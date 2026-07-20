#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from run_lri_experiment import load_datasets, positive_pairs, sample_negative_pairs  # noqa: E402


def stable_dataset_seed(name: str, seed: int) -> int:
    return seed + sum((i + 1) * ord(ch) for i, ch in enumerate(name))


def sequence_chunks(seq: str, chunk_len: int) -> Sequence[str]:
    return [seq[start : start + chunk_len] for start in range(0, len(seq), chunk_len)] or ["X"]


def resample_tokens(tokens: torch.Tensor, max_residues: int) -> torch.Tensor:
    if tokens.shape[0] <= max_residues:
        return tokens
    idx = torch.linspace(0, tokens.shape[0] - 1, max_residues).round().long()
    return tokens[idx]


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
    return (x * mask.unsqueeze(-1).to(x.dtype)).sum(dim=1) / denom


def masked_max_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    fill = torch.finfo(x.dtype).min
    x = x.masked_fill(~mask.unsqueeze(-1), fill)
    return x.max(dim=1).values


class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


class PairwiseRankingLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos = logits[targets > 0.5]
        neg = logits[targets <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            return logits.new_tensor(0.0)
        return nn.functional.softplus(neg[:, None] - pos[None, :]).mean()


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GlobalEncoder(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
            nn.GELU(),
            ResidualBlock(out_dim, dropout),
            ResidualBlock(out_dim, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalExpert(nn.Module):
    def __init__(self, dim: int, dropout: float, dilation: int):
        super().__init__()
        padding = dilation * 2
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=5, padding=padding, dilation=dilation),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2, dilation=1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalMoEEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_dim: int,
        num_experts: int,
        dropout: float,
        dilations: Sequence[int],
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)
        self.experts = nn.ModuleList(
            [LocalExpert(out_dim, dropout, dilations[i % len(dilations)]) for i in range(num_experts)]
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_experts),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def project_tokens(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.input_norm(tokens))
        return x * mask.unsqueeze(-1).to(x.dtype)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, return_tokens: bool = False):
        gate_input = masked_mean(tokens, mask)
        weights = torch.softmax(self.gate(gate_input), dim=1)
        token_features = self.project_tokens(tokens, mask)
        x = token_features.transpose(1, 2)
        expert_vectors = []
        for expert in self.experts:
            y = expert(x).transpose(1, 2)
            expert_vectors.append(masked_max_pool(y, mask))
        stacked = torch.stack(expert_vectors, dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        local_vec = self.out(fused)
        if return_tokens:
            return local_vec, token_features
        return local_vec


class ProteinFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, global_vec: torch.Tensor, local_vec: torch.Tensor) -> torch.Tensor:
        beta = self.gate(torch.cat([global_vec, local_vec], dim=1))
        return beta * global_vec + (1.0 - beta) * local_vec


class PairAwareLocalInteraction(nn.Module):
    def __init__(self, dim: int, attention_dim: int, dropout: float):
        super().__init__()
        self.scale = attention_dim ** -0.5
        self.ligand_norm = nn.LayerNorm(dim)
        self.receptor_norm = nn.LayerNorm(dim)
        self.ligand_query = nn.Linear(dim, attention_dim)
        self.receptor_key = nn.Linear(dim, attention_dim)
        self.ligand_value = nn.Linear(dim, dim)
        self.receptor_value = nn.Linear(dim, dim)
        self.align = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(dim, dropout),
        )

    def forward(
        self,
        ligand_tokens: torch.Tensor,
        ligand_mask: torch.Tensor,
        receptor_tokens: torch.Tensor,
        receptor_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ligand = self.ligand_norm(ligand_tokens)
        receptor = self.receptor_norm(receptor_tokens)
        scores = torch.bmm(
            self.ligand_query(ligand),
            self.receptor_key(receptor).transpose(1, 2),
        ) * self.scale
        valid = ligand_mask.unsqueeze(2) & receptor_mask.unsqueeze(1)
        masked_scores = scores.masked_fill(~valid, -1e4)

        ligand_to_receptor = torch.softmax(masked_scores, dim=2)
        receptor_to_ligand = torch.softmax(masked_scores.transpose(1, 2), dim=2)
        ligand_context = torch.bmm(ligand_to_receptor, self.receptor_value(receptor))
        receptor_context = torch.bmm(receptor_to_ligand, self.ligand_value(ligand))

        ligand_aligned = self.align(
            torch.cat(
                [
                    ligand_tokens,
                    ligand_context,
                    torch.abs(ligand_tokens - ligand_context),
                    ligand_tokens * ligand_context,
                ],
                dim=2,
            )
        )
        receptor_aligned = self.align(
            torch.cat(
                [
                    receptor_tokens,
                    receptor_context,
                    torch.abs(receptor_tokens - receptor_context),
                    receptor_tokens * receptor_context,
                ],
                dim=2,
            )
        )
        ligand_pair = masked_max_pool(ligand_aligned, ligand_mask)
        receptor_pair = masked_max_pool(receptor_aligned, receptor_mask)

        valid_count = valid.sum(dim=(1, 2)).clamp_min(1).to(scores.dtype)
        score_max = scores.masked_fill(~valid, torch.finfo(scores.dtype).min).flatten(1).max(dim=1).values
        score_mean = scores.masked_fill(~valid, 0.0).sum(dim=(1, 2)) / valid_count
        stats = torch.stack([score_max, score_mean], dim=1)
        return ligand_pair, receptor_pair, stats


class PairAttentionMoELRIClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        protein_dim: int,
        num_experts: int,
        hidden_dim: int,
        attention_dim: int,
        dropout: float,
        dilations: Sequence[int],
    ):
        super().__init__()
        self.global_encoder = GlobalEncoder(embed_dim, protein_dim, dropout)
        self.local_encoder = LocalMoEEncoder(embed_dim, protein_dim, num_experts, dropout, dilations)
        self.fusion = ProteinFusion(protein_dim)
        self.pair_interaction = PairAwareLocalInteraction(protein_dim, attention_dim, dropout)
        pair_dim = protein_dim * 8 + 2
        global_pair_dim = protein_dim * 4
        local_pair_dim = protein_dim * 4 + 2
        self.global_head = self.make_head(global_pair_dim, hidden_dim, dropout)
        self.local_head = self.make_head(local_pair_dim, hidden_dim, dropout)
        self.joint_head = self.make_head(pair_dim, hidden_dim, dropout)
        self.gate = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    @staticmethod
    def make_head(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_protein(
        self, global_vec: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.global_encoder(global_vec)
        r, token_features = self.local_encoder(tokens, mask, return_tokens=True)
        return self.fusion(g, r), token_features

    def forward(
        self,
        ligand_global: torch.Tensor,
        receptor_global: torch.Tensor,
        ligand_tokens: torch.Tensor,
        ligand_mask: torch.Tensor,
        receptor_tokens: torch.Tensor,
        receptor_mask: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor:
        ligand, ligand_token_features = self.encode_protein(ligand_global, ligand_tokens, ligand_mask)
        receptor, receptor_token_features = self.encode_protein(receptor_global, receptor_tokens, receptor_mask)
        ligand_pair, receptor_pair, pair_stats = self.pair_interaction(
            ligand_token_features,
            ligand_mask,
            receptor_token_features,
            receptor_mask,
        )
        global_pair = torch.cat([ligand, receptor, torch.abs(ligand - receptor), ligand * receptor], dim=1)
        local_pair = torch.cat(
            [
                ligand_pair,
                receptor_pair,
                torch.abs(ligand_pair - receptor_pair),
                ligand_pair * receptor_pair,
                pair_stats,
            ],
            dim=1,
        )
        pair = torch.cat([global_pair, local_pair], dim=1)
        global_logits = self.global_head(global_pair).squeeze(1)
        local_logits = self.local_head(local_pair).squeeze(1)
        joint_logits = self.joint_head(pair).squeeze(1)
        branch_logits = torch.stack([global_logits, local_logits, joint_logits], dim=1)
        gate_weights = torch.softmax(self.gate(pair), dim=1)
        logits = (branch_logits * gate_weights).sum(dim=1)
        if return_all:
            return {
                "logits": logits,
                "global_logits": global_logits,
                "local_logits": local_logits,
                "joint_logits": joint_logits,
                "gate_weights": gate_weights,
            }
        return logits


class LRIPairDataset(Dataset):
    def __init__(self, pairs: np.ndarray, labels: np.ndarray):
        self.pairs = pairs.astype(np.int64)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return int(self.pairs[idx, 0]), int(self.pairs[idx, 1]), float(self.labels[idx])


class PairCollator:
    def __init__(
        self,
        ligand_global: np.ndarray,
        receptor_global: np.ndarray,
        ligand_local: List[torch.Tensor],
        receptor_local: List[torch.Tensor],
    ):
        self.ligand_global = torch.from_numpy(ligand_global.astype(np.float32))
        self.receptor_global = torch.from_numpy(receptor_global.astype(np.float32))
        self.ligand_local = ligand_local
        self.receptor_local = receptor_local

    @staticmethod
    def pad_token_list(token_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = [t.shape[0] for t in token_list]
        max_len = max(lengths)
        dim = token_list[0].shape[1]
        batch = torch.zeros(len(token_list), max_len, dim, dtype=torch.float16)
        mask = torch.zeros(len(token_list), max_len, dtype=torch.bool)
        for i, tokens in enumerate(token_list):
            length = tokens.shape[0]
            batch[i, :length] = tokens
            mask[i, :length] = True
        return batch, mask

    def __call__(self, batch):
        ligand_idx = [item[0] for item in batch]
        receptor_idx = [item[1] for item in batch]
        labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        ligand_global = self.ligand_global[ligand_idx]
        receptor_global = self.receptor_global[receptor_idx]
        ligand_tokens, ligand_mask = self.pad_token_list([self.ligand_local[i] for i in ligand_idx])
        receptor_tokens, receptor_mask = self.pad_token_list([self.receptor_local[i] for i in receptor_idx])
        return ligand_global, receptor_global, ligand_tokens, ligand_mask, receptor_tokens, receptor_mask, labels


@dataclass
class RunConfig:
    data_dir: str
    embedding_dir: str
    out_dir: str
    model_name: str
    max_residues: int
    chunk_len: int
    neg_ratio: float
    test_size: float
    val_size: float
    seed: int
    epochs: int
    patience: int
    batch_size: int
    lr: float
    weight_decay: float
    protein_dim: int
    hidden_dim: int
    pair_attention_dim: int
    num_experts: int
    dropout: float
    focal_alpha: float
    focal_gamma: float
    aux_loss_weight: float
    rank_loss_weight: float
    positive_target: float
    negative_target: float
    consistency_weight: float


def load_global_embeddings(embedding_dir: Path, role: str, ids: List[str], model_name: str) -> np.ndarray:
    safe_model = model_name.replace("/", "__")
    path = embedding_dir / f"human_{role}_{safe_model}.npz"
    data = np.load(path, allow_pickle=False)
    saved_ids = data["ids"].astype(str).tolist()
    if saved_ids != ids:
        raise ValueError(f"Embedding ID order mismatch for {path}")
    return data["embeddings"].astype(np.float32)


def extract_local_embeddings(
    ids: List[str],
    seqs: Dict[str, str],
    model,
    esm_protein,
    logits_config,
    device: str,
    chunk_len: int,
    max_residues: int,
    desc: str,
) -> List[torch.Tensor]:
    local = []
    with torch.inference_mode():
        for pid in tqdm(ids, desc=desc):
            pieces = []
            for chunk in sequence_chunks(seqs[pid], chunk_len):
                protein = esm_protein(sequence=chunk)
                encoded = model.encode(protein)
                output = model.logits(encoded, logits_config(sequence=True, return_embeddings=True))
                emb = output.embeddings.detach()
                if emb.ndim == 3:
                    emb = emb[0]
                if emb.shape[0] >= len(chunk) + 2:
                    emb = emb[1 : 1 + len(chunk)]
                elif emb.shape[0] > len(chunk):
                    emb = emb[: len(chunk)]
                pieces.append(emb.float().cpu())
            tokens = torch.cat(pieces, dim=0)
            tokens = resample_tokens(tokens, max_residues).half().contiguous()
            local.append(tokens)
    return local


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            ligand_global, receptor_global, ligand_tokens, ligand_mask, receptor_tokens, receptor_mask, y = batch
            ligand_global = ligand_global.to(device)
            receptor_global = receptor_global.to(device)
            ligand_tokens = ligand_tokens.to(device).float()
            receptor_tokens = receptor_tokens.to(device).float()
            ligand_mask = ligand_mask.to(device)
            receptor_mask = receptor_mask.to(device)
            logits = model(
                ligand_global,
                receptor_global,
                ligand_tokens,
                ligand_mask,
                receptor_tokens,
                receptor_mask,
            )
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())
    y_true = np.concatenate(labels).astype(np.int64)
    p = np.concatenate(probs)
    pred = (p >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "average_precision": float(average_precision_score(y_true, p)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics, y_true, p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("outputs/lri_esmc_600m/embeddings"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/human_multibranch_moe_esmc_600m"))
    parser.add_argument("--model-name", default="esmc_600m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-residues", type=int, default=512)
    parser.add_argument("--chunk-len", type=int, default=1022)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--protein-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--pair-attention-dim", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--aux-loss-weight", type=float, default=0.25)
    parser.add_argument("--rank-loss-weight", type=float, default=0.05)
    parser.add_argument("--positive-target", type=float, default=1.0)
    parser.add_argument("--negative-target", type=float, default=0.0)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260520)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_datasets(args.data_dir, ["human"])["human"]
    ligand_global = load_global_embeddings(args.embedding_dir, "ligand", dataset.ligand_ids, args.model_name)
    receptor_global = load_global_embeddings(args.embedding_dir, "receptor", dataset.receptor_ids, args.model_name)

    print("loading ESM C for residue-level embeddings", flush=True)
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    esm_model = ESMC.from_pretrained(args.model_name).to(args.device)
    esm_model.eval()
    ligand_local = extract_local_embeddings(
        dataset.ligand_ids,
        dataset.ligand_sequences,
        esm_model,
        ESMProtein,
        LogitsConfig,
        args.device,
        args.chunk_len,
        args.max_residues,
        "local human/ligand",
    )
    receptor_local = extract_local_embeddings(
        dataset.receptor_ids,
        dataset.receptor_sequences,
        esm_model,
        ESMProtein,
        LogitsConfig,
        args.device,
        args.chunk_len,
        args.max_residues,
        "local human/receptor",
    )
    del esm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("finished residue-level embeddings", flush=True)

    rng_seed = stable_dataset_seed("human", args.seed)
    pos = positive_pairs(dataset.matrix)
    neg = sample_negative_pairs(dataset.matrix, int(len(pos) * args.neg_ratio), rng_seed)
    pairs = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)])
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=args.test_size, random_state=rng_seed, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=args.val_size / (1.0 - args.test_size),
        random_state=rng_seed + 1,
        stratify=labels[train_idx],
    )

    collator = PairCollator(ligand_global, receptor_global, ligand_local, receptor_local)
    train_loader = DataLoader(
        LRIPairDataset(pairs[train_idx], labels[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(rng_seed),
        num_workers=0,
    )
    val_loader = DataLoader(
        LRIPairDataset(pairs[val_idx], labels[val_idx]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    test_loader = DataLoader(
        LRIPairDataset(pairs[test_idx], labels[test_idx]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    model = PairAttentionMoELRIClassifier(
        embed_dim=ligand_global.shape[1],
        protein_dim=args.protein_dim,
        num_experts=args.num_experts,
        hidden_dim=args.hidden_dim,
        attention_dim=args.pair_attention_dim,
        dropout=args.dropout,
        dilations=[1, 2, 4, 8],
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = FocalLossWithLogits(alpha=args.focal_alpha, gamma=args.focal_gamma)
    rank_loss_fn = PairwiseRankingLoss()

    best_state = None
    best_epoch = 0
    best_val_ap = -math.inf
    patience_left = args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            ligand_global_b, receptor_global_b, ligand_tokens, ligand_mask, receptor_tokens, receptor_mask, y = batch
            ligand_global_b = ligand_global_b.to(args.device)
            receptor_global_b = receptor_global_b.to(args.device)
            ligand_tokens = ligand_tokens.to(args.device).float()
            receptor_tokens = receptor_tokens.to(args.device).float()
            ligand_mask = ligand_mask.to(args.device)
            receptor_mask = receptor_mask.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                ligand_global_b,
                receptor_global_b,
                ligand_tokens,
                ligand_mask,
                receptor_tokens,
                receptor_mask,
                return_all=True,
            )
            logits = outputs["logits"]
            soft_y = torch.where(
                y > 0.5,
                torch.full_like(y, args.positive_target),
                torch.full_like(y, args.negative_target),
            )
            loss = loss_fn(logits, soft_y)
            if args.aux_loss_weight > 0:
                aux_loss = (
                    loss_fn(outputs["global_logits"], soft_y)
                    + loss_fn(outputs["local_logits"], soft_y)
                    + loss_fn(outputs["joint_logits"], soft_y)
                ) / 3.0
                loss = loss + args.aux_loss_weight * aux_loss
            if args.rank_loss_weight > 0:
                loss = loss + args.rank_loss_weight * rank_loss_fn(logits, y)
            if args.consistency_weight > 0:
                repeat_outputs = model(
                    ligand_global_b,
                    receptor_global_b,
                    ligand_tokens,
                    ligand_mask,
                    receptor_tokens,
                    receptor_mask,
                    return_all=True,
                )
                repeat_logits = repeat_outputs["logits"]
                repeat_loss = loss_fn(repeat_logits, soft_y)
                if args.aux_loss_weight > 0:
                    repeat_aux_loss = (
                        loss_fn(repeat_outputs["global_logits"], soft_y)
                        + loss_fn(repeat_outputs["local_logits"], soft_y)
                        + loss_fn(repeat_outputs["joint_logits"], soft_y)
                    ) / 3.0
                    repeat_loss = repeat_loss + args.aux_loss_weight * repeat_aux_loss
                if args.rank_loss_weight > 0:
                    repeat_loss = repeat_loss + args.rank_loss_weight * rank_loss_fn(repeat_logits, y)
                consistency_loss = nn.functional.mse_loss(
                    torch.sigmoid(logits),
                    torch.sigmoid(repeat_logits),
                )
                loss = 0.5 * (loss + repeat_loss) + args.consistency_weight * consistency_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics, _, _ = evaluate(model, val_loader, args.device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **val_metrics}
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={row['loss']:.5f} "
            f"val_auc={row['roc_auc']:.5f} val_aupr={row['average_precision']:.5f} "
            f"val_f1={row['f1']:.5f}",
            flush=True,
        )
        if row["average_precision"] > best_val_ap:
            best_val_ap = row["average_precision"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping at epoch={epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics, y_val, val_probs = evaluate(model, val_loader, args.device)
    test_metrics, y_test, test_probs = evaluate(model, test_loader, args.device)

    config = RunConfig(
        data_dir=str(args.data_dir),
        embedding_dir=str(args.embedding_dir),
        out_dir=str(args.out_dir),
        model_name=args.model_name,
        max_residues=args.max_residues,
        chunk_len=args.chunk_len,
        neg_ratio=args.neg_ratio,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        protein_dim=args.protein_dim,
        hidden_dim=args.hidden_dim,
        pair_attention_dim=args.pair_attention_dim,
        num_experts=args.num_experts,
        dropout=args.dropout,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        aux_loss_weight=args.aux_loss_weight,
        rank_loss_weight=args.rank_loss_weight,
        positive_target=args.positive_target,
        negative_target=args.negative_target,
        consistency_weight=args.consistency_weight,
    )
    result = {
        "dataset": "human",
        "model": "ESMC_LocalCNNMoE_MultiBranchRank",
        "matrix_shape": list(dataset.matrix.shape),
        "matrix_positives": int(dataset.matrix.sum()),
        "embedding_dim": int(ligand_global.shape[1]),
        "positive_samples": int(len(pos)),
        "negative_samples": int(len(neg)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "best_epoch": int(best_epoch),
        "config": asdict(config),
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
    }

    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (args.out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "model",
                "best_epoch",
                "test_roc_auc",
                "test_average_precision",
                "test_recall",
                "test_precision",
                "test_accuracy",
                "test_f1",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        )
        writer.writerow(
            [
                "human",
                "ESMC_LocalCNNMoE_MultiBranchRank",
                best_epoch,
                f"{test_metrics['roc_auc']:.6f}",
                f"{test_metrics['average_precision']:.6f}",
                f"{test_metrics['recall']:.6f}",
                f"{test_metrics['precision']:.6f}",
                f"{test_metrics['accuracy']:.6f}",
                f"{test_metrics['f1']:.6f}",
                test_metrics["tn"],
                test_metrics["fp"],
                test_metrics["fn"],
                test_metrics["tp"],
            ]
        )
    torch.save({"model_state": model.state_dict(), "result": result}, args.out_dir / "human_multibranch_moe.pt")
    np.savez_compressed(
        args.out_dir / "test_predictions.npz",
        y_test=y_test,
        probs=test_probs,
        pairs=pairs[test_idx],
        y_val=y_val,
        val_probs=val_probs,
        val_pairs=pairs[val_idx],
    )
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"Done. Results written to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
