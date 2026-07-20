#!/usr/bin/env python3
import argparse
import csv
import gzip
import heapq
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class LRIDataset:
    name: str
    ligand_ids: List[str]
    receptor_ids: List[str]
    ligand_sequences: Dict[str, str]
    receptor_sequences: Dict[str, str]
    ligand_symbols: Dict[str, str]
    receptor_symbols: Dict[str, str]
    matrix: np.ndarray


def clean_sequence(seq: str) -> str:
    seq = re.sub(r"\s+", "", seq).upper()
    return "".join(ch if ch in VALID_AA else "X" for ch in seq)


def read_csv(path: Path) -> List[List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [
            [cell.strip() for cell in row]
            for row in csv.reader(f)
            if row and not all(cell.strip() == "" for cell in row)
        ]


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header = None
    parts: List[str] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, clean_sequence("".join(parts))))
                header = line[1:].strip()
                parts = []
            else:
                parts.append(line)
    if header is not None:
        records.append((header, clean_sequence("".join(parts))))
    return records


def load_relation_matrix(path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    rows = read_csv(path)
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"Invalid relation matrix: {path}")
    receptor_ids = rows[0][1:]
    ligand_ids: List[str] = []
    matrix_rows: List[List[int]] = []
    for row in rows[1:]:
        ligand_ids.append(row[0])
        values = [1 if value in {"1", "1.0"} else 0 for value in row[1:]]
        if len(values) != len(receptor_ids):
            raise ValueError(f"Bad row width in {path}: {row[0]}")
        matrix_rows.append(values)
    return ligand_ids, receptor_ids, np.asarray(matrix_rows, dtype=np.uint8)


def load_human(data_dir: Path) -> LRIDataset:
    root = data_dir / "human"
    ligand_ids, receptor_ids, matrix = load_relation_matrix(root / "human_Related_name.csv")

    ligand_records = read_fasta(root / "feature" / "ligand.txt")
    receptor_records = read_fasta(root / "feature" / "receptor.txt")
    ligand_sequences = {header.split()[0]: seq for header, seq in ligand_records}
    receptor_sequences = {header.split()[0]: seq for header, seq in receptor_records}

    ligand_symbols = {}
    for row in read_csv(root / "ligand_gen.csv"):
        if len(row) >= 3:
            ligand_symbols[row[2]] = row[1]

    receptor_symbols = {}
    for row in read_csv(root / "receptor_gen.csv"):
        if len(row) >= 3:
            receptor_symbols[row[2]] = row[0]

    return LRIDataset(
        name="human",
        ligand_ids=ligand_ids,
        receptor_ids=receptor_ids,
        ligand_sequences=ligand_sequences,
        receptor_sequences=receptor_sequences,
        ligand_symbols=ligand_symbols,
        receptor_symbols=receptor_symbols,
        matrix=matrix,
    )


def load_ordered_sequence_dataset(
    name: str,
    root: Path,
    relation_name: str,
    ligand_gene_name: str,
    receptor_gene_name: str,
    ligand_fasta_name: str,
    receptor_fasta_name: str,
) -> LRIDataset:
    ligand_ids, receptor_ids, matrix = load_relation_matrix(root / relation_name)
    ligand_genes = [row[0] for row in read_csv(root / ligand_gene_name)]
    receptor_genes = [row[0] for row in read_csv(root / receptor_gene_name)]
    ligand_records = read_fasta(root / ligand_fasta_name)
    receptor_records = read_fasta(root / receptor_fasta_name)

    if len(ligand_ids) != len(ligand_records):
        raise ValueError(f"{name}: ligand ID count and FASTA count do not match")
    if len(receptor_ids) != len(receptor_records):
        raise ValueError(f"{name}: receptor ID count and FASTA count do not match")

    ligand_sequences = {pid: seq for pid, (_, seq) in zip(ligand_ids, ligand_records)}
    receptor_sequences = {pid: seq for pid, (_, seq) in zip(receptor_ids, receptor_records)}
    ligand_symbols = {pid: gene for pid, gene in zip(ligand_ids, ligand_genes)}
    receptor_symbols = {pid: gene for pid, gene in zip(receptor_ids, receptor_genes)}

    return LRIDataset(
        name=name,
        ligand_ids=ligand_ids,
        receptor_ids=receptor_ids,
        ligand_sequences=ligand_sequences,
        receptor_sequences=receptor_sequences,
        ligand_symbols=ligand_symbols,
        receptor_symbols=receptor_symbols,
        matrix=matrix,
    )


def load_dataset4(data_dir: Path) -> LRIDataset:
    root = data_dir / "dataset 4"
    if not root.exists():
        root = data_dir / "dataset4"
    ligand_ids, receptor_ids, matrix = load_relation_matrix(root / "ligand-receptor interaction.csv")
    ligand_records = read_fasta(root / "ligand sequence.txt")
    receptor_records = read_fasta(root / "receptor sequence.txt")

    if len(ligand_ids) != len(ligand_records):
        raise ValueError("dataset4: ligand ID count and FASTA count do not match")
    if len(receptor_ids) != len(receptor_records):
        raise ValueError("dataset4: receptor ID count and FASTA count do not match")

    ligand_sequences = {pid: seq for pid, (_, seq) in zip(ligand_ids, ligand_records)}
    receptor_sequences = {pid: seq for pid, (_, seq) in zip(receptor_ids, receptor_records)}
    ligand_symbols = {pid: pid for pid in ligand_ids}
    receptor_symbols = {pid: pid for pid in receptor_ids}

    return LRIDataset(
        name="dataset4",
        ligand_ids=ligand_ids,
        receptor_ids=receptor_ids,
        ligand_sequences=ligand_sequences,
        receptor_sequences=receptor_sequences,
        ligand_symbols=ligand_symbols,
        receptor_symbols=receptor_symbols,
        matrix=matrix,
    )


def load_datasets(data_dir: Path, names: Sequence[str]) -> Dict[str, LRIDataset]:
    datasets: Dict[str, LRIDataset] = {}
    for name in names:
        if name == "human":
            datasets[name] = load_human(data_dir)
        elif name == "mouse":
            datasets[name] = load_ordered_sequence_dataset(
                name="mouse",
                root=data_dir / "mouse",
                relation_name="Related.csv",
                ligand_gene_name="l-gene650.csv",
                receptor_gene_name="r-gene588.csv",
                ligand_fasta_name="ligand650.csv",
                receptor_fasta_name="receptor588.csv",
            )
        elif name == "mouse-heart":
            datasets[name] = load_ordered_sequence_dataset(
                name="mouse-heart",
                root=data_dir / "mouse-heart",
                relation_name="Related.csv",
                ligand_gene_name="l-gene574.csv",
                receptor_gene_name="r-gene559.csv",
                ligand_fasta_name="ligand574.csv",
                receptor_fasta_name="receptor559.csv",
            )
        elif name in {"dataset4", "dataset 4"}:
            datasets["dataset4"] = load_dataset4(data_dir)
        else:
            raise ValueError(f"Unknown dataset: {name}")
    return datasets


def sequence_chunks(seq: str, chunk_len: int) -> Iterable[str]:
    if not seq:
        yield "X"
        return
    for start in range(0, len(seq), chunk_len):
        yield seq[start : start + chunk_len]


class ESMCEmbedder:
    def __init__(self, model_name: str, device: str, chunk_len: int):
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig

        self.ESMProtein = ESMProtein
        self.LogitsConfig = LogitsConfig
        self.device = torch.device(device)
        self.chunk_len = chunk_len
        self.model = ESMC.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.dim = None

    @torch.inference_mode()
    def embed_chunk(self, seq: str) -> np.ndarray:
        protein = self.ESMProtein(sequence=seq)
        protein_tensor = self.model.encode(protein)
        output = self.model.logits(
            protein_tensor, self.LogitsConfig(sequence=True, return_embeddings=True)
        )
        emb = output.embeddings
        if isinstance(emb, torch.Tensor):
            token_emb = emb
        else:
            token_emb = torch.as_tensor(emb)
        token_emb = token_emb.detach()
        if token_emb.ndim == 3:
            token_emb = token_emb[0]
        token_emb = token_emb.float()

        # ESM C returns special-token embeddings as well. Prefer the amino-acid
        # span if present; otherwise fall back to all non-empty token positions.
        if token_emb.shape[0] >= len(seq) + 2:
            token_emb = token_emb[1 : 1 + len(seq)]
        elif token_emb.shape[0] > len(seq):
            token_emb = token_emb[: len(seq)]
        pooled = token_emb.mean(dim=0).cpu().numpy().astype(np.float32)
        self.dim = pooled.shape[0]
        return pooled

    def embed_sequence(self, seq: str) -> np.ndarray:
        vectors: List[np.ndarray] = []
        weights: List[int] = []
        for chunk in sequence_chunks(seq, self.chunk_len):
            vectors.append(self.embed_chunk(chunk))
            weights.append(len(chunk))
        return np.average(np.stack(vectors), axis=0, weights=np.asarray(weights)).astype(np.float32)


class TransformersESMEmbedder:
    def __init__(self, model_name: str, device: str, chunk_len: int):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        self.chunk_len = chunk_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def embed_chunk(self, seq: str) -> np.ndarray:
        encoded = self.tokenizer(
            seq,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=self.chunk_len + 2,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        output = self.model(**encoded)
        token_emb = output.last_hidden_state[0].float()
        mask = encoded["attention_mask"][0].bool()
        valid_idx = torch.where(mask)[0]
        if len(valid_idx) >= len(seq) + 2:
            token_emb = token_emb[1 : 1 + len(seq)]
        else:
            token_emb = token_emb[valid_idx]
        return token_emb.mean(dim=0).cpu().numpy().astype(np.float32)

    def embed_sequence(self, seq: str) -> np.ndarray:
        vectors: List[np.ndarray] = []
        weights: List[int] = []
        for chunk in sequence_chunks(seq, self.chunk_len):
            vectors.append(self.embed_chunk(chunk))
            weights.append(len(chunk))
        return np.average(np.stack(vectors), axis=0, weights=np.asarray(weights)).astype(np.float32)


def make_embedder(args):
    if args.embedder == "esmc":
        return ESMCEmbedder(args.model_name, args.device, args.chunk_len)
    if args.embedder == "transformers-esm":
        return TransformersESMEmbedder(args.model_name, args.device, args.chunk_len)
    raise ValueError(args.embedder)


def embedding_file(out_dir: Path, dataset_name: str, role: str, model_name: str) -> Path:
    safe_model = model_name.replace("/", "__")
    return out_dir / "embeddings" / f"{dataset_name}_{role}_{safe_model}.npz"


def get_or_create_embeddings(
    dataset: LRIDataset,
    role: str,
    ids: List[str],
    seqs: Dict[str, str],
    embedder,
    out_dir: Path,
    model_name: str,
    overwrite: bool,
) -> np.ndarray:
    path = embedding_file(out_dir, dataset.name, role, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        data = np.load(path, allow_pickle=False)
        saved_ids = data["ids"].astype(str).tolist()
        if saved_ids == ids:
            return data["embeddings"].astype(np.float32)
        print(f"[warn] Recomputing {path}; saved IDs do not match current dataset order.")

    vectors: List[np.ndarray] = []
    missing = [pid for pid in ids if pid not in seqs]
    if missing:
        raise ValueError(f"{dataset.name}/{role}: missing sequences for {missing[:5]}")

    for pid in tqdm(ids, desc=f"embedding {dataset.name}/{role}"):
        vectors.append(embedder.embed_sequence(seqs[pid]))
    embeddings = np.stack(vectors).astype(np.float32)
    np.savez_compressed(path, ids=np.asarray(ids, dtype=str), embeddings=embeddings)
    return embeddings


def positive_pairs(matrix: np.ndarray) -> np.ndarray:
    return np.argwhere(matrix == 1).astype(np.int64)


def sample_negative_pairs(matrix: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_lig, n_rec = matrix.shape
    target = min(n, int((matrix == 0).sum()))
    negatives = set()
    # Rejection sampling is efficient because positives are only about 0.5%.
    while len(negatives) < target:
        batch = max(4096, (target - len(negatives)) * 2)
        lig = rng.integers(0, n_lig, size=batch)
        rec = rng.integers(0, n_rec, size=batch)
        for i, j in zip(lig, rec):
            if matrix[i, j] == 0:
                negatives.add((int(i), int(j)))
                if len(negatives) >= target:
                    break
    return np.asarray(sorted(negatives), dtype=np.int64)


def pair_features(
    ligand_embeddings: np.ndarray,
    receptor_embeddings: np.ndarray,
    pairs: np.ndarray,
    batch_size: int = 65536,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for start in range(0, len(pairs), batch_size):
        p = pairs[start : start + batch_size]
        lig = ligand_embeddings[p[:, 0]]
        rec = receptor_embeddings[p[:, 1]]
        chunks.append(np.concatenate([lig, rec, np.abs(lig - rec), lig * rec], axis=1))
    return np.concatenate(chunks, axis=0).astype(np.float32)


class PairMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def evaluate_classifier(model: nn.Module, x: np.ndarray, y: np.ndarray, device: str) -> Dict[str, float]:
    model.eval()
    probs: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), 16384):
            xb = torch.from_numpy(x[start : start + 16384]).to(device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    p = np.concatenate(probs)
    pred = (p >= 0.5).astype(np.int64)
    metrics = {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def train_one_dataset(
    dataset: LRIDataset,
    ligand_embeddings: np.ndarray,
    receptor_embeddings: np.ndarray,
    args,
    out_dir: Path,
) -> Tuple[PairMLP, StandardScaler, Dict[str, object]]:
    rng_seed = args.seed + abs(hash(dataset.name)) % 10000
    pos = positive_pairs(dataset.matrix)
    neg = sample_negative_pairs(dataset.matrix, int(len(pos) * args.neg_ratio), rng_seed)
    pairs = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate(
        [np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)], axis=0
    )

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=args.test_size,
        random_state=rng_seed,
        stratify=labels,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=args.val_size / (1.0 - args.test_size),
        random_state=rng_seed + 1,
        stratify=labels[train_idx],
    )

    x_train = pair_features(ligand_embeddings, receptor_embeddings, pairs[train_idx])
    x_val = pair_features(ligand_embeddings, receptor_embeddings, pairs[val_idx])
    x_test = pair_features(ligand_embeddings, receptor_embeddings, pairs[test_idx])
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    device = torch.device(args.device)
    model = PairMLP(x_train.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    generator = torch.Generator().manual_seed(rng_seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    best_state = None
    best_val_ap = -math.inf
    best_epoch = 0
    patience_left = args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_classifier(model, x_val, y_val, args.device)
        val_ap = val_metrics["average_precision"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **val_metrics})
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate_classifier(model, x_val, y_val, args.device)
    test_metrics = evaluate_classifier(model, x_test, y_test, args.device)
    metadata: Dict[str, object] = {
        "dataset": dataset.name,
        "matrix_shape": list(dataset.matrix.shape),
        "matrix_positives": int(dataset.matrix.sum()),
        "matrix_density": float(dataset.matrix.mean()),
        "positive_samples": int(len(pos)),
        "negative_samples": int(len(neg)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "embedding_dim": int(ligand_embeddings.shape[1]),
        "pair_feature_dim": int(x_train.shape[1]),
        "best_epoch": int(best_epoch),
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
    }

    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "metadata": metadata,
            "args": vars(args),
        },
        model_dir / f"{dataset.name}_mlp.pt",
    )
    return model, scaler, metadata


def score_zero_pairs_topk(
    dataset: LRIDataset,
    ligand_embeddings: np.ndarray,
    receptor_embeddings: np.ndarray,
    model: PairMLP,
    scaler: StandardScaler,
    args,
    out_dir: Path,
) -> Path:
    model.eval()
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    top_heap: List[Tuple[float, int, int]] = []
    batch_pairs: List[Tuple[int, int]] = []

    def flush_batch():
        if not batch_pairs:
            return
        pairs = np.asarray(batch_pairs, dtype=np.int64)
        feats = pair_features(ligand_embeddings, receptor_embeddings, pairs)
        feats = scaler.transform(feats).astype(np.float32)
        with torch.inference_mode():
            xb = torch.from_numpy(feats).to(args.device)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
        for (li, ri), score in zip(batch_pairs, probs):
            item = (float(score), int(li), int(ri))
            if len(top_heap) < args.top_k:
                heapq.heappush(top_heap, item)
            elif item[0] > top_heap[0][0]:
                heapq.heapreplace(top_heap, item)
        batch_pairs.clear()

    n_lig, n_rec = dataset.matrix.shape
    for li in tqdm(range(n_lig), desc=f"scoring zeros {dataset.name}"):
        zero_cols = np.where(dataset.matrix[li] == 0)[0]
        for ri in zero_cols:
            batch_pairs.append((li, int(ri)))
            if len(batch_pairs) >= args.predict_batch_size:
                flush_batch()
    flush_batch()

    top = sorted(top_heap, reverse=True)
    output_path = pred_dir / f"{dataset.name}_top{args.top_k}_unknown_pairs.csv.gz"
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "score",
                "ligand_id",
                "ligand_symbol",
                "receptor_id",
                "receptor_symbol",
            ]
        )
        for rank, (score, li, ri) in enumerate(top, start=1):
            ligand_id = dataset.ligand_ids[li]
            receptor_id = dataset.receptor_ids[ri]
            writer.writerow(
                [
                    rank,
                    f"{score:.8f}",
                    ligand_id,
                    dataset.ligand_symbols.get(ligand_id, ""),
                    receptor_id,
                    dataset.receptor_symbols.get(receptor_id, ""),
                ]
            )
    return output_path


def write_summary(out_dir: Path, summaries: List[Dict[str, object]]) -> None:
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "ligands",
                "receptors",
                "positives",
                "density",
                "embedding_dim",
                "best_epoch",
                "test_roc_auc",
                "test_average_precision",
                "test_f1",
                "test_precision",
                "test_recall",
                "test_accuracy",
            ]
        )
        for m in summaries:
            test = m["test"]
            shape = m["matrix_shape"]
            writer.writerow(
                [
                    m["dataset"],
                    shape[0],
                    shape[1],
                    m["matrix_positives"],
                    f"{m['matrix_density']:.8f}",
                    m["embedding_dim"],
                    m["best_epoch"],
                    f"{test['roc_auc']:.6f}",
                    f"{test['average_precision']:.6f}",
                    f"{test['f1']:.6f}",
                    f"{test['precision']:.6f}",
                    f"{test['recall']:.6f}",
                    f"{test['accuracy']:.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/lri_esmc"))
    parser.add_argument("--datasets", nargs="+", default=["human", "mouse", "mouse-heart"])
    parser.add_argument("--embedder", choices=["esmc", "transformers-esm"], default="esmc")
    parser.add_argument("--model-name", default="esmc_600m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-len", type=int, default=1022)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--overwrite-embeddings", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets(args.data_dir, args.datasets)
    print("Loaded datasets:")
    for ds in datasets.values():
        print(
            f"  {ds.name}: {len(ds.ligand_ids)} ligands x {len(ds.receptor_ids)} receptors, "
            f"positives={int(ds.matrix.sum())}"
        )

    start = time.time()
    embedder = make_embedder(args)
    summaries: List[Dict[str, object]] = []
    prediction_paths = []

    for dataset in datasets.values():
        ligand_embeddings = get_or_create_embeddings(
            dataset,
            "ligand",
            dataset.ligand_ids,
            dataset.ligand_sequences,
            embedder,
            args.out_dir,
            args.model_name,
            args.overwrite_embeddings,
        )
        receptor_embeddings = get_or_create_embeddings(
            dataset,
            "receptor",
            dataset.receptor_ids,
            dataset.receptor_sequences,
            embedder,
            args.out_dir,
            args.model_name,
            args.overwrite_embeddings,
        )
        model, scaler, metadata = train_one_dataset(
            dataset, ligand_embeddings, receptor_embeddings, args, args.out_dir
        )
        prediction_paths.append(
            str(
                score_zero_pairs_topk(
                    dataset,
                    ligand_embeddings,
                    receptor_embeddings,
                    model,
                    scaler,
                    args,
                    args.out_dir,
                )
            )
        )
        summaries.append(metadata)
        write_summary(args.out_dir, summaries)

    run_info = {
        "elapsed_seconds": round(time.time() - start, 2),
        "args": vars(args),
        "prediction_paths": prediction_paths,
    }
    with (args.out_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2, default=str)
    write_summary(args.out_dir, summaries)
    print(f"Done. Results written to: {args.out_dir}")


if __name__ == "__main__":
    main()
