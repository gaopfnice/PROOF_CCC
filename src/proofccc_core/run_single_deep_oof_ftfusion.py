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

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent))
from run_cv_multibranch_simmemory import (  # noqa: E402
    best_threshold,
    build_graph_diffusion_features,
    load_global_embeddings_generic,
    load_or_extract_local,
    logit,
    make_loader,
    select_memory_features,
    train_multibranch_fold,
    write_curve_csv,
)
from run_human_multibranch_memory_fusion import (  # noqa: E402
    build_memory_features,
    evaluate as evaluate_probs,
)
from run_human_multibranch_moe import PairCollator, evaluate as evaluate_torch_model, stable_dataset_seed  # noqa: E402
from run_lri_experiment import load_datasets, positive_pairs, sample_negative_pairs  # noqa: E402


@dataclass
class RunConfig:
    datasets: List[str]
    model_name: str
    seed: int
    test_size: float
    neg_ratio: float
    inner_oof_folds: int
    inner_val_size: float
    max_residues: int
    chunk_len: int
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
    topks: List[int]
    svd_rank: int
    memory_feature_mode: str
    graph_topks: List[int]
    graph_self_weight: float
    threshold_metric: str
    threshold_beta: float
    loss_type: str
    bce_mix_weight: float
    asl_gamma_pos: float
    asl_gamma_neg: float
    asl_clip: float
    early_stop_metric: str
    ft_d_token: int
    ft_layers: int
    ft_heads: int
    ft_dropout: float
    ft_lr: float
    ft_weight_decay: float
    ft_epochs: int
    ft_patience: int
    ft_batch_size: int
    ft_focal_gamma: float
    fusion_val_size: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")


class NumericFeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, None] * self.weight[None, :, :] + self.bias[None, :, :]


class FTTransformerFusion(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_token: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.tokenizer = NumericFeatureTokenizer(n_features, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(encoded[:, 0]).squeeze(1)


class FocalBCEWithLogits(nn.Module):
    def __init__(self, gamma: float = 1.5):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        return (((1.0 - pt) ** self.gamma) * bce).mean()


def build_fusion_features(
    eval_pairs: np.ndarray,
    eval_base_probs: np.ndarray,
    ligand_global: np.ndarray,
    receptor_global: np.ndarray,
    train_pairs: np.ndarray,
    train_labels: np.ndarray,
    matrix_shape: Tuple[int, int],
    args,
) -> Tuple[np.ndarray, List[str]]:
    memory_all, memory_names = build_memory_features(
        eval_pairs,
        ligand_global,
        receptor_global,
        train_pairs,
        train_labels,
        matrix_shape,
        args.topks,
        args.svd_rank,
    )
    memory_all, memory_names = select_memory_features(memory_all, memory_names, args.memory_feature_mode)
    graph_all, graph_names = build_graph_diffusion_features(
        eval_pairs,
        ligand_global,
        receptor_global,
        train_pairs,
        train_labels,
        matrix_shape,
        args.graph_topks,
        args.graph_self_weight,
    )
    base = np.concatenate([eval_base_probs[:, None], logit(eval_base_probs)[:, None]], axis=1)
    inter = graph_all * eval_base_probs[:, None]
    features = np.concatenate([base, memory_all, graph_all, inter], axis=1).astype(np.float32)
    names = ["base_prob", "base_logit"] + memory_names + graph_names + [f"base_x_{name}" for name in graph_names]
    return features, names


@torch.no_grad()
def predict_ft(model: nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x.astype(np.float32))), batch_size=batch_size, shuffle=False)
    probs = []
    for (xb,) in loader:
        logits = model(xb.to(device))
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs).astype(np.float32)


def train_ft_fusion(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args,
    seed: int,
) -> Tuple[FTTransformerFusion, Dict[str, object], np.ndarray]:
    set_seed(seed)
    model = FTTransformerFusion(
        n_features=x_train.shape[1],
        d_token=args.ft_d_token,
        n_layers=args.ft_layers,
        n_heads=args.ft_heads,
        dropout=args.ft_dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=args.ft_weight_decay)
    loss_fn = FocalBCEWithLogits(args.ft_focal_gamma)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=args.ft_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state = None
    best_epoch = 0
    best_ap = -math.inf
    patience_left = args.ft_patience
    history = []
    for epoch in range(1, args.ft_epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_probs = predict_ft(model, x_val, args.ft_batch_size, args.device)
        val_ap = float(average_precision_score(y_val, val_probs))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_average_precision": val_ap}
        history.append(row)
        print(f"ft_epoch={epoch:03d} loss={row['loss']:.5f} val_aupr={val_ap:.5f}", flush=True)
        if val_ap > best_ap + 1e-8:
            best_ap = val_ap
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.ft_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"ft early stopping at epoch={epoch}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_probs = predict_ft(model, x_val, args.ft_batch_size, args.device)
    return model, {"best_epoch": best_epoch, "best_val_average_precision": best_ap, "history": history}, val_probs


def write_metrics_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "threshold_metric", "threshold", "roc_auc", "average_precision", "recall", "precision", "accuracy", "f1"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(dataset_name: str, dataset, args, config: RunConfig, esm_model, ESMProtein, LogitsConfig) -> Dict[str, object]:
    dataset_seed = stable_dataset_seed(dataset_name, args.seed)
    dataset_dir = args.out_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"===== DATASET {dataset_name} =====", flush=True)

    ligand_global = load_global_embeddings_generic(args.embedding_dir, dataset_name, "ligand", dataset.ligand_ids, args.model_name)
    receptor_global = load_global_embeddings_generic(args.embedding_dir, dataset_name, "receptor", dataset.receptor_ids, args.model_name)
    ligand_local = load_or_extract_local(
        dataset_name,
        "ligand",
        dataset.ligand_ids,
        dataset.ligand_sequences,
        args.cache_dir,
        esm_model,
        ESMProtein,
        LogitsConfig,
        args.device,
        args.chunk_len,
        args.max_residues,
    )
    receptor_local = load_or_extract_local(
        dataset_name,
        "receptor",
        dataset.receptor_ids,
        dataset.receptor_sequences,
        args.cache_dir,
        esm_model,
        ESMProtein,
        LogitsConfig,
        args.device,
        args.chunk_len,
        args.max_residues,
    )

    pos = positive_pairs(dataset.matrix)
    neg = sample_negative_pairs(dataset.matrix, int(len(pos) * args.neg_ratio), dataset_seed)
    pairs = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)])
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=args.test_size,
        random_state=dataset_seed,
        stratify=labels,
    )

    oof_features = []
    oof_labels = []
    test_feature_ensemble = []
    feature_names: List[str] = []
    inner = StratifiedKFold(n_splits=args.inner_oof_folds, shuffle=True, random_state=dataset_seed + 17)
    train_idx = np.asarray(train_idx)
    for inner_fold, (inner_train_rel, hold_rel) in enumerate(inner.split(train_idx, labels[train_idx])):
        fold_seed = dataset_seed + 1000 + inner_fold * 97
        inner_train_idx = train_idx[inner_train_rel]
        hold_idx = train_idx[hold_rel]
        fit_idx, val_idx = train_test_split(
            inner_train_idx,
            test_size=args.inner_val_size,
            random_state=fold_seed,
            stratify=labels[inner_train_idx],
        )
        print(f"--- {dataset_name} inner_oof_fold={inner_fold + 1}/{args.inner_oof_folds} ---", flush=True)
        model, info, _, _ = train_multibranch_fold(
            ligand_global,
            receptor_global,
            ligand_local,
            receptor_local,
            pairs[fit_idx],
            labels[fit_idx],
            pairs[val_idx],
            labels[val_idx],
            args,
            fold_seed,
        )
        collator = PairCollator(ligand_global, receptor_global, ligand_local, receptor_local)
        hold_loader = make_loader(pairs[hold_idx], labels[hold_idx], args.batch_size, collator, fold_seed, False)
        _, _, hold_base_probs = evaluate_torch_model(model, hold_loader, args.device)
        hold_features, feature_names = build_fusion_features(
            pairs[hold_idx],
            hold_base_probs,
            ligand_global,
            receptor_global,
            pairs[inner_train_idx],
            labels[inner_train_idx],
            dataset.matrix.shape,
            args,
        )
        oof_features.append(hold_features)
        oof_labels.append(labels[hold_idx])

        test_loader = make_loader(pairs[test_idx], labels[test_idx], args.batch_size, collator, fold_seed, False)
        _, _, fold_test_base_probs = evaluate_torch_model(model, test_loader, args.device)
        fold_test_features, _ = build_fusion_features(
            pairs[test_idx],
            fold_test_base_probs,
            ligand_global,
            receptor_global,
            pairs[inner_train_idx],
            labels[inner_train_idx],
            dataset.matrix.shape,
            args,
        )
        test_feature_ensemble.append(fold_test_features)

        if args.save_base_models:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "dataset": dataset_name,
                    "inner_fold": inner_fold,
                    "config": asdict(config),
                    "train_info": info,
                },
                dataset_dir / f"inner_base_model_fold{inner_fold}.pt",
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    x_oof = np.concatenate(oof_features, axis=0).astype(np.float32)
    y_oof = np.concatenate(oof_labels, axis=0).astype(np.int64)
    x_test = np.mean(np.stack(test_feature_ensemble, axis=0), axis=0).astype(np.float32)
    y_test = labels[test_idx].astype(np.int64)
    np.savez_compressed(
        dataset_dir / "feature_bank.npz",
        y_oof=y_oof,
        x_oof=x_oof,
        y_test=y_test,
        x_test=x_test,
        test_pairs=pairs[test_idx],
        feature_names=np.asarray(feature_names),
    )
    if args.feature_only:
        result = {
            "dataset": dataset_name,
            "model": "DeepOOF_FeatureBank",
            "feature_count": int(x_oof.shape[1]),
            "train_oof_samples": int(len(y_oof)),
            "test_samples": int(len(y_test)),
            "feature_bank_path": str(dataset_dir / "feature_bank.npz"),
        }
        with (dataset_dir / "feature_bank.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(
            f"FEATURE_BANK {dataset_name} train_oof={len(y_oof)} test={len(y_test)} features={x_oof.shape[1]}",
            flush=True,
        )
        return result

    fusion_train_idx, fusion_val_idx = train_test_split(
        np.arange(len(y_oof)),
        test_size=args.fusion_val_size,
        random_state=dataset_seed + 777,
        stratify=y_oof,
    )
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x_oof[fusion_train_idx]).astype(np.float32)
    x_val = scaler.transform(x_oof[fusion_val_idx]).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    print(f"training FT-Transformer fusion: train={len(fusion_train_idx)} val={len(fusion_val_idx)} test={len(y_test)} features={x_oof.shape[1]}", flush=True)
    ft_model, ft_info, val_probs = train_ft_fusion(
        x_fit,
        y_oof[fusion_train_idx],
        x_val,
        y_oof[fusion_val_idx],
        args,
        dataset_seed + 20000,
    )
    test_probs = predict_ft(ft_model, x_test_scaled, args.ft_batch_size, args.device)
    threshold = best_threshold(y_oof[fusion_val_idx], val_probs, args.threshold_metric, args.threshold_beta)
    test_metrics = evaluate_probs(y_test, test_probs, threshold)

    alt_rows = []
    for threshold_metric in ["rate0.92", "rate1.0", "f1", "balance4", "accuracy", "fbeta1.3"]:
        alt_threshold = best_threshold(y_oof[fusion_val_idx], val_probs, threshold_metric, args.threshold_beta)
        metrics = evaluate_probs(y_test, test_probs, alt_threshold)
        alt_rows.append(
            {
                "dataset": dataset_name,
                "threshold_metric": threshold_metric,
                "threshold": alt_threshold,
                **{key: metrics[key] for key in ["roc_auc", "average_precision", "recall", "precision", "accuracy", "f1"]},
            }
        )

    torch.save(
        {
            "model_state": ft_model.state_dict(),
            "feature_names": feature_names,
            "config": asdict(config),
            "ft_info": ft_info,
        },
        dataset_dir / "ftfusion_model.pt",
    )
    joblib.dump({"scaler": scaler, "feature_names": feature_names, "config": asdict(config)}, dataset_dir / "feature_scaler.joblib")
    np.savez_compressed(
        dataset_dir / "predictions.npz",
        y_oof=y_oof,
        x_oof=x_oof,
        y_test=y_test,
        x_test=x_test,
        x_test_scaled=x_test_scaled,
        test_pairs=pairs[test_idx],
        test_probs=test_probs,
        val_probs=val_probs,
        y_val=y_oof[fusion_val_idx],
        feature_names=np.asarray(feature_names),
    )
    write_curve_csv(dataset_dir / "curves", "ftfusion_test", y_test, test_probs)
    write_metrics_csv(dataset_dir / "threshold_sweep_metrics.csv", alt_rows)
    result = {
        "dataset": dataset_name,
        "model": "DeepOOF_FTTransformerFusion",
        "threshold_metric": args.threshold_metric,
        "threshold": float(threshold),
        "test": test_metrics,
        "ft_info": ft_info,
        "feature_count": int(x_oof.shape[1]),
        "feature_names": feature_names,
        "threshold_sweep": alt_rows,
    }
    with (dataset_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(
        f"RESULT {dataset_name} AUC={test_metrics['roc_auc']:.6f} AUPR={test_metrics['average_precision']:.6f} "
        f"REC={test_metrics['recall']:.6f} PRE={test_metrics['precision']:.6f} "
        f"ACC={test_metrics['accuracy']:.6f} F1={test_metrics['f1']:.6f}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("outputs/lri_esmc_600m/embeddings"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/local_cache_esmc"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/single_deep_oof_ftfusion"))
    parser.add_argument("--datasets", nargs="+", default=["human", "mouse", "mouse-heart"])
    parser.add_argument("--model-name", default="esmc_600m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--inner-oof-folds", type=int, default=5)
    parser.add_argument("--inner-val-size", type=float, default=0.18)
    parser.add_argument("--max-residues", type=int, default=512)
    parser.add_argument("--chunk-len", type=int, default=1022)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--protein-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--pair-attention-dim", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--aux-loss-weight", type=float, default=0.15)
    parser.add_argument("--rank-loss-weight", type=float, default=0.02)
    parser.add_argument("--positive-target", type=float, default=1.0)
    parser.add_argument("--negative-target", type=float, default=0.0)
    parser.add_argument("--topks", type=int, nargs="+", default=[5, 10, 25, 50])
    parser.add_argument("--svd-rank", type=int, default=64)
    parser.add_argument("--memory-feature-mode", choices=["all", "positive_only"], default="positive_only")
    parser.add_argument("--graph-topks", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    parser.add_argument("--graph-self-weight", type=float, default=0.35)
    parser.add_argument("--threshold-metric", default="rate1.0")
    parser.add_argument("--threshold-beta", type=float, default=1.0)
    parser.add_argument("--loss-type", choices=["focal", "bce", "bce_focal", "asl"], default="focal")
    parser.add_argument("--bce-mix-weight", type=float, default=0.50)
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asl-gamma-neg", type=float, default=2.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--early-stop-metric", choices=["average_precision", "f1", "accuracy", "paf"], default="average_precision")
    parser.add_argument("--fusion-val-size", type=float, default=0.18)
    parser.add_argument("--ft-d-token", type=int, default=64)
    parser.add_argument("--ft-layers", type=int, default=3)
    parser.add_argument("--ft-heads", type=int, default=4)
    parser.add_argument("--ft-dropout", type=float, default=0.20)
    parser.add_argument("--ft-lr", type=float, default=1e-3)
    parser.add_argument("--ft-weight-decay", type=float, default=1e-4)
    parser.add_argument("--ft-epochs", type=int, default=180)
    parser.add_argument("--ft-patience", type=int, default=25)
    parser.add_argument("--ft-batch-size", type=int, default=256)
    parser.add_argument("--ft-focal-gamma", type=float, default=1.5)
    parser.add_argument("--save-base-models", action="store_true")
    parser.add_argument("--feature-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    config = RunConfig(
        datasets=args.datasets,
        model_name=args.model_name,
        seed=args.seed,
        test_size=args.test_size,
        neg_ratio=args.neg_ratio,
        inner_oof_folds=args.inner_oof_folds,
        inner_val_size=args.inner_val_size,
        max_residues=args.max_residues,
        chunk_len=args.chunk_len,
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
        topks=args.topks,
        svd_rank=args.svd_rank,
        memory_feature_mode=args.memory_feature_mode,
        graph_topks=args.graph_topks,
        graph_self_weight=args.graph_self_weight,
        threshold_metric=args.threshold_metric,
        threshold_beta=args.threshold_beta,
        loss_type=args.loss_type,
        bce_mix_weight=args.bce_mix_weight,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_clip=args.asl_clip,
        early_stop_metric=args.early_stop_metric,
        ft_d_token=args.ft_d_token,
        ft_layers=args.ft_layers,
        ft_heads=args.ft_heads,
        ft_dropout=args.ft_dropout,
        ft_lr=args.ft_lr,
        ft_weight_decay=args.ft_weight_decay,
        ft_epochs=args.ft_epochs,
        ft_patience=args.ft_patience,
        ft_batch_size=args.ft_batch_size,
        ft_focal_gamma=args.ft_focal_gamma,
        fusion_val_size=args.fusion_val_size,
    )
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    print("loading ESM C for residue-level local embeddings", flush=True)
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    datasets = load_datasets(args.data_dir, args.datasets)
    esm_model = ESMC.from_pretrained(args.model_name).to(args.device)
    esm_model.eval()

    results = {}
    for dataset_name, dataset in datasets.items():
        results[dataset_name] = run_dataset(dataset_name, dataset, args, config, esm_model, ESMProtein, LogitsConfig)
    with (args.out_dir / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Done. Results written to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
