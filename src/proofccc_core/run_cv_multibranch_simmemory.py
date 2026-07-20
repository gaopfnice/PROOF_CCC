#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import torch
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))
from run_lri_experiment import load_datasets, positive_pairs, sample_negative_pairs  # noqa: E402
from run_human_multibranch_moe import (  # noqa: E402
    FocalLossWithLogits,
    LRIPairDataset,
    PairAttentionMoELRIClassifier,
    PairCollator,
    PairwiseRankingLoss,
    evaluate as evaluate_torch_model,
    extract_local_embeddings,
    stable_dataset_seed,
)
from run_human_multibranch_memory_fusion import (  # noqa: E402
    build_memory_features,
    evaluate as evaluate_probs,
    oof_predict,
)


@dataclass
class CVConfig:
    datasets: List[str]
    repeats: int
    folds: int
    inner_val_size: float
    neg_ratio: float
    seed: int
    model_name: str
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
    fusion_feature_block: str
    fusion_model_type: str
    gb_learning_rate: float
    gb_max_depth: int
    gb_subsample: float
    gb_n_estimators: int
    select_by: str
    threshold_metric: str
    threshold_beta: float
    loss_type: str
    bce_mix_weight: float
    asl_gamma_pos: float
    asl_gamma_neg: float
    asl_clip: float
    early_stop_metric: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")


class MixedBCEFocalLossWithLogits(nn.Module):
    def __init__(self, bce_weight: float, focal_alpha: float, focal_gamma: float):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLossWithLogits(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        focal_loss = self.focal(logits, targets)
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * focal_loss


class AsymmetricLossWithLogits(nn.Module):
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 2.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pos_probs = probs.clamp(min=self.eps, max=1.0 - self.eps)
        neg_probs = (1.0 - probs + self.clip).clamp(min=self.eps, max=1.0)
        pos_weight = torch.pow(1.0 - probs, self.gamma_pos)
        neg_weight = torch.pow(probs, self.gamma_neg)
        pos_loss = targets * pos_weight * torch.log(pos_probs)
        neg_loss = (1.0 - targets) * neg_weight * torch.log(neg_probs)
        return -(pos_loss + neg_loss).mean()


def make_loss_fn(args) -> nn.Module:
    if args.loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    if args.loss_type == "bce_focal":
        return MixedBCEFocalLossWithLogits(args.bce_mix_weight, args.focal_alpha, args.focal_gamma)
    if args.loss_type == "asl":
        return AsymmetricLossWithLogits(args.asl_gamma_pos, args.asl_gamma_neg, args.asl_clip)
    return FocalLossWithLogits(alpha=args.focal_alpha, gamma=args.focal_gamma)


def load_global_embeddings_generic(embedding_dir: Path, dataset_name: str, role: str, ids: List[str], model_name: str) -> np.ndarray:
    safe_model = model_name.replace("/", "__")
    path = embedding_dir / f"{dataset_name}_{role}_{safe_model}.npz"
    data = np.load(path, allow_pickle=False)
    saved_ids = data["ids"].astype(str).tolist()
    if saved_ids != ids:
        raise ValueError(f"Embedding ID order mismatch for {path}")
    return data["embeddings"].astype(np.float32)


def load_or_extract_local(
    dataset_name: str,
    role: str,
    ids: List[str],
    sequences: Dict[str, str],
    cache_dir: Path,
    model,
    esm_protein,
    logits_config,
    device: str,
    chunk_len: int,
    max_residues: int,
) -> List[torch.Tensor]:
    safe_role = role.replace("/", "_")
    cache_path = cache_dir / f"{dataset_name}_{safe_role}_local_{max_residues}_{chunk_len}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload["ids"] == ids:
            print(f"loaded cached local embeddings: {cache_path}", flush=True)
            return payload["tokens"]
        print(f"cache ID mismatch, rebuilding: {cache_path}", flush=True)
    local = extract_local_embeddings(
        ids,
        sequences,
        model,
        esm_protein,
        logits_config,
        device,
        chunk_len,
        max_residues,
        f"local {dataset_name}/{role}",
    )
    torch.save({"ids": ids, "tokens": local}, cache_path)
    print(f"saved local embedding cache: {cache_path}", flush=True)
    return local


def make_loader(
    pairs: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    collator: PairCollator,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        LRIPairDataset(pairs, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=generator,
        num_workers=0,
    )


def train_multibranch_fold(
    ligand_global: np.ndarray,
    receptor_global: np.ndarray,
    ligand_local: List[torch.Tensor],
    receptor_local: List[torch.Tensor],
    train_pairs: np.ndarray,
    train_labels: np.ndarray,
    val_pairs: np.ndarray,
    val_labels: np.ndarray,
    args,
    fold_seed: int,
) -> Tuple[PairAttentionMoELRIClassifier, Dict[str, object], np.ndarray, np.ndarray]:
    set_seed(fold_seed)
    collator = PairCollator(ligand_global, receptor_global, ligand_local, receptor_local)
    train_loader = make_loader(train_pairs, train_labels, args.batch_size, collator, fold_seed, True)
    val_loader = make_loader(val_pairs, val_labels, args.batch_size, collator, fold_seed, False)
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
    loss_fn = make_loss_fn(args)
    rank_loss_fn = PairwiseRankingLoss()

    best_state = None
    best_epoch = 0
    best_val_score = -math.inf
    patience_left = args.patience
    history = []
    def early_stop_score(metrics: Dict[str, float]) -> float:
        if args.early_stop_metric == "f1":
            return float(metrics["f1"])
        if args.early_stop_metric == "accuracy":
            return float(metrics["accuracy"])
        if args.early_stop_metric == "paf":
            return float((metrics["precision"] + metrics["accuracy"] + metrics["f1"]) / 3.0)
        return float(metrics["average_precision"])

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
            soft_y = torch.where(
                y > 0.5,
                torch.full_like(y, args.positive_target),
                torch.full_like(y, args.negative_target),
            )
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics, _, val_probs = evaluate_torch_model(model, val_loader, args.device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **val_metrics}
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={row['loss']:.5f} "
            f"val_auc={row['roc_auc']:.5f} val_aupr={row['average_precision']:.5f} "
            f"val_f1={row['f1']:.5f}",
            flush=True,
        )
        score = early_stop_score(row)
        if score > best_val_score:
            best_val_score = score
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
    val_metrics, y_val, val_probs = evaluate_torch_model(model, val_loader, args.device)
    info = {
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "early_stop_metric": args.early_stop_metric,
        "val": val_metrics,
        "history": history,
    }
    return model, info, y_val, val_probs


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def best_threshold(y_true: np.ndarray, probs: np.ndarray, metric: str, beta: float = 1.0) -> float:
    if len(probs) == 0:
        return 0.5
    metric_name = metric
    if metric.startswith("fbeta") and metric != "fbeta":
        metric_beta = metric.replace("fbeta", "", 1).lstrip("_")
        if metric_beta:
            try:
                beta = float(metric_beta)
                metric_name = "fbeta"
            except ValueError:
                metric_name = metric
    rate_scale = 1.0
    if metric.startswith("rate") and metric != "rate":
        metric_rate = metric.replace("rate", "", 1).lstrip("_")
        if metric_rate:
            try:
                rate_scale = float(metric_rate)
                metric_name = "rate"
            except ValueError:
                metric_name = metric
    precision_floor = None
    if metric.startswith("pmin") and metric != "pmin":
        metric_floor = metric.replace("pmin", "", 1).lstrip("_")
        if metric_floor:
            try:
                precision_floor = float(metric_floor)
                metric_name = "pmin"
            except ValueError:
                metric_name = metric
    thresholds = np.unique(np.clip(probs.astype(np.float64), 1e-6, 1.0 - 1e-6))
    if len(thresholds) > 512:
        quantiles = np.linspace(0.0, 1.0, 512)
        thresholds = np.unique(np.quantile(thresholds, quantiles))

    y_true = y_true.astype(np.int64)
    positives = max(int(y_true.sum()), 1)
    negatives = max(int((1 - y_true).sum()), 1)
    best_score = -np.inf
    best_t = 0.5
    beta2 = beta * beta
    target_rate = float(y_true.mean())
    for threshold in thresholds:
        pred = probs >= threshold
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
        accuracy = (tp + tn) / len(y_true)
        if metric_name == "accuracy":
            score = accuracy
        elif metric_name == "balanced_accuracy":
            score = 0.5 * (recall + tn / negatives)
        elif metric_name == "mcc":
            denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
            score = ((tp * tn) - (fp * fn)) / denom
        elif metric_name == "fbeta":
            score = (1.0 + beta2) * precision * recall / (beta2 * precision + recall + 1e-12)
        elif metric_name == "f1_acc":
            score = 0.5 * (f1 + accuracy)
        elif metric_name == "paf":
            score = (precision + accuracy + f1) / 3.0
        elif metric_name == "balance4":
            score = (precision + recall + accuracy + f1) / 4.0
        elif metric_name == "geom4":
            score = (max(precision, 1e-12) * max(recall, 1e-12) * max(accuracy, 1e-12) * max(f1, 1e-12)) ** 0.25
        elif metric_name == "min4":
            score = min(precision, recall, accuracy, f1)
        elif metric_name == "prevalence":
            score = -abs(float(pred.mean()) - target_rate)
        elif metric_name == "rate":
            score = -abs(float(pred.mean()) - target_rate * rate_scale)
        elif metric_name == "pmin":
            floor = 0.88 if precision_floor is None else precision_floor
            if precision >= floor:
                score = f1 + 0.01 * recall
            else:
                score = precision - floor
        else:
            score = f1
        # Prefer the slightly more conservative threshold when validation scores tie.
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and threshold > best_t):
            best_score = score
            best_t = float(threshold)
    return best_t


def is_negative_memory_feature(name: str) -> bool:
    return "_neg_" in name or name.endswith("_neg_degree") or "neg_degree" in name


def select_memory_features(
    memory_all: np.ndarray,
    memory_names: List[str],
    mode: str,
) -> Tuple[np.ndarray, List[str]]:
    if mode == "all":
        return memory_all, memory_names
    if mode != "positive_only":
        raise ValueError(f"unknown memory_feature_mode: {mode}")
    keep = np.asarray([not is_negative_memory_feature(name) for name in memory_names], dtype=bool)
    return memory_all[:, keep], [name for name, use_feature in zip(memory_names, keep) if use_feature]


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def build_transition(embeddings: np.ndarray, topk: int, self_weight: float) -> np.ndarray:
    norm = normalize_rows(embeddings)
    sim = norm @ norm.T
    n = sim.shape[0]
    if n <= 1:
        return np.eye(n, dtype=np.float32)
    work = sim.copy()
    np.fill_diagonal(work, -np.inf)
    k = min(max(int(topk), 1), n - 1)
    idx = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    order = np.argsort(-work[rows, idx], axis=1)
    idx = idx[rows, order]
    weights = np.clip(work[rows, idx], 0.0, None).astype(np.float32)

    transition = np.zeros((n, n), dtype=np.float32)
    transition[rows, idx] = weights
    if self_weight > 0:
        transition[np.arange(n), np.arange(n)] = float(self_weight)
    row_sum = transition.sum(axis=1, keepdims=True)
    empty = row_sum[:, 0] <= 1e-8
    if np.any(empty):
        transition[empty, :] = 0.0
        transition[empty, np.arange(n)[empty]] = 1.0
        row_sum = transition.sum(axis=1, keepdims=True)
    return transition / (row_sum + 1e-8)


def build_graph_diffusion_features(
    selected_pairs: np.ndarray,
    ligand_global: np.ndarray,
    receptor_global: np.ndarray,
    train_pairs: np.ndarray,
    train_labels: np.ndarray,
    shape: Tuple[int, int],
    graph_topks: Sequence[int],
    self_weight: float,
) -> Tuple[np.ndarray, List[str]]:
    pos_matrix = np.zeros(shape, dtype=np.float32)
    for (ligand_idx, receptor_idx), label in zip(train_pairs, train_labels):
        if int(label) == 1:
            pos_matrix[int(ligand_idx), int(receptor_idx)] = 1.0

    ligand_idx = selected_pairs[:, 0].astype(np.int64)
    receptor_idx = selected_pairs[:, 1].astype(np.int64)
    columns = []
    names: List[str] = []

    ligand_degree = pos_matrix.sum(axis=1)
    receptor_degree = pos_matrix.sum(axis=0)
    density = float(pos_matrix.sum() / max(pos_matrix.size, 1))
    for name, values in [
        ("graph_ligand_degree_log", np.log1p(ligand_degree[ligand_idx])),
        ("graph_receptor_degree_log", np.log1p(receptor_degree[receptor_idx])),
        (
            "graph_expected_degree_score",
            (ligand_degree[ligand_idx] + 1.0) * (receptor_degree[receptor_idx] + 1.0) * density,
        ),
    ]:
        columns.append(values[:, None].astype(np.float32))
        names.append(name)

    for topk in graph_topks:
        wl = build_transition(ligand_global, topk, self_weight)
        wr = build_transition(receptor_global, topk, self_weight)
        ligand_only = wl @ pos_matrix
        receptor_only = pos_matrix @ wr.T
        two_side = wl @ pos_matrix @ wr.T
        left = ligand_only[ligand_idx, receptor_idx]
        right = receptor_only[ligand_idx, receptor_idx]
        both = two_side[ligand_idx, receptor_idx]
        feature_items = [
            (f"graph_ligand_diffusion_k{topk}", left),
            (f"graph_receptor_diffusion_k{topk}", right),
            (f"graph_twoside_diffusion_k{topk}", both),
            (f"graph_diffusion_mean_k{topk}", (left + right + both) / 3.0),
            (f"graph_diffusion_max_k{topk}", np.maximum(np.maximum(left, right), both)),
            (f"graph_diffusion_agreement_k{topk}", np.minimum(np.minimum(left, right), both)),
            (f"graph_diffusion_lr_gap_k{topk}", np.abs(left - right)),
        ]
        for name, values in feature_items:
            columns.append(values[:, None].astype(np.float32))
            names.append(name)
    return np.concatenate(columns, axis=1).astype(np.float32), names


def train_strict_fusion(
    val_base_probs: np.ndarray,
    test_base_probs: np.ndarray,
    val_pairs: np.ndarray,
    test_pairs: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    ligand_global: np.ndarray,
    receptor_global: np.ndarray,
    memory_train_pairs: np.ndarray,
    memory_train_labels: np.ndarray,
    matrix_shape: Tuple[int, int],
    args,
    fold_seed: int,
) -> Tuple[object, Dict[str, object], np.ndarray, np.ndarray]:
    all_eval_pairs = np.concatenate([val_pairs, test_pairs], axis=0)
    memory_all, memory_names = build_memory_features(
        all_eval_pairs,
        ligand_global,
        receptor_global,
        memory_train_pairs,
        memory_train_labels,
        matrix_shape,
        args.topks,
        args.svd_rank,
    )
    memory_all, memory_names = select_memory_features(memory_all, memory_names, args.memory_feature_mode)
    memory_val = memory_all[: len(val_pairs)]
    memory_test = memory_all[len(val_pairs) :]
    base_val = np.concatenate([val_base_probs[:, None], logit(val_base_probs)[:, None]], axis=1)
    base_test = np.concatenate([test_base_probs[:, None], logit(test_base_probs)[:, None]], axis=1)
    feature_names = ["base_prob", "base_logit"] + memory_names
    if args.fusion_feature_block == "base_memory":
        x_val = np.concatenate([base_val, memory_val], axis=1)
        x_test = np.concatenate([base_test, memory_test], axis=1)
    elif args.fusion_feature_block == "base_memory_graph_inter":
        graph_all, graph_names = build_graph_diffusion_features(
            all_eval_pairs,
            ligand_global,
            receptor_global,
            memory_train_pairs,
            memory_train_labels,
            matrix_shape,
            args.graph_topks,
            args.graph_self_weight,
        )
        graph_val = graph_all[: len(val_pairs)]
        graph_test = graph_all[len(val_pairs) :]
        graph_interact_val = graph_val * val_base_probs[:, None]
        graph_interact_test = graph_test * test_base_probs[:, None]
        x_val = np.concatenate([base_val, memory_val, graph_val, graph_interact_val], axis=1)
        x_test = np.concatenate([base_test, memory_test, graph_test, graph_interact_test], axis=1)
        feature_names = feature_names + graph_names + [f"base_x_{name}" for name in graph_names]
    else:
        raise ValueError(f"unknown fusion_feature_block: {args.fusion_feature_block}")

    candidates = []
    if args.fusion_model_type == "gb_fixed":
        candidates.append(
            (
                f"gb_lr{args.gb_learning_rate}_d{args.gb_max_depth}_sub{args.gb_subsample}_n{args.gb_n_estimators}",
                GradientBoostingClassifier(
                    n_estimators=args.gb_n_estimators,
                    learning_rate=args.gb_learning_rate,
                    max_depth=args.gb_max_depth,
                    subsample=args.gb_subsample,
                    random_state=fold_seed,
                ),
            )
        )
    elif args.fusion_model_type == "logreg":
        for c_value in args.fusion_c_values:
            candidates.append(
                (
                    f"logreg_C{c_value}",
                    make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=c_value, max_iter=3000, class_weight="balanced"),
                    ),
                )
            )
    else:
        raise ValueError(f"unknown fusion_model_type: {args.fusion_model_type}")

    best = None
    best_model = None
    best_val_probs = None
    all_results = []
    for name, model in candidates:
        oof_probs, cv_ap = oof_predict(model, x_val, y_val, fold_seed)
        threshold = best_threshold(y_val, oof_probs, args.threshold_metric, args.threshold_beta)
        cv_val_metrics = evaluate_probs(y_val, oof_probs, threshold)
        fitted = clone(model)
        fitted.fit(x_val, y_val)
        fit_val_probs = fitted.predict_proba(x_val)[:, 1]
        record = {
            "name": name,
            "cv_val_average_precision": cv_ap,
            "cv_val": cv_val_metrics,
            "fit_val_average_precision": float(average_precision_score(y_val, fit_val_probs)),
            "threshold": float(threshold),
        }
        all_results.append(record)
        if args.select_by == "cv_aupr":
            score = cv_ap
        elif args.select_by == "val_accuracy":
            score = cv_val_metrics["accuracy"]
        elif args.select_by == "val_paf":
            score = (cv_val_metrics["precision"] + cv_val_metrics["accuracy"] + cv_val_metrics["f1"]) / 3.0
        elif args.select_by == "val_balance4":
            score = (
                cv_val_metrics["precision"]
                + cv_val_metrics["recall"]
                + cv_val_metrics["accuracy"]
                + cv_val_metrics["f1"]
            ) / 4.0
        else:
            score = cv_val_metrics["f1"]
        if best is None or score > best["selection_score"]:
            best = {**record, "selection_score": float(score)}
            best_model = fitted
            best_val_probs = fit_val_probs

    assert best is not None and best_model is not None and best_val_probs is not None
    best_test_probs = best_model.predict_proba(x_test)[:, 1]
    best_test_metrics = evaluate_probs(y_test, best_test_probs, best["threshold"])
    result = {
        "selected_model": best["name"],
        "selected_by": args.select_by,
        "threshold": best["threshold"],
        "threshold_metric": args.threshold_metric,
        "threshold_beta": float(args.threshold_beta),
        "memory_feature_mode": args.memory_feature_mode,
        "fusion_feature_block": args.fusion_feature_block,
        "fusion_model_type": args.fusion_model_type,
        "graph_topks": list(args.graph_topks),
        "graph_self_weight": float(args.graph_self_weight),
        "feature_names": feature_names,
        "val": best["cv_val"],
        "test": best_test_metrics,
        "all_results": all_results,
    }
    return best_model, result, best_val_probs, best_test_probs


def aggregate_metrics(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    keys = ["roc_auc", "average_precision", "recall", "precision", "accuracy", "f1"]
    out = {}
    for key in keys:
        values = np.asarray([row["test"][key] for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return out


def write_rows_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_best_fold_artifacts(dataset_dir: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    best = max(rows, key=lambda row: float(row["test"]["average_precision"]))
    best_dir = dataset_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)

    src_fold_dir = dataset_dir / f"repeat_{int(best['repeat']):02d}" / f"fold_{int(best['fold']):02d}"
    copy_items = [
        (Path(best.get("model_path", "")), best_dir / "multibranch_model.pt"),
        (Path(best.get("fusion_path", "")), best_dir / "fusion_model.joblib"),
        (src_fold_dir / "metrics.json", best_dir / "metrics.json"),
        (src_fold_dir / "predictions.npz", best_dir / "predictions.npz"),
    ]
    for src, dst in copy_items:
        if src and src.is_file():
            shutil.copy2(src, dst)
    curve_src = src_fold_dir / "curves"
    if curve_src.exists():
        shutil.copytree(curve_src, best_dir / "curves", dirs_exist_ok=True)
    with (best_dir / "best_fold.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": best["dataset"],
                "repeat": int(best["repeat"]),
                "fold": int(best["fold"]),
                "selection_metric": "test_average_precision",
                "test": best["test"],
                "model_path": str(best_dir / "multibranch_model.pt"),
                "fusion_path": str(best_dir / "fusion_model.joblib"),
                "source_fold_dir": str(src_fold_dir),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def write_curve_csv(curve_dir: Path, prefix: str, y_true: np.ndarray, probs: np.ndarray) -> None:
    curve_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)

    fpr, tpr, roc_thresholds = roc_curve(y_true, probs)
    with (curve_dir / f"{prefix}_roc_curve.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fpr", "tpr", "threshold"])
        for i, (fpr_value, tpr_value) in enumerate(zip(fpr, tpr)):
            threshold = roc_thresholds[i] if i < len(roc_thresholds) else ""
            writer.writerow([f"{fpr_value:.10g}", f"{tpr_value:.10g}", f"{threshold:.10g}"])

    precision, recall, pr_thresholds = precision_recall_curve(y_true, probs)
    with (curve_dir / f"{prefix}_pr_curve.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["precision", "recall", "threshold"])
        for i, (precision_value, recall_value) in enumerate(zip(precision, recall)):
            threshold = pr_thresholds[i] if i < len(pr_thresholds) else ""
            writer.writerow([f"{precision_value:.10g}", f"{recall_value:.10g}", "" if threshold == "" else f"{threshold:.10g}"])


def fold_metric_row(result: Dict[str, object]) -> Dict[str, object]:
    t = result["test"]
    fusion = result["fusion"]
    return {
        "dataset": result["dataset"],
        "repeat": result["repeat"],
        "fold": result["fold"],
        "selected_model": fusion["selected_model"],
        "selected_by": fusion["selected_by"],
        "threshold_metric": fusion["threshold_metric"],
        "threshold": fusion["threshold"],
        "best_epoch": result["best_epoch"],
        "roc_auc": t["roc_auc"],
        "average_precision": t["average_precision"],
        "recall": t["recall"],
        "precision": t["precision"],
        "accuracy": t["accuracy"],
        "f1": t["f1"],
        "model_path": result["model_path"],
        "fusion_path": result["fusion_path"],
    }


def write_dataset_artifacts(dataset_dir: Path, dataset_name: str, config: CVConfig, rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    aggregate = aggregate_metrics(rows) if rows else {}
    csv_dir = dataset_dir / "csv"
    curves_dir = dataset_dir / "curves"
    csv_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = [fold_metric_row(row) for row in rows]
    fold_fields = [
        "dataset",
        "repeat",
        "fold",
        "selected_model",
        "selected_by",
        "threshold_metric",
        "threshold",
        "best_epoch",
        "roc_auc",
        "average_precision",
        "recall",
        "precision",
        "accuracy",
        "f1",
        "model_path",
        "fusion_path",
    ]
    write_rows_csv(csv_dir / "fold_metrics.csv", fold_fields, fold_rows)

    aggregate_rows = []
    for metric_name, stats in aggregate.items():
        aggregate_rows.append(
            {
                "dataset": dataset_name,
                "metric": metric_name,
                "mean": stats.get("mean", 0.0),
                "std": stats.get("std", 0.0),
            }
        )
    write_rows_csv(csv_dir / "aggregate_metrics.csv", ["dataset", "metric", "mean", "std"], aggregate_rows)

    y_all = []
    base_all = []
    fusion_all = []
    pair_all = []
    split_rows = []
    for result in rows:
        pred_path = dataset_dir / f"repeat_{int(result['repeat']):02d}" / f"fold_{int(result['fold']):02d}" / "predictions.npz"
        if not pred_path.exists():
            continue
        pred = np.load(pred_path)
        y_test = np.asarray(pred["y_test"], dtype=np.int64)
        base_probs = np.asarray(pred["base_test_probs"], dtype=np.float32)
        fusion_probs = np.asarray(pred["fusion_test_probs"], dtype=np.float32)
        test_pairs = np.asarray(pred["test_pairs"], dtype=np.int64)
        y_all.append(y_test)
        base_all.append(base_probs)
        fusion_all.append(fusion_probs)
        pair_all.append(test_pairs)
        for idx, ((ligand_idx, receptor_idx), label, base_prob, fusion_prob) in enumerate(
            zip(test_pairs, y_test, base_probs, fusion_probs)
        ):
            split_rows.append(
                {
                    "dataset": dataset_name,
                    "repeat": result["repeat"],
                    "fold": result["fold"],
                    "sample_index": idx,
                    "ligand_idx": int(ligand_idx),
                    "receptor_idx": int(receptor_idx),
                    "label": int(label),
                    "base_prob": float(base_prob),
                    "fusion_prob": float(fusion_prob),
                }
            )
    if y_all:
        y_concat = np.concatenate(y_all)
        base_concat = np.concatenate(base_all)
        fusion_concat = np.concatenate(fusion_all)
        pair_concat = np.concatenate(pair_all)
        np.savez_compressed(
            dataset_dir / "all_test_predictions.npz",
            y_test=y_concat,
            base_test_probs=base_concat,
            fusion_test_probs=fusion_concat,
            test_pairs=pair_concat,
        )
        write_curve_csv(curves_dir, "base_all_test", y_concat, base_concat)
        write_curve_csv(curves_dir, "fusion_all_test", y_concat, fusion_concat)
    prediction_fields = [
        "dataset",
        "repeat",
        "fold",
        "sample_index",
        "ligand_idx",
        "receptor_idx",
        "label",
        "base_prob",
        "fusion_prob",
    ]
    write_rows_csv(csv_dir / "test_predictions.csv", prediction_fields, split_rows)
    write_best_fold_artifacts(dataset_dir, rows)

    dataset_result = {"dataset": dataset_name, "config": asdict(config), "aggregate": aggregate, "folds": rows}
    with (dataset_dir / "cv_results.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_result, f, ensure_ascii=False, indent=2)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("outputs/lri_esmc_600m/embeddings"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cv_multibranch_simmemory"))
    parser.add_argument("--datasets", nargs="+", default=["human", "mouse", "mouse-heart"])
    parser.add_argument("--model-name", default="esmc_600m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-residues", type=int, default=512)
    parser.add_argument("--chunk-len", type=int, default=1022)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-val-size", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=20260520)
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
    parser.add_argument("--fusion-c-values", type=float, nargs="+", default=[0.03, 0.06, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--memory-feature-mode", choices=["all", "positive_only"], default="all")
    parser.add_argument("--graph-topks", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    parser.add_argument("--graph-self-weight", type=float, default=0.35)
    parser.add_argument(
        "--fusion-feature-block",
        choices=["base_memory", "base_memory_graph_inter"],
        default="base_memory",
    )
    parser.add_argument("--fusion-model-type", choices=["logreg", "gb_fixed"], default="logreg")
    parser.add_argument("--gb-learning-rate", type=float, default=0.03)
    parser.add_argument("--gb-max-depth", type=int, default=2)
    parser.add_argument("--gb-subsample", type=float, default=0.75)
    parser.add_argument("--gb-n-estimators", type=int, default=260)
    parser.add_argument(
        "--select-by",
        choices=["cv_aupr", "val_f1", "val_accuracy", "val_paf", "val_balance4"],
        default="val_f1",
    )
    parser.add_argument("--threshold-metric", default="f1")
    parser.add_argument("--threshold-beta", type=float, default=1.0)
    parser.add_argument("--loss-type", choices=["focal", "bce", "bce_focal", "asl"], default="focal")
    parser.add_argument("--bce-mix-weight", type=float, default=0.50)
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asl-gamma-neg", type=float, default=2.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument(
        "--early-stop-metric",
        choices=["average_precision", "f1", "accuracy", "paf"],
        default="average_precision",
    )
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--start-repeat", type=int, default=0)
    parser.add_argument("--max-repeat", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.out_dir / "local_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = CVConfig(
        datasets=args.datasets,
        repeats=args.repeats,
        folds=args.folds,
        inner_val_size=args.inner_val_size,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
        model_name=args.model_name,
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
        fusion_feature_block=args.fusion_feature_block,
        fusion_model_type=args.fusion_model_type,
        gb_learning_rate=args.gb_learning_rate,
        gb_max_depth=args.gb_max_depth,
        gb_subsample=args.gb_subsample,
        gb_n_estimators=args.gb_n_estimators,
        select_by=args.select_by,
        threshold_metric=args.threshold_metric,
        threshold_beta=args.threshold_beta,
        loss_type=args.loss_type,
        bce_mix_weight=args.bce_mix_weight,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_clip=args.asl_clip,
        early_stop_metric=args.early_stop_metric,
    )
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    datasets = load_datasets(args.data_dir, args.datasets)

    print("loading ESM C for residue-level local embeddings", flush=True)
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    esm_model = ESMC.from_pretrained(args.model_name).to(args.device)
    esm_model.eval()

    all_dataset_summaries = {}
    repeat_stop = args.repeats if args.max_repeat is None else min(args.repeats, args.max_repeat)
    for dataset_name, dataset in datasets.items():
        dataset_dir = args.out_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"===== DATASET {dataset_name} =====", flush=True)
        ligand_global = load_global_embeddings_generic(
            args.embedding_dir, dataset_name, "ligand", dataset.ligand_ids, args.model_name
        )
        receptor_global = load_global_embeddings_generic(
            args.embedding_dir, dataset_name, "receptor", dataset.receptor_ids, args.model_name
        )
        ligand_local = load_or_extract_local(
            dataset_name,
            "ligand",
            dataset.ligand_ids,
            dataset.ligand_sequences,
            cache_dir,
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
            cache_dir,
            esm_model,
            ESMProtein,
            LogitsConfig,
            args.device,
            args.chunk_len,
            args.max_residues,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pos = positive_pairs(dataset.matrix)
        dataset_rows: List[Dict[str, object]] = []
        dataset_summary_path = dataset_dir / "summary.csv"
        if not dataset_summary_path.exists():
            with dataset_summary_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "dataset",
                        "repeat",
                        "fold",
                        "selected_model",
                        "best_epoch",
                        "roc_auc",
                        "average_precision",
                        "recall",
                        "precision",
                        "accuracy",
                        "f1",
                        "threshold",
                        "model_path",
                        "fusion_path",
                    ]
                )

        for repeat in range(args.start_repeat, repeat_stop):
            repeat_seed = stable_dataset_seed(dataset_name, args.seed + repeat * 1009)
            neg = sample_negative_pairs(dataset.matrix, int(len(pos) * args.neg_ratio), repeat_seed)
            pairs = np.concatenate([pos, neg], axis=0)
            labels = np.concatenate([np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)])
            splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=repeat_seed)
            for fold, (outer_train_idx, test_idx) in enumerate(splitter.split(pairs, labels)):
                fold_seed = repeat_seed + fold * 97
                fold_dir = dataset_dir / f"repeat_{repeat:02d}" / f"fold_{fold:02d}"
                metrics_path = fold_dir / "metrics.json"
                if metrics_path.exists():
                    print(f"skip existing {dataset_name} repeat={repeat} fold={fold}", flush=True)
                    with metrics_path.open("r", encoding="utf-8") as f:
                        dataset_rows.append(json.load(f))
                    continue
                fold_dir.mkdir(parents=True, exist_ok=True)
                print(f"--- {dataset_name} repeat={repeat + 1}/{args.repeats} fold={fold + 1}/{args.folds} ---", flush=True)
                fit_idx, val_idx = train_test_split(
                    outer_train_idx,
                    test_size=args.inner_val_size,
                    random_state=fold_seed,
                    stratify=labels[outer_train_idx],
                )
                model, train_info, y_val_base, val_base_probs = train_multibranch_fold(
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
                test_loader = make_loader(pairs[test_idx], labels[test_idx], args.batch_size, collator, fold_seed, False)
                base_test_metrics, y_test, test_base_probs = evaluate_torch_model(model, test_loader, args.device)

                fusion_model, fusion_info, val_fusion_probs, test_fusion_probs = train_strict_fusion(
                    val_base_probs,
                    test_base_probs,
                    pairs[val_idx],
                    pairs[test_idx],
                    labels[val_idx],
                    labels[test_idx],
                    ligand_global,
                    receptor_global,
                    pairs[fit_idx],
                    labels[fit_idx],
                    dataset.matrix.shape,
                    args,
                    fold_seed,
                )

                model_path = fold_dir / "multibranch_model.pt"
                fusion_path = fold_dir / "fusion_model.joblib"
                if args.save_models:
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "dataset": dataset_name,
                            "repeat": repeat,
                            "fold": fold,
                            "config": asdict(config),
                            "train_info": train_info,
                        },
                        model_path,
                    )
                    joblib.dump(
                        {
                            "model": fusion_model,
                            "fusion_info": fusion_info,
                            "dataset": dataset_name,
                            "repeat": repeat,
                            "fold": fold,
                            "config": asdict(config),
                        },
                        fusion_path,
                    )
                else:
                    model_path = Path("")
                    fusion_path = Path("")

                result = {
                    "dataset": dataset_name,
                    "repeat": repeat,
                    "fold": fold,
                    "model": "ESMC_MultiBranchRank_SimMemoryFusion_CV",
                    "matrix_shape": list(dataset.matrix.shape),
                    "matrix_positives": int(dataset.matrix.sum()),
                    "positive_samples": int(len(pos)),
                    "negative_samples": int(len(neg)),
                    "fit_samples": int(len(fit_idx)),
                    "val_samples": int(len(val_idx)),
                    "test_samples": int(len(test_idx)),
                    "best_epoch": int(train_info["best_epoch"]),
                    "base_val": train_info["val"],
                    "base_test": base_test_metrics,
                    "fusion": fusion_info,
                    "test": fusion_info["test"],
                    "model_path": str(model_path),
                    "fusion_path": str(fusion_path),
                }
                with metrics_path.open("w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                np.savez_compressed(
                    fold_dir / "predictions.npz",
                    y_val=labels[val_idx],
                    base_val_probs=val_base_probs,
                    fusion_val_probs=val_fusion_probs,
                    val_pairs=pairs[val_idx],
                    y_test=labels[test_idx],
                    base_test_probs=test_base_probs,
                    fusion_test_probs=test_fusion_probs,
                    test_pairs=pairs[test_idx],
                )
                write_curve_csv(fold_dir / "curves", "base_test", labels[test_idx], test_base_probs)
                write_curve_csv(fold_dir / "curves", "fusion_test", labels[test_idx], test_fusion_probs)
                dataset_rows.append(result)
                t = result["test"]
                with dataset_summary_path.open("a", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            dataset_name,
                            repeat,
                            fold,
                            fusion_info["selected_model"],
                            train_info["best_epoch"],
                            f"{t['roc_auc']:.6f}",
                            f"{t['average_precision']:.6f}",
                            f"{t['recall']:.6f}",
                            f"{t['precision']:.6f}",
                            f"{t['accuracy']:.6f}",
                            f"{t['f1']:.6f}",
                            f"{fusion_info['threshold']:.6f}",
                            str(model_path),
                            str(fusion_path),
                        ]
                    )
                print(
                    f"FOLD_RESULT {dataset_name} repeat={repeat} fold={fold} "
                    f"AUC={t['roc_auc']:.6f} AUPR={t['average_precision']:.6f} "
                    f"REC={t['recall']:.6f} PRE={t['precision']:.6f} "
                    f"ACC={t['accuracy']:.6f} F1={t['f1']:.6f}",
                    flush=True,
                )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        aggregate = write_dataset_artifacts(dataset_dir, dataset_name, config, dataset_rows)
        all_dataset_summaries[dataset_name] = aggregate
        print(f"===== SUMMARY {dataset_name} {json.dumps(aggregate, ensure_ascii=False)} =====", flush=True)

    with (args.out_dir / "all_dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_dataset_summaries, f, ensure_ascii=False, indent=2)
    all_csv_rows = []
    for dataset_name, aggregate in all_dataset_summaries.items():
        for metric_name, stats in aggregate.items():
            all_csv_rows.append(
                {
                    "dataset": dataset_name,
                    "metric": metric_name,
                    "mean": stats.get("mean", 0.0),
                    "std": stats.get("std", 0.0),
                }
            )
    write_rows_csv(args.out_dir / "csv" / "all_dataset_summary.csv", ["dataset", "metric", "mean", "std"], all_csv_rows)
    print(f"Done. CV results written to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
