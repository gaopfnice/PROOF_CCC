from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_KEYS = [
    "roc_auc",
    "average_precision",
    "recall",
    "precision",
    "accuracy",
    "f1",
    "brier",
    "ece",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ece_score(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    y_true = y_true.astype(np.float64)
    probs = np.clip(probs.astype(np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(probs), 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not np.any(mask):
            continue
        conf = float(probs[mask].mean())
        acc = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(conf - acc)
    return float(ece)


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = y_true.astype(np.int64)
    probs = np.clip(probs.astype(np.float64), 1e-12, 1.0 - 1e-12)
    pred = probs >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else float("nan"),
        "average_precision": float(average_precision_score(y_true, probs)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": ece_score(y_true, probs),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_threshold(y_true: np.ndarray, probs: np.ndarray, metric: str = "balance4", beta: float = 1.0) -> float:
    y_true = y_true.astype(np.int64)
    probs = np.clip(probs.astype(np.float64), 1e-6, 1.0 - 1e-6)
    thresholds = np.unique(probs)
    if len(thresholds) > 512:
        thresholds = np.unique(np.quantile(thresholds, np.linspace(0.0, 1.0, 512)))
    positives = max(int(y_true.sum()), 1)
    negatives = max(int((1 - y_true).sum()), 1)
    target_rate = float(y_true.mean())
    metric_name = metric
    beta2 = beta * beta
    best_score = -np.inf
    best_t = 0.5
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
        elif metric_name == "f1":
            score = f1
        elif metric_name == "fbeta":
            score = (1.0 + beta2) * precision * recall / (beta2 * precision + recall + 1e-12)
        elif metric_name == "paf":
            score = (precision + accuracy + f1) / 3.0
        elif metric_name == "balance4":
            score = (precision + recall + accuracy + f1) / 4.0
        elif metric_name == "min4":
            score = min(precision, recall, accuracy, f1)
        elif metric_name == "prevalence":
            score = -abs(float(pred.mean()) - target_rate)
        else:
            score = f1
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and threshold > best_t):
            best_score = score
            best_t = float(threshold)
    return best_t


def threshold_metrics(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    y_test: np.ndarray,
    test_probs: np.ndarray,
    rules: Sequence[str] = ("fixed_0.5", "balance4", "f1", "paf", "min4"),
) -> List[Dict[str, float]]:
    rows = []
    for rule in rules:
        if rule == "fixed_0.5":
            threshold = 0.5
        else:
            threshold = best_threshold(y_val, val_probs, rule, 1.0)
        rows.append({"threshold_rule": rule, **binary_metrics(y_test, test_probs, threshold)})
    return rows


def topk_metrics(y_true: np.ndarray, probs: np.ndarray, ks: Sequence[int]) -> List[Dict[str, float]]:
    order = np.argsort(-probs)
    total_pos = max(int(y_true.sum()), 1)
    rows = []
    for k in ks:
        kk = min(int(k), len(order))
        if kk <= 0:
            continue
        hit = int(y_true[order[:kk]].sum())
        rows.append(
            {
                "k": kk,
                "precision_at_k": hit / kk,
                "recall_at_k": hit / total_pos,
                "hits_at_k": hit,
                "positives": total_pos,
            }
        )
    return rows


def calibration_bins(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> List[Dict[str, float]]:
    y_true = y_true.astype(np.float64)
    probs = np.clip(probs.astype(np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        rows.append(
            {
                "bin": idx,
                "bin_low": float(lo),
                "bin_high": float(hi),
                "count": count,
                "mean_probability": float(probs[mask].mean()) if count else float("nan"),
                "fraction_positive": float(y_true[mask].mean()) if count else float("nan"),
            }
        )
    return rows


def summarize(rows: Sequence[Dict], group_keys: Sequence[str], metric_keys: Sequence[str] = METRIC_KEYS) -> List[Dict]:
    grouped: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)
    out = []
    for key, items in sorted(grouped.items()):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["n"] = len(items)
        for metric in metric_keys:
            vals = np.asarray([float(item[metric]) for item in items if metric in item and item[metric] != ""], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            rec[f"{metric}_mean"] = float(vals.mean())
            rec[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(rec)
    return out


def list_fold_dirs(root: Path, dataset: str) -> List[Path]:
    dataset_root = root / dataset
    if not dataset_root.exists():
        return []
    folds = sorted(dataset_root.glob("repeat_*/fold_*"))
    return [fold for fold in folds if (fold / "feature_bank.npz").exists()]


def load_feature_bank(fold_dir: Path) -> Dict:
    z = np.load(fold_dir / "feature_bank.npz", allow_pickle=True)
    payload = {
        "x_oof": z["x_oof"].astype(np.float32),
        "y_oof": z["y_oof"].astype(np.int64),
        "x_test": z["x_test"].astype(np.float32),
        "y_test": z["y_test"].astype(np.int64),
        "test_pairs": z["test_pairs"].astype(np.int64),
        "feature_names": z["feature_names"].astype(str).tolist(),
    }
    meta_path = fold_dir / "feature_bank.json"
    if meta_path.exists():
        payload["meta"] = load_json(meta_path)
    return payload


def parse_repeat_fold(fold_dir: Path) -> Tuple[int, int]:
    repeat = int(fold_dir.parent.name.split("_")[-1])
    fold = int(fold_dir.name.split("_")[-1])
    return repeat, fold


def feature_mask(feature_names: Sequence[str], variant: str) -> np.ndarray:
    names = list(feature_names)
    is_base = np.asarray([name in {"base_prob", "base_logit"} for name in names], dtype=bool)
    is_graph = np.asarray([name.startswith("graph_") or name.startswith("base_x_graph_") for name in names], dtype=bool)
    is_memory = ~(is_base | is_graph)
    if variant == "full":
        return np.ones(len(names), dtype=bool)
    if variant in {"base_only", "without_deepoof_fusion"}:
        return np.asarray([name == "base_prob" for name in names], dtype=bool)
    if variant == "without_memory":
        return ~is_memory
    if variant == "without_graph_diffusion":
        return ~is_graph
    if variant == "memory_only":
        return is_memory
    if variant == "graph_only":
        return is_graph
    if variant == "base_plus_memory":
        return is_base | is_memory
    if variant == "base_plus_graph":
        return is_base | is_graph
    raise ValueError(f"unknown feature variant: {variant}")


def pair_set_from_matrix(matrix: np.ndarray) -> set[Tuple[int, int]]:
    rows, cols = np.where(matrix == 1)
    return set(zip(rows.astype(int).tolist(), cols.astype(int).tolist()))
