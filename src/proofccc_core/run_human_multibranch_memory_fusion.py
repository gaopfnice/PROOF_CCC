#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from run_lri_experiment import load_datasets, positive_pairs, sample_negative_pairs  # noqa: E402
from run_human_multibranch_moe import load_global_embeddings, stable_dataset_seed  # noqa: E402


def evaluate(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (probs >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "average_precision": float(average_precision_score(y_true, probs)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    if len(thresholds) == 0:
        return 0.5
    best = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best])


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def topk_indices(sim: np.ndarray, k: int) -> np.ndarray:
    k = min(k, sim.shape[1] - 1)
    work = sim.copy()
    np.fill_diagonal(work, -np.inf)
    idx = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    row = np.arange(work.shape[0])[:, None]
    order = np.argsort(-work[row, idx], axis=1)
    return idx[row, order]


def svd_prior(pos_matrix: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(pos_matrix.astype(np.float32), full_matrices=False)
    rank = min(rank, len(s))
    scale = np.sqrt(s[:rank])
    return u[:, :rank] * scale, vt[:rank, :].T * scale


def build_memory_features(
    selected_pairs: np.ndarray,
    ligand_global: np.ndarray,
    receptor_global: np.ndarray,
    train_pairs: np.ndarray,
    train_labels: np.ndarray,
    shape: Tuple[int, int],
    topks: List[int],
    svd_rank: int,
) -> Tuple[np.ndarray, List[str]]:
    ligand_norm = normalize_rows(ligand_global)
    receptor_norm = normalize_rows(receptor_global)
    ligand_sim = ligand_norm @ ligand_norm.T
    receptor_sim = receptor_norm @ receptor_norm.T
    max_k_ligand = min(max(topks), ligand_sim.shape[0] - 1)
    max_k_receptor = min(max(topks), receptor_sim.shape[0] - 1)
    ligand_top = topk_indices(ligand_sim, max_k_ligand)
    receptor_top = topk_indices(receptor_sim, max_k_receptor)

    pos_matrix = np.zeros(shape, dtype=np.float32)
    neg_matrix = np.zeros(shape, dtype=np.float32)
    for (ligand_idx, receptor_idx), label in zip(train_pairs, train_labels):
        if label == 1:
            pos_matrix[ligand_idx, receptor_idx] = 1.0
        else:
            neg_matrix[ligand_idx, receptor_idx] = 1.0

    ligand_latent, receptor_latent = svd_prior(pos_matrix, svd_rank)
    ligand_degree = pos_matrix.sum(axis=1)
    receptor_degree = pos_matrix.sum(axis=0)
    ligand_neg_degree = neg_matrix.sum(axis=1)
    receptor_neg_degree = neg_matrix.sum(axis=0)

    columns = []
    names: List[str] = []
    ligand_idx = selected_pairs[:, 0]
    receptor_idx = selected_pairs[:, 1]

    seq_cos = (ligand_norm[ligand_idx] * receptor_norm[receptor_idx]).sum(axis=1)
    columns.append(seq_cos[:, None])
    names.append("seq_cross_cosine")

    left = ligand_latent[ligand_idx]
    right = receptor_latent[receptor_idx]
    dot = (left * right).sum(axis=1)
    svd_cos = dot / ((np.linalg.norm(left, axis=1) + 1e-8) * (np.linalg.norm(right, axis=1) + 1e-8))
    for name, values in [
        ("svd_dot", dot),
        ("svd_cosine", svd_cos),
        ("ligand_pos_degree", np.log1p(ligand_degree[ligand_idx])),
        ("receptor_pos_degree", np.log1p(receptor_degree[receptor_idx])),
        ("ligand_neg_degree", np.log1p(ligand_neg_degree[ligand_idx])),
        ("receptor_neg_degree", np.log1p(receptor_neg_degree[receptor_idx])),
    ]:
        columns.append(values[:, None].astype(np.float32))
        names.append(name)

    for k in topks:
        k_ligand = min(k, ligand_top.shape[1])
        k_receptor = min(k, receptor_top.shape[1])
        neigh_lig = ligand_top[ligand_idx, :k_ligand]
        neigh_rec = receptor_top[receptor_idx, :k_receptor]
        lig_w = np.clip(ligand_sim[ligand_idx[:, None], neigh_lig], 0.0, None)
        rec_w = np.clip(receptor_sim[receptor_idx[:, None], neigh_rec], 0.0, None)
        lig_pos = pos_matrix[neigh_lig, receptor_idx[:, None]]
        lig_neg = neg_matrix[neigh_lig, receptor_idx[:, None]]
        rec_pos = pos_matrix[ligand_idx[:, None], neigh_rec]
        rec_neg = neg_matrix[ligand_idx[:, None], neigh_rec]
        weighted_lig_pos = (lig_w * lig_pos).sum(axis=1) / (lig_w.sum(axis=1) + 1e-8)
        weighted_lig_neg = (lig_w * lig_neg).sum(axis=1) / (lig_w.sum(axis=1) + 1e-8)
        weighted_rec_pos = (rec_w * rec_pos).sum(axis=1) / (rec_w.sum(axis=1) + 1e-8)
        weighted_rec_neg = (rec_w * rec_neg).sum(axis=1) / (rec_w.sum(axis=1) + 1e-8)
        feature_items = [
            (f"ligand_neighbor_pos_k{k}", weighted_lig_pos),
            (f"ligand_neighbor_neg_k{k}", weighted_lig_neg),
            (f"receptor_neighbor_pos_k{k}", weighted_rec_pos),
            (f"receptor_neighbor_neg_k{k}", weighted_rec_neg),
            (f"ligand_neighbor_pos_cov_k{k}", lig_pos.mean(axis=1)),
            (f"receptor_neighbor_pos_cov_k{k}", rec_pos.mean(axis=1)),
            (f"ligand_neighbor_neg_cov_k{k}", lig_neg.mean(axis=1)),
            (f"receptor_neighbor_neg_cov_k{k}", rec_neg.mean(axis=1)),
        ]
        for name, values in feature_items:
            columns.append(values[:, None].astype(np.float32))
            names.append(name)

    return np.concatenate(columns, axis=1).astype(np.float32), names


def load_base_predictions(base_dirs: List[Path], val_pairs: np.ndarray, test_pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    val_columns = []
    test_columns = []
    names = []
    for base_dir in base_dirs:
        pred_path = base_dir / "test_predictions.npz"
        if not pred_path.exists():
            raise FileNotFoundError(f"missing prediction file: {pred_path}")
        data = np.load(pred_path)
        if not np.array_equal(data["val_pairs"], val_pairs):
            raise ValueError(f"validation pair order mismatch for {pred_path}")
        if not np.array_equal(data["pairs"], test_pairs):
            raise ValueError(f"test pair order mismatch for {pred_path}")
        val_probs = np.asarray(data["val_probs"], dtype=np.float32)
        test_probs = np.asarray(data["probs"], dtype=np.float32)
        stem = base_dir.name
        val_columns.extend([val_probs[:, None], np.log(np.clip(val_probs, 1e-6, 1 - 1e-6) / np.clip(1 - val_probs, 1e-6, 1.0))[:, None]])
        test_columns.extend([test_probs[:, None], np.log(np.clip(test_probs, 1e-6, 1 - 1e-6) / np.clip(1 - test_probs, 1e-6, 1.0))[:, None]])
        names.extend([f"{stem}_prob", f"{stem}_logit"])
    if len(base_dirs) > 1:
        val_stack = np.stack([np.load(base_dir / "test_predictions.npz")["val_probs"] for base_dir in base_dirs], axis=1)
        test_stack = np.stack([np.load(base_dir / "test_predictions.npz")["probs"] for base_dir in base_dirs], axis=1)
        for stat_name, func in [("mean", np.mean), ("max", np.max), ("min", np.min), ("std", np.std)]:
            val_columns.append(func(val_stack, axis=1)[:, None].astype(np.float32))
            test_columns.append(func(test_stack, axis=1)[:, None].astype(np.float32))
            names.append(f"base_{stat_name}")
    return np.concatenate(val_columns, axis=1), np.concatenate(test_columns, axis=1), names


def oof_predict(model, x: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, float]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probs = np.zeros(len(y), dtype=np.float32)
    for train_idx, hold_idx in folds.split(x, y):
        fold_model = clone(model)
        fold_model.fit(x[train_idx], y[train_idx])
        probs[hold_idx] = fold_model.predict_proba(x[hold_idx])[:, 1]
    return probs, float(average_precision_score(y, probs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("outputs/lri_esmc_600m/embeddings"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/human_multibranch_memory_fusion"))
    parser.add_argument(
        "--base-dirs",
        nargs="+",
        type=Path,
        default=[Path("outputs/human_multibranch_tune_expert6_softloss")],
    )
    parser.add_argument("--model-name", default="esmc_600m")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--topks", type=int, nargs="+", default=[5, 10, 25, 50])
    parser.add_argument("--svd-rank", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--fast-only", action="store_true")
    parser.add_argument(
        "--select-by",
        choices=["cv_aupr", "full_val_aupr", "val_f1", "val_accuracy"],
        default="cv_aupr",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("loading data and global embeddings", flush=True)
    rng_seed = stable_dataset_seed("human", args.seed)
    dataset = load_datasets(args.data_dir, ["human"])["human"]
    ligand_global = load_global_embeddings(args.embedding_dir, "ligand", dataset.ligand_ids, args.model_name)
    receptor_global = load_global_embeddings(args.embedding_dir, "receptor", dataset.receptor_ids, args.model_name)

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
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    val_pairs = pairs[val_idx]
    test_pairs = pairs[test_idx]

    print("loading MultiBranchRank base predictions", flush=True)
    base_val, base_test, base_names = load_base_predictions(args.base_dirs, val_pairs, test_pairs)
    print("building train-set similarity memory features", flush=True)
    all_eval_pairs = np.concatenate([val_pairs, test_pairs], axis=0)
    memory_all, memory_names = build_memory_features(
        all_eval_pairs,
        ligand_global,
        receptor_global,
        pairs[train_idx],
        labels[train_idx],
        dataset.matrix.shape,
        args.topks,
        args.svd_rank,
    )
    memory_val = memory_all[: len(val_pairs)]
    memory_test = memory_all[len(val_pairs) :]
    x_val = np.concatenate([base_val, memory_val], axis=1)
    x_test = np.concatenate([base_test, memory_test], axis=1)
    feature_names = base_names + memory_names
    print(f"meta features val={x_val.shape} test={x_test.shape}", flush=True)

    candidates = []
    for c_value in [0.03, 0.06, 0.1, 0.2, 0.5, 1.0, 2.0]:
        candidates.append(
            (
                f"logreg_C{c_value}",
                make_pipeline(StandardScaler(), LogisticRegression(C=c_value, max_iter=3000, class_weight="balanced")),
            )
        )
    if not args.fast_only:
        for max_iter, lr, leaf, l2 in [(90, 0.04, 15, 0.1), (120, 0.03, 15, 0.2), (120, 0.04, 31, 0.1)]:
            candidates.append(
                (
                    f"hgb_i{max_iter}_lr{lr}_leaf{leaf}_l2{l2}",
                    make_pipeline(
                        QuantileTransformer(
                            n_quantiles=min(512, len(y_val)),
                            output_distribution="normal",
                            random_state=args.seed,
                        ),
                        HistGradientBoostingClassifier(
                            max_iter=max_iter,
                            learning_rate=lr,
                            max_leaf_nodes=leaf,
                            l2_regularization=l2,
                            random_state=args.seed,
                        ),
                    ),
                )
            )
        for leaf in [2, 4, 8, 12]:
            candidates.append(
                (
                    f"extratrees_leaf{leaf}",
                    ExtraTreesClassifier(
                        n_estimators=600,
                        min_samples_leaf=leaf,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=args.seed,
                        n_jobs=args.n_jobs,
                    ),
                )
            )

    results = []
    base_mean_val = base_val[:, 0] if base_val.shape[1] == 2 else base_val[:, -4]
    base_mean_test = base_test[:, 0] if base_test.shape[1] == 2 else base_test[:, -4]
    threshold = best_f1_threshold(y_val, base_mean_val)
    base_val_metrics = evaluate(y_val, base_mean_val, threshold)
    base_metrics = evaluate(y_test, base_mean_test, threshold)
    results.append(
        {
            "name": "base_multibranch",
            "cv_val_average_precision": float(average_precision_score(y_val, base_mean_val)),
            "val_average_precision_full": float(average_precision_score(y_val, base_mean_val)),
            "val": base_val_metrics,
            "test": base_metrics,
        }
    )
    print(
        f"base_multibranch valAUPR={results[-1]['val_average_precision_full']:.6f} "
        f"testAUPR={base_metrics['average_precision']:.6f} testAUC={base_metrics['roc_auc']:.6f} "
        f"F1={base_metrics['f1']:.6f} PRE={base_metrics['precision']:.6f} REC={base_metrics['recall']:.6f}",
        flush=True,
    )

    def selection_score(record: Dict[str, object]) -> float:
        if args.select_by == "cv_aupr":
            return float(record["cv_val_average_precision"])
        if args.select_by == "full_val_aupr":
            return float(record["val_average_precision_full"])
        if args.select_by == "val_f1":
            # For fitted meta-model candidates, this is computed from out-of-fold
            # validation predictions, avoiding in-sample threshold selection.
            return float(record["val"]["f1"])
        if args.select_by == "val_accuracy":
            return float(record["val"]["accuracy"])
        raise ValueError(args.select_by)

    best_record = results[-1]
    best_probs = base_mean_test
    best_val_probs = base_mean_val
    for name, model in candidates:
        print(f"fit candidate {name}", flush=True)
        oof_probs, cv_ap = oof_predict(model, x_val, y_val, args.seed)
        cv_threshold = best_f1_threshold(y_val, oof_probs)
        cv_val_metrics = evaluate(y_val, oof_probs, cv_threshold)
        fitted = clone(model)
        fitted.fit(x_val, y_val)
        val_probs = fitted.predict_proba(x_val)[:, 1]
        test_probs = fitted.predict_proba(x_test)[:, 1]
        fit_threshold = best_f1_threshold(y_val, val_probs)
        fit_val_metrics = evaluate(y_val, val_probs, fit_threshold)
        test_metrics = evaluate(y_test, test_probs, cv_threshold)
        record = {
            "name": name,
            "cv_val_average_precision": cv_ap,
            "val_average_precision_full": float(average_precision_score(y_val, val_probs)),
            "val": cv_val_metrics,
            "fit_val": fit_val_metrics,
            "test": test_metrics,
        }
        results.append(record)
        print(
            f"{name} cvValAUPR={cv_ap:.6f} fullValAUPR={record['val_average_precision_full']:.6f} "
            f"cvValF1={cv_val_metrics['f1']:.6f} cvValAcc={cv_val_metrics['accuracy']:.6f} "
            f"testAUPR={test_metrics['average_precision']:.6f} testAUC={test_metrics['roc_auc']:.6f} "
            f"F1={test_metrics['f1']:.6f} PRE={test_metrics['precision']:.6f} REC={test_metrics['recall']:.6f}",
            flush=True,
        )
        if selection_score(record) > selection_score(best_record):
            best_record = record
            best_probs = test_probs
            best_val_probs = val_probs

    result = {
        "dataset": "human",
        "model": "ESMC_MultiBranchRank_SimMemoryFusion",
        "base_dirs": [str(path) for path in args.base_dirs],
        "config": {
            "data_dir": str(args.data_dir),
            "embedding_dir": str(args.embedding_dir),
            "out_dir": str(args.out_dir),
            "seed": args.seed,
            "topks": args.topks,
            "svd_rank": args.svd_rank,
            "select_by": args.select_by,
        },
        "feature_names": feature_names,
        "selected_by": args.select_by,
        "selected_model": best_record["name"],
        "val": {
            "cv_average_precision": best_record["cv_val_average_precision"],
            "full_average_precision": best_record["val_average_precision_full"],
            "metrics": best_record["val"],
        },
        "test": best_record["test"],
        "all_results": results,
    }
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    np.savez_compressed(
        args.out_dir / "test_predictions.npz",
        y_test=y_test,
        probs=best_probs,
        pairs=test_pairs,
        y_val=y_val,
        val_probs=best_val_probs,
        val_pairs=val_pairs,
    )
    print(
        f"BEST_BY_{args.select_by}",
        best_record["name"],
        json.dumps(best_record["test"], ensure_ascii=False),
        flush=True,
    )
    print(f"Done. Results written to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
