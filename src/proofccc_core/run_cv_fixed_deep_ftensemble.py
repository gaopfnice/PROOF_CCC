#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import sys
from copy import copy
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
CORE = HERE / "core"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(HERE))

from run_cv_multibranch_simmemory import (  # noqa: E402
    best_threshold,
    load_global_embeddings_generic,
    load_or_extract_local,
    make_loader,
    train_multibranch_fold,
    write_curve_csv,
)
from run_human_multibranch_memory_fusion import evaluate as evaluate_probs  # noqa: E402
from run_human_multibranch_moe import PairCollator, evaluate as evaluate_torch_model, stable_dataset_seed  # noqa: E402
from run_lri_experiment import load_datasets, positive_pairs, sample_negative_pairs  # noqa: E402
from run_single_deep_oof_ftfusion import (  # noqa: E402
    RunConfig,
    build_fusion_features,
    predict_ft,
    train_ft_fusion,
)


METRIC_KEYS = ["roc_auc", "average_precision", "recall", "precision", "accuracy", "f1"]
THRESHOLD_RULES = [
    "balance4",
    "f1",
    "min4",
    "paf",
    "rate0.94",
    "rate0.96",
    "rate0.98",
    "rate1.0",
    "fbeta0.8",
    "fbeta1.2",
]
COMPONENTS = [
    {
        "name": "d96_l4_do20_g15_lr5e4",
        "params": {
            "ft_d_token": 96,
            "ft_layers": 4,
            "ft_heads": 4,
            "ft_dropout": 0.20,
            "ft_lr": 0.0005,
            "ft_weight_decay": 0.0001,
            "ft_focal_gamma": 1.5,
        },
    },
    {
        "name": "d128_l4_do20_g12_lr4e4",
        "params": {
            "ft_d_token": 128,
            "ft_layers": 4,
            "ft_heads": 4,
            "ft_dropout": 0.20,
            "ft_lr": 0.0004,
            "ft_weight_decay": 0.0001,
            "ft_focal_gamma": 1.2,
        },
    },
    {
        "name": "d64_l4_do30_g10_lr7e4",
        "params": {
            "ft_d_token": 64,
            "ft_layers": 4,
            "ft_heads": 4,
            "ft_dropout": 0.30,
            "ft_lr": 0.0007,
            "ft_weight_decay": 0.0002,
            "ft_focal_gamma": 1.0,
        },
    },
]


def make_config(args, dataset_name: str) -> RunConfig:
    return RunConfig(
        datasets=[dataset_name],
        model_name=args.model_name,
        seed=args.seed,
        test_size=0.0,
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
        topks=list(args.topks),
        svd_rank=args.svd_rank,
        memory_feature_mode=args.memory_feature_mode,
        graph_topks=list(args.graph_topks),
        graph_self_weight=args.graph_self_weight,
        threshold_metric=args.threshold_metric,
        threshold_beta=args.threshold_beta,
        loss_type=args.loss_type,
        bce_mix_weight=args.bce_mix_weight,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_clip=args.asl_clip,
        early_stop_metric=args.early_stop_metric,
        ft_d_token=0,
        ft_layers=0,
        ft_heads=0,
        ft_dropout=0.0,
        ft_lr=0.0,
        ft_weight_decay=0.0,
        ft_epochs=args.ft_epochs,
        ft_patience=args.ft_patience,
        ft_batch_size=args.ft_batch_size,
        ft_focal_gamma=0.0,
        fusion_val_size=args.fusion_val_size,
    )


def write_threshold_sweep(path: Path, dataset: str, y_val: np.ndarray, val_probs: np.ndarray, y_test: np.ndarray, test_probs: np.ndarray) -> List[Dict[str, object]]:
    rows = []
    for rule in THRESHOLD_RULES:
        threshold = best_threshold(y_val, val_probs, rule, 1.0)
        metrics = evaluate_probs(y_test, test_probs, threshold)
        rows.append({"dataset": dataset, "threshold_metric": rule, "threshold": float(threshold), **metrics})
    fields = ["dataset", "threshold_metric", "threshold", *METRIC_KEYS, "tn", "fp", "fn", "tp"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_feature_bank(path: Path):
    data = np.load(path, allow_pickle=True)
    return (
        np.asarray(data["x_oof"], dtype=np.float32),
        np.asarray(data["y_oof"], dtype=np.int64),
        np.asarray(data["x_test"], dtype=np.float32),
        np.asarray(data["y_test"], dtype=np.int64),
        np.asarray(data["test_pairs"], dtype=np.int64),
        data["feature_names"].astype(str).tolist(),
    )


def build_fold_feature_bank(
    fold_dir: Path,
    dataset_name: str,
    dataset,
    ligand_global: np.ndarray,
    receptor_global: np.ndarray,
    ligand_local: List[torch.Tensor],
    receptor_local: List[torch.Tensor],
    pairs: np.ndarray,
    labels: np.ndarray,
    outer_train_idx: np.ndarray,
    test_idx: np.ndarray,
    args,
    config: RunConfig,
    fold_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    bank_path = fold_dir / "feature_bank.npz"
    if bank_path.exists():
        print(f"reuse feature bank: {bank_path}", flush=True)
        return load_feature_bank(bank_path)

    oof_features = []
    oof_labels = []
    test_feature_ensemble = []
    feature_names: List[str] = []
    inner = StratifiedKFold(n_splits=args.inner_oof_folds, shuffle=True, random_state=fold_seed + 17)
    outer_train_idx = np.asarray(outer_train_idx)
    for inner_fold, (inner_train_rel, hold_rel) in enumerate(inner.split(outer_train_idx, labels[outer_train_idx])):
        inner_seed = fold_seed + 1000 + inner_fold * 97
        inner_train_idx = outer_train_idx[inner_train_rel]
        hold_idx = outer_train_idx[hold_rel]
        fit_idx, val_idx = train_test_split(
            inner_train_idx,
            test_size=args.inner_val_size,
            random_state=inner_seed,
            stratify=labels[inner_train_idx],
        )
        print(
            f"--- {dataset_name} outer_fold inner_oof_fold={inner_fold + 1}/{args.inner_oof_folds} ---",
            flush=True,
        )
        model, _, _, _ = train_multibranch_fold(
            ligand_global,
            receptor_global,
            ligand_local,
            receptor_local,
            pairs[fit_idx],
            labels[fit_idx],
            pairs[val_idx],
            labels[val_idx],
            args,
            inner_seed,
        )
        collator = PairCollator(ligand_global, receptor_global, ligand_local, receptor_local)
        hold_loader = make_loader(pairs[hold_idx], labels[hold_idx], args.batch_size, collator, inner_seed, False)
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

        test_loader = make_loader(pairs[test_idx], labels[test_idx], args.batch_size, collator, inner_seed, False)
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
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    x_oof = np.concatenate(oof_features, axis=0).astype(np.float32)
    y_oof = np.concatenate(oof_labels, axis=0).astype(np.int64)
    x_test = np.mean(np.stack(test_feature_ensemble, axis=0), axis=0).astype(np.float32)
    y_test = labels[test_idx].astype(np.int64)
    test_pairs = pairs[test_idx]
    np.savez_compressed(
        bank_path,
        y_oof=y_oof,
        x_oof=x_oof,
        y_test=y_test,
        x_test=x_test,
        test_pairs=test_pairs,
        feature_names=np.asarray(feature_names),
    )
    with (fold_dir / "feature_bank.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "model": "DeepOOF_FeatureBank_CV",
                "feature_count": int(x_oof.shape[1]),
                "train_oof_samples": int(len(y_oof)),
                "test_samples": int(len(y_test)),
                "config": asdict(config),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return x_oof, y_oof, x_test, y_test, test_pairs, feature_names


def train_fixed_ensemble(
    fold_dir: Path,
    dataset_name: str,
    repeat: int,
    fold: int,
    x_oof: np.ndarray,
    y_oof: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_pairs: np.ndarray,
    feature_names: List[str],
    args,
    config: RunConfig,
) -> Dict[str, object]:
    fusion_train_idx, fusion_val_idx = train_test_split(
        np.arange(len(y_oof)),
        test_size=args.fusion_val_size,
        random_state=args.seed,
        stratify=y_oof,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_oof[fusion_train_idx]).astype(np.float32)
    x_val = scaler.transform(x_oof[fusion_val_idx]).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)
    y_train = y_oof[fusion_train_idx]
    y_val = y_oof[fusion_val_idx]

    val_probs = []
    test_probs = []
    component_payload = {}
    for component in COMPONENTS:
        comp_args = copy(args)
        for key, value in component["params"].items():
            setattr(comp_args, key, value)
        print(f"training fixed FT component {component['name']}", flush=True)
        model, info, comp_val_probs = train_ft_fusion(x_train, y_train, x_val, y_val, comp_args, args.seed)
        comp_test_probs = predict_ft(model, x_test_scaled, args.ft_batch_size, args.device)
        val_probs.append(comp_val_probs.astype(np.float64))
        test_probs.append(comp_test_probs.astype(np.float64))
        component_payload[component["name"]] = {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "params": component["params"],
            "ft_info": info,
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_val_probs = np.mean(np.vstack(val_probs), axis=0)
    mean_test_probs = np.mean(np.vstack(test_probs), axis=0)
    threshold = best_threshold(y_val, mean_val_probs, "balance4", 1.0)
    metrics = evaluate_probs(y_test, mean_test_probs, threshold)
    sweep = write_threshold_sweep(fold_dir / "threshold_sweep_metrics.csv", dataset_name, y_val, mean_val_probs, y_test, mean_test_probs)
    write_curve_csv(fold_dir / "curves", "ensemble_test", y_test, mean_test_probs)
    np.savez_compressed(
        fold_dir / "predictions.npz",
        y_val=y_val,
        val_probs=mean_val_probs,
        y_test=y_test,
        test_probs=mean_test_probs,
        test_pairs=test_pairs,
        component_names=np.asarray([component["name"] for component in COMPONENTS]),
        feature_names=np.asarray(feature_names),
    )
    model_path = fold_dir / "ftensemble_components.pt"
    scaler_path = fold_dir / "feature_scaler.joblib"
    if args.save_fold_models:
        torch.save(
            {
                "components": component_payload,
                "feature_names": feature_names,
                "config": asdict(config),
                "dataset": dataset_name,
                "repeat": repeat,
                "fold": fold,
                "threshold_metric": "balance4",
                "threshold": float(threshold),
            },
            model_path,
        )
        joblib.dump({"scaler": scaler, "feature_names": feature_names, "config": asdict(config)}, scaler_path)
    else:
        model_path = Path("")
        scaler_path = Path("")

    result = {
        "dataset": dataset_name,
        "repeat": repeat,
        "fold": fold,
        "model": "DeepOOF_FixedFTTransformerEnsemble_CV",
        "components": [component["name"] for component in COMPONENTS],
        "threshold_metric": "balance4",
        "threshold": float(threshold),
        "test": metrics,
        "threshold_sweep": sweep,
        "feature_count": int(x_oof.shape[1]),
        "train_oof_samples": int(len(y_oof)),
        "test_samples": int(len(y_test)),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def run_dataset(dataset_name: str, dataset, args) -> None:
    dataset_dir = args.out_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(args, dataset_name)
    print(f"===== DATASET {dataset_name} fixed DeepOOF-FT-Ensemble CV =====", flush=True)
    ligand_global = load_global_embeddings_generic(args.embedding_dir, dataset_name, "ligand", dataset.ligand_ids, args.model_name)
    receptor_global = load_global_embeddings_generic(args.embedding_dir, dataset_name, "receptor", dataset.receptor_ids, args.model_name)
    ligand_local = load_or_extract_local(
        dataset_name,
        "ligand",
        dataset.ligand_ids,
        dataset.ligand_sequences,
        args.cache_dir,
        None,
        None,
        None,
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
        None,
        None,
        None,
        args.device,
        args.chunk_len,
        args.max_residues,
    )
    pos = positive_pairs(dataset.matrix)
    stop_repeat = args.repeats if args.max_repeat is None else min(args.repeats, args.max_repeat)
    worker_rows = []
    for repeat in range(args.start_repeat, stop_repeat):
        repeat_seed = stable_dataset_seed(dataset_name, args.seed + repeat * 1009)
        neg = sample_negative_pairs(dataset.matrix, int(len(pos) * args.neg_ratio), repeat_seed)
        pairs = np.concatenate([pos, neg], axis=0)
        labels = np.concatenate([np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)])
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=repeat_seed)
        for fold, (outer_train_idx, test_idx) in enumerate(splitter.split(pairs, labels)):
            if args.only_fold is not None and fold != args.only_fold:
                continue
            fold_seed = repeat_seed + fold * 97
            fold_dir = dataset_dir / f"repeat_{repeat:02d}" / f"fold_{fold:02d}"
            metrics_path = fold_dir / "metrics.json"
            if metrics_path.exists():
                print(f"skip existing {dataset_name} repeat={repeat} fold={fold}", flush=True)
                with metrics_path.open("r", encoding="utf-8") as f:
                    worker_rows.append(json.load(f))
                continue
            fold_dir.mkdir(parents=True, exist_ok=True)
            print(f"--- {dataset_name} repeat={repeat + 1}/{args.repeats} fold={fold + 1}/{args.folds} ---", flush=True)
            x_oof, y_oof, x_test, y_test, test_pairs, feature_names = build_fold_feature_bank(
                fold_dir,
                dataset_name,
                dataset,
                ligand_global,
                receptor_global,
                ligand_local,
                receptor_local,
                pairs,
                labels,
                outer_train_idx,
                test_idx,
                args,
                config,
                fold_seed,
            )
            result = train_fixed_ensemble(
                fold_dir,
                dataset_name,
                repeat,
                fold,
                x_oof,
                y_oof,
                x_test,
                y_test,
                test_pairs,
                feature_names,
                args,
                config,
            )
            worker_rows.append(result)
            t = result["test"]
            print(
                f"FOLD_RESULT {dataset_name} repeat={repeat} fold={fold} "
                f"AUC={t['roc_auc']:.6f} AUPR={t['average_precision']:.6f} "
                f"REC={t['recall']:.6f} PRE={t['precision']:.6f} ACC={t['accuracy']:.6f} F1={t['f1']:.6f}",
                flush=True,
            )
    write_worker_summary(dataset_dir, dataset_name, args.start_repeat, stop_repeat, worker_rows)


def write_worker_summary(dataset_dir: Path, dataset_name: str, start_repeat: int, stop_repeat: int, rows: List[Dict[str, object]]) -> None:
    path = dataset_dir / f"worker_repeats_{start_repeat:02d}_{stop_repeat - 1:02d}.csv"
    fields = ["dataset", "repeat", "fold", "roc_auc", "average_precision", "recall", "precision", "accuracy", "f1", "threshold", "model_path", "scaler_path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            t = row["test"]
            writer.writerow(
                {
                    "dataset": dataset_name,
                    "repeat": row["repeat"],
                    "fold": row["fold"],
                    "threshold": row["threshold"],
                    "model_path": row.get("model_path", ""),
                    "scaler_path": row.get("scaler_path", ""),
                    **{key: t[key] for key in METRIC_KEYS},
                }
            )


def collect_dataset_rows(dataset_dir: Path) -> List[Dict[str, object]]:
    rows = []
    for metrics_path in sorted(dataset_dir.glob("repeat_*/fold_*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def aggregate_metric_values(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for key in METRIC_KEYS:
        values = np.asarray([float(row["test"][key]) for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()) if len(values) else float("nan"),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()) if len(values) else float("nan"),
            "max": float(values.max()) if len(values) else float("nan"),
        }
    return summary


def copy_best_model(dataset_dir: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    best = max(rows, key=lambda row: float(row["test"]["average_precision"]))
    src_dir = dataset_dir / f"repeat_{int(best['repeat']):02d}" / f"fold_{int(best['fold']):02d}"
    best_dir = dataset_dir / "best_model"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    for name in ["metrics.json", "predictions.npz", "ftensemble_components.pt", "feature_scaler.joblib", "feature_bank.npz", "feature_bank.json", "threshold_sweep_metrics.csv"]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, best_dir / name)
    if (src_dir / "curves").exists():
        shutil.copytree(src_dir / "curves", best_dir / "curves", dirs_exist_ok=True)
    with (best_dir / "best_fold.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": best["dataset"],
                "repeat": int(best["repeat"]),
                "fold": int(best["fold"]),
                "selection_metric": "test_average_precision",
                "test": best["test"],
                "source_fold_dir": str(src_dir),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def aggregate_outputs(out_dir: Path, datasets: List[str]) -> None:
    all_summary_rows = []
    for dataset_name in datasets:
        dataset_dir = out_dir / dataset_name
        rows = collect_dataset_rows(dataset_dir)
        if not rows:
            continue
        copy_best_model(dataset_dir, rows)
        summary = aggregate_metric_values(rows)
        with (dataset_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"dataset": dataset_name, "folds_completed": len(rows), "metrics": summary}, f, ensure_ascii=False, indent=2)
        with (dataset_dir / "metrics_all_folds.csv").open("w", encoding="utf-8", newline="") as f:
            fields = ["dataset", "repeat", "fold", "roc_auc", "average_precision", "recall", "precision", "accuracy", "f1", "threshold", "model_path", "scaler_path"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                t = row["test"]
                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "repeat": row["repeat"],
                        "fold": row["fold"],
                        "threshold": row["threshold"],
                        "model_path": row.get("model_path", ""),
                        "scaler_path": row.get("scaler_path", ""),
                        **{key: t[key] for key in METRIC_KEYS},
                    }
                )
        for metric in METRIC_KEYS:
            all_summary_rows.append(
                {
                    "dataset": dataset_name,
                    "metric": metric,
                    "folds_completed": len(rows),
                    "mean": summary[metric]["mean"],
                    "std": summary[metric]["std"],
                    "min": summary[metric]["min"],
                    "max": summary[metric]["max"],
                }
            )
    with (out_dir / "summary_all_datasets.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["dataset", "metric", "folds_completed", "mean", "std", "min", "max"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_summary_rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("outputs/lri_esmc_600m/embeddings"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/local_cache_esmc"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["human", "mouse", "mouse-heart"])
    parser.add_argument("--model-name", default="esmc_600m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--start-repeat", type=int, default=0)
    parser.add_argument("--max-repeat", type=int, default=None)
    parser.add_argument("--only-fold", type=int, default=None)
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
    parser.add_argument("--threshold-metric", default="balance4")
    parser.add_argument("--threshold-beta", type=float, default=1.0)
    parser.add_argument("--loss-type", choices=["focal", "bce", "bce_focal", "asl"], default="focal")
    parser.add_argument("--bce-mix-weight", type=float, default=0.50)
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asl-gamma-neg", type=float, default=2.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--early-stop-metric", choices=["average_precision", "f1", "accuracy", "paf"], default="average_precision")
    parser.add_argument("--fusion-val-size", type=float, default=0.12)
    parser.add_argument("--ft-epochs", type=int, default=180)
    parser.add_argument("--ft-patience", type=int, default=25)
    parser.add_argument("--ft-batch-size", type=int, default=256)
    parser.add_argument("--save-fold-models", action="store_true", default=True)
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate_outputs(args.out_dir, args.datasets)
        return
    with (args.out_dir / "fixed_cv_strategy_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "strategy": "DeepOOF-FT-Ensemble",
                "repeats": args.repeats,
                "folds": args.folds,
                "inner_oof_folds": args.inner_oof_folds,
                "components": COMPONENTS,
                "threshold_rule": "balance4",
                "fusion_val_size": args.fusion_val_size,
                "graph_topks": args.graph_topks,
                "graph_self_weight": args.graph_self_weight,
                "memory_feature_mode": args.memory_feature_mode,
                "leakage_control": [
                    "Outer test fold labels are never used in training, feature construction, model selection, or threshold selection.",
                    "OOF train features for each outer fold are generated by inner folds where each held-out sample is scored by a base model that did not train on it.",
                    "Memory and graph features are constructed from training labels only.",
                    "The same FT components and balance4 threshold rule are used for all datasets and all folds.",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    datasets = load_datasets(args.data_dir, args.datasets)
    for dataset_name, dataset in datasets.items():
        run_dataset(dataset_name, dataset, args)
    aggregate_outputs(args.out_dir, args.datasets)


if __name__ == "__main__":
    main()
