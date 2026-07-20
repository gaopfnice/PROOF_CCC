from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
CORE = EXPERIMENT_ROOT / "core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from common import ensure_dir, summarize, write_csv, write_json  # noqa: E402
from run_cv_fixed_deep_ftensemble import (  # noqa: E402
    aggregate_outputs,
    build_fold_feature_bank,
    make_config,
    train_fixed_ensemble,
    write_worker_summary,
)
from run_cv_multibranch_simmemory import load_global_embeddings_generic, load_or_extract_local  # noqa: E402
from run_human_multibranch_moe import stable_dataset_seed  # noqa: E402
from run_lri_experiment import load_datasets, positive_pairs, sample_negative_pairs  # noqa: E402


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}


def base_args(cli) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=cli.data_dir,
        embedding_dir=cli.embedding_dir,
        cache_dir=cli.cache_dir,
        out_dir=cli.out_dir,
        datasets=cli.datasets,
        model_name=cli.model_name,
        device=cli.device,
        seed=cli.seed,
        repeats=cli.repeats,
        folds=cli.folds,
        start_repeat=0,
        max_repeat=None,
        only_fold=None,
        neg_ratio=cli.neg_ratio,
        inner_oof_folds=cli.inner_oof_folds,
        inner_val_size=0.18,
        max_residues=512,
        chunk_len=1022,
        epochs=cli.epochs,
        patience=cli.patience,
        batch_size=16,
        protein_dim=256,
        hidden_dim=384,
        pair_attention_dim=128,
        num_experts=6,
        dropout=0.25,
        lr=2e-4,
        weight_decay=1e-4,
        focal_alpha=0.5,
        focal_gamma=1.5,
        aux_loss_weight=0.15,
        rank_loss_weight=0.02,
        positive_target=1.0,
        negative_target=0.0,
        topks=[5, 10, 25, 50],
        svd_rank=64,
        memory_feature_mode="positive_only",
        graph_topks=[5, 10, 25, 50, 100],
        graph_self_weight=0.35,
        threshold_metric="balance4",
        threshold_beta=1.0,
        loss_type="focal",
        bce_mix_weight=0.50,
        asl_gamma_pos=0.0,
        asl_gamma_neg=2.0,
        asl_clip=0.05,
        early_stop_metric="average_precision",
        fusion_val_size=0.12,
        ft_epochs=cli.ft_epochs,
        ft_patience=cli.ft_patience,
        ft_batch_size=256,
        save_fold_models=True,
    )


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def hard_negative_pairs(
    matrix: np.ndarray,
    n: int,
    ligand_global: np.ndarray,
    receptor_global: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positives = (matrix == 1).astype(np.float32)
    lig_sim = normalize_rows(ligand_global) @ normalize_rows(ligand_global).T
    rec_sim = normalize_rows(receptor_global) @ normalize_rows(receptor_global).T
    scores = lig_sim @ positives @ rec_sim.T
    scores = scores.astype(np.float32)
    scores[matrix == 1] = -np.inf
    scores += rng.random(scores.shape, dtype=np.float32) * 1e-6
    flat = scores.ravel()
    n = min(int(n), int(np.isfinite(flat).sum()))
    if n <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    idx = np.argpartition(flat, -n)[-n:]
    idx = idx[np.argsort(-flat[idx])]
    lig, rec = np.unravel_index(idx, matrix.shape)
    return np.stack([lig, rec], axis=1).astype(np.int64)


def aac_global_and_local(ids: Sequence[str], sequences: Dict[str, str], max_residues: int) -> Tuple[np.ndarray, List[torch.Tensor]]:
    globals_ = []
    locals_ = []
    for item_id in ids:
        seq = sequences[item_id]
        residues = [aa for aa in seq[:max_residues] if aa in AA_INDEX]
        if not residues:
            residues = ["A"]
        mat = np.zeros((len(residues), len(AA)), dtype=np.float32)
        for i, aa in enumerate(residues):
            mat[i, AA_INDEX[aa]] = 1.0
        globals_.append(mat.mean(axis=0))
        locals_.append(torch.from_numpy(mat))
    return np.stack(globals_, axis=0).astype(np.float32), locals_


def apply_representation_mode(dataset_name: str, dataset, args, mode: str):
    if mode == "no_esmc_aac":
        ligand_global, ligand_local = aac_global_and_local(dataset.ligand_ids, dataset.ligand_sequences, args.max_residues)
        receptor_global, receptor_local = aac_global_and_local(dataset.receptor_ids, dataset.receptor_sequences, args.max_residues)
        return ligand_global, receptor_global, ligand_local, receptor_local

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
    if mode == "full":
        return ligand_global, receptor_global, ligand_local, receptor_local
    if mode == "global_only":
        ligand_local = [torch.zeros_like(x) for x in ligand_local]
        receptor_local = [torch.zeros_like(x) for x in receptor_local]
        return ligand_global, receptor_global, ligand_local, receptor_local
    if mode == "local_only":
        ligand_global = np.zeros_like(ligand_global)
        receptor_global = np.zeros_like(receptor_global)
        return ligand_global, receptor_global, ligand_local, receptor_local
    raise ValueError(f"unknown representation mode: {mode}")


def make_pairs(dataset, neg_ratio: float, negative_mode: str, seed: int, ligand_global: np.ndarray, receptor_global: np.ndarray):
    pos = positive_pairs(dataset.matrix)
    n_neg = int(len(pos) * neg_ratio)
    if negative_mode == "random":
        neg = sample_negative_pairs(dataset.matrix, n_neg, seed)
    elif negative_mode == "hard":
        neg = hard_negative_pairs(dataset.matrix, n_neg, ligand_global, receptor_global, seed)
    else:
        raise ValueError(f"unknown negative mode: {negative_mode}")
    pairs = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)])
    return pairs, labels


def cold_start_splits(
    pairs: np.ndarray,
    labels: np.ndarray,
    mode: str,
    folds: int,
    seed: int,
) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    if mode == "pair":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(pairs, labels)):
            yield fold, train_idx, test_idx
        return

    if mode == "leave_ligand_out":
        groups = np.unique(pairs[:, 0])
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (_, test_group_idx) in enumerate(splitter.split(groups)):
            test_ligs = set(groups[test_group_idx].tolist())
            test_mask = np.asarray([int(pair[0]) in test_ligs for pair in pairs], dtype=bool)
            yield fold, np.where(~test_mask)[0], np.where(test_mask)[0]
        return

    if mode == "leave_receptor_out":
        groups = np.unique(pairs[:, 1])
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (_, test_group_idx) in enumerate(splitter.split(groups)):
            test_recs = set(groups[test_group_idx].tolist())
            test_mask = np.asarray([int(pair[1]) in test_recs for pair in pairs], dtype=bool)
            yield fold, np.where(~test_mask)[0], np.where(test_mask)[0]
        return

    if mode == "leave_both_out":
        ligands = np.unique(pairs[:, 0])
        receptors = np.unique(pairs[:, 1])
        lig_splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        rec_splitter = KFold(n_splits=folds, shuffle=True, random_state=seed + 37)
        lig_folds = list(lig_splitter.split(ligands))
        rec_folds = list(rec_splitter.split(receptors))
        for fold in range(folds):
            test_ligs = set(ligands[lig_folds[fold][1]].tolist())
            test_recs = set(receptors[rec_folds[fold][1]].tolist())
            test_mask = np.asarray([(int(l) in test_ligs) and (int(r) in test_recs) for l, r in pairs], dtype=bool)
            train_mask = np.asarray([(int(l) not in test_ligs) and (int(r) not in test_recs) for l, r in pairs], dtype=bool)
            yield fold, np.where(train_mask)[0], np.where(test_mask)[0]
        return

    raise ValueError(f"unknown split mode: {mode}")


def valid_split(labels: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> bool:
    if len(train_idx) < 20 or len(test_idx) < 20:
        return False
    return len(np.unique(labels[train_idx])) == 2 and len(np.unique(labels[test_idx])) == 2


def run_cv_experiment(
    experiment_name: str,
    dataset_name: str,
    dataset,
    args,
    split_mode: str,
    negative_mode: str,
    representation_mode: str,
    neg_ratio: float,
    seed_offset: int = 0,
) -> List[Dict]:
    dataset_dir = ensure_dir(args.out_dir / experiment_name / dataset_name)
    config = make_config(args, dataset_name)
    ligand_global, receptor_global, ligand_local, receptor_local = apply_representation_mode(dataset_name, dataset, args, representation_mode)
    rows = []
    for repeat in range(args.repeats):
        repeat_seed = stable_dataset_seed(dataset_name, args.seed + seed_offset + repeat * 1009)
        pairs, labels = make_pairs(dataset, neg_ratio, negative_mode, repeat_seed, ligand_global, receptor_global)
        for fold, train_idx, test_idx in cold_start_splits(pairs, labels, split_mode, args.folds, repeat_seed):
            if not valid_split(labels, train_idx, test_idx):
                print(f"skip invalid split {experiment_name}/{dataset_name} repeat={repeat} fold={fold}", flush=True)
                continue
            fold_seed = repeat_seed + fold * 97
            fold_dir = ensure_dir(dataset_dir / f"repeat_{repeat:02d}" / f"fold_{fold:02d}")
            metrics_path = fold_dir / "metrics.json"
            if metrics_path.exists():
                rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                continue
            print(
                f"RUN {experiment_name} dataset={dataset_name} repeat={repeat} fold={fold} "
                f"split={split_mode} neg={negative_mode} repr={representation_mode} ratio={neg_ratio}",
                flush=True,
            )
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
                train_idx,
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
            result["supplemental_experiment"] = {
                "experiment_name": experiment_name,
                "split_mode": split_mode,
                "negative_mode": negative_mode,
                "representation_mode": representation_mode,
                "neg_ratio": neg_ratio,
                "seed_offset": seed_offset,
            }
            metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(result)
            t = result["test"]
            print(
                f"RESULT {experiment_name}/{dataset_name} repeat={repeat} fold={fold} "
                f"AUC={t['roc_auc']:.6f} AUPR={t['average_precision']:.6f} "
                f"REC={t['recall']:.6f} PRE={t['precision']:.6f} ACC={t['accuracy']:.6f} F1={t['f1']:.6f}",
                flush=True,
            )
    write_worker_summary(dataset_dir, dataset_name, 0, args.repeats, rows)
    return rows


def combine_embeddings(source_name, source, target_name, target, args):
    src_lg, src_rg, src_ll, src_rl = apply_representation_mode(source_name, source, args, "full")
    tgt_lg, tgt_rg, tgt_ll, tgt_rl = apply_representation_mode(target_name, target, args, "full")
    ligand_global = np.concatenate([src_lg, tgt_lg], axis=0)
    receptor_global = np.concatenate([src_rg, tgt_rg], axis=0)
    ligand_local = [*src_ll, *tgt_ll]
    receptor_local = [*src_rl, *tgt_rl]
    return ligand_global, receptor_global, ligand_local, receptor_local, len(src_lg), len(src_rg)


def run_transfer_experiment(experiment_name: str, source_name: str, target_name: str, datasets: Dict, args) -> Dict:
    source = datasets[source_name]
    target = datasets[target_name]
    dataset_dir = ensure_dir(args.out_dir / "transfer" / experiment_name)
    ligand_global, receptor_global, ligand_local, receptor_local, lig_offset, rec_offset = combine_embeddings(source_name, source, target_name, target, args)
    src_seed = stable_dataset_seed(source_name, args.seed)
    tgt_seed = stable_dataset_seed(target_name, args.seed + 991)
    src_pairs, src_labels = make_pairs(source, args.neg_ratio, "random", src_seed, ligand_global[:lig_offset], receptor_global[:rec_offset])
    tgt_pairs, tgt_labels = make_pairs(target, args.neg_ratio, "random", tgt_seed, ligand_global[lig_offset:], receptor_global[rec_offset:])
    tgt_pairs = tgt_pairs.copy()
    tgt_pairs[:, 0] += lig_offset
    tgt_pairs[:, 1] += rec_offset
    pairs = np.concatenate([src_pairs, tgt_pairs], axis=0)
    labels = np.concatenate([src_labels, tgt_labels], axis=0)
    train_idx = np.arange(len(src_pairs))
    test_idx = np.arange(len(src_pairs), len(pairs))
    matrix_shape = (ligand_global.shape[0], receptor_global.shape[0])
    pseudo_dataset = SimpleNamespace(matrix=np.zeros(matrix_shape, dtype=np.int8))
    config = make_config(args, source_name)
    fold_dir = ensure_dir(dataset_dir / "repeat_00" / "fold_00")
    x_oof, y_oof, x_test, y_test, test_pairs, feature_names = build_fold_feature_bank(
        fold_dir,
        f"{source_name}_to_{target_name}",
        pseudo_dataset,
        ligand_global,
        receptor_global,
        ligand_local,
        receptor_local,
        pairs,
        labels,
        train_idx,
        test_idx,
        args,
        config,
        src_seed,
    )
    result = train_fixed_ensemble(
        fold_dir,
        f"{source_name}_to_{target_name}",
        0,
        0,
        x_oof,
        y_oof,
        x_test,
        y_test,
        test_pairs,
        feature_names,
        args,
        config,
    )
    result["supplemental_experiment"] = {
        "experiment_name": experiment_name,
        "source": source_name,
        "target": target_name,
        "split_mode": "cross_dataset_transfer",
    }
    (fold_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_worker_summary(dataset_dir, f"{source_name}_to_{target_name}", 0, 1, [result])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retraining-based supplemental LRI experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("outputs/lri_esmc_600m/embeddings"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/local_cache_esmc"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/supplemental_experiments/retraining"))
    parser.add_argument("--datasets", nargs="+", default=["human", "mouse", "mouse-heart"])
    parser.add_argument("--model-name", default="esmc_600m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--ft-epochs", type=int, default=180)
    parser.add_argument("--ft-patience", type=int, default=25)
    parser.add_argument("--inner-oof-folds", type=int, default=5)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["neg_seed", "neg_ratio", "hard_negative", "cold_start", "representation", "transfer"],
    )
    args_cli = parser.parse_args()

    args = base_args(args_cli)
    ensure_dir(args.out_dir)
    datasets = load_datasets(args.data_dir, sorted(set(args.datasets + ["dataset4"])))
    all_rows = []

    if "neg_seed" in args_cli.experiments:
        for seed_offset in [0, 101, 202]:
            for dataset_name in args_cli.datasets:
                rows = run_cv_experiment(
                    f"robustness/neg_seed_{seed_offset}",
                    dataset_name,
                    datasets[dataset_name],
                    args,
                    split_mode="pair",
                    negative_mode="random",
                    representation_mode="full",
                    neg_ratio=1.0,
                    seed_offset=seed_offset,
                )
                all_rows.extend(rows)

    if "neg_ratio" in args_cli.experiments:
        for ratio in [1.0, 2.0, 5.0]:
            for dataset_name in args_cli.datasets:
                rows = run_cv_experiment(
                    f"robustness/neg_ratio_{ratio:g}",
                    dataset_name,
                    datasets[dataset_name],
                    args,
                    split_mode="pair",
                    negative_mode="random",
                    representation_mode="full",
                    neg_ratio=ratio,
                )
                all_rows.extend(rows)

    if "hard_negative" in args_cli.experiments:
        for dataset_name in args_cli.datasets:
            rows = run_cv_experiment(
                "robustness/hard_negative",
                dataset_name,
                datasets[dataset_name],
                args,
                split_mode="pair",
                negative_mode="hard",
                representation_mode="full",
                neg_ratio=1.0,
            )
            all_rows.extend(rows)

    if "cold_start" in args_cli.experiments:
        for mode in ["leave_ligand_out", "leave_receptor_out", "leave_both_out"]:
            for dataset_name in args_cli.datasets:
                rows = run_cv_experiment(
                    f"cold_start/{mode}",
                    dataset_name,
                    datasets[dataset_name],
                    args,
                    split_mode=mode,
                    negative_mode="random",
                    representation_mode="full",
                    neg_ratio=1.0,
                )
                all_rows.extend(rows)

    if "representation" in args_cli.experiments:
        for mode in ["global_only", "local_only", "no_esmc_aac"]:
            for dataset_name in args_cli.datasets:
                rows = run_cv_experiment(
                    f"ablation_representation/{mode}",
                    dataset_name,
                    datasets[dataset_name],
                    args,
                    split_mode="pair",
                    negative_mode="random",
                    representation_mode=mode,
                    neg_ratio=1.0,
                )
                all_rows.extend(rows)

    if "transfer" in args_cli.experiments:
        for source_name, target_name in [("human", "dataset4"), ("mouse", "mouse-heart")]:
            if source_name in datasets and target_name in datasets:
                all_rows.append(run_transfer_experiment(f"{source_name}_to_{target_name}", source_name, target_name, datasets, args))

    flat_rows = []
    for row in all_rows:
        t = row["test"]
        exp = row.get("supplemental_experiment", {})
        flat_rows.append(
            {
                "experiment_name": exp.get("experiment_name", ""),
                "dataset": row["dataset"],
                "repeat": row["repeat"],
                "fold": row["fold"],
                "roc_auc": t["roc_auc"],
                "average_precision": t["average_precision"],
                "recall": t["recall"],
                "precision": t["precision"],
                "accuracy": t["accuracy"],
                "f1": t["f1"],
                "threshold": row["threshold"],
            }
        )
    write_csv(args.out_dir / "all_retraining_metrics.csv", flat_rows)
    write_csv(args.out_dir / "summary_retraining_metrics.csv", summarize(flat_rows, ["experiment_name", "dataset"]))
    write_json(args.out_dir / "manifest.json", {"args": vars(args_cli), "n_results": len(flat_rows)})
    print(f"wrote retraining supplemental results to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
