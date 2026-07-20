from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import copy
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
CORE = EXPERIMENT_ROOT / "core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from common import ensure_dir, write_json  # noqa: E402
from run_cv_fixed_deep_ftensemble import COMPONENTS, make_config  # noqa: E402
from run_cv_multibranch_simmemory import (  # noqa: E402
    load_global_embeddings_generic,
    load_or_extract_local,
    make_loader,
    train_multibranch_fold,
)
from run_extended_cv_experiments import base_args, hard_negative_pairs, make_pairs  # noqa: E402
from run_human_multibranch_moe import PairCollator, stable_dataset_seed  # noqa: E402
from run_lri_experiment import load_datasets, positive_pairs, sample_negative_pairs  # noqa: E402
from run_single_deep_oof_ftfusion import build_fusion_features, predict_ft, train_ft_fusion  # noqa: E402


def predict_base_probs(model, loader, device: str) -> np.ndarray:
    model.eval()
    probs = []
    with torch.inference_mode():
        for batch in loader:
            ligand_global, receptor_global, ligand_tokens, ligand_mask, receptor_tokens, receptor_mask, _ = batch
            logits = model(
                ligand_global.to(device),
                receptor_global.to(device),
                ligand_tokens.to(device).float(),
                ligand_mask.to(device),
                receptor_tokens.to(device).float(),
                receptor_mask.to(device),
            )
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs).astype(np.float64)


def candidate_pairs(dataset, ligand_global, receptor_global, mode: str, max_candidates: int | None, seed: int) -> np.ndarray:
    zeros = np.argwhere(dataset.matrix == 0).astype(np.int64)
    if max_candidates is None or max_candidates >= len(zeros):
        if mode in {"all", "random"}:
            return zeros
    rng = np.random.default_rng(seed)
    if mode == "random":
        idx = rng.choice(len(zeros), size=min(max_candidates or len(zeros), len(zeros)), replace=False)
        return zeros[idx]
    if mode in {"hard_prefilter", "hard"}:
        return hard_negative_pairs(dataset.matrix, max_candidates or 50000, ligand_global, receptor_global, seed)
    if mode == "all":
        return zeros
    raise ValueError(f"unknown candidate mode: {mode}")


def exclude_pairs(candidates: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    excluded_set = {(int(i), int(j)) for i, j in excluded}
    keep = [idx for idx, (i, j) in enumerate(candidates) if (int(i), int(j)) not in excluded_set]
    return candidates[np.asarray(keep, dtype=np.int64)]


def build_unknown_feature_bank(
    fold_dir: Path,
    dataset_name: str,
    dataset,
    ligand_global: np.ndarray,
    receptor_global: np.ndarray,
    ligand_local: List[torch.Tensor],
    receptor_local: List[torch.Tensor],
    train_pairs: np.ndarray,
    train_labels: np.ndarray,
    candidates: np.ndarray,
    args,
    config,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    bank_path = fold_dir / "unknown_feature_bank.npz"
    if bank_path.exists():
        z = np.load(bank_path, allow_pickle=True)
        return (
            z["x_oof"].astype(np.float32),
            z["y_oof"].astype(np.int64),
            z["x_candidates"].astype(np.float32),
            z["candidate_pairs"].astype(np.int64),
            z["feature_names"].astype(str).tolist(),
        )

    oof_features = []
    oof_labels = []
    candidate_features = []
    feature_names: List[str] = []
    inner = StratifiedKFold(n_splits=args.inner_oof_folds, shuffle=True, random_state=seed + 17)
    all_train_idx = np.arange(len(train_labels))
    collator = PairCollator(ligand_global, receptor_global, ligand_local, receptor_local)
    dummy_candidate_labels = np.zeros(len(candidates), dtype=np.int64)

    for inner_fold, (inner_train_idx, hold_idx) in enumerate(inner.split(all_train_idx, train_labels)):
        inner_seed = seed + 1000 + inner_fold * 97
        fit_idx, val_idx = train_test_split(
            inner_train_idx,
            test_size=args.inner_val_size,
            random_state=inner_seed,
            stratify=train_labels[inner_train_idx],
        )
        print(f"unknown prediction inner_oof_fold={inner_fold + 1}/{args.inner_oof_folds}", flush=True)
        model, _, _, _ = train_multibranch_fold(
            ligand_global,
            receptor_global,
            ligand_local,
            receptor_local,
            train_pairs[fit_idx],
            train_labels[fit_idx],
            train_pairs[val_idx],
            train_labels[val_idx],
            args,
            inner_seed,
        )
        hold_loader = make_loader(train_pairs[hold_idx], train_labels[hold_idx], args.batch_size, collator, inner_seed, False)
        hold_base_probs = predict_base_probs(model, hold_loader, args.device)
        hold_features, feature_names = build_fusion_features(
            train_pairs[hold_idx],
            hold_base_probs,
            ligand_global,
            receptor_global,
            train_pairs[inner_train_idx],
            train_labels[inner_train_idx],
            dataset.matrix.shape,
            args,
        )
        oof_features.append(hold_features)
        oof_labels.append(train_labels[hold_idx])

        candidate_loader = make_loader(candidates, dummy_candidate_labels, args.batch_size, collator, inner_seed, False)
        candidate_base_probs = predict_base_probs(model, candidate_loader, args.device)
        fold_candidate_features, _ = build_fusion_features(
            candidates,
            candidate_base_probs,
            ligand_global,
            receptor_global,
            train_pairs[inner_train_idx],
            train_labels[inner_train_idx],
            dataset.matrix.shape,
            args,
        )
        candidate_features.append(fold_candidate_features)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    x_oof = np.concatenate(oof_features, axis=0).astype(np.float32)
    y_oof = np.concatenate(oof_labels, axis=0).astype(np.int64)
    x_candidates = np.mean(np.stack(candidate_features, axis=0), axis=0).astype(np.float32)
    np.savez_compressed(
        bank_path,
        x_oof=x_oof,
        y_oof=y_oof,
        x_candidates=x_candidates,
        candidate_pairs=candidates,
        feature_names=np.asarray(feature_names),
    )
    write_json(
        fold_dir / "unknown_feature_bank.json",
        {
            "dataset": dataset_name,
            "candidate_count": int(len(candidates)),
            "feature_count": int(x_candidates.shape[1]),
            "config": config.__dict__ if hasattr(config, "__dict__") else str(config),
        },
    )
    return x_oof, y_oof, x_candidates, candidates, feature_names


def train_ft_and_predict_candidates(fold_dir: Path, x_oof: np.ndarray, y_oof: np.ndarray, x_candidates: np.ndarray, args, config, feature_names):
    train_idx, val_idx = train_test_split(
        np.arange(len(y_oof)),
        test_size=args.fusion_val_size,
        random_state=args.seed,
        stratify=y_oof,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_oof[train_idx]).astype(np.float32)
    x_val = scaler.transform(x_oof[val_idx]).astype(np.float32)
    x_candidate_scaled = scaler.transform(x_candidates).astype(np.float32)
    y_train = y_oof[train_idx]
    y_val = y_oof[val_idx]
    candidate_probs = []
    component_payload = {}
    for component in COMPONENTS:
        comp_args = copy(args)
        for key, value in component["params"].items():
            setattr(comp_args, key, value)
        model, info, _ = train_ft_fusion(x_train, y_train, x_val, y_val, comp_args, args.seed)
        probs = predict_ft(model, x_candidate_scaled, args.ft_batch_size, args.device)
        candidate_probs.append(probs.astype(np.float64))
        component_payload[component["name"]] = {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "params": component["params"],
            "ft_info": info,
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mean_probs = np.mean(np.vstack(candidate_probs), axis=0)
    torch.save(
        {
            "components": component_payload,
            "feature_names": feature_names,
            "config": config.__dict__ if hasattr(config, "__dict__") else str(config),
        },
        fold_dir / "unknown_ftensemble_components.pt",
    )
    joblib.dump({"scaler": scaler, "feature_names": feature_names}, fold_dir / "unknown_feature_scaler.joblib")
    return mean_probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the final model and predict high-confidence unknown LRI candidates.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("outputs/lri_esmc_600m/embeddings"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/local_cache_esmc"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/supplemental_experiments/high_confidence"))
    parser.add_argument("--datasets", nargs="+", default=["human"])
    parser.add_argument("--model-name", default="esmc_600m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--candidate-mode", choices=["hard_prefilter", "random", "all"], default="hard_prefilter")
    parser.add_argument("--max-candidates", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--min-score", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--ft-epochs", type=int, default=180)
    parser.add_argument("--ft-patience", type=int, default=25)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--inner-oof-folds", type=int, default=5)
    cli = parser.parse_args()

    args = base_args(cli)
    args.out_dir = cli.out_dir
    ensure_dir(args.out_dir)
    datasets = load_datasets(args.data_dir, cli.datasets)
    for dataset_name, dataset in datasets.items():
        dataset_dir = ensure_dir(args.out_dir / dataset_name)
        config = make_config(args, dataset_name)
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
        seed = stable_dataset_seed(dataset_name, args.seed)
        train_pairs, train_labels = make_pairs(dataset, args.neg_ratio, "random", seed, ligand_global, receptor_global)
        candidates = candidate_pairs(dataset, ligand_global, receptor_global, cli.candidate_mode, cli.max_candidates, seed + 57)
        train_negative_pairs = train_pairs[train_labels == 0]
        candidates = exclude_pairs(candidates, train_negative_pairs)
        fold_dir = ensure_dir(dataset_dir / "train_all_predict_unknown")
        x_oof, y_oof, x_candidates, candidate_idx_pairs, feature_names = build_unknown_feature_bank(
            fold_dir,
            dataset_name,
            dataset,
            ligand_global,
            receptor_global,
            ligand_local,
            receptor_local,
            train_pairs,
            train_labels,
            candidates,
            args,
            config,
            seed,
        )
        probs = train_ft_and_predict_candidates(fold_dir, x_oof, y_oof, x_candidates, args, config, feature_names)
        order = np.argsort(-probs)
        selected = [idx for idx in order if probs[idx] >= cli.min_score][: cli.top_k]
        if len(selected) < cli.top_k:
            selected = order[: cli.top_k].tolist()
        csv_path = dataset_dir / f"{dataset_name}_high_confidence_unknown_top{cli.top_k}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["rank", "ligand", "receptor", "ligand_index", "receptor_index", "score"])
            writer.writeheader()
            for rank, idx in enumerate(selected, start=1):
                lig_i, rec_i = candidate_idx_pairs[idx]
                writer.writerow(
                    {
                        "rank": rank,
                        "ligand": dataset.ligand_ids[int(lig_i)],
                        "receptor": dataset.receptor_ids[int(rec_i)],
                        "ligand_index": int(lig_i),
                        "receptor_index": int(rec_i),
                        "score": float(probs[idx]),
                    }
                )
        np.savez_compressed(
            dataset_dir / "unknown_candidate_predictions.npz",
            candidate_pairs=candidate_idx_pairs,
            probs=probs,
        )
        write_json(
            dataset_dir / "manifest.json",
            {
                "dataset": dataset_name,
                "candidate_mode": cli.candidate_mode,
                "candidate_count": int(len(candidate_idx_pairs)),
                "top_k": cli.top_k,
                "min_score": cli.min_score,
                "csv": str(csv_path),
            },
        )
        print(f"wrote high-confidence candidates: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
