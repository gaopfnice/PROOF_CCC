#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CORE = HERE / "core"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(HERE))

from run_cv_multibranch_simmemory import load_or_extract_local  # noqa: E402
from run_lri_experiment import ESMCEmbedder, get_or_create_embeddings, load_datasets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/lri_esmc_600m"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/local_cache_esmc"))
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--model-name", default="esmc_600m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-len", type=int, default=1022)
    parser.add_argument("--max-residues", type=int, default=512)
    parser.add_argument("--overwrite-embeddings", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets(args.data_dir, args.datasets)

    embedder = ESMCEmbedder(args.model_name, args.device, args.chunk_len)
    from esm.sdk.api import ESMProtein, LogitsConfig

    for dataset in datasets.values():
        print(
            f"PREPARE {dataset.name}: {len(dataset.ligand_ids)} ligands x "
            f"{len(dataset.receptor_ids)} receptors positives={int(dataset.matrix.sum())}",
            flush=True,
        )
        get_or_create_embeddings(
            dataset,
            "ligand",
            dataset.ligand_ids,
            dataset.ligand_sequences,
            embedder,
            args.out_dir,
            args.model_name,
            args.overwrite_embeddings,
        )
        get_or_create_embeddings(
            dataset,
            "receptor",
            dataset.receptor_ids,
            dataset.receptor_sequences,
            embedder,
            args.out_dir,
            args.model_name,
            args.overwrite_embeddings,
        )
        load_or_extract_local(
            dataset.name,
            "ligand",
            dataset.ligand_ids,
            dataset.ligand_sequences,
            args.cache_dir,
            embedder.model,
            ESMProtein,
            LogitsConfig,
            args.device,
            args.chunk_len,
            args.max_residues,
        )
        load_or_extract_local(
            dataset.name,
            "receptor",
            dataset.receptor_ids,
            dataset.receptor_sequences,
            args.cache_dir,
            embedder.model,
            ESMProtein,
            LogitsConfig,
            args.device,
            args.chunk_len,
            args.max_residues,
        )
        print(f"PREPARE_DONE {dataset.name}", flush=True)


if __name__ == "__main__":
    main()
