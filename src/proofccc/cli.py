"""Command-line entry point for communication scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .communication import (
    CommunicationConfig,
    aggregate_expression,
    build_receptor_conditioned_scores,
    compute_receptor_conditioned_tf_response,
    matrix_from_scores,
)


def _read_gene_by_cell(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    if table.shape[1] < 2:
        raise ValueError("expression CSV must contain a gene column and at least one cell")
    gene_col = table.columns[0]
    return table.set_index(gene_col)


def communication_main() -> None:
    parser = argparse.ArgumentParser(description="Compute PROOF-CCC communication scores")
    parser.add_argument("--expression", type=Path, required=True, help="genes-by-cells raw-count CSV")
    parser.add_argument("--metadata", type=Path, required=True, help="CSV with cell and cell_type columns")
    parser.add_argument("--pairs", type=Path, required=True, help="CSV with ligand/receptor/model_probability")
    parser.add_argument("--tf-list", type=Path, required=True, help="one-column CSV/TXT of TF gene symbols")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-zero-rctf", action="store_true", help="retain expression-matched contexts without positive RCTF")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--specificity-power", type=float, default=0.5)
    args = parser.parse_args()

    expression = _read_gene_by_cell(args.expression)
    expression.index = expression.index.astype(str).str.upper()
    metadata = pd.read_csv(args.metadata)
    if not {"cell", "cell_type"}.issubset(metadata.columns):
        raise ValueError("metadata must contain cell and cell_type columns")
    cell_types = metadata.set_index("cell")["cell_type"]
    pairs = pd.read_csv(args.pairs)
    pairs["ligand"] = pairs["ligand"].astype(str).str.upper()
    pairs["receptor"] = pairs["receptor"].astype(str).str.upper()
    tf_table = pd.read_csv(args.tf_list, header=None)
    tf_genes = set(tf_table.iloc[:, 0].astype(str).str.upper())
    config = CommunicationConfig(
        alpha_model_probability=args.alpha,
        rho_tf_response=args.rho,
        lambda_specificity=args.specificity_power,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mean_log, detection = aggregate_expression(expression, cell_types)
    raw, normalized, summary = compute_receptor_conditioned_tf_response(
        expression,
        cell_types,
        pairs["receptor"],
        tf_genes,
        cell_type_order=list(mean_log.columns),
        config=config,
    )
    scores = build_receptor_conditioned_scores(
        pairs,
        mean_log,
        detection,
        raw,
        normalized,
        require_tf_support=not args.allow_zero_rctf,
        config=config,
    )
    mean_log.to_csv(args.out_dir / "celltype_mean_log1p_expression.csv")
    detection.to_csv(args.out_dir / "celltype_detection_rate.csv")
    raw.to_csv(args.out_dir / "receptor_conditioned_tf_response_raw.csv")
    normalized.to_csv(args.out_dir / "receptor_conditioned_tf_response_minmax.csv")
    summary.to_csv(args.out_dir / "receptor_conditioned_tf_response_summary.csv", index=False)
    scores.to_csv(args.out_dir / "lr_celltype_communication_scores.csv", index=False)
    matrix_from_scores(scores, aggregation="sum").to_csv(args.out_dir / "total_communication_matrix.csv")
    matrix_from_scores(scores, aggregation="mean").to_csv(args.out_dir / "mean_communication_matrix.csv")


if __name__ == "__main__":
    communication_main()
