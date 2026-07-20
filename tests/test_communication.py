from __future__ import annotations

from pathlib import Path

import pandas as pd

from proofccc.communication import (
    aggregate_expression,
    build_receptor_conditioned_scores,
    compute_receptor_conditioned_tf_response,
    matrix_from_scores,
)


ROOT = Path(__file__).resolve().parents[1]


def test_example_communication_pipeline() -> None:
    expression = pd.read_csv(ROOT / "data/example/expression.csv").set_index("gene")
    metadata = pd.read_csv(ROOT / "data/example/metadata.csv")
    labels = metadata.set_index("cell")["cell_type"]
    pairs = pd.read_csv(ROOT / "data/example/pairs.csv")
    tf_genes = pd.read_csv(ROOT / "data/example/tf_genes.csv", header=None).iloc[:, 0]

    mean_log, detection = aggregate_expression(expression, labels)
    raw, normalized, summary = compute_receptor_conditioned_tf_response(
        expression, labels, pairs["receptor"], tf_genes
    )
    scores = build_receptor_conditioned_scores(
        pairs, mean_log, detection, raw, normalized, require_tf_support=True
    )
    matrix = matrix_from_scores(scores)

    assert raw.loc["REC1", "Receiver"] > 0
    assert not summary.empty
    assert not scores.empty
    assert scores["communication_score"].gt(0).all()
    assert matrix.loc["Sender", "Receiver"] > 0
