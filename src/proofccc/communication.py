"""Receptor-conditioned cell-cell communication scoring used by PROOF-CCC.

The implementation mirrors the production GSE103322 analysis while exposing
dataset-independent functions. Expression must be supplied as a genes-by-cells
matrix of non-negative counts. The reported TF term is an expression-based
association and must not be interpreted as a causal receptor-to-TF effect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CommunicationConfig:
    """Parameters fixed in the production communication analysis."""

    alpha_model_probability: float = 1.0
    lambda_specificity: float = 0.5
    rho_tf_response: float = 0.5
    tail_fraction: float = 0.25
    min_cells_per_tail: int = 5
    top_tfs_per_receptor: int = 50
    epsilon: float = 1e-8


def _validate_expression(expression: pd.DataFrame) -> pd.DataFrame:
    if expression.empty:
        raise ValueError("expression matrix is empty")
    if expression.index.has_duplicates:
        expression = expression.groupby(level=0, sort=False).sum()
    expression = expression.apply(pd.to_numeric, errors="raise").astype(np.float32)
    if not np.isfinite(expression.to_numpy()).all():
        raise ValueError("expression matrix contains non-finite values")
    if (expression.to_numpy() < 0).any():
        raise ValueError("expression must contain non-negative counts")
    return expression


def aggregate_expression(
    expression: pd.DataFrame,
    cell_types: pd.Series,
    cell_type_order: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cell-type mean log1p expression and detection frequency.

    Parameters
    ----------
    expression
        Genes-by-cells raw count matrix.
    cell_types
        Cell-type labels indexed by the same cell identifiers as the columns of
        ``expression``.
    cell_type_order
        Optional deterministic output order.
    """

    expression = _validate_expression(expression)
    missing = expression.columns.difference(cell_types.index)
    if len(missing):
        raise ValueError(f"metadata is missing {len(missing)} expression cells")
    labels = cell_types.loc[expression.columns].astype(str)
    order = list(cell_type_order) if cell_type_order is not None else list(dict.fromkeys(labels))
    values = expression.to_numpy(dtype=np.float32, copy=False)
    mean_log: dict[str, np.ndarray] = {}
    detection: dict[str, np.ndarray] = {}
    for cell_type in order:
        mask = labels.to_numpy() == cell_type
        if not mask.any():
            continue
        subset = values[:, mask]
        mean_log[cell_type] = np.log1p(subset).mean(axis=1)
        detection[cell_type] = (subset > 0).mean(axis=1)
    return (
        pd.DataFrame(mean_log, index=expression.index),
        pd.DataFrame(detection, index=expression.index),
    )


def minmax_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the production global min-max normalization to a matrix."""

    values = frame.to_numpy(dtype=float)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=float)
    if finite.any():
        vmin = float(values[finite].min())
        vmax = float(values[finite].max())
        if not math.isclose(vmin, vmax):
            out[finite] = (values[finite] - vmin) / (vmax - vmin)
        elif vmax > 0:
            out[finite] = 1.0
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def compute_receptor_conditioned_tf_response(
    expression: pd.DataFrame,
    cell_types: pd.Series,
    receptors: Iterable[str],
    tf_genes: Iterable[str],
    cell_type_order: Sequence[str] | None = None,
    config: CommunicationConfig = CommunicationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate the receptor-conditioned TF-expression support term.

    Within each receiver cell type, cells are split into receptor-high and
    receptor-low tails. For TF ``t``, the association is the high-minus-low
    difference in log1p expression divided by the TF standard deviation. Only
    positive associations are retained; the top associations weight the mean TF
    expression in that receiver population.
    """

    expression = _validate_expression(expression)
    missing = expression.columns.difference(cell_types.index)
    if len(missing):
        raise ValueError(f"metadata is missing {len(missing)} expression cells")
    labels = cell_types.loc[expression.columns].astype(str)
    order = list(cell_type_order) if cell_type_order is not None else list(dict.fromkeys(labels))
    tf_present = sorted(set(map(str, tf_genes)) & set(expression.index))
    receptor_present = sorted(set(map(str, receptors)) & set(expression.index))
    if not tf_present:
        raise ValueError("none of the supplied TF genes occur in the expression matrix")
    raw_response = pd.DataFrame(0.0, index=receptor_present, columns=order)
    records: list[dict[str, object]] = []

    for cell_type in order:
        cells = expression.columns[labels.to_numpy() == cell_type].tolist()
        if not cells:
            continue
        n_cells = len(cells)
        n_tail = max(config.min_cells_per_tail, int(math.ceil(n_cells * config.tail_fraction)))
        n_tail = min(n_tail, max(1, n_cells // 2))
        tf_matrix = np.log1p(expression.loc[tf_present, cells].to_numpy(dtype=np.float32, copy=False))
        tf_mean = tf_matrix.mean(axis=1)
        tf_std = tf_matrix.std(axis=1) + config.epsilon

        for receptor in receptor_present:
            receptor_values = np.log1p(
                expression.loc[receptor, cells].to_numpy(dtype=np.float32, copy=False)
            )
            if float(receptor_values.max() - receptor_values.min()) <= config.epsilon:
                records.append(
                    {
                        "receiver": cell_type,
                        "receptor": receptor,
                        "n_cells": n_cells,
                        "n_high_cells": 0,
                        "n_low_cells": 0,
                        "positive_tf_count": 0,
                        "used_tf_count": 0,
                        "rctf_raw": 0.0,
                        "top_tfs": "",
                    }
                )
                continue
            rank = np.argsort(receptor_values)
            low_idx, high_idx = rank[:n_tail], rank[-n_tail:]
            association = (
                tf_matrix[:, high_idx].mean(axis=1) - tf_matrix[:, low_idx].mean(axis=1)
            ) / tf_std
            positive_idx = np.flatnonzero(association > 0)
            used_idx = np.asarray([], dtype=int)
            rctf_raw = 0.0
            if len(positive_idx):
                top_k = min(config.top_tfs_per_receptor, len(positive_idx))
                used_idx = positive_idx[np.argsort(-association[positive_idx])[:top_k]]
                weights = association[used_idx]
                rctf_raw = float(
                    np.dot(weights, tf_mean[used_idx]) / (weights.sum() + config.epsilon)
                )
            raw_response.at[receptor, cell_type] = rctf_raw
            records.append(
                {
                    "receiver": cell_type,
                    "receptor": receptor,
                    "n_cells": n_cells,
                    "n_high_cells": int(n_tail),
                    "n_low_cells": int(n_tail),
                    "positive_tf_count": int(len(positive_idx)),
                    "used_tf_count": int(len(used_idx)),
                    "rctf_raw": rctf_raw,
                    "top_tfs": ";".join(tf_present[i] for i in used_idx[:10]),
                }
            )
    return raw_response, minmax_frame(raw_response), pd.DataFrame(records)


def build_receptor_conditioned_scores(
    pairs: pd.DataFrame,
    mean_log: pd.DataFrame,
    detection: pd.DataFrame,
    rctf_raw: pd.DataFrame,
    rctf_norm: pd.DataFrame,
    require_tf_support: bool = True,
    config: CommunicationConfig = CommunicationConfig(),
) -> pd.DataFrame:
    """Compute directional LR-cell-context scores.

    Required pair columns are ``ligand``, ``receptor`` and
    ``model_probability``; ``source`` is optional. The score is

    ``P(L,R)^alpha * harmonic(availability_L, availability_R) *
    (1 + rho * RCTF_norm(R,r))``.
    """

    required = {"ligand", "receptor", "model_probability"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs table lacks required columns: {sorted(missing)}")
    pairs = pairs.copy()
    if "source" not in pairs:
        pairs["source"] = "unspecified"
    if not mean_log.index.equals(detection.index) or list(mean_log.columns) != list(detection.columns):
        raise ValueError("mean_log and detection must use identical genes and cell types")

    ligand_total = mean_log.sum(axis=1) + config.epsilon
    receptor_total = mean_log.sum(axis=1) + config.epsilon
    rows: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        ligand, receptor = str(pair.ligand), str(pair.receptor)
        if ligand not in mean_log.index or receptor not in mean_log.index:
            continue
        probability = float(pair.model_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"model_probability outside [0,1] for {ligand}-{receptor}")
        for sender in mean_log.columns:
            ligand_expr = float(mean_log.at[ligand, sender])
            ligand_det = float(detection.at[ligand, sender])
            ligand_spec = float(mean_log.at[ligand, sender] / ligand_total.at[ligand])
            ligand_availability = math.sqrt(max(ligand_expr, 0.0) * max(ligand_det, 0.0)) * (
                max(ligand_spec, 0.0) ** config.lambda_specificity
            )
            for receiver in mean_log.columns:
                receptor_expr = float(mean_log.at[receptor, receiver])
                receptor_det = float(detection.at[receptor, receiver])
                receptor_spec = float(mean_log.at[receptor, receiver] / receptor_total.at[receptor])
                receptor_availability = math.sqrt(
                    max(receptor_expr, 0.0) * max(receptor_det, 0.0)
                ) * (max(receptor_spec, 0.0) ** config.lambda_specificity)
                matching = (
                    2.0
                    * ligand_availability
                    * receptor_availability
                    / (ligand_availability + receptor_availability + config.epsilon)
                )
                tf_raw = float(rctf_raw.at[receptor, receiver]) if receptor in rctf_raw.index else 0.0
                tf_norm = float(rctf_norm.at[receptor, receiver]) if receptor in rctf_norm.index else 0.0
                tf_supported = tf_raw > 0.0
                if require_tf_support and not tf_supported:
                    continue
                tf_factor = 1.0 + config.rho_tf_response * tf_norm
                score = probability**config.alpha_model_probability * matching * tf_factor
                if score <= 0:
                    continue
                rows.append(
                    {
                        "sender": sender,
                        "receiver": receiver,
                        "ligand": ligand,
                        "receptor": receptor,
                        "pair": f"{ligand}_{receptor}",
                        "source": str(pair.source),
                        "model_probability": probability,
                        "ligand_expr_logmean": ligand_expr,
                        "receptor_expr_logmean": receptor_expr,
                        "ligand_detection": ligand_det,
                        "receptor_detection": receptor_det,
                        "ligand_specificity": ligand_spec,
                        "receptor_specificity": receptor_spec,
                        "lr_matching": matching,
                        "tf_supported": tf_supported,
                        "rctf_raw": tf_raw,
                        "rctf_norm": tf_norm,
                        "rctf_factor": tf_factor,
                        "communication_score": score,
                    }
                )
    return pd.DataFrame(rows)


def matrix_from_scores(
    score_table: pd.DataFrame,
    value_col: str = "communication_score",
    aggregation: str = "sum",
) -> pd.DataFrame:
    """Aggregate LR-context scores into a sender-by-receiver matrix."""

    if aggregation not in {"sum", "mean"}:
        raise ValueError("aggregation must be 'sum' or 'mean'")
    if score_table.empty:
        return pd.DataFrame()
    grouped = score_table.groupby(["sender", "receiver"])[value_col]
    values = grouped.sum() if aggregation == "sum" else grouped.mean()
    return values.unstack(fill_value=0.0)
