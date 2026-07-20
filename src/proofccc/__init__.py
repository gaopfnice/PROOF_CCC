"""Public, reusable interfaces for PROOF-CCC."""

from .communication import (
    CommunicationConfig,
    aggregate_expression,
    build_receptor_conditioned_scores,
    compute_receptor_conditioned_tf_response,
    matrix_from_scores,
)

__all__ = [
    "CommunicationConfig",
    "aggregate_expression",
    "build_receptor_conditioned_scores",
    "compute_receptor_conditioned_tf_response",
    "matrix_from_scores",
]

__version__ = "0.1.0"
