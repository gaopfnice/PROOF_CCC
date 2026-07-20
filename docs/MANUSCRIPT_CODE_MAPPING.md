# Manuscript-to-code mapping

This table identifies the source that implements each methodological component.
It is intended to prevent a simplified example from being mistaken for the
production LRI model.

| Manuscript component | Primary implementation |
|---|---|
| Benchmark matrix and sequence loading | `src/proofccc_core/run_lri_experiment.py` |
| ESM-C 600M global embedding extraction | `src/proofccc_core/run_lri_experiment.py` |
| Residue-level feature extraction and cache | `src/proofccc_core/run_cv_multibranch_simmemory.py` |
| Pair-aware multi-branch encoder, residue attention, experts and gate | `src/proofccc_core/run_human_multibranch_moe.py` |
| Positive-memory and graph-diffusion features | `src/proofccc_core/run_cv_multibranch_simmemory.py` |
| Nested inner OOF contextual construction | `src/proofccc_core/run_single_deep_oof_ftfusion.py` and `src/proofccc_core/run_cv_fixed_deep_ftensemble.py` |
| FT-Transformer fusion and fixed three-member ensemble | `src/proofccc_core/run_single_deep_oof_ftfusion.py` and `src/proofccc_core/run_cv_fixed_deep_ftensemble.py` |
| Repeated stratified pair-level benchmark | `src/proofccc_core/run_cv_fixed_deep_ftensemble.py` |
| Unannotated-pair prioritization | `src/proofccc_core/run_high_confidence_prediction.py` |
| Cell-type expression availability | `src/proofccc/communication.py` |
| Receptor-conditioned TF-expression support | `src/proofccc/communication.py` |
| Directional LRI and cell-type communication scores | `src/proofccc/communication.py` |

The scripts in `scripts/` are stable launchers. Files in `src/proofccc_core/`
retain the final experiment structure and cross-import names, whereas
`src/proofccc/communication.py` provides a cleaned reusable implementation of
the published communication equations.
