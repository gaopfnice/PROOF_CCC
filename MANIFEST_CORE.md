# Core-source provenance

The files below were copied from the final experiment snapshot dated 2026-05-20
and retained without algorithmic rewriting:

| File | Role |
|---|---|
| `run_lri_experiment.py` | Dataset loading, ESM-C global features, base utilities |
| `run_human_multibranch_moe.py` | Pair-aware multi-branch encoder and expert fusion |
| `run_human_multibranch_memory_fusion.py` | Positive-memory feature construction |
| `run_cv_multibranch_simmemory.py` | Fold-local model fitting, memory selection, graph diffusion |
| `run_single_deep_oof_ftfusion.py` | Nested OOF feature bank and FT-Transformer fusion |
| `run_cv_fixed_deep_ftensemble.py` | Final repeated outer-CV and three-member ensemble |
| `prepare_dataset_embeddings.py` | Frozen ESM-C feature extraction and caching |
| `run_extended_cv_experiments.py` | Candidate construction helpers used by final prediction |
| `run_high_confidence_prediction.py` | Ranking of unannotated LR candidates |
`src/proofccc/communication.py` is a path-independent extraction of the
production communication equations with validation and tests. It preserves the
same default parameters and transformations. The original case scripts bundled
data parsing and plotting with hard-coded server paths, so they are not part of
this source release; the reusable scoring implementation is provided instead.
