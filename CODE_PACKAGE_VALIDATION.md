# Code-package validation

Validation date: 2026-07-20

## Scope

This release is the source-code package for the PROOF-CCC ligand-receptor
interaction prediction path and receptor-conditioned communication scoring
path. It intentionally excludes plotting scripts, large public datasets,
cached ESM-C features, pretrained ESM-C parameters, and trained checkpoints.

## Checks completed

- All Python files under `src/`, `scripts/`, and `tests/` passed
  `python -m compileall`.
- The small communication example passed the direct end-to-end test in
  `tests/test_communication.py`.
- `scripts/score_communication.py` completed on the included example and wrote
  cell-type expression summaries, receptor-conditioned TF support, LRI-level
  directional scores, and total/mean communication matrices.
- No `matplotlib`, `seaborn`, `plotly`, `plt`, or `savefig` imports were found.
- No local `/home/zqgaopengfei`, server-IP, or Windows user paths were found in
  the release tree.

## Checks requiring the final GPU/data host

The complete 20-repeat by five-fold benchmark and high-confidence candidate
prediction were not rerun in the local document-audit environment because the
large benchmark inputs, cached residue embeddings, ESM-C 600M parameters, GPU
runtime, and checkpoints are intentionally external. Before creating the
immutable public release, the authors should:

1. create the environment on the final CUDA host;
2. run `pytest -q`;
3. run one benchmark smoke fold using `--repeats 1 --max-repeat 1 --only-fold 0`;
4. confirm the full command against the archived split manifest;
5. record `python`, CUDA, driver, GPU, `torch`, `esm`, NumPy, pandas,
   scikit-learn, SciPy, joblib, and tqdm versions;
6. archive the split manifest, prediction-level results, checkpoints, and
   checksums in a DOI-bearing repository.

Passing source compilation confirms syntax and import structure, not numerical
identity with the server run. Numerical reproduction requires the external
assets listed in `docs/DATA_AND_WEIGHTS.md`.
