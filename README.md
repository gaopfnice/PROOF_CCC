# PROOF-CCC

PROOF-CCC is a leakage-controlled framework for ligand-receptor interaction
(LRI) prioritization followed by directional, receptor-conditioned cell-cell
communication (CCC) scoring. This repository contains the core source used for
the final benchmark and case-analysis workflow. It deliberately excludes
plotting scripts, large datasets, cached protein embeddings, model checkpoints,
and the 600M-parameter ESM-C weights.

## What is included

The LRI prediction path contains:

1. frozen ESM-C global and residue-level protein representations;
2. a pair-aware multi-branch encoder with bidirectional residue attention,
   six convolutional experts, and adaptive expert gating;
3. nested out-of-fold (OOF) contextual learning;
4. positive-memory features constructed from training positives only;
5. ligand-receptor graph-diffusion features constructed inside the training
   boundary;
6. a fixed three-member FT-Transformer ensemble for final probability output;
7. repeated stratified pair-level evaluation and unannotated-pair ranking.

The CCC path combines model-derived LRI confidence, sender/receiver expression
availability, harmonic ligand-receptor matching, and receiver-side
receptor-conditioned TF-expression support. The TF term is associative, not
causal.

## Repository layout

```text
PROOF_CCC/
├── src/
│   ├── proofccc/                 # cleaned reusable communication API
│   └── proofccc_core/            # exact final experiment scripts
├── scripts/
│   ├── prepare_embeddings.py
│   ├── train_benchmark.py
│   ├── predict_lri.py
│   └── score_communication.py
├── configs/benchmark.example.yaml
├── data/example/                 # tiny executable CCC example only
├── docs/
│   ├── DATA_AND_WEIGHTS.md
│   └── TUTORIAL.md
├── tests/
├── environment.yml
├── requirements.txt
└── pyproject.toml
```

## Quick start

For a lightweight test of the communication layer:

```bash
conda env create -f environment.yml
conda activate proofccc
pip install -e .
pytest -q

proofccc-communication \
  --expression data/example/expression.csv \
  --metadata data/example/metadata.csv \
  --pairs data/example/pairs.csv \
  --tf-list data/example/tf_genes.csv \
  --out-dir outputs/example_communication
```

Full embedding, benchmark, LRI prediction, and CCC commands are documented in
[`docs/TUTORIAL.md`](docs/TUTORIAL.md). Required external assets are documented
without ambiguity in [`docs/DATA_AND_WEIGHTS.md`](docs/DATA_AND_WEIGHTS.md).
The correspondence between manuscript modules and source files is listed in
[`docs/MANUSCRIPT_CODE_MAPPING.md`](docs/MANUSCRIPT_CODE_MAPPING.md).

## Reproducibility boundary

The final benchmark used 20 independent repeats and five stratified outer folds
per repeat. Each repeat retained all curated positive interactions and sampled a
new equal-sized set of zero-valued matrix entries as working negatives. This is
pair-level generalization: the same ligand or receptor may occur in training and
test pairs. It is not a protein cold-start evaluation.

All label-derived memory and graph features are training-boundary features.
Nested inner OOF models generate contextual features for outer training pairs,
and outer-test contextual features are averaged over the inner models. The final
probability is the mean of three fixed FT-Transformer members. AUROC and average
precision are computed from unthresholded probabilities; operating-point
metrics use a validation-only threshold.

## Large files and third-party assets

- **ESM-C weights:** not included. The code requests `esmc_600m` through the
  official `esm` package.
- **Benchmark matrices and sequences:** not included. They must be obtained from
  the original resources cited in the paper and arranged using the documented
  filenames.
- **GEO/STOmicsDB data:** not included; access them from the stated repositories.
- **Checkpoints and cached embeddings:** not included in this Git archive. For a
  paper release, deposit them with checksums in a DOI-bearing repository.

The manuscript must not claim that this Git repository alone contains all data,
weights, and trained models unless those assets are actually deposited.

## Environment

The release provides a pinned Python 3.10 reference environment. GPU execution
uses PyTorch 2.3.1 with CUDA 12.1 and ESM 3.1.1. These pins should be verified on
the final public machine before tagging the release; the manuscript should also
state the tested GPU model, driver, operating system, and random seed.

## Citation

Citation metadata is provided in `CITATION.cff`. Add the manuscript DOI and
confirm author order before publishing the immutable GitHub release.

## License

MIT. Confirm that redistribution of every third-party input and model is allowed
under its own license; this license applies only to the code in this repository.
