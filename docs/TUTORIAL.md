# PROOF-CCC tutorial

## 1. Installation

Python 3.10 and a CUDA-capable GPU are recommended for ESM-C feature extraction
and nested training. The pinned files are a reproducibility environment for the
public package; record the final GPU driver and CUDA toolkit in the release DOI.

```bash
conda env create -f environment.yml
conda activate proofccc
pip install -e .
pytest -q
```

CPU-only installation is sufficient for the communication-scoring example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Prepare benchmark inputs

Follow `DATA_AND_WEIGHTS.md`. The loader validates matrix dimensions and
sequence counts. Do not reorder the relation matrix independently of the FASTA
or sequence files.

## 3. Extract frozen ESM-C features

```bash
python scripts/prepare_embeddings.py \
  --data-dir data \
  --out-dir outputs/lri_esmc_600m \
  --cache-dir outputs/local_cache_esmc \
  --datasets human mouse mouse-heart dataset4 \
  --model-name esmc_600m \
  --device cuda \
  --chunk-len 1022 \
  --max-residues 512
```

The 600M-parameter weights are downloaded by the ESM package and are not part
of this Git repository.

## 4. Run the final benchmark pipeline

```bash
python scripts/train_benchmark.py \
  --data-dir data \
  --embedding-dir outputs/lri_esmc_600m/embeddings \
  --cache-dir outputs/local_cache_esmc \
  --out-dir outputs/final_benchmark \
  --datasets human mouse mouse-heart dataset4 \
  --repeats 20 \
  --folds 5 \
  --inner-oof-folds 5 \
  --neg-ratio 1.0 \
  --device cuda
```

For each repeat, all curated positives are retained and a new 1:1 sample of
zero-valued matrix entries is drawn as the working negative set. Stratified
five-fold outer evaluation produces 100 records per dataset. Label-derived
memory and graph features are constructed only inside their training boundary.
Within each outer training fold, five inner models produce OOF features; the
outer-test features are averaged across those inner models. A fixed
three-member FT-Transformer ensemble produces the final probabilities. The
classification threshold is chosen on validation data only and does not affect
AUROC or average precision.

## 5. Rank unannotated LR candidates

```bash
python scripts/predict_lri.py \
  --data-dir data \
  --embedding-dir outputs/lri_esmc_600m/embeddings \
  --cache-dir outputs/local_cache_esmc \
  --out-dir outputs/high_confidence \
  --datasets human \
  --device cuda \
  --candidate-mode hard_prefilter \
  --max-candidates 50000 \
  --top-k 5000 \
  --min-score 0.99
```

Important: the production script first selects candidates meeting
`--min-score`. If fewer than `--top-k` qualify, it fills the file with the
highest-ranked candidates up to `top-k`. Therefore the output filename means
"top-ranked candidates"; downstream analyses that require an absolute cutoff
must filter the `score` column explicitly (the HNSCC main analysis used 0.90).

## 6. Compute receptor-conditioned communication strength

Run the included small example:

```bash
proofccc-communication \
  --expression data/example/expression.csv \
  --metadata data/example/metadata.csv \
  --pairs data/example/pairs.csv \
  --tf-list data/example/tf_genes.csv \
  --out-dir outputs/example_communication
```

Input schemas:

- expression: first column is the gene symbol; remaining columns are cells and
  values are non-negative counts;
- metadata: `cell` and `cell_type` columns;
- pairs: `ligand`, `receptor`, `model_probability`, and optional `source`;
- TF list: one gene symbol per row, without a required header.

For ligand `L`, receptor `R`, sender `s`, and receiver `r`, availability is

```text
A(g,c) = sqrt(mean_log1p(g,c) * detection(g,c)) * specificity(g,c)^lambda
```

and expression matching is the harmonic mean of ligand and receptor
availability. Receiver cells are split into the top and bottom receptor-expression
tails. Positive standardized TF-expression differences are used to obtain
`RCTF_raw`, followed by global min-max normalization. The final score is

```text
Score(L,R,s,r) = P(L,R)^alpha * LR_matching(L,R,s,r)
                 * (1 + rho * RCTF_norm(R,r)).
```

The default main analysis retains only contexts with `RCTF_raw > 0`. Use
`--allow-zero-rctf` only for the stated diagnostic analysis. RCTF is an
expression association, not evidence of causal receptor-to-TF regulation.

## 7. Provenance of the source tree

`src/proofccc_core/` preserves the exact final experiment scripts and their
cross-import names. `src/proofccc/` contains the cleaned reusable communication
API. Plotting scripts and historical exploratory scripts are intentionally not
included.
