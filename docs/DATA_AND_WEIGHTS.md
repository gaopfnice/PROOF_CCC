# Data and model-weight manifest

Large or redistribution-restricted assets are intentionally not committed to
this repository. Their absence is explicit; the code does not silently replace
them with synthetic data.

## ESM-C weights

The final experiments used the frozen `esmc_600m` encoder. The weights are not
stored in Git because of their size. `ESMC.from_pretrained("esmc_600m")` is
called by `src/proofccc_core/run_lri_experiment.py`; the first run downloads the
model through the `esm` package. Users must accept the provider's license and
configure any required access token/cache according to the official ESM
documentation. A local cache can be reused across all four datasets.

## Benchmark datasets

The four processed benchmark matrices are not redistributed in this archive.
Obtain them from the original resources/studies cited in the manuscript, retain
their original identifiers, and create the following layout:

```text
data/
  human/
    human_Related_name.csv
    ligand_gen.csv
    receptor_gen.csv
    feature/ligand.txt
    feature/receptor.txt
  mouse/
    Related.csv
    l-gene650.csv
    r-gene588.csv
    ligand650.csv
    receptor588.csv
  mouse-heart/
    Related.csv
    l-gene574.csv
    r-gene559.csv
    ligand574.csv
    receptor559.csv
  dataset 4/
    ligand-receptor interaction.csv
    ligand sequence.txt
    receptor sequence.txt
```

Dataset-specific filenames are not invented here; they are the filenames read
by the final loader. The binary relation CSV uses ligand identifiers as rows,
receptor identifiers as columns, and `1` for curated interactions. Sequence
files must preserve the same order used by the relation matrix.

## Case-study data

GSE103322 and GSE301741 must be downloaded from GEO under their respective
data-use conditions. The generic communication CLI does not require a specific
accession: it accepts a genes-by-cells count matrix, cell metadata, a scored LRI
table, and a TF list. This decouples the published scoring method from the
original local server paths.

## Files that should be deposited with a paper release

For full computational reproducibility, archive the following in Zenodo or a
similar repository and place the DOI in the manuscript:

1. checksummed processed benchmark inputs;
2. the exact train/validation/test pair indices for all 100 outer folds;
3. cached global and residue-level ESM-C features, or a script plus manifest to
   regenerate them;
4. final model checkpoints and fitted preprocessing objects;
5. prediction-level `y_true` and `y_score` outputs for every fold and method if
   empirical ROC/PR curves are shown;
6. the curated-plus-predicted LRI table used for each case analysis;
7. the TF annotation version and checksum.

Until those assets are deposited, the repository should be described as
containing the complete source code and small executable example, not as a
complete data-and-checkpoint reproduction bundle.
