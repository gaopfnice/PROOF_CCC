# Public-release checklist

- [ ] Replace placeholder authors and repository URL in `CITATION.cff`.
- [ ] Add manuscript DOI and software archive DOI.
- [ ] Verify `environment.yml` on the final CUDA host and record `conda list --explicit`.
- [ ] Deposit checksummed split manifests, checkpoints, predictions, and processed data.
- [ ] Confirm that no SSH keys, passwords, access tokens, absolute private paths, or patient identifiers are tracked.
- [ ] Confirm third-party dataset and ESM-C redistribution terms.
- [ ] Run `pytest -q` and `python -m compileall src scripts`.
- [ ] Tag the exact version cited in the manuscript.
