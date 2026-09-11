# Release checks

Checked on 11 September 2026 using Python 3.12.14, NumPy 2.4.2 and PyTorch 2.10.0.
These checks replay existing evidence and test the exported code. They do not add
new scientific test scenes or repeat the original model-selection study.

- **Eight contract tests passed:** saved-model predictions and edits match the
  archived fixture; the linear edit follows the outer-product formula; tanh acts
  after the complete affine sum; no-op edits and bounds behave correctly;
  queried-answer fields and future observations do not enter explanatory inputs;
  swaps preserve the scene's letters; regenerated query paths match the archive.
- **Offline audit passed:** all reported behavioral means and confidence intervals,
  assignment-identification accuracy, state R² and immediate next-letter agreement
  agree with the bundled evidence. The figure was regenerated and visually checked.
- **Full CPU replay passed:** regenerated the 400-scene test set and the separate
  400-scene assignment set (398 eligible), then evaluated both shipped TPRs and
  both reference conditions. All behavioral accuracy counts, their confidence
  intervals, assignment accuracy and immediate prediction agreement match the
  original MPS results. Across 184 compared numerical entries, the largest absolute
  difference is 1.20×10⁻⁷ (rounding in continuous state/probability metrics).
  [Machine-readable comparison](../results/release_validation.json)
- **Short fit-and-evaluate check passed:** collected 12 training and six validation
  scenes, trained each model for two epochs, and evaluated on eight scenes. This
  checks that refitting runs end to end; its accuracies are not research results.
- **Portability:** the tests and evidence audit also ran from a separate copy
  without the parent workspace, upstream files or generated run data. Package
  metadata was checked by building a wheel without downloading dependencies.
- **Upstream assets:** the local-copy fetch path verified both pinned SHA-256 hashes.
  The HTTP download path is implemented but was not exercised in this release check.

Full 100-epoch refitting was not repeated for packaging. Historical fits used MPS;
this release's full replay and short training check used CPU. A fresh environment
installing all dependencies from the internet was not tested.

To check that the source, fitted models and archived evidence match the release:

```bash
python scripts/verify_release.py
```

The manifest excludes regenerated summaries/figures, this validation record, and
ignored runtime data. It detects accidental changes; it is not a signed attestation.
