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

The README was subsequently reorganized to present the basic TPR results before
the tanh comparison. The added methods diagram was rendered and visually checked;
its plotting script is included. The explanatory changes do not alter experimental
code, fitted models or results.

The expanded symbol definitions and revised methods figure were checked against
`CleanTPR.edit`, `probe_path`, and the five-copy evaluation code: both TPR
predictions share context, their difference is added to the actual state and then
clipped, and each query consumes five intervening updates plus one outgoing-query
update without observing the queried or edited locations. The plot captions now
define axes, intervals, reference markers and the all-five scoring rule. The
regenerated evidence audit passed; numerical results remain unchanged.

## Final three-figure report audit

The final report follows scene/GRU/TPR relationship (Figure 1), intervention (Figure 2),
then results (Figure 3). Figure 1 was checked against Ventura et al.'s Figure 1B
and scene-generation methods, McCoy et al.'s Figure 2.1, and this repository's
sampler and TPR computation. It is an original schematic: no source artwork or
fitted activation values are presented as data. The caption distinguishes the
six-distinct-letter tests from the source paper's broader scene distribution.

All three generated figures were visually inspected; layout collisions in the
initial setup draft were corrected. The text and figure conventions agree on
inputs, state dimensions, context, TPR-side subtraction, clipping, timing,
independent queries and scoring. The eight contract tests and saved-evidence
audit passed again. All report equations parsed locally and relative file links
resolved. The release hashes and ZIP contents were verified; the ZIP contains
only the intended release files, including `.gitignore`. GitHub's live renderer
was not exercised by this local check, and no new scientific experiments were run.

A further layout and terminology review checked all three figures at 840-pixel
README width and rendered the PDF exports separately. Main annotation bounds
were checked for overlap and clipping, followed by visual inspection of labels,
arrows, axes and box padding. No annotation overlaps or canvas clipping remained.
Figure 1 now explicitly distinguishes the scene, the GRU that predicts it, and
the TPR approximation of the GRU's state. Its tile addition shows one binding
plus the sum of the others, removing a distracting omitted-term ellipsis.
The TPR is not presented as a second predictor of the scene or as a complete
model of recurrent state updates. The README and all figure descriptions use
the same distinction. These are presentation changes; results are unchanged.
