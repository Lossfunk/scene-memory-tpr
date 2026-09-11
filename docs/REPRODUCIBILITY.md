# Reproducing the foray

There are three levels: audit bundled evidence, replay bundled fitted models, or
refit the two approximators. The frozen scene network is never trained here.
All commands below run from the repository root, after the README environment setup.

## Audit without upstream assets

```bash
python -m unittest discover -s tests -v
python scripts/audit_results.py --plot
python scripts/draw_methods.py
```

The tests check saved-model forward/edit parity, the exact linear tensor edit,
tanh placement, no-op/bound behavior, absence of queried-answer dependence,
prefix causality, swap inventory preservation, and probe-path regeneration with
no forbidden refresh. The audit recomputes behavioral endpoints and confidence
intervals, paired gains, state R², assignment identification, and immediate output
agreement from the bundled evidence and compares them with the archived report.
It writes `results/audited.json`, `results/metrics.csv` and `figures/results.{png,pdf}`.
These are regenerated summaries of old evidence, not fresh experiments.
`draw_methods.py` separately regenerates the explanatory methods diagram.

## Replay the fitted models

```bash
python scripts/fetch_assets.py
python scripts/run.py evaluate --device auto --work runs/reproduction
```

Only the pinned upstream model source and checkpoint are fetched. Both SHA-256
hashes are checked before use, and existing mismatched files are left untouched.
For an existing local copy of the upstream tree, use:

```bash
python scripts/fetch_assets.py --from-directory /path/to/minimal_world_model_interp
```

Alternatively set `SCENE_MODEL_ROOT` to that tree before starting the runner;
the same hashes are required. Upstream assets, generated data and rerun outputs
are ignored by Git. They are not part of the public release.

Replay regenerates the 400-scene causal cohort and separate matching cohort from
the recorded seeds, runs the shipped linear/tanh models and no-edit/donor controls,
and saves detailed predictions, states, probes and results under `runs/reproduction`.
The training mean needed for R² is included with the fitted models, so replay
does not require fitting or regenerating training targets.

The default `auto` chooses MPS if available, otherwise CPU. `--device mps` fails
explicitly when MPS is unavailable; `--device cpu` forces CPU. CUDA support has not
been implemented or tested. Exact historical numerical identity across different
devices and library versions is not guaranteed; the archived evidence defines the
reported result. This remains a replay of known scenes, not a fresh confirmatory test.

## Refit only these two models

```bash
python scripts/run.py fit --device auto --work runs/refit
python scripts/run.py evaluate --device auto --work runs/refit --model-dir runs/refit/models
```

This collects training/validation targets, creates the ridge/Tucker initializer,
fits the two configured models, and evaluates those new fits. Generated artifacts
are kept separate from `models/` and `results/`. Existing fitted files and completed
evaluations are not overwritten. Use a new work directory for another run.
The original 18-candidate selection context is preserved in `configs/original_study.json`
and `results/original_*`; the focused runner intentionally does not execute it.

A small engineering smoke check exercises collection, two short fits, and evaluation:

```bash
python scripts/run.py fit --smoke --device cpu --work runs/smoke
python scripts/run.py evaluate --smoke --device cpu --work runs/smoke --model-dir runs/smoke/models
```

Smoke results are not scientific evidence. The smoke and full configurations cannot
share a work directory. Snapshot times and the intervention protocol are unchanged.

## Bundled evidence

| File | Contents |
| --- | --- |
| `models/*.pt` | Two original seed-51 fitted TPRs, with fit metadata; these are not target GRU checkpoints |
| `models/training_mean.npy` | Training factual-state mean for the reported R² baseline |
| `results/reported.json` | Focused archived model/control results, including original numerical metrics |
| `results/evidence.npz` | Per-query logits for both models and no-edit/donor controls; exact probes, answers and target masks; per-snapshot squared-error/baseline sums; per-family 6×6 matching distances; immediate-agreement indicators |
| `tests/fixture.npz` | Eight original scene families with observed inputs, states, reference predictions and edits |
| `results/original_selection.json` | Validation selection rule and selected model |
| `results/original_validation.json` | Full validation record at selection, retained for transparency |
| `results/original_pretest_seal.json` | Historical file hashes; only the focused subset is distributed here |
| `results/provenance.json` | Source/data hashes and scoped packaging changes |
| `results/release_manifest.json` | SHA-256 hashes of source, fitted models and archived evidence |

Array conventions: probe labels `[400,2,5,6]`, actions `[400,2,5,6,2]`, and logits
`[400,2,5,26]`; manipulation axis is replacement then swap. The five queries have
their own independently cloned paths. Factual reconstruction statistics cover
1,200 scene-time rows (400 scenes × three times); assignment distances are
`[398,6,6]`, with predicted variants as rows and actual variants as columns.
No training data, paper PDFs, account details, or private filesystem links are
required. Full activation arrays are regenerated through replay instead of
being committed as large binaries.

## Environment and numerical provenance

The archived experiment used Python 3.12.14, NumPy 2.4.2 and PyTorch 2.10.0 on
Apple MPS. `requirements.txt` pins the main packages used; `requirements-macos.lock.txt`
records the fuller original environment. The source package has looser dependency
bounds for convenience, which are not a claim of testing every supported version.
Use the pinned requirements for the closest replay.

The release extracts relevant functions from the frozen experiment source,
preserving the model and fit computation. Unused queried-answer/age enrichment
was removed from the collector; the clean model never consumed those fields.
Older unused helper branches remain in a few archived core modules, but the
focused configuration and runner fit only the linear/tanh pair with clipped edits.
No generic history model or alternate experimental suite is executed.

See [release validation](VALIDATION.md) for the checks actually run while preparing
this repository. Source hashes establish provenance, not evidence that every
unshipped historical candidate can be reproduced by the focused runner.

## Publish your copy

The folder is an initialized Git repository with no remote configured. Review the
files, then commit it using your Git identity and add your GitHub remote:

```bash
git add .
git commit -m "Release simple scene TPR study"
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

`YOUR_GITHUB_REPOSITORY_URL` is your repository URL. The optional upstream assets
and all `runs/` directories remain ignored. Nothing is published by the analysis
or replay commands.
