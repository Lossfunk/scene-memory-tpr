# Editing scene memory with a simple tensor-product representation

**A short empirical study: can an explicit sum of letter–position bindings change what a frozen scene network recalls?** A linear tensor-product representation (TPR) lets us change which letter the network recalls at a chosen location. Adding tanh improves this ability. Neither model predicts the entire network state exactly or changes recall without errors.

The [scene paper](https://escholarship.org/uc/item/1mj18812) by Ventura, Bosch, Kietzmann and Thorat studies a recurrent network that predicts the next letter from the current letter and a saccade-like displacement. Motivated by [McCoy et al.'s TPR framework](https://arxiv.org/abs/2608.29530v1), we ask whether an explicit binding decomposition can support **controlled, location-specific changes in recall**. This is a small follow-up on the public checkpoint, not a replication of every paper result.

## The formulation

For locations observed so far:

$$
M=\sum_i f(\ell_i)\otimes r(p_i),\qquad
r(p)=R\phi(p),\qquad
z=b+Uq+W\,\mathrm{vec}(M).
$$

$$
\widehat H_{\mathrm{linear}}=z,
\qquad
\widehat H_{\mathrm{tanh}}=\tanh(z).
$$

Letters have **26-dimensional** learned filler vectors; positions have **16-dimensional** learned roles built from 27 fixed smooth coordinate features. Each observed location contributes once with weight one. The 1,536 reconstructed coordinates cover all three GRU layers. The context $q$ contains observed current-input, position, movement and timing information; it supplies no queried-letter answer.

Why this form? A static scene is a set of letter–position assignments, so one outer product per observed assignment is a direct, testable hypothesis. Shared smooth roles allow continuous positions without a separate learned vector for each scene. Tanh is a minimal bounded comparison: it acts **after the entire affine sum**, matching the GRU's $[-1,1]$ range. It adds no parameters, but makes state-space edits depend on the baseline activation. Both models have 770,644 parameters; full-dimensional letter vectors do not demonstrate alphabet compression.

We fit both models to network states and to the differences between states produced by running the same viewing sequence on original and changed scenes. The scene network itself is never retrained. The tanh configuration was selected on validation within an earlier **18-specification study**. This repository isolates the linear/tanh comparison; [selection records](results/original_selection.json) and the [original configuration](configs/original_study.json) preserve that context. See [methods](docs/METHODS.md) for the exact features, loss and optimizer.

## Manipulations

At step 34, after 35 input updates, choose previously observed locations with at least three visits, excluding the current location and the already-specified next destination.

- **Replacement:** change letter $a$ to a letter $b$ absent from the scene, at position $p$: $\Delta M=[f(b)-f(a)]\otimes r(p)$.
- **Swap:** exchange letters $a,b$ at positions $p,s$: $\Delta M=[f(b)-f(a)]\otimes[r(p)-r(s)]$. The scene still contains the same letters.

Predict the state for the original and changed assignments, subtract the two predictions, and add that difference to the actual network state:

$$
H'=\mathrm{clip}_{[-1,1]}\left(H+\widehat H(M+\Delta M,q)-\widehat H(M,q)\right).
$$

Clipping is separate from tanh and is used in **both** comparisons. Let the network take five further steps, then ask it to predict the letter at a location. It does not see that letter, or either edited letter, along the way. Test each location on a separate copy of the edited state. This prevents earlier queries from teaching answers to later ones. Score five locations: one replacement and four unchanged locations, or two swapped and three unchanged locations. Exclude the already-specified next destination, which must immediately be observed. [Exact protocol and code links](docs/METHODS.md#intervention-and-probe-timing)

## Observed results

**400 test scenes not used for fitting or selection**, separate from 1,000 training and 250 validation scenes; six unique letters per scene. Error bars are 95% bootstrap intervals, resampling scenes rather than treating queries on the same scene as independent observations.

![Recall and preservation after a single TPR edit, with 95% scene-bootstrap intervals](figures/results.png)

Replacement accuracy is **85.75% → 89.00%**, and average swap-target accuracy is **76.1% → 80.9%**, from linear to tanh. Paired gains are **3.25 points [1.00, 5.50]** and **4.75 points [2.88, 6.75]**, respectively. Both swapped letters are recalled correctly in **63.25% → 68.75%** of scenes, testing each on a separate copy. The figure’s last row requires all five answers to be correct, including unchanged letters, and averages the replacement and swap tasks.

| How well does the model predict the state? | Linear | Tanh |
| --- | ---: | ---: |
| State R²: improvement over predicting the training mean | 0.594 | 0.620 |
| Predicted state is closest to its matching actual state, among six assignments | 90.28% | 91.75% |
| Same next-letter answer after replacing the entire state with the predicted state | 47.75% | 64.25% |

For the six-way test, rearrange three letters while keeping the same positions and viewing sequence. Compare each predicted state with the six actual states: is its own assignment closest? There are 398 eligible scenes in a separate 400-scene set; chance is 16.67%. Unchanged-location accuracy after edits is about 95–96%, versus **98.13%/98.42% without editing**. For comparison, running the network from the beginning on the changed scene gives 99.50% replacement and 98.38% swap accuracy per location. This shows that the network can usually answer these queries when it has actually experienced the changed scene.

[All values and intervals](results/audited.json) · [CSV](results/metrics.csv) · [Figure PDF](figures/results.pdf) · [Evidence format](docs/REPRODUCIBILITY.md#bundled-evidence)

## Interpretation and limits

The simplest TPR captures enough about which letter is where to change recall fairly selectively. Tanh improves recall at edited locations, but errors remain and accuracy at unchanged locations drops by about two to three percentage points. Crucially, **adding a predicted change to the real state works much better than replacing the entire state with a prediction**. The former keeps whatever the TPR fails to explain in the original state.

This supports a useful approximation of letter–position bindings, not proof that the network literally stores or reads a TPR. Training examples change letters throughout the viewing history, so the learned differences may include effects of seeing different letters as well as remembering different assignments. A static scene memory could still be only one part of the recurrent state. Evidence comes from one checkpoint whose training condition remains unverified.

Next: withhold particular letter–region combinations during fitting, test repeated letters, and ask whether the same factors explain newly learned or overwritten assignments without rewriting past observations. Also test whether the network reads letters through these factors, and repeat the study on checkpoints with documented training conditions.

## Run it

From this directory, using Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/audit_results.py --plot
```

The audit rebuilds headline statistics and the figure from bundled evidence; **no upstream download is needed**. To replay the shipped fitted models against the frozen target:

```bash
python scripts/fetch_assets.py
python scripts/run.py evaluate --device auto
```

`auto` uses Apple MPS when available and CPU otherwise. The fetch downloads and verifies two pinned upstream files (about 61 MB); they remain ignored by Git. [Refitting, smoke checks, provenance and reproducibility limits](docs/REPRODUCIBILITY.md)

Original analysis code is MIT-licensed. The upstream model and checkpoint are not redistributed or relicensed; see [sources and attribution](THIRD_PARTY.md). Results were obtained on 10 September 2026; this focused release was assembled on 11 September 2026.
