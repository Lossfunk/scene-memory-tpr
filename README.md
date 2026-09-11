# Editing scene memory with a simple tensor-product representation

**Can a simple model of “which letter is where” explain a recurrent network’s memory well enough to change what it recalls?** We fit a tensor-product representation (TPR) to a trained scene network, calculate changes in that explanatory model, and apply them to the network’s hidden state. The basic linear TPR gets surprisingly far: the network recalls the intended replacement letter in **85.75%** of tests, while accuracy at unchanged locations remains about **96%**. We first examine this result, then ask whether adding a bounded nonlinearity, tanh, helps further.

## Why this experiment?

In [Ventura, Bosch, Kietzmann and Thorat’s scene study](https://escholarship.org/uc/item/1mj18812), a recurrent network observes letters one at a time and predicts the next letter from the current letter and a saccade-like displacement. This task requires remembering letters and their locations. We use the **publicly released trained GRU from that study**, available in [the authors’ scene-network repository](https://github.com/KietzmannLab/minimal_world_model_interp); its exact training condition is not documented in the checkpoint itself.

Motivated by [McCoy et al.’s TPR framework](https://arxiv.org/abs/2608.29530v1), we test a concrete hypothesis: a sum of letter–position bindings can model this memory. Matching recorded states is one test. A stronger test is to change a binding in the TPR model and ask whether the predicted change makes the GRU recall the new assignment, while preserving the others. The GRU’s weights stay fixed throughout.

## Start with the simplest TPR

For the locations observed so far, bind each letter vector to a position vector and sum:

$$
M=\sum_i f(\ell_i)\otimes r(p_i),\qquad r(p)=R\phi(p).
$$

Here, $f$ is a learned letter vector, $r$ is a learned position vector, and $\otimes$ is the outer product. Each observed location contributes once. This is a natural starting point for a static scene: it explicitly represents the assignments we want to explain and manipulate.

A linear map translates this tensor into a prediction of the GRU’s hidden state:

$$
\widehat H(M,q)=b+Uq+W\,\mathrm{vec}(M).
$$

The context $q$ accounts for the current input, position, outgoing movement, time and number of observed locations; it contains no queried-letter answer. Letter vectors have 26 dimensions and position vectors have 16. Positions are represented through 27 fixed smooth coordinate features, allowing the same learned map to work across continuous positions. The predicted state has 1,536 entries, covering all three GRU layers. [Exact features and dimensions](docs/METHODS.md#representation-and-training)

We fit this **model of the GRU** using recorded GRU states, plus the differences between states produced by running the same viewing sequence on original and changed scenes. There are 1,000 training and 250 validation scenes. Only the TPR model is fitted; the GRU is never retrained.

## Test the model by changing recall

![Left: change a letter assignment in the TPR model and subtract its two state predictions. Right: add that predicted difference to the actual GRU state, then test recall.](figures/methods.png)

At step 34, after 35 input updates, choose familiar locations with at least three visits, excluding the current location and the already-specified next destination. Make either change **in the TPR**:

- **Replacement:** at position $p$, replace letter $a$ with a letter $b$ absent from the scene. The tensor change is $\Delta M=[f(b)-f(a)]\otimes r(p)$.
- **Swap:** exchange letters $a,b$ at positions $p,s$. The tensor change is $\Delta M=[f(b)-f(a)]\otimes[r(p)-r(s)]$. Keeping the same letters makes this a test of their assignments to locations.

Next, calculate the difference between the TPR model’s two state predictions:

$$
\Delta\widehat H=\widehat H(M+\Delta M,q)-\widehat H(M,q).
$$

**Both predictions and their subtraction are computed in the explanatory TPR model.** We then add that predicted difference to the actual GRU state $H$:

$$
H'=\mathrm{clip}_{[-1,1]}(H+\Delta\widehat H).
$$

Clipping keeps every hidden-state entry within the GRU’s range. We do not subtract a separately recorded “changed-scene” GRU state, or replace the entire state with the TPR prediction. This tests whether the change proposed by the TPR has the intended effect on the GRU’s behavior.

Let the edited GRU take five further steps, then predict a letter at a queried location. It never sees the queried or edited letters along the way. Test each location on a separate copy of the edited state, so one query cannot teach an answer to another. Score five locations: one replacement and four unchanged locations, or two swapped and three unchanged locations. The already-specified next destination is excluded because it must immediately be observed. [Exact timing and code](docs/METHODS.md#intervention-and-probe-timing)

## How well does the basic TPR do?

On **400 test scenes not used for fitting or selection**, each containing six distinct letters:

| GRU recall after an edit proposed by the linear TPR | Accuracy |
| --- | ---: |
| Intended replacement letter | **85.75%** |
| Intended swapped letter, averaged over the two locations | **76.1%** |
| Both swapped letters correct, testing each on a separate copy | **63.25%** |
| Unchanged letters after replacement | **96.06%** |
| Unchanged letters after swapping | **95.58%** |

Without editing, accuracy on those unchanged locations is 98.13% and 98.42%, respectively. Thus, the TPR makes substantial, fairly selective changes to recall, with a two-to-three-point cost at other locations. As a reference, if the GRU actually experiences the changed scene from the beginning, it scores 99.50% on replacements and 98.38% per swapped location. There is still room to improve the artificial edits.

**Does the TPR also predict the right representation?** Rearrange three familiar letters to create six assignments with the same positions and viewing sequence. For each predicted state, ask whether its matching actual GRU state is closer than the other five. The linear TPR identifies the correct assignment **90.28%** of the time, versus 16.67% chance, across 398 eligible scenes from a separate 400-scene set. Its state R² is **0.594**, measured against predicting the training-set mean state.

However, if we replace the *entire* GRU state with the TPR prediction, the next-letter answer agrees with the original GRU’s answer only **47.75%** of the time. This is a different test from adding a predicted change to the real state: adding a change retains the parts of the real state that the TPR does not explain. The model captures useful assignment structure without yet accounting for the whole recurrent state.

## Does tanh help further?

The GRU’s hidden-state entries lie in $[-1,1]$, whereas a linear TPR prediction can extend beyond that range. To respect this constraint, we fit a second model that applies **tanh after the complete sum**:

$$
\widehat H_{\mathrm{tanh}}(M,q)=\tanh\left(b+Uq+W\,\mathrm{vec}(M)\right).
$$

The letter–position decomposition stays the same. Tanh adds no parameters and is included during fitting. We still subtract the two model predictions and add their difference to the actual GRU state. The final clipping step is used for **both** models; it is separate from bounding the predictions with tanh.

| Measure | Basic linear TPR | TPR with tanh |
| --- | ---: | ---: |
| Replacement letter recalled | 85.75% | **89.00%** |
| Swapped letter recalled, per location | 76.1% | **80.9%** |
| Both swapped letters correct | 63.25% | **68.75%** |
| Unchanged letters after replacement | 96.06% | 96.19% |
| Unchanged letters after swapping | 95.58% | 95.42% |
| Correct assignment among six state alternatives | 90.28% | 91.75% |
| State R² | 0.594 | 0.620 |
| Same next-letter answer after replacing the entire state | 47.75% | 64.25% |

Tanh improves replacement accuracy by **3.25 percentage points** (95% interval: 1.00–5.50) and swap accuracy per location by **4.75 points** (2.88–6.75). Accuracy at unchanged locations stays similar. These intervals compare the models on the same scenes and resample scenes, not individual queries. [Comparison plot with intervals](figures/results.png) · [All results](results/audited.json) · [CSV](results/metrics.csv)

Both models have 770,644 parameters. The tanh configuration was selected using validation results within an earlier 18-specification study; this focused report presents it alongside the matched linear control. [Selection record](results/original_selection.json) · [Original study configuration](configs/original_study.json)

## What does this tell us?

**The basic TPR already provides a useful model of letter–position memory:** its predictions distinguish assignments, and its proposed edits often change recall at the intended location while largely preserving other letters. Respecting the GRU’s activation bounds with tanh improves this result further. That is meaningful progress toward an editable model of the network’s memory.

It does not yet establish that the GRU literally stores or reads a TPR. Full-dimensional letter vectors can support separate position maps for each letter, and the fitting examples change letters throughout the viewing history, so predicted differences may include effects of past observations. The incomplete prediction of the entire state leaves open whether a static scene-memory component coexists with other recurrent processing. Results also concern one scene-network checkpoint with unverified training-condition provenance.

Next steps are to withhold particular letter–region combinations during fitting, test repeated letters, and test newly learned or overwritten assignments without rewriting past observations. We should also ask whether the GRU reads letters through the fitted factors, and repeat the study on checkpoints with documented training conditions.

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

The included results let you reproduce the statistics and comparison plot without downloading the scene network. To run the recall experiments with the fitted TPR models, first download the trained scene network:

```bash
python scripts/fetch_assets.py
python scripts/run.py evaluate --device auto
```

`auto` uses Apple MPS when available and CPU otherwise. See [reproducing the experiments](docs/REPRODUCIBILITY.md) for fitting the TPR models from scratch, and [methods](docs/METHODS.md) for the full experimental specification.

Analysis code is MIT-licensed. See [sources and attribution](THIRD_PARTY.md) for the scene network and related work.

## Acknowledgment

The analyses, code, and report were developed with assistance from Astra, using **medium** and **xhigh** reasoning settings.
