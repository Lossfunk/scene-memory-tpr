# Exact methods

## Target and coordinates

The frozen public target has three 512-unit GRU layers (1,536 concatenated hidden
coordinates; 5,108,976 parameters). Its configuration is inferred from the pinned
checkpoint and loaded strictly. The target state is the raw recurrent state after
processing the current letter and its outgoing displacement; it predicts the next
letter. Inference uses the public projections, native PyTorch GRU, normalization
and output head, with dropout disabled. See [adapter](../src/binding_dynamics/model.py)
and [upstream hashes](../configs/upstream.json). The checkpoint's training condition
is not determined by its filename or the present results.

Our [scene generator](../src/binding_dynamics/episodes.py) creates six distinct
letters sampled from 26, at positions in [-4,4]² with minimum separation 0.25.
The first location is the origin. Each next visited location is sampled uniformly
from the other five; adjacent repeats are excluded. There are 100 input updates.
This is our specified evaluation distribution, not a claim of exact identity to
all original paper samplers. Positions used in explanatory features are integrated
from observed displacements. Unobserved locations are masked; stable node IDs are
bookkeeping only.

## Representation and training

The model is in [clean_tpr.py](../src/binding_dynamics/clean_tpr.py).
`f` is a 26×26 embedding table; `R` maps 27 position features to 16 roles; `W` maps
26×16=416 tensor entries to 1,536 state entries. Bias has 1,536 entries; the context
map is 1,536×84. Total: **770,644** parameters for either link.

For $p=(x,y)$, the fixed position features are:

$$
\phi(p)=[1,x/4,y/4,\{\sin(\pi k^Tp/4)\}_k,
\{\cos(\pi k^Tp/4)\}_k].
$$

The twelve wave vectors $k$ are
`(.5,0), (0,.5), (1,0), (0,1), (1,1), (1,-1), (2,0), (0,2), (2,2), (2,-2), (4,0), (0,4)`.
The context concatenates current-letter one-hot (26), current and next position
features (27 each), outgoing displacement divided by 8 (2), time divided by 100
(1), and observed-location count divided by 6 (1): 84 total.
[Exact feature implementation](../src/binding_dynamics/features.py)

An observed location contributes one term irrespective of observation count.
For each scene and snapshot t=34,64,99, choose two eligible locations observed at
least three times and distinct from current/next. Construct original, single
replacement and two-location swap variants. Replacement letters are absent from
the scene. Each variant is naturally replayed from zero state with the same path;
changed letters appear throughout its past. The current context and geometry
are identical across variants. [Collection](../src/binding_dynamics/data.py)

Training minimizes state MSE over all variants plus MSE between predicted and
actual (single−original, swap−original) differences, with equal weights. It uses
no output/KL training loss. This is supervised counterfactual fitting, not merely
unsupervised discovery from factual activations.

Initialize with training-only standardized full letter×position ridge regression
(penalty .01), followed by Tucker projections into the chosen filler/role ranks.
Add Gaussian factor perturbations of scale .001 with fitting seed 51. Use AdamW
(learning rate .001, weight decay 1e−5), an additional dimension-group L2,1 penalty
of 1e−5 on filler/role factors, gradient-norm clipping at 5, and batches of 64
scene-time families containing all three variants. Train at most 100 epochs,
patience 12, retaining the minimum validation state-plus-difference objective
with an improvement threshold of 1e−7. [Fitting code](../src/binding_dynamics/clean_tpr_fit.py)

All times and variants of a scene stay in one split. Seeds are 413001/413002/413003
for train/validation/test, with 1,000/250/400 scenes. A separate matching cohort
uses seed 413004 and 400 scenes. [Focused configuration](../configs/experiment.json)

The original study compared 18 specifications, with additional fit seeds for the
selected model. Selection maximized validation delay-five all-five joint accuracy
among bounded-reconstruction and bounded-edit candidates, with parameter-count
and state-MSE tie-breaks. The linear model is a matched control. This release
ships seed-51 linear and tanh fits only. It does **not** portray them as the sole
models considered before observing validation. [Historical validation](../results/original_validation.json),
[selection](../results/original_selection.json), [pretest seal](../results/original_pretest_seal.json).
Refitting these two arms alone is not a rerun of the full original selection study.

## Intervention and probe timing

At t=34 the state has processed input letters 0..34 and outgoing movements 0..34.
Its pending destination is node `visits[35]`. Edited locations cannot be that node
or the currently observed node, and each has at least three prior observations.

For each original state, construct replacement and swap predictions by changing
the relevant map labels while holding context fixed. Add their predicted state
differences once to the actual state and clip every coordinate into [-1,1]. Tanh,
when used, bounds each reconstruction *before* taking their difference. Clipping
alone is not equivalent. Zero-change edits are exact no-ops. Filler vectors and
role vectors are not individually passed through tanh.

Score all five locations except the pending destination. For each location and
manipulation, clone the same edited state independently, then consume six inputs:
five intervening updates followed by one outgoing query update. The final query
letter is never fed in. All edited locations and that branch's queried location
are excluded from the six observed source nodes. Unchanged nonqueried locations
can still be visited. No zero displacement is introduced.

Probe RNG uses seed sequence `[split_seed, scene_id, delay, query_node, 811]`.
All compared conditions receive exactly the same branch inputs. The next label
is scored after the final outgoing movement. [Probe construction](../src/binding_dynamics/interventions.py),
[five-location evaluation](../src/binding_dynamics/evaluation.py).

The replacement task has one target and four unchanged queries. The swap task has
two targets and three unchanged queries. “Both swap targets correct” means both
independent branches succeed. “All five correct” means all five independent
branches succeed; the primary joint statistic averages that indicator across the
replacement and swap tasks, then scenes. It is not sequential recall on one branch.

No-edit controls retain the original state. Donor controls use the natural
counterfactual state. A donor changes the network's perceived past as well as its
map assignment; it is not unique ground truth for a semantic edit preserving the
past. Neither donor agreement nor successful artificial writing establishes how
the network naturally learns or overwrites bindings.

## Metrics and uncertainty

For full-state R², sum squared error over factual test snapshots and coordinates;
the baseline is the training factual-state mean. Immediate agreement replaces the
whole hidden state with its prediction, applies the frozen output head, and asks
whether its argmax agrees with the original argmax. This differs from delayed
difference editing, which retains the actual residual state.

The assignment test permutes three familiar dormant letters among their positions,
giving six natural variants per family. Inventory, geometry, context and viewing
schedule stay fixed. Each predicted variant is compared with all six actual
states by squared Euclidean distance. Ties receive fractional credit. Average
over six candidates within each family, then families. Two scenes fail the
three-eligible-location criterion, leaving 398 families; exclusions do not depend
on outcomes. Chance top-1 is 1/6. [Metric implementation](../src/binding_dynamics/metrics.py)

Intervals use 4,000 bootstrap resamples of entire scene families, seed 82319,
and percentile limits 2.5/97.5. Paired differences subtract model scores within
the same family before resampling. Probe branches and assignment permutations
are not treated as independent scene observations. State R² is reported as a
descriptive point estimate. [Offline audit](../scripts/audit_results.py)

The results are on scene-held-out data; they do not establish performance on
letter–region combinations withheld from approximation training or on bindings
absent from target training. Full-dimensional fillers can support independent
smooth maps for each letter. Exact factorization, unbinding, and systematic reuse
require further controls. Failure to explain the whole state also does not exclude
a static TPR memory component alongside other recurrent activity.
