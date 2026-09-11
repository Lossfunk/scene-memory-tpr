# Sources and attribution

This repository contains original analysis code, explanatory TPR fits, and derived
experimental outputs. The MIT license applies to this repository's contributions;
it does not relicense the upstream scene model, its checkpoint, or either paper.

The frozen target is from **Linda Ariel Ventura, Victoria Bosch, Tim C. Kietzmann,
and Sushrut Thorat**, *Path Integration and Object-Location Binding Emerge in an
Action-Conditioned Predictive Sequence Network* (CogSci 2026):

- [Conference paper](https://escholarship.org/uc/item/1mj18812)
- [Public scene repository](https://github.com/KietzmannLab/minimal_world_model_interp)
- Pinned revision: `0a1755cf6195cfd335838f3c48b79a7b2d7507fb`.

Upstream model source and checkpoint are **not bundled or tracked**. The optional
fetch script obtains just `setup/model.py` and `paper_data/model/final_epoch_79.pth`
from that revision and checks the hashes in [upstream.json](configs/upstream.json).
The pinned asset inventory did not include a license file; consult the upstream
project for its terms. This repository does not assert the historical training
condition of the checkpoint or reproduce the paper's restricted-k experiment.

The representational approach is motivated by **R. Thomas McCoy, Paul Soulos,
Tal Linzen, and Paul Smolensky**, *The Emergent Symbolic Structure of Artificial
Neural Networks* ([arXiv:2608.29530v1](https://arxiv.org/abs/2608.29530v1)). That work
already includes TPR-based interventions and combination-generalization tests.
The contribution here is a small empirical application to the scene network,
with explicit letter–position edits and protected recall measurements. No
DISCOVER source code or paper PDF is redistributed here.

Please cite the scene paper when using its network and the TPR paper when drawing
on its framework, along with this repository's version or commit for these results.

## Explanatory figures

Figure 1 is an original drawing informed by Ventura et al.'s Figure 1B (the
scene and prediction task) and McCoy et al.'s Figure 2.1 (filler–role products,
summation and mapping to a neural representation). Its scene illustrates this
repository's six-distinct-letter evaluation setting; the source scene paper also
uses four or five locations and repeated letters. The colored TPR tiles are an
illustration of the construction, not learned values. No source-paper artwork
is reproduced. Figures 2 and 3 depict this repository's procedure and results.
All three figures can be regenerated using the included plotting scripts.
