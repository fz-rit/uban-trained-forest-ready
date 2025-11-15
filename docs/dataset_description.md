## Semantic3D
- [dataset page](https://semantic3d.ethz.ch/)
- 8 classes (1-7); label 0 is 'unlabeled', 4.2 billion points (train + test)
- 15 scans for training, 15 for testing

## ForestSemantic
- [dataset page](https://zenodo.org/records/15193973)
- 6 classes (1-6), 0.35 billion points (train + test); there were about 0.01% points with strange label=7, which we simply ignored.
- 3 plots; 2 for training, 1 for testing; each plot was from co-registered 5 scans.

## DigiForests
- [dataset page](https://www.ipb.uni-bonn.de/data/digiforest-dataset/)
- 5 classes (1-5): 1-ground, 2-shrub, 3-stem, 4-canopy, 5-miscellaneous; label 0 is 'unlabeled'.
- 5 aerial (plots c1, d2, m1, m3, m5; no labels) + 6 ground plots (plots c1\*3, d2\*3, m1, m3, m4, m5; with labels).
- We can probably use the ground points for supervised fine-tuning and the aerial points for UDA.