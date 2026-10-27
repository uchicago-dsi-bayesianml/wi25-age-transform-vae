# Earlier architectures

Attempts from late February and early March 2025 that the team kept for the record. None of them produced the figures in the main README.

- `model_architecture*.ipynb`: a β-VAE that was meant to discover an age direction without supervision, with a VGG perceptual loss in two of the versions. That perceptual loss is the "deep feature consistent" idea behind the `dfc` prefix that stuck to the final notebook names.
- `conditional-dfc-model-test.ipynb` and the `viewer*` notebooks: the first conditional models and their viewers, at 64×64 resolution.

They show how the design moved from unsupervised disentanglement to explicit conditioning on age and gender.
