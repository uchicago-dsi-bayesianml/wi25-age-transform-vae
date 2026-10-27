<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Conditional Variational Autoencoders for Age Transformation in Facial Imagery

> **Example project from Winter 2025.** This is a copy of [bhstoller/AgeTransform-VAE](https://github.com/bhstoller/AgeTransform-VAE), shown with the authors' work and commit history intact. Course staff made the edits listed under [Changes by course staff](#changes-by-course-staff).

Given one photo of a face, can a generative model show what that person looked like younger, or will look like older, while keeping them recognizable? The team answered this with a conditional variational autoencoder (CVAE) trained on 55,000 labeled face images, conditioned on age and gender.

![Age transformation examples](https://github.com/user-attachments/assets/29226943-95e9-4ce7-8449-c5c19ad1604d)

## Summary

A CVAE with a convolutional encoder and decoder was trained from scratch in PyTorch on the MORPH-2 dataset for 500 epochs. The encoder maps an image and its age and gender to a 256-dimensional latent code. The decoder reconstructs the image from that code and a condition vector, so decoding the same code with a different age produces the same person at a different age. The main engineering effort went into preprocessing: aligning eyes, masking the background, and matching brightness and color histograms to the training set, so that new photos, including the team's own, land in the distribution the model learned. Reconstructions are blurry, as expected from a VAE trained with a pixel-wise loss, but identity is preserved and the aging effect is visible and smooth. See [Results](#results).

## Team and contributions

- **Bradley Stoller** ([@bhstoller](https://github.com/bhstoller))
- **Cassandra Maldonado** ([@CassandraMaldonado](https://github.com/CassandraMaldonado))
- **John Melel** ([@johnmelel](https://github.com/johnmelel))
- **Kyler Rosen** ([@KyKatRo](https://github.com/KyKatRo))

The project was built between February 6 and March 13, 2025. The [commit history](https://github.com/uchicago-dsi-bayesianml/wi25-age-transform-vae/commits/main) records who built what.

## Getting started

Tested on an Apple Silicon Mac (the `mps` backend). CUDA and CPU also work; CPU training is slow.

1. Clone the repository and install the dependencies.

   ```sh
   git clone https://github.com/uchicago-dsi-bayesianml/wi25-age-transform-vae.git
   cd wi25-age-transform-vae
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

   `dlib` compiles from source and needs CMake and a C++ compiler. If that fails, `pip install dlib-bin` installs a prebuilt wheel.

2. Get the data. See [Data](#data). The notebooks expect it under `Dataset/`.

3. Download dlib's face landmark model into the repository root. It is 99 MB and is not committed.

   ```sh
   curl -LO http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   ```

4. Run the notebooks in this order.

   | Notebook | What it does |
   |---|---|
   | `fix_csv.ipynb` | Strips the Kaggle path prefix from the index files. |
   | `conditional-dfc-model-large-images.ipynb` | Trains the CVAE for 500 epochs and writes one checkpoint per epoch to `checkpoints/`. Expect several gigabytes. |
   | `conditional-dfc-model-viewer-large-images.ipynb` | Loads a checkpoint, reconstructs and ages test images, and runs your own photos through the alignment pipeline. Produces the figures and GIFs. |
   | `figures.ipynb` | Training curves from the saved checkpoints, and the latent interpolation figure. |

   To age your own photos, put them in `Dataset/Team pics/` with a `team.csv` that has the columns `filepath`, `age` and `gender`, as in the index files.

## Data

Data files are never committed to this repository. Git ignores `Dataset/` and all checkpoints.

- **Source:** MORPH-2, about 55,000 longitudinal face images with age and gender labels. Kurt Ricanek and Tesfaye Tessema (2023). MORPH-2 [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/3576052
- **Terms:** MORPH is distributed under an academic license. Check the terms on Kaggle before downloading, and don't redistribute the images.
- **How to get it:** download from Kaggle and unpack into `Dataset/`. The Kaggle release ships index files with `filepath`, `age` and `gender` columns. The notebooks read them as `Dataset/Index/train.csv`, `validation.csv` and `test.csv`, so rename them to lowercase on a case-sensitive filesystem.
- **Where to put it:** `Dataset/Index/` for the CSV files, `Dataset/Images/` for the images, and `Dataset/Team pics/` for your own photos.

The `EDA/` folder also explores CACD and IMDB-WIKI, two other face datasets the team considered before settling on MORPH-2.

## Project structure

| Path | Contents |
|---|---|
| `dataset.py` | `MorphII_Dataset`, a PyTorch dataset that reads an index CSV and returns an image with its `[age/100, gender]` condition. |
| `conditional-dfc-model-large-images.ipynb` | Model definition and training loop. |
| `conditional-dfc-model-viewer-large-images.ipynb` | Evaluation, the face alignment pipeline, aging sweeps, GIFs. |
| `figures.ipynb` | Training curves and latent interpolation. |
| `fix_csv.ipynb`, `morphII_lighting_distribution_analysis.ipynb`, `morphii_train_stats.csv` | Index cleanup, and the brightness and histogram statistics of the training set that the alignment pipeline matches new photos to. |
| `EDA/` | Exploration of MORPH-2, CACD and IMDB-WIKI, and an eye-finder prototype. |
| `old_architectures/` | Earlier attempts: a β-VAE, a perceptual-loss variant, and 64×64 models. See the note inside. |
| `figures/` | All figures in this README, and the script that draws the architecture diagram. |

The `dfc` in the notebook names is left over from an earlier deep-feature-consistent variant with a VGG perceptual loss, kept in `old_architectures/`. The final model uses a pixel-wise reconstruction loss.

## Method

![CVAE architecture](figures/architecture.png)

**Encoder.** Four stride-2 convolutional blocks (16, 32, 64 and 128 channels, each with batch normalization and LeakyReLU) take a 128×128 image down to 128×8×8. The flattened features are concatenated with the condition vector and two linear layers produce the mean and log-variance of a 256-dimensional Gaussian over the latent code.

**Decoder.** The latent code and the condition vector pass through a linear layer back to 128×8×8, then four transposed convolutions mirror the encoder up to 128×128×3.

**Loss.** Mean squared reconstruction error plus the KL divergence between the encoder's Gaussian and a standard normal prior, averaged over the batch. Adam with a learning rate of 0.0008, batch size 128, 500 epochs. Images are normalized to [-1, 1].

**Preprocessing for new photos.** The model only generalizes to photos that look like its training set. For each new photo, the pipeline detects the face and 68 landmarks with dlib, rotates and scales so the eyes sit at fixed positions, masks the face, replaces the background with a flat grey, and adjusts brightness and matches the color histogram inside the mask to the training-set statistics in `morphii_train_stats.csv`.

### What is Bayesian here

The VAE is amortized variational inference. The encoder is the approximate posterior over the latent code, a Gaussian whose mean and variance are functions of the image and the condition. The decoder is the likelihood of the image given the code and the condition, and the prior on the code is a standard normal. The training loss is the negative evidence lower bound: a reconstruction term that is the Gaussian log-likelihood up to a constant, which is where the mean squared error comes from, plus the KL term that keeps the posterior close to the prior. The reparameterization trick makes a one-sample Monte Carlo estimate of this bound differentiable.

Conditioning the encoder and decoder on age and gender is what makes age editing possible. The KL term pushes the code to carry only what the condition does not explain, so at decode time the code holds identity and the condition holds age. Sampling the code several times for the same photo gives different plausible reconstructions. That spread is the posterior uncertainty over the latent representation.

## Results

Training converges within roughly 100 epochs and is flat afterwards. The reconstruction term dominates the loss.

![Training curves](figures/training_curve.png)

Reconstructions of held-out test images and of the team's own photos. They are blurry, which is the usual effect of a pixel-wise loss averaging over plausible details, but the identifying features survive.

![Reconstruction of test images](figures/test_reconstruction.png)

![Reconstruction of the team's photos](figures/team_reconstruction.png)

Interpolating between the latent codes of two team members moves smoothly from one face to the other.

![Latent interpolation](figures/latent_interpolation.png)

Decoding one latent code with a sweep of age values. Age labels on the figures follow the authors' mapping of the condition value to years.

![Aging of test images](figures/test_age.png)

![Aging of the team's photos](figures/team_age.png)

### Age variation GIFs

| Batu | Brad | Casey |
|---|---|---|
| <img src="figures/gifs/age_variation_Batu.gif" width="250"> | <img src="figures/gifs/age_variation_Brad.gif" width="250"> | <img src="figures/gifs/age_variation_Casey.gif" width="250"> |

| John | Kyler |
|---|---|
| <img src="figures/gifs/age_variation_John.gif" width="250"> | <img src="figures/gifs/age_variation_Kyler.gif" width="250"> |

## Challenges

**Backgrounds, framing and lighting.** The most time-consuming part of the project. Varied backgrounds, head positions and exposure levels kept the model from learning a clean representation of age, and new photos taken outside the dataset's conditions reconstructed poorly. The alignment and masking pipeline described under [Method](#method) was the answer.

**Disentanglement.** The team first tried to make a β-VAE discover an age direction in an unsupervised latent space, on IMDB-WIKI and then MORPH-2. The age factor never separated cleanly, so they switched to conditioning on age explicitly. Those attempts are in `old_architectures/`.

**Dataset imbalance.** MORPH-2 has far fewer images of older people and an uneven racial composition. The team noted this as a limitation. The final model does not correct for it, so aging results are weakest for the under-represented groups.

## Conclusion and future directions

An age-conditioned VAE can produce controllable, identity-preserving aging from a single photo, provided the input is brought into the training distribution first. Sharper results would need a perceptual or adversarial reconstruction term, which the team explored but did not finish. Other directions are additional conditioning attributes, class-balanced training to address the imbalance above, and applying the same pipeline to other image domains.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Winter 2025. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository and made these edits. The authors' commits are unchanged.

- Rewrote this README in the course's project layout, keeping the authors' text where it matched the code. Removed statements about weighted losses, data augmentation and an age-conditioned loss, which the code never had. Added the sections on getting started, data, project structure and what is Bayesian here.
- Removed an unrelated notebook added in November 2025, a 99 MB dlib model file that is now downloaded in [Getting started](#getting-started), and HTML fragments of an earlier β-VAE presentation that the final write-up had superseded.
- Removed one cell from `figures.ipynb` that drew a bar chart of "performance metrics" from hardcoded numbers rather than measurements.
- Added `requirements.txt`, a note in `old_architectures/`, and the program banner.

## License

MIT, held by the authors. See [LICENSE](LICENSE).
