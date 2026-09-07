# Multi-Model Generative AI Framework for Digital Evidence Analysis and Intelligence Generation

## 1. Project Overview

This academic project studies how several generative deep-learning approaches behave on digital image evidence. It uses the CASIA v2.0 Image Tampering Detection Dataset to investigate deterministic reconstruction and compression, probabilistic latent representations, image denoising, and synthetic-image generation.

The current review build implements:

1. a convolutional Autoencoder (AE), including a denoising experiment;
2. a convolutional Variational Autoencoder (VAE), with baseline V1 and KL-warm-up V2 experiments; and
3. a Deep Convolutional Generative Adversarial Network (DCGAN).

The repository also contains evaluation scripts, saved experimental results, faculty-demo notebooks, and a Streamlit interface. It does **not** yet implement a complete forensic intelligence-generation system. Transformer and Diffusion components are future work, not current features.

## 2. Current Project Status

| Module | Status | Purpose |
| --- | --- | --- |
| CASIA dataset pipeline | Completed | Inventory, validation, portable stratified splits, and reusable loaders |
| Standard Autoencoder | Completed | Deterministic reconstruction and 24× latent compression |
| Denoising Autoencoder experiment | Completed | Reconstruction of clean images from Gaussian-noise inputs |
| VAE V1/V2 | Historical baselines | Earlier probabilistic reconstruction and sampling experiments |
| VAE V5 forensic | **Current / completed** | Authentic-only reconstruction learning and exploratory anomaly analysis |
| DCGAN | Completed | Adversarial generation of 64×64 synthetic images |
| Quantitative evaluation | Completed | Reconstruction metrics, KL, FID, and Inception Score where applicable |
| Streamlit GUI | Completed | Cached inference and interactive demonstrations for AE, VAE V5, and DCGAN |
| Auxiliary ResNet-18 classifier notebook | Experimental | Separate real-vs-fake classification study; not part of the generative pipeline or GUI |
| Transformer | Planned | Future sequence/context component |
| Diffusion model | Planned | Future image-generation component |
| Integrated intelligence generation | Planned | Future multi-model reasoning and reporting layer |

## 3. Dataset — CASIA v2.0

The project uses the **CASIA v2.0 Image Tampering Detection Dataset**, obtained from Kaggle under <code>divg07/casia-20-image-tampering-detection-dataset</code>. CASIA was selected because it provides a substantial set of authentic and manipulated images, together with tampering ground-truth masks, and is widely suitable for course-level image-forensics experiments.

| Category | Count |
| --- | ---: |
| Authentic images | 7,491 |
| Tampered images | 5,123 |
| Model-input images | **12,614** |
| Ground-truth masks | 5,123 |
| All supported image and mask files | 17,737 |
| Corrupt/unreadable files found | 0 |

The inventory records RGB authentic and tampered images in JPG, TIF, and BMP formats. The most common dimensions are 384×256 and 256×384. Corresponding ground-truth masks are PNG files held separately from normal model input.

The discovered directories are:

- <code>CASIA2/Au</code> — authentic images;
- <code>CASIA2/Tp</code> — tampered images; and
- <code>CASIA2/CASIA 2 Groundtruth</code> — PNG masks.

Masks are excluded by the inventory/split pipeline rather than inferred from an image label. This prevents target-mask leakage into AE, VAE, or GAN inputs. The original dataset is intentionally not committed to GitHub.

## 4. Dataset Preparation

The reusable pipeline is implemented by <code>src/explore_dataset.py</code>, <code>src/prepare_ae_data.py</code>, and <code>src/ae_dataset.py</code>.

For AE and VAE:

- images are opened safely with Pillow and converted to RGB;
- images are resized to 128×128;
- tensors remain in the [0, 1] range;
- ImageNet normalization is not applied; and
- authentic/tampered labels are retained as metadata, not reconstruction targets.

For DCGAN, the same split manifests are reused, but images are resized to 64×64 and normalized to [-1, 1] for the generator's Tanh output.

The dataset is stratified with random seed 42:

| Split | Total | Authentic (0) | Tampered (1) |
| --- | ---: | ---: | ---: |
| Train | 8,830 | 5,244 | 3,586 |
| Validation | 1,892 | 1,124 | 768 |
| Test | 1,892 | 1,123 | 769 |
| **Total** | **12,614** | **7,491** | **5,123** |

The CSV manifests under <code>data/splits/</code> store portable relative paths with <code>image_path</code>, <code>label</code>, and <code>class_name</code>. Repository validation found zero cross-split duplicate paths and zero included mask paths.

## 5. Autoencoder (AE)

### Purpose

The standard AE learns a deterministic mapping from an image to a lower-dimensional representation and back to a reconstruction. Its labels are preserved only for later group-level exploratory analysis.

### Architecture

The encoder uses strided convolution blocks and the decoder uses transposed convolutions:

<pre>
Input 3×128×128
  → convolutional encoder
  → latent 32×8×8
  → transposed-convolution decoder
  → output 3×128×128 with Sigmoid
</pre>

| Property | Value |
| --- | ---: |
| Input values per image | 49,152 |
| Latent values per image | 2,048 |
| Compression ratio | 24× |
| Parameters | 265,571 |

### Final standard-AE training

The current history contains the newer 50-epoch-configured experiment, not only the earlier 20-epoch run. Early stopping ended training after epoch 46, with the best validation checkpoint at epoch 42.

| Setting/result | Value |
| --- | ---: |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | MSE |
| Batch size | 32 |
| Maximum epochs | 50 |
| Epochs completed | 46 |
| Early-stopping patience | 4 |
| First train loss | 0.02793490 |
| Final train loss | 0.00363512 |
| Best validation loss | 0.00355497 |
| Best epoch | 42 |
| Recorded device | Tesla T4 |
| Recorded training time | 1,906.4 s (about 31.8 min) |

The training time and GPU are recorded in the faculty-demo notebook; the checkpoint itself records epoch 42 and the validation loss.

### Standard-AE evaluation

The complete held-out test set of 1,892 images produced:

| Group | MSE ↓ | PSNR ↑ | SSIM ↑ |
| --- | ---: | ---: | ---: |
| Overall | **0.00353242** | **25.3077 dB** | **0.761558** |
| Authentic (1,123) | 0.00342173 | 25.4430 dB | 0.766052 |
| Tampered (769) | 0.00369406 | 25.1100 dB | 0.754995 |

These results supersede the older AE values still hard-coded in the current GUI comparison table.

### Denoising experiment

The denoising experiment starts from the standard AE checkpoint and uses configurable Gaussian noise (recorded standard deviation 0.10). Its test comparison reports noisy-input MSE 0.00904314 and denoised-output MSE 0.00375649, with denoised PSNR 24.9341 dB and SSIM 0.732429. This is an image-restoration experiment, not a forgery-classification result.

### Interpretation

The AE is **not trained or validated as a forgery classifier**. Differences between authentic and tampered reconstruction statistics are exploratory group observations and do not prove that an individual image is forged.

## 6. Variational Autoencoder (VAE)

### Purpose and architecture

The current VAE is **VAE V5 — Forensic Reconstruction and Anomaly Analysis**. It learns reconstruction from authentic training/validation images only, then compares reconstruction behavior on the untouched mixed test split. It is an anomaly-analysis module, not a calibrated tampering classifier.

VAE V5 processes 3×128×128 RGB inputs through four convolutional stages (32, 64, 128, and 256 channels). Residual blocks, GroupNorm, and SiLU activations are used throughout. Separate fully connected heads produce 256-dimensional μ and log-variance vectors. The decoder combines a latent projection with encoder skip features, bilinear upsampling, convolutional residual blocks, and a final Sigmoid output.

| Property | VAE V5 |
| --- | ---: |
| Latent dimension | 256 |
| Parameters | 17,599,971 |
| Reconstruction mode | Deterministic z=μ with encoder skips |
| Training scope | Authentic only |

<pre>
σ = exp(0.5 × logvar)
ε ~ N(0, I)
z = μ + σ × ε
</pre>

### Loss and KL warm-up

The final V5 loss is the exact hybrid objective from the source experiment:

<pre>
Reconstruction = 0.5 × MSE + 0.5 × L1
KL = mean[-0.5 × sum(1 + logvar - μ² - exp(logvar))]
Total = Reconstruction + β × KL
</pre>

β increases linearly from 0.00005 to 0.00030 over the first 20 epochs and remains at 0.00030 afterward. KL regularizes the latent distribution; lower KL does not automatically mean better reconstruction quality.

### Training configuration and checkpoint provenance

The source notebook specifies 80 fixed epochs, AdamW, learning rate 0.0001, weight decay 0.00001, batch size 32, gradient clipping at 1.0, ReduceLROnPlateau, no early stopping, and checkpoint selection by authentic-validation MSE.

The notebook's stored output reports best epoch 18, validation MSE 0.002313, and PSNR 26.36 dB. However, the checkpoint actually supplied to this repository records **epoch 80**, validation MSE **0.001409014**, validation PSNR **28.510845 dB**, and validation KL **0.00201394**. The application uses the checkpoint metadata and documents the notebook figures only as historical provenance.

### Canonical test evaluation

The current checkpoint was evaluated deterministically on the repository's exact 1,892-image test split:

| Group | MSE ↓ | PSNR ↑ | SSIM ↑ | Mean KL |
| --- | ---: | ---: | ---: | ---: |
| Overall (1,892) | **0.00106170** | **31.3124 dB** | **0.950181** | 0.00198055 |
| Authentic (1,123) | 0.00089763 | 31.8360 dB | 0.952742 | 0.00186138 |
| Tampered (769) | 0.00130129 | 30.5479 dB | 0.946442 | 0.00215458 |

Reconstruction-MSE ROC-AUC is **0.604618** and SSIM-error ROC-AUC is **0.593219**. The error distributions overlap substantially. These values show limited exploratory separation and do not make VAE V5 a reliable standalone forgery detector or turn reconstruction error into a tampering probability.

The large notebook did not calculate a valid final V5 FID. The clean evaluation script supports opt-in FID with <code>--run-fid</code>, but the current V5 FID remains **N/A** rather than reusing the old V2 value.

Although the model defines a probabilistic latent and prior-sampling path, the trained decoder depends heavily on encoder skip features. Prior-only samples generated with zero skips are nearly uniform/dark, consistent with the checkpoint's very small KL and weak use of the latent variable. V5's strong reconstruction results must therefore not be interpreted as strong unconditional generation.

### Historical baselines

VAE V1 and V2 checkpoints/results remain in the repository for comparison. V2 previously reported MSE 0.02816802, PSNR 15.9348 dB, SSIM 0.290189, and FID 321.358307. They are no longer the current GUI model.

## 7. DCGAN

### Purpose and architecture

The DCGAN learns adversarial image generation from the CASIA training split. It is unconditional: authentic/tampered labels are not supplied to the generator or discriminator.

The generator accepts normal random noise with shape <code>[batch, 100, 1, 1]</code>:

<pre>
100
  → 512×4×4
  → 256×8×8
  → 128×16×16
  → 64×32×32
  → 3×64×64
</pre>

It uses transposed convolutions, BatchNorm, ReLU, and a final Tanh. The discriminator follows the reverse spatial progression with convolutions, LeakyReLU(0.2), BatchNorm except in its first block, and a final real/fake logit.

| Property | Value |
| --- | ---: |
| Image size | 64×64 RGB |
| Latent dimension | 100 |
| Generator parameters | 3,576,704 |
| Discriminator parameters | 2,765,568 |

Weights use standard DCGAN initialization. Training uses BCEWithLogitsLoss and separate Adam optimizers.

### Training and evaluation

| Setting/result | Value |
| --- | ---: |
| Batch size | 64 |
| Epochs | 30 |
| Learning rate | 0.0002 |
| Adam β1 / β2 | 0.5 / 0.999 |
| Final generator loss | 3.3741 |
| Final discriminator loss | 0.2748 |
| Recorded GPU | Tesla T4 |
| Recorded training time | 677.68 s |
| FID ↓ | 169.8786 |
| Inception Score ↑ | 2.9495 ± 0.1316 |

Generator and discriminator losses represent different adversarial objectives; their magnitudes should not be ranked as though they were the same metric. The filenames use the <code>best_*</code> convention, but the current training code writes the final epoch-30 generator and discriminator states rather than selecting them using a validation metric.

FID is Fréchet Inception Distance, for which lower is generally better. Inception Score rewards confident and diverse ImageNet-class predictions, but both metrics rely on ImageNet-oriented Inception features and therefore have domain limitations for CASIA forensic imagery. The current images show learned color and scene structure but limited detail and realism.

## 8. Quantitative Model Comparison

| Metric | AE | VAE V5 | DCGAN |
| --- | ---: | ---: | ---: |
| MSE ↓ | 0.00353242 | 0.00106170 | N/A |
| PSNR ↑ | 25.3077 dB | 31.3124 dB | N/A |
| SSIM ↑ | 0.761558 | 0.950181 | N/A |
| KL | N/A | 0.00198055 | N/A |
| FID ↓ | N/A | N/A | 169.8786 |
| Inception Score ↑ | N/A | N/A | 2.9495 ± 0.1316 |

These metrics do not define a single overall ranking because the models solve different tasks:

- the AE is best suited among the current models for deterministic reconstruction;
- VAE V5 combines skip-connected reconstruction, a probabilistic latent space, prior sampling, and exploratory anomaly analysis; and
- DCGAN is dedicated to adversarial generation. A current V5 FID is not yet available.

AE reconstruction metrics should not be compared directly with DCGAN FID or Inception Score.

## 9. Streamlit GUI

Run <code>app.py</code> to open five tabs:

1. **Project Overview** — CASIA counts and cards explaining each generative model.
2. **Autoencoder** — accepts JPG, JPEG, PNG, BMP, TIF, and TIFF images; reconstructs at 128×128; displays original/reconstruction, MSE, PSNR, SSIM, compression, and parameter count.
3. **VAE V5** — reconstructs an uploaded image using deterministic <code>z=μ</code> and encoder skips, displays reconstruction metrics with an anomaly-analysis disclaimer, and can sample a synthetic image from <code>N(0,I)</code>.
4. **GAN** — requires no uploaded image; generates a selectable grid of one to eight samples from random 100-dimensional noise and displays stored FID/IS information.
5. **Model Comparison** — summarizes model objectives and metrics, with a warning that different objectives require different metrics.

Models are loaded once with <code>@st.cache_resource</code>, automatically use CUDA when available, switch to evaluation mode, and perform inference without gradients. The wrappers report missing checkpoints, invalid uploads, unsupported formats, and model-load failures through readable Streamlit messages.

The GUI currently loads:

- <code>checkpoints/best_autoencoder.pth</code>;
- <code>checkpoints/best_vae_v5_forensic.pth</code>;
- <code>checkpoints/best_generator.pth</code>; and
- <code>checkpoints/best_discriminator.pth</code>.

All generated images are synthetic research outputs and are **not genuine forensic evidence**.

## 10. Project Directory Structure

The important current files are organized as follows:

<pre>
Digital_Evidence/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   └── splits/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── src/
│   ├── explore_dataset.py
│   ├── prepare_ae_data.py
│   ├── ae_dataset.py
│   ├── autoencoder.py
│   ├── train_autoencoder.py
│   ├── evaluate_autoencoder.py
│   ├── denoising_dataset.py
│   ├── train_denoising_autoencoder.py
│   ├── evaluate_denoising_autoencoder.py
│   ├── vae.py
│   ├── vae_legacy.py
│   ├── train_vae.py
│   ├── train_vae_v2.py
│   ├── evaluate_vae.py
│   ├── evaluate_vae_v2.py
│   ├── dcgan.py
│   ├── gan_dataset.py
│   ├── train_dcgan.py
│   ├── evaluate_dcgan.py
│   ├── ae_inference.py
│   ├── vae_inference.py
│   └── gan_inference.py
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_autoencoder_training_colab.ipynb
│   ├── 03_vae_training_colab.ipynb
│   ├── 04_vae_v2_training_colab.ipynb
│   ├── Autoencoder_CASIA_Complete_Demo.ipynb
│   ├── VAE_CASIA_Complete_Demo.ipynb
│   ├── GAN_CASIA_Complete_Demo.ipynb
│   └── AI_GENERATED_IMAGE_EVIDENCE_ANALYSIS.ipynb
├── checkpoints/
│   ├── best_autoencoder.pth
│   ├── best_denoising_autoencoder.pth
│   ├── best_vae.pth
│   ├── best_vae_v2.pth
│   ├── best_vae_v5_forensic.pth
│   ├── best_generator.pth
│   └── best_discriminator.pth
├── results/
│   ├── dataset_summary.json
│   ├── ae_test_metrics.json
│   ├── dae_test_metrics.json
│   ├── vae_test_metrics.json
│   ├── vae_v2_test_metrics.json
│   ├── vae_v5_test_metrics.json
│   ├── gan_test_metrics.json
│   └── training histories and per-image CSV files
└── outputs/
    ├── ae/
    ├── vae/
    ├── vae_v5/
    └── gan/
</pre>

Raw CASIA data and generated output images are ignored by the current Git configuration. They may exist locally but are not guaranteed to be present in a fresh clone.

## 11. Important Files

| File | Role |
| --- | --- |
| <code>src/explore_dataset.py</code> | Discovers the real CASIA structure, validates images, writes inventory/summary results, and creates sample plots |
| <code>src/prepare_ae_data.py</code> | Builds and validates the fixed stratified split manifests |
| <code>src/ae_dataset.py</code> | Portable manifest-backed PyTorch dataset and data-loader factory |
| <code>src/autoencoder.py</code> | Convolutional AE architecture |
| <code>src/train_autoencoder.py</code> | Standard-AE training, validation, early stopping, checkpoints, and plots |
| <code>src/evaluate_autoencoder.py</code> | Full-test AE metrics, per-image results, reconstructions, and group comparisons |
| <code>src/denoising_dataset.py</code> | Configurable Gaussian-noise input wrapper |
| <code>src/vae.py</code> | Final residual/skip-connected VAE V5, exact loss, and beta schedule |
| <code>src/vae_legacy.py</code> | Historical V1/V2 architecture compatibility |
| <code>src/train_vae.py</code> | Authentic-only VAE V5 training and validation-MSE checkpointing |
| <code>src/train_vae_v2.py</code> | Historical KL-warm-up VAE V2 training |
| <code>src/evaluate_vae.py</code> | VAE V5 reconstruction, group metrics, ROC-AUC, plots, generation, and optional FID |
| <code>src/evaluate_vae_v2.py</code> | V2 evaluation and V1/V2 comparison |
| <code>src/dcgan.py</code> | Generator, discriminator, and DCGAN weight initialization |
| <code>src/gan_dataset.py</code> | 64×64, [-1,1] GAN data pipeline |
| <code>src/train_dcgan.py</code> | Adversarial training, checkpoints, loss history, and fixed-noise grids |
| <code>src/evaluate_dcgan.py</code> | FID and Inception Score evaluation |
| <code>src/*_inference.py</code> | Cached-GUI-friendly model loading, preprocessing, metrics, and generation |
| <code>app.py</code> | Unified Streamlit demonstration interface |

Architecture and data-pipeline validation scripts also remain under <code>src/</code> for smoke testing.

## 12. Installation

Python dependencies are declared in <code>requirements.txt</code>.

~~~bash
python -m venv .venv
~~~

Windows PowerShell:

~~~powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
~~~

Linux/macOS:

~~~bash
source .venv/bin/activate
pip install -r requirements.txt
~~~

TorchMetrics and Torch-Fidelity are included for the optional VAE/DCGAN FID and Inception Score evaluations.

## 13. Running the GUI

From the repository root:

~~~bash
streamlit run app.py
~~~

The application automatically selects CUDA when available and otherwise uses CPU. Place the required checkpoints in <code>checkpoints/</code>. The dataset itself is not required for single-image AE/VAE inference or unconditional GAN generation.

## 14. Training Models

The scripts have repository-relative defaults. Run them from the repository root after preparing CASIA and the split CSVs. GPU/Colab is recommended for full training.

### Dataset inventory and splits

~~~bash
python src/explore_dataset.py --dataset-root data/raw
python src/prepare_ae_data.py
~~~

### Autoencoder

~~~bash
python src/train_autoencoder.py --max-epochs 50 --patience 4
python src/evaluate_autoencoder.py
~~~

Use <code>--smoke-test</code> before a full AE run. The separate denoising workflow is:

~~~bash
python src/train_denoising_autoencoder.py
python src/evaluate_denoising_autoencoder.py
~~~

### VAE

~~~bash
python src/train_vae.py
python src/evaluate_vae.py
~~~

The current training script defaults to authentic-only VAE V5: 80 epochs, latent dimension 256, AdamW, hybrid MSE/L1 reconstruction, and a 20-epoch beta warm-up. Use <code>python src/evaluate_vae.py --run-fid</code> only when TorchMetrics/Torch-Fidelity and a suitable GPU are available. Historical V1/V2 scripts, architecture compatibility, and checkpoints remain for reference but are not used by the GUI.

### DCGAN

~~~bash
python src/train_dcgan.py
python src/evaluate_dcgan.py
~~~

The complete demo notebooks default to loading existing artifacts where their demo flags are disabled. Existing checkpoints can be used by the GUI without retraining.

## 15. Evaluation Metrics

| Metric | Meaning | Direction/usage |
| --- | --- | --- |
| MSE | Mean squared pixel reconstruction error | Lower is better; AE/VAE reconstruction |
| PSNR | Peak signal-to-noise ratio derived from reconstruction error | Higher is better; AE/VAE reconstruction |
| SSIM | Structural similarity between original and reconstruction | Closer to 1 is better; AE/VAE reconstruction |
| KL | Kullback–Leibler regularization of the VAE posterior | Balance term, not a standalone image-quality ranking |
| FID | Fréchet distance between Inception-feature distributions | Lower is generally better; VAE/DCGAN generation |
| IS | Inception-based confidence/diversity score | Higher is generally better; DCGAN generation |

Not every metric applies to every model. Reconstruction metrics and distribution-level generation metrics answer different questions.

## 16. Experimental Results

### Autoencoder

- Best checkpoint epoch: 42
- Best validation MSE: 0.00355497
- Final completed-epoch train MSE: 0.00363512
- Test MSE / PSNR / SSIM: 0.00353242 / 25.3077 dB / 0.761558
- Recorded run: 46 epochs completed on Tesla T4 in 1,906.4 seconds

### VAE V5 forensic

- Current checkpoint metadata: epoch 80, validation MSE 0.001409014, PSNR 28.510845 dB, KL 0.00201394
- Source-notebook stored run: best epoch 18, validation MSE 0.002313, PSNR 26.36 dB
- Canonical test MSE / PSNR / SSIM: 0.00106170 / 31.3124 dB / 0.950181
- Test KL / hybrid total loss: 0.00198055 / 0.01129045
- Reconstruction-MSE / SSIM-error ROC-AUC: 0.604618 / 0.593219
- FID: N/A; the source notebook did not contain a valid final V5 calculation

### DCGAN

- Final epoch: 30
- Final generator/discriminator loss: 3.3741 / 0.2748
- FID: 169.8786
- Inception Score: 2.9495 ± 0.1316
- Recorded run: Tesla T4, 677.68 seconds

The JSON and CSV files under <code>results/</code> retain more precision, full histories, and per-image reconstruction measurements.

## 17. Visual Results

Selected locally generated artifacts are shown below.

### Autoencoder reconstruction

![Autoencoder original and reconstructed test images](outputs/ae/test_reconstruction_grid.png)

### VAE V5 reconstruction

![VAE V5 authentic and tampered reconstruction examples](outputs/vae_v5/reconstruction_grid.png)

### DCGAN final generated samples

![DCGAN final synthetic samples](outputs/gan/final_generated_samples.png)

Additional available plots include:

- AE training, denoising, and authentic-versus-tampered comparisons under <code>outputs/ae/</code>;
- current VAE V5 reconstruction, generated samples, error distributions, and ROC curves under <code>outputs/vae_v5/</code>, with historical VAE artifacts under <code>outputs/vae/</code>; and
- DCGAN loss curves and fixed-noise samples from epochs 5, 10, 15, 20, 25, and 30 under <code>outputs/gan/</code>.

Because <code>outputs/</code> is currently ignored by Git, these embedded images render only where the files are present (or if selected artifacts are later force-added to version control).

## 18. Known Issues and Limitations

### Technical limitations

- CASIA raw data and generated outputs are ignored by Git, so a fresh clone requires dataset setup and may not display the embedded README images.
- <code>notebooks/01_dataset_exploration.ipynb</code> currently contains no notebook cells; the working exploration implementation and records are the Python script, inventory CSV, and summary JSON.
- The GUI comparison table contains older AE values (MSE 0.00413435, PSNR 24.5603, SSIM 0.725357), while the latest evaluation files report the improved values documented here.
- <code>notebooks/02_autoencoder_training_colab.ipynb</code> retains the earlier 20-epoch experiment; the current final 50-epoch-configured run is documented in the complete AE demo notebook and latest history/checkpoint.
- The V5 source notebook used a 1,893-image test set and its stored best-run metadata differs from the supplied checkpoint; the clean project evaluation instead uses the canonical 1,892-image split and reports both provenance records.
- GAN checkpoints created in Colab contain platform-specific path metadata. <code>src/gan_inference.py</code> includes a compatibility mapping for local GUI loading, while direct local use of <code>src/evaluate_dcgan.py</code> may require a matching Python environment or equivalent compatibility handling.

### Model limitations

- VAE V5 reconstructs strongly through encoder skip connections, but its MSE ROC-AUC of 0.604618 still provides only limited authentic/tampered separation.
- The V5 decoder was trained with image skip features. Prior generation has no source-image skips, uses zero-valued skips, and currently produces visually degenerate dark samples; reconstruction quality and generation quality must not be conflated.
- A valid final V5 FID is not currently available.
- DCGAN samples capture coarse visual structure but have limited sharpness and realism. Final loss behavior is consistent with a comparatively strong discriminator, although generator and discriminator losses cannot be compared directly.
- The AE, VAE, and DCGAN are not trained as validated tampering detectors.
- CASIA is the only dataset used by the generative modules, limiting evidence of cross-dataset generalization.
- The current implementation does not yet provide Transformer, Diffusion, or integrated forensic-intelligence generation.

### Evaluation limitations

- AE/VAE authentic-versus-tampered reconstruction differences are exploratory and do not establish per-image classification capability.
- FID and Inception Score do not demonstrate forensic usefulness, authenticity, or evidential validity.
- ImageNet-based Inception features may not align well with CASIA's forensic domain.
- All results describe this dataset, split, implementation, and checkpoint configuration; external validity has not been established.
- This is an academic prototype, not a production or legally validated forensic system.

## 19. Problems Encountered and Solutions

Repository artifacts support the following development history:

| Problem/context | Implemented response |
| --- | --- |
| Ground-truth PNG masks could be confused with model inputs | Inventory-based class discovery, explicit exclusion, and zero-leakage assertions |
| Local Windows paths did not transfer cleanly to Colab | Split manifests use relative paths and reconstruct paths from a configurable dataset root |
| CPU full training was impractical | Smoke tests were run locally and full experiments were executed on a Tesla T4 in Colab |
| Baseline VAE reconstruction was weak | VAE V5 added residual blocks, GroupNorm/SiLU, skip-connected decoding, a hybrid MSE/L1 loss, and a 20-epoch KL warm-up |
| Multiple VAE experiments and conflicting notebook outputs | V1/V2 artifacts are retained as historical baselines; V5 uses separate paths and runtime evaluation trusts the supplied checkpoint plus canonical manifests |
| GUI checkpoint loading across Colab/local Python path classes | GAN inference adds a restricted compatibility mapping during checkpoint loading |
| Three model families require different interactions | The Streamlit interface uses dedicated inference wrappers and task-specific tabs |

The auxiliary classifier notebook also contains DataLoader multiprocessing cleanup assertions in stored output. It is separate from the generative modules and has no corresponding classifier checkpoint in this repository.

## 20. Model Checkpoints

| Checkpoint | Purpose | Current GUI use |
| --- | --- | --- |
| <code>checkpoints/best_autoencoder.pth</code> | Standard AE, best validation checkpoint at epoch 42 | Yes |
| <code>checkpoints/best_denoising_autoencoder.pth</code> | Gaussian-noise denoising AE experiment | No |
| <code>checkpoints/best_vae.pth</code> | Fixed-β VAE V1 baseline | No |
| <code>checkpoints/best_vae_v2.pth</code> | Historical KL-warm-up VAE V2, best epoch 43 | No |
| <code>checkpoints/best_vae_v5_forensic.pth</code> | Current authentic-only forensic reconstruction VAE V5 | **Yes** |
| <code>checkpoints/best_generator.pth</code> | Final trained DCGAN generator state | Yes |
| <code>checkpoints/best_discriminator.pth</code> | Final trained DCGAN discriminator state | Loaded for model validation/information |

The repository also contains a duplicate denoising checkpoint under <code>results/</code>; the canonical inference path is the checkpoint-directory file shown above.

## 21. Current Review Deliverables

Ready for faculty review:

- verified CASIA inventory and fixed train/validation/test manifests;
- modular PyTorch data pipelines;
- trained standard and denoising AEs;
- historical VAE V1/V2 baselines and the current authentic-only VAE V5;
- trained DCGAN generator and discriminator;
- full-test reconstruction results and per-image CSVs;
- VAE/DCGAN distribution-level generation metrics;
- saved model checkpoints, histories, plots, and sample grids;
- complete AE, VAE, and GAN faculty-demo notebooks; and
- a unified Streamlit inference and comparison dashboard.

The auxiliary ResNet-18 notebook is preserved as a separate exploratory classifier and is not presented as a completed generative-model deliverable.

## 22. Future Work

The implementation roadmap is explicitly separated from current functionality:

1. **Current phase:** AE, denoising AE, VAE, DCGAN, evaluations, notebooks, and GUI.
2. **Future phase:** Transformer-based component for contextual or sequence-level processing.
3. **Future phase:** Diffusion-based image-generation experiment.
4. **Future phase:** integrated multi-model digital-evidence analysis and richer intelligence generation.

Potential model improvements include stronger VAE encoders/decoders, carefully validated perceptual or hybrid reconstruction losses, improved GAN stabilization, higher output resolution, cross-dataset validation, domain-aware alternatives to generic Inception metrics, forensic feature analysis, and calibrated comparisons with future Transformer/Diffusion modules.

## 23. Responsible Use and Disclaimer

This repository is an academic research prototype. Generated VAE and DCGAN images are synthetic outputs, not genuine forensic evidence. Reconstruction errors, latent variables, FID, Inception Score, and visual examples must not be used alone to decide whether evidence is authentic, tampered, admissible, or attributable to a person. The system has not been validated for production investigations or legal decision-making.

## 24. Technologies Used

- Python
- PyTorch and Torchvision
- NumPy and Pandas
- Pillow
- scikit-image and scikit-learn
- Matplotlib
- Streamlit
- Jupyter Notebook / Google Colab
- KaggleHub
- TorchMetrics and Torch-Fidelity for generation evaluation

## 25. Team and Academic Context

This is a Semester 7 Generative AI course project titled **“Multi-Model Generative AI Framework for Digital Evidence Analysis and Intelligence Generation.”** The repository does not currently contain verified team-member names, institutional details, or role assignments, so none are attributed here. Those details can be added when confirmed.
