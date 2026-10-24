# Multi-Model Generative AI Framework for Digital Evidence Analysis and Intelligence Generation

Semester 7 Generative AI project prepared for Review 2. The repository combines four trained image architectures with a separate pretrained text Transformer integration and a pretrained Diffusion scenario simulation:

1. Convolutional Autoencoder — deterministic reconstruction and compression.
2. Variational Autoencoder V5 — probabilistic latent representation, deterministic reconstruction, and image-conditioned stochastic reconstructions.
3. DCGAN Generator — trained 64×64 CASIA synthetic image patch synthesis and latent-space manifold interpolation.
4. Vision Transformer V2 — patch-token self-attention reconstruction and exploratory attention visualization.
5. Evidence Intelligence Transformer — no-key pretrained FLAN-T5 integration for pasted text and digitally extractable PDFs.
6. Synthetic Evidence Generation (Diffusion) — pretrained Stable Diffusion 1.5 integration for controlled forensic incident scenario simulation.

All six model families are fully integrated and demonstrable through the professional Streamlit GUI.

## Responsible-use boundary

This is an academic research prototype, not a forensic decision system. Reconstruction error is not a tampering probability. Attention is not proof or definitive localization of manipulation. Generated language may contain errors. No model output should be the sole basis for authenticity, guilt, admissibility, attribution, or another legal/forensic conclusion.

The app implements privacy-oriented controls and GDPR-oriented design principles; it does not claim formal GDPR compliance.

## Verified repository state

### Dataset and canonical split

CASIA v2.0 model inputs contain 12,614 RGB images. Ground-truth mask PNGs are excluded from model inputs.

| Split | Total | Authentic | Tampered |
| --- | ---: | ---: | ---: |
| Train | 8,830 | 5,244 | 3,586 |
| Validation | 1,892 | 1,124 | 768 |
| Test | 1,892 | 1,123 | 769 |

The manifests in `data/splits/` use repository-relative paths and seed 42. Raw CASIA data is intentionally ignored by Git.

### Same-split image reconstruction comparison

Every row below comes from the same canonical 1,892-image test manifest. Values are generated from verified JSON files by `src/build_model_comparison.py`.

| Model | MSE ↓ | PSNR ↑ | SSIM ↑ | Parameters | Epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quality Autoencoder V1 | 0.00094603 | 31.1951 dB | 0.938654 | 2,371,715 | 79 |
| VAE V5 Final | 0.00070007 | 32.9535 dB | 0.961139 | 17,599,971 | 79 |
| Vision Transformer V2 | 0.00220733 | 27.2652 dB | 0.844864 | 3,591,168 | 50 |

Authoritative artifacts:

- `results/image_model_comparison.json`
- `results/image_model_comparison.csv`
- `results/quality_ae_v1_test_metrics.json`
- `results/vae_v5_final_test_metrics.json`
- `results/transformer_v2_test_metrics.json`
- `results/transformer_v2_test_per_image_metrics.csv`

MSE, PSNR, and SSIM measure reconstruction fidelity only. The VAE's exploratory MSE ROC-AUC is 0.5154 and the newly measured Vision Transformer MSE ROC-AUC is 0.5413, both indicating weak reconstruction-error ranking rather than validated forgery detection.

## Model details

### Quality Autoencoder V1 Final

- Input: `3 × 128 × 128`
- Latent tensor: `32 × 16 × 16` (8,192 values)
- Compression ratio: `6×`
- Parameters: 2,371,715
- Active checkpoint: `checkpoints/best_quality_autoencoder_v1.pth`
- Best checkpoint epoch: 79, selected by validation SSIM
- Validation MSE / SSIM: 0.00095790 / 0.942674
- Loss: `0.65 × L1 + 0.25 × (1 - SSIM) + 0.10 × edge loss`
- Role: deterministic reconstruction and compression, not classification

The GUI labels outputs as `Original Image` and `Reconstructed Image` and does not issue an authenticity verdict.

![AE canonical reconstruction examples](docs/results/ae_reconstruction_examples.png)

The model uses residual blocks and resize-convolution upsampling with no
encoder-to-decoder skip connections. Compared with the preserved 24× RTX-80 model,
it deliberately trades compression for substantially stronger measured and visual
reconstruction fidelity. The RTX-80 checkpoint remains available as a historical
baseline.

![Quality AE training curves](docs/results/quality_ae_training_curve.png)

![Quality AE canonical test reconstructions](docs/results/quality_ae_test_reconstructions.png)

### VAE V5 Final

- Input: `3 × 128 × 128`
- Latent dimension: 256, with learned `mu` and `logvar`
- Parameters: 17,599,971
- Active checkpoint: `checkpoints/VAE_V5_FINAL.pth`
- Checkpoint epoch: 79
- Reconstruction loss: `0.5 × MSE + 0.5 × L1`
- Total loss: reconstruction loss plus beta-weighted KL
- KL beta warm-up: 0.00005 to 0.00030 over 20 epochs

The Review-2 inference path encodes the uploaded image once and calculates:

```text
std = exp(0.5 × logvar)
z = mu + temperature × std × epsilon,  epsilon ~ N(0, I)
```

Every stochastic decode reuses the same image-conditioned skip tensors. Samples are seed-reproducible and are presented as probabilistic latent reconstructions—not new forensic facts. The GUI always presents the original image, deterministic reconstruction, and three posterior samples. Per-sample absolute-difference heatmaps use their own disclosed measured color scales to make subtle changes visible without modifying the reconstructed images. A five-image CPU smoke measurement at temperature 1.0 produced a mean absolute pixel delta of 0.001197 from the deterministic reconstruction (range 0.000259–0.003526 across 15 samples). This confirms real stochastic behavior while documenting that strong skip connections often make the visual differences subtle.

Prior-only generation is retained as a separate limitation demonstration; it can be dark/weak because the trained decoder relies heavily on encoder skips.

![VAE image-conditioned stochastic reconstructions](docs/results/vae_stochastic_variations.png)

### DCGAN Generator (Synthetic Image Synthesis)

- Architecture: Deep Convolutional GAN (Radford et al. 2015)
- Latent noise dimension: 100 ($z \sim \mathcal{N}(0, \mathbf{I})$)
- Native output resolution: $64 \times 64$ RGB
- Generator parameters: 3,576,704 (5 transposed convolutional layers with BatchNorm2d, ReLU, and Tanh)
- Discriminator parameters: 2,765,568 (5 convolutional layers with LeakyReLU 0.2 and scalar logit output)
- Active checkpoints: `checkpoints/best_generator.pth` (14.3 MB) and `checkpoints/best_discriminator.pth` (11.1 MB)
- Training: 30 epochs on CASIA (NVIDIA Tesla T4, 677.68 seconds)
- Adversarial loss: Binary cross-entropy with logits (`BCEWithLogitsLoss`)
- Final training loss: Generator 3.3741 · Discriminator 0.2748
- Verified test evaluation (1,892 real vs. 1,892 synthetic CASIA images):
  - Fréchet Inception Distance (FID): 169.88
  - Inception Score (IS): 2.95 ± 0.13
- Core capabilities:
  - Multi-sample synthetic generation (1, 4, 9, or 16 images) with seed reproducibility;
  - Spherical linear interpolation (SLERP) across the continuous 100-D latent manifold;
  - Disclosed native resolution ($64 \times 64$) with 4× nearest-neighbor upscaling ($256 \times 256$) for pixel-accurate inspection;
  - Embedded educational provenance watermark banner and audit logging (`outputs/gan/gan_generation_log.csv`);
  - Historical training evolution display across epochs 5, 10, 15, 20, 25, and 30.
- Boundary: Synthetic image generation only. Does not classify real vs. tampered images, does not detect forgery, and does not produce authentic digital evidence.

Authoritative artifacts:
- `results/gan_training_summary.json`
- `results/gan_training_history.csv`
- `results/gan_architecture_summary.json`
- `results/gan_test_metrics.json`
- `outputs/gan/gan_loss_curve.png`
- `outputs/gan/final_generated_samples.png`

### Vision Transformer V2

- Input: `3 × 128 × 128`
- Patches: 64 non-overlapping `16 × 16` tokens
- Embedding dimension: 256
- Attention heads: 8
- Encoder/decoder layers: 4/2
- Feed-forward dimension: 512
- Parameters: 3,591,168
- Active checkpoint: `checkpoints/transformer_v2_final.pth`
- Checkpoint epoch: 50

`src/evaluate_transformer_v2.py` performed the full canonical evaluation. The checkpoint's 0.00206004 value remains correctly identified as validation MSE; it is not used as a test metric.

![Vision Transformer reconstruction and attention](docs/results/vision_transformer_example.png)

Attention shows learned patch relationships and must not be interpreted as definitive manipulation localization.

### Evidence Intelligence Transformer

The text module is separate from the trained Vision Transformer. Its flow is:

```text
TXT / digitally extractable PDF
  → validated in-memory extraction
  → token-aware chunks
  → tokenizer
  → pretrained google/flan-t5-small
  → evidence-oriented generated output
```

Features include:

- pasted UTF-8 text, `.txt`, and digitally extractable `.pdf` inputs;
- clear scanned/image-only PDF error when no digital text is found;
- a structured default analyst prompt covering summary, entities, observations, relevance, and limitations;
- a disclosed hybrid presentation: Transformer-generated summary plus conservative source-text indicators, reducing pressure on the compact model to invent entities;
- token-aware chunking with processed/truncated counts;
- configurable model through `EVIDENCE_MODEL_ID`;
- deterministic beam generation on CPU or CUDA;
- no API key for the core path; and
- a mandatory limitation/responsible-use note in every generated result.

The default model is deliberately small for CPU/Streamlit portability, so output may be terse or extractive. It is a pretrained integration, not a language model trained from scratch by the team. A real local demo output and its measured runtime metadata are stored in `docs/results/evidence_transformer_demo.md`.

### Synthetic Evidence Generation (Diffusion)

- Base Model: Latent Diffusion Model — Stable Diffusion 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`)
- Provenance: Pretrained foundation model (RunwayML / Stability AI); not custom-trained or fine-tuned by the team
- Parameters: 1,066,235,307 (~1.07B parameters across CLIP text encoder, U-Net, and VAE)
- Latent Compression: $8\times$ spatial compression ($512 \times 512 \times 3 \to 64 \times 64 \times 4$) via `AutoencoderKL`
- Reverse Denoising: Iterative U-Net latent noise prediction with Classifier-Free Guidance (CFG scale 7.5)
- Output Resolution: $512 \times 512$ RGB
- Checkpoint: `checkpoints/Diffusion_UNet.pth` contains the exported pretrained U-Net denoiser (859.5M FP16 parameters; 1.60 GB; SHA-256: `29ac918c...`)
- Features:
  - Incident scenario presets (traffic collisions, office intrusions, roadside accidents, warehouse breaches);
  - Reverse diffusion latent capture and progressive VAE decoding visualization (steps 1, 5, 10, 15, 20, 25);
  - Permanent `SYNTHETIC DIGITAL EVIDENCE · NOT GENUINE` digital provenance watermark banner;
  - Generation audit logging (`outputs/diffusion/generation_log.csv`);
  - Verified precomputed canonical gallery from audited Tesla T4 benchmark runs (10 steps: 2.73s, 20 steps: 5.01s, 30 steps: 8.13s).
- Boundary: Synthetic scenario simulation only. Does not classify images as authentic or tampered and does not issue forensic verdicts. Full technical audit: `DIFFUSION_IMPLEMENTATION_AUDIT.md`.

## Streamlit application

The professional GUI has ten pages:

1. Project Overview
2. Autoencoder
3. Variational Autoencoder
4. GAN — Synthetic Image Generation
5. Vision Transformer
6. Evidence Intelligence Transformer
7. Synthetic Evidence Generation (Diffusion)
8. Model Evaluation
9. Privacy & Ethical AI
10. Project Information

The three image pages support normal uploads and a browsable, reproducible 24-image canonical test gallery (12 authentic and 12 tampered) when local CASIA data is available. If raw data is absent, the app explains why the gallery is unavailable and retains manual upload. Model loading is cached, CPU fallback is automatic, and missing files/LFS pointers produce actionable errors.

Privacy-oriented controls include:

- authorization/consent before evidence input;
- allowlisted extensions and 10 MB limits;
- Pillow/PDF content validation;
- in-memory/session processing with no automatic permanent evidence save by project code;
- explicit session evidence cleanup;
- visible model, token, truncation, and limitation information; and
- deployment guidance for HTTPS, authentication, access control, logs, region, retention, and governance.

Run from the repository root:

```powershell
streamlit run app.py
```

## Installation

Python 3.10+ is supported.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first evidence-language-model run downloads public FLAN-T5 weights from Hugging Face unless the model is already cached. No API key is required. Set `EVIDENCE_MODEL_ID` to another local or compatible Hugging Face seq2seq model if needed.

## Checkpoints and Git LFS

`checkpoints/VAE_V5_FINAL.pth` and `checkpoints/best_quality_autoencoder_v1.pth`
are managed by Git LFS. GitHub source ZIP files may contain only small text pointers
instead of model bytes. Clone/fetch with LFS:

```powershell
git lfs install
git lfs pull --include="checkpoints/VAE_V5_FINAL.pth,checkpoints/best_quality_autoencoder_v1.pth"
```

The active inference wrappers inspect checkpoint headers before calling `torch.load`. If a pointer is found, the error reports the expected size and exact LFS fetch command.

Checkpoint locations can be overridden without editing code:

| Environment variable | Default |
| --- | --- |
| `AE_CHECKPOINT_PATH` | `checkpoints/best_quality_autoencoder_v1.pth` |
| `VAE_CHECKPOINT_PATH` | `checkpoints/VAE_V5_FINAL.pth` |
| `VISION_TRANSFORMER_CHECKPOINT_PATH` | `checkpoints/transformer_v2_final.pth` |

Current local audit found real bytes for all active checkpoints. Historical AE, denoising, GAN, and VAE assets are preserved; no working checkpoint was overwritten or deleted.

## Reproducible evaluation and artifacts

From the repository root:

```powershell
python src/evaluate_autoencoder.py
python src/evaluate_vae.py
python src/evaluate_transformer_v2.py
python src/build_model_comparison.py
python src/generate_review2_artifacts.py --include-text
```

The full evaluators require the local CASIA data referenced by `data/splits/`. Selected presentation assets are committed under `docs/results/`; bulk outputs remain ignored.

![Canonical image-model comparison](docs/results/image_model_comparison.png)

## Validation

Fast local validation:

```powershell
python -m compileall -q app.py src tests
python -m unittest discover -s tests -v
python src/smoke_review2.py
```

All repository tests use Python's built-in `unittest` discovery; no separate pytest command is required.

Review-2 validation performed for this build includes:

- all source syntax compilation;
- parser/chunking/checkpoint unit tests;
- real AE, VAE, Vision Transformer, and FLAN-T5 CPU inference;
- exact canonical manifest/path/count validation;
- full 1,892-image Vision Transformer evaluation; and
- Streamlit smoke tests for every page plus real canonical-demo inference paths.

## Repository layout

```text
app.py                         Unified Streamlit interface
src/
  quality_autoencoder.py      Active Quality Autoencoder V1 architecture
  autoencoder.py              Preserved historical 24× convolutional AE architecture
  vae.py                      VAE V5 architecture, loss, beta schedule
  transformer.py              Vision Transformer V2 architecture
  *_inference.py              Cached-GUI-friendly image inference
  document_parser.py          Safe TXT/PDF extraction
  evidence_transformer.py     Pretrained language Transformer wrapper
  smoke_review2.py            Real all-model and TXT/PDF smoke inference
  evaluate_transformer_v2.py Canonical Vision Transformer evaluation
  build_model_comparison.py   Verified same-split comparison builder
  generate_review2_artifacts.py Reproducible selected artifacts
tests/                         Focused unit/smoke tests
data/splits/                   Fixed portable manifests
checkpoints/                   Active and preserved trained weights
results/                       Verified metrics and per-image records
docs/results/                  Small presentation-ready result set
notebooks/                     Preserved training/demo notebooks
```

## Preserved and future scope

- GAN code, notebook, checkpoints, training history, and evaluation results are preserved for the later 60-mark phase. GAN is not in the Review-2 GUI.
- Historical VAE and denoising experiments remain available but are not presented as the current final models.
- Diffusion remains future work and is not claimed as implemented.
- Cross-dataset validation, calibrated tampering classifiers, OCR for scanned PDFs, and production governance remain outside the verified Review-2 scope.
