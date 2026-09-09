"""Streamlit GUI for the Digital Evidence Generative AI Framework."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ae_inference import AutoencoderInference
from transformer_inference import TransformerInference
from vae_inference import VAEInference


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@st.cache_resource(show_spinner="Loading Autoencoder checkpoint…")
def load_ae() -> AutoencoderInference:
    return AutoencoderInference(ROOT / "checkpoints" / "best_autoencoder.pth", DEVICE)


@st.cache_resource(show_spinner="Loading final VAE V5 checkpoint…")
def load_vae() -> VAEInference:
    return VAEInference(
        ROOT / "checkpoints" / "VAE_V5_FINAL.pth", DEVICE, 256
    )


@st.cache_resource(show_spinner="Loading Transformer V2 checkpoint…")
def load_transformer() -> TransformerInference:
    return TransformerInference(
        ROOT / "checkpoints" / "transformer_v2_final.pth", DEVICE
    )


def safe_load(loader, label: str):
    try:
        return loader(), None
    except FileNotFoundError as exc:
        return None, f"{label} checkpoint is missing. {exc}"
    except Exception as exc:
        return None, f"{label} could not be loaded: {exc}"


def uploaded_image(uploaded) -> Image.Image | None:
    if uploaded is None: return None
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        st.error(f"Unsupported file type: {suffix or 'unknown'}. Upload JPG, JPEG, PNG, BMP, TIF, or TIFF.")
        return None
    try:
        image = Image.open(uploaded)
        image.load()
        return image
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"The uploaded image could not be read: {exc}")
        return None


def metric_cards(metrics: dict[str, float]) -> None:
    columns = st.columns(3)
    columns[0].metric("MSE ↓", f"{metrics['mse']:.6f}")
    columns[1].metric("PSNR ↑", f"{metrics['psnr']:.3f} dB")
    columns[2].metric("SSIM ↑", f"{metrics['ssim']:.4f}")


st.set_page_config(page_title="Digital Evidence Generative AI Framework", page_icon="🔬", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding:1.3rem 1.5rem;border-radius:16px;background:linear-gradient(120deg,#102a43,#176b87);color:white;margin-bottom:1.2rem;}
.hero h1 {margin:0;font-size:2.25rem}.hero p {margin:.35rem 0 0;color:#d9edf2;font-size:1.05rem}
.model-card {border:1px solid #d7e1e8;border-radius:13px;padding:1rem;min-height:125px;background:#f8fbfc;}
.notice {border-left:4px solid #176b87;padding:.65rem .9rem;background:#eff8fa;border-radius:6px;}
</style>
<div class="hero"><h1>Digital Evidence Generative AI Framework</h1>
<p>Autoencoder • Variational Autoencoder • Transformer</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Framework")
    st.caption("Multi-Model Generative AI Framework for Digital Evidence Analysis and Intelligence Generation")
    st.divider()
    st.write("**Runtime device**")
    st.code(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    st.info("Generated images are synthetic research outputs, not genuine forensic evidence.")

overview_tab, ae_tab, vae_tab, transformer_tab, comparison_tab = st.tabs([
    "Project Overview", "Autoencoder", "VAE", "Transformer", "Model Comparison"
])

with overview_tab:
    st.subheader("CASIA v2.0 Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total images", "12,614"); c2.metric("Authentic", "7,491"); c3.metric("Tampered", "5,123")
    st.markdown("This system demonstrates complementary generative AI techniques for digital image evidence analysis, reconstruction, compression, probabilistic representation, and synthetic generation.")
    cards = st.columns(3)
    cards[0].markdown('<div class="model-card"><h3>Autoencoder</h3><p>Image reconstruction and 24× latent compression.</p></div>', unsafe_allow_html=True)
    cards[1].markdown('<div class="model-card"><h3>VAE V5 Final</h3><p>High-quality reconstruction, exploratory anomaly analysis, and probabilistic generation.</p></div>', unsafe_allow_html=True)
    cards[2].markdown('<div class="model-card"><h3>Transformer V2</h3><p>Patch-attention reconstruction and exploratory anomaly analysis.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="notice">Authentic/tampered labels support exploratory comparisons only. Reconstruction error alone does not prove forgery.</div>', unsafe_allow_html=True)

with ae_tab:
    st.subheader("Autoencoder Reconstruction")
    st.caption("Preprocessing: RGB → 128×128 → tensor in [0,1]")
    ae, ae_error = safe_load(load_ae, "Autoencoder")
    if ae_error: st.error(ae_error)
    else:
        info = st.columns(2); info[0].metric("Compression ratio", "24×"); info[1].metric("Parameters", "265,571")
        upload = st.file_uploader("Upload an image for AE reconstruction", type=["jpg","jpeg","png","bmp","tif","tiff"], key="ae_upload")
        image = uploaded_image(upload)
        if image is not None:
            try:
                original, reconstructed, metrics = ae.reconstruct(image)
                left, right = st.columns(2)
                left.image(original, caption="Original image", width="stretch")
                right.image(reconstructed, caption="Reconstructed image", width="stretch", clamp=True)
                metric_cards(metrics)
                with st.expander("How to read these metrics"):
                    st.write("Lower MSE is better. Higher PSNR is better. SSIM closer to 1 is better.")
                    st.warning("These values do not classify an image as authentic or tampered.")
            except Exception as exc: st.error(f"AE reconstruction failed: {exc}")

with vae_tab:
    st.subheader("VAE V5 Final — Reconstruction and Exploratory Analysis")
    st.caption("Final mixed-data reconstruction checkpoint • RGB 128×128 • [0,1]")
    vae, vae_error = safe_load(load_vae, "VAE V5")
    if vae_error: st.error(vae_error)
    else:
        st.metric("Latent dimension", "256")
        upload = st.file_uploader("Upload an image for VAE reconstruction", type=["jpg","jpeg","png","bmp","tif","tiff"], key="vae_upload")
        image = uploaded_image(upload)
        if image is not None:
            try:
                original, reconstructed, metrics = vae.reconstruct(image)
                left, right = st.columns(2)
                left.image(original, caption="Original", width="stretch")
                right.image(reconstructed, caption="VAE reconstruction (z = μ)", width="stretch", clamp=True)
                metric_cards(metrics)
                st.caption("MSE is shown as reconstruction error, not as a tampering probability.")
                st.warning(
                    "This forensic reconstruction indicator is exploratory and is not a "
                    "validated tampering probability or standalone forgery decision."
                )
            except Exception as exc: st.error(f"VAE reconstruction failed: {exc}")
        st.divider()
        if st.button("Generate Synthetic Image", type="primary", key="vae_generate"):
            try:
                st.image(vae.generate(), caption="Synthetic VAE-generated image", width=420, clamp=True)
                st.warning("Synthetic research output — not real forensic evidence.")
                st.caption(
                    "VAE V5 was optimized for skip-connected reconstruction. Prior-only "
                    "samples have no image skip features and may be visually degenerate."
                )
            except Exception as exc: st.error(f"VAE generation failed: {exc}")

with transformer_tab:
    st.subheader("Transformer V2 — Reconstruction and Attention Analysis")
    st.caption("RGB 128×128 • 16×16 patches • tensor in [0,1]")
    transformer, transformer_error = safe_load(load_transformer, "Transformer V2")
    if transformer_error:
        st.error(transformer_error)
    else:
        architecture = st.columns(3)
        architecture[0].metric("Patches", "64 (8×8)")
        architecture[1].metric("Embedding dimension", "256")
        architecture[2].metric("Attention heads", "8")
        layers = st.columns(2)
        layers[0].metric("Encoder layers", "4")
        layers[1].metric("Decoder layers", "2")
        upload = st.file_uploader(
            "Upload an image for Transformer reconstruction",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key="transformer_upload",
        )
        image = uploaded_image(upload)
        if image is not None:
            try:
                original, reconstructed, metrics, latent, attention = transformer.reconstruct(image)
                left, right = st.columns(2)
                left.image(original, caption="Original image", width="stretch")
                right.image(reconstructed, caption="Transformer reconstruction", width="stretch", clamp=True)
                metric_cards(metrics)
                indicators = st.columns(2)
                indicators[0].metric("Reconstruction error", f"{metrics['reconstruction_error']:.6f}")
                indicators[1].metric("Reference threshold", f"{metrics['anomaly_threshold']:.6f}")
                figure, axes = plt.subplots(1, 2, figsize=(9, 4))
                axes[0].imshow(attention, cmap="viridis")
                axes[0].set_title("8×8 patch attention")
                axes[1].imshow(original.resize((128, 128)))
                axes[1].imshow(
                    Image.fromarray((attention * 255).astype("uint8")).resize((128, 128)),
                    cmap="jet", alpha=0.45,
                )
                axes[1].set_title("Attention overlay")
                for axis in axes: axis.axis("off")
                figure.tight_layout()
                st.pyplot(figure)
                plt.close(figure)
                st.warning(
                    "Reconstruction error and attention are exploratory forensic indicators. "
                    "They are not tampering probabilities and do not confirm manipulation."
                )
            except Exception as exc:
                st.error(f"Transformer inference failed: {exc}")

with comparison_tab:
    st.subheader("Model Comparison")
    comparison = pd.DataFrame([
        {"Model":"Autoencoder", "Purpose":"Reconstruction + compression", "MSE":"0.00353242", "PSNR":"25.3077", "SSIM":"0.761558", "FID":"—", "Inception Score":"—"},
        {"Model":"VAE V5 Final", "Purpose":"Reconstruction + exploratory anomaly analysis + generation", "MSE":"0.00070007", "PSNR":"32.9535", "SSIM":"0.961139", "FID":"N/A", "Inception Score":"—"},
        {"Model":"Transformer V2", "Purpose":"Patch-attention reconstruction + anomaly indicator", "MSE":"0.002060*", "PSNR":"N/A", "SSIM":"N/A", "FID":"—", "Inception Score":"—"},
    ])
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.caption("*Transformer MSE is its saved best validation MSE. Its saved reconstruction-error ROC-AUC is 0.5455, indicating limited forensic separation.")
    st.info("These metrics measure different model objectives and should not all be compared directly.")
    with st.expander("Interpretation"):
        st.write("AE, VAE, and Transformer reconstruction metrics compare output images with their inputs. Their reconstruction errors measure reconstruction behavior, not tampering probability.")

st.divider()
st.caption(".")
