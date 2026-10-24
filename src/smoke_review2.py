"""Run real CPU smoke inference for every active Review-2 model and document flow."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from ae_inference import AutoencoderInference
from document_parser import parse_evidence_file
from evidence_transformer import DEFAULT_PROMPT, RESPONSIBLE_USE_NOTE, EvidenceTransformer
from transformer_inference import TransformerInference
from vae_inference import VAEInference

ROOT = Path(__file__).resolve().parents[1]


def _sample_image() -> tuple[Image.Image, str]:
    frame = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    for relative_path in frame["image_path"]:
        path = ROOT / str(relative_path)
        if path.is_file():
            with Image.open(path) as opened:
                return opened.convert("RGB").copy(), path.relative_to(ROOT).as_posix()
    raise FileNotFoundError(
        "No canonical test image is available locally. Restore CASIA under data/raw or use GUI upload."
    )


def _text_pdf(text: str) -> bytes:
    """Create a minimal text PDF for a dependency-independent extraction smoke test."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def run() -> dict[str, object]:
    device = torch.device("cpu")
    image, source = _sample_image()
    report: dict[str, object] = {"device": str(device), "image": source}

    ae = AutoencoderInference(ROOT / "checkpoints" / "best_quality_autoencoder_v1.pth", device)
    _, ae_output, ae_metrics = ae.reconstruct(image)
    report["autoencoder"] = {"shape": list(ae_output.shape), **ae_metrics}
    del ae
    gc.collect()

    vae = VAEInference(ROOT / "checkpoints" / "VAE_V5_FINAL.pth", device, 256)
    _, deterministic, variations, vae_metrics = vae.reconstruct_variations(
        image, n_samples=3, temperature=1.0, seed=42
    )
    report["vae"] = {
        "deterministic_shape": list(deterministic.shape),
        "variation_count": len(variations),
        "mean_absolute_deltas": [
            item["mean_abs_delta_from_deterministic"] for item in variations
        ],
        "difference_map_shapes": [list(item["difference_map"].shape) for item in variations],
        **vae_metrics,
    }
    del vae
    gc.collect()

    vision = TransformerInference(ROOT / "checkpoints" / "transformer_v2_final.pth", device)
    _, vision_output, vision_metrics, latent, attention = vision.reconstruct(image)
    report["vision_transformer"] = {
        "shape": list(vision_output.shape),
        "latent_shape": list(latent.shape),
        "attention_shape": list(attention.shape),
        **vision_metrics,
    }
    del vision
    gc.collect()

    evidence = (
        "At 09:15 on 12 March 2026, analyst Mira Patel collected device EX-17. "
        "The transfer log records SHA-256 abc123. A later note states collection began at "
        "10:15, but no timezone is recorded."
    )
    language_model = EvidenceTransformer(device=device)
    text_result = language_model.generate(evidence, DEFAULT_PROMPT, 160)
    parsed_pdf = parse_evidence_file("review2_smoke.pdf", _text_pdf(evidence))
    pdf_result = language_model.generate(parsed_pdf.text, DEFAULT_PROMPT, 160)
    for result in (text_result, pdf_result):
        if RESPONSIBLE_USE_NOTE not in result.output or "### Evidence Summary" not in result.output:
            raise RuntimeError("Evidence output is missing its structured or responsible-use content.")
    report["evidence_transformer"] = {
        "model": text_result.model_id,
        "text": text_result.to_dict(),
        "pdf": {**pdf_result.to_dict(), "extracted_pages": parsed_pdf.page_count},
        "text_output": text_result.output,
        "pdf_output": pdf_result.output,
    }

    # Diffusion service verification
    from models.diffusion_service import (
        DIFFUSION_MODEL_ID,
        DiffusionService,
        apply_synthetic_watermark,
        load_canonical_gallery,
    )

    diff_service = DiffusionService(
        model_id=DIFFUSION_MODEL_ID,
        checkpoint_unet_path=ROOT / "checkpoints" / "Diffusion_UNet.pth",
        device=device,
    )
    hw_status = diff_service.get_hardware_status()
    gallery = load_canonical_gallery(ROOT / "outputs" / "diffusion")

    # Verify watermarking pipeline
    dummy_input = Image.new("RGB", (128, 128), color=(80, 120, 160))
    wm_test = apply_synthetic_watermark(dummy_input, text="SMOKE TEST WATERMARK")

    report["diffusion_service"] = {
        "model_id": DIFFUSION_MODEL_ID,
        "device": str(device),
        "hardware_status": hw_status,
        "canonical_gallery_count": len(gallery),
        "watermark_verified": wm_test.size == (128, 128),
        "checkpoint_exists": (ROOT / "checkpoints" / "Diffusion_UNet.pth").is_file(),
    }

    # DCGAN service verification
    from models.gan_service import (
        GANService,
        GANGenerationConfig,
        apply_gan_watermark,
    )

    gan_service = GANService(
        checkpoint_path=ROOT / "checkpoints" / "best_generator.pth",
        device=device,
    )
    gan_info = gan_service.get_model_info()
    gan_gen = gan_service.generate(GANGenerationConfig(count=1, seed=42))
    gan_interp = gan_service.interpolate(seed_a=42, seed_b=99, steps=4)

    report["dcgan_service"] = {
        "model_type": gan_info["model_type"],
        "latent_dim": gan_info["latent_dim"],
        "generator_parameters": gan_info["generator_parameters"],
        "epochs_completed": gan_info["epochs_completed"],
        "sample_shape": list(gan_gen.images[0].size),
        "upscaled_shape": list(gan_gen.upscaled_images[0].size),
        "interpolation_count": len(gan_interp),
        "latency_seconds": gan_gen.generation_time_seconds,
    }

    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
