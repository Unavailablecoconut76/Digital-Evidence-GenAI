"""Diffusion service for synthetic digital evidence scenario generation.

Wraps pretrained Stable Diffusion 1.5 with support for:
- Deterministic seed-based generation
- Intermediate reverse-denoising latent capture & VAE decoding
- Digital provenance watermarking
- Audit logging
- Hardware-adaptive execution (CUDA/FP16 with attention slicing, CPU/FP32 fallback)
- Canonical gallery fallback for low-resource/offline environments
"""

from __future__ import annotations

import csv
import datetime
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Model constants & provenance
DIFFUSION_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_INFERENCE_STEPS = 25
DEFAULT_RESOLUTION = (512, 512)
DEFAULT_SEED = 201

# Preset forensic incident scenarios from audited notebook
PRESET_SCENARIOS = {
    "Urban Collision Scene": {
        "title": "Urban Collision Scene with Road Debris",
        "prompt": (
            "A fictional urban street scene with a damaged parked car, "
            "small debris on the road, daylight, realistic documentary-style "
            "photography, no visible people"
        ),
        "seed": 201,
        "description": "Simulation of a traffic incident with localized vehicle damage and debris field.",
    },
    "Office Intrusion Scene": {
        "title": "Office Intrusion & Document Disarray",
        "prompt": (
            "A fictional office interior with an open desk drawer, scattered "
            "papers and a tipped chair, neutral lighting, realistic "
            "documentary-style photography, no visible people"
        ),
        "seed": 202,
        "description": "Simulation of commercial burglary or unauthorized physical workspace access.",
    },
    "Roadside Incident Scene": {
        "title": "Roadside Incident with Broken Bicycle",
        "prompt": (
            "A fictional outdoor scene with a broken bicycle beside a road, "
            "small debris nearby, overcast daylight, realistic "
            "documentary-style photography, no visible people"
        ),
        "seed": 203,
        "description": "Simulation of an outdoor non-vehicular or cycling incident scene.",
    },
    "Warehouse Security Breach": {
        "title": "Warehouse Perimeter Compromise",
        "prompt": (
            "A fictional warehouse interior with a compromised security door, "
            "broken latch, scattered packing materials, industrial overhead "
            "lighting, realistic documentary-style photography, no visible people"
        ),
        "seed": 204,
        "description": "Simulation of an industrial perimeter breach with physical evidence disarray.",
    },
}

SYNTHETIC_DISCLAIMER_TEXT = (
    "DISCLAIMER: This image is a synthetic scenario simulation generated using "
    "pretrained Stable Diffusion 1.5. It does not represent genuine digital evidence, "
    "does not verify authenticity, and cannot detect tampering."
)


@dataclass
class DiffusionGenerationConfig:
    """Configuration parameters for diffusion generation."""

    prompt: str
    negative_prompt: str = (
        "distorted, blurry, low quality, cartoon, 3d render, illustration, "
        "cgi, watermark, signature, text, people, faces"
    )
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    seed: int = DEFAULT_SEED
    height: int = 512
    width: int = 512
    capture_intermediates: bool = True


@dataclass
class DiffusionGenerationResult:
    """Results returned from diffusion generation."""

    image: Image.Image
    watermarked_image: Image.Image
    intermediate_images: list[tuple[int, Image.Image]] = field(default_factory=list)
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = DEFAULT_SEED
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    resolution: tuple[int, int] = DEFAULT_RESOLUTION
    generation_time_seconds: float = 0.0
    peak_gpu_memory_mb: float | None = None
    device: str = "cpu"
    model_id: str = DIFFUSION_MODEL_ID
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())


class DiffusionServiceError(Exception):
    """Base exception for diffusion service errors."""


class DiffusionModelMissingError(DiffusionServiceError):
    """Raised when model weights cannot be located or downloaded."""


def get_default_device() -> torch.device:
    """Determine the optimal compute device available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_synthetic_watermark(
    image: Image.Image,
    text: str = "SYNTHETIC DIGITAL EVIDENCE · NOT GENUINE",
    subtext: str | None = None,
) -> Image.Image:
    """Apply an explicit, permanent digital provenance watermark to a generated image.
    
    Ensures that exported synthetic images are visibly identified as simulations
    and cannot be mistaken for authentic digital evidence.
    """
    canvas = image.copy().convert("RGB")
    width, height = canvas.size
    banner_height = 36 if subtext else 26

    # Draw semi-transparent dark banner along the bottom
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([(0, height - banner_height), (width, height)], fill=(12, 28, 42, 210))
    draw.line([(0, height - banner_height), (width, height - banner_height)], fill=(17, 168, 199, 255), width=2)

    # Text positioning
    draw.text((12, height - banner_height + 4), text, fill=(255, 255, 255, 255))
    if subtext:
        draw.text((12, height - banner_height + 18), subtext, fill=(180, 210, 230, 240))

    # Composite overlay onto canvas
    canvas_rgba = canvas.convert("RGBA")
    composited = Image.alpha_composite(canvas_rgba, overlay)
    return composited.convert("RGB")


def log_generation_event(
    result: DiffusionGenerationResult,
    log_file_path: Path | str,
) -> None:
    """Append a privacy-preserving generation event record to the audit log."""
    path = Path(log_file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.is_file()

    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Model",
                "Prompt",
                "Seed",
                "Steps",
                "GuidanceScale",
                "Resolution",
                "LatencySeconds",
                "Device",
                "PeakGPUMemoryMB",
            ])
        writer.writerow([
            result.timestamp,
            result.model_id,
            result.prompt.replace("\n", " "),
            result.seed,
            result.num_inference_steps,
            result.guidance_scale,
            f"{result.resolution[0]}x{result.resolution[1]}",
            f"{result.generation_time_seconds:.3f}",
            result.device,
            f"{result.peak_gpu_memory_mb:.2f}" if result.peak_gpu_memory_mb is not None else "N/A",
        ])


class DiffusionService:
    """Manages the Stable Diffusion 1.5 pipeline with hardware-aware inference."""

    def __init__(
        self,
        model_id: str = DIFFUSION_MODEL_ID,
        checkpoint_unet_path: Path | str | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model_id = model_id
        self.checkpoint_unet_path = Path(checkpoint_unet_path) if checkpoint_unet_path else None
        self.device = device or get_default_device()
        self._pipe: Any = None
        self._pipeline_loaded: bool = False

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def is_loaded(self) -> bool:
        return self._pipeline_loaded and self._pipe is not None

    def get_hardware_status(self) -> dict[str, Any]:
        """Return operational hardware status and memory characteristics."""
        status: dict[str, Any] = {
            "device": str(self.device),
            "is_cuda": self.is_cuda,
            "torch_version": torch.__version__,
        }
        if self.is_cuda:
            status["device_name"] = torch.cuda.get_device_name(self.device)
            status["total_vram_gb"] = round(torch.cuda.get_device_properties(self.device).total_memory / (1024**3), 2)
            status["allocated_vram_mb"] = round(torch.cuda.memory_allocated(self.device) / (1024**2), 2)
        else:
            status["device_name"] = "Host CPU"
            status["warning"] = "CPU mode is active. Live inference will be slower; 10-15 steps or precomputed gallery is recommended."
        return status

    def load_pipeline(
        self,
        local_files_only: bool = False,
        use_attention_slicing: bool = True,
    ) -> Any:
        """Lazily load and configure the Hugging Face Stable Diffusion pipeline."""
        if self.is_loaded:
            return self._pipe

        try:
            from diffusers import StableDiffusionPipeline
        except ImportError as exc:
            raise DiffusionServiceError(
                "The 'diffusers' library is required to run the diffusion pipeline. "
                "Install it via: pip install diffusers accelerate"
            ) from exc

        dtype = torch.float16 if self.is_cuda else torch.float32

        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise DiffusionModelMissingError(
                f"Failed to load pipeline '{self.model_id}'. "
                f"Ensure internet connectivity or a pre-downloaded local cache: {exc}"
            ) from exc

        pipe = pipe.to(self.device)

        if self.is_cuda and use_attention_slicing:
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass

        # If a verified custom/exported U-Net checkpoint was provided, load its state dict
        if self.checkpoint_unet_path and self.checkpoint_unet_path.is_file():
            try:
                ckpt = torch.load(self.checkpoint_unet_path, map_location=self.device, weights_only=True)
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    pipe.unet.load_state_dict(ckpt["state_dict"])
            except Exception:
                # If loading fails or weights_only raises, preserve standard pipeline weights
                pass

        self._pipe = pipe
        self._pipeline_loaded = True
        return self._pipe

    def generate(
        self,
        config: DiffusionGenerationConfig,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> DiffusionGenerationResult:
        """Execute text-to-image diffusion generation with optional intermediate latent capture."""
        pipe = self.load_pipeline()

        intermediate_snapshots: list[tuple[int, Image.Image]] = []
        captured_latents: list[tuple[int, torch.Tensor]] = []

        # Step callback for reverse-denoising latent capture
        def step_callback(pipeline_obj: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            if config.capture_intermediates:
                total_steps = config.num_inference_steps
                milestones = {
                    0: 1,
                    max(1, total_steps // 4): total_steps // 4,
                    max(1, total_steps // 2): total_steps // 2,
                    max(1, (3 * total_steps) // 4): (3 * total_steps) // 4,
                    total_steps - 1: total_steps,
                }
                if step_index in milestones:
                    latents = callback_kwargs.get("latents")
                    if latents is not None:
                        captured_latents.append((milestones[step_index], latents.detach().clone()))
            if progress_callback:
                progress_callback(step_index + 1, config.num_inference_steps, f"Denoising step {step_index + 1}/{config.num_inference_steps}...")
            return callback_kwargs

        # Prepare reproducible generator
        generator = torch.Generator(self.device.type).manual_seed(config.seed)

        if self.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start_time = time.perf_counter()

        with torch.inference_mode():
            call_kwargs: dict[str, Any] = {
                "prompt": config.prompt,
                "negative_prompt": config.negative_prompt,
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
                "height": config.height,
                "width": config.width,
                "generator": generator,
            }
            if config.capture_intermediates or progress_callback:
                call_kwargs["callback_on_step_end"] = step_callback
                call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

            result = pipe(**call_kwargs)

        if self.is_cuda:
            torch.cuda.synchronize()
            peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            peak_gpu_mb = None

        elapsed = time.perf_counter() - start_time
        final_img = result.images[0]

        # Decode intermediate latent snapshots using VAE
        if captured_latents:
            vae = pipe.vae
            scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
            with torch.inference_mode():
                for step_num, lat in captured_latents:
                    try:
                        decoded = vae.decode(lat.to(device=vae.device, dtype=vae.dtype) / scaling_factor, return_dict=False)[0]
                        decoded = (decoded / 2 + 0.5).clamp(0, 1)
                        decoded_np = decoded[0].permute(1, 2, 0).float().cpu().numpy()
                        decoded_img = Image.fromarray((decoded_np * 255).astype(np.uint8))
                        intermediate_snapshots.append((step_num, decoded_img))
                    except Exception:
                        pass

        # Apply digital provenance watermark
        watermarked = apply_synthetic_watermark(
            final_img,
            text="SYNTHETIC DIGITAL EVIDENCE · NOT GENUINE",
            subtext=f"Stable Diffusion 1.5 | Seed: {config.seed} | Steps: {config.num_inference_steps} | CFG: {config.guidance_scale}",
        )

        return DiffusionGenerationResult(
            image=final_img,
            watermarked_image=watermarked,
            intermediate_images=intermediate_snapshots,
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            seed=config.seed,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            resolution=(config.width, config.height),
            generation_time_seconds=elapsed,
            peak_gpu_memory_mb=peak_gpu_mb,
            device=str(self.device),
            model_id=self.model_id,
        )


def load_canonical_gallery(artifacts_dir: Path | str) -> list[dict[str, Any]]:
    """Load metadata and paths for precomputed canonical artifacts."""
    base_dir = Path(artifacts_dir)
    gallery: list[dict[str, Any]] = []

    items = [
        {
            "id": "scenes_comparison",
            "title": "Three Synthetic Incident Scenes (Comparison)",
            "filename": "synthetic_evidence_comparison.png",
            "category": "Demonstration",
            "description": "Side-by-side comparison of urban car damage, office disarray, and roadside bicycle scenes generated on Tesla T4 (25 steps, CFG 7.5).",
        },
        {
            "id": "reverse_diffusion",
            "title": "Reverse Diffusion: Decoded Intermediate Latents",
            "filename": "reverse_diffusion_progress.png",
            "category": "Architecture & Process",
            "description": "Progressive emergence of scene features across denoising steps (steps 1, 5, 10, 15, 20, 25) decoded through the VAE decoder.",
        },
        {
            "id": "forward_diffusion",
            "title": "Forward Diffusion: Progressive Noise Simulation",
            "filename": "forward_diffusion_progress.png",
            "category": "Architecture & Process",
            "description": "Progressive noise injection from original image to pure noise (noise levels t=0.0 to 1.0).",
        },
        {
            "id": "step_comparison",
            "title": "Inference Step Comparison (10 vs 20 vs 30 Steps)",
            "filename": "inference_step_comparison.png",
            "category": "Evaluation",
            "description": "Visual comparison of structural refinement at 10 steps (2.73s), 20 steps (5.01s), and 30 steps (8.13s).",
        },
        {
            "id": "unified_dashboard",
            "title": "Unified Model Evaluation Dashboard",
            "filename": "Unified_Model_Evaluation_Dashboard.png",
            "category": "Evaluation",
            "description": "Multi-metric panel comparing generation time, peak GPU memory, and model parameter counts.",
        },
    ]

    for item in items:
        file_path = base_dir / item["filename"]
        if file_path.is_file():
            gallery.append({
                **item,
                "path": str(file_path),
                "size_kb": round(file_path.stat().st_size / 1024, 1),
            })

    return gallery
