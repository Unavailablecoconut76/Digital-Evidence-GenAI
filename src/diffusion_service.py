"""Re-export diffusion service module for convenience."""

from models.diffusion_service import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_RESOLUTION,
    DEFAULT_SEED,
    DIFFUSION_MODEL_ID,
    PRESET_SCENARIOS,
    SYNTHETIC_DISCLAIMER_TEXT,
    DiffusionGenerationConfig,
    DiffusionGenerationResult,
    DiffusionModelMissingError,
    DiffusionService,
    DiffusionServiceError,
    apply_synthetic_watermark,
    get_default_device,
    load_canonical_gallery,
    log_generation_event,
)

__all__ = [
    "DEFAULT_GUIDANCE_SCALE",
    "DEFAULT_INFERENCE_STEPS",
    "DEFAULT_RESOLUTION",
    "DEFAULT_SEED",
    "DIFFUSION_MODEL_ID",
    "PRESET_SCENARIOS",
    "SYNTHETIC_DISCLAIMER_TEXT",
    "DiffusionGenerationConfig",
    "DiffusionGenerationResult",
    "DiffusionModelMissingError",
    "DiffusionService",
    "DiffusionServiceError",
    "apply_synthetic_watermark",
    "get_default_device",
    "load_canonical_gallery",
    "log_generation_event",
]
