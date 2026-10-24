"""Re-export DCGAN service module for convenience."""

from models.gan_service import (
    DEFAULT_DISCRIMINATOR_CHECKPOINT,
    DEFAULT_DISPLAY_UPSCALED_SIZE,
    DEFAULT_GENERATOR_CHECKPOINT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LATENT_DIM,
    GAN_AUDIT_LOG_CSV,
    SYNTHETIC_GAN_DISCLAIMER,
    GANCheckpointMissingError,
    GANGenerationConfig,
    GANGenerationResult,
    GANService,
    GANServiceError,
    apply_gan_watermark,
    log_gan_generation_event,
    slerp,
)

__all__ = [
    "DEFAULT_DISCRIMINATOR_CHECKPOINT",
    "DEFAULT_DISPLAY_UPSCALED_SIZE",
    "DEFAULT_GENERATOR_CHECKPOINT",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_LATENT_DIM",
    "GAN_AUDIT_LOG_CSV",
    "SYNTHETIC_GAN_DISCLAIMER",
    "GANCheckpointMissingError",
    "GANGenerationConfig",
    "GANGenerationResult",
    "GANService",
    "GANServiceError",
    "apply_gan_watermark",
    "log_gan_generation_event",
    "slerp",
]
