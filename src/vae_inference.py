"""Reusable inference wrapper for the final forensic reconstruction VAE V5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ae_inference import _state_dict, image_metrics, preprocess_image
from vae import DEFAULT_LATENT_DIM, VAEV5


class VAEInference:
    """Load VAE V5 once and expose deterministic reconstruction and sampling."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device,
        latent_dim: int = DEFAULT_LATENT_DIM,
    ) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"VAE V5 checkpoint not found: {path}")
        self.device = device
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict):
            latent_dim = int(checkpoint.get("latent_dim", latent_dim))
            self.checkpoint_metadata = {
                key: checkpoint.get(key)
                for key in ("epoch", "val_mse", "val_psnr", "val_kl", "beta", "l1_weight")
            }
        else:
            self.checkpoint_metadata = {}
        self.latent_dim = latent_dim
        self.model = VAEV5(latent_dim).to(device)
        incompatible = self.model.load_state_dict(_state_dict(checkpoint), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"VAE V5 checkpoint mismatch: {incompatible}")
        self.model.eval()

    def reconstruct(
        self, image: Image.Image
    ) -> tuple[Image.Image, np.ndarray, dict[str, float]]:
        original_image, tensor = preprocess_image(image, 128)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            reconstructed, _, _ = self.model.reconstruct(tensor, deterministic=True)
        metrics = image_metrics(tensor, reconstructed)
        metrics["reconstruction_error"] = metrics["mse"]
        output = reconstructed.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
        return original_image, output, metrics

    def generate(self) -> np.ndarray:
        """Decode a prior sample with zero skips; output is synthetic research data."""
        with torch.no_grad():
            z = torch.randn(1, self.latent_dim, device=self.device)
            generated = self.model.decode(z)
        return generated.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
