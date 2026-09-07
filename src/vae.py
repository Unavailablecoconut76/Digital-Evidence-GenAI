"""Final VAE V5 model for CASIA forensic reconstruction and anomaly analysis.

Parameter names and tensor shapes match the trained forensic V5 checkpoint.
Reconstruction uses encoder skip features. Prior sampling uses zero-valued
skips because no source image is available, preserving checkpoint compatibility.
"""

from __future__ import annotations

import warnings
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_LATENT_DIM = 256
DEFAULT_L1_WEIGHT = 0.5
DEFAULT_BETA_START = 0.00005
DEFAULT_BETA_END = 0.00030
DEFAULT_KL_WARMUP_EPOCHS = 20


class ResidualBlock(nn.Module):
    """Two-convolution residual block used throughout VAE V5."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + x)


class VAEV5(nn.Module):
    """Residual, skip-connected VAE V5 for 128 x 128 RGB images."""

    feature_channels = 256
    feature_size = 8

    def __init__(self, latent_dim: int = DEFAULT_LATENT_DIM) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.e1 = self._encoder_block(3, 32)
        self.e2 = self._encoder_block(32, 64)
        self.e3 = self._encoder_block(64, 128)
        self.e4 = self._encoder_block(128, 256)

        self.feature_dim = self.feature_channels * self.feature_size * self.feature_size
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.feature_dim)

        self.d4 = self._decoder_block(256, 256)
        self.d3 = self._decoder_block(384, 128)
        self.d2 = self._decoder_block(192, 64)
        self.d1 = self._decoder_block(96, 32)
        self.final = nn.Conv2d(32, 3, 3, padding=1)

    @staticmethod
    def _encoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            ResidualBlock(out_channels),
        )

    @staticmethod
    def _decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            ResidualBlock(out_channels),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return mu, logvar, and all four encoder feature maps."""
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        flattened = e4.view(e4.size(0), -1)
        return self.fc_mu(flattened), self.fc_logvar(flattened), e1, e2, e3, e4

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    @staticmethod
    def _zero_skip(z: torch.Tensor, channels: int, size: int) -> torch.Tensor:
        return torch.zeros(z.size(0), channels, size, size, device=z.device, dtype=z.dtype)

    def decode(
        self,
        z: torch.Tensor,
        e1: torch.Tensor | None = None,
        e2: torch.Tensor | None = None,
        e3: torch.Tensor | None = None,
        e4: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode with image skips for reconstruction or zero skips for sampling."""
        supplied = (e1, e2, e3, e4)
        if any(item is None for item in supplied) and not all(item is None for item in supplied):
            raise ValueError("Provide every encoder skip tensor or none of them")
        if all(item is None for item in supplied):
            e1 = self._zero_skip(z, 32, 64)
            e2 = self._zero_skip(z, 64, 32)
            e3 = self._zero_skip(z, 128, 16)
            e4 = self._zero_skip(z, 256, 8)

        # e4 is intentionally accepted to match the final notebook API; the
        # learned decoder starts from fc_decode(z), as in the source notebook.
        del e4
        assert e1 is not None and e2 is not None and e3 is not None
        x = self.fc_decode(z).view(-1, 256, 8, 8)
        x = self.d4(x)
        x = F.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d3(torch.cat([x, e3], dim=1))
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d2(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d1(torch.cat([x, e1], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.final(x))

    def reconstruct(
        self, x: torch.Tensor, deterministic: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, e1, e2, e3, e4 = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        return self.decode(z, e1, e2, e3, e4), mu, logvar

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.reconstruct(x, deterministic=False)


def vae_v5_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = DEFAULT_BETA_END,
    l1_weight: float = DEFAULT_L1_WEIGHT,
) -> dict[str, torch.Tensor]:
    """Exact V5 hybrid reconstruction, mean per-sample KL, and total loss."""
    if not 0.0 <= l1_weight <= 1.0:
        raise ValueError("l1_weight must be in [0, 1]")
    mse = F.mse_loss(reconstruction, target, reduction="mean")
    l1 = F.l1_loss(reconstruction, target, reduction="mean")
    reconstruction_loss = (1.0 - l1_weight) * mse + l1_weight * l1
    kl = -0.5 * torch.mean(
        torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )
    return {
        "total": reconstruction_loss + beta * kl,
        "reconstruction": reconstruction_loss,
        "mse": mse,
        "l1": l1,
        "kl": kl,
    }


def beta_for_epoch(
    epoch: int,
    warmup_epochs: int = DEFAULT_KL_WARMUP_EPOCHS,
    beta_start: float = DEFAULT_BETA_START,
    beta_end: float = DEFAULT_BETA_END,
) -> float:
    """Linear V5 KL warm-up, reaching beta_end on warmup_epochs."""
    if epoch < 1 or warmup_epochs < 1:
        raise ValueError("epoch and warmup_epochs must be positive")
    if epoch > warmup_epochs:
        return beta_end
    return beta_start + (beta_end - beta_start) * (epoch - 1) / max(warmup_epochs - 1, 1)


def __getattr__(name: str):
    """Keep historical V1/V2 scripts importable without making them current."""
    if name == "ConvolutionalVAE":
        try:
            from .vae_legacy import ConvolutionalVAE
        except ImportError:
            from vae_legacy import ConvolutionalVAE
        warnings.warn(
            "ConvolutionalVAE is the historical V1/V2 architecture; use VAEV5.",
            DeprecationWarning,
            stacklevel=2,
        )
        return ConvolutionalVAE
    raise AttributeError(name)


__all__ = [
    "VAEV5", "ResidualBlock", "vae_v5_loss", "beta_for_epoch",
    "DEFAULT_LATENT_DIM", "DEFAULT_L1_WEIGHT", "DEFAULT_BETA_START",
    "DEFAULT_BETA_END", "DEFAULT_KL_WARMUP_EPOCHS",
]
