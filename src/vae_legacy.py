"""Historical convolutional VAE retained only for V1/V2 compatibility."""

from __future__ import annotations

import torch
from torch import nn


class ConvolutionalVAE(nn.Module):
    """Original 128-dimensional VAE used by the historical V1/V2 scripts."""

    encoded_channels = 128
    encoded_size = 8

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.flattened_feature_dim = 128 * 8 * 8
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(self.flattened_feature_dim, latent_dim)
        self.logvar_head = nn.Linear(self.flattened_feature_dim, latent_dim)
        self.decoder_projection = nn.Linear(latent_dim, self.flattened_feature_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.flatten(self.encode_features(x), start_dim=1)
        return self.mu_head(features), self.logvar_head(features)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        projected = self.decoder_projection(z).view(
            z.shape[0], self.encoded_channels, self.encoded_size, self.encoded_size
        )
        return self.decoder(projected)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


__all__ = ["ConvolutionalVAE"]
