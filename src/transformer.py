"""Transformer Forensic Autoencoder V2 extracted from the project notebook."""

from __future__ import annotations

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=128, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class PositionalEmbedding(nn.Module):
    """Learnable positional embedding implied by the notebook/checkpoint layout."""

    def __init__(self, num_patches=64, embed_dim=256):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_embedding


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm1(x)
        attended, weights = self.attention(
            normalized, normalized, normalized, need_weights=True
        )
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, weights


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=512, num_layers=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attention_maps = []
        for layer in self.layers:
            x, weights = layer(x)
            attention_maps.append(weights)
        return self.final_norm(x), attention_maps


class ImageRepresentation(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.mean(dim=1))


class TransformerForensicAutoencoder(nn.Module):
    """V2 model preserving all encoded patch tokens for reconstruction."""

    def __init__(
        self, image_size=128, patch_size=16, in_channels=3, embed_dim=256,
        num_heads=8, encoder_layers=4, decoder_layers=2, ff_dim=512, dropout=0.1,
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.positional_embedding = PositionalEmbedding(num_patches, embed_dim)
        self.transformer_encoder = TransformerEncoder(
            embed_dim, num_heads, ff_dim, encoder_layers, dropout
        )
        self.image_representation = ImageRepresentation(embed_dim)
        self.decoder_position_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.decoder_position_embedding, std=0.02)
        self.transformer_decoder_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.patch_projection = nn.Linear(embed_dim, patch_size * patch_size * in_channels)
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        tokens = self.positional_embedding(self.patch_embedding(x))
        encoded, attention_maps = self.transformer_encoder(tokens)
        latent = self.image_representation(encoded)
        decoder_tokens = encoded + self.decoder_position_embedding
        for layer in self.transformer_decoder_layers:
            decoder_tokens, _ = layer(decoder_tokens)
        decoder_tokens = self.decoder_norm(decoder_tokens)
        patches = self.patch_projection(decoder_tokens).view(
            batch_size, self.num_patches, self.in_channels,
            self.patch_size, self.patch_size,
        )
        side = self.image_size // self.patch_size
        patches = patches.view(
            batch_size, side, side, self.in_channels, self.patch_size, self.patch_size
        )
        reconstructed = patches.permute(0, 3, 1, 4, 2, 5).contiguous().view(
            batch_size, self.in_channels, self.image_size, self.image_size
        )
        return torch.sigmoid(reconstructed), latent, attention_maps


__all__ = ["TransformerForensicAutoencoder"]
