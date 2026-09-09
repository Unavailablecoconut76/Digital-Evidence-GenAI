"""Validate final VAE V5 shapes, loss, sampling, and checkpoint compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from vae import VAEV5, vae_v5_loss


def validate() -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path("checkpoints/VAE_V5_FINAL.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    latent_dim = int(checkpoint["latent_dim"])
    model = VAEV5(latent_dim).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    inputs = torch.rand(2, 3, 128, 128, device=device)
    with torch.no_grad():
        reconstruction, mu, logvar = model.reconstruct(inputs, deterministic=True)
        z = model.reparameterize(mu, logvar)
        generated = model.decode(torch.randn(2, latent_dim, device=device))
        losses = vae_v5_loss(
            reconstruction, inputs, mu, logvar,
            beta=float(checkpoint["beta"]),
            l1_weight=float(checkpoint["l1_weight"]),
        )

    finite = all(torch.isfinite(tensor).all() for tensor in (reconstruction, mu, logvar, z, generated))
    assert reconstruction.shape == inputs.shape == generated.shape
    assert mu.shape == logvar.shape == z.shape == (2, 256)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    assert finite and 0.0 <= float(reconstruction.min()) <= float(reconstruction.max()) <= 1.0

    summary = {
        "model": "VAEV5",
        "checkpoint": str(checkpoint_path),
        "input_shape": list(inputs.shape),
        "mu_shape": list(mu.shape),
        "logvar_shape": list(logvar.shape),
        "latent_shape": list(z.shape),
        "output_shape": list(reconstruction.shape),
        "generated_shape": list(generated.shape),
        "latent_dim": latent_dim,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "losses": {name: float(value) for name, value in losses.items()},
        "output_range": [float(reconstruction.min()), float(reconstruction.max())],
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "finite": bool(finite),
        "checkpoint_metadata": {
            key: checkpoint.get(key) for key in ("epoch", "val_mse", "val_psnr", "val_kl")
        },
        "device": str(device),
    }
    output = Path("results/vae_v5_architecture_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
