"""Train the final authentic-only VAE V5 on the existing CASIA manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_dataset import AESample, AutoencoderImageDataset
from vae import (
    DEFAULT_BETA_END,
    DEFAULT_BETA_START,
    DEFAULT_KL_WARMUP_EPOCHS,
    DEFAULT_L1_WEIGHT,
    DEFAULT_LATENT_DIM,
    VAEV5,
    beta_for_epoch,
    vae_v5_loss,
)


HISTORY_FIELDS = [
    "epoch", "train_total_loss", "train_reconstruction_loss", "train_mse",
    "train_l1", "train_kl_loss", "validation_total_loss",
    "validation_reconstruction_loss", "validation_mse", "validation_l1",
    "validation_kl_loss", "validation_psnr", "beta", "learning_rate",
    "epoch_seconds",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def authentic_loader(
    manifest: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> tuple[DataLoader[AESample], int]:
    """Build a loader containing label-0 records without opening every image."""
    dataset = AutoencoderImageDataset(manifest, image_size)
    indices = [index for index, row in enumerate(dataset.records) if int(row["label"]) == 0]
    subset = Subset(dataset, indices)
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    return loader, len(indices)


def run_epoch(
    model: VAEV5,
    loader: DataLoader[AESample],
    device: torch.device,
    beta: float,
    optimizer: Optimizer | None = None,
    max_batches: int | None = None,
    l1_weight: float | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("total", "reconstruction", "mse", "l1", "kl")}
    count = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            if training:
                reconstruction, mu, logvar = model(images)
            elif hasattr(model, "reconstruct"):
                reconstruction, mu, logvar = model.reconstruct(images, deterministic=True)
            else:
                mu, logvar = model.encode(images)
                reconstruction = model.decode(mu)
            if l1_weight is None:
                mse = torch.nn.functional.mse_loss(reconstruction, images)
                kl = -0.5 * torch.sum(
                    1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1
                ).mean()
                losses = {
                    "total": mse + beta * kl,
                    "reconstruction": mse,
                    "mse": mse,
                    "l1": torch.zeros_like(mse),
                    "kl": kl,
                }
            else:
                losses = vae_v5_loss(
                    reconstruction, images, mu, logvar,
                    beta=beta, l1_weight=l1_weight,
                )
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = images.size(0)
        for key in totals:
            totals[key] += float(losses[key].detach()) * batch_size
        count += batch_size

    if count == 0:
        raise RuntimeError("No authentic images were processed")
    return {key: value / count for key, value in totals.items()}


def save_history(history: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def save_curves(history: list[dict[str, float | int]], path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(epochs, [row["train_reconstruction_loss"] for row in history], label="Train")
    axes[0, 0].plot(epochs, [row["validation_reconstruction_loss"] for row in history], label="Validation")
    axes[0, 0].set(title="Hybrid Reconstruction Loss", ylabel="0.5 MSE + 0.5 L1")
    axes[0, 1].plot(epochs, [row["train_kl_loss"] for row in history], label="Train")
    axes[0, 1].plot(epochs, [row["validation_kl_loss"] for row in history], label="Validation")
    axes[0, 1].set(title="KL Divergence", ylabel="Mean KL per image")
    axes[1, 0].plot(epochs, [row["validation_mse"] for row in history], label="Validation MSE")
    axes[1, 0].plot(epochs, [row["validation_psnr"] for row in history], label="Validation PSNR")
    axes[1, 0].set(title="Validation Reconstruction Metrics")
    axes[1, 1].plot(epochs, [row["beta"] for row in history], label="Beta")
    axes[1, 1].set(title="KL Warm-up", ylabel="Beta")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, train_count = authentic_loader(
        args.splits_dir / "train.csv", args.image_size, args.batch_size,
        args.num_workers, True, args.seed,
    )
    validation_loader, validation_count = authentic_loader(
        args.splits_dir / "validation.csv", args.image_size, args.batch_size,
        args.num_workers, False, args.seed,
    )
    if train_count != args.expected_train_authentic or validation_count != args.expected_validation_authentic:
        raise AssertionError(
            f"Authentic counts are {train_count}/{validation_count}, expected "
            f"{args.expected_train_authentic}/{args.expected_validation_authentic}"
        )

    model = VAEV5(args.latent_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    epochs = 1 if args.smoke_test else args.max_epochs
    max_batches = args.smoke_batches if args.smoke_test else None
    checkpoint_path = (
        args.checkpoint_path.with_name(args.checkpoint_path.stem + "_smoke_test.pth")
        if args.smoke_test else args.checkpoint_path
    )
    history: list[dict[str, float | int]] = []
    best_mse = float("inf")
    best_epoch = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        beta = beta_for_epoch(epoch, args.kl_warmup_epochs, args.beta_start, args.beta_end)
        training = run_epoch(
            model, train_loader, device, beta, optimizer, max_batches,
            l1_weight=args.l1_weight,
        )
        validation = run_epoch(
            model, validation_loader, device, beta, None, max_batches,
            l1_weight=args.l1_weight,
        )
        validation_psnr = -10.0 * math.log10(max(validation["mse"], 1e-12))
        scheduler.step(validation["mse"])
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_total_loss": training["total"],
            "train_reconstruction_loss": training["reconstruction"],
            "train_mse": training["mse"],
            "train_l1": training["l1"],
            "train_kl_loss": training["kl"],
            "validation_total_loss": validation["total"],
            "validation_reconstruction_loss": validation["reconstruction"],
            "validation_mse": validation["mse"],
            "validation_l1": validation["l1"],
            "validation_kl_loss": validation["kl"],
            "validation_psnr": validation_psnr,
            "beta": beta,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)

        improved = validation["mse"] < best_mse
        if improved:
            best_mse, best_epoch = validation["mse"], epoch
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_mse": validation["mse"],
                    "val_psnr": validation_psnr,
                    "val_l1": validation["l1"],
                    "val_reconstruction": validation["reconstruction"],
                    "val_kl": validation["kl"],
                    "latent_dim": args.latent_dim,
                    "image_size": args.image_size,
                    "l1_weight": args.l1_weight,
                    "beta": beta,
                    "training_scope": "authentic_only",
                },
                checkpoint_path,
            )
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train MSE {training['mse']:.6f} | "
            f"Val MSE {validation['mse']:.6f} | PSNR {validation_psnr:.2f} dB | "
            f"KL {validation['kl']:.4f} | beta {beta:.5f} | "
            f"{time.perf_counter()-epoch_started:.1f}s" + (" | BEST" if improved else ""),
            flush=True,
        )

    training_seconds = time.perf_counter() - started
    save_history(history, args.history_path)
    save_curves(history, args.curve_path)
    summary = {
        "model": "VAEV5 authentic-only forensic reconstruction",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_mse": best_mse,
        "best_validation_psnr": -10.0 * math.log10(max(best_mse, 1e-12)),
        "training_seconds": training_seconds,
        "authentic_train_images": train_count,
        "authentic_validation_images": validation_count,
        "checkpoint_path": str(checkpoint_path),
        "config": {
            "image_size": args.image_size, "batch_size": args.batch_size,
            "latent_dim": args.latent_dim, "optimizer": "AdamW",
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs, "early_stopping": False,
            "gradient_clip": 1.0, "scheduler": "ReduceLROnPlateau",
            "l1_weight": args.l1_weight, "beta_start": args.beta_start,
            "beta_end": args.beta_end, "kl_warmup_epochs": args.kl_warmup_epochs,
            "seed": args.seed,
        },
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/best_vae_v5_forensic.pth"))
    parser.add_argument("--history-path", type=Path, default=Path("results/vae_v5_training_history.csv"))
    parser.add_argument("--summary-path", type=Path, default=Path("results/vae_v5_training_summary.json"))
    parser.add_argument("--curve-path", type=Path, default=Path("outputs/vae_v5/training_curves.png"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--kl-warmup-epochs", type=int, default=DEFAULT_KL_WARMUP_EPOCHS)
    parser.add_argument("--beta-start", type=float, default=DEFAULT_BETA_START)
    parser.add_argument("--beta-end", type=float, default=DEFAULT_BETA_END)
    parser.add_argument("--l1-weight", type=float, default=DEFAULT_L1_WEIGHT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-train-authentic", type=int, default=5244)
    parser.add_argument("--expected-validation-authentic", type=int, default=1124)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
