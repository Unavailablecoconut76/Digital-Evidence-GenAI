"""Unit tests for the DCGAN inference service."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.models.gan_service import (
    DEFAULT_GENERATOR_CHECKPOINT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LATENT_DIM,
    GANCheckpointMissingError,
    GANGenerationConfig,
    GANGenerationResult,
    GANService,
    apply_gan_watermark,
    slerp,
)


class TestGANService(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoint_path = DEFAULT_GENERATOR_CHECKPOINT
        cls.checkpoint_exists = cls.checkpoint_path.is_file()
        if cls.checkpoint_exists:
            cls.service = GANService(checkpoint_path=cls.checkpoint_path, device=torch.device("cpu"))
            cls.service.load_model()
        else:
            cls.service = None

    def test_slerp_interpolation(self) -> None:
        z1 = torch.randn(1, 100)
        z2 = torch.randn(1, 100)
        z_mid = slerp(0.5, z1, z2)
        self.assertEqual(z_mid.shape, (1, 100))
        # At alpha=0, should equal z1
        z_start = slerp(0.0, z1, z2)
        torch.testing.assert_close(z_start, z1, atol=1e-5, rtol=1e-5)
        # At alpha=1, should equal z2
        z_end = slerp(1.0, z1, z2)
        torch.testing.assert_close(z_end, z2, atol=1e-5, rtol=1e-5)

    def test_watermark_applied(self) -> None:
        img = Image.new("RGB", (256, 256), color=(100, 150, 200))
        wm_img = apply_gan_watermark(img, seed=123)
        self.assertEqual(wm_img.size, (256, 256))
        # Image should be modified (banner added at bottom)
        orig_arr = np.array(img)
        wm_arr = np.array(wm_img)
        self.assertFalse(np.array_equal(orig_arr, wm_arr))

    def test_missing_checkpoint_raises(self) -> None:
        fake_service = GANService(checkpoint_path=Path("checkpoints/non_existent_gan.pth"), device=torch.device("cpu"))
        with self.assertRaises((GANCheckpointMissingError, FileNotFoundError)):
            fake_service.load_model()

    def test_checkpoint_loaded_and_parameters(self) -> None:
        if not self.checkpoint_exists:
            self.skipTest("GAN checkpoint not present locally")
        self.assertTrue(self.service.is_loaded)
        info = self.service.get_model_info()
        self.assertEqual(info["latent_dim"], 100)
        self.assertEqual(info["epochs_completed"], 30)
        self.assertEqual(info["generator_parameters"], 3576704)

    def test_generation_shape_and_seed_reproducibility(self) -> None:
        if not self.checkpoint_exists:
            self.skipTest("GAN checkpoint not present locally")
        config1 = GANGenerationConfig(count=1, seed=42)
        config2 = GANGenerationConfig(count=1, seed=42)
        res1 = self.service.generate(config1)
        res2 = self.service.generate(config2)

        self.assertEqual(len(res1.images), 1)
        self.assertEqual(res1.images[0].size, DEFAULT_IMAGE_SIZE)
        self.assertEqual(res1.upscaled_images[0].size, (256, 256))
        self.assertEqual(res1.resolution, (64, 64))

        # Check reproducibility: same seed gives identical pixels
        arr1 = np.array(res1.images[0])
        arr2 = np.array(res2.images[0])
        np.testing.assert_array_equal(arr1, arr2)

    def test_multi_sample_generation(self) -> None:
        if not self.checkpoint_exists:
            self.skipTest("GAN checkpoint not present locally")
        config = GANGenerationConfig(count=4, seed=100)
        res = self.service.generate(config)
        self.assertEqual(len(res.images), 4)
        self.assertEqual(len(res.upscaled_images), 4)
        self.assertEqual(len(res.watermarked_images), 4)
        # Ensure images are distinct
        arr0 = np.array(res.images[0])
        arr1 = np.array(res.images[1])
        self.assertFalse(np.array_equal(arr0, arr1))

    def test_latent_interpolation(self) -> None:
        if not self.checkpoint_exists:
            self.skipTest("GAN checkpoint not present locally")
        interp_images = self.service.interpolate(seed_a=42, seed_b=99, steps=5)
        self.assertEqual(len(interp_images), 5)
        for img in interp_images:
            self.assertEqual(img.size, (256, 256))


if __name__ == "__main__":
    unittest.main()
