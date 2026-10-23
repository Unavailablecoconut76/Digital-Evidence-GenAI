"""Unit tests for diffusion service module."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.diffusion_service import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_SEED,
    DIFFUSION_MODEL_ID,
    PRESET_SCENARIOS,
    DiffusionGenerationConfig,
    DiffusionGenerationResult,
    DiffusionModelMissingError,
    DiffusionService,
    apply_synthetic_watermark,
    load_canonical_gallery,
    log_generation_event,
)


class TestDiffusionService(unittest.TestCase):
    """Test suite for diffusion service functionality."""

    def test_preset_scenarios_structure(self):
        self.assertGreaterEqual(len(PRESET_SCENARIOS), 3)
        for key, scenario in PRESET_SCENARIOS.items():
            self.assertIn("title", scenario)
            self.assertIn("prompt", scenario)
            self.assertIn("seed", scenario)
            self.assertIn("description", scenario)
            self.assertTrue(scenario["prompt"].startswith("A fictional"))

    def test_generation_config_defaults(self):
        config = DiffusionGenerationConfig(prompt="Test forensic prompt")
        self.assertEqual(config.num_inference_steps, DEFAULT_INFERENCE_STEPS)
        self.assertEqual(config.guidance_scale, DEFAULT_GUIDANCE_SCALE)
        self.assertEqual(config.seed, DEFAULT_SEED)
        self.assertEqual(config.width, 512)
        self.assertEqual(config.height, 512)

    def test_apply_synthetic_watermark(self):
        dummy = Image.new("RGB", (256, 256), color=(100, 150, 200))
        watermarked = apply_synthetic_watermark(dummy, text="TEST WATERMARK", subtext="Subtext info")
        self.assertEqual(watermarked.size, (256, 256))
        self.assertEqual(watermarked.mode, "RGB")
        # Banner should have altered bottom pixels
        dummy_pixel = dummy.getpixel((128, 250))
        wm_pixel = watermarked.getpixel((128, 250))
        self.assertNotEqual(dummy_pixel, wm_pixel)

    def test_log_generation_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_log.csv"
            dummy_img = Image.new("RGB", (64, 64), color="gray")
            res = DiffusionGenerationResult(
                image=dummy_img,
                watermarked_image=dummy_img,
                prompt="Fictional burglary scene",
                seed=42,
                num_inference_steps=20,
                guidance_scale=7.5,
                resolution=(64, 64),
                generation_time_seconds=1.234,
                peak_gpu_memory_mb=None,
                device="cpu",
                model_id=DIFFUSION_MODEL_ID,
            )
            log_generation_event(res, log_path)
            self.assertTrue(log_path.is_file())

            with open(log_path, mode="r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
                self.assertEqual(len(rows), 2)  # Header + 1 entry
                self.assertEqual(rows[0][0], "Timestamp")
                self.assertEqual(rows[1][1], DIFFUSION_MODEL_ID)
                self.assertEqual(rows[1][3], "42")
                self.assertEqual(rows[1][4], "20")

    def test_canonical_gallery_loading(self):
        artifacts_dir = ROOT / "outputs" / "diffusion"
        gallery = load_canonical_gallery(artifacts_dir)
        self.assertIsInstance(gallery, list)
        if artifacts_dir.is_dir() and any(artifacts_dir.glob("*.png")):
            self.assertGreater(len(gallery), 0)
            first = gallery[0]
            self.assertIn("title", first)
            self.assertIn("path", first)
            self.assertTrue(Path(first["path"]).is_file())

    def test_hardware_status(self):
        service = DiffusionService()
        status = service.get_hardware_status()
        self.assertIn("device", status)
        self.assertIn("is_cuda", status)
        self.assertIn("torch_version", status)

    def test_missing_model_error_handling(self):
        service = DiffusionService(model_id="nonexistent-model-org/nonexistent-model-id")
        with self.assertRaises(DiffusionModelMissingError):
            service.load_pipeline(local_files_only=True)


if __name__ == "__main__":
    unittest.main()
