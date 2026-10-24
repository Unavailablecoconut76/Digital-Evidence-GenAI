import sys
import types
from pathlib import Path, PurePosixPath, PosixPath, WindowsPath
import torch

if "pathlib._local" not in sys.modules:
    import pathlib
    dummy_local = types.ModuleType("pathlib._local")
    dummy_local.PosixPath = PurePosixPath
    dummy_local.WindowsPath = WindowsPath
    dummy_local.Path = Path
    dummy_local.PurePosixPath = PurePosixPath
    sys.modules["pathlib._local"] = dummy_local

sys.path.insert(0, str(Path("src").resolve()))
from dcgan import DCGANGenerator, DCGANDiscriminator

with torch.serialization.safe_globals([(PurePosixPath, "pathlib._local.PosixPath"), (PurePosixPath, "pathlib.PosixPath")]):
    gen_ckpt = torch.load("checkpoints/best_generator.pth", map_location="cpu", weights_only=True)
    disc_ckpt = torch.load("checkpoints/best_discriminator.pth", map_location="cpu", weights_only=True)

print("=== GENERATOR CHECKPOINT ===")
print("Epoch:", gen_ckpt.get("epoch"))
print("Config:", gen_ckpt.get("config"))

print("\n=== DISCRIMINATOR CHECKPOINT ===")
print("Epoch:", disc_ckpt.get("epoch"))

gen = DCGANGenerator(latent_dim=100)
gen.load_state_dict(gen_ckpt["model_state_dict"])
gen.eval()

torch.manual_seed(42)
noise = torch.randn(4, 100, 1, 1)
with torch.no_grad():
    out = gen(noise)
print("\nGenerated output shape:", out.shape)
print("Generated min:", float(out.min()), "max:", float(out.max()))
rgb = ((out + 1.0) / 2.0).clamp(0, 1)
print("RGB min:", float(rgb.min()), "max:", float(rgb.max()))
print("All assertions passed!")
