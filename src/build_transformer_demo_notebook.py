"""Build the load-only Transformer V2 faculty demonstration notebook."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "Transformer" / "Multi_Model_Generative_AI_Framework_for_Digital_Evidence_Analysis_and_Intelligence_Generation.ipynb"
HISTORY = ROOT / "results" / "transformer_v2_training_history.csv"
OUTPUT = ROOT / "notebooks" / "Transformer" / "Transformer_CASIA_Complete_Demo.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def extract_history() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    output_text = "\n".join(
        "".join(output.get("text", []))
        for output in notebook["cells"][34].get("outputs", [])
    )
    pattern = re.compile(
        r"Epoch \[(\d+)/50\] \| Train MSE: ([0-9.]+) \| "
        r"Val MSE: ([0-9.]+) \| LR: ([0-9.eE+-]+)"
    )
    rows = [
        {"epoch": int(epoch), "train_mse": float(train),
         "validation_mse": float(validation), "learning_rate": float(lr)}
        for epoch, train, validation, lr in pattern.findall(output_text)
    ]
    if len(rows) != 50:
        raise RuntimeError(f"Expected 50 Transformer epochs, found {len(rows)}")
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def build() -> None:
    cells = [
        md("""# Transformer V2 — Complete CASIA Faculty Demo

This notebook presents the trained **Transformer Forensic Autoencoder V2** using CASIA v2.0. It loads `transformer_v2_final.pth`, explains patch attention, shows the verified 50-epoch training history, reconstructs held-out images, calculates MSE/PSNR/SSIM, and visualizes attention and reconstruction differences.

**Reconstruction error and attention are exploratory forensic indicators—not tampering probabilities or proof of manipulation. No retraining occurs.**"""),
        code('''# Imports, reproducibility, and project discovery
from pathlib import Path
import json, random, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from IPython.display import display
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
candidates=[Path.cwd(),*Path.cwd().parents,Path('/content/Digital-Evidence-GenAI'),Path('/content/AI-Digital-Evidence-Intelligence-Platform')]
PROJECT_ROOT=next((p.resolve() for p in candidates if (p/'src'/'transformer.py').is_file()),None)
if PROJECT_ROOT is None: raise FileNotFoundError('Run this notebook inside the project repository.')
sys.path.insert(0,str(PROJECT_ROOT/'src'))
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Project:',PROJECT_ROOT); print('Device:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('Mode: trained-checkpoint demonstration (no retraining)')'''),
        md("""## 1. CASIA dataset and fixed manifests

- Total images: 12,614
- Authentic: 7,491
- Tampered: 5,123
- RGB → 128×128 → tensor in `[0,1]`
- Seed: 42
- Ground-truth masks are excluded from model input.

The repository's canonical 70/15/15 manifests are reused. Labels are retained only for exploratory comparison."""),
        code('''frames={name:pd.read_csv(PROJECT_ROOT/'data'/'splits'/f'{name}.csv') for name in ('train','validation','test')}
rows=[]
for name,frame in frames.items(): rows.append({'Split':name.title(),'Total':len(frame),'Authentic':int((frame.label==0).sum()),'Tampered':int((frame.label==1).sum())})
display(pd.DataFrame(rows)); print('Canonical total:',sum(row['Total'] for row in rows))'''),
        md("""## 2. Transformer V2 architecture

```text
3×128×128 image
→ 64 non-overlapping 16×16 patches
→ 64×256 embedded tokens + learned positions
→ 4 Transformer encoder layers (8 attention heads)
→ 256-dimensional global representation for analysis
→ 2 Transformer decoder layers using all 64 encoded tokens
→ 64 reconstructed RGB patches
→ 3×128×128 Sigmoid reconstruction
```

Keeping all encoded patch tokens in the decoder preserves spatial information better than reconstructing from only one global vector."""),
        code('''from transformer import TransformerForensicAutoencoder
CHECKPOINT=PROJECT_ROOT/'checkpoints'/'transformer_v2_final.pth'
checkpoint=torch.load(CHECKPOINT,map_location=DEVICE,weights_only=True)
model=TransformerForensicAutoencoder(image_size=checkpoint['image_size'],patch_size=checkpoint['patch_size'],embed_dim=checkpoint['embed_dim'],num_heads=checkpoint['num_heads'],encoder_layers=checkpoint['encoder_layers'],decoder_layers=checkpoint['decoder_layers'],ff_dim=checkpoint['ff_dim']).to(DEVICE)
incompatible=model.load_state_dict(checkpoint['model_state_dict'],strict=True); model.eval()
configuration={key:checkpoint[key] for key in ('architecture','image_size','patch_size','num_patches','embed_dim','num_heads','encoder_layers','decoder_layers','ff_dim')}
display(pd.DataFrame(configuration.items(),columns=['Property','Value']))
print('Parameters:',f'{sum(p.numel() for p in model.parameters()):,}'); print('Strict loading: PASS'); print('Missing/unexpected keys:',incompatible.missing_keys,incompatible.unexpected_keys)'''),
        md("""## 3. Training configuration and objective

- Authentic-only reconstruction learning in the source experiment
- Epochs: 50
- Batch size: 64
- Optimizer: AdamW
- Learning rate: 0.0001
- Weight decay: 0.00001
- Loss: MSE
- Scheduler: ReduceLROnPlateau
- Best checkpoint selected by validation MSE"""),
        code('''metadata={key:checkpoint.get(key) for key in ('epoch','train_mse','validation_mse','anomaly_threshold','roc_auc','accuracy','precision','recall','f1_score','balanced_accuracy')}
display(pd.DataFrame(metadata.items(),columns=['Checkpoint field','Value']))
print('Checkpoint:',CHECKPOINT)'''),
        md("## 4. Verified training and validation loss curves"),
        code('''history=pd.read_csv(PROJECT_ROOT/'results'/'transformer_v2_training_history.csv'); display(history.tail())
ax=history.plot(x='epoch',y=['train_mse','validation_mse'],figsize=(11,5),grid=True,title='Transformer V2 — 50-Epoch MSE History'); ax.set_ylabel('MSE loss'); plt.show()
best=history.loc[history.validation_mse.idxmin()]; print('Epochs:',len(history)); print('First/final training MSE:',history.train_mse.iloc[0],history.train_mse.iloc[-1]); print('Best epoch/MSE:',int(best.epoch),best.validation_mse)'''),
        md("""## 5. Source evaluation provenance

The source notebook reports validation reconstruction **PSNR 27.5793 dB** and **SSIM 0.8498**. Its final exported checkpoint reports reconstruction-error ROC-AUC **0.5455** and threshold **0.0044039**.

The source notebook's forensic evaluation used 6,248 images (1,125 authentic and all 5,123 tampered images), which is **not the repository's canonical 1,892-image held-out test manifest**. These stored classification values are therefore shown as checkpoint provenance, not relabeled as canonical test results."""),
        md("## 6. Load balanced examples from the canonical held-out test split"),
        code('''from ae_dataset import AutoencoderImageDataset
dataset=AutoencoderImageDataset(PROJECT_ROOT/'data'/'splits'/'test.csv',128)
selected=[]
for index,record in enumerate(dataset.records):
 label=int(record['label'])
 if sum(int(item[1])==label for item in selected)<3: selected.append((index,label))
 if len(selected)==6: break
images=torch.stack([dataset[index]['image'] for index,_ in selected]).to(DEVICE)
labels=[label for _,label in selected]
with torch.no_grad(): reconstructed,latent,attention_maps=model(images)
print('Input:',tuple(images.shape)); print('Reconstruction:',tuple(reconstructed.shape)); print('Latent:',tuple(latent.shape)); print('Attention layers:',len(attention_maps),'| final map:',tuple(attention_maps[-1].shape))'''),
        md("## 7. Original versus reconstructed images and per-image metrics"),
        code('''from skimage.metrics import structural_similarity
records=[]; fig,axes=plt.subplots(6,2,figsize=(8,20))
for i,label in enumerate(labels):
 original=images[i].cpu().permute(1,2,0).numpy(); output=reconstructed[i].cpu().permute(1,2,0).numpy().clip(0,1); mse=float(np.mean((original-output)**2)); psnr=float(-10*np.log10(max(mse,1e-12))); ssim=float(structural_similarity(original,output,channel_axis=2,data_range=1.0)); records.append({'Class':'Authentic' if label==0 else 'Tampered','MSE':mse,'PSNR':psnr,'SSIM':ssim}); axes[i,0].imshow(original); axes[i,0].set_title('Original — '+records[-1]['Class']); axes[i,1].imshow(output); axes[i,1].set_title(f'Reconstructed — MSE {mse:.5f}'); axes[i,0].axis('off'); axes[i,1].axis('off')
plt.tight_layout(); plt.show(); display(pd.DataFrame(records))'''),
        md("""## 8. Reconstruction-error indicator

The saved threshold is the 95th percentile of authentic validation reconstruction errors. A value above it can be flagged for additional review, but it is **not a tampering probability** and does not confirm manipulation."""),
        code('''threshold=float(checkpoint['anomaly_threshold']); indicators=pd.DataFrame(records); indicators['Reference threshold']=threshold; indicators['Exploratory indicator']=np.where(indicators.MSE>threshold,'Above reference range','Within reference range'); display(indicators[['Class','MSE','Reference threshold','Exploratory indicator']]); print('Checkpoint ROC-AUC:',checkpoint['roc_auc'],'— limited separation')'''),
        md("## 9. Reconstruction difference maps"),
        code('''fig,axes=plt.subplots(2,3,figsize=(13,8))
for ax,i in zip(axes.flat,range(6)):
 difference=torch.mean(torch.abs(images[i]-reconstructed[i]),dim=0).cpu(); ax.imshow(difference,cmap='hot'); ax.set_title(('Authentic' if labels[i]==0 else 'Tampered')+' difference'); ax.axis('off')
plt.suptitle('Pixel reconstruction differences — not tampering localization'); plt.tight_layout(); plt.show()'''),
        md("""## 10. Patch-attention visualization

Attention describes relationships learned between patches. It must not be interpreted as definitive tampering localization."""),
        code('''index=0; matrix=attention_maps[-1][index].cpu().numpy(); patch_attention=matrix.mean(axis=0).reshape(8,8); patch_attention=(patch_attention-patch_attention.min())/(np.ptp(patch_attention)+1e-8); original=images[index].cpu().permute(1,2,0).numpy(); overlay=np.array(Image.fromarray((patch_attention*255).astype(np.uint8)).resize((128,128),Image.Resampling.BILINEAR))/255
fig,axes=plt.subplots(1,3,figsize=(14,4)); axes[0].imshow(original); axes[0].set_title('Original'); axes[1].imshow(patch_attention,cmap='viridis'); axes[1].set_title('8×8 patch attention'); axes[2].imshow(original); axes[2].imshow(overlay,cmap='jet',alpha=.45); axes[2].set_title('Attention overlay'); [ax.axis('off') for ax in axes]; plt.tight_layout(); plt.show()'''),
        md("## 11. Faculty summary"),
        code('''print('Transformer V2 checkpoint:',CHECKPOINT)
print('Epochs:',checkpoint['epoch']); print('Final training MSE:',checkpoint['train_mse']); print('Best validation MSE:',checkpoint['validation_mse']); print('Source validation PSNR/SSIM: 27.5793 dB / 0.8498'); print('Reference threshold:',checkpoint['anomaly_threshold']); print('Source reconstruction-error ROC-AUC:',checkpoint['roc_auc']); print('Conclusion: useful reconstruction and attention analysis, but reconstruction error alone is not reliable tampering detection.')'''),
        md("""## 12. Conclusion and limitations

- Transformer V2 reconstructs images using full spatial patch tokens.
- Self-attention provides an interpretable patch-relationship visualization.
- The saved ROC-AUC is close to random ranking, so forensic separation is limited.
- Attention and bright difference-map regions do not prove localization of manipulation.
- CASIA is the primary dataset and cross-dataset generalization remains limited.
- This is an academic research prototype, not a production or legal forensic decision system."""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Created {OUTPUT.relative_to(ROOT)} with {len(cells)} cells")


if __name__ == "__main__":
    extract_history()
    build()
