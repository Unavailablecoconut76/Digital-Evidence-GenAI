"""Generate the three load-only faculty demonstration notebooks."""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}

def py(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}

def save(path: Path, cells: list[dict]) -> None:
    payload = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Created", path.relative_to(ROOT), "cells:", len(cells))

SETUP = '''from pathlib import Path
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
PROJECT_ROOT=next((p.resolve() for p in candidates if (p/'src').is_dir() and (p/'checkpoints').is_dir()),None)
if PROJECT_ROOT is None: raise FileNotFoundError('Run inside the project repository.')
sys.path.insert(0,str(PROJECT_ROOT/'src'))
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Project:',PROJECT_ROOT)
print('Device:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('LOAD-ONLY DEMO: models will not be retrained.')'''

SPLITS = '''frames={name:pd.read_csv(PROJECT_ROOT/'data'/'splits'/f'{name}.csv') for name in ('train','validation','test')}
rows=[]
for name,frame in frames.items(): rows.append({'split':name.title(),'total':len(frame),'authentic':int((frame.label==0).sum()),'tampered':int((frame.label==1).sum())})
display(pd.DataFrame(rows)); print('Total:',sum(r['total'] for r in rows),'| seed: 42')
print('Masks are excluded; labels are metadata, not reconstruction targets.')'''

def extract_vae_history() -> None:
    destination=ROOT/'results'/'vae_v5_final_training_history.csv'
    source=ROOT/'bin'/'notebooks'/'VAE_experiments'/'VAE_V5_External_Tampered_vs_Authentic.ipynb'
    if destination.exists(): return
    if not source.exists(): raise FileNotFoundError('Archived VAE training notebook is required once to extract its verified history.')
    nb=json.loads(source.read_text(encoding='utf-8'))
    text='\n'.join(''.join(o.get('text',[])) for o in nb['cells'][32].get('outputs',[]))
    pattern=re.compile(r'Epoch\s+(\d+)\s+\|\s+Train MSE:\s+([0-9.]+)\s+\|\s+Val MSE:\s+([0-9.]+)\s+\|\s+PSNR:\s+([0-9.]+) dB\s+\|\s+Val KL:\s+([0-9.]+)\s+\|\s+β:\s+([0-9.]+)\s+\|\s+LR:\s+([0-9.eE+-]+)')
    rows=[dict(epoch=int(e),train_mse=float(tm),validation_mse=float(vm),validation_psnr=float(ps),validation_kl=float(kl),beta=float(b),learning_rate=float(lr)) for e,tm,vm,ps,kl,b,lr in pattern.findall(text)]
    if len(rows)!=80: raise RuntimeError(f'Expected 80 VAE epochs, found {len(rows)}')
    with destination.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)

def ae_notebook() -> list[dict]:
    return [
        md('# Autoencoder — Complete CASIA Faculty Demo\n\nLoads trained AE artifacts and shows architecture, training, evaluation, reconstruction, class-wise exploration, and denoising. No retraining; no forgery classification.'),
        py(SETUP), md('## 1. Dataset and fixed splits\n\nRGB → 128×128 → tensor `[0,1]`.'), py(SPLITS),
        md('## 2. Architecture and checkpoint\n\n`3×128×128 → 32×8×8 → 3×128×128`; 24× compression; Sigmoid output.'),
        py('''from autoencoder import ConvolutionalAutoencoder
CHECKPOINT=PROJECT_ROOT/'checkpoints'/'best_autoencoder.pth'
ckpt=torch.load(CHECKPOINT,map_location=DEVICE,weights_only=True)
model=ConvolutionalAutoencoder().to(DEVICE); model.load_state_dict(ckpt['model_state_dict'],strict=True); model.eval()
print('Checkpoint:',CHECKPOINT); print('Best epoch:',ckpt.get('epoch')); print('Validation MSE:',ckpt.get('validation_loss')); print('Config:',ckpt.get('config')); print('Parameters:',f'{sum(p.numel() for p in model.parameters()):,}')'''),
        md('## 3. Training losses'),
        py('''history=pd.read_csv(PROJECT_ROOT/'results'/'ae_training_history.csv'); display(history.tail())
history.plot(x='epoch',y=['train_loss','validation_loss'],figsize=(10,5),grid=True,title='AE training and validation MSE'); plt.ylabel('MSE'); plt.show()
print('Epochs:',len(history),'| First/final train:',history.train_loss.iloc[0],history.train_loss.iloc[-1],'| Best validation:',history.validation_loss.min())'''),
        md('## 4. Complete test metrics'),
        py('''metrics=json.loads((PROJECT_ROOT/'results'/'ae_test_metrics.json').read_text())
counts={'Overall':metrics['number_of_test_images'],'Authentic':metrics['number_authentic'],'Tampered':metrics['number_tampered']}
display(pd.DataFrame([{'group':name,'count':counts[name],**metrics[key]} for name,key in [('Overall','overall'),('Authentic','authentic'),('Tampered','tampered')]])[['group','count','mse_mean','mse_std','psnr_mean','psnr_std','ssim_mean','ssim_std']])'''),
        md('## 5. Fresh original vs reconstructed examples'),
        py('''from ae_dataset import AutoencoderImageDataset
from torch.utils.data import DataLoader
batch=next(iter(DataLoader(AutoencoderImageDataset(PROJECT_ROOT/'data'/'splits'/'test.csv',128),batch_size=6,shuffle=False,num_workers=0))); x=batch['image'].to(DEVICE)
with torch.no_grad(): y=model(x)
fig,axes=plt.subplots(6,2,figsize=(7,18))
for i in range(6):
 label='Authentic' if int(batch['label'][i])==0 else 'Tampered'; axes[i,0].imshow(x[i].cpu().permute(1,2,0)); axes[i,0].set_title('Original — '+label); axes[i,1].imshow(y[i].cpu().permute(1,2,0).clamp(0,1)); axes[i,1].set_title('Reconstructed')
 for ax in axes[i]: ax.axis('off')
plt.tight_layout(); plt.show(); print('Batch shape:',tuple(x.shape),'| MSE:',torch.mean((y-x)**2).item())'''),
        md('## 6. Authentic vs tampered exploration\n\nDifferences are exploratory and do not make AE a tampering detector.'),
        py('''per=pd.read_csv(PROJECT_ROOT/'results'/'ae_test_per_image_metrics.csv'); display(per.groupby('class_name')[['mse','psnr','ssim']].agg(['mean','std']))
for name,g in per.groupby('class_name'): plt.hist(g.mse,bins=40,alpha=.55,label=name)
plt.title('AE reconstruction-MSE distributions'); plt.legend(); plt.grid(alpha=.2); plt.show()'''),
        md('## 7. Denoising AE results'),
        py('''dae=json.loads((PROJECT_ROOT/'results'/'dae_test_metrics.json').read_text()); display(pd.DataFrame([dae]).T.rename(columns={0:'value'}))
for rel in ('outputs/ae/denoising_comparison_grid.png','outputs/ae/dae_training_curve.png'):
 path=PROJECT_ROOT/rel
 if path.is_file(): display(Image.open(path))'''),
        md('## Conclusion\n\nThe AE is the deterministic reconstruction and compression module. Its reconstruction statistics are not proof of tampering.')]

def vae_notebook() -> list[dict]:
    return [
        md('# VAE V5 Final — Complete CASIA Faculty Demo\n\nLoads `VAE_V5_FINAL.pth` and shows architecture, verified 80-epoch training, hybrid reconstruction/KL losses, canonical test results, reconstructions, exploratory ROC analysis, and synthetic sampling. No retraining.'),
        py(SETUP), md('## 1. Dataset and fixed splits\n\nThe final checkpoint used mixed CASIA reconstruction training. Labels remain analysis metadata.'), py(SPLITS),
        md('## 2. Architecture and checkpoint\n\nResidual/GroupNorm/SiLU encoder, 256-dimensional `mu` and `logvar`, reparameterization `z=mu+sigma×epsilon`, skip-connected decoder, Sigmoid RGB output.'),
        py('''from vae import VAEV5
CHECKPOINT=PROJECT_ROOT/'checkpoints'/'VAE_V5_FINAL.pth'; ckpt=torch.load(CHECKPOINT,map_location=DEVICE,weights_only=True)
model=VAEV5(int(ckpt.get('latent_dim',256))).to(DEVICE); model.load_state_dict(ckpt['model_state_dict'],strict=True); model.eval()
print('Checkpoint:',CHECKPOINT); print('Epoch:',ckpt.get('epoch')); print('Validation MSE/PSNR/KL:',ckpt.get('val_mse'),ckpt.get('val_psnr'),ckpt.get('val_kl')); print('Beta/L1 weight:',ckpt.get('beta'),ckpt.get('l1_weight')); print('Parameters:',f'{sum(p.numel() for p in model.parameters()):,}')'''),
        md('## 3. Loss and KL warm-up\n\n`Reconstruction = 0.5×MSE + 0.5×L1`; `Total = Reconstruction + beta×KL`. Beta increases from 0.00005 to 0.00030 over 20 epochs. Lower KL does not automatically mean better reconstruction.'),
        md('## 4. Verified training history'),
        py('''history=pd.read_csv(PROJECT_ROOT/'results'/'vae_v5_final_training_history.csv'); display(history.tail())
fig,axes=plt.subplots(2,2,figsize=(13,8)); history.plot(x='epoch',y=['train_mse','validation_mse'],ax=axes[0,0],grid=True,title='MSE'); history.plot(x='epoch',y='validation_psnr',ax=axes[0,1],grid=True,title='Validation PSNR'); history.plot(x='epoch',y='validation_kl',ax=axes[1,0],grid=True,title='Validation KL'); history.plot(x='epoch',y='beta',ax=axes[1,1],grid=True,title='Beta warm-up'); plt.tight_layout(); plt.show()
best=history.loc[history.validation_mse.idxmin()]; print('Epochs:',len(history),'| Best epoch:',int(best.epoch),'| Best MSE:',best.validation_mse,'| PSNR:',best.validation_psnr)'''),
        md('## 5. Complete canonical test metrics'),
        py('''metrics=json.loads((PROJECT_ROOT/'results'/'vae_v5_final_test_metrics.json').read_text()); display(pd.DataFrame([{'group':'Overall',**metrics['overall']},{'group':'Authentic',**metrics['authentic']},{'group':'Tampered',**metrics['tampered']}])[['group','count','mse_mean','mse_std','psnr_mean','ssim_mean','kl_loss_mean','total_loss_mean']]); print('ROC-AUC:',metrics['roc_auc']); print('FID:',metrics['fid'] if metrics['fid'] is not None else 'N/A')'''),
        md('## 6. Fresh reconstruction examples'),
        py('''from ae_dataset import AutoencoderImageDataset
from torch.utils.data import DataLoader
batch=next(iter(DataLoader(AutoencoderImageDataset(PROJECT_ROOT/'data'/'splits'/'test.csv',128),batch_size=6,shuffle=False,num_workers=0))); x=batch['image'].to(DEVICE)
with torch.no_grad(): y,mu,logvar=model.reconstruct(x,deterministic=True)
fig,axes=plt.subplots(6,2,figsize=(7,18))
for i in range(6):
 label='Authentic' if int(batch['label'][i])==0 else 'Tampered'; axes[i,0].imshow(x[i].cpu().permute(1,2,0)); axes[i,0].set_title('Original — '+label); axes[i,1].imshow(y[i].cpu().permute(1,2,0).clamp(0,1)); axes[i,1].set_title('VAE V5 reconstruction')
 for ax in axes[i]: ax.axis('off')
plt.tight_layout(); plt.show(); print('Input/mu/logvar/output:',tuple(x.shape),tuple(mu.shape),tuple(logvar.shape),tuple(y.shape))'''),
        md('## 7. Forensic reconstruction analysis\n\nROC-AUC is exploratory—not a calibrated tampering probability or standalone detector.'),
        py('''per=pd.read_csv(PROJECT_ROOT/'results'/'vae_v5_final_test_per_image_metrics.csv'); display(per.groupby('class_name')[['mse','psnr','ssim']].agg(['mean','median','std']))
fig,axes=plt.subplots(1,2,figsize=(12,4))
for name,g in per.groupby('class_name'): axes[0].hist(g.mse,bins=45,alpha=.55,label=name); axes[1].hist(1-g.ssim,bins=45,alpha=.55,label=name)
axes[0].set_title('MSE'); axes[1].set_title('SSIM error'); [ax.legend() for ax in axes]; plt.show()'''),
        md('## 8. Synthetic latent samples\n\nSynthetic research outputs—not genuine forensic evidence. Prior-only generation may be weak because the decoder learned strong image skip connections.'),
        py('''with torch.no_grad(): samples=model.decode(torch.randn(16,256,device=DEVICE))
fig,axes=plt.subplots(4,4,figsize=(8,8))
for ax,img in zip(axes.flat,samples.cpu()): ax.imshow(img.permute(1,2,0).clamp(0,1)); ax.axis('off')
plt.suptitle('Synthetic VAE V5 research outputs'); plt.tight_layout(); plt.show()'''),
        md('## 9. Saved full-test plots'),
        py('''for rel in ('outputs/vae_v5_final/reconstruction_grid.png','outputs/vae_v5_final/authentic_vs_tampered_error.png','outputs/vae_v5_final/roc_curves.png'):
 path=PROJECT_ROOT/rel
 if path.is_file(): print(rel); display(Image.open(path))'''),
        md('## Conclusion\n\nVAE V5 Final gives strong reconstruction but weak MSE-based authentic/tampered separation. It is a reconstruction and exploratory anomaly-analysis module, not a tampering detector.')]

def gan_notebook() -> list[dict]:
    return [
        md('# DCGAN — Complete CASIA Faculty Demo\n\nLoads the trained generator/discriminator, plots adversarial losses and epoch progression, reports FID/IS, and produces fresh synthetic samples. No retraining.'),
        py(SETUP), md('## 1. Dataset and preprocessing\n\nRGB 64×64, normalized to `[-1,1]`; unconditional training without class labels.'), py(SPLITS),
        md('## 2. Architecture and checkpoints\n\nGenerator: `100×1×1 → 512×4×4 → 256×8×8 → 128×16×16 → 64×32×32 → 3×64×64`. Discriminator reverses the progression and returns logits.'),
        py('''from gan_inference import GANInference
G=PROJECT_ROOT/'checkpoints'/'best_generator.pth'; D=PROJECT_ROOT/'checkpoints'/'best_discriminator.pth'; gan=GANInference(G,D,DEVICE,100)
print('Generator:',G,'| Parameters:',f'{sum(p.numel() for p in gan.generator.parameters()):,}'); print('Discriminator:',D,'| Parameters:',f'{sum(p.numel() for p in gan.discriminator.parameters()):,}'); print('Latent dimension: 100')'''),
        md('## 3. Training configuration\n\n30 epochs; batch 64; Adam 0.0002; betas `(0.5,0.999)`; BCEWithLogitsLoss; DCGAN initialization; fixed noise for progress.'),
        md('## 4. Adversarial training losses\n\nG and D losses have different objectives and must not be compared as if they were the same score.'),
        py('''history=pd.read_csv(PROJECT_ROOT/'results'/'gan_training_history.csv'); display(history.tail()); history.plot(x='epoch',y=['generator_loss','discriminator_loss'],figsize=(10,5),grid=True,title='DCGAN adversarial losses'); plt.show(); print('Epochs:',len(history),'| Final G/D:',history.generator_loss.iloc[-1],history.discriminator_loss.iloc[-1],'| Time:',history.epoch_time_seconds.sum(),'s')'''),
        md('## 5. Generated progress every five epochs'),
        py('''paths=[PROJECT_ROOT/'outputs'/'gan'/f'generated_epoch_{e:02d}.png' for e in (5,10,15,20,25,30)]; fig,axes=plt.subplots(2,3,figsize=(15,9))
for ax,path in zip(axes.flat,paths):
 if path.is_file(): ax.imshow(Image.open(path)); ax.set_title(path.stem.replace('_',' ').title())
 ax.axis('off')
plt.tight_layout(); plt.show()'''),
        md('## 6. Fresh synthetic generation\n\nSynthetic research outputs—not genuine forensic evidence.'),
        py('''samples=gan.generate(8); fig,axes=plt.subplots(2,4,figsize=(10,5))
for ax,img in zip(axes.flat,samples): ax.imshow(img); ax.axis('off')
plt.suptitle('Synthetic DCGAN research outputs'); plt.tight_layout(); plt.show()'''),
        md('## 7. FID and Inception Score'),
        py('''metrics=json.loads((PROJECT_ROOT/'results'/'gan_test_metrics.json').read_text()); display(pd.DataFrame([metrics]).T.rename(columns={0:'value'})); print('Lower FID is generally better. IS uses ImageNet features and is not domain-specific forensic validation.')'''),
        md('## 8. Final saved grid'), py("path=PROJECT_ROOT/'outputs'/'gan'/'final_generated_samples.png'\nif path.is_file(): display(Image.open(path))"),
        md('## Conclusion\n\nDCGAN is the dedicated synthetic-generation module. Its outputs remain experimental and FID/IS do not establish forensic usefulness.')]

def main() -> None:
    extract_vae_history()
    save(ROOT/'notebooks'/'AE'/'Autoencoder_CASIA_Complete_Demo.ipynb',ae_notebook())
    save(ROOT/'notebooks'/'VAE'/'VAE_CASIA_Complete_Demo.ipynb',vae_notebook())
    save(ROOT/'notebooks'/'GAN'/'GAN_CASIA_Complete_Demo.ipynb',gan_notebook())

if __name__=='__main__': main()
