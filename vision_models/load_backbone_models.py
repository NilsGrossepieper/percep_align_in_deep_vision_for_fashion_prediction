"""
This script loads the backbone deep vision models and saves them in the corresponding folders
"""
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Paths & imports
import os, sys, shutil, torch
from pathlib import Path

# Make your project src importable
sys.path.append('/content/drive/MyDrive/perceptual-vits-fashion-forecasting/src')

try:
    from project_root.path import PROJECT_ROOT as _PR
    PROJECT_ROOT = Path(_PR)
except Exception:
    PROJECT_ROOT = Path('/content/drive/MyDrive/perceptual-vits-fashion-forecasting')

REQ_DIR  = PROJECT_ROOT / 'src' / 'vision_models' / 'dreamsim' / 'requirements.txt'
VITS_DIR = PROJECT_ROOT / 'vision_models' / 'vits_backbone_models'
CNNS_DIR = PROJECT_ROOT / 'vision_models' / 'cnn_backbone_models'

# Install requirements
!pip -q install -r "{REQ_DIR}"
!pip -q install torchvision timm torch torchvision transformers

# Import after installs
from vision_models.dreamsim.dreamsim.model import download_weights
import torchvision.models as tvm
import timm

# Download helpers
def _safe_del(path: Path):
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

def download_vits(vits_dir: Path):
    vits_dir.mkdir(parents=True, exist_ok=True)

    # DreamSim bundles (formats your loader expects)
    download_weights(str(vits_dir), dreamsim_type='ensemble')       # dino_b16, clip_b16, openclip_b16
    download_weights(str(vits_dir), dreamsim_type='clip_vitb32')    # CLIP-B/32
    download_weights(str(vits_dir), dreamsim_type='open_clip_vitb32')  # OpenCLIP-B/32

    # Additional DINO models
    print("\n[+] Downloading DINOv2 ViT-B/14 ...")
    dino_v2 = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
    torch.save(dino_v2.state_dict(), vits_dir / "dinov2_vitb14_pretrain.pth")
    del dino_v2
    print("    [OK] DINOv2 ViT-B/14 →", vits_dir / "dinov2_vitb14_pretrain.pth")

    print("\n[+] Downloading DINOv3 ViT-B/16 ...")
    dino_v3 = timm.create_model('vit_base_patch16_clip_224.laion2b_ft_in12k_in1k', pretrained=True)
    torch.save(dino_v3.state_dict(), vits_dir / "dinov3_vitb16_pretrain.pth")
    del dino_v3
    print("    [OK] DINOv3 ViT-B/16 →", vits_dir / "dinov3_vitb16_pretrain.pth")

    # Clean up LoRA adapter folders/archives (you only need base backbones here)
    to_delete = [
        vits_dir / "ensemble_lora",
        vits_dir / "dino_vitb16_single_lora",
        vits_dir / "open_clip_vitb32_single_lora",
        vits_dir / "clip_vitb32_single_lora",
        vits_dir / "dino_vitb16_patch_lora",
        vits_dir / "dinov2_vitb14_single_lora",
        vits_dir / "dinov2_vitb14_patch_lora",
        vits_dir / "synclr_vitb16_single_lora",
        vits_dir / "pretrained.zip",
    ]
    for p in to_delete:
        _safe_del(p)

    print("\n[VITS READY]", vits_dir)
    for name in [
        'dino_vitb16_pretrain.pth',
        'clip_vitb16_pretrain.pth.tar',
        'open_clip_vitb16_pretrain.pth.tar',
        'clip_vitb32_pretrain.pth.tar',
        'open_clip_vitb32_pretrain.pth.tar',
        'dinov2_vitb14_pretrain.pth',
        'dinov3_vitb16_pretrain.pth',
    ]:
        path = vits_dir / name
        print(" -", path, "✓" if path.exists() else "MISSING")

def download_cnn_backbones(cnns_dir: Path):
    cnns_dir.mkdir(parents=True, exist_ok=True)
    r18 = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    torch.save(r18.state_dict(), cnns_dir / "resnet18_imagenet.pth")
    print("[OK] ResNet18 →", cnns_dir / "resnet18_imagenet.pth")

    r50 = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    torch.save(r50.state_dict(), cnns_dir / "resnet50_imagenet.pth")
    print("[OK] ResNet50 →", cnns_dir / "resnet50_imagenet.pth")

    cxt = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    torch.save(cxt.state_dict(), cnns_dir / "convnext_tiny_imagenet.pth")
    print("[OK] ConvNeXt-Tiny →", cnns_dir / "convnext_tiny_imagenet.pth")
    print("\n[CNNS READY]", cnns_dir)

# Run all downloads
download_vits(VITS_DIR)
download_cnn_backbones(CNNS_DIR)
