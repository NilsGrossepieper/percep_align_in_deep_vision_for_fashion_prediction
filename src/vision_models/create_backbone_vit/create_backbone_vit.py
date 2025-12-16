"""
Creates create_backbone_vits.py

This script creates an untrained backbone vision transformer, either based on
a single model (dino_vitb16,clip_vitb16,open_clip_vitb16) or a combined model.
Besides this it also creates the correct preprocessing function for the model.
"""

import logging
from pathlib import Path

import torch
from torchvision import transforms

from vision_models.dreamsim.dreamsim.model import PerceptualModel
from vision_models.dreamsim.util.utils import get_preprocess, get_preprocess_fn


def create_backbone_vit(
    model_type: str,
    feat_type: str,
    stride: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    project_root: str = '/content/drive/MyDrive/perceptual-vits-fashion-forecasting'
  ):
  """Creates a backbone vision transformer and a preprocessing function.

  Args:
    model_type: str, either dino_vitb16, clip_vitb16, open_clip_vitb16
    feat_type: str, eithervcls, embedding or else
    stride: str, either 16 or 32
    device: str, either cuda or cpu
    project_root: str, path to the project root

    Returns:
    model: PerceptualModel, a backbone vision transformer
    preprocess: function, a preprocessing function for the model
  """
  logging.info(f'Creating backbone vision transformer for {model_type}')

  PROJECT_ROOT = Path(project_root)
  BACKBONE_DIR = PROJECT_ROOT / 'vision_models' / 'vits_backbone_models'

  # Build a VANILLA model (no training)
  model = PerceptualModel(
    model_type = model_type,
    feat_type = feat_type,
    stride = stride,
    hidden_size = 1,
    lora = False,
    baseline = True,
    load_dir = str(BACKBONE_DIR),
    normalize_embeds = True,
    device = device
  ).to(device).eval()

  # Proper preprocessing: resize to 224, ToTensor only (no normalization here)
  preprocess_key = get_preprocess(model_type)
  preprocess = get_preprocess_fn(
    preprocess=preprocess_key,
    load_size=224,
    interpolation=transforms.InterpolationMode.BICUBIC
  )

  return model, preprocess
