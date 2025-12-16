"""
model.py

This module defines a CNN-based perceptual embedding model (ResNet/ConvNeXt)
with an optional small residual MLP head. It mirrors the ViT perceptual path
where possible.

Main tasks:
  - Build a torchvision backbone that outputs pooled image embeddings
  - Optionally load local ImageNet checkpoints, else fallback to torchvision weights
  - Optionally apply a residual MLP head on top of embeddings
  - Provide a preprocessing function (resize → to tensor → normalize)
  - Expose an 'embed' API and a forward pass returning cosine distance
"""

import logging
from pathlib import Path
from typing import Tuple, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models as tvm, transforms

# Defaults
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD  = [0.229, 0.224, 0.225]

# Pooled embedding dims
EMBED_DIMS_CNN = {
  'resnet18': 512,
  'resnet50': 2048,
  'convnext_tiny': 768,
}

# Expected checkpoint filenames in load_dir
CKPT_FILENAMES = {
  'resnet18':      'resnet18_imagenet.pth',
  'resnet50':      'resnet50_imagenet.pth',
  'convnext_tiny': 'convnext_tiny_imagenet.pth',
}


# Small MLP head
class MLP(nn.Module):
  def __init__(self, in_features: int, hidden_size: int = 512):
    super().__init__()
    self.fc1 = nn.Linear(in_features, hidden_size, bias=True)
    self.fc2 = nn.Linear(hidden_size, in_features, bias=True)

  def forward(self, x):
    return self.fc2(F.relu(self.fc1(x))) + x


def _build_backbone(model_type: str) -> nn.Module:
  """
  Returns a CNN backbone that outputs pooled features:
    - ResNet: replace .fc with Identity (so forward -> [B, D])
    - ConvNeXt-Tiny: flatten classifier input (GAP output) and return [B, D]
  """
  logging.info(f"Building CNN backbone: {model_type}")
  if model_type == 'resnet18':
    m = tvm.resnet18(weights=None)
    m.fc = nn.Identity()
    return m
  elif model_type == 'resnet50':
    m = tvm.resnet50(weights=None)
    m.fc = nn.Identity()
    return m
  elif model_type == 'convnext_tiny':
    m = tvm.convnext_tiny(weights=None)

    new_cls = nn.Sequential(
        m.classifier[0],     # LayerNorm2d
        nn.Flatten(1),       # produce [B, D]
        # no Linear here
    )
    m.classifier = new_cls
    return m
  else:
    logging.error(f"Unsupported CNN model_type: {model_type}")
    raise ValueError(f"Unsupported CNN model_type: {model_type}")


def _load_weights_if_available(model: nn.Module, model_type: str, load_dir: Union[str, Path]) -> bool:
  """
  Try to load state_dict from load_dir / CKPT_FILENAMES[model_type].
  Returns True if loaded, False otherwise.
  """
  ckpt_name = CKPT_FILENAMES[model_type]
  ckpt_path = Path(load_dir) / ckpt_name
  if ckpt_path.exists():
    sd = torch.load(str(ckpt_path), map_location='cpu')
    model.load_state_dict(sd, strict=False)
    logging.info(f"Loaded local checkpoint for {model_type} from {ckpt_path}")
    return True
  else:
    logging.warning(f"No local checkpoint found for {model_type} at {ckpt_path}")
    return False


def _fallback_load_torchvision_weights(model: nn.Module, model_type: str):
  """
  If no local weights, pull torchvision ImageNet-1K weights.
  """
  if model_type == 'resnet18':
    w = tvm.ResNet18_Weights.IMAGENET1K_V1
    ref = tvm.resnet18(weights=w); ref.fc = nn.Identity()
  elif model_type == 'resnet50':
    w = tvm.ResNet50_Weights.IMAGENET1K_V1
    ref = tvm.resnet50(weights=w); ref.fc = nn.Identity()
  elif model_type == 'convnext_tiny':
    w = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    ref = tvm.convnext_tiny(weights=w)
    ref.classifier = nn.Sequential(ref.classifier[0], nn.Flatten(1))
  else:
    logging.error(f"Unsupported CNN model_type: {model_type}")
    raise ValueError(model_type)
  model.load_state_dict(ref.state_dict(), strict=True)


def normalize_embedding(embed: torch.Tensor) -> torch.Tensor:
  # zero-mean per sample, then L2 normalize per row
  embed = embed - embed.mean(dim=1, keepdim=True)
  denom = embed.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
  return embed / denom


# CNN Perceptual Model
class CNNPerceptualModel(nn.Module):
  """
  Vanilla / MLP CNN model that outputs cosine distances between embeddings,
  matching the ViT PerceptualModel interface where possible.

  Notes:
    - feat_type is fixed to global-average-pooled "gap" features.
    - normalization is expected in the *preprocess function* returned by the factory.
  """
  def __init__(
              self,
              model_type: str,
              hidden_size: int,
              mlp: bool,
              normalize_embeds: bool,
              device: str,
              load_dir: Optional[Union[str, Path]] = None,
              **kwargs
              ):
    super().__init__()

    if model_type not in EMBED_DIMS_CNN:
      raise ValueError(f"Unsupported model_type={model_type}. "
                        f"Choose from {list(EMBED_DIMS_CNN.keys())}")

    self.model_type = model_type
    self.hidden_size = hidden_size
    self.mlp = mlp
    self.normalize_embeds = normalize_embeds
    self.device = torch.device(device)

    logging.info(f"Initializing CNNPerceptualModel with {model_type}, mlp={mlp}")

    # backbone
    self.backbone = _build_backbone(model_type).to(self.device)

    loaded = False
    if load_dir is not None:
      loaded = _load_weights_if_available(self.backbone, model_type, load_dir)
    if not loaded:
      _fallback_load_torchvision_weights(self.backbone, model_type)

    self.backbone.eval().requires_grad_(False)


    self.embed_dim = EMBED_DIMS_CNN[model_type]

    # head
    if self.mlp:
      logging.info(f"Adding residual MLP head with hidden size {hidden_size}")
      self.head = MLP(in_features=self.embed_dim, hidden_size=self.hidden_size)
    else:
      self.head = nn.Identity()

    # for safety ensure only head is trainable (vanilla=Identity -> trains nothing)
    self.head.requires_grad_(not isinstance(self.head, nn.Identity))

  @torch.no_grad()
  def _backbone_embed(self, x: torch.Tensor) -> torch.Tensor:
    """
    x is expected preprocessed (normalized) already: [B,3,224,224]
    Returns [B, D]
    """
    return self.backbone(x)

  def embed(self, img: torch.Tensor) -> torch.Tensor:
    """
    Returns [B, D] embedding. (No image normalization here; handled by preprocess.)
    """
    with torch.no_grad():
      feats = self._backbone_embed(img.to(self.device))
    embed = self.head(feats)
    if embed.dim() == 1:
      embed = embed.unsqueeze(0)
    if self.normalize_embeds:
      embed = normalize_embedding(embed)
    return embed

  def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
    """
    Cosine distance between two images' embeddings. Higher = more different.
    """
    ea = self.embed(img_a)
    eb = self.embed(img_b)
    return 1 - F.cosine_similarity(ea, eb, dim=-1)


def create_backbone_cnn(
                        model_type: str = 'resnet18',
                        hidden_size: int = 512,
                        mlp: bool = True,
                        normalize_embeds: bool = True,
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                        project_root: str = '/content/drive/MyDrive/perceptual-vits-fashion-forecasting'
                        ) -> Tuple[nn.Module, Callable]:
  """Create a backbone cnn model and preprocess function.

  Args:
    model_type: str, either resnet18, resnet50 or convnext_tiny
    hidden_size: int, size of mlp head
    mlp: bool, whether to add a trainable mlp head
    normalize_embeds: bool, whether to normalize embeddings
    device: str, either 'cuda' or 'cpu'
    project_root: str, root of the project

  Returns:
  model: nn.Module, backbone cnn model
  preprocess: Callable, preprocessing function
  """
  logging.info(f'Creating backbone CNN for {model_type}')

  PROJECT_ROOT = Path(project_root)
  BACKBONE_DIR = PROJECT_ROOT / 'vision_models' / 'cnn_backbone_models'

  # model
  model = CNNPerceptualModel(
    model_type=model_type,
    hidden_size=hidden_size,
    mlp=mlp,
    normalize_embeds=normalize_embeds,
    device=device,
    load_dir=BACKBONE_DIR
  )

  model.eval()
  model.backbone.requires_grad_(False)
  model.head.requires_grad_(mlp)

  # preprocessing
  t = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
  ])

  def preprocess(pil_img):
    pil_img = pil_img.convert('RGB')
    return t(pil_img)

  return model, preprocess
