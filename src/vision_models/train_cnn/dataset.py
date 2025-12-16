"""
dataset.py

This module defines the dataset class for the nights dataset, for NIGHTS,
fashiontriplets, NIGHTS+fashion triplets plus for synthetic fashion data.

Main tasks:
  - Creates the dataset for NIGHTS
  - Creates the dataset for NIGHTS+fashion triplets
  - Creates the dataset for NIGHTS+fashion triplets
  - Creates the dataset for synthetic fashion data
"""

import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable, Optional

# preprocessing
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD  = [0.229, 0.224, 0.225]

def preprocess(pil_img, load_size):
  """
  Preprocess the images for the CNN models
  """
  t = transforms.Compose([
    transforms.Resize((load_size, load_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
  ])
  pil_img = pil_img.convert('RGB')
  return t(pil_img)


# Dataset for nights
class NightsDataset(Dataset):
  def __init__(
            self,
            root_dir: str,
            random_seed: int,
            second_root_dir: str = None,
            split: str = 'train',
            load_size: int = 224,
            preprocessing_function: Optional[Callable] = preprocess,
            **kwargs
              ):

    self.root_dir = root_dir
    self.csv = pd.read_csv(os.path.join(self.root_dir, 'data.csv'))
    self.csv = self.csv[self.csv['votes'] >= 6] # Filter out triplets with less than 6 unanimous votes
    self.seed = random_seed
    self.second_root_dir = second_root_dir
    self.split = split
    self.load_size = load_size
    self.preprocessing_function = preprocessing_function

    if self.split == 'train' or self.split == 'val' or self.split == 'test':
      self.csv = self.csv[self.csv['split'] == split]
    elif split == 'test_imagenet':
      self.csv = self.csv[self.csv['split'] == 'test']
      self.csv = self.csv[self.csv['is_imagenet'] == True]
    elif split == 'test_no_imagenet':
      self.csv = self.csv[self.csv['split'] == 'test']
      self.csv = self.csv[self.csv['is_imagenet'] == False]
    else:
      raise ValueError(f"Invalid split: {split}")

    self.csv['ref_abs']   = self.csv['ref_path'].apply(lambda p: os.path.join(self.root_dir, p))
    self.csv['left_abs']  = self.csv['left_path'].apply(lambda p: os.path.join(self.root_dir, p))
    self.csv['right_abs'] = self.csv['right_path'].apply(lambda p: os.path.join(self.root_dir, p))
    self.csv['right_vote'] = self.csv['right_vote'].astype(np.float32)
    self.csv = self.csv.reset_index(drop=True)

  def __len__(self):
    return len(self.csv)

  def __getitem__(self, idx):
    row = self.csv.iloc[idx]
    p = row['right_vote']
    with Image.open(row['ref_abs']) as im_ref:
      img_ref = self.preprocessing_function(im_ref, self.load_size)
    with Image.open(row['left_abs']) as im_left:
      img_left = self.preprocessing_function(im_left, self.load_size)
    with Image.open(row['right_abs']) as im_right:
      img_right = self.preprocessing_function(im_right, self.load_size)
    return img_ref, img_left, img_right, p, row['id']


# Dataset for fashion triplets
class FashionTripletsDataset(Dataset):
  def __init__(
            self,
            root_dir: str,
            random_seed: int,
            second_root_dir: str = None,
            split: str = 'train',
            load_size: int = 224,
            preprocessing_function: Optional[Callable] = preprocess,
            **kwargs
              ):

    self.root_dir = root_dir
    self.seed = random_seed
    self.csv = pd.read_csv(os.path.join(self.root_dir, f"data_clean_{self.seed}.csv"))
    self.second_root_dir = second_root_dir
    self.split = split
    self.load_size = load_size
    self.preprocessing_function = preprocessing_function

    if self.split == 'train' or self.split == 'val' or self.split == 'test':
      self.csv = self.csv[self.csv['split'] == split]
    else:
      raise ValueError(f"Invalid split: {split}")

    self.csv['ref_abs']   = self.csv['ref_path'].apply(lambda p: os.path.join(self.root_dir, p))
    self.csv['left_abs']  = self.csv['left_path'].apply(lambda p: os.path.join(self.root_dir, p))
    self.csv['right_abs'] = self.csv['right_path'].apply(lambda p: os.path.join(self.root_dir, p))
    self.csv['right_vote'] = self.csv['right_vote'].astype(np.float32)
    self.csv = self.csv.reset_index(drop=True)

  def __len__(self):
    return len(self.csv)

  def __getitem__(self, idx):
    row = self.csv.iloc[idx]
    p = row['right_vote']
    with Image.open(row['ref_abs']) as im_ref:
      img_ref = self.preprocessing_function(im_ref, self.load_size)
    with Image.open(row['left_abs']) as im_left:
      img_left = self.preprocessing_function(im_left, self.load_size)
    with Image.open(row['right_abs']) as im_right:
      img_right = self.preprocessing_function(im_right, self.load_size)
    return img_ref, img_left, img_right, p, row['id']


# Dataset for nights + fashion triplets
class NightsFashionTripletsDataset(Dataset):
  def __init__(
            self,
            root_dir: str,
            random_seed: int,
            second_root_dir: str,
            split: str = 'train',
            load_size: int = 224,
            preprocessing_function: Optional[Callable] = preprocess,
            **kwargs
              ):

    self.root_dir = root_dir
    self.seed = random_seed
    self.second_root_dir = second_root_dir
    self.split = split
    self.load_size = load_size
    self.preprocessing_function = preprocessing_function

    # Prepare NIGHTS
    self.nights = pd.read_csv(os.path.join(self.root_dir, 'data.csv'))
    self.nights = self.nights[self.nights['votes'] >= 6]
    self.nights = self.nights.drop(columns=['is_imagenet', 'prompt'], errors='ignore')
    self.nights['ref_path']  = self.nights['ref_path'].apply(lambda p: os.path.join(root_dir, p))
    self.nights['left_path'] = self.nights['left_path'].apply(lambda p: os.path.join(root_dir, p))
    self.nights['right_path']= self.nights['right_path'].apply(lambda p: os.path.join(root_dir, p))

    # Prepare fashion triplets
    self.fashion = pd.read_csv(os.path.join(self.second_root_dir, f"data_clean_{self.seed}.csv"))
    self.fashion = self.fashion.drop(columns=['win_rate'], errors='ignore')
    self.fashion['ref_path']  = self.fashion['ref_path'].apply(lambda p: os.path.join(self.second_root_dir, p))
    self.fashion['left_path'] = self.fashion['left_path'].apply(lambda p: os.path.join(self.second_root_dir, p))
    self.fashion['right_path']= self.fashion['right_path'].apply(lambda p: os.path.join(self.second_root_dir, p))

    if self.split == 'train' or self.split == 'val' or self.split == 'test':
      self.nights = self.nights[self.nights['split'] == split]
      self.fashion = self.fashion[self.fashion['split'] == split]
    else:
      raise ValueError(f'Invalid split: {split}')

    # Remove as many NIGHTS as we add fashion triplets
    len_fashion = len(self.fashion)
    self.nights = self.nights.drop(
      self.nights.sample(n=len_fashion, random_state=self.seed).index
    )

    # Combine and shuffle data
    self.csv = pd.concat([self.nights, self.fashion], ignore_index=True)\
      .sample(frac=1, random_state=self.seed)\
      .reset_index(drop=True)
    self.csv['right_vote'] = self.csv['right_vote'].astype(np.float32)
    self.csv = self.csv.reset_index(drop=True)

  def __len__(self):
    return len(self.csv)

  def __getitem__(self, idx):
    row = self.csv.iloc[idx]
    p = row['right_vote']
    with Image.open(row['ref_path']) as im_ref:
      img_ref = self.preprocessing_function(im_ref, self.load_size)
    with Image.open(row['left_path']) as im_left:
      img_left = self.preprocessing_function(im_left, self.load_size)
    with Image.open(row['right_path']) as im_right:
      img_right = self.preprocessing_function(im_right, self.load_size)
    return img_ref, img_left, img_right, p, row['id']
