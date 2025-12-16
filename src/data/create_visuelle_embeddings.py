"""
create_visuelle_embeddings.py

This script computes or loads precomputed image embeddings for the Visuelle2.0 dataset.
Main tasks:
    - Load the list of unique image paths from processed data
    - Load or create embeddings using a DreamSim vision model
    - Handle missing or corrupted image files
    - Save embeddings to a CSV file for downstream use
"""

import logging
import os
from PIL import Image

import pandas as pd
import torch
from tqdm import tqdm

from vision_models.create_backbone_vit.create_backbone_vit import create_backbone_vit
from vision_models.dreamsim.dreamsim.model import PerceptualModel
from vision_models.train_cnn.model import create_backbone_cnn


def create_visuelle_embeddings(
    model_family: str,
    train: bool,
    load_vit: bool,
    model_type: str,
    training_method: str,
    feat_type: str,
    stride: str,
    hidden_size: int,
    dataset_name: str,
    tag: str,
    best: bool,
    project_root: str = '/content/drive/MyDrive/perceptual-vits-fashion-forecasting'
) -> str:
  """Compute or load image embeddings for all unique images in Visuelle2 dataset.

  Args:
      model_family (str): Whether we use a CNN or Vision Transformer (ViT) model.
      train (str): Whether we used a trained vision model or backbone models.
      load_vit (bool): Whether to load a pre-trained Vision Transformer (ViT) model.
      model_type (str): String with the vision models used.
      training_method (str): String with the training method used mlp, lora or vanilla.
      feat_type (str): String with the feature type used.
      stride (int): Integer with the stride used.
      hidden_size (int): Integer with the hidden size used.
      dataset_name (str): String with the dataset name.
      tag (str): String with the tag used.
      best (bool): If true we use best model else last model.
      project_root (str, optional): Root directory of the project.

  Returns:
      str: Path to the saved embeddings CSV file.
  """
  # Output file location
  embeddings_dir = os.path.join(project_root, 'datasets', 'visuelle2', 'embeddings')
  os.makedirs(embeddings_dir, exist_ok=True)

  if model_family == 'vit':
    experiment_name = f"{str(tag)}_{str(model_type)}_{str(training_method)}_{str(feat_type)}_{str(stride)}_{str(dataset_name)}"
  elif model_family =='cnn':
    experiment_name = f"{str(tag)}_{str(model_type)}_{str(training_method)}_{str(dataset_name)}"
  else:
    raise ValueError(f"Unknown model family: {model_family}")
  embedding_filename = f'{experiment_name}_embeddings.csv'
  emb_filepath = os.path.join(embeddings_dir, embedding_filename)

  logging.info(f'Creating embeddings for {experiment_name}')

  # Load existing embeddings
  if os.path.exists(emb_filepath):
    logging.info(f'Loading precomputed embeddings from {emb_filepath}')
    return emb_filepath

  logging.info('No precomputed embeddings found. Creating new embeddings...')

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  if model_family == 'cnn':
    logging.info('Using CNN model')

    # MLP head only if this run was trained with MLP; otherwise backbone/vanilla
    mlp_flag = (training_method == 'mlp') if train else False

    # Build the inference model
    model, preprocess = create_backbone_cnn(
      model_type=model_type,
      hidden_size=hidden_size,
      mlp=mlp_flag,
      normalize_embeds=True,
      device=device,
      project_root=project_root
    )

    if train:
      logging.info('Loading CNN model trained with MLP')
      # choose best/last checkpoint from the training run
      exp_dir = os.path.join(project_root, 'vision_models', 'cnn_training', experiment_name)
      ckpt_path = os.path.join(exp_dir, 'best.ckpt' if best else 'last.ckpt')

      if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
          f"Checkpoint not found: {ckpt_path}. "
          f"Check tag/model_type/training_method/dataset_name or finish training first."
        )

      # Load Lightning checkpoint → extract cnn_model.* weights
      ckpt = torch.load(ckpt_path, map_location=device)
      sd = ckpt.get('state_dict', {})
      stripped = {
        k.replace('cnn_model.', '', 1): v
        for k, v in sd.items() if k.startswith('cnn_model.')
      }
      missing, unexpected = model.load_state_dict(stripped, strict=False)
      if missing:
        logging.warning(f"CNN load_state_dict missing keys: {missing[:6]}{' ...' if len(missing)>6 else ''}")
      if unexpected:
        logging.warning(f"CNN load_state_dict unexpected keys: {unexpected[:6]}{' ...' if len(unexpected)>6 else ''}")
    else:
      logging.info('Using CNN backbone only (no training checkpoint loaded)')

  # Load Vision Transformer
  elif model_family == 'vit':
    logging.info('Using Vision Transformer model')

    # Load the vanilla backbone model
    model, preprocess = create_backbone_vit(
      model_type = model_type,
      feat_type = feat_type,
      stride = stride,
      device = device,
      project_root = project_root
    )

    # Check if we should use a trained model
    if train:
      # Check if we should use a pretrained model from the dreamsim repo
      if load_vit and dataset_name=='nights':
        logging.info('Loading pretrained Vision Transformer model')
        from dreamsim import dreamsim
        model, preprocess = dreamsim(pretrained=True, dreamsim_type=model_type, device=device)

      # Use the self trained models
      else:
        # Load the model trained with mlp
        if training_method == 'mlp':
          logging.info('Loading Vision Transformer model trained with MLP')

          exp_dir = os.path.join(project_root, 'vision_models', 'vits_training', experiment_name)
          ckpt_path = os.path.join(exp_dir, 'best.ckpt' if best else 'last.ckpt')
          if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
              f"Checkpoint not found: {ckpt_path}. "
              f"Check tag/model_type/training_method/dataset_name or finish training first."
            )

          # Load a backbone model
          model = PerceptualModel(
              model_type = model_type,
              feat_type = feat_type,
              stride = stride,
              hidden_size = hidden_size,
              lora = False,
              baseline = False,
              project_root = project_root,
              normalize_embeds = True,
              device = device
              )

          # Load lightning checkpoints
          ckpt = torch.load(ckpt_path, map_location=device)
          sd = ckpt.get('state_dict', {})
          stripped = {
            k.replace('perceptual_model.', '', 1): v
            for k, v in sd.items() if k.startswith('perceptual_model.')
          }
          missing, unexpected = model.load_state_dict(stripped, strict=False)
          if missing:
            logging.warning(f"ViT load_state_dict missing/unexpected keys: {missing[:6]}{' ...' if len(missing)>6 else ''}")
          if unexpected:
            logging.warning(f"ViT load_state_dict missing/unexpected keys: {unexpected[:6]}{' ...' if len(unexpected)>6 else ''}")

        # Load the model trained with lora
        if training_method == 'lora':
          from peft import PeftModel
          logging.info('Loading Vision Transformer model trained with LoRA')

          # Create the path
          exp_dir = os.path.join(project_root, 'vision_models', 'vits_training', experiment_name, 'lora_adapters')
          ckpt_path = os.path.join(exp_dir, 'best' if best else 'last')
          if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
              f"Checkpoint not found: {ckpt_path}. "
              f"Check tag/model_type/training_method/dataset_name or finish training first."
            )

          # Load a backbone model
          model = PerceptualModel(
              model_type = model_type,
              feat_type = feat_type,
              stride = stride,
              hidden_size = hidden_size,
              lora = True,
              baseline = False,
              project_root = project_root,
              normalize_embeds = True,
              device = device
              )

          # Load the LoRA adapter (folder contains adapter_config.json + adapter_model.safetensors)
          model = PeftModel.from_pretrained(model, ckpt_path)

  # Load image paths
  processed_dir = os.path.join(project_root, 'datasets', 'visuelle2', 'processed_data')
  image_paths_csv = os.path.join(processed_dir, 'image_df.csv')
  if not os.path.exists(image_paths_csv):
    logging.error(f"Image paths CSV not found: {image_paths_csv}. Run process_visuelle first.")
    raise FileNotFoundError(f"Image paths CSV not found: {image_paths_csv}")

  df = pd.read_csv(image_paths_csv)
  if 'image_path' not in df.columns:
    logging.error("Missing required column: 'image_path' in the input CSV.")
    raise ValueError("Missing required column: 'image_path' in the input CSV.")

  # Embedd the good files
  model.to(device)
  model.eval()

  # Compute embeddings
  all_records, bad_files = [], []
  with torch.no_grad():
      for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding images"):
          img_path = os.path.join(project_root, 'datasets', 'visuelle2', 'images', row['image_path'])
          try:
            with Image.open(img_path) as img:
              img = img.convert('RGB')
              x = preprocess(img)
              if x.dim() == 3:        # [C,H,W] -> add batch
                  input_tensor = x.unsqueeze(0)
              elif x.dim() == 4:      # [B,C,H,W] -> already batched (e.g., DreamSim)
                  input_tensor = x
              else:
                  raise ValueError(f"Unexpected tensor shape from preprocess: {tuple(x.shape)}")
              input_tensor = input_tensor.to(device)
            embedding = model.embed(input_tensor).detach().cpu().numpy().squeeze()
            record = [row['image_path']] + embedding.tolist()
            all_records.append(record)
          except Exception as e:
              logging.warning(f'Skipped {img_path}: {e}')
              bad_files.append(img_path)
              continue
          if idx % 100 == 0:
              logging.info(f'Processed {idx} images')

  # Save list of skipped files
  if bad_files:
    bad_path = os.path.join(embeddings_dir, f'{experiment_name}_bad_images.txt')
    with open(bad_path, 'w') as f:
      for file in bad_files:
        f.write(file + '\n')
    logging.warning(f'Skipped {len(bad_files)} files. See {bad_path}')

  if not all_records:
    logging.error('No valid embeddings created.')
    return None

  # Save as DataFrame
  emb_dim = len(all_records[0]) - 1
  columns = ['image_path'] + [f'emb_{i}' for i in range(emb_dim)]
  df_emb = pd.DataFrame(all_records, columns=columns)
  df_emb.to_csv(emb_filepath, index=False)

  logging.info(f'Saved embeddings to {emb_filepath}')

  return emb_filepath
