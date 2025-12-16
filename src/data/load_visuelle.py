"""
load_visuelle.py

This script loads preprocessed Visuelle2.0 data and embeddings to prepare
training, validation, and test datasets for forecasting models.

Main tasks:
  - Merge preprocessed data with computed embeddings
  - Optionally apply PCA for dimensionality reduction
  - Normalize embeddings if needed
  - Build final feature matrices (X) and target arrays (y)
  - Return processed datasets with normalization scalars
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def normalize(X: np.ndarray) -> np.ndarray:
  """L2-normalize each row of a 2D NumPy array.

  Args:
    X (np.ndarray): Input 2D array where each row will be normalized.

  Returns:
    np.ndarray: L2-normalized array.
  """
  norms = np.linalg.norm(X, axis=1, keepdims=True)
  return X / np.clip(norms, 1e-8, None)


def build_features(
    df: pd.DataFrame,
    using_year_int: bool,
    using_year_dummies: bool,
    using_season_dummies: bool,
    using_price_float: bool,
    using_category_dummies: bool,
    using_color_dummies: bool,
    using_fabric_dummies: bool,
    using_store_int: bool,
    using_store_dummies: bool,
    using_week_dummies: bool,
    dummy_normalization: bool,
    emb_data_dict: dict,
    train_scaler_dict: dict = None
) -> tuple[np.ndarray, dict]:
  """Builds feature matrix for forecasting models.

  This function:
    - Selects embedding columns and optional feature groups
    - Normalizes integer features (year, store count)
    - Scales dummy variables if requested
    - Returns NumPy feature matrix and scalers for consistency

  Args:
    df (pd.DataFrame): Input dataset.
    using_* (bool): Flags for including different feature groups.
    dummy_normalization (bool): Whether to scale dummy variables.
    emb_data_dict (dict): Contains embedding mean and std for scaling.
    train_scaler_dict (dict, optional): Precomputed scalers for integer features.

  Returns:
    tuple[np.ndarray, dict]:
    - Feature matrix for the dataset
    - Updated scaler dictionary for reproducibility
  """
  if train_scaler_dict is None:
    train_scaler_dict = {'year_int': None, 'store_int': None, 'price_float': None}

  # Map feature groups to column selectors
  feature_groups = {
    'year_int': ['year'],
    'year_dummies': [col for col in df.columns if col.startswith('year_')],
    'season_dummies': [col for col in df.columns if col.startswith('fs_')],
    'price_float': ['price'],
    'category_dummies': [col for col in df.columns if col.startswith('cat_')],
    'color_dummies': [col for col in df.columns if col.startswith('col_')],
    'fabric_dummies': [col for col in df.columns if col.startswith('fab_')],
    'store_int': ['store_int'],
    'store_dummies': [col for col in df.columns if col.startswith('store_') and col != 'store_int'],
    'week_dummies': [col for col in df.columns if col.startswith('week_')],
  }

  # Map feature group flags
  flags = {
    'year_int': using_year_int,
    'year_dummies': using_year_dummies,
    'season_dummies': using_season_dummies,
    'price_float': using_price_float,
    'category_dummies': using_category_dummies,
    'color_dummies': using_color_dummies,
    'fabric_dummies': using_fabric_dummies,
    'store_int': using_store_int,
    'store_dummies': using_store_dummies,
    'week_dummies': using_week_dummies
  }

  # Start with embedding columns
  embedding_cols = [col for col in df.columns if col.startswith("emb_")]
  features = embedding_cols.copy()

  # Add selected feature groups
  for group_name, columns in feature_groups.items():
    if not flags[group_name] or not columns:
      continue

    features += columns

    # Normalize integer variables
    if group_name in ['year_int', 'store_int', 'price_float']:
      for col in columns:
        if train_scaler_dict[group_name] is None:
          mean_val = df[col].mean()
          std_val = df[col].std()
          if std_val == 0:
            std_val = 1.0
          train_scaler_dict[group_name] = [mean_val, std_val]

        df[col] = ((df[col] - train_scaler_dict[group_name][0]) / train_scaler_dict[group_name][1]
                   ) * emb_data_dict['embedding'][1] + emb_data_dict['embedding'][0]

    # Scale dummy variables if enabled
    elif dummy_normalization:
      df[columns] = (df[columns] * 2 - 1) * emb_data_dict['embedding'][1]

  return df[features].to_numpy(), train_scaler_dict


def load_visuelle(
    process_data_dict: dict,
    embedding_path: str,
    using_year_int: bool,
    using_year_dummies: bool,
    using_season_dummies: bool,
    using_price_float: bool,
    using_category_dummies: bool,
    using_color_dummies: bool,
    using_fabric_dummies: bool,
    using_store_int: bool,
    using_store_dummies: bool,
    using_week_dummies: bool,
    pca: bool,
    n_components: float,
    visualize_pca: bool,
    melt_data: bool,
    dummy_normalization: bool,
    normalize_embeddings_manually: bool,
    project_root: str
) -> dict:
  """Loads and prepares Visuelle2 dataset for forecasting.

  Steps:
  - Loads train, val, test splits
  - Merges embeddings
  - Applies PCA (optional)
  - Normalizes embeddings (optional)
  - Builds feature matrices and targets

  Args:
    process_data_dict: Dictionary with dataset paths.
    embedding_path: Path to embeddings CSV.
    using_*: Flags for feature inclusion.
    pca: Whether to apply PCA to embeddings.
    n_components: PCA components or variance fraction.
    visualize_pca: Plot explained variance.
    melt_data: Whether data is melted (single week target).
    dummy_normalization: Scale dummy variables.
    normalize_embeddings_manually: Manually normalize embeddings.
    project_root: Root directory of the project.

  Returns:
    dict: Processed features, targets, scalers, and dataframes.

  Raises:
    FileNotFoundError: If dataset or embedding files are missing.
    ValueError: If merging results in data loss or empty outputs.
  """
  logging.info("Loading and preparing Visuelle2 dataset...")

  if using_week_dummies and not melt_data:
    logging.error("Invalid configuration: Week dummies require melted data.")
    raise ValueError("Week dummies are only available when data is melted.")

  # Dataset paths
  train_path =process_data_dict['dataset_paths']['train_data_path']
  val_path = process_data_dict['dataset_paths']['val_data_path']
  test_path = process_data_dict['dataset_paths']['test_data_path']

  # Check file existence
  for path in [train_path, val_path, test_path, embedding_path]:
    if not os.path.exists(path):
      logging.error(f"Missing required file: {path}")
      raise FileNotFoundError(f"File does not exist: {path}")

  # Load CSVs
  train_df = pd.read_csv(train_path)
  val_df = pd.read_csv(val_path)
  test_df = pd.read_csv(test_path)
  embedding_df = pd.read_csv(embedding_path)

  logging.info(f"Load training data from {train_path}")
  logging.info(f"Load validation data from {val_path}")
  logging.info(f"Load test data from {test_path}")
  logging.info(f"Load embeddings from {embedding_path}")

  if 'image_path' not in embedding_df.columns:
    logging.error("Embedding CSV is missing 'image_path' column.")
    raise ValueError("Embedding data must have 'image_path' column.")

  # Merge on 'image_path'
  train_df['merge_idx'] = np.arange(len(train_df))
  val_df['merge_idx'] = np.arange(len(val_df))
  test_df['merge_idx'] = np.arange(len(test_df))

  train_merged = pd.merge(train_df, embedding_df, on='image_path', how='inner') \
                  .sort_values('merge_idx') \
                  .reset_index(drop=True)
  val_merged = pd.merge(val_df, embedding_df, on='image_path', how='inner') \
                .sort_values('merge_idx') \
                .reset_index(drop=True)
  test_merged = pd.merge(test_df, embedding_df, on='image_path', how='inner') \
                  .sort_values('merge_idx') \
                  .reset_index(drop=True)

  # Log the lenght after merging
  logging.info(f"Length of training data before merging is {train_df.shape[0]} rows.")
  logging.info(f"Length of training data after merging is {train_merged.shape[0]} rows.")
  logging.info(f"Length of validation data before merging is {val_df.shape[0]} rows.")
  logging.info(f"Length of validation data after merging is {val_merged.shape[0]} rows.")
  logging.info(f"Length of testing data before merging is {test_df.shape[0]} rows.")
  logging.info(f"Length of testing data after merging is {test_merged.shape[0]} rows.")

  # Raise an error if the merged data has less rows then before merge
  allowed_loss = 13
  if train_df.shape[0] - train_merged.shape[0] > allowed_loss:
    raise ValueError('Observations lost after merge in training data.')
  if val_df.shape[0] - val_merged.shape[0] > allowed_loss:
    raise ValueError('Observations lost after merge in validation data.')
  if test_df.shape[0] - test_merged.shape[0] > allowed_loss:
    raise ValueError('Observations lost after merge in test data.')

  if train_merged.shape[0] == 0 or val_merged.shape[0] == 0 or test_merged.shape[0] == 0:
    raise ValueError("Merge resulted in zero rows for one or more splits. Check your image_path columns!")

  embedding_cols = [col for col in train_merged.columns if col.startswith("emb_")]

  # PCA
  if pca:
    pca = PCA(n_components=n_components)
    pca.fit(train_merged[embedding_cols])

    train_pca = pca.transform(train_merged[embedding_cols])
    val_pca = pca.transform(val_merged[embedding_cols])
    test_pca = pca.transform(test_merged[embedding_cols])
    pca_dim = train_pca.shape[1]

    logging.info(f"PCA embedding dim: {pca_dim}")

    # Create new column names
    new_cols = [f"emb_pca_{i}" for i in range(train_pca.shape[1])]

    # Add transformed data as new columns and drop old one
    train_pca_df = pd.DataFrame(train_pca, columns=new_cols, index=train_merged.index)
    train_merged = pd.concat([train_merged.drop(columns=embedding_cols), train_pca_df], axis=1)
    val_pca_df = pd.DataFrame(val_pca, columns=new_cols, index=val_merged.index)
    val_merged = pd.concat([val_merged.drop(columns=embedding_cols), val_pca_df], axis=1)
    test_pca_df = pd.DataFrame(test_pca, columns=new_cols, index=test_merged.index)
    test_merged = pd.concat([test_merged.drop(columns=embedding_cols), test_pca_df], axis=1)

    # Update embedding column list
    embedding_cols = new_cols

    if visualize_pca:
      plt.plot(np.cumsum(pca.explained_variance_ratio_))
      plt.xlabel("Number of Components")
      plt.ylabel("Cumulative Explained Variance")
      plt.show()
  else:
      pca_dim = None

  # Normalize embeddings to have unit length of one
  if normalize_embeddings_manually:
    train_merged[embedding_cols] = normalize(train_merged[embedding_cols].values)
    val_merged[embedding_cols] = normalize(val_merged[embedding_cols].values)
    test_merged[embedding_cols] = normalize(test_merged[embedding_cols].values)

  # Embedding scalars
  emb_mean = train_merged[embedding_cols].mean().mean()
  emb_std = train_merged[embedding_cols].std().mean()
  emb_data_dict = {'embedding': [emb_mean, emb_std]}

  # Feature matrices
  X_train, train_df_dict = build_features(
    train_merged,
    using_year_int,
    using_year_dummies,
    using_season_dummies,
    using_price_float,
    using_category_dummies,
    using_color_dummies,
    using_fabric_dummies,
    using_store_int,
    using_store_dummies,
    using_week_dummies,
    dummy_normalization,
    emb_data_dict,
    train_scaler_dict=None
  )

  X_val, train_df_dict = build_features(
    val_merged,
    using_year_int,
    using_year_dummies,
    using_season_dummies,
    using_price_float,
    using_category_dummies,
    using_color_dummies,
    using_fabric_dummies,
    using_store_int,
    using_store_dummies,
    using_week_dummies,
    dummy_normalization,
    emb_data_dict,
    train_scaler_dict=train_df_dict
  )

  X_test, train_df_dict = build_features(
    test_merged,
    using_year_int,
    using_year_dummies,
    using_season_dummies,
    using_price_float,
    using_category_dummies,
    using_color_dummies,
    using_fabric_dummies,
    using_store_int,
    using_store_dummies,
    using_week_dummies,
    dummy_normalization,
    emb_data_dict,
    train_scaler_dict=train_df_dict
  )

  if melt_data:
    y_train, y_val, y_test = train_merged['sales_sum'].to_numpy(), val_merged['sales_sum'].to_numpy(), test_merged['sales_sum'].to_numpy()
  else:
    sales_cols = [col for col in train_merged.columns if col.startswith('sum_')]
    if not sales_cols:
      logging.error("No sales columns found in dataset.")
      raise ValueError("No sales columns found in dataset.")
    y_train, y_val, y_test = train_merged[sales_cols].to_numpy(), val_merged[sales_cols].to_numpy(), test_merged[sales_cols].to_numpy()

  # Validate feature matrices
  for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
    if X.shape[0] == 0 or X.shape[1] == 0:
      logging.error(f"{name} features are empty after processing.")
      raise ValueError(f"{name} features are empty.")
    if y.shape[0] == 0:
      logging.error(f"{name} targets are empty after processing.")
      raise ValueError(f"{name} targets are empty.")

  logging.info(f"Loaded data: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

  return {
      'scalars': {
          'embedding_norm': emb_data_dict,
          'train_df_norm': train_df_dict,
          'pca_dim': pca_dim
      },
      'train_df': {
        'X_train': X_train,
        'y_train': y_train
      },
      'val_df': {
        'X_val': X_val,
        'y_val': y_val
      },
      'test_df': {
        'X_test': X_test,
        'y_test': y_test
      },
      'data_frames': {
        'train_df': train_merged,
        'val_df': val_merged,
        'test_df': test_merged
      }
  }
