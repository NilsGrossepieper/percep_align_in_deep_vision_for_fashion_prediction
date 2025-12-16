"""
train_ridge_regressor.py

This script handles training a Ridge regressor model for forecasting.
Main tasks:
  - Define Optuna objective for hyperparameter search
  - TPE search to find the best alpha
  - Train a final Ridge model with the best parameters
  - Save the trained model and study for future use
"""

import logging
import os
import time

import joblib
import numpy as np
import optuna
from optuna.samplers import GridSampler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Optional
import wandb

from models.ridge_regressor import ridge_regressor


def ridge_objective(
  trial: optuna.Trial,
  X_train: np.ndarray,
  y_train: np.ndarray,
  X_val: np.ndarray,
  y_val: np.ndarray,
  error_metric: str,
  alpha_values: list[float],
  melt_data: bool,
  forecasting_seed: int
) -> float:
  """Objective function for Optuna to optimize Ridge regressor hyperparameters.

  Args:
    trial (optuna.Trial): Current Optuna trial.
    X_train (np.ndarray): Training features.
    y_train (np.ndarray): Training targets.
    X_val (np.ndarray): Validation features.
    y_val (np.ndarray): Validation targets.
    error_metric (str): Error metric ('mse' or 'mae') for evaluation.
    alpha_values (list[float]): Candidate alpha values for Ridge regressor.
    melt_data (bool): Whether data is melted (single week).
    forecasting_seed (int): Random seed for reproducibility.

  Returns:
    float: Validation error score for the trial.
  """
  alpha = trial.suggest_categorical('alpha', alpha_values)
  logging.info(f"Testing Ridge regressor with {alpha=}")

  model = ridge_regressor(alpha, melt_data, forecasting_seed, final_run=False)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_val)

  if error_metric == 'mse':
    error_score = mean_squared_error(y_val, y_pred)
  elif error_metric == 'mae':
    error_score = mean_absolute_error(y_val, y_pred)
  else:
    logging.error(f"Unknown error metric: {error_metric}")
    raise ValueError(f"Unknown error metric: {error_metric}")

  trial.set_user_attr('alpha', float(alpha))
  trial.report(float(error_score), step=0)

  return  float(error_score)


def run_gridsearch_ridge(
  X_train: np.ndarray,
  y_train: np.ndarray,
  X_val: np.ndarray,
  y_val: np.ndarray,
  error_metric: str,
  alpha_values: list[float],
  melt_data: bool,
  forecasting_seed: int,
) -> optuna.Study:
  """Run exhaustive grid search for Ridge regressor using Optuna.

  Args:
    X_train (np.ndarray): Training features.
    y_train (np.ndarray): Training targets.
    X_val (np.ndarray): Validation features.
    y_val (np.ndarray): Validation targets.
    error_metric (str): Error metric ('mse' or 'mae') for evaluation.
    alpha_values (list[float]): Candidate alpha values for Ridge regressor.
    melt_data (bool): Whether data is melted (single week).
    forecasting_seed (int): Random seed for reproducibility.

  Returns:
    optuna.Study: The completed Optuna study with optimization results.
  """
  logging.info("Starting grid search for Ridge regressor hyperparameters.")

  search_space = {'alpha': alpha_values}
  sampler = GridSampler(search_space)
  study = optuna.create_study(direction='minimize', sampler=sampler)

  objective_func = lambda trial: ridge_objective(
    trial, X_train, y_train, X_val, y_val, error_metric, alpha_values, melt_data, forecasting_seed
  )

  try:
    study.optimize(objective_func, n_trials=len(alpha_values), n_jobs=-1, show_progress_bar=True)
  except KeyboardInterrupt:
      logging.warning('Optimization interruption by user. Returning current best study.')
  except Exception as e:
      logging.error(f'An error occurred during optimization: {e}')
      raise e

  return study


def train_ridge_regressor(
    dict_data: dict,
    project_name: str,
    experiment_name: str,
    wandb_bool: bool,
    error_metric: str,
    alpha_values: list[float],
    melt_data: bool,
    pca_dim: Optional[int],
    forecasting_seed: int,
    project_root: str,
) -> dict:
  """Train a Ridge regressor model with grid search hyperparameter tuning.

  Steps:
  - Load training and validation datasets
  - Run Optuna grid search to find the best alpha
  - Train a final model with the best hyperparameters
  - Save model and study artifacts for reuse

  Args:
    dict_data (dict): Dataset dictionary with train/val features and targets.
    project_name (str): Name of the project.
    experiment_name (str): Name of the experiment (used for saving artifacts).
    wandb_bool (bool): Whether to log to WandB.
    error_metric (str): Metric to optimize ('mse' or 'mae').
    alpha_values (list[float]): List of alpha values to search.
    melt_data (bool): Whether data is melted (single-week format).
    pca_dim (Optional[int]): Dimensionality of PCA features.
    forecasting_seed (int): Random seed for reproducibility.
    project_root (str): Root project directory.

  Returns:
    dict: Paths to saved model/study and best hyperparameters.
  """
  if error_metric not in ['mse', 'mae']:
    logging.error(f"Unknown error metric: {error_metric}")
    raise ValueError(f"Unknown error metric: {error_metric}")

  # Directories for saving
  model_dir = os.path.join(project_root, 'models', experiment_name)
  study_dir = os.path.join(project_root, 'studies', experiment_name)
  os.makedirs(model_dir, exist_ok=True)
  os.makedirs(study_dir, exist_ok=True)

  model_path = os.path.join(model_dir, 'ridge_regressor_model.joblib')
  study_path = os.path.join(study_dir, 'ridge_regressor_study.joblib')

  # Load existing model/study if available
  if os.path.exists(model_path) and os.path.exists(study_path):
    logging.info(f"Using existing Ridge regressor model: {model_path}")
    logging.info(f"Using existing Optuna study: {study_path}")
    study = joblib.load(study_path)
    time_hyperparameter_search = 0.0
    time_final_training = 0.0

  else:
    # Load data
    X_train = dict_data['train_df']['X_train']
    y_train = dict_data['train_df']['y_train']
    X_val = dict_data['val_df']['X_val']
    y_val = dict_data['val_df']['y_val']

    # Start hyperparameter training
    logging.info(f"Starting hyperparameter search for Ridge regressor...")
    start_time_hyperparameter_search = time.time()

    study = run_gridsearch_ridge(
      X_train,
      y_train,
      X_val,
      y_val,
      error_metric,
      alpha_values,
      melt_data,
      forecasting_seed
    )

    time_hyperparameter_search = time.time() - start_time_hyperparameter_search
    logging.info(f"Hyperparameter search took {time_hyperparameter_search:.2f} seconds.")
    logging.info(f"Best alpha: {study.best_params['alpha']}")
    logging.info(f"Best {error_metric.upper()}: {study.best_value}")

    # Final training using train + val
    X_trainval = np.concatenate((X_train, X_val), axis=0)
    y_trainval = np.concatenate((y_train, y_val), axis=0)

    logging.info(f"Starting final model training with best alpha: {study.best_params['alpha']}...")
    model = ridge_regressor(
      study.best_params['alpha'], melt_data, forecasting_seed, final_run=True
    )

    start_time_final_training = time.time()
    model.fit(X_trainval, y_trainval)
    time_final_training = time.time() - start_time_final_training

    logging.info(f"Final training took {time_final_training:.2f} seconds.")
    logging.info(f"Saving final Ridge model to {model_path} and study to {study_path}.")

    joblib.dump(model, model_path)
    joblib.dump(study, study_path)

    # Log results in WandB
    if wandb_bool:
      run_name = f"{experiment_name}_ridge_regressor_model_configs"
      with wandb.init(
        project = project_name,
        group = experiment_name,
        name = run_name,
        job_type = 'model configs',
        tags = [project_name, 'ridge_regressor_model', 'model_configs', 'prediction'],
        notes = f"Configurations of the final Ridge regressor model for {experiment_name}.",
        reinit = True
      ) as run:
        run.log({
          'ridge_regressor_config': {
            'best_alpha': study.best_params['alpha'],
            f"best_{error_metric}": study.best_value,
            'time_hyperparameter_search': time_hyperparameter_search,
            'time_final_training': time_final_training,
            'pca_dim': pca_dim
          }
        })

        model_artifact = wandb.Artifact(
          name=f"{experiment_name}_ridge_regressor_model",
          type='model',
          description='Final Ridge regressor model',
          metadata={
            'alpha': study.best_params['alpha'],
            f"{error_metric}": study.best_value,
            'pca_dim': pca_dim
          }
        )
        model_artifact.add_file(model_path)
        run.log_artifact(model_artifact)

        study_artifact = wandb.Artifact(
          name = f"{experiment_name}_ridge_regressor_study",
          type = 'study',
          description = 'Optuna study for Ridge regressor',
          metadata = {
            'alpha': study.best_params['alpha'],
            f"{error_metric}": study.best_value,
            'pca_dim': pca_dim
          }
        )
        study_artifact.add_file(study_path)
        run.log_artifact(study_artifact)

  return {
      'model_path': model_path,
      'study_path': study_path,
      'best_params': study.best_params,
      f"best_{error_metric}": study.best_value,
      'training_time': {
        'time_hyperparameter_search': time_hyperparameter_search,
        'time_final_training': time_final_training
      }
  }
