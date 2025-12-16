"""
train_knn_regressor.py

This script handles training a kNN regressor model for forecasting.
Main tasks:
  - Define Optuna objective for hyperparameter search
  - TPE search to find the best number of neighbors and weights
  - Train a final kNN model with the best parameters
  - Save the trained model and study for future use
"""

import logging
import os
import time

import joblib
import numpy as np
import optuna
from optuna.samplers import GridSampler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Optional
import wandb

from models.knn_regressor import knn_regressor


def knn_objective(
  trial: optuna.Trial,
  X_train: np.ndarray,
  y_train: np.ndarray,
  X_val: np.ndarray,
  y_val: np.ndarray,
  error_metric: str,
  max_neighbors: int,
  weights: list[str]
) -> float:
  """Objective function Optuna to optimize kNN hyperparameters.

  Args:
    trial (optuna.Trial): Current Optuna trial.
    X_train (np.ndarray): Training features.
    y_train (np.ndarray): Training targets.
    X_val (np.ndarray): Validation features.
    y_val (np.ndarray): Validation targets.
    error_metric (str): Error metric ('mse' or 'mae') for evaluation.
    max_neighbors (int): Maximum number of neighbors to try.
    weights (list[str]): List of weight objective_functions to try.

  Returns:
    float: Validation error score for the trial.
  """
  n_neighbors = trial.suggest_categorical('n_neighbors', list(range(1, max_neighbors + 1)))
  chosen_weights = trial.suggest_categorical('weights', weights)

  model = knn_regressor(n_neighbors, chosen_weights, final_run=False)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_val)

  if error_metric == 'mse':
    error_score = mean_squared_error(y_val, y_pred)
  elif error_metric == 'mae':
    error_score = mean_absolute_error(y_val, y_pred)
  else:
    logging.error(f"Unknown error metric: {error_metric}")
    raise ValueError(f"Unknown error metric: {error_metric}")

  trial.set_user_attr('n_neighbors', int(n_neighbors))
  trial.set_user_attr('weights', str(chosen_weights))
  trial.report(float(error_score), step=0)

  return float(error_score)


def run_gridsearch_knn(
  X_train: np.ndarray,
  y_train: np.ndarray,
  X_val: np.ndarray,
  y_val: np.ndarray,
  error_metric: str,
  max_neighbors: int,
  weights: list[str],
) -> optuna.Study:
  """Run exhaustive grid search for kNN regressor using Optuna.

  Args:
    X_train (np.ndarray): Training features.
    y_train (np.ndarray): Training targets.
    X_val (np.ndarray): Validation features.
    y_val (np.ndarray): Validation targets.
    error_metric (str): Error metric ('mse' or 'mae') for evaluation.
    max_neighbors (int): Maximum number of neighbors to try.
    weights (list[str]): List of weight objective_functions to try.

  Returns:
    optuna.Study: The completed Optuna study with optimization results.
  """
  logging.info("Starting grid search for kNN regressor hyperparameters.")

  search_space = {
    'n_neighbors': list(range(1, max_neighbors + 1)),
    'weights': weights
  }
  total_trials = len(search_space['n_neighbors']) * len(weights)
  sampler = GridSampler(search_space)
  study = optuna.create_study(direction='minimize', sampler=sampler)

  objective_func = lambda trial: knn_objective(
    trial, X_train, y_train, X_val, y_val, error_metric, max_neighbors, weights
  )

  try:
    study.optimize(objective_func, n_trials=total_trials, n_jobs=-1, show_progress_bar=True)
  except KeyboardInterrupt:
      logging.warning('Optimization interruption by user. Returning current best study.')
  except Exception as e:
      logging.error(f'An error occurred during optimization: {e}')
      raise e

  return study


def train_knn_regressor(
  dict_data: dict,
  project_name: str,
  experiment_name: str,
  wandb_bool: bool,
  error_metric: str,
  max_neighbors: int,
  weights: list[str],
  pca_dim: Optional[int],
  project_root: str,
) -> dict:
  """Train a kNN regressor model with grid search hyperparameter tuning.

  Steps:
  - Load training and validation datasets
  - Run Optuna grid search to find the best alpha
  - Train a final model with the best hyperparameters
  - Save model and study artifacts for reuse

  Args:
    dict_data (dict): Dataset dictionary with train/val features and targets.
    project_name (str): Name of the Optuna project.
    experiment_name (str): Name of the Optuna experiment.
    wandb_bool (bool): Whether to use Weights & Biases for logging.
    error_metric (str): Metric to optimize ('mse' or 'mae').
    max_neighbors (int): Maximum number of neighbors to try.
    weights (list[str]): List of weight objective_functions to try.
    pca_dim (Optional[int]): Dimensionality of PCA features.
    project_root (str): Root directory of the project.

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

  model_path = os.path.join(model_dir, 'knn_regressor_model.joblib')
  study_path = os.path.join(study_dir, 'knn_regressor_study.joblib')

  # Load existing model/study if available
  if os.path.exists(model_path) and os.path.exists(study_path):
    logging.info(f"Using existing kNN regressor model: {model_path}")
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
    logging.info(f"Starting hyperparameter search for knn regressor...")
    start_time_hyperparameter_search = time.time()

    study = run_gridsearch_knn(
      X_train,
      y_train,
      X_val,
      y_val,
      error_metric,
      max_neighbors,
      weights
    )

    time_hyperparameter_search = time.time() - start_time_hyperparameter_search
    logging.info(f"Hyperparameter search took {time_hyperparameter_search:.2f} seconds.")
    logging.info(f"Best parameters: "
      f"{study.best_params['n_neighbors']=} {study.best_params['weights']=}"
    )
    logging.info(f"Best {error_metric.upper()}: {study.best_value}")

    # Final training using train + val
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    logging.info(f"Starting final model training with best parameters: "
      f"{study.best_params['n_neighbors']=} {study.best_params['weights']=}"
    )
    model = knn_regressor(
      study.best_params['n_neighbors'],
      study.best_params['weights'],
      final_run=True
    )

    start_time_final_training = time.time()
    model.fit(X_trainval, y_trainval)
    time_final_training = time.time() - start_time_final_training

    logging.info(f"Final training took {time_final_training:.2f} seconds.")
    logging.info(f"Saving final kNN model to {model_path} and study to {study_path}.")

    joblib.dump(model, model_path)
    joblib.dump(study, study_path)

    # Log results in WandB
    if wandb_bool:
      run_name = f"{experiment_name}_knn_regressor_model_configs"
      with wandb.init(
          project = project_name,
          group = experiment_name,
          name = run_name,
          job_type = 'model configs',
          tags = [project_name, 'knn_regressor_model', 'model_configs', 'prediction'],
          notes = f"Configurations of the final kNN regressor model for {experiment_name}.",
          reinit=True
      ) as run:
        run.log({
        'knn_regressor_configs': {
          'best_n_neighbors': study.best_params['n_neighbors'],
          'best_weights': study.best_params['weights'],
          f"best_{error_metric}": study.best_value,
          'time_hyperparameter_search': time_hyperparameter_search,
          'time_final_training': time_final_training,
          'pca_dim': pca_dim
          }
        })

        model_artefact = wandb.Artifact(
            name = f"{experiment_name}_knn_regressor_model",
            type='model',
            description='Final kNN regressor Model',
            metadata = {
                'best_n_neighbors': study.best_params['n_neighbors'],
                'best_weights': study.best_params['weights'],
                f"best_{error_metric}": study.best_value,
                'pca_dim': pca_dim
            }
        )
        model_artefact.add_file(model_path)
        run.log_artifact(model_artefact)

        study_artefact = wandb.Artifact(
            name = f"{experiment_name}_knn_regressor_study",
            type='study',
            description='Optuna Study for kNN regressor',
            metadata = {
                'best_n_neighbors': study.best_params['n_neighbors'],
                'best_weights': study.best_params['weights'],
                f"best_{error_metric}": study.best_value,
                'pca_dim': pca_dim
            }
          )
        study_artefact.add_file(study_path)
        run.log_artifact(study_artefact)

  return {
    'model_path': model_path,
    'study_path': study_path,
    'best_params': study.best_params,
    f"best_{error_metric}": study.best_value,
    'error_metric': error_metric,
    'training_time': {
        'time_hyperparameter_search': time_hyperparameter_search,
        'time_final_training': time_final_training
    }
  }
