"""
train_gradient_boosting_regressor.py

This script handles training a Gradient Boosting model for forecasting.
Main tasks:
  - Define Optuna objective for hyperparameter search
  - TPE search to find the best hyperparameter
  - Train a final Gradient Boosting model with the best parameters
  - Save the trained model and study for future use
"""

import logging
import os
import time

import joblib
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Optional
import wandb

from models.gradient_boosting_regressor import gradient_boosting_regressor


def gradient_boosting_objective(
  trial: optuna.Trial,
  X_train: np.ndarray,
  y_train: np.ndarray,
  X_val: np.ndarray,
  y_val: np.ndarray,
  error_metric: str,
  n_estimators: list[int],
  learning_rate: list[float],
  max_depth: list[int],
  min_samples_split: list[int],
  min_samples_leaf: list[int],
  melt_data: bool,
  forecasting_seed: int
  ) -> float:
  """Objective function for Optuna to optimize Gradient Boosting hyperparameters.

  Args:
    trial (optuna.Trial): Current Optuna trial.
    X_train (np.ndarray): Training features.
    y_train (np.ndarray): Training targets.
    X_val (np.ndarray): Validation features.
    y_val (np.ndarray): Validation targets.
    error_metric (str): Error metric ('mse' or 'mae') for evaluation.
    n_estimators (list[int]): List of n_estimators values to try.
    learning_rate (list[float]): List of learning_rate values to try.
    max_depth (list[int]): List of max_depth values to try.
    min_samples_split (list[int]): List of min_samples_split values to try.
    min_samples_leaf (list[int]): List of min_samples_leaf values to try.
    melt_data (bool): Whether data is melted (single week).
    forecasting_seed (int): Random seed for reproducibility.

  Returns:
    float: Validation error score for the trial.
  """
  choosen_n_estimators = trial.suggest_categorical('n_estimators', n_estimators)
  choosen_learning_rate = trial.suggest_categorical('learning_rate', learning_rate)
  choosen_max_depth = trial.suggest_categorical('max_depth', max_depth)
  choosen_min_samples_split = trial.suggest_categorical('min_samples_split', min_samples_split)
  choosen_min_samples_leaf = trial.suggest_categorical('min_samples_leaf', min_samples_leaf)

  model = gradient_boosting_regressor(
    choosen_n_estimators,
    choosen_learning_rate,
    choosen_max_depth,
    choosen_min_samples_split,
    choosen_min_samples_leaf,
    melt_data,
    forecasting_seed,
    final_run=False
  )
  model.fit(X_train, y_train)

  y_pred = model.predict(X_val)

  if error_metric == 'mse':
    error_score = mean_squared_error(y_val, y_pred)
  elif error_metric == 'mae':
    error_score = mean_absolute_error(y_val, y_pred)
  else:
    logging.error(f"Unknown error metric: {error_metric}")
    raise ValueError(f"Unknown error metric: {error_metric}")

  trial.set_user_attr('n_estimators', int(choosen_n_estimators))
  trial.set_user_attr('learning_rate', float(choosen_learning_rate))
  trial.set_user_attr('max_depth', int(choosen_max_depth))
  trial.set_user_attr('min_samples_split', int(choosen_min_samples_split))
  trial.set_user_attr('min_samples_leaf', int(choosen_min_samples_leaf))
  trial.report(float(error_score), step=0)

  return float(error_score)


def run_tpesampler_gradient_boosting(
  X_train: np.ndarray,
  y_train: np.ndarray,
  X_val: np.ndarray,
  y_val: np.ndarray,
  error_metric: str,
  n_trials: int,
  n_estimators: list[int],
  learning_rate: list[float],
  max_depth: list[int],
  min_samples_split: list[int],
  min_samples_leaf: list[int],
  melt_data: bool,
  forecasting_seed: int,
):
  """Run exhaustive hyperparameter search for Gradient Boosting regressor using Optuna.

  Args:
    X_train (np.ndarray): Training features.
    y_train (np.ndarray): Training targets.
    X_val (np.ndarray): Validation features.
    y_val (np.ndarray): Validation targets.
    error_metric (str): Error metric ('mse' or 'mae') for evaluation.
    n_trials (int): Number of trials for Optuna.
    n_estimators (list[int]): List of n_estimators values to try.
    learning_rate (list[float]): List of learning_rate values to try.
    max_depth (list[int]): List of max_depth values to try.
    min_samples_split (list[int]): List of min_samples_split values to try.
    melt_data (bool): Whether data is melted (single week).
    forecasting_seed (int): Random seed for reproducibility.

  Returns:
    optuna.Study: The completed Optuna study with optimization results.
  """
  logging.info(f"Running hyperparameter search with {n_trials} trials.")

  study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=forecasting_seed)
  )

  objective_func = lambda trial: gradient_boosting_objective(
    trial,
    X_train,
    y_train,
    X_val,
    y_val,
    error_metric,
    n_estimators,
    learning_rate,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    melt_data,
    forecasting_seed
  )

  try:
    study.optimize(objective_func, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)
  except KeyboardInterrupt:
    logging.warning('Optimization interruption by user. Returning current best study.')
  except Exception as e:
    logging.error(f'An error occurred during optimization: {e}')
    raise e

  return study


def train_gradient_boosting_regressor(
  dict_data: dict,
  project_name: str,
  experiment_name: str,
  wandb_bool: bool,
  error_metric: str,
  n_trials: int,
  n_estimators: list[int],
  learning_rate: list[float],
  max_depth: list[int],
  min_samples_split: list[int],
  min_samples_leaf: list[int],
  melt_data: bool,
  pca_dim: Optional[int],
  forecasting_seed: int,
  project_root: str,
):
  """Train a Gradient Boosting regressor model with hyperparameter tuning.

  Steps:
  - Load training and validation datasets
  - Run Optuna grid search to find the best alpha
  - Train a final model with the best hyperparameters
  - Save model and study artifacts for reuse

  Args:
    dict_data (dict): Dataset dictionary with train/val features and targets.
    project_name (str): Name of the WandB project.
    experiment_name (str): Name of the WandB experiment.
    wandb_bool (bool): Whether to use WandB for logging.
    error_metric (str): Metric to optimize ('mse' or 'mae').
    n_trials (int): Number of trials for Optuna.
    n_estimators (list[int]): List of n_estimators values to try.
    learning_rate (list[float]): List of learning_rate values to try.
    max_depth (list[int]): List of max_depth values to try.
    min_samples_split (list[int]): List of min_samples_split values to try.
    min_samples_leaf (list[int]): List of min_samples_leaf values to try.
    melt_data (bool): Whether data is melted (single week).
    pca_dim (Optional[int]): Dimensionality of PCA features.
    forecasting_seed (int): Random seed for reproducibility.
    project_root (str): Root directory of the project.

  Returns:
    dict: Paths to saved model/study and best hyperparameters.
  """
  if error_metric not in ['mse', 'mae']:
    logging.error(f"Unknown error metric: {error_metric}")
    raise ValueError(f"Unknown error metric: {error_metric}")

  if pca_dim is None:
    logging.info("PCA embedding dim: None (PCA disabled)")
  else:
    logging.info(f"PCA embedding dim: {pca_dim}")

  # Directories for saving
  model_dir = os.path.join(project_root, 'models', experiment_name)
  study_dir = os.path.join(project_root, 'studies', experiment_name)
  os.makedirs(model_dir, exist_ok=True)
  os.makedirs(study_dir, exist_ok=True)

  model_path = os.path.join(model_dir, 'gradient_boosting_regressor_model.joblib')
  study_path = os.path.join(study_dir, 'gradient_boosting_regressor_study.joblib')

  # Load existing model/study if available
  if os.path.exists(model_path) and os.path.exists(study_path):
    logging.info(f"Using existing Gradient Boosting regressor model: {model_path}")
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
    logging.info('Starting hyperparameter search for Gradient Boosting regressor...')
    start_time_hyperparameter_search = time.time()

    study = run_tpesampler_gradient_boosting(
      X_train,
      y_train,
      X_val,
      y_val,
      error_metric,
      n_trials,
      n_estimators,
      learning_rate,
      max_depth,
      min_samples_split,
      min_samples_leaf,
      melt_data,
      forecasting_seed,
    )

    time_hyperparameter_search = time.time() - start_time_hyperparameter_search
    logging.info(f"Hyperparameter search took {time_hyperparameter_search:.2f} seconds.")
    logging.info(f"Best parameters: "
      f"{study.best_params['n_estimators']=} {study.best_params['learning_rate']=}, "
      f"{study.best_params['max_depth']=}, {study.best_params['min_samples_split']=}, "
      f"{study.best_params['min_samples_leaf']=}"
    )
    logging.info(f"Best {error_metric.upper()}: {study.best_value}")

    # Final training using train + val
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    logging.info(f"Starting final model training with best parameters: "
      f"{study.best_params['n_estimators']=} {study.best_params['learning_rate']=}, "
      f"{study.best_params['max_depth']}, {study.best_params['min_samples_split']=}, "
      f"{study.best_params['min_samples_leaf']=}"
    )

    model = gradient_boosting_regressor(
      study.best_params['n_estimators'],
      study.best_params['learning_rate'],
      study.best_params['max_depth'],
      study.best_params['min_samples_split'],
      study.best_params['min_samples_leaf'],
      melt_data,
      forecasting_seed,
      final_run=True
    )

    start_time_final_training = time.time()
    model.fit(X_trainval, y_trainval)
    time_final_training = time.time() - start_time_final_training

    logging.info(f"Final training took {time_final_training:.2f} seconds.")
    logging.info(f"Saving final Gradient Boosting model to {model_path} and study to {study_path}.")

    joblib.dump(model, model_path)
    joblib.dump(study, study_path)

    # Log results in WandB
    if wandb_bool:
      run_name = f"{experiment_name}_gradient_boosting_regressor_model_configs"
      with wandb.init(
          project = project_name,
          group = experiment_name,
          name = run_name,
          job_type = 'model_configs',
          tags = [project_name, 'gradient_boosting_regressor_model', 'model_configs', 'prediction'],
          notes = f"Configurations of the final Gradient Boosting regressor model for {experiment_name}."
      ) as run:
        run.log({
          'gradient_boosting_configs': {
            'best_n_estimators': study.best_params['n_estimators'],
            'best_learning_rate': study.best_params['learning_rate'],
            'best_max_depth': study.best_params['max_depth'],
            'best_min_samples_split': study.best_params['min_samples_split'],
            'best_min_samples_leaf': study.best_params['min_samples_leaf'],
            f"best_{error_metric}": study.best_value,
            'time_hyperparameter_search': time_hyperparameter_search,
            'time_final_training': time_final_training,
            'pca_dim': pca_dim
          }
        })

        model_artifact = wandb.Artifact(
            name = f"{experiment_name}_gradient_boosting_regressor_model",
            type = 'model',
            metadata = {
                'best_n_estimators': study.best_params['n_estimators'],
                'best_learning_rate': study.best_params['learning_rate'],
                'best_max_depth': study.best_params['max_depth'],
                'best_min_samples_split': study.best_params['min_samples_split'],
                'best_min_samples_leaf': study.best_params['min_samples_leaf'],
                f"best_{error_metric}": study.best_value,
                'pca_dim': pca_dim
            }
        )
        model_artifact.add_file(model_path)
        run.log_artifact(model_artifact)

        study_artifact = wandb.Artifact(
            name = f"{experiment_name}_gradient_boosting_regressor_study",
            type = 'study',
            metadata = {
                'best_n_estimators': study.best_params['n_estimators'],
                'best_learning_rate': study.best_params['learning_rate'],
                'best_max_depth': study.best_params['max_depth'],
                'best_min_samples_split': study.best_params['min_samples_split'],
                'best_min_samples_leaf': study.best_params['min_samples_leaf'],
                f"best_{error_metric}": study.best_value,
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
