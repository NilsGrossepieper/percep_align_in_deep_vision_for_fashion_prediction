"""
test_metrics.py

Provides evaluation metrics and helper functions for forecasting models.
- Compute wape_pct and Tracking Signal
- Evaluate weekly metrics (melted or wide)
- Save CSV/JSON/NPY
- Optional Weights & Biases logging (one run per evaluation)
"""

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)
import wandb

EPSILON = 1e-8


def calc_mape_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculate Mean Absolute Percentage Error (mape_pct).

  Args:
    y_true (np.ndarray): Ground truth values.
    y_pred (np.ndarray): Predicted values.

  Returns:
    float: mape_pct value (percentage).
    """
  return 100.0 * np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, EPSILON)))


def calc_wape_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculate Weighted Absolute Percentage Error (wape_pct).

  Args:
    y_true (np.ndarray): Ground truth values.
    y_pred (np.ndarray): Predicted values.

  Returns:
    float: wape_pct value (percentage).
  """
  denom = np.sum(np.abs(y_true))
  return 100.0 * (np.sum(np.abs(y_true - y_pred)) / np.maximum(denom, EPSILON))


def cal_tracking_signal(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculate Tracking Signal for forecast bias detection.

  Args:
    y_true (np.ndarray): Ground truth values.
    y_pred (np.ndarray): Predicted values.

  Returns:
    float: Tracking signal value.
  """
  cumulative_error = np.sum(y_true - y_pred)
  mae = mean_absolute_error(y_true, y_pred)
  return cumulative_error / max(mae, EPSILON)


def calc_error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scalar: float = 53.0
) -> tuple[float, float, float, float, float, float, float, float, float]:
  """Calculate a set of error metrics for model evaluation.

  This function scales predictions and targets, then computes:
    - MAE, MSE, RMSE
    - mape_pct, wape_pct
    - Tracking Signal
    - R-squared
    - Residual Std, Standard Deviation of ground truth

  Args:
    y_true (np.ndarray): Ground truth values.
    y_pred (np.ndarray): Predicted values.
    scalar (float, optional): Scaling factor for sales values. Defaults to 53.0.

  Returns:
    tuple: Rounded metrics in the following order:
        (MAE, MSE, RMSE, mape_pct, wape_pct, Tracking Signal, R-squared, Residual Std, Standard Std)
  """
  # Scale values
  y_true, y_pred = y_true* scalar, y_pred * scalar

  # Core metrics
  mae = mean_absolute_error(y_true, y_pred)
  mse = mean_squared_error(y_true, y_pred)
  rmse = np.sqrt(mse)

  # Percentage-based metrics
  mape_pct = calc_mape_pct(y_true, y_pred)
  wape_pct = calc_wape_pct(y_true, y_pred)

  # Other metrics
  ts = cal_tracking_signal(y_true, y_pred) # Tracking Signal
  r2 = r2_score(y_true, y_pred)

  # standard deviations
  residual_std = float(np.std(y_true - y_pred, ddof=1)) if y_true.size > 1 else 0.0
  standard_std = float(np.std(y_true, ddof=1)) if y_true.size > 1 else 0.0

  return (
    round(mae, 3),
    round(mse, 3),
    round(rmse, 3),
    round(mape_pct, 3),
    round(wape_pct, 3),
    round(ts, 3),
    round(r2, 3),
    round(residual_std, 3),
    round(standard_std, 3)
  )


def evaluate_weekly_metrics(
    dict_model: dict,
    dict_data: dict,
    project_name: str,
    experiment_name: str,
    melt_data: bool,
    wandb_bool: bool,
    project_root: str,
    scalar: float = 53.0
) -> dict:
  """Evaluate forecasting model performance on a weekly basis.

  Steps:
  - Load model and predictions (or reuse cached results)
  - Compute weekly metrics (MAE, MSE, RMSE, etc.)
  - Compute average and overall global metrics
  - Save results to CSV, JSON, and NumPy formats

  Args:
      dict_model (dict): Dictionary with model info including saved model path.
      dict_data (dict): Dataset dictionary containing test data and DataFrames.
      project_name (str): Name of the project for folder structure.
      experiment_name (str): Name of the experiment for folder structure.
      melt_data (bool): Whether the data is in long (melted) format.
      wandb_bool (bool): Whether to log metrics to Weights & Biases.
      project_root (str, optional): Root directory of the project.
      scalar (float, optional): Scaling factor for sales numbers. Defaults to 53.0.

  Returns:
      dict: Contains per-week results, average and global metrics, and file paths.
  """
  logging.info(f"Evaluating metrics for model: {dict_model['model_path']}")

  # Build output folders
  results_dir = os.path.join(project_root, 'experiments', 'results', experiment_name)
  forecasts_dir = os.path.join(project_root, 'experiments', 'forecasts', experiment_name)
  os.makedirs(results_dir, exist_ok=True)
  os.makedirs(forecasts_dir, exist_ok=True)

  # Compose descriptive filenames (same structure as model saving)
  model_name = os.path.splitext(os.path.basename(dict_model['model_path']))[0]
  results_csv_path = os.path.join(results_dir, f"results_{model_name}.csv")
  avg_json_path = os.path.join(results_dir, f"avg_results_{model_name}.json")
  global_json_path = os.path.join(results_dir, f"global_results_{model_name}.json")
  y_pred_path = os.path.join(forecasts_dir, f"forecasts_{model_name}.npy")

   # Load cached results if they exist
  if all([os.path.exists(p) for p in [
    results_csv_path, avg_json_path, global_json_path, y_pred_path
  ]]):
    logging.info(f"Found cached results for '{model_name}', loading...")
    results = pd.read_csv(results_csv_path).to_dict(orient='records')
    with open(avg_json_path, 'r') as f:
      avg_results = json.load(f)
    with open(global_json_path, 'r') as f:
      global_results = json.load(f)
    y_pred = np.load(y_pred_path)
  else:
    # Load test data
    X_test = dict_data['test_df']['X_test']
    y_test = dict_data['test_df']['y_test']

    # Load and predict
    prediction_model = joblib.load(dict_model['model_path'])
    try:
      y_pred = prediction_model.predict(X_test)
    except Exception as e:
      logging.error(f"Prediction failed: {e}")
      raise RuntimeError(f"Prediction failed: {e}")

    results = []

    if melt_data:
      # Handle melted long-format data
      test_df = dict_data['data_frames']['test_df']
      if test_df is None:
        logging.error('test_df must be provided')
        raise ValueError('test_df must be provided if melt is True')

      if not np.allclose(test_df['sales_sum'].values, y_test):
        logging.error('test_df and y_test must be the same')
        raise ValueError('test_df and y_test must be the same')

      weeks = np.sort(test_df['week'].astype(int).unique())
      for week in weeks:
        index = test_df.index[test_df['week'] == week].to_numpy()
        week_y_true = y_test[index]
        week_pred = y_pred[index]

        (mae, mse, rmse, mape_pct, wape_pct, ts, r2, residual_std, standard_std
         ) = calc_error_metrics(week_y_true, week_pred, scalar)
        results.append({'week': week+1,
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'mape_pct': mape_pct,
                        'wape_pct': wape_pct,
                        'ts': ts,
                        'r2': r2,
                        'residual_std': residual_std,
                        'standard_std': standard_std})

    else:
      # Hanlde wide format
      if y_pred.shape != y_test.shape:
        raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y_test {y_test.shape}")
      weeks = y_test.shape[1]
      for week in range(weeks):
        week_y_true= y_test[:, week]
        week_pred = y_pred[:, week]
        (mae, mse, rmse, mape_pct, wape_pct, ts, r2, residual_std, standard_std
          ) = calc_error_metrics(week_y_true, week_pred, scalar)
        results.append({'week': week+1,
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'mape_pct': mape_pct,
                        'wape_pct': wape_pct,
                        'ts': ts,
                        'r2': r2,
                        'residual_std': residual_std,
                        'standard_std': standard_std})

    # Average metrics
    df_res = pd.DataFrame(results).sort_values('week')
    avg_results = {
      k: round(float(df_res[k].mean()), 3)
      for k in ['mae', 'mse', 'rmse', 'mape_pct', 'wape_pct', 'ts', 'r2', 'residual_std', 'standard_std']
    }

    # Global (absolute) metrics
    (mae, mse, rmse, mape_pct, wape_pct, ts, r2, resid_std, target_std) = calc_error_metrics(
      y_test.flatten(), y_pred.flatten(), scalar
    )
    global_results = {
      'mae': mae, 'mse': mse, 'rmse': rmse,
      'mape_pct': mape_pct, 'wape_pct': wape_pct, 'ts': ts,
      'r2': r2, 'residual_std': resid_std, 'standard_std': target_std
    }

    # Save everything
    pd.DataFrame(results).to_csv(results_csv_path, index=False)
    with open(avg_json_path, 'w') as f:
      json.dump(avg_results, f, indent=2)
    with open(global_json_path, 'w') as f:
      json.dump(global_results, f, indent=2)
    np.save(y_pred_path, y_pred)
    logging.info(
    f"Results saved: {results_csv_path}, {avg_json_path}, {global_json_path}, {y_pred_path}"
    )

  # log everything with wandb
  if wandb_bool:
    run_name = f"{experiment_name}__{model_name}__eval"
    with wandb.init(
      project=project_name,
      group=experiment_name,
      name=run_name,
      job_type='evaluate model',
      tags=[project_name, model_name, 'model_results', 'prediction'],
      notes=f"Evaluation of {model_name} for experiment {experiment_name}.",
      reinit=True
    ) as run:
      # define axes so weekly charts are clean
      wandb.define_metric('week')
      wandb.define_metric(f"{model_name}_weekly/*", step_metric='week')

      # weekly logs with explicit week step
      for row in results:
        week = int(row['week'])
        payload = {
          'week': week,
          f"{model_name}_weekly/mae": row['mae'],
          f"{model_name}_weekly/mse": row['mse'],
          f"{model_name}_weekly/rmse": row['rmse'],
          f"{model_name}_weekly/mape_pct": row['mape_pct'],
          f"{model_name}_weekly/wape_pct": row['wape_pct'],
          f"{model_name}_weekly/ts": row['ts'],
          f"{model_name}_weekly/r2": row['r2'],
          f"{model_name}_weekly/residual_std": row['residual_std'],
          f"{model_name}_weekly/standard_std": row['standard_std'],
        }
        wandb.log(payload)

      # average & global summaries
      wandb.run.log({f"{model_name}_avg/{k}": v for k, v in avg_results.items()})
      wandb.run.log({f"{model_name}_global/{k}": v for k, v in global_results.items()})

      # table
      table = wandb.Table(columns=[
        'week', 'mae', 'mse', 'rmse', 'mape_pct', 'wape_pct', 'ts', 'r2', 'residual_std', 'standard_std'
      ])
      for row in results:
        table.add_data(
          int(row['week']), row['mae'], row['mse'], row['rmse'],
          row['mape_pct'], row['wape_pct'], row['ts'],
          row['r2'], row['residual_std'], row['standard_std']
        )
      wandb.log({f"{model_name}_weekly_metrics_table": table})

      # artifact
      art = wandb.Artifact(f"{experiment_name}_{model_name}_results", type='results')
      art.add_file(results_csv_path)
      art.add_file(avg_json_path)
      art.add_file(global_json_path)
      art.add_file(y_pred_path)
      run.log_artifact(art)

  return {
      'results': results,
      'avg_results': avg_results,
      'global_results': global_results,
      'results_csv_path': results_csv_path,
      'avg_json_path': avg_json_path,
      'global_json_path': global_json_path,
      'y_pred_path': y_pred_path
  }
