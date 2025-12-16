"""
statistical_tests.py

This module provides statistical evaluation utilities for comparing forecasting models.

It implements:
  • Core error metric calculations (MAE, MSE, RMSE, MAPE, WAPE, R², Tracking Signal, etc.)
  • Paired bootstrap tests:
        - pooled version (sampling full products, pooling all rows)
        - per-product mean version (computing metrics per SKU, then aggregating)
  • Non-parametric Wilcoxon signed-rank test on per-product metrics

Each test quantifies whether a trained model significantly differs
from a baseline (vanilla) model in prediction accuracy.

Results are saved as CSV files for each predictive model, including
confidence intervals, p-values, and effect size measures.
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from scipy.stats import wilcoxon, rankdata


EPSILON = 1e-8
LIST_PRED_MODELS = ['gradient_boosting_regressor_model', 'knn_regressor_model',
                    'random_forest_regressor_model', 'ridge_regressor_model']
test_parameters = {
  'paired_bootstrap_pooled': 5000,
  'paired_bootstrap_per_product': 5000
}

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
        (MAE, MSE, RMSE, mape_pct, wape_pct, Tracking Signal, R-squared, Residual Std, Standard Std for y_pred and y_true)
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
  standard_std_y_pred= float(np.std(y_pred, ddof=1)) if y_pred.size > 1 else 0.0
  standard_std_y_true= float(np.std(y_true, ddof=1)) if y_true.size > 1 else 0.0

  return (
    round(mae, 3),
    round(mse, 3),
    round(rmse, 3),
    round(mape_pct, 3),
    round(wape_pct, 3),
    round(ts, 3),
    round(r2, 3),
    round(residual_std, 3),
    round(standard_std_y_pred, 3),
    round(standard_std_y_true, 3)
  )


def paired_bootstrap_pooled(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_vanilla: np.ndarray,
    y_pred_trained: np.ndarray,
    product_col: str = 'external_code',
    B: int = 5000,
    scalar: float = 53.0,
    random_state: int = 123,
):
    """
    Pooled paired-bootstrap test comparing two models (vanilla vs trained).
    Resamples products (with replacement), pools their rows, and computes global metrics.

    Returns:
        dict with arrays of bootstrap diffs and summary stats for MAE, MSE, RMSE, MAPE, WAPE
    """
    # Alignment and dtype checks
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_vanilla = np.asarray(y_pred_vanilla, dtype=np.float64).reshape(-1)
    y_pred_trained = np.asarray(y_pred_trained, dtype=np.float64).reshape(-1)

    if not np.array_equal(test_df["sales_sum"].to_numpy().reshape(-1), y_true):
        raise ValueError("test_df['sales_sum'] and y_true are not aligned.")
    if not (len(y_true) == len(y_pred_vanilla) == len(y_pred_trained) == len(test_df)):
        raise ValueError("Length mismatch among inputs.")

    # Keep immutable originals to sample from every replicate
    y_true0 = y_true.copy()
    y_v0 = y_pred_vanilla.copy()
    y_t0 = y_pred_trained.copy()

    # Group indices by product so resampling replicates full product rows
    groups = test_df.groupby(product_col).indices  # dict: product_id -> np.array(row_idx)
    product_ids = np.array(list(groups.keys()))
    n_products = len(product_ids)

    rng = np.random.default_rng(random_state)

    diffs_mae = np.empty(B, dtype=float)
    diffs_mse = np.empty(B, dtype=float)
    diffs_rmse = np.empty(B, dtype=float)
    diffs_mape = np.empty(B, dtype=float)
    diffs_wape = np.empty(B, dtype=float)

    for b in range(B):
        # Sample products WITH replacement; then concatenate their row indices
        sampled = rng.choice(product_ids, size=n_products, replace=True)
        idx_list = [groups[p] for p in sampled]
        idx = np.concatenate(idx_list)

        # Always index the ORIGINAL arrays; scale ONCE per replicate
        yt = y_true0[idx] * scalar
        yv = y_v0[idx] * scalar
        yt_hat = y_t0[idx] * scalar

        # MAE
        mae_v = mean_absolute_error(yt, yv)
        mae_t = mean_absolute_error(yt, yt_hat)
        diffs_mae[b] = mae_t - mae_v

        # MSE
        mse_v = mean_squared_error(yt, yv)
        mse_t = mean_squared_error(yt, yt_hat)
        diffs_mse[b] = mse_t - mse_v

        # RMSE
        rmse_v = np.sqrt(mean_squared_error(yt, yv))
        rmse_t = np.sqrt(mean_squared_error(yt, yt_hat))
        diffs_rmse[b] = rmse_t - rmse_v

        # MAPE
        mape_v = calc_mape_pct(yt, yv)
        mape_t = calc_mape_pct(yt, yt_hat)
        diffs_mape[b] = mape_t - mape_v

        # WAPE
        wape_v = calc_wape_pct(yt, yv)
        wape_t = calc_wape_pct(yt, yt_hat)
        diffs_wape[b] = wape_t - wape_v

    def summarize(diffs: np.ndarray):
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_left = np.mean(diffs <= 0.0)
        p_right = np.mean(diffs >= 0.0)
        p_two = min(1.0, 2 * min(p_left, p_right))
        return {
            'diffs': diffs,
            'ci': (float(lo), float(hi)),
            'tail_mass_left': float(p_left),   # one-sided: trained better if diff < 0
            'tail_mass_right': float(p_right),
            'p_two_sided': float(p_two),
        }

    return {
        'mae':  summarize(diffs_mae),
        'mse': summarize(diffs_mse),
        'rmse': summarize(diffs_rmse),
        'mape': summarize(diffs_mape),
        'wape': summarize(diffs_wape),
    }


def paired_bootstrap_per_product_mean(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_vanilla: np.ndarray,
    y_pred_trained: np.ndarray,
    product_col: str = 'external_code',
    B: int = 5000,
    scalar: float = 53.0,
    random_state: int = 123,
    metric: str = 'mae',           # 'mae' | 'mse' | 'rmse' | 'mape' | 'wape'
    weight_by: str | None = None,  # None | 'rows' | 'sales'  (for weighted mean across products)
):
    """
    Paired bootstrap where we compute a metric per product and then aggregate across products.
    """
    # Flatten & checks
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yv = np.asarray(y_pred_vanilla, dtype=np.float64).reshape(-1)
    yt = np.asarray(y_pred_trained, dtype=np.float64).reshape(-1)

    if not np.array_equal(test_df['sales_sum'].to_numpy().reshape(-1), y_true):
        raise ValueError("test_df['sales_sum'] and y_true misaligned.")
    if not (len(y_true) == len(yv) == len(yt) == len(test_df)):
        raise ValueError("Length mismatch among inputs.")

    # Precompute index lists per product
    groups = test_df.groupby(product_col).indices
    product_ids = np.array(list(groups.keys()))
    n_products = len(product_ids)

    # Optional weights per product
    if weight_by == 'rows':
        prod_w = {p: float(len(groups[p])) for p in product_ids}
    elif weight_by == 'sales':
        prod_w = {
            p: float(np.sum(np.abs(y_true[groups[p]] * scalar))) for p in product_ids
        }
    else:
        prod_w = {p: 1.0 for p in product_ids}

    rng = np.random.default_rng(random_state)
    diffs = np.empty(B, dtype=float)

    def per_product_metric(idx):
        yt_true = y_true[idx] * scalar
        yt_v    = yv[idx]     * scalar
        yt_t    = yt[idx]     * scalar

        if metric == 'mae':
            mv = np.mean(np.abs(yt_true - yt_v))
            mt = np.mean(np.abs(yt_true - yt_t))
        elif metric == 'mse':
            mv = np.mean((yt_true - yt_v) ** 2)
            mt = np.mean((yt_true - yt_t) ** 2)
        elif metric == 'rmse':
            mv = np.sqrt(np.mean((yt_true - yt_v) ** 2))
            mt = np.sqrt(np.mean((yt_true - yt_t) ** 2))
        elif metric == 'mape':
            mv = calc_mape_pct(yt_true, yt_v)
            mt = calc_mape_pct(yt_true, yt_t)
        elif metric == 'wape':
            mv = calc_wape_pct(yt_true, yt_v)
            mt = calc_wape_pct(yt_true, yt_t)
        else:
            raise ValueError("Unknown metric.")
        return mv, mt

    for b in range(B):
        # Resample products
        sampled = rng.choice(product_ids, size=n_products, replace=True)
        vals_v, vals_t, weights = [], [], []

        for p in sampled:
            idx = groups[p]
            mv, mt = per_product_metric(idx)
            vals_v.append(mv)
            vals_t.append(mt)
            weights.append(prod_w[p])

        vals_v = np.asarray(vals_v)
        vals_t = np.asarray(vals_t)
        weights = np.asarray(weights, dtype=float)

        if weight_by is None:
            agg_v = vals_v.mean()
            agg_t = vals_t.mean()
        else:
            # Weighted mean across products
            w = np.maximum(weights, EPSILON)
            agg_v = np.sum(vals_v * w) / np.sum(w)
            agg_t = np.sum(vals_t * w) / np.sum(w)

        diffs[b] = agg_t - agg_v  # negative => trained better for error metrics

    # Summarize
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_left  = np.mean(diffs <= 0.0)
    p_right = np.mean(diffs >= 0.0)
    p_two   = min(1.0, 2 * min(p_left, p_right))

    return {
        'diffs': diffs,
        'ci': (float(lo), float(hi)),
        'tail_mass_left': float(p_left),
        'tail_mass_right': float(p_right),
        'p_two_sided': float(p_two),
        'meta': {'metric': metric, 'weight_by': weight_by}
    }


def wilcoxon_per_product(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_vanilla: np.ndarray,
    y_pred_trained: np.ndarray,
    product_col: str = 'external_code',
    scalar: float = 53.0,
    metric: str = 'mae',         # 'mae' | 'mse' | 'rmse' | 'mape' | 'wape'
    alternative: str = 'two-sided',  # 'two-sided' | 'less' | 'greater'
):
    """
    Wilcoxon signed-rank test on per-product metric differences (trained - vanilla).
    Returns dict with p-value, statistic, effect size, and small diagnostics.
    """

    # ensure 1-D float arrays
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yv     = np.asarray(y_pred_vanilla, dtype=np.float64).reshape(-1)
    yt     = np.asarray(y_pred_trained, dtype=np.float64).reshape(-1)

    # alignment checks
    if not np.array_equal(test_df['sales_sum'].to_numpy().reshape(-1), y_true):
        raise ValueError("test_df['sales_sum'] and y_true misaligned.")
    if not (len(y_true) == len(yv) == len(yt) == len(test_df)):
        raise ValueError("Length mismatch among inputs.")

    # group row indices per product
    groups = test_df.groupby(product_col).indices
    product_ids = list(groups.keys())

    def per_product_metric(idx):
        yt_true = y_true[idx] * scalar
        yt_v    = yv[idx]     * scalar
        yt_t    = yt[idx]     * scalar

        if metric == 'mae':
            mv = np.mean(np.abs(yt_true - yt_v))
            mt = np.mean(np.abs(yt_true - yt_t))
        elif metric == 'mse':
            mv = np.mean((yt_true - yt_v) ** 2)
            mt = np.mean((yt_true - yt_t) ** 2)
        elif metric == 'rmse':
            mv = np.sqrt(np.mean((yt_true - yt_v) ** 2))
            mt = np.sqrt(np.mean((yt_true - yt_t) ** 2))
        elif metric == 'mape':
            mv = calc_mape_pct(yt_true, yt_v)   # keep consistent with your helpers
            mt = calc_mape_pct(yt_true, yt_t)
        elif metric == 'wape':
            mv = calc_wape_pct(yt_true, yt_v)
            mt = calc_wape_pct(yt_true, yt_t)
        else:
            raise ValueError("Unknown metric.")
        return mv, mt

    # compute per-product metrics
    mv_list, mt_list = [], []
    for p in product_ids:
        idx = groups[p]
        mv, mt = per_product_metric(idx)
        mv_list.append(mv)
        mt_list.append(mt)

    mv = np.asarray(mv_list, dtype=float)
    mt = np.asarray(mt_list, dtype=float)

    # paired differences (trained - vanilla); lower is better for error metrics
    d = mt - mv

    # drop exact zeros (Wilcoxon convention when zero_method='wilcox')
    mask_nz = (d != 0.0) & np.isfinite(d)
    d_nz = d[mask_nz]

    n_used = int(d_nz.size)
    if n_used == 0:
        # No non-zero pairs: no evidence of difference
        return {
            'n_pairs': int(len(d)),
            'n_used':  n_used,
            'median_diff': float(0.0),
            'statistic': float('nan'),
            'p_value': 1.0,
            'effect_size_rank_biserial': float(0.0),
            'alternative': alternative,
            'metric': metric
        }

    # Wilcoxon signed-rank p-value (SciPy handles exact/asymptotic as needed)
    stat, pval = wilcoxon(d_nz, zero_method='wilcox', alternative=alternative, correction=False)

    # Effect size (rank-biserial correlation)
    # Compute ranks of |d| and sum by sign
    ranks = rankdata(np.abs(d_nz), method='average')
    W_plus  = ranks[d_nz > 0].sum()
    W_minus = ranks[d_nz < 0].sum()
    r_rb = (W_plus - W_minus) / (W_plus + W_minus)  # ∈ [-1, 1]

    # Hodges–Lehmann estimator for paired case is simply the median of d
    median_diff = float(np.median(d))

    return {
        'n_pairs': int(len(d)),
        'n_used':  n_used,
        'median_diff': median_diff,
        'statistic': float(stat),
        'p_value': float(pval),
        'effect_size_rank_biserial': float(r_rb),
        'alternative': alternative,
        'metric': metric
    }


def test_for_significance(
    dict_load_vanilla: dict,
    dict_load_run_one: dict,
    dict_load_run_two: dict,
    dict_load_run_three: dict,
    experiment_name: str,
    vanilla_run_name: str,
    run_one_name: str,
    run_two_name: str,
    run_three_name: str,
    test_parameters: dict,
    random_seed: int,
    project_root: str,
    scalar: float = 53.0
):
  """
    Evaluate whether the trained models significantly outperform the vanilla baseline.

  This function loads saved models and their test data, computes global and weekly
  error metrics, and applies multiple statistical tests comparing the vanilla and
  trained model predictions.

  Specifically, it performs:
    • Global error metric calculation for vanilla and trained models
    • Pooled paired bootstrap test (global metric differences)
    • Per-product paired bootstrap test (average per SKU)
    • Wilcoxon signed-rank test (per-product paired differences)
    • Weekly performance analysis (without significance testing)

  All results are saved as CSV files inside:
      experiments/statistical_tests/<experiment_name>/

  Args:
      dict_load_vanilla (dict): Dictionary containing test data for the vanilla model.
      dict_load_run_one (dict): Dictionary containing test data for the first trained run.
      dict_load_run_two (dict): Dictionary containing test data for the second trained run.
      dict_load_run_three (dict): Dictionary containing test data for the third trained run.
      experiment_name (str): Name of the experiment for saving results.
      vanilla_run_name (str): Folder name of the vanilla model.
      run_one_name (str): Folder name of the first trained model.
      run_two_name (str): Folder name of the second trained model.
      run_three_name (str): Folder name of the third trained model.
      test_parameters (dict): Dictionary containing bootstrap parameters (e.g., number of samples).
      random_seed (int): Random seed for reproducibility.
      project_root (str): Root directory of the project.
      scalar (float, optional): Scaling factor for sales values. Defaults to 53.0.

  Returns:
      None. The function writes all result tables (global, bootstrap, Wilcoxon, weekly)
      as CSV files to the specified experiment directory and logs the save paths.
  """
  logging.info('Testing for significance...')

  # Load the df, X and y values
  test_df = dict_load_vanilla['data_frames']['test_df']
  y_true = dict_load_vanilla['test_df']['y_test']
  X_test_vanilla = dict_load_vanilla['test_df']['X_test']
  X_test_run_one = dict_load_run_one['test_df']['X_test']
  X_test_run_two = dict_load_run_two['test_df']['X_test']
  X_test_run_three = dict_load_run_three['test_df']['X_test']

  # Build output folder
  results_dir = os.path.join(project_root, 'experiments', 'statistical_tests', experiment_name)
  os.makedirs(results_dir, exist_ok=True)

  # Loop over the prediction models
  for pred_model in LIST_PRED_MODELS:

    # Load the prediction models
    vanilla_prediction_model_path = os.path.join(project_root, 'models', vanilla_run_name, f'{pred_model}.joblib')
    if not os.path.isfile(vanilla_prediction_model_path):
      raise FileNotFoundError(f"Prediction model file not found: {vanilla_prediction_model_path}")
    run_one_prediction_model_path = os.path.join(project_root, 'models', run_one_name, f'{pred_model}.joblib')
    if not os.path.isfile(run_one_prediction_model_path):
      raise FileNotFoundError(f"Prediction model file not found: {run_one_prediction_model_path}")
    run_two_prediction_model_path = os.path.join(project_root, 'models', run_two_name, f'{pred_model}.joblib')
    if not os.path.isfile(run_two_prediction_model_path):
      raise FileNotFoundError(f"Prediction model file not found: {run_two_prediction_model_path}")
    run_three_prediction_model_path = os.path.join(project_root, 'models', run_three_name, f'{pred_model}.joblib')
    if not os.path.isfile(run_three_prediction_model_path):
      raise FileNotFoundError(f"Prediction model file not found: {run_three_prediction_model_path}")

    # Get the prediction and average the three runs
    y_pred_vanilla = joblib.load(vanilla_prediction_model_path).predict(X_test_vanilla)
    y_pred_run_one = joblib.load(run_one_prediction_model_path).predict(X_test_run_one)
    y_pred_run_two = joblib.load(run_two_prediction_model_path).predict(X_test_run_two)
    y_pred_run_three = joblib.load(run_three_prediction_model_path).predict(X_test_run_three)
    y_pred_average = (y_pred_run_one + y_pred_run_two + y_pred_run_three) / 3

    # First calculate the global metrics
    (mae, mse, rmse, mape_pct, wape_pct, ts, r2, resid_std, standard_std_y_pred,
     standard_std_y_true) = calc_error_metrics(y_true.flatten(),
                                               y_pred_vanilla.flatten(), scalar)
    global_results_vanilla = {
      'mae': mae, 'mse': mse, 'rmse': rmse,
      'mape_pct': mape_pct, 'wape_pct': wape_pct, 'ts': ts,
      'r2': r2, 'residual_std': resid_std,
      'standard_std_y_pred': standard_std_y_pred, 'standard_std_y_true': standard_std_y_true
    }

    (mae, mse, rmse, mape_pct, wape_pct, ts, r2, resid_std, standard_std_y_pred,
     standard_std_y_true) = calc_error_metrics(y_true.flatten(),
                                               y_pred_average.flatten(), scalar)
    global_results_trained = {
      'mae': mae, 'mse': mse, 'rmse': rmse,
      'mape_pct': mape_pct, 'wape_pct': wape_pct, 'ts': ts,
      'r2': r2, 'residual_std': resid_std,
      'standard_std_y_pred': standard_std_y_pred, 'standard_std_y_true': standard_std_y_true
    }

    # Transform results into a df
    metric_cols = set(global_results_vanilla.keys())
    rows = [
      {'type': 'vanilla', **{k: global_results_vanilla.get(k, np.nan) for k in metric_cols}},
      {'type': 'trained', **{k: global_results_trained.get(k, np.nan) for k in metric_cols}},
      ]
    df = pd.DataFrame(rows)[['type'] + list(metric_cols)]

    # Save df
    out_path = os.path.join(results_dir, f"global_results_{pred_model}.csv")
    df.to_csv(out_path, index=False)
    logging.info(f"Saved global results at {out_path}")


    # Calculate the pooled paired boostrap test results
    results_dict = paired_bootstrap_pooled(
      test_df = test_df,
      y_true = y_true,
      y_pred_vanilla = y_pred_vanilla,
      y_pred_trained = y_pred_average,
      product_col = 'external_code',
      B = test_parameters['paired_bootstrap_pooled'],
      scalar = scalar,
      random_state = random_seed
    )

    # Extract summary stats from paired bootstrap
    rows = []
    for metric, stats in results_dict.items():
        lo, hi = stats['ci']
        rows.append({
            'metric': metric,
            'ci_lower': lo,
            'ci_upper': hi,
            'tail_mass_left': stats['tail_mass_left'],
            'tail_mass_right': stats['tail_mass_right'],
            'p_two_sided': stats['p_two_sided']
        })

    df_bootstrap = pd.DataFrame(rows)

    # Save as csv
    out_path = os.path.join(results_dir, f"paired_bootstrap_pooled_{pred_model}.csv")
    df_bootstrap.to_csv(out_path, index=False)
    logging.info(f"Saved paired bootstrap pooled results at {out_path}")


    # Paired bootstrap (per-product mean) — run for all metrics and save one CSV per model
    metrics_to_run = ['mae', 'mse', 'rmse', 'mape', 'wape']  # pick what you need

    rows = []
    for metric_name in metrics_to_run:
        res = paired_bootstrap_per_product_mean(
            test_df=test_df,
            y_true=y_true,
            y_pred_vanilla=y_pred_vanilla,
            y_pred_trained=y_pred_average,
            product_col='external_code',
            B=test_parameters['paired_bootstrap_per_product'],
            scalar=scalar,
            random_state=random_seed,
            metric=metric_name,
            weight_by=None,  # or 'sales' / 'rows' if you want a weighted average across products
        )
        lo, hi = res['ci']
        rows.append({
            'metric': metric_name,
            'ci_lower': lo,
            'ci_upper': hi,
            'tail_mass_left':  res['tail_mass_left'],
            'tail_mass_right': res['tail_mass_right'],
            'p_two_sided':     res['p_two_sided'],
            'n_bootstrap':     int(len(res['diffs'])),
            'aggregation':     'per_product_mean',
            'weight_by':       res['meta']['weight_by'] if 'meta' in res else None,
        })

    df_per_product = pd.DataFrame(rows)

    # Save as csv
    out_path = os.path.join(results_dir, f"paired_bootstrap_per_product_{pred_model}.csv")
    df_per_product.to_csv(out_path, index=False)
    logging.info(f"Saved paired bootstrap per-product results at {out_path}")


    # Wilcoxon per-product: run for selected metrics and save one CSV per model
    metrics_to_run = ['mae', 'rmse', 'mape', 'wape']  # choose what you want to report

    rows = []
    for metric_name in metrics_to_run:
        res = wilcoxon_per_product(
            test_df=test_df,
            y_true=y_true,
            y_pred_vanilla=y_pred_vanilla,
            y_pred_trained=y_pred_average,
            product_col='external_code',
            scalar=scalar,
            metric=metric_name,
            alternative='two-sided',   # or 'less'/'greater' if you want directional tests
        )
        rows.append({
            'metric': metric_name,
            'n_pairs': res['n_pairs'],
            'n_used': res['n_used'],  # pairs with non-zero diffs used by Wilcoxon
            'median_diff': res['median_diff'],
            'wilcoxon_statistic': res['statistic'],
            'p_two_sided': res['p_value'],
            'rank_biserial': res['effect_size_rank_biserial'],
            'alternative': res['alternative']
        })

    df_wilcoxon = pd.DataFrame(rows)

    # Save as csv (one file per model)
    out_path = os.path.join(results_dir, f"wilcoxon_per_product_{pred_model}.csv")
    df_wilcoxon.to_csv(out_path, index=False)
    logging.info(f"Saved Wilcoxon per-product results at {out_path}")


    # Second calculate the metrics per week
    results_vanilla = []
    results_trained = []

    # Check for correct size
    if not np.array_equal(test_df['sales_sum'], y_true):
      raise ValueError("test_df['sales_sum] and y_true must be the same.")

    weeks = np.sort(test_df['week'].astype(int).unique())
    for week in weeks:
      index = test_df.index[test_df['week'] == week].to_numpy()
      week_y_true = y_true[index]
      week_pred_vanilla = y_pred_vanilla[index]
      week_pred_trained = y_pred_average[index]

      (mae, mse, rmse, mape_pct, wape_pct, ts, r2, resid_std, standard_std_y_pred,
      standard_std_y_true) = calc_error_metrics(week_y_true, week_pred_vanilla, scalar)
      results_vanilla.append({'week': week+1,
                      'mae': mae,
                      'mse': mse,
                      'rmse': rmse,
                      'mape_pct': mape_pct,
                      'wape_pct': wape_pct,
                      'ts': ts,
                      'r2': r2,
                      'residual_std': resid_std,
                      'standard_std_y_pred': standard_std_y_pred,
                      'standard_std_y_true': standard_std_y_true})

      (mae, mse, rmse, mape_pct, wape_pct, ts, r2, resid_std, standard_std_y_pred,
      standard_std_y_true) = calc_error_metrics(week_y_true, week_pred_trained, scalar)
      results_trained.append({'week': week+1,
                      'mae': mae,
                      'mse': mse,
                      'rmse': rmse,
                      'mape_pct': mape_pct,
                      'wape_pct': wape_pct,
                      'ts': ts,
                      'r2': r2,
                      'residual_std': resid_std,
                      'standard_std_y_pred': standard_std_y_pred,
                      'standard_std_y_true': standard_std_y_true})

    # Save weekly metrics as results
    results_vanilla_csv_path = os.path.join(results_dir, f"weekly_vanilla_results_{pred_model}.csv")
    results_trained_csv_path = os.path.join(results_dir, f"weekly_trained_results_{pred_model}.csv")
    pd.DataFrame(results_vanilla).to_csv(results_vanilla_csv_path, index=False)
    pd.DataFrame(results_trained).to_csv(results_trained_csv_path, index=False)
    logging.info(f"Saved weekly vanilla results at {results_vanilla_csv_path}")
    logging.info(f"Saved weekly trained results at {results_trained_csv_path}")
