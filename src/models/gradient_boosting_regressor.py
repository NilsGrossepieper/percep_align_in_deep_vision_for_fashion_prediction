"""
gradient_boosting_regressor.py

This module defines a Gradient Boosting regressor model for forecasting.
It supports both single-output (melted data) and multi-output scenarios.
"""

import logging
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


def gradient_boosting_regressor(
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    melt_data: bool,
    forecasting_seed: int,
    final_run: bool
) -> GradientBoostingRegressor | MultiOutputRegressor:
  """Initializes a Gradient Boosting regressor for forecasting.

  Args:
    n_estimators (int): Number of boosting stages.
    learning_rate (float): Learning rate for boosting.
    max_depth (int): Maximum depth of individual estimators.
    min_samples_split (int): Minimum samples to split an internal node.
    min_samples_leaf (int): Minimum samples at a leaf node.
    melt_data (bool): If True, use single-output regression.
    forecasting_seed (int): Random seed for reproducibility.
    final_run (bool): If True, uses all CPU cores.

  Returns:
    GradientBoostingRegressor | MultiOutputRegressor: Configured regressor instance.
  """
  logging.info(
    f"Initializing Gradient Boosting Regressor with: "
    f"{n_estimators=}, {learning_rate=}, {max_depth=}, "
    f"{min_samples_split=}, {min_samples_leaf=}, "
    f"{melt_data=}, {final_run=}, {forecasting_seed=}"
  )

  base_gb = GradientBoostingRegressor(
    loss='squared_error',
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    subsample=0.8,
    criterion='friedman_mse',
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    min_weight_fraction_leaf=0.0,
    max_depth=max_depth,
    min_impurity_decrease=0.0,
    init=None,
    random_state=forecasting_seed,
    max_features='sqrt',
    alpha=0.9,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False,
    tol=0.0001,
    ccp_alpha=0.0
  )

  # Return appropriate regressor
  if melt_data:
    return base_gb
  else:
    return MultiOutputRegressor(base_gb, n_jobs=(-1 if final_run else 1))
