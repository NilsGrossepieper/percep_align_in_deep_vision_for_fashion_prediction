"""
random_forest_regressor.py

This module defines a Random Forest regressor model for forecasting.
It supports both single-output (melted data) and multi-output scenarios.
"""

import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def random_forest_regressor(
  n_estimators: int,
  max_depth: int,
  min_samples_split: int,
  min_samples_leaf: int,
  melt_data: bool,
  forecasting_seed: int,
  final_run: bool
  ) -> RandomForestRegressor | MultiOutputRegressor:
  """Initializes a Random Forest regressor for forecasting.

  Args:
    n_estimators (int): Number of trees in the forest.
    max_depth (int): Maximum depth of the trees.
    min_samples_split (int): Minimum samples to split an internal node.
    min_samples_leaf (int): Minimum samples at a leaf node.
    melt_data (bool): If True, use single-output regression.
    forecasting_seed (int): Random seed for reproducibility.
    final_run (bool): If True, uses all CPU cores.

  Returns:
    RandomForestRegressor | MultiOutputRegressor: Configured regressor instance.
  """
  logging.info(
    f"Initializing Random Forest regressor with: "
    f"{n_estimators=}, {max_depth=}, {min_samples_split=}, "
    f"{min_samples_leaf=}, {melt_data=}, {final_run=}"
  )

  base_rf = RandomForestRegressor(
    n_estimators=n_estimators,
    criterion='squared_error',
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    min_weight_fraction_leaf=0.0,
    max_features='sqrt',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=(-1 if final_run else 1),
    random_state=forecasting_seed,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
  )

  # Return appropriate regressor
  if melt_data:
    return base_rf
  else:
    return MultiOutputRegressor(base_rf)
