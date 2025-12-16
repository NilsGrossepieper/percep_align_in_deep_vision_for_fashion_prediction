"""
ridge_regressor.py

This module defines a Ridge regression model for forecasting.
It supports both single-output (melted data) and multi-output scenarios.
"""

import logging
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor


def ridge_regressor(
  alpha: float,
  melt_data: bool,
  forecasting_seed: int,
  final_run: bool
  ) -> Ridge | MultiOutputRegressor:
  """Initializes a Ridge regressor for forecasting.

  Args:
    alpha (float): Regularization strength.
    melt_data (bool): If True, use single-output regression.
    forecasting_seed (int): Random seed for reproducibility. Defaults to 42.
    final_run (bool): If True, uses all CPU cores. Defaults to False.

  Returns:
    Ridge | MultiOutputRegressor: Configured regressor instance.
    """
  logging.info(
      f"Initializing Ridge regressor with: "
      f"{alpha=}, {melt_data=}, {final_run=}"
      f"{forecasting_seed=}"
  )

  base_ridge = Ridge(
    alpha=alpha,
    fit_intercept=True,
    copy_X=True,
    max_iter=10000,
    tol=0.0001,
    solver='auto',
    positive=False,
    random_state=forecasting_seed
  )

  # Return appropriate regressor
  if melt_data:
      return base_ridge
  else:
      return MultiOutputRegressor(base_ridge, n_jobs=(-1 if final_run else 1))
