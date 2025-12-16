"""
knn_regressor.py

This module defines a K-Nearest Neighbors (KNN) regressor model for forecasting.
"""

import logging
from sklearn.neighbors import KNeighborsRegressor


def knn_regressor(
  n_neighbors: int,
  weights: str,
  final_run: bool,
) -> KNeighborsRegressor:
  """Initializes a K-Nearest Neighbors regressor.

  Args:
    n_neighbors (int): Number of neighbors to use.
    weights (str): Weight function ('uniform' or 'distance').
    final_run (bool): If True, use all CPU cores.

  Returns:
    KNeighborsRegressor: Configured KNN regressor instance.
  """
  n_jobs = -1 if final_run else 1

  logging.info(
    f"Initializing KNN regressor with: "
    f"{n_neighbors=}, {weights=}, {final_run=}"
  )

  return KNeighborsRegressor(
    n_neighbors=n_neighbors,
    weights=weights,
    algorithm='auto',
    leaf_size=30,
    metric='cosine',
    metric_params=None,
    n_jobs=n_jobs
  )
