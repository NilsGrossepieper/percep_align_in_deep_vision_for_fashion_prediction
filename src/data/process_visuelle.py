"""
process_visuelle.py

This script processes the Visuelle2.0 dataset for fashion sales forecasting.
Main tasks:
    - Load and validate raw data
    - Cap weekly sales values by stock availability
    - Create dummy variables for season, stores, and product metadata
    - Aggregate data to product level
    - Split into train/val/test sets for forecasting
    - Optionally melt data for single-week predictions
"""

import logging
import os

import numpy as np
import pandas as pd


def remove_invalid_rows(
    df: pd.DataFrame,
    week_cols: list[str]
    ) -> pd.DataFrame:
  """Removes rows with missing image paths or missing week sales data.

  Args:
      df (pd.DataFrame): Raw sales dataset.
      week_cols (list[str]): List of week column names (e.g., ['0', '1', ...]).

  Returns:
      pd.DataFrame: Cleaned dataframe with invalid rows removed.
  """
  logging.info("Removing rows with missing image paths or week sales values...")

  # Filter rows with valid image paths
  df = df[df['image_path'].notna() & (df['image_path'] != '')]

  # Filter rows with complete week sales values
  for col in week_cols:
      df = df[df[col].notna() & (df[col] != "")]

  return df


def cap_sales_by_stock(
    df: pd.DataFrame,
    week_cols: list[str],
    restock_col: str = 'restock'
    ) -> pd.DataFrame:
  """Caps weekly sales values based on available stock.

  Ensures that cumulative sales do not exceed the available stock.
  If sales exceed stock in a week, later weeks are set to zero.

  Args:
      df (pd.DataFrame): Sales dataframe.
      week_cols (list[str]): Week column names.
      restock_col (str, optional): Column with stock levels. Defaults to 'restock'.

  Returns:
      pd.DataFrame: Dataframe with capped weekly sales.
  """
  logging.info("Capping weekly sales values by stock availability...")

  df = df.copy()
  restocks = df[restock_col].values
  sales = df[week_cols].values
  sales = np.abs(sales)

  # Cap cumulative sales so it doesn't exceed restock quantity
  for i in range(sales.shape[0]):
    sales_cumsum = sales[i].cumsum()
    if sales_cumsum[-1] > restocks[i]:
      cap_idx = np.argmax(sales_cumsum > restocks[i])
      sold_before = sales_cumsum[cap_idx - 1] if cap_idx > 0 else 0
      remaining = max(restocks[i] - sold_before, 0)
      sales[i, cap_idx] = remaining
      # Zero out sales for subsequent weeks
      if cap_idx + 1 < sales.shape[1]:
        sales[i, cap_idx + 1:] = 0

  df.loc[:, week_cols] = sales
  return df


def add_store_columns(df: pd.DataFrame) -> pd.DataFrame:
  """Adds dummy variables for store presence and total store count.

  Ensures each product has correct store count representation and creates
  one-hot encoded columns for each store.

  Args:
      df (pd.DataFrame): Dataset with 'external_code' and 'retail' columns.

  Returns:
      pd.DataFrame: Dataset with store dummy variables and store count per product.

  Raises:
      ValueError: If aggregated store dummies do not match unique store counts.
  """
  logging.info("Creating store dummy variables and store counts...")

  # Remove duplicates to ensure one entry per product/store pair
  df_unique = df[['external_code', 'retail']].drop_duplicates()

  # Generate dummy variables for store presence
  store_dummies = pd.get_dummies(df_unique['retail'], prefix='store', dtype='uint8')
  df_unique = pd.concat([df_unique, store_dummies], axis=1)

  # Aggregate dummy variables to product level
  dummy_cols = [col for col in df_unique.columns if col.startswith('store_')]
  product_dummies = df_unique.groupby('external_code')[dummy_cols].max()
  product_store_sum = product_dummies.sum(axis=1)
  store_count_by_group = df_unique.groupby('external_code')['retail'].nunique()

  # Validate aggregated store counts
  if (product_store_sum != store_count_by_group).any():
    logging.error("Mismatch between store dummy sum and store count detected.")
    raise ValueError("Store dummy sum does not match store_count for some products.")

  # Merge dummy variables back into the main dataframe
  df['store_int'] = df['external_code'].map(store_count_by_group)
  df = df.merge(product_dummies, left_on='external_code', right_index=True, how='left')

  return df


def add_season_columns(df: pd.DataFrame) -> pd.DataFrame:
  """Adds dummy variables for fashion season and year extracted from the 'season' column.

  Args:
      df (pd.DataFrame): Dataset with 'season' column containing values like 'SS19'.

  Returns:
      pd.DataFrame: Dataset with one-hot encoded columns for fashion season and year.

  Raises:
      ValueError: If a product ends up with more than one season or year dummy set.
  """
  logging.info("Creating season and year dummy variables...")

  # Split season string into season type (SS/AW) and year
  split = df['season'].str.split('1', n=1, expand=True)
  df['fashion_season'] = split[0]
  df['year_str'] = '1' + split[1]
  df['year'] = df['year_str'].astype(int)

  # Generate dummy variables for season and year
  season_dummies = pd.get_dummies(df['fashion_season'], prefix='fs', dtype='uint8')
  year_dummies = pd.get_dummies(df['year'], prefix='year', dtype='uint8')
  df = pd.concat([df, season_dummies, year_dummies], axis=1)

  # Drop intermediate helper columns
  df = df.drop(columns=['fashion_season', 'year_str'])

  # Validate that each product has exactly one season and one year dummy
  for prefix in ['fs_', 'year_']:
      dummies = [col for col in df.columns if col.startswith(prefix)]
      if dummies:
          sums = df[dummies].sum(axis=1)
          if (sums != 1).any():
            logging.error(f"Multiple {prefix} dummies set for some products.")
            raise ValueError(
              f"Warning: Some products have more than one {prefix} dummy set! Check your input data!"
              "Check your input data!"
            )

  return df


def add_release_date_columns(df: pd.DataFrame) -> pd.DataFrame:
  """Adds dummy variables for release year, month, and week from 'release_date' column.

  Args:
      df (pd.DataFrame): Dataset containing a 'release_date' column.

  Returns:
      pd.DataFrame: Dataset with one-hot encoded columns for release year, month, and week.

  Raises:
      ValueError: If a product ends up with multiple dummies set for any release time unit.
  """
  logging.info("Creating release date dummy variables (year, month, week)...")

  # Convert release_date to datetime for feature extraction
  df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

  # Extract features
  df['release_year'] = df['release_date'].dt.year
  df['release_month'] = df['release_date'].dt.month
  df['release_week'] = df['release_date'].dt.isocalendar().week.astype(int)

  # One-hot encode release time units
  release_year_dummies = pd.get_dummies(df['release_year'], prefix='release_year', dtype='uint8')
  release_month_dummies = pd.get_dummies(df['release_month'], prefix='release_month', dtype='uint8')
  release_week_dummies = pd.get_dummies(df['release_week'], prefix='release_week', dtype='uint8')
  df = pd.concat([df, release_year_dummies, release_month_dummies, release_week_dummies], axis=1)

  # Validate that only one dummy is active for each temporal unit
  for prefix in ['release_year_', 'release_month_', 'release_week_']:
      dummies = [col for col in df.columns if col.startswith(prefix)]
      if dummies:
          sums = df[dummies].sum(axis=1)
          if (sums != 1).any():
            logging.error(f"Multiple {prefix} dummies set for some products.")
            raise ValueError(
              f"Warning: Some products have more than one {prefix} dummy set! "
                "Check your input data!"
              )

  return df


def add_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
  """Adds dummy variables for product metadata (category, color, fabric).

  Args:
      df (pd.DataFrame): Dataset containing 'category', 'color', and 'fabric' columns.

  Returns:
      pd.DataFrame: Dataset with one-hot encoded metadata columns.

  Raises:
      ValueError: If a product ends up with multiple dummies set for any metadata attribute.
  """
  logging.info("Creating dummy variables for product metadata (category, color, fabric)...")

  # One-hot encode metadata attributes
  category_dummies = pd.get_dummies(df['category'], prefix='cat', dtype='uint8')
  color_dummies = pd.get_dummies(df['color'], prefix='col', dtype='uint8')
  fabric_dummies = pd.get_dummies(df['fabric'], prefix='fab', dtype='uint8')
  df = pd.concat([df, category_dummies, color_dummies, fabric_dummies], axis=1)

  # Validate that only one dummy is active for each metadata type
  for prefix in ['cat_', 'col_', 'fab_']:
      dummies = [col for col in df.columns if col.startswith(prefix)]
      if dummies:
          sums = df[dummies].sum(axis=1)
          if (sums != 1).any():
            logging.error(f"Multiple {prefix} dummies set for some products.")
            raise ValueError(
              f"Warning: Some products have more than one {prefix} dummy set! "
              "Check your input data!"
            )

  return df


def aggregate_product_level(df: pd.DataFrame,
                            week_cols: list[str]
                            ) -> pd.DataFrame:
  """Aggregates data to product level with statistical summaries for weekly sales.

  Groups data by 'external_code' and computes multiple statistics (sum, mean,
  std, quantiles, etc.) for each weekly sales column. Also aggregates
  categorical and dummy columns appropriately.

  Args:
      df (pd.DataFrame): Dataset containing weekly sales and product metadata.
      week_cols (list[str]): List of week column names (e.g., ['0', '1', ...]).

  Returns:
      pd.DataFrame: Aggregated dataframe with product-level features and statistics.
  """
  logging.info("Aggregating data to product level with weekly sales statistics...")

  # Identify dummy columns for stores, seasons, years, and metadata
  store_dummies = [col for col in df.columns if col.startswith('store_') and col.split('_')[-1].isdigit()]
  season_dummies = [col for col in df.columns if col.startswith('fs_')]
  year_dummies = [col for col in df.columns if col.startswith('year_')]
  category_dummies = [col for col in df.columns if col.startswith('cat_')]
  color_dummies = [col for col in df.columns if col.startswith('col_')]
  fabric_dummies = [col for col in df.columns if col.startswith('fab_')]

  # Build aggregation dictionary for non-weekly columns
  agg_dict = {
      'season': 'first',
      'year': 'first',
      'release_date': 'first',
      'price': 'mean',
      'category': 'first',
      'color': 'first',
      'fabric': 'first',
      'image_path': 'first',
      'store_int': 'first',
      **{col: 'max' for col in store_dummies + season_dummies + year_dummies +
         category_dummies + color_dummies + fabric_dummies},
  }

  # Define statistics to compute for each weekly sales column
  week_stats = ['sum', 'mean', 'std', 'min',
                (lambda x: x.quantile(0.25)), 'median',
                (lambda x: x.quantile(0.75)), 'max']

  # Add weekly sales aggregation to dictionary
  for col in week_cols:
    agg_dict[col] = week_stats

  # Sort to ensure deterministic aggregation (earliest release_date is selected)
  df['release_date'] = pd.to_datetime(df['release_date'])
  df = df.sort_values(['external_code', 'release_date'])

  # Perform aggregation
  df_product = df.groupby('external_code').agg(agg_dict).reset_index()

  # Flatten MultiIndex columns from multiple aggregations
  new_cols = []
  lambda_counter = 0
  for col in df_product.columns:
    if isinstance(col, tuple):
      base, stat = col
      if stat == '<lambda_0>':
        stat_name = 'q25'
        lambda_counter += 1
      elif stat == '<lambda_1>':
        stat_name = 'q75'
        lambda_counter += 1
      elif stat == 'sum' or stat == 'mean' or stat == 'std' or stat == 'min' or stat == 'max' or stat == 'median':
        stat_name = stat
      elif stat == '':
        stat_name = ''
      else:
        stat_name = stat
      if stat_name:
        new_cols.append(f"{stat_name}_{base}" if base in week_cols else base)
      else:
        new_cols.append(base)
    else:
      new_cols.append(col)
  df_product.columns = new_cols

  # Reorder columns for logical structure
  week_stat_prefixes = ['sum', 'mean', 'std', 'min', 'q25', 'median', 'q75', 'max']
  week_stat_cols = []
  for stat in week_stat_prefixes:
      week_stat_cols += [f"{stat}_{w}" for w in week_cols if f"{stat}_{w}" in df_product.columns]

  ordered_cols = [
      'external_code',
      'season',
      'year',
      *sorted(season_dummies, reverse=True), # Ensure seasons like SS/AW appear in order
      *sorted(year_dummies),
      'release_date',
      'price',
      'category',
      'color',
      'fabric',
      *sorted(category_dummies),
      *sorted(color_dummies),
      *sorted(fabric_dummies),
      'image_path',
      'store_int',
      *sorted(store_dummies, key=lambda x: int(x.split('_')[-1])),
      *week_stat_cols
  ]

  # Only keep columns that exist in the dataframe
  ordered_cols = [c for c in ordered_cols if c in df_product.columns]
  df_product = df_product[ordered_cols]

  return df_product


def make_walk_forward_split(
    df: pd.DataFrame,
    target_season: str,
    seasons: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Creates train, validation, and test splits using a walk-forward strategy.

  The split is based on chronological order of release dates.
  The test set contains items from the target season.
  The validation set includes the most recent items from the two seasons
  before the target season.

  Args:
      df (pd.DataFrame): Full dataset with 'season' and 'release_date' columns.
      target_season (str): The season to use for the test set ('SS19' or 'AW19').
      seasons (list[str]): Ordered list of all seasons.

  Returns:
      tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
          Train, validation, and test datasets.

  Raises:
      ValueError: If target_season is invalid or
                  if there are fewer than two previous seasons for validation.
  """
  logging.info("Creating walk-forward train/validation/test splits...")

  if target_season not in ['SS19', 'AW19']:
    logging.error(f"Invalid target season: {target_season}. Only 'SS19' and 'AW19' are supported.")
    raise ValueError("target_season must be 'SS19' or 'AW19'.")

  # Sort data by release date
  df['release_date'] = pd.to_datetime(df['release_date'])
  df = df.sort_values(by='release_date').reset_index(drop=True)

  # Select data up to and including the target season
  idx_season = seasons.index(target_season)
  selected_seasons = seasons[:idx_season+1]
  total_data = df[df['season'].isin(selected_seasons)].copy()

  # Define test set (all items from target season)
  test_data = total_data[total_data['season'] == target_season].copy()
  test_indices = set(test_data.index)

  # Remaining data for train/validation
  trainval_data = total_data[~total_data.index.isin(test_indices)].copy()

  # Validation set size = 10% of total data
  val_size = int(len(total_data) * 0.1)

   # Ensure we have at least two previous seasons
  prev_seasons = seasons[max(0, idx_season-2):idx_season]
  if len(prev_seasons) < 2:
    logging.error("Need at least two previous seasons for validation split.")
    raise ValueError("Need at least two previous seasons for validation split.")

  # Split validation set evenly between the previous two seasons
  items_per_season = val_size // 2
  val_data_parts = []

  for prev_season in prev_seasons:
      season_items = trainval_data[trainval_data['season'] == prev_season].sort_values(by='release_date', ascending=False)
      val_season = season_items.head(items_per_season)
      val_data_parts.append(val_season)

  # Combine validation parts
  val_data = pd.concat(val_data_parts)
  val_indices = set(val_data.index)

  # Training set = all remaining items
  train_indices = set(trainval_data.index) - val_indices
  train_data = trainval_data.loc[list(train_indices)].copy()

  # Final sorting for reproducibility
  train_data = train_data.sort_values(by='release_date').reset_index(drop=True)
  val_data = val_data.sort_values(by='release_date').reset_index(drop=True)
  test_data = test_data.sort_values(by='release_date').reset_index(drop=True)

  logging.info(
    "Walk-forward split created: "
    f"{len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples."
  )

  return train_data, val_data, test_data


def make_release_date_split(
    df: pd.DataFrame,
    target_season: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Creates train, validation, and test splits based on release date percentages.

  The dataset is split into:
      - 10% test
      - 10% validation
      - Remaining for training

  If target season is SS19, AW19 items are excluded.

  Args:
      df (pd.DataFrame): Full dataset with 'season' and 'release_date' columns.
      target_season (str): The season to use for the test set ('SS19' or 'AW19').

  Returns:
      tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
          Train, validation, and test datasets.

  Raises:
      ValueError: If target_season is invalid.
  """
  logging.info("Creating release date train/validation/test splits...")

  valid_seasons = ['SS19', 'AW19']
  if target_season not in valid_seasons:
    logging.error(f"Invalid target season: {target_season}. Must be 'SS19' or 'AW19'.")
    raise ValueError("Invalid target season.")

  # Prepare dataset
  data = df[df['season'] != 'AW19'].copy() if target_season == 'SS19' else df.copy()
  data['release_date'] = pd.to_datetime(data['release_date'])
  data = data.sort_values(by='release_date').reset_index(drop=True)

  n_total = len(data)
  n_test = int(n_total * 0.10)
  n_val = int(n_total * 0.10)

  # Split into test, validation, train
  test_data = data.iloc[-n_test:].copy()
  val_data = data.iloc[-n_test-n_val:-n_test].copy()
  train_data = data.iloc[:-n_test-n_val].copy()

  # Final sorting for reproducibility
  train_data = train_data.sort_values(by='release_date').reset_index(drop=True)
  val_data = val_data.sort_values(by='release_date').reset_index(drop=True)
  test_data = test_data.sort_values(by='release_date').reset_index(drop=True)

  logging.info(
    f"Release-date split created: "
    f"{len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples."
  )

  return train_data, val_data, test_data


def make_standard_split(
    concat_merge_data: pd.DataFrame,
    target_season: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Creates train, validation, and test splits using a standard date-based strategy.

  The dataset is split into:
      - 10% test (latest release dates)
      - 10% validation (latest dates from remaining data)
      - Remaining for training

  If target season is SS19, AW19 items are excluded.

  Args:
      concat_merge_data (pd.DataFrame): Combined dataset with 'season' and 'release_date' columns.
      target_season (str): The season to use for the test set ('SS19' or 'AW19').

  Returns:
      tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
          Train, validation, and test datasets.

  Raises:
      ValueError: If target_season is invalid.
  """
  logging.info('Creating standard date-based train/validation/test split...')

  if target_season not in ['SS19', 'AW19']:
    logging.error(f"Invalid target season: {target_season}. Must be 'SS19' or 'AW19'.")
    raise ValueError('Invalid target season.')

  # Prepare dataset
  data = concat_merge_data[concat_merge_data['season'] != 'AW19'].copy() if target_season == 'SS19' else concat_merge_data.copy()
  data['release_date'] = pd.to_datetime(data['release_date'])
  data = data.sort_values('release_date').reset_index(drop=True)

  n = len(data)
  n_test = int(n * 0.1)
  n_val = int(n * 0.1)

  # Test split
  test_start_idx = max(0, min(n - n_test, n-1))
  first_test_date = data.iloc[test_start_idx]['release_date']
  test_mask = data['release_date'] >= first_test_date
  test_data = data[test_mask]
  data_left = data[~test_mask]

  # Validation split
  n_left = len(data_left)
  val_start_idx = max(0, min(n_left - n_val, n_left-1))
  first_val_date = data_left.iloc[val_start_idx]['release_date']
  val_mask = data_left['release_date'] >= first_val_date
  val_data = data_left[val_mask]
  train_data = data_left[~val_mask]

  # Final sorting by release date for reproducibility
  train_data = train_data.sort_values(by='release_date').reset_index(drop=True)
  val_data = val_data.sort_values(by='release_date').reset_index(drop=True)
  test_data = test_data.sort_values(by='release_date').reset_index(drop=True)

  logging.info(
    f"Standard split created: "
    f"{len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples."
  )

  return train_data, val_data, test_data


def make_week_based_split(
    concat_merge_data: pd.DataFrame,
    target_season: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """
  Creates train, validation, and test splits using a week-based strategy.

  The dataset is split into:
      - 10% test (latest release dates)
      - 10% validation (latest dates from remaining data)
      - Remaining for training
  """
  logging.info('Creating week data-based train/validation/test split...')

  if target_season not in ['SS19', 'AW19']:
    logging.error(f"Invalid target season: {target_season}. Must be 'SS19' or 'AW19'.")
    raise ValueError('Invalid target season.')

  # Prepare dataset
  data = concat_merge_data[concat_merge_data['season'] != 'AW19'].copy() if target_season == 'SS19' else concat_merge_data.copy()
  data['release_date'] = pd.to_datetime(data['release_date'])
  data = data.sort_values("release_date", ascending=True)
  data['sales_week'] = data['release_date'] + pd.to_timedelta(data['week'], unit='W')
  data = data.sort_values('sales_week').reset_index(drop=True)

  n = len(data)
  n_test = int(n * 0.1)
  n_val = int(n * 0.1)

  # Test split
  test_start_idx = max(0, min(n - n_test, n-1))
  first_test_date = data.iloc[test_start_idx]['sales_week']
  test_mask = data['sales_week'] >= first_test_date
  test_data = data[test_mask]
  data_left = data[~test_mask]

  # Validation split
  n_left = len(data_left)
  val_start_idx = max(0, min(n_left - n_val, n_left-1))
  first_val_date = data_left.iloc[val_start_idx]['sales_week']
  val_mask = data_left['sales_week'] >= first_val_date
  val_data = data_left[val_mask]
  train_data = data_left[~val_mask]

  # Final sorting by release date for reproducibility
  train_data = train_data.sort_values(by='sales_week').reset_index(drop=True)
  val_data = val_data.sort_values(by='sales_week').reset_index(drop=True)
  test_data = test_data.sort_values(by='sales_week').reset_index(drop=True)

  # Drop  the sales_week column again for consistency
  train_data = train_data.drop('sales_week', axis=1)
  val_data = val_data.drop('sales_week', axis=1)
  test_data = test_data.drop('sales_week', axis=1)

  logging.info(
    f"Week based split created: "
    f"{len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples."
  )

  return train_data, val_data, test_data



def make_default_split(data: pd.DataFrame):
  """
  Default split:
    - Test: last 5% from AW19 + last 5% from SS19  (strictly 2019; may be < target if 2019 is short)
    - Val : previous 5% from AW19 + previous 5% from SS19; if short, backfill from AW18/SS18 (latest first)
    - Train: remaining rows
  """
  data = data.copy()
  data['release_date'] = pd.to_datetime(data['release_date'])
  # Establish a single global index and stable chronological order
  data = data.sort_values('release_date').reset_index(drop=True)

  per_season_len = int(len(data) * 0.05)
  if per_season_len <= 0:
      raise ValueError("Dataset too small for 5% per-season val/test splits.")

  # Season slices (already in chronological order due to global sort)
  aw19 = data[data['season'] == 'AW19']
  ss19 = data[data['season'] == 'SS19']
  aw18 = data[data['season'] == 'AW18']
  ss18 = data[data['season'] == 'SS18']

  # TEST (strictly 2019)
  test_aw_idx = aw19.index[-per_season_len:] if len(aw19) >= per_season_len else aw19.index
  test_ss_idx = ss19.index[-per_season_len:] if len(ss19) >= per_season_len else ss19.index
  test_idx = test_aw_idx.union(test_ss_idx)

  # Remove test picks from 2019 before constructing validation
  aw19_rem = aw19.loc[aw19.index.difference(test_aw_idx)]
  ss19_rem = ss19.loc[ss19.index.difference(test_ss_idx)]

  # VALIDATION (2019 preceding test; backfill from 2018 if needed)
  # AW side
  val_aw19_take = aw19_rem.index[-per_season_len:]
  shortage_aw = per_season_len - len(val_aw19_take)
  if shortage_aw > 0:
      backfill_aw18 = aw18.index[-shortage_aw:]
      val_aw_idx = val_aw19_take.append(backfill_aw18)
  else:
      val_aw_idx = val_aw19_take

  # SS side
  val_ss19_take = ss19_rem.index[-per_season_len:]
  shortage_ss = per_season_len - len(val_ss19_take)
  if shortage_ss > 0:
      backfill_ss18 = ss18.index[-shortage_ss:]
      val_ss_idx = val_ss19_take.append(backfill_ss18)
  else:
      val_ss_idx = val_ss19_take

  val_idx = val_aw_idx.union(val_ss_idx)
  # Ensure no overlap between val and test
  val_idx = val_idx.difference(test_idx)

  # TRAIN = everything else
  train_idx = data.index.difference(test_idx.union(val_idx))

  test_data  = data.loc[test_idx].sort_values('release_date').reset_index(drop=True)
  val_data   = data.loc[val_idx].sort_values('release_date').reset_index(drop=True)
  train_data = data.loc[train_idx].sort_values('release_date').reset_index(drop=True)

  return train_data, val_data, test_data


def melt_and_merge_week_stats(
    df: pd.DataFrame,
    n_weeks: int = 12
) -> pd.DataFrame:
  """Melt and merge weekly sales statistics into a structured DataFrame.

  For each product and week, this function:
      - Melts statistics (sum, mean, median, quantiles, etc.) from wide to long format
      - Merges these melted stats into a single DataFrame
      - Adds dummy variables for weeks
      - Ensures proper ordering of columns

  Args:
      df (pd.DataFrame): DataFrame containing weekly stats columns for multiple weeks.
      n_weeks (int, optional): Number of weekly columns for each statistic. Defaults to 12.

  Returns:
      pd.DataFrame: A DataFrame with merged weekly stats and week dummy variables.

  Raises:
      ValueError: If multiple week dummies are set for the same product-week combination.
  """
  logging.info("Melting and merging weekly sales statistics...")

  # Define the stats and output column names
  week_stats = [
    ('sum',    'sales_sum'),
    ('std',    'sales_std'),
    ('mean',   'sales_mean'),
    ('min',    'sales_min'),
    ('q25',    'sales_q1'),
    ('median', 'sales_median'),
    ('q75',    'sales_q3'),
    ('max',    'sales_max'),
  ]

  # Build all column names to exclude from id_cols
  all_week_cols = {prefix: [f"{prefix}_{i}" for i in range(n_weeks)] for prefix, _ in week_stats}
  exclude_cols = set(c for cols in all_week_cols.values() for c in cols)
  id_cols = [col for col in df.columns if col not in exclude_cols]

  # Melt each stat and collect in a dictionary
  melted_dfs = {}
  for prefix, out_name in week_stats:
    melt = pd.melt(
      df,
      value_vars=all_week_cols[prefix],
      id_vars=id_cols,
      var_name='week',
      value_name=out_name
      )
    melt['week'] = melt['week'].str.extract(r'(\d+)$').astype(int)

    # Keep full columns for sum, minimal for other stats
    if prefix == 'sum':
      keep = [col for col in melt.columns if col not in exclude_cols]
    else:
      keep = ['external_code', 'week', out_name]
    melted_dfs[out_name] = melt[keep]

  # Merge all melted DataFrames step by step
  from functools import reduce
  merged = reduce(lambda left, right: pd.merge(left, right, on=['external_code', 'week'], how='inner'), melted_dfs.values())

  # Add week dummies
  week_dummies = pd.get_dummies(merged['week'], prefix='week', dtype='uint8')
  merged = pd.concat([merged, week_dummies], axis=1)

  # Validate week dummies
  week_dummy_cols = [col for col in merged.columns if col.startswith('week_')]
  if week_dummy_cols:
    sums = merged[week_dummy_cols].sum(axis=1)
    if (sums != 1).any():
      logging.error("Multiple week dummies set for some products.")
      raise ValueError(
        "Some products have more than one week dummy set. "
        "Check your input data."
      )

  # Sort rows
  merged = merged.sort_values(by=['external_code', 'week'])

  # Order columns for output
  store_dummy_cols = [
    col for col in merged.columns
    if col.startswith('store_') and col.split('_')[-1].isdigit()
  ]
  store_dummy_cols = sorted(store_dummy_cols, key=lambda x: int(x.split('_')[-1]))

  ordered_cols = [
    'external_code',
    'season',
    'year',
    *sorted([col for col in merged.columns if col.startswith('fs_')], reverse=True),
    *sorted([col for col in merged.columns if col.startswith('year_')]),
    'release_date',
    'price',
    'category',
    'color',
    'fabric',
    *sorted([col for col in merged.columns if col.startswith('cat_')]),
    *sorted([col for col in merged.columns if col.startswith('col_')]),
    *sorted([col for col in merged.columns if col.startswith('fab_')]),
    'image_path',
    'store_int',
    *store_dummy_cols,
    'week',
    *sorted([col for col in merged.columns if col.startswith('week_')]),
    'sales_sum',
    'sales_std',
    'sales_mean',
    'sales_min',
    'sales_q1',
    'sales_median',
    'sales_q3',
    'sales_max'
  ]
  ordered_cols = [c for c in ordered_cols if c in merged.columns]
  merged = merged[ordered_cols]

  logging.info("Melt and merge completed successfully.")

  return merged


VALID_SEASONS = ['SS17', 'AW17', 'SS18', 'AW18', 'SS19', 'AW19']
ALLOWED_SPLIT_METHODS = ['season', 'release_date', 'standard', 'week', 'default']
WEEK_COLUMNS = [str(i) for i in range(12)]


def process_visuelle(
    season: str,
    split_method: str,
    melt_data: bool,
    project_root: str
) -> dict:
  """Main function to process the Visuelle2 dataset for forecasting experiments.

  This function:
      - Validates inputs
      - Loads and cleans data
      - Adds store, season, and metadata columns
      - Aggregates data at product level
      - Splits dataset into train, validation, and test sets
      - Optionally melts data for single-week predictions
      - Saves preprocessed files

  Args:
      season (str): Forecasting season ('SS19' or 'AW19').
      split_method (str): Splitting method ('season', 'release_date', 'standard').
      melt_data (bool): Whether to melt data for single-week predictions.
      project_root (str): Root directory of the project.

  Returns:
      dict: Dictionary with paths to image DataFrame, scalar, and dataset CSVs.

  Raises:
      ValueError: If inputs are invalid or cleaned data is empty.
      FileNotFoundError: If required scalar or dataset files are missing.
      KeyError: If required columns are missing from the dataset.
  """
  logging.info("Starting Visuelle2 data processing...")

  # Validate forecasting season
  if season not in VALID_SEASONS[-2:] + ['None']:
    logging.error(f"Invalid forecasting season: {season}")
    raise ValueError("Forecasting season must be 'SS19' or 'AW19'.")

  # Validate split method
  if split_method not in ALLOWED_SPLIT_METHODS:
    logging.error(f"Invalid split method: {split_method}")
    raise ValueError(f"Split method must be one of {ALLOWED_SPLIT_METHODS}.")

  # Dataset and scalar paths
  dataset_dir = os.path.join(project_root, 'datasets', 'visuelle2')
  scalar_path = os.path.join(dataset_dir, 'stfore_sales_norm_scalar.npy')
  os.makedirs(dataset_dir, exist_ok=True)

  if not os.path.isfile(scalar_path):
    logging.error(f"Normalization scalar file not found: {scalar_path}")
    raise FileNotFoundError(f"Normalization scalar file not found: {scalar_path}.")

  scalar = np.load(scalar_path)
  scalar = scalar.item() if np.isscalar(scalar) or scalar.size == 1 else scalar

  # Experiment paths
  processed_dir = os.path.join(dataset_dir, 'processed_data')
  experiment_name = f"melt_{str(melt_data).lower()}_{season.lower()}_{split_method}_split"
  experiment_dir = os.path.join(processed_dir, experiment_name)
  os.makedirs(experiment_dir, exist_ok=True)

  train_data_path = os.path.join(experiment_dir, 'visuelle2_train.csv')
  val_data_path = os.path.join(experiment_dir, 'visuelle2_val.csv')
  test_data_path = os.path.join(experiment_dir, 'visuelle2_test.csv')
  image_df_path = os.path.join(processed_dir, 'image_df.csv')

  output_files = [train_data_path, val_data_path, test_data_path, image_df_path]

  # If processed files already exist
  if all(os.path.isfile(f) for f in output_files):
    logging.info('All processed files already exist. Skipping processing.')
  else:
    logging.info('Processing raw data...')

    train_csv_path = os.path.join(dataset_dir, 'stfore_train.csv')
    test_csv_path = os.path.join(dataset_dir, 'stfore_test.csv')
    price_csv_path = os.path.join(dataset_dir, 'price_discount_series.csv')

    # Validate raw files
    if not os.path.isfile(train_csv_path):
      logging.error(f"Training CSV not found: {train_csv_path}")
      raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")
    if not os.path.isfile(test_csv_path):
      logging.error(f"Test CSV not found: {test_csv_path}")
      raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")
    if not os.path.isfile(price_csv_path):
      logging.error(f"Price CSV not found: {price_csv_path}")
      raise FileNotFoundError(f"Price CSV not found: {price_csv_path}")

    # Load raw data
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)
    price_data = pd.read_csv(price_csv_path)
    price_data = price_data[['external_code', 'retail', 'price']]

    concat_data = pd.concat([train_data, test_data], ignore_index=True)
    concat_merge_data = pd.merge(concat_data,
                                 price_data,
                                 on=['external_code',
                                     'retail'],
                                 how='left',
                                 validate='m:1')

    if not concat_data.equals(concat_merge_data.drop(columns=['price'])):
      logging.error('Error merging datasets.')
      raise ValueError('Error merging datasets.')

    # Validate required columns
    required_columns = [
        'image_path', 'external_code', 'season', 'category',
        'color', 'fabric', 'release_date', 'price', 'restock'
        ]
    for col in required_columns + WEEK_COLUMNS:
        if col not in concat_merge_data.columns:
          logging.error(f"Missing column: {col}")
          raise KeyError(f"Column '{col}' missing in loaded CSV!")

    # Clean invalid rows
    concat_merge_data = remove_invalid_rows(concat_merge_data, WEEK_COLUMNS)
    if concat_merge_data.empty:
      logging.error("Dataset is empty after cleaning.")
      raise ValueError("After cleaning, the dataset is empty.")

    # Apply transformations
    concat_merge_data = cap_sales_by_stock(concat_merge_data, WEEK_COLUMNS, restock_col='restock')
    concat_merge_data = add_store_columns(concat_merge_data)
    concat_merge_data = add_season_columns(concat_merge_data)
    # concat_merge_data = add_release_date_columns(concat_merge_data)
    concat_merge_data = add_meta_columns(concat_merge_data)
    concat_merge_data = aggregate_product_level(concat_merge_data, WEEK_COLUMNS)

    concat_merge_data['release_date'] = pd.to_datetime(concat_merge_data['release_date'])

    # Split datasets
    if split_method == 'season':
      train_data, val_data, test_data = make_walk_forward_split(concat_merge_data, season, VALID_SEASONS)
    elif split_method == 'release_date':
      train_data, val_data, test_data = make_release_date_split(concat_merge_data, season)
    elif split_method == 'standard':
      train_data, val_data, test_data = make_standard_split(concat_merge_data, season)
    elif split_method == 'week':
      raise ValueError('Week-based split not supported anymore.')
    #  train_data, val_data, test_data = make_week_based_split(concat_merge_data, season)
    elif split_method == 'default':
      train_data, val_data, test_data = make_default_split(concat_merge_data)

    # Melt data for single-week predictions
    if melt_data:
      train_data = melt_and_merge_week_stats(train_data)
      val_data = melt_and_merge_week_stats(val_data)
      test_data = melt_and_merge_week_stats(test_data)

      train_data = train_data.sort_values(by=['external_code', 'week'])
      val_data = val_data.sort_values(by=['external_code', 'week'])
      test_data = test_data.sort_values(by=['external_code', 'week'])

    else:
      train_data = train_data.sort_values(by='external_code')
      val_data = val_data.sort_values(by='external_code')
      test_data = test_data.sort_values(by='external_code')

    logging.info(
    f"Split sizes — Train: {len(train_data)}, "
    f"Val: {len(val_data)}, Test: {len(test_data)}"
    )

    # Save image DataFrame if not exists
    if not os.path.isfile(image_df_path):
      image_paths = concat_merge_data['image_path'].unique()
      image_df = pd.DataFrame({'image_path': image_paths})
      image_df.to_csv(image_df_path, index=False)

    # Save datasets
    train_data.to_csv(train_data_path, index=False)
    val_data.to_csv(val_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)
    logging.info(f"Saved preprocessed datasets in {experiment_name} folder.")

  return {
    'image_df_path': image_df_path,
    'scalar': scalar,
    'dataset_paths': {
      'train_data_path': train_data_path,
      'val_data_path': val_data_path,
      'test_data_path': test_data_path
    }
  }
