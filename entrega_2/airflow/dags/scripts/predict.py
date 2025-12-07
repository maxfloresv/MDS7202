import pandas as pd
import numpy as np
import joblib
from sklearn.base import clone
from sklearn.metrics import f1_score

def find_best_threshold_binary(y_true, y_probs):
  """
  Finds the threshold that maximizes Binary F1 Score.

  Parameters
  ----------
  y_true : array-like
    The true validation labels.
  y_probs : array-like
    The predicted probabilities for the validation set.
  """
  best_f1 = 0
  best_thresh = 0.5
  
  for thresh in np.arange(0.3, 0.96, 0.01):
    y_pred_temp = (y_probs >= thresh).astype(int)
    score = f1_score(y_true, y_pred_temp, average='binary')
    
    if score > best_f1:
      best_f1 = score
      best_thresh = thresh
          
  return best_thresh, best_f1

def generate_test_data(W: int, Y: int, **kwargs):
  """
  Generates test data with full advanced Feature Engineering.

  Parameters
  ----------
  W : int
    The week to generate test data for.
  Y : int
    The year to generate test data for.
  """
  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')

  df_path = ti.xcom_pull(key='final_df_path')
  df = pd.read_parquet(df_path, engine='pyarrow')

  last_known_year = df['year'].max()
  last_known_week = df[df['year'] == last_known_year]['week'].max()

  mask = (df['year'] == last_known_year) & (df['week'] == last_known_week)
  
  static_cols = [
    'customer_id', 
    'product_id', 
    'X', 
    'Y', 
    'num_deliver_per_week', 
    'size', 
    'customer_type', 
    'category', 
    'sub_category', 
    'segment', 
    'package', 
    'brand'
  ]
  candidates = df[mask][static_cols].copy()

  candidates = candidates.drop_duplicates(
    subset=['customer_id', 'product_id'], 
    keep='last'
  )

  candidates['week'] = W
  candidates['year'] = Y
  candidates['items'] = np.nan
  candidates['label'] = np.nan

  df_window = pd.concat([df, candidates], ignore_index=True)
  df_window = df_window.sort_values(by=[
    'customer_id', 'product_id', 'year', 'week'
  ])

  df_window['week_abs'] = df_window['year'].astype(int) * 52 + df_window['week'].astype(int)

  # Recency (R)
  df_window['week_of_purchase'] = np.where(df_window['items'] > 0, df_window['week_abs'], np.nan)
  df_window['last_purchase_week_abs'] = df_window.groupby(
    ['customer_id', 'product_id']
  )['week_of_purchase'].ffill()
  df_window['last_purchase_week_abs_shifted'] = df_window.groupby(
    ['customer_id', 'product_id']
  )['last_purchase_week_abs'].shift(1)
  df_window['weeks_since_last_purchase'] = (
    df_window['week_abs'] - df_window['last_purchase_week_abs_shifted']
  )
  df_window['weeks_since_last_purchase'] = df_window['weeks_since_last_purchase'].fillna(1000).astype(int)
  
  df_window.drop(columns=[
    'week_of_purchase',
    'last_purchase_week_abs',
    'last_purchase_week_abs_shifted'
  ], inplace=True)

  # Frequency (F) & Monetary (M)
  df_window['accumulated_purchase_count'] = df_window.groupby(
    ['customer_id', 'product_id']
  )['label'].transform(
    lambda x: x.shift(1).expanding().sum()
  ).fillna(0)
  df_window['accumulated_items_volume'] = df_window.groupby(
    ['customer_id', 'product_id']
  )['items'].transform(
      lambda x: x.shift(1).expanding().sum()
  ).fillna(0)

  # Periodicity (P)
  purchases_only = df_window[df_window['items'] > 0].copy()
  purchases_only = purchases_only.sort_values(by=[
    'customer_id', 'product_id', 'week_abs'
  ])
  
  purchases_only['prev_purchase_week'] = purchases_only.groupby(
    ['customer_id', 'product_id']
  )['week_abs'].shift(1)
  purchases_only['inter_purchase_gap'] = (
    purchases_only['week_abs'] - purchases_only['prev_purchase_week']
  )
  
  purchases_only['gap_mean'] = purchases_only.groupby(
    ['customer_id', 'product_id']
  )['inter_purchase_gap'].transform(lambda x: x.expanding().mean())
  purchases_only['gap_std'] = purchases_only.groupby(
    ['customer_id', 'product_id']
  )['inter_purchase_gap'].transform(lambda x: x.expanding().std())

  periodicity_cols = [
    'customer_id', 
    'product_id', 
    'week_abs', 
    'gap_mean', 
    'gap_std'
  ]
  df_window = df_window.merge(
    purchases_only[periodicity_cols], 
    on=['customer_id', 'product_id', 'week_abs'], 
    how='left', 
    suffixes=('', '_new')
  )

  df_window['periodicity_mean'] = df_window.groupby(
    ['customer_id', 'product_id']
  )['gap_mean'].transform(lambda x: x.ffill().shift(1)).fillna(-1)
  df_window['periodicity_std'] = df_window.groupby(
    ['customer_id', 'product_id']
  )['gap_std'].transform(
    lambda x: x.ffill().shift(1)
  ).fillna(-1)

  df_window['items_lag_1'] = df_window.groupby(['customer_id', 'product_id'])['items'].shift(1)  
  df_window['items_rolling_mean_4w'] = df_window.groupby(
    ['customer_id', 'product_id']
  )['items'].transform(
    lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
  )
  df_window['items_rolling_mean_12w'] = df_window.groupby(
    ['customer_id', 'product_id']
  )['items'].transform(
    lambda x: x.shift(1).rolling(window=12, min_periods=1).mean()
  )
  df_window['items_expanding_mean'] = df_window.groupby(
    ['customer_id', 'product_id']
  )['items'].transform(
    lambda x: x.shift(1).expanding().mean()
  )
  df_window['purchased_lag_1'] = df_window.groupby(
    ['customer_id', 'product_id']
  )['label'].shift(1)

  fill_cols = [
    'items_lag_1', 
    'items_rolling_mean_4w', 
    'items_rolling_mean_12w', 
    'items_expanding_mean', 
    'purchased_lag_1'
  ]

  df_window.drop(columns=[
    'week_abs',
    'gap_mean',
    'gap_std'
  ], inplace=True)
  df_window[fill_cols] = df_window[fill_cols].fillna(0)

  X_test = df_window[(df_window['year'] == Y) & (df_window['week'] == W)].copy()

  X_test['week_sin'] = np.sin(2 * np.pi * W / 52)
  X_test['week_cos'] = np.cos(2 * np.pi * W / 52)

  X_full_path = ti.xcom_pull(key='X_full_path')
  X_full_schema = pd.read_parquet(X_full_path, engine='pyarrow')[:0]
  objective_columns = X_full_schema.columns.tolist()
  X_test = X_test[objective_columns]
  
  splits_dir = f"{base_dir}/splits"
  X_test_path_out = f"{splits_dir}/X_test_week_{W}_year_{Y}.parquet"
  X_test.to_parquet(X_test_path_out, engine='pyarrow', index=False)
  ti.xcom_push(key='X_test_path', value=X_test_path_out)

def generate_week_predictions(**kwargs):
  """
  Generates the predictions for the week using the optimized model.
  """
  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')

  X_test_path = ti.xcom_pull(key='X_test_path')
  X_test = pd.read_parquet(X_test_path, engine='pyarrow')

  optimized_model_path = ti.xcom_pull(key='optimized_trained_model_path')
  optimized_model = joblib.load(optimized_model_path)

  X_full_path = ti.xcom_pull(key='X_full_path')
  X_full = pd.read_parquet(X_full_path, engine='pyarrow')
  
  y_full_path = ti.xcom_pull(key='y_full_path')
  y_full = pd.read_parquet(y_full_path, engine='pyarrow').iloc[:, 0]

  last_row = X_full.iloc[-1]

  target_year = last_row['year']
  target_sin = last_row['week_sin']
  target_cos = last_row['week_cos']

  year_mask = (X_full['year'] == target_year)
  week_mask = (
    np.isclose(X_full['week_sin'], target_sin, atol=1e-4) 
    & np.isclose(X_full['week_cos'], target_cos, atol=1e-4)
  )
  mask = year_mask & week_mask

  # Uses the last period available to validate the model.
  X_shadow_train = X_full[~mask]
  y_shadow_train = y_full[~mask]
  X_shadow_val = X_full[mask]
  y_shadow_val = y_full[mask]

  shadow_model = clone(optimized_model)
  shadow_model.fit(X_shadow_train, y_shadow_train)

  probs_val = shadow_model.predict_proba(X_shadow_val)[:, 1]
  best_thresh, best_f1 = find_best_threshold_binary(y_shadow_val, probs_val)
  print(f"Best threshold: {best_thresh}, Best F1: {best_f1}")

  probs_test = optimized_model.predict_proba(X_test)[:, 1]
  y_pred_test = (probs_test >= best_thresh).astype(int)
  y_pred = pd.DataFrame(y_pred_test, columns=['prediction'])

  mask = y_pred['prediction'] == 1
  projection = X_test[mask][['customer_id', 'product_id']]

  # Cast to int to avoid decimal interpretation when saving to CSV.
  projection['customer_id'] = projection['customer_id'].astype(int)
  projection['product_id'] = projection['product_id'].astype(int)

  predictions_dir = f"{base_dir}/predictions"
  predictions_path_out = f"{predictions_dir}/test_week.csv"
  projection.to_csv(predictions_path_out, header=False, index=False)

  ti.xcom_push(key='predictions_path', value=predictions_path_out)