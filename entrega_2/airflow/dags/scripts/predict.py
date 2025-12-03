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
  Generates the input DataFrame for the given week (W) and year (Y) for each customer-product combination.

  Parameters
  ----------
  W : int
    The week to generate the test data for.
  Y : int
    The year to generate the test data for.
  """
  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')

  df_path = ti.xcom_pull(key='final_df_path')
  df = pd.read_parquet(df_path, engine='pyarrow')

  last_known_year = df['year'].max()
  last_known_week = df[df['year'] == last_known_year]['week'].max()

  mask = (df['year'] == last_known_year) & (df['week'] == last_known_week)
  candidates = df[mask][[
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
  ]].copy()

  candidates = candidates.drop_duplicates(
    subset=['customer_id', 'product_id'],
    keep='last'
  )

  candidates['week'] = W
  candidates['year'] = Y

  # Dummy values to concatenate with the existing dataframe.
  candidates['items'] = np.nan
  candidates['label'] = np.nan

  df_window = pd.concat([df, candidates], ignore_index=True)
  # Recuperates the last occurrence to be consistent with lags and rolling features.
  df_window = df_window.sort_values(
    by=['customer_id', 'product_id', 'year', 'week']
  )

  df_window['items_lag_1'] = df_window.groupby(
    ['customer_id', 'product_id']
  )['items'].shift(1)

  df_window['items_rolling_mean_4w'] = df_window.groupby(
    ['customer_id', 'product_id']
  )['items'].transform(
    lambda x: x.shift(1).rolling(
      window=4,
      min_periods=1
    ).mean()
  )
  df_window['purchased_lag_1'] = df_window.groupby(
    ['customer_id', 'product_id']
  )['label'].shift(1)

  features_to_fill = ['items_lag_1', 'items_rolling_mean_4w', 'purchased_lag_1']
  df_window[features_to_fill] = df_window[features_to_fill].fillna(0)

  X_test = df_window[
    (df_window['year'] == Y) & (df_window['week'] == W)
  ].copy()

  week_sin = np.sin(2 * np.pi * W / 52)
  week_cos = np.cos(2 * np.pi * W / 52)
  X_test['week_sin'] = week_sin
  X_test['week_cos'] = week_cos
  
  X_full_path = ti.xcom_pull(key='X_full_path')
  X_full_schema = pd.read_parquet(X_full_path, engine='pyarrow')[:0]
  objective_columns = X_full_schema.columns.tolist()

  X_test = X_test[objective_columns]
  print(X_test.head())

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