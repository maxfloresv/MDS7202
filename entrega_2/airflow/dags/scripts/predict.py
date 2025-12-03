import pandas as pd
import numpy as np
import joblib

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
  
  X_fusion_path = ti.xcom_pull(key='X_fusion_path')
  X_fusion = pd.read_parquet(X_fusion_path, engine='pyarrow')

  # Recovers the week from the sine and cosine values.
  angles = np.arctan2(X_fusion['week_sin'], X_fusion['week_cos'])
  angles[angles < 0] += 2 * np.pi
  X_fusion['week_recovered'] = (angles / (2 * np.pi)) * 52
  X_fusion['week_recovered'] = X_fusion['week_recovered'].round().astype(int)
  X_fusion.loc[X_fusion['week_recovered'] == 0, 'week_recovered'] = 52
  
  week_sin = np.sin(2 * np.pi * W / 52)
  week_cos = np.cos(2 * np.pi * W / 52)

  dynamic_features = ['week_sin', 'week_cos', 'year', 'week_recovered']
  static_features = [
    col for col in X_fusion.columns 
    if col not in dynamic_features
  ]
  
  # Drop duplicates to get only one row per customer-product combination for the given week and year.
  # Keeps the last occurrence to be consistent with lags and rolling features.
  X_test = X_fusion.sort_values(
      by=['customer_id', 'product_id', 'year', 'week_recovered']
  )[static_features].drop_duplicates(
    subset=['customer_id', 'product_id'],
    keep='last'
  ).copy()
  
  X_test['week_sin'] = week_sin
  X_test['week_cos'] = week_cos
  X_test['year'] = Y
  
  final_columns = [col for col in X_fusion.columns if col != 'week_recovered']
  X_test = X_test[final_columns]
  print(X_test.head())

  splits_dir = f"{base_dir}/splits"
  X_test_path_out = f"{splits_dir}/X_test_week_{W}_year_{Y}.parquet"
  X_test.to_parquet(X_test_path_out, engine='pyarrow', index=False)

  ti.xcom_push(key='X_test_path', value=X_test_path_out)

def generate_week_predictions(**kwargs):
  """
  Generates the predictions for the week in the pipeline.
  """
  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')

  X_test_path = ti.xcom_pull(key='X_test_path')
  X_test = pd.read_parquet(X_test_path, engine='pyarrow')

  optimized_model_path = ti.xcom_pull(key='optimized_trained_model_path')
  optimized_model = joblib.load(optimized_model_path)

  y_pred = pd.DataFrame(
    optimized_model.predict(X_test), 
    columns=['prediction']
  )

  mask = y_pred['prediction'] == 1
  projection = X_test[mask][['customer_id', 'product_id']]

  # Cast to int to avoid decimal interpretation when saving to CSV.
  projection['customer_id'] = projection['customer_id'].astype(int)
  projection['product_id'] = projection['product_id'].astype(int)

  predictions_dir = f"{base_dir}/predictions"
  predictions_path_out = f"{predictions_dir}/test_week.csv"
  projection.to_csv(predictions_path_out, header=False, index=False)

  ti.xcom_push(key='predictions_path', value=predictions_path_out)