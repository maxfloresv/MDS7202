import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

REFERENCE_DATA_DIR = "/data"

def apply_shap_values(**kwargs):
  """
  Applies SHAP values to the validation set and saves them.
  """
  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')

  optimized_model_path = ti.xcom_pull(
    key='optimized_trained_model_path'
  )
  optimized_model = joblib.load(optimized_model_path)

  lgbm_model = optimized_model.named_steps['classifier']
  explainer = shap.TreeExplainer(lgbm_model)

  X_val_path = ti.xcom_pull(key='X_val_path')
  X_val = pd.read_parquet(X_val_path, engine='pyarrow')

  X_val_tf = optimized_model.named_steps['preprocessor'].transform(X_val)
  shap_values = explainer.shap_values(X_val_tf)

  studies_dir = f"{base_dir}/studies"
  shap_values_path_out = f"{studies_dir}/shap_values.parquet"
  pd.DataFrame(
    shap_values, 
    columns=optimized_model.named_steps['preprocessor'].get_feature_names_out()
  ).to_parquet(shap_values_path_out, engine='pyarrow')
  ti.xcom_push(key='shap_values_path', value=shap_values_path_out)

  preprocessed_dir = f"{base_dir}/preprocessed"
  X_val_tf_path_out = f"{preprocessed_dir}/X_val_tf.parquet"
  pd.DataFrame(
    X_val_tf,
    columns=optimized_model.named_steps['preprocessor'].get_feature_names_out()
  ).to_parquet(X_val_tf_path_out, engine='pyarrow')
  ti.xcom_push(key='X_val_tf_path', value=X_val_tf_path_out)

def generate_shap_summary(**kwargs):
  """
  Generates a summary of the SHAP values.
  """
  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')

  optimized_model_path = ti.xcom_pull(
    key='optimized_trained_model_path'
  )
  optimized_model = joblib.load(optimized_model_path)

  shap_values_path = ti.xcom_pull(key='shap_values_path')
  shap_values = pd.read_parquet(shap_values_path, engine='pyarrow')

  shap_values_dense = shap_values.to_numpy()

  X_val_tf_path = ti.xcom_pull(key='X_val_tf_path')
  X_val_tf = pd.read_parquet(X_val_tf_path, engine='pyarrow')

  X_val_tf = X_val_tf.to_numpy()

  shap.summary_plot(
    shap_values_dense, 
    X_val_tf,
    feature_names=optimized_model.named_steps['preprocessor'].get_feature_names_out()
  )

  images_dir = f"{base_dir}/images"
  summary_plot_path_out = f"{images_dir}/shap_summary.png"

  plt.savefig(summary_plot_path_out, bbox_inches='tight')
  plt.close()
  ti.xcom_push(key='shap_summary_path', value=summary_plot_path_out)