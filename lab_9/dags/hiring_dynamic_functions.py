import os
import pandas as pd
import gradio as gr
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42
TARGET_COLUMN = 'HiringDecision'
BASE_PATH = os.environ.get('AIRFLOW_HOME', '/opt/airflow')

def create_folders(**kwargs) -> None:
  """
  Create necessary folders for the current execution date.
  """
  ts = kwargs['ts']
  ti = kwargs['ti']
  # Prevent issues with special characters in folder names.
  safe_ts = ts.replace(":", "_").replace("+", "_").replace("T", "_")
  base_dir = f"{BASE_PATH}/{safe_ts}"
  os.makedirs(f"{base_dir}/raw", exist_ok=True)
  os.makedirs(f"{base_dir}/preprocessed", exist_ok=True)
  os.makedirs(f"{base_dir}/splits", exist_ok=True)
  os.makedirs(f"{base_dir}/models", exist_ok=True)
  ti.xcom_push(key='base_dir', value=base_dir)

def loads_ands_merge(**kwargs) -> None:
  """
  Load and merge multiple raw data files into a single DataFrame.
  """
  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')
  all_files = [f for f in os.listdir(f'{base_dir}/raw') if f.endswith('.csv')]
  df_list = [pd.read_csv(f'{base_dir}/raw/{file}') for file in all_files]
  merged_df = pd.concat(df_list, ignore_index=True)
  merged_df.to_csv(f'{base_dir}/preprocessed/merged_data.csv', index=False)
def split_data(**kwargs) -> None:
  """
  Split the raw data into training and testing sets, and saves them as CSV files.
  """
  ti= kwatgs['ti']
  base_dir = ti.xcom_pull(key='base_dir')
  df=pd.read_csv(f'{base_dir}/preprocessed/merged_data.csv')

  X = df.drop(columns=[TARGET_COLUMN])
  y=df[TARGET_COLUMN]
  X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    stratify=y, 
    random_state=RANDOM_STATE
  )
  train_data=pd.concat([X_train, y_train], axis=1)
  test_data=pd.concat([X_test, y_test], axis=1)
  train_path = f"{base_dir}/splits/train_data.csv"
  test_path = f"{base_dir}/splits/test_data.csv"
  train_data.to_csv(train_path, index=False)
  test_data.to_csv(test_path, index=False)

def train_model(**kwargs)-> None:
    X_train = train_data.drop(columns=[TARGET_COLUMN])
    y_train = train_data[TARGET_COLUMN]
    
