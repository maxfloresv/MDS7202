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
  os.makedirs(f"{base_dir}/splits", exist_ok=True)
  os.makedirs(f"{base_dir}/models", exist_ok=True)
  ti.xcom_push(key='base_dir', value=base_dir)

def split_data(**kwargs) -> None:
  """
  Split the raw data into training and testing sets, and saves them as CSV files.
  """
  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')
  df = pd.read_csv(f'{base_dir}/raw/data_1.csv')

  X = df.drop(columns=[TARGET_COLUMN])
  y = df[TARGET_COLUMN]
  X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    stratify=y, 
    random_state=RANDOM_STATE
  )

  train_data = pd.concat([X_train, y_train], axis=1)
  test_data = pd.concat([X_test, y_test], axis=1)

  train_path = f"{base_dir}/splits/train_data.csv"
  test_path = f"{base_dir}/splits/test_data.csv"

  train_data.to_csv(train_path, index=False)
  test_data.to_csv(test_path, index=False)

def preprocess_and_train(**kwargs) -> None:
  """ 
  Applies preprocessing to the data and trains a RandomForest model.
  """
  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')
  train_data = pd.read_csv(f'{base_dir}/splits/train_data.csv')
  test_data = pd.read_csv(f'{base_dir}/splits/test_data.csv')

  X_train = train_data.drop(columns=[TARGET_COLUMN])
  y_train = train_data[TARGET_COLUMN]
  X_test = test_data.drop(columns=[TARGET_COLUMN])
  y_test = test_data[TARGET_COLUMN]

  categorical = [
    'Gender', 
    'EducationLevel', 
    'PreviousCompanies', 
    'RecruitmentStrategy'
  ]
  numerical = [
    col for col in X_train.columns if col not in categorical
  ]

  num_pipe = Pipeline(steps=[('scaler', MinMaxScaler())])

  ct = ColumnTransformer(
    transformers=[
      ('num', num_pipe, numerical)
    ], remainder='passthrough'
  )

  pipe = Pipeline(steps=[
    ('preprocessor', ct),
    ('model', RandomForestClassifier(random_state=RANDOM_STATE))
  ])

  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred, pos_label=1)

  print(f"Accuracy (test): {accuracy}")
  print(f"F1-Score de la clase positiva (test): {f1}")

  model_path = f"{base_dir}/models/random_forest.joblib"
  joblib.dump(pipe, model_path)

def predict(file, model_path) -> dict:
  """
  Auxiliary function to make predictions using the trained model.

  Parameters
  ----------
  file : Uploaded file in JSON format containing input features.
  model_path : str
    Path to the trained model file.

  Returns
  -------
  dict
    A dictionary containing the prediction result.
  """
  pipeline = joblib.load(model_path)
  input_data = pd.read_json(file)
  predictions = pipeline.predict(input_data)
  print(f'La prediccion es: {predictions}')
  labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

  return {'Predicción': labels[0]}

def gradio_interface(**kwargs) -> None:
  """
  Launches a Gradio interface for making predictions with the trained model.
  """
  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')
  model_path = f"{base_dir}/models/random_forest.joblib"

  interface = gr.Interface(
    fn=lambda file: predict(file, model_path),
    inputs=gr.File(label="Sube un archivo JSON"),
    outputs="json",
    title="Hiring Decision Prediction",
    description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no."
  )
  interface.launch(share=True)