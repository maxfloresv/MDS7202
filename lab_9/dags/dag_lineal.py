from airflow import DAG
import datetime as dt

from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator

from hiring_functions import (
  create_folders,
  split_data,
  preprocess_and_train,
  gradio_interface
)

with DAG(
  dag_id='linear_hiring', 
  description='Hiring Decision Pipeline (Linear Version)',
  start_date=dt.datetime(2024, 10, 1),
  schedule=None,
  catchup=False,
) as dag:
  initial_task = EmptyOperator(task_id='start_pipeline', retries=2)  

  create_folders_task = PythonOperator(
    task_id='create_folders',
    python_callable=create_folders
  )

  download_data_task = BashOperator(
    task_id='download_data',
    bash_command="curl -o {{ ti.xcom_pull(key='base_dir') }}/raw/data_1.csv " 
      "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
  )

  split_data_task = PythonOperator(
    task_id='split_data',
    python_callable=split_data
  )

  preprocess_and_train_task = PythonOperator(
    task_id='preprocess_and_train',
    python_callable=preprocess_and_train
  )

  gradio_interface_task = PythonOperator(
    task_id='gradio_interface',
    python_callable=gradio_interface
  )

  (
    initial_task 
    >> create_folders_task 
    >> download_data_task 
    >> split_data_task
    >> preprocess_and_train_task 
    >> gradio_interface_task
  )