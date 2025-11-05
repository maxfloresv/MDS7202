import datetime as dt

from airflow import DAG
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator

from hiring_dynamic_functions import (
  data_branching,
  create_folders,
  load_and_merge,
  split_data,
  train_model,
  evaluate_models
)

with DAG(
  dag_id='dynamic_hiring', 
  description='Hiring Decision Pipeline (Dynamic Version)',
  start_date=dt.datetime(2024, 10, 1),
  schedule_interval='0 15 5 * *', 
  catchup=True
) as dag:
  initial_task = EmptyOperator(task_id='start_pipeline', retries=2)  

  create_folders_task = PythonOperator(
    task_id='create_folders',
    python_callable=create_folders
  )

  branch_task_1 = BranchPythonOperator(
    task_id='data_branching',
    python_callable=data_branching,
    provide_context=True,
    dag=dag
  )

  download_data_1_task = BashOperator(
    task_id='download_dataset_1',
    bash_command="curl -o {{ ti.xcom_pull(key='base_dir') }}/raw/data_1.csv " 
      "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
  )

  download_data_2_task = BashOperator(
    task_id='download_dataset_2',
    bash_command="curl -o {{ ti.xcom_pull(key='base_dir') }}/raw/data_2.csv " 
      "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"
  )

  load_and_merge_task = PythonOperator(
    task_id='load_and_merge',
    python_callable=load_and_merge,
    trigger_rule='one_success'
  )

  split_data_task = PythonOperator(
    task_id='split_data',
    python_callable=split_data
  )

  train_rf_task = PythonOperator(
    task_id='train_random_forest',
    python_callable=train_model,
    op_kwargs={'model': RandomForestClassifier}
  )

  train_svc_task = PythonOperator(
    task_id='train_svc',
    python_callable=train_model,
    op_kwargs={'model': SVC}
  )

  train_logreg_task = PythonOperator(
    task_id='train_logistic_regression',
    python_callable=train_model,
    op_kwargs={'model': LogisticRegression}
  )

  evaluate_models_task = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    trigger_rule='all_success'
  )

  (
    initial_task
    >> create_folders_task
    >> branch_task_1
    >> [download_data_1_task, download_data_2_task]
    >> load_and_merge_task
    >> split_data_task
    >> [train_rf_task, train_svc_task, train_logreg_task]
    >> evaluate_models_task
  )