from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from scripts.preprocessing import (
  create_folders,
  preprocess,
  generate_base_dataframe,
  clean_base_dataframe_types
)

with DAG(
    dag_id='sodai-test-preprocessing',
    schedule_interval='@weekly',
    start_date=datetime(2025, 11, 18),
    tags=['sodai', 'preprocessing']
) as dag:
    initial_task = EmptyOperator(task_id='initial_task')

    create_folders = PythonOperator(
      task_id='create_folders', 
      python_callable=create_folders
    )

    dl_customers = BashOperator(
      task_id='dl_customers',
      retries=3,
      bash_command="gdown 1EYohIzpbdWzJ5dCMi5U3ntV0-GkzcQdN -O "
        "{{ ti.xcom_pull(key='base_dir') }}/raw/clientes.parquet"
    )
    dl_products = BashOperator(
      task_id='dl_products',
      retries=3,
      bash_command="gdown 1SdnvFMqjUN1YOtsHbrC3H7G7ElIFmhp2 -O "
        "{{ ti.xcom_pull(key='base_dir') }}/raw/productos.parquet"
    )
    dl_transactions = BashOperator(
      task_id='dl_transactions',
      retries=3,
      bash_command="gdown 150vO-Jav--uiskCxIeUksqqIhS5TYQ_s -O "
        "{{ ti.xcom_pull(key='base_dir') }}/raw/transacciones.parquet"
    )

    preprocess = PythonOperator(
      task_id='preprocess',
      python_callable=preprocess
    )

    generate_base_dataframe = PythonOperator(
      task_id='generate_base_dataframe',
      python_callable=generate_base_dataframe
    )

    clean_base_dataframe_types = PythonOperator(
      task_id='clean_base_dataframe_types',
      python_callable=clean_base_dataframe_types
    )

    (
      initial_task 
      >> create_folders
      >> [dl_customers, dl_products, dl_transactions]
      >> preprocess
      >> generate_base_dataframe
      >> clean_base_dataframe_types
    )