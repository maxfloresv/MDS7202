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
from scripts.split_and_apply_transformations import (
  split_data,
  create_data_transformations
)
from scripts.train_and_optimize import (
  construct_model,
  save_optimization_study,
  generate_optuna_plots,
  setup_optimized_model
)
from scripts.interpret import (
  apply_shap_values,
  generate_shap_summary
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

    dl_transactions = BashOperator(
      task_id='dl_transactions',
      retries=3,
      bash_command="gdown 1Cwczy6VepuOmEqJ94pcJP-njx-9l61Q6 -O "
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

    split_data = PythonOperator(
      task_id='split_data',
      python_callable=split_data
    )

    create_data_transformations = PythonOperator(
      task_id='create_data_transformations',
      python_callable=create_data_transformations
    )

    construct_model = PythonOperator(
      task_id='construct_model',
      python_callable=construct_model
    )

    save_optimization_study = PythonOperator(
      task_id='save_optimization_study',
      python_callable=save_optimization_study
    )

    generate_optuna_plots = PythonOperator(
      task_id='generate_optuna_plots',
      python_callable=generate_optuna_plots
    )

    setup_optimized_model = PythonOperator(
      task_id='setup_optimized_model',
      python_callable=setup_optimized_model
    )

    apply_shap_values = PythonOperator(
      task_id='apply_shap_values',
      python_callable=apply_shap_values
    )

    generate_shap_summary = PythonOperator(
      task_id='generate_shap_summary',
      python_callable=generate_shap_summary
    )

    (
      initial_task 
      >> create_folders
      >> dl_transactions
      >> preprocess
      >> generate_base_dataframe
      >> clean_base_dataframe_types
      >> split_data
      >> create_data_transformations
      >> construct_model
      >> save_optimization_study
      >> generate_optuna_plots
      >> setup_optimized_model
      >> apply_shap_values
      >> generate_shap_summary
    )