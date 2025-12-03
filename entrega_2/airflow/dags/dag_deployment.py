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
  prepare_full_data,
  create_preprocessor_template
)
from scripts.train_and_optimize import (
  construct_model_template,
  save_optimization_study,
  generate_optuna_plots,
  setup_optimized_model
)
from scripts.predict import (
  generate_test_data,
  generate_week_predictions
)

with DAG(
  dag_id='sodai-test-preprocessing',
  schedule_interval='@weekly',
  start_date=datetime(2025, 12, 3),
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
      bash_command="gdown 1dkUPLVXq_patD2RAKsb7ix-kQyRigcUo -O "
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

    prepare_full_data = PythonOperator(
      task_id='prepare_full_data',
      python_callable=prepare_full_data
    )

    create_preprocessor_template = PythonOperator(
      task_id='create_preprocessor_template',
      python_callable=create_preprocessor_template
    )

    construct_model_template = PythonOperator(
      task_id='construct_model_template',
      python_callable=construct_model_template
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

    generate_test_data = PythonOperator(
      task_id='generate_test_data',
      python_callable=generate_test_data,
      op_kwargs={'W': 3, 'Y': 2025}
    )

    generate_week_predictions = PythonOperator(
      task_id='generate_week_predictions',
      python_callable=generate_week_predictions
    )

    (
      initial_task 
      >> create_folders
      >> dl_transactions
      >> preprocess
      >> generate_base_dataframe
      >> clean_base_dataframe_types
      >> prepare_full_data
      >> create_preprocessor_template
      >> construct_model_template
      >> save_optimization_study
      >> generate_optuna_plots
      >> setup_optimized_model
      >> generate_test_data
      >> generate_week_predictions
    )