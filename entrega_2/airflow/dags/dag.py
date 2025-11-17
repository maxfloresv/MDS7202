from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.task_group import TaskGroup
from scripts.preprocessing import (
    create_folders,
    extract_and_load,
    negative_items,
    filter_active_entities,
    create_weekly_base,
    create_labels,
    merge_features,
    save_preprocessed_data
)

with DAG(
    dag_id='sodai-model-pipeline',
    schedule_interval='@weekly',
    #start_date=datetime(2024, 1, 1),
    #catchup=False,
) as dag:
    
    create_folders = PythonOperator(task_id='create_folders', python_callable=create_folders)
    with TaskGroup('preprocessing') as preprocessing:
        extract = PythonOperator(task_id='extract_and_load',python_callable=extract_and_load)
        negative = PythonOperator(task_id='negative_items', python_callable=negative_items)
        filter_entities = PythonOperator(task_id='filter_active_entities', python_callable=filter_active_entities)
        weekly = PythonOperator(task_id='create_weekly_base', python_callable=create_weekly_base)
        labels = PythonOperator(task_id='create_labels', python_callable=create_labels)
        merge = PythonOperator(task_id='merge_features', python_callable=merge_features)
        save = PythonOperator(task_id='save_preprocessed_data', python_callable=save_preprocessed_data)
        extract >> negative >> filter_entities >> weekly >> labels >> merge >> save
    # TO DO: IMPLEMENTAR EL RESTO DE LAS FUNCIONES Y LAS TAREAS DEL DAG
    with TaskGroup('drift_detection') as drift:
        check_drift = PythonOperator(task_id='check_drift', ...)
        decide_retrain = BranchPythonOperator(task_id='decide_retrain', ...)
        
        check_drift >> decide_retrain

    with TaskGroup('retraining') as retraining:
        split= PythonOperator(task_id='split_data', ...)
        scaling = PythonOperator(task_id='scaling_data', ...)
        optimize = PythonOperator(task_id='optimize_hyperparams', ...)
        train = PythonOperator(task_id='train_model', ...)
        track = PythonOperator(task_id='track_results', ...)    
        split >> scaling >> optimize >> train >> track

    with TaskGroup('prediction') as prediction:
        load_model = PythonOperator(task_id='load_model', ...)
        predict = PythonOperator(task_id='generate_predictions', ...)
        save_preds = PythonOperator(task_id='save_predictions', ...)  
        load_model >> predict >> save_preds

    create_folders >> preprocessing >> drift >> retraining >> prediction