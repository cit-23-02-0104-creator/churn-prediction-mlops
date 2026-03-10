from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'jayani',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_pipeline_task(script_name):
    
    os.system(f"python /usr/local/airflow/src/{script_name}")

with DAG(
    'customer_churn_dag',
    default_args=default_args,
    description='MLOps Pipeline for Churn Prediction',
    schedule=timedelta(days=1),
    catchup=False
) as dag:

    # 1. Data Ingestion 
    ingestion = PythonOperator(
        task_id='data_ingestion',
        python_callable=run_pipeline_task,
        op_args=['preprocessing.py'] 
    )

    # 2. Data Validation 
    validation = PythonOperator(
        task_id='data_validation',
        python_callable=run_pipeline_task,
        op_args=['preprocessing.py']
    )

    # 3. Feature Engineering
    feature_eng = PythonOperator(
        task_id='feature_engineering',
        python_callable=run_pipeline_task,
        op_args=['preprocessing.py']
    )

    # 4. Model Training
    train = PythonOperator(
        task_id='model_training',
        python_callable=run_pipeline_task,
        op_args=['train.py']
    )

    # 5. Model Evaluation
    evaluate = PythonOperator(
        task_id='model_evaluation',
        python_callable=run_pipeline_task,
        op_args=['evaluate.py']
    )

    # 6. Model Registration
    registration = PythonOperator(
        task_id='model_registration',
        python_callable=run_pipeline_task,
        op_args=['evaluate.py']
    )

    # Task Dependencies 
    ingestion >> validation >> feature_eng >> train >> evaluate >> registration