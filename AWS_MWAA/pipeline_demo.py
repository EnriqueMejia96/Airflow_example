import airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os
import boto3
import tempfile

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 26),
    'email': ['enriquemejiagamarra@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_workflow_demo',
    default_args=default_args,
    description='A simple ML pipeline',
    schedule_interval='0 17 * * *',
)

def load_data():
    data = load_iris()
    return data.data.tolist(), data.target.tolist()

def preprocess_data(ti):
    X, y = ti.xcom_pull(task_ids='load_data')
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X).tolist()
    return X_scaled, y

def train_model(ti):
    X_scaled, y = ti.xcom_pull(task_ids='preprocess_data')
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)

    # Use Boto3 to upload the model to S3
    s3_client = boto3.client('s3')
    bucket_name = os.environ.get('BUCKET_NAME')  # Ensure this is set in your MWAA environment
    object_name = 'airflow_demo_dmc/logistic_model.joblib'
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        # Save the model to a temporary file
        dump(model, tmp.name)
        # Upload the temporary file to S3
        s3_client.upload_file(tmp.name, bucket_name, object_name)
    
    return f's3://{bucket_name}/{object_name}'

def evaluate_model(ti):
    model_path = ti.xcom_pull(task_ids='train_model')

    # Extract bucket name and object name from the model_path
    _, _, bucket_name, object_name = model_path.split('/', 3)
    
    s3_client = boto3.client('s3')
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        # Download the model from S3 to a temporary file
        s3_client.download_file(bucket_name, object_name, tmp.name)
        # Load the model from the temporary file
        model = load(tmp.name)

    X_scaled, y = ti.xcom_pull(task_ids='preprocess_data')
    predictions = model.predict(X_scaled)
    accuracy = accuracy_score(y, predictions)
    return accuracy

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

load_data_task >> preprocess_data_task >> train_model_task >> evaluate_model_task
