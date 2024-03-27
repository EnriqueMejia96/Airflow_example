import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 26),
    'email ':['enriquemejiagamarra@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_workflow',
    default_args=default_args,
    description='A simple ML pipeline',
    schedule_interval=timedelta(days=1),
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
    model_path = '/opt/airflow/dags/data/logistic_model.joblib'
    dump(model, model_path)
    return model_path

def evaluate_model(ti):
    model_path = ti.xcom_pull(task_ids='train_model')
    model = load(model_path)
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
