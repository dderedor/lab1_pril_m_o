import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import OrdinalEncoder

from airflow import DAG
from airflow.operators.python import PythonOperator

from train_model import train

def download_data():
    AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', os.path.expanduser('~/airflow'))
    data_path = os.path.join(AIRFLOW_HOME, 'data', 'mpg.csv')
    url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv'
    df = pd.read_csv(url)
    df.to_csv(data_path, index=False)
    print(f"Data saved to {data_path}, shape: {df.shape}")

def clear_data():
    AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', os.path.expanduser('~/airflow'))
    raw_path = os.path.join(AIRFLOW_HOME, 'data', 'mpg.csv')
    clean_path = os.path.join(AIRFLOW_HOME, 'data', 'mpg_clear.csv')
    
    df = pd.read_csv(raw_path)
    df = df.dropna().reset_index(drop=True)
    
    enc = OrdinalEncoder()
    df['origin_enc'] = enc.fit_transform(df[['origin']])
    df = df.drop(columns=['origin', 'name'])
    
    df.to_csv(clean_path, index=False)
    print(f"Cleaned data saved to {clean_path}")

default_args = {
    'owner': 'masha',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 10),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'schedule_interval': '@daily',
}

dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='ML pipeline for mpg prediction',
    catchup=False,
)

t1 = PythonOperator(task_id='download_data', python_callable=download_data, dag=dag)
t2 = PythonOperator(task_id='clear_data', python_callable=clear_data, dag=dag)
t3 = PythonOperator(task_id='train_model', python_callable=train, dag=dag)

t1 >> t2 >> t3
