import os
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import OrdinalEncoder

from airflow import DAG
from airflow.operators.python import PythonOperator


from train_model_fish import train

# Функции загрузки и очистки 

def download_data():
    """Скачивает датасет Fish Market и сохраняет в /data/fish.csv"""
    AIRFLOW_HOME = os.environ['AIRFLOW_HOME']
    url = 'https://raw.githubusercontent.com/plotly/datasets/master/fish-market.csv'
    df = pd.read_csv(url)
    output_path = os.path.join(AIRFLOW_HOME, 'data', 'fish.csv')
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}, shape: {df.shape}")
    return output_path

def clear_data(**context):
    """Очистка данных: удаление выбросов, кодирование категорий"""
    ti = context['ti']
    input_path = ti.xcom_pull(task_ids='download_data')
    df = pd.read_csv(input_path)

    # Удалим выбросы по весу (очень маленькие или большие)
    df = df[(df['Weight'] > 10) & (df['Weight'] < 2000)]
    # Сброс индекса
    df = df.reset_index(drop=True)

    # Кодируем категориальный признак Species (порода рыбы)
    cat_col = ['Species']
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_col])
    df_encoded = ordinal.transform(df[cat_col])
    df['Species_enc'] = df_encoded
    df = df.drop(columns=['Species'])  # заменяем исходный столбец на закодированный

    # Сохраняем очищенный файл
    AIRFLOW_HOME = os.environ['AIRFLOW_HOME']
    output_path = os.path.join(AIRFLOW_HOME, 'data', 'fish_clear.csv')
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    return output_path



default_args = {
    'owner': 'masha',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 10),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_fish_pipeline',
    default_args=default_args,
    description='ML pipeline for fish weight prediction',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['ml', 'fish'],
)

download_task = PythonOperator(
    task_id='download_data',
    python_callable=download_data,
    dag=dag,
)

clear_task = PythonOperator(
    task_id='clear_data',
    python_callable=clear_data,
    provide_context=True,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train,
    provide_context=True,
    dag=dag,
)

download_task >> clear_task >> train_task
