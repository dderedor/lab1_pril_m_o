import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature
import joblib

def scale_frame(frame):

    df = frame.copy()
    # Все столбцы, кроме целевой (Weight), считаем признаками
    X = df.drop(columns=['Weight'])
    y = df['Weight']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scaled = scaler.fit_transform(X.values)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))
    return X_scaled, y_scaled, power_trans

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train(**context):
    """Обучение модели
    ti = context['ti']
    input_path = ti.xcom_pull(task_ids='clear_data')
    df = pd.read_csv(input_path)

    # Масштабируем
    X, y, power_trans = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Гиперпараметры для перебора
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ['squared_error', 'huber', 'epsilon_insensitive'],
        "fit_intercept": [False, True],
    }

    # Устанавливаем эксперимент MLflow
    mlflow.set_experiment("fish_weight_prediction")

    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train.reshape(-1))
        best = clf.best_estimator_

        # Предсказание на валидации
        y_pred_scaled = best.predict(X_val)
        y_pred = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_val_orig = power_trans.inverse_transform(y_val)

        rmse, mae, r2 = eval_metrics(y_val_orig, y_pred)

        # Логирование параметров
        mlflow.log_param("alpha", best.alpha)
        mlflow.log_param("l1_ratio", best.l1_ratio)
        mlflow.log_param("penalty", best.penalty)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Сохраняем модель через MLflow
        signature = infer_signature(X_train, best.predict(X_train))
        mlflow.sklearn.log_model(best, "model", signature=signature)

        # Дополнительно сохраняем pickle в папку models
        model_path = os.path.join(os.environ['AIRFLOW_HOME'], 'models', 'fish_model.pkl')
        with open(model_path, "wb") as f:
            joblib.dump(best, f)
        print(f"Model saved to {model_path}")
