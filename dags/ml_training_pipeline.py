from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
try:
    from airflow.operators.bash import BashOperator
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import BranchPythonOperator, PythonOperator
except ImportError:
    pass

# Константи 

# Шлях до ML-проєкту всередині контейнера (змонтований через docker-compose)
ML_PROJECT = os.getenv("ML_PROJECT_ROOT", "/opt/airflow")

# Пороги якості для автоматичної реєстрації моделі
F1_THRESHOLD = float(os.getenv("F1_THRESHOLD", "0.60"))
ROC_AUC_THRESHOLD = float(os.getenv("ROC_AUC_THRESHOLD", "0.70"))

# Python-інтерпретатор у середовищі з ML-залежностями
PYTHON = os.getenv("ML_PYTHON", "python3")

# Дефолтні аргументи DAG 

default_args = {
    "owner": "developer",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# Python-функції для PythonOperator 


def check_data_availability(**kwargs):
    """
    Крок 1 — Sensor/Check: перевірка наявності вхідних даних.
    Перевіряє що data/sample/train.csv існує і не порожній.
    """
    train_path = Path(ML_PROJECT) / "data" / "sample" / "train.csv"
    test_path = Path(ML_PROJECT) / "data" / "sample" / "test.csv"

    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Дані не знайдено: {path}\n"
                "Запустіть: python src/prepare.py data/sample/creditcard_sample.csv data/sample"
            )
        if path.stat().st_size < 1000:
            raise ValueError(f"Файл підозріло малий: {path}")

    print(f"Дані доступні: {train_path} ({train_path.stat().st_size // 1024} KB)")
    print(f"Дані доступні: {test_path} ({test_path.stat().st_size // 1024} KB)")


def evaluate_and_branch(**kwargs):
    """
    Крок 4 — Evaluation & Branching: читає metrics.json і вирішує
    чи реєструвати модель в MLflow Registry.

    Повертає task_id наступного кроку для BranchPythonOperator.
    """
    metrics_path = Path(ML_PROJECT) / "metrics.json"

    if not metrics_path.exists():
        print("metrics.json не знайдено — пропускаємо реєстрацію")
        return "notify_low_quality"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    f1 = float(metrics.get("f1", 0))
    roc_auc = float(metrics.get("roc_auc", 0))

    print(f"Метрики: F1={f1:.4f} (поріг={F1_THRESHOLD}) | ROC-AUC={roc_auc:.4f} (поріг={ROC_AUC_THRESHOLD})")

    # Передаємо метрики через XCom для наступних кроків
    kwargs["ti"].xcom_push(key="f1", value=f1)
    kwargs["ti"].xcom_push(key="roc_auc", value=roc_auc)

    if f1 >= F1_THRESHOLD and roc_auc >= ROC_AUC_THRESHOLD:
        print("Якість прийнятна - реєстрація моделі")
        return "register_model"
    else:
        print(f"Якість нижча за поріг - сповіщення")
        return "notify_low_quality"


def register_model_fn(**kwargs):
    import mlflow
    import joblib
    from mlflow.tracking import MlflowClient

    ti = kwargs["ti"]
    f1 = ti.xcom_pull(task_ids="evaluate_model", key="f1") or 0.0
    roc_auc = ti.xcom_pull(task_ids="evaluate_model", key="roc_auc") or 0.0

    mlflow_tracking_path = os.path.join(ML_PROJECT, "mlruns")
    os.makedirs(mlflow_tracking_path, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_tracking_path}")
    
    experiment_name = "CreditCard_Fraud_Airflow"
    model_name = "CreditCardFraudDetector"
    model_path = Path(ML_PROJECT) / "model.pkl"

    try:
        client = MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            print(f"Creating new experiment: {experiment_name}")
            experiment_id = client.create_experiment(experiment_name)
        else:
            experiment_id = exp.experiment_id
            
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(experiment_id=experiment_id, run_name="airflow_registration"):
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            if model_path.exists():
                model = joblib.load(model_path)
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=model_name
                )
                print("Model logged and registered.")
            else:
                raise FileNotFoundError(f"Model not found at {model_path}")

        versions = client.get_latest_versions(model_name, stages=["None"])
        if versions:
            latest_v = versions[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=latest_v,
                stage="Staging"
            )
            print(f"Success: {model_name} v{latest_v} -> Staging")

    except Exception as e:
        print(f"Error during MLflow registration: {str(e)}")
        raise


def notify_low_quality_fn(**kwargs):
    ti = kwargs["ti"]
    f1 = ti.xcom_pull(task_ids="evaluate_model", key="f1") or "N/A"
    roc_auc = ti.xcom_pull(task_ids="evaluate_model", key="roc_auc") or "N/A"

    message = (
        f"Quality Gate НЕ пройдено!\n"
        f"  F1={f1} (поріг={F1_THRESHOLD})\n"
        f"  ROC-AUC={roc_auc} (поріг={ROC_AUC_THRESHOLD})\n"
        f"  Модель НЕ зареєстрована в MLflow Registry."
    )
    print(message)


# Визначення DAG 

with DAG(
    dag_id="ml_training_pipeline",
    description="Credit Card Fraud Detection — повний ML-пайплайн",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 2 * * *",   # щодня о 02:00; None — лише ручний запуск
    catchup=False,
    tags=["mlops", "creditcard", "lab5"],
) as dag:

    # Task 1: перевірка даних 
    check_data = PythonOperator(
        task_id="check_data",
        python_callable=check_data_availability,
    )

    # Task 2: підготовка даних через src/prepare.py 
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=(
            f"cd {ML_PROJECT} && "
            f"{PYTHON} src/prepare.py "
            f"data/sample/creditcard_sample.csv data/sample"
        ),
    )

    # Task 3: тренування через src/train.py (CI-режим на sample) 
    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {ML_PROJECT} && "
            f"{PYTHON} src/train.py data/sample models "
            f"--ci-mode --max-rows 5000 --n_estimators 100 --max_depth 10"
        ),
    )

    # Task 4: оцінка та розгалуження 
    evaluate_model = BranchPythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_and_branch,
    )

    # Task 5a: реєстрація моделі
    register_model = PythonOperator(
        task_id="register_model",
        python_callable=register_model_fn,
    )

    # Task 5b: сповіщення про низьку якість
    notify_low_quality = PythonOperator(
        task_id="notify_low_quality",
        python_callable=notify_low_quality_fn,
    )

    # Task 6: завершення пайплайну 
    pipeline_end = EmptyOperator(
        task_id="pipeline_end",
        trigger_rule="none_failed_min_one_success",
    )

    # Залежності (DAG-граф)
    check_data >> prepare_data >> train_model >> evaluate_model
    evaluate_model >> [register_model, notify_low_quality]
    [register_model, notify_low_quality] >> pipeline_end