import argparse
import os
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]  # на рівень вище src/
db_path = project_root / "mlflow.db"

# 1. Парсинг аргументів командного рядка

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RandomForest for Credit Card Fraud Detection"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/creditcard.csv",
        help="Шлях до CSV файлу з даними",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Кількість дерев у RandomForest (default: 100)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Максимальна глибина дерева (default: None — без обмежень)",
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2,
        help="Мінімальна кількість зразків для розщеплення вузла (default: 2)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Частка тестової вибірки (default: 0.2)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state для відтворюваності (default: 42)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Поріг класифікації (default: 0.5)",
    )
    return parser.parse_args()


# 2. Завантаження та передобробка даних

def load_and_preprocess(data_path: str, test_size: float, random_state: int):
    print(f"Завантаження даних: {data_path}")

    if not os.path.exists(data_path):
        print(f"Файл не знайдено: {data_path}")
        print("Завантажте creditcard.csv з Kaggle та помістіть у data/raw/")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Розмір датасету: {df.shape}")
    print(f"Шахрайство: {df['Class'].sum()} записів ({df['Class'].mean()*100:.3f}%)")

    # Ознаки та цільова змінна
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Масштабування Amount та Time (V1-V28 вже масштабовані PCA)
    scaler = StandardScaler()
    X = X.copy()
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

    # Розділення
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, df.columns.drop("Class").tolist()


# 3. Побудова графіків

def plot_confusion_matrix(y_true, y_pred, output_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Передбачений клас")
    ax.set_ylabel("Справжній клас")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix збережено: {output_path}")


def plot_feature_importance(model, feature_names: list, output_path: str, top_n: int = 15):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(top_n),
        importances[indices][::-1],
        color="steelblue",
        edgecolor="black",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1])
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Топ-{top_n} важливих ознак")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance збережено: {output_path}")


# 4. Основна функція тренування

def train(args):
    # Завантаження даних
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(
        args.data_path, args.test_size, args.random_state
    )

    # Вага класів для боротьби з дисбалансом
    class_weight = {0: 1, 1: int((y_train == 0).sum() / (y_train == 1).sum())}
    print(f"Ваги класів: {class_weight}")

    # Ініціалізація MLflow
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment("CreditCard_Fraud_Detection")

    with mlflow.start_run():

        # ── Теги (метадані запуску) ──────────────────
        mlflow.set_tag("author", "student")
        mlflow.set_tag("dataset_version", "kaggle_v1")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("task", "binary_classification")
        mlflow.set_tag("dataset_name", "creditcard-fraud")
        mlflow.set_tag("imbalance_strategy", "class_weight")

        # Логування гіперпараметрів
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("class_weight_fraud", class_weight[1])

        # Навчання моделі
        print("Навчання моделі...")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            class_weight=class_weight,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Передбачення
        # Train метрики (для аналізу overfitting)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_train_pred = (y_train_proba >= args.threshold).astype(int)

        # Test метрики
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= args.threshold).astype(int)

        # Обчислення метрик
        metrics_train = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_f1": f1_score(y_train, y_train_pred),
            "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
            "train_recall": recall_score(y_train, y_train_pred),
            "train_roc_auc": roc_auc_score(y_train, y_train_proba),
            "train_avg_precision": average_precision_score(y_train, y_train_proba),
        }

        metrics_test = {
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_f1": f1_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_test_pred),
            "test_roc_auc": roc_auc_score(y_test, y_test_proba),
            "test_avg_precision": average_precision_score(y_test, y_test_proba),
        }

        # Вивід результатів
        print("\nTRAIN метрики")
        for k, v in metrics_train.items():
            print(f"  {k}: {v:.4f}")

        print("\nTEST метрики")
        for k, v in metrics_test.items():
            print(f"  {k}: {v:.4f}")

        print("\nClassification Report (Test)")
        print(classification_report(y_test, y_test_pred, target_names=["Normal", "Fraud"]))

        # Логування метрик у MLflow
        mlflow.log_metrics(metrics_train)
        mlflow.log_metrics(metrics_test)

        # Зберігаємо max_depth як числовий параметр для порівняння на графіку
        mlflow.log_metric("max_depth_numeric", args.max_depth if args.max_depth else 0)

        # Графіки
        os.makedirs("models", exist_ok=True)

        cm_path = "models/confusion_matrix.png"
        plot_confusion_matrix(y_test, y_test_pred, cm_path)
        mlflow.log_artifact(cm_path)

        fi_path = "models/feature_importance.png"
        plot_feature_importance(model, feature_names, fi_path)
        mlflow.log_artifact(fi_path)

        # Логування моделі
        mlflow.sklearn.log_model(
            model,
            "random_forest_model",
            registered_model_name="CreditCardFraudDetector",
        )

        run_id = mlflow.active_run().info.run_id
        print(f"\nЗапуск завершено. Run ID: {run_id}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
