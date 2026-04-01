import argparse
import os
import sys
import json

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
from pathlib import Path
import joblib

project_root = Path(__file__).resolve().parents[1]
db_path = project_root / "mlflow.db"


# 1. Парсинг аргументів командного рядка

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RandomForest for Credit Card Fraud Detection (DVC stage)"
    )
    # Позиційні аргументи — для сумісності з dvc.yaml
    parser.add_argument(
        "prepared_dir",
        type=str,
        help="Папка з підготовленими даними (train.csv, test.csv)",
    )
    parser.add_argument(
        "models_dir",
        type=str,
        help="Папка для збереження моделі та артефактів",
    )
    # Гіперпараметри (опціональні)
    parser.add_argument("--n_estimators", type=int, default=144)
    parser.add_argument("--max_depth", type=int, default=15)
    parser.add_argument("--min_samples_split", type=int, default=8)
    parser.add_argument("--min_samples_leaf", type=int, default=3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        help="Обмежити кількість рядків train/test (для CI-режиму)",
    )
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="CI-режим: зберігати model.pkl + metrics.json + confusion_matrix.png у корінь",
    )
    return parser.parse_args()


# 2. Завантаження та передобробка даних

def load_prepared(prepared_dir: str, max_rows: int = 5000):
    train_path = os.path.join(prepared_dir, "train.csv")
    test_path = os.path.join(prepared_dir, "test.csv")

    for p in [train_path, test_path]:
        if not os.path.exists(p):
            print(f"Файл не знайдено — {p}")
            sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]
    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]

    feature_names = X_train.columns.tolist()
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Ознаки: {len(feature_names)}")
    if max_rows:
        print(f"CI-режим: обмежено до {max_rows} рядків")
    return X_train, X_test, y_train, y_test, feature_names


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


# 4. Збереження CI-артефактів

def save_ci_artifacts(model, y_test, y_test_pred, y_test_proba):
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_test_pred)), 4),
        "f1": round(float(f1_score(y_test, y_test_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_test_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_test_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_test_proba)), 4),
    }
    metrics_path = project_root / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"CI: metrics.json → {metrics}")

    plot_confusion_matrix(y_test, y_test_pred, str(project_root / "confusion_matrix.png"))

    joblib.dump(model, project_root / "model.pkl")
    print(f"CI: model.pkl збережено")


# 5. Null-контекст для CI

class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


# 6. Основна функція тренування

def train(args):
    X_train, X_test, y_train, y_test, feature_names = load_prepared(
        args.prepared_dir, max_rows=args.max_rows
    )
    os.makedirs(args.models_dir, exist_ok=True)

    # Ваги класів (боротьба з дисбалансом)
    class_weight = {0: 1, 1: int((y_train == 0).sum() / (y_train == 1).sum())}
    print(f"Ваги класів: {class_weight}")

    # MLflow
    if not args.ci_mode:
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
        mlflow.set_experiment("CreditCard_Fraud_Detection")

    run_ctx = mlflow.start_run() if not args.ci_mode else _NullContext()

    with run_ctx:
        if not args.ci_mode:
            # Теги
            mlflow.set_tag("author", "student")
            mlflow.set_tag("dataset_version", "kaggle_v1")
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.set_tag("task", "binary_classification")
            mlflow.set_tag("pipeline", "dvc")
            mlflow.set_tag("imbalance_strategy", "class_weight")

            # Гіперпараметри
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
            mlflow.log_param("min_samples_split", args.min_samples_split)
            mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_param("threshold", args.threshold)
            mlflow.log_param("class_weight_fraud", class_weight[1])
            mlflow.log_param("prepared_dir", args.prepared_dir)

        # Навчання
        print("Навчання моделі...")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=class_weight,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Передбачення
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_train_pred = (y_train_proba >= args.threshold).astype(int)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= args.threshold).astype(int)

        # Метрики
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

        print("\nTRAIN метрики")
        for k, v in metrics_train.items():
            print(f"  {k}: {v:.4f}")
        print("\nTEST метрики")
        for k, v in metrics_test.items():
            print(f"  {k}: {v:.4f}")
        print("\nClassification Report (Test)")
        print(classification_report(y_test, y_test_pred, target_names=["Normal", "Fraud"]))

        if not args.ci_mode:
            mlflow.log_metrics(metrics_train)
            mlflow.log_metrics(metrics_test)
            mlflow.log_metric("max_depth_numeric", args.max_depth if args.max_depth else 0)

        # Артефакти
        cm_path = os.path.join(args.models_dir, "confusion_matrix.png")
        plot_confusion_matrix(y_test, y_test_pred, cm_path)

        fi_path = os.path.join(args.models_dir, "feature_importance.png")
        plot_feature_importance(model, feature_names, fi_path)

        if not args.ci_mode:
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(fi_path)
            # Збереження моделі
            mlflow.sklearn.log_model(
                model,
                "random_forest_model",
                registered_model_name="CreditCardFraudDetector",
            )
            run_id = mlflow.active_run().info.run_id
            print(f"\nЗапуск завершено. Run ID: {run_id}")

        if args.ci_mode:
            save_ci_artifacts(model, y_test, y_test_pred, y_test_proba)
            print("CI-режим завершено.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
