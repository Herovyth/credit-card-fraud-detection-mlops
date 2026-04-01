import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Константи

DATA_PATH = os.getenv("DATA_PATH", "data/raw/creditcard.csv")
F1_THRESHOLD = float(os.getenv("F1_THRESHOLD", "0.70"))

# Очікувані колонки датасету Credit Card Fraud
REQUIRED_COLUMNS = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}


# Fixtures


@pytest.fixture(scope="module")
def sample_df():
    path = PROJECT_ROOT / DATA_PATH
    assert path.exists(), (
        f"Sample CSV не знайдено: {path}\n" "Запустіть: python scripts/make_sample.py"
    )
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def metrics():
    path = PROJECT_ROOT / "metrics.json"
    assert path.exists(), "metrics.json не знайдено. Спочатку запустіть train.py"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# PRE-TRAIN ТЕСТИ — перевірка вхідних даних (без тренування)


class TestDataPreTrain:

    def test_file_exists(self):
        path = PROJECT_ROOT / DATA_PATH
        assert path.exists(), f"Файл не знайдено: {path}"

    def test_schema_columns(self, sample_df):
        missing = REQUIRED_COLUMNS - set(sample_df.columns)
        assert not missing, f"Відсутні колонки: {sorted(missing)}"

    def test_minimum_rows(self, sample_df):
        assert (
            sample_df.shape[0] >= 100
        ), f"Занадто мало рядків: {sample_df.shape[0]} (мінімум 100)"

    def test_no_critical_nulls(self, sample_df):
        critical = ["Class", "Amount", "Time"]
        for col in critical:
            null_count = sample_df[col].isnull().sum()
            assert null_count == 0, f"Колонка '{col}' містить {null_count} пропусків"

    def test_target_binary(self, sample_df):
        unique = set(sample_df["Class"].unique())
        assert unique <= {0, 1}, f"Неочікувані значення у Class: {unique}"

    def test_both_classes_present(self, sample_df):
        counts = sample_df["Class"].value_counts()
        assert 0 in counts.index, "Клас 0 (Normal) відсутній у вибірці"
        assert 1 in counts.index, "Клас 1 (Fraud) відсутній у вибірці"
        assert (
            counts[1] >= 5
        ), f"Занадто мало прикладів класу Fraud: {counts[1]} (мінімум 5)"

    def test_amount_non_negative(self, sample_df):
        neg = (sample_df["Amount"] < 0).sum()
        assert neg == 0, f"Знайдено {neg} рядків з від'ємним Amount"

    def test_feature_ranges(self, sample_df):
        v_cols = [f"V{i}" for i in range(1, 29)]
        for col in v_cols:
            assert np.isfinite(
                sample_df[col]
            ).all(), f"Колонка {col} містить inf або NaN"


# POST-TRAIN ТЕСТИ — перевірка артефактів та Quality Gate


class TestArtifactsPostTrain:

    def test_model_pkl_exists(self):
        assert (
            PROJECT_ROOT / "models/best_model.pkl"
        ).exists(), "best_model.pkl не знайдено"

    def test_metrics_json_exists(self):
        assert (PROJECT_ROOT / "metrics.json").exists(), "metrics.json не знайдено"

    def test_confusion_matrix_exists(self):
        assert (
            PROJECT_ROOT / "models/confusion_matrix.png"
        ).exists(), "confusion_matrix.png не знайдено"

    def test_metrics_json_has_required_keys(self, metrics):
        required = {"accuracy", "f1", "precision", "recall", "roc_auc"}
        missing = required - set(metrics.keys())
        assert not missing, f"metrics.json не містить ключів: {missing}"

    def test_metrics_values_in_range(self, metrics):
        for key, val in metrics.items():
            assert 0.0 <= float(val) <= 1.0, f"Метрика '{key}' поза межами [0,1]: {val}"

    def test_model_can_be_loaded(self):
        model = joblib.load(PROJECT_ROOT / "models/best_model.pkl")
        assert hasattr(model, "predict"), "Завантажений об'єкт не є sklearn-моделлю"
        assert hasattr(model, "predict_proba"), "Модель не підтримує predict_proba"

    def test_model_output_shape(self, sample_df):
        model = joblib.load(PROJECT_ROOT / "models/best_model.pkl")
        train_df = pd.read_csv(PROJECT_ROOT / "data" / "prepared" / "train.csv")
        X = train_df.drop(columns=["Class"]).head(10)
        preds = model.predict(X)
        assert preds.shape == (10,), f"Неочікувана форма виходу: {preds.shape}"

    def test_quality_gate_f1(self, metrics):
        f1 = float(metrics["f1"])
        assert (
            f1 >= F1_THRESHOLD
        ), f"Quality Gate не пройдено: f1={f1:.4f} < threshold={F1_THRESHOLD:.2f}"

    def test_quality_gate_roc_auc(self, metrics):
        roc_auc = float(metrics["roc_auc"])
        assert roc_auc >= 0.85, f"ROC-AUC занадто низький: {roc_auc:.4f} < 0.85"
