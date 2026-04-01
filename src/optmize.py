"""
optimize.py - HPO з Optuna + MLflow + Hydra
Лабораторна робота №3 | MLOps

Використання:
    python src/optimize.py
    python src/optimize.py hpo.n_trials=50 hpo.sampler=random
    python src/optimize.py model.type=logistic_regression hpo.metric=roc_auc
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import hashlib
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import hydra


def file_md5(path: str) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


# 1. Відтворюваність

def set_global_seed(seed: int) -> None:
    """Фіксує seed у Python та NumPy для відтворюваності."""
    random.seed(seed)
    np.random.seed(seed)


# 2. Завантаження підготовлених даних (результат prepare.py з ЛР2)

def load_prepared(prepared_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Читає train.csv / test.csv, які були створені src/prepare.py у ЛР2.
    Шлях розраховується відносно кореня проєкту (де знаходиться src/),
    щоб Hydra не ламав відносні шляхи на Windows.
    """
    project_root = Path(__file__).resolve().parents[1]
    abs_prepared = project_root / prepared_dir

    train_path = str(abs_prepared / "train.csv")
    test_path  = str(abs_prepared / "test.csv")

    for p in [train_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Файл не знайдено: {p}\n"
                "Спочатку запустіть: python src/prepare.py data/raw/creditcard.csv data/prepared"
            )

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Class"]).values
    y_train = train_df["Class"].values
    X_test  = test_df.drop(columns=["Class"]).values
    y_test  = test_df["Class"].values

    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Ознаки: {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test


# 3. Побудова моделі

def build_model(model_type: str, params: Dict[str, Any], seed: int) -> Any:
    """
    Повертає sklearn-модель за типом та словником гіперпараметрів.
    Для logistic_regression огортає в Pipeline зі StandardScaler.
    """
    if model_type == "random_forest":
        return RandomForestClassifier(
            random_state=seed, n_jobs=-1, class_weight="balanced", **params
        )
    if model_type == "logistic_regression":
        clf = LogisticRegression(random_state=seed, max_iter=500, **params)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    raise ValueError(f"Невідомий model.type='{model_type}'. Очікується: random_forest | logistic_regression")


# 4. Оцінка моделі

def evaluate(
    model: Any,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    metric: str,
) -> float:
    """Навчає модель на train, оцінює на test."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if metric == "f1":
        return float(f1_score(y_test, y_pred, average="binary"))
    if metric == "roc_auc":
        y_score = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )
        return float(roc_auc_score(y_test, y_score))
    raise ValueError(f"Непідтримувана метрика '{metric}'. Використовуйте: f1 | roc_auc")


def evaluate_cv(
    model: Any,
    X: np.ndarray, y: np.ndarray,
    metric: str, seed: int, n_splits: int = 5,
) -> float:
    """5-fold Stratified CV — опційний режим (hpo.use_cv=true)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = [
        evaluate(clone(model), X[tr], y[tr], X[te], y[te], metric)
        for tr, te in cv.split(X, y)
    ]
    return float(np.mean(scores))


# 5. Search space

def suggest_params(trial: optuna.Trial, model_type: str, cfg: DictConfig) -> Dict[str, Any]:
    """
    Описує простір пошуку для кожного типу моделі.
    Межі беруться з config/config.yaml -> hpo.<model_type>.
    """
    if model_type == "random_forest":
        sp = cfg.hpo.random_forest
        return {
            "n_estimators":      trial.suggest_int("n_estimators",      sp.n_estimators.low,      sp.n_estimators.high),
            "max_depth":         trial.suggest_int("max_depth",         sp.max_depth.low,         sp.max_depth.high),
            "min_samples_split": trial.suggest_int("min_samples_split", sp.min_samples_split.low, sp.min_samples_split.high),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf",  sp.min_samples_leaf.low,  sp.min_samples_leaf.high),
        }
    if model_type == "logistic_regression":
        sp = cfg.hpo.logistic_regression
        return {
            "C":       trial.suggest_float("C", sp.C.low, sp.C.high, log=True),
            "solver":  trial.suggest_categorical("solver",  list(sp.solver)),
            "penalty": trial.suggest_categorical("penalty", list(sp.penalty)),
        }
    raise ValueError(f"Невідомий model.type='{model_type}'")


# 6. Sampler

def make_sampler(sampler_name: str, seed: int, grid_space: Dict = None) -> optuna.samplers.BaseSampler:
    name = sampler_name.lower()
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "grid":
        if not grid_space:
            raise ValueError("Для sampler='grid' потрібно задати grid_space у конфігурації.")
        return optuna.samplers.GridSampler(search_space=grid_space)
    raise ValueError(f"Невідомий sampler '{sampler_name}'. Очікується: tpe | random | grid")


# 7. Objective function (child runs всередині parent run)

def objective_factory(cfg: DictConfig, X_train, X_test, y_train, y_test):
    """
    Фабрика: повертає objective-функцію для Optuna.
    Кожен trial логується як nested (child) MLflow run.
    """
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cfg.model.type, cfg)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type",   cfg.model.type)
            mlflow.set_tag("sampler",      cfg.hpo.sampler)
            mlflow.set_tag("seed",         cfg.seed)
            mlflow.log_params(params)

            model = build_model(cfg.model.type, params=params, seed=cfg.seed)

            if cfg.hpo.use_cv:
                X_all = np.concatenate([X_train, X_test], axis=0)
                y_all = np.concatenate([y_train, y_test], axis=0)
                score = evaluate_cv(model, X_all, y_all, cfg.hpo.metric, cfg.seed, cfg.hpo.cv_folds)
            else:
                score = evaluate(model, X_train, y_train, X_test, y_test, cfg.hpo.metric)

            mlflow.log_metric(cfg.hpo.metric, score)

        return score

    return objective


# 8. Реєстрація моделі (опційно)

def register_model_if_enabled(model_uri: str, model_name: str, stage: str) -> None:
    """Реєструє модель у MLflow Model Registry та переводить у Staging."""
    client = mlflow.tracking.MlflowClient()
    mv = mlflow.register_model(model_uri, model_name)
    client.transition_model_version_stage(name=model_name, version=mv.version, stage=stage)
    client.set_model_version_tag(model_name, mv.version, "registered_by", "lab3")
    print(f"Модель '{model_name}' v{mv.version} -> {stage}")


# 9. main

def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    print(f"Конфігурація:\n{OmegaConf.to_yaml(cfg)}")

    # Абсолютний корінь проєкту — незалежно від того, куди Hydra змінить cwd
    project_root = Path(__file__).resolve().parents[1]

    # MLflow — абсолютний шлях до SQLite-бази
    tracking_uri = f"sqlite:///{(project_root / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Дані — ті самі, що підготував prepare.py у ЛР2
    X_train, X_test, y_train, y_test = load_prepared(cfg.data.prepared_dir)

    # Grid sampler потребує явного search_space
    grid_space = None
    if cfg.hpo.sampler.lower() == "grid":
        m = cfg.model.type
        raw = OmegaConf.to_container(cfg.hpo.grid[m], resolve=True)
        grid_space = {k: list(v) for k, v in raw.items()}

    sampler = make_sampler(cfg.hpo.sampler, seed=cfg.seed, grid_space=grid_space)

    mlflow.set_tag("train_data_md5", file_md5(str(project_root / cfg.data.prepared_dir / "train.csv")))
    mlflow.set_tag("test_data_md5",  file_md5(str(project_root / cfg.data.prepared_dir / "test.csv")))

    # Parent run (Study)
    with mlflow.start_run(run_name=f"hpo_{cfg.hpo.sampler}_{cfg.model.type}") as parent_run:
        mlflow.set_tag("model_type", cfg.model.type)
        mlflow.set_tag("sampler",    cfg.hpo.sampler)
        mlflow.set_tag("seed",       cfg.seed)
        mlflow.set_tag("n_trials",   cfg.hpo.n_trials)
        mlflow.set_tag("metric",     cfg.hpo.metric)

        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config_resolved.json")

        # Optuna Study
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction=cfg.hpo.direction,
            sampler=sampler,
            study_name=f"{cfg.hpo.sampler}_{cfg.model.type}_seed{cfg.seed}",
        )
        objective = objective_factory(cfg, X_train, X_test, y_train, y_test)
        study.optimize(objective, n_trials=cfg.hpo.n_trials)

        # Результати
        best_trial = study.best_trial
        print(f"\nНайкращий trial #{best_trial.number}: {cfg.hpo.metric}={best_trial.value:.4f}")
        print(f"Найкращі параметри: {best_trial.params}")

        mlflow.log_metric(f"best_{cfg.hpo.metric}", float(best_trial.value))
        mlflow.log_dict(best_trial.params, "best_params.json")

        # Фінальна модель
        best_model = build_model(cfg.model.type, params=best_trial.params, seed=cfg.seed)
        final_score = evaluate(best_model, X_train, y_train, X_test, y_test, cfg.hpo.metric)
        mlflow.log_metric(f"final_{cfg.hpo.metric}", final_score)
        print(f"Фінальна модель: {cfg.hpo.metric}={final_score:.4f}")

        # Зберігаємо .pkl — абсолютний шлях
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = str(models_dir / "best_model.pkl")
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_model, artifact_path="model")

        if cfg.mlflow.register_model:
            model_uri = f"runs:/{parent_run.info.run_id}/model"
            register_model_if_enabled(model_uri, cfg.mlflow.model_name, cfg.mlflow.stage)

        print(f"\nParent Run ID: {parent_run.info.run_id}")


# Hydra entry point

@hydra.main(version_base=None, config_path="../config", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
