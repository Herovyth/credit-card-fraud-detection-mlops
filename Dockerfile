##
# Single multi-stage Dockerfile with multiple targets:
# - builder: builds wheels for heavy ML deps once
# - runtime: minimal python:slim image to run scripts (train/prepare/evaluate)
# - airflow: apache/airflow image with the same ML deps so DAG BashOperator works
##

FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to build some Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Prefer ML requirements if present; fallback to requirements.txt
COPY requirements-ml.txt ./requirements-ml.txt
COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip wheel setuptools \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-ml.txt


FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*

# Copy only what is needed to run scripts
COPY src/       ./src/
COPY config/    ./config/
COPY dvc.yaml   ./dvc.yaml
COPY .dvc/      ./.dvc/
COPY data/raw/*.dvc ./data/raw/

LABEL maintainer="developer" \
      project="creditcard-fraud-mlops"

CMD ["python", "--version"]


FROM apache/airflow:2.9.1-python3.11 AS airflow

USER root

# Не ставимо всі wheels з ML-builder: це перезаписує SQLAlchemy, Werkzeug, numpy тощо і ламає Airflow.
# Ставимо лише потрібні для DAG пакети з constraints Airflow 2.9.1 — так збережуться Flask 2.2 / Werkzeug 2.2.
COPY requirements-airflow-extras.txt /requirements-airflow-extras.txt
# Без офіційного constraints-файлу: mlflow/dvc часто конфліктують з pinned packaging/sqlalchemy у constraints.
# Тримємо Airflow на тій же мажорній версії; після ML-додатків відкочуємо Flask/Werkzeug (mlflow тягне Flask 3).
RUN python -m pip install --no-cache-dir \
    "apache-airflow==2.9.1" \
    -r /requirements-airflow-extras.txt \
    && python -m pip install --no-cache-dir --force-reinstall \
        "Flask==2.2.5" \
        "Werkzeug==2.2.3" \
        "Jinja2==3.1.3" \
        "MarkupSafe==2.1.5" \
        "click==8.1.7"

USER airflow