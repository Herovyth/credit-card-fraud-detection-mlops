FROM python:3.11-slim AS builder

WORKDIR /build

# Системні залежності для компіляції
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Встановлюємо в окрему директорію
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime

WORKDIR /app

# Копіюємо встановлені пакети зі stage builder
COPY --from=builder /install /usr/local

# Копіюємо лише потрібний код (без venv, mlruns, __pycache__ тощо)
COPY src/       ./src/
COPY config/    ./config/
COPY dvc.yaml   ./dvc.yaml
COPY .dvc/      ./.dvc/
COPY data/raw/*.dvc ./data/raw/

# Мітки образу
LABEL maintainer="developer" \
      project="creditcard-fraud-mlops"

# За замовчуванням — допомога; конкретна команда передається через docker run
CMD ["python", "--version"]