# MLOps Lab 1 — Credit Card Fraud Detection

## Датасет
**Credit Card Fraud Detection** — виявлення шахрайських транзакцій.
- Джерело: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- ~284 807 транзакцій

## Структура проєкту
```
mlops_lab_1/
├── .gitignore
├── requirements.txt
├── README.md
├── venv/               # (не в Git)
├── data/
│   └── raw/
│       └── creditcard.csv  # (не в Git)
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   └── train.py
├── mlruns/             # (не в Git)
└── models/             # (не в Git)
```

## Швидкий старт

### 1. Встановлення залежностей
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/MacOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Завантаження даних
Завантажте `creditcard.csv` з Kaggle та помістіть у `data/raw/creditcard.csv`.

### 3. EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Тренування моделі
```bash
# Базовий запуск
python src/train.py

# З гіперпараметрами
python src/train.py --n_estimators 200 --max_depth 10 --threshold 0.3

# Повний набір аргументів
python src/train.py --n_estimators 100 --max_depth 5 --threshold 0.5 --test_size 0.2 --random_state 42
```

### 5. Перегляд результатів у MLflow UI
```bash
mlflow ui
# Відкрийте браузер: http://127.0.0.1:5000
```
