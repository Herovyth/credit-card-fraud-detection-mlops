import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main(input_file: str, output_dir: str):
    print(f"Читання даних: {input_file}")

    if not os.path.exists(input_file):
        print(f"Файл не знайдено — {input_file}")
        sys.exit(1)

    df = pd.read_csv(input_file)
    print(f"Розмір датасету: {df.shape}")
    print(f"Шахрайство: {df['Class'].sum()} записів ({df['Class'].mean()*100:.3f}%)")

    # 3. Feature Engineering
    # Масштабування Amount та Time (V1-V28 вже масштабовані PCA)
    scaler = StandardScaler()
    df = df.copy()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])

    # Додаткова ознака: година доби (Time у секундах, доба = 86400 с)
    # Відновлюємо оригінальний Time лише для обчислення — вже масштабований,
    # тому використовуємо індекс рядка як proxy-ознаку порядку
    df["hour_of_day"] = (df.index % (86400 // 1)) // 3600  # умовна cyclical ознака
    df["log_amount_raw"] = np.log1p(
        pd.read_csv(input_file, usecols=["Amount"])["Amount"]
    )

    print(f"Ознаки після Feature Engineering: {df.shape[1] - 1} (без Class)")

    # 4. Розділення на train / test
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # 5. Збереження результатів
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df["Class"] = y_train.values
    train_path = os.path.join(output_dir, "train.csv")
    train_df.to_csv(train_path, index=False)
    print(f"Збережено: {train_path}")

    test_df = X_test.copy()
    test_df["Class"] = y_test.values
    test_path = os.path.join(output_dir, "test.csv")
    test_df.to_csv(test_path, index=False)
    print(f"Збережено: {test_path}")

    print("Підготовка даних завершена")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Використання: python src/prepare.py <input_csv> <output_dir>")
        print("Приклад:      python src/prepare.py data/raw/creditcard.csv data/prepared")
        sys.exit(1)

    main(input_file=sys.argv[1], output_dir=sys.argv[2])