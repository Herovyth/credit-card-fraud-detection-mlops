"""
scripts/make_sample.py — Генерація малої підвибірки для CI
Запускати локально один раз перед першим push.

Використання:
    python scripts/make_sample.py
    python scripts/make_sample.py --n-rows 5000 --input data/raw/creditcard.csv
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/creditcard.csv")
    parser.add_argument("--output", default="data/sample/creditcard_sample.csv")
    parser.add_argument("--n-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = PROJECT_ROOT / args.input
    dst = PROJECT_ROOT / args.output

    if not src.exists():
        print(f"ПОМИЛКА: {src} не знайдено. Завантажте creditcard.csv з Kaggle.")
        return

    df = pd.read_csv(src)
    print(
        f"Повний датасет: {df.shape} | шахрайство: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)"
    )

    # Стратифікована вибірка — зберігає пропорцію класів
    fraud = df[df["Class"] == 1]
    normal = df[df["Class"] == 0]

    n_fraud = max(10, int(args.n_rows * df["Class"].mean()))
    n_normal = args.n_rows - n_fraud

    sample = (
        pd.concat(
            [
                normal.sample(n=n_normal, random_state=args.seed),
                fraud.sample(n=min(n_fraud, len(fraud)), random_state=args.seed),
            ]
        )
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(dst, index=False)
    print(f"Sample збережено: {dst}")
    print(
        f"Розмір: {sample.shape} | шахрайство: {sample['Class'].sum()} ({sample['Class'].mean()*100:.2f}%)"
    )
    print(f"\nДодай файл у Git:")
    print(f"  git add {args.output}")
    print(f"  git commit -m 'Add CI sample dataset'")


if __name__ == "__main__":
    main()
