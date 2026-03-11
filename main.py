#!/usr/bin/env python3
"""Train and evaluate missile alarm prediction models."""

from src.load_data import load_iran_alarms
from src.features import build_dataset
from src.model import train_and_evaluate


def main():
    print("Loading data...")
    df = load_iran_alarms()
    print(f"Loaded {len(df)} alarms, {df['date'].nunique()} days, {df['city'].nunique()} cities\n")

    print("Building features...")
    dataset = build_dataset(df)
    print(f"Dataset: {len(dataset)} samples, {len(dataset.columns)} columns\n")

    # Temporal split: train on earlier days, test on last 3
    dates = sorted(dataset["date"].unique())
    test_dates = dates[-3:]  # last 3 days
    train_dates = dates[:-3]

    print(f"Train dates: {train_dates[0]} to {train_dates[-1]}")
    print(f"Test dates:  {test_dates[0]} to {test_dates[-1]}\n")

    results, models = train_and_evaluate(dataset, train_dates, test_dates)


if __name__ == "__main__":
    main()
