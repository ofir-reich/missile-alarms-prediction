#!/usr/bin/env python3
"""Predict remaining alarms per city for today at the current hour."""

import sys
from datetime import datetime

import numpy as np
from sklearn.linear_model import Ridge

from src.load_data import load_iran_alarms
from src.features import build_dataset, FEATURE_COLS


def main():
    print("Loading data...")
    df = load_iran_alarms()

    today = datetime.now().strftime("%Y-%m-%d")
    today_alarms = df[df["date"] == today]

    if len(today_alarms) == 0:
        print(f"No alarms recorded for today ({today}). Nothing to predict.")
        sys.exit(0)

    print(f"Today ({today}): {len(today_alarms)} alarms so far, {today_alarms['city'].nunique()} cities\n")

    # Use current hour (rounded down to nearest cutoff)
    current_hour = datetime.now().hour
    cutoff_hour = (current_hour // 3) * 3
    print(f"Current hour: {current_hour}, using cutoff: {cutoff_hour}\n")

    # Build dataset with just this cutoff hour
    print("Building features...")
    dataset = build_dataset(df, cutoff_hours=[cutoff_hour])

    # Train on all days except today
    dates = sorted(dataset["date"].unique())
    train_dates = [d for d in dates if d != today]
    today_data = dataset[dataset["date"] == today]
    train_data = dataset[dataset["date"].isin(train_dates)]

    if len(today_data) == 0:
        print(f"No data for today at cutoff hour {cutoff_hour}.")
        sys.exit(0)

    X_train = train_data[FEATURE_COLS].values
    y_train = train_data["target"].values
    X_today = today_data[FEATURE_COLS].values

    # Train Ridge model
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)

    predictions = model.predict(X_today)
    predictions = np.maximum(predictions, 0).round(1)

    # Output predictions sorted by predicted remaining alarms
    today_data = today_data.copy()
    today_data["predicted_remaining"] = predictions

    results = today_data[["city", "city_alarms_so_far", "predicted_remaining"]].sort_values(
        "predicted_remaining", ascending=False
    )

    print(f"Predictions for {today} after hour {cutoff_hour}:")
    print(f"{'City':40s} {'Alarms so far':>15s} {'Predicted remaining':>20s}")
    print("-" * 78)
    for _, row in results.head(30).iterrows():
        print(f"{row['city']:40s} {row['city_alarms_so_far']:>15.0f} {row['predicted_remaining']:>20.1f}")

    if len(results) > 30:
        print(f"\n... and {len(results) - 30} more cities")

    total_predicted = predictions.sum()
    print(f"\nTotal predicted remaining alarms (all cities): {total_predicted:.0f}")


if __name__ == "__main__":
    main()
