import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from .features import FEATURE_COLS


def naive_baseline(X, y_train):
    """Predict the mean of training targets for all samples."""
    return np.full(len(X), y_train.mean())


def city_avg_baseline(test):
    """Predict max(city_historical_avg - city_alarms_so_far, 0)."""
    return np.maximum(test["city_historical_avg"] - test["city_alarms_so_far"], 0).values


def rate_baselines(test):
    """Return the pre-computed rate baseline predictions from features."""
    y_rate = np.maximum(test["city_rate_pred"].values, 0)
    y_adj = np.maximum(test["adjusted_rate_pred"].values, 0)
    return y_rate, y_adj


def train_and_evaluate(dataset, train_dates, test_dates):
    """Train models on train_dates, evaluate on test_dates."""
    train = dataset[dataset["date"].isin(train_dates)]
    test = dataset[dataset["date"].isin(test_dates)]

    X_train = train[FEATURE_COLS].values
    y_train = train["target"].values
    X_test = test[FEATURE_COLS].values
    y_test = test["target"].values

    print(f"Train: {len(train)} samples ({len(train_dates)} days)")
    print(f"Test:  {len(test)} samples ({len(test_dates)} days)")
    print(f"Target stats - train mean: {y_train.mean():.2f}, test mean: {y_test.mean():.2f}")
    print()

    results = {}

    # === Baselines ===
    print("--- Baselines ---")
    y_pred_naive = naive_baseline(X_test, y_train)
    results["Naive (mean)"] = _eval("Naive (mean)", y_test, y_pred_naive)

    y_pred_city_avg = city_avg_baseline(test)
    results["City avg baseline"] = _eval("City avg baseline", y_test, y_pred_city_avg)

    y_pred_prev = np.maximum(test["city_prev_day_total"] - test["city_alarms_so_far"], 0).values
    results["City prev-day"] = _eval("City prev-day", y_test, y_pred_prev)

    y_rate, y_adj = rate_baselines(test)
    results["City 24h rate"] = _eval("City 24h rate", y_test, y_rate)
    results["Adjusted rate"] = _eval("Adjusted rate", y_test, y_adj)

    # === Direct models (predict remaining alarms) ===
    print("\n--- Direct models (target = remaining alarms) ---")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["LinReg"] = _eval("LinReg", y_test, y_pred_lr)

    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    results["Ridge"] = _eval("Ridge", y_test, y_pred_ridge)

    xgb = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred_xgb = xgb.predict(X_test)
    results["XGBoost"] = _eval("XGBoost", y_test, y_pred_xgb)

    # === Rate models (predict daily total, scale by hours_remaining/24) ===
    print("\n--- Rate models (target = daily total, scale by hours_rem/24) ---")
    y_train_daily = train["target_daily_total"].values
    y_test_daily = test["target_daily_total"].values
    hours_rem = test["hours_remaining"].values
    alarms_so_far = test["city_alarms_so_far"].values

    ridge_rate = Ridge(alpha=10.0)
    ridge_rate.fit(X_train, y_train_daily)
    y_pred_daily_ridge = ridge_rate.predict(X_test)
    y_pred_remaining_ridge = np.maximum(y_pred_daily_ridge * hours_rem / 24, 0)
    results["Ridge (rate)"] = _eval("Ridge (rate)", y_test, y_pred_remaining_ridge)

    # Also try: predicted_remaining = max(predicted_daily - alarms_so_far, 0)
    y_pred_remaining_ridge2 = np.maximum(y_pred_daily_ridge - alarms_so_far, 0)
    results["Ridge (rate-sub)"] = _eval("Ridge (rate-sub)", y_test, y_pred_remaining_ridge2)

    xgb_rate = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    xgb_rate.fit(X_train, y_train_daily, eval_set=[(X_test, y_test_daily)], verbose=False)
    y_pred_daily_xgb = xgb_rate.predict(X_test)
    y_pred_remaining_xgb = np.maximum(y_pred_daily_xgb * hours_rem / 24, 0)
    results["XGBoost (rate)"] = _eval("XGBoost (rate)", y_test, y_pred_remaining_xgb)

    y_pred_remaining_xgb2 = np.maximum(y_pred_daily_xgb - alarms_so_far, 0)
    results["XGBoost (rate-sub)"] = _eval("XGBoost (rate-sub)", y_test, y_pred_remaining_xgb2)

    # === Feature importances ===
    print("\nTop XGBoost (direct) feature importances:")
    imp = xgb.feature_importances_
    for name, score in sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1])[:15]:
        print(f"  {name:30s} {score:.4f}")

    print("\nRidge (direct) top coefficients:")
    for name, coef in sorted(zip(FEATURE_COLS, ridge.coef_), key=lambda x: -abs(x[1]))[:10]:
        print(f"  {name:30s} {coef:+.4f}")

    # === Scatter plots ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    _scatter_ax(axes[0, 0], y_test, y_pred_ridge, "Ridge (direct)")
    _scatter_ax(axes[0, 1], y_test, y_pred_xgb, "XGBoost (direct)")
    _scatter_ax(axes[1, 0], y_test, y_pred_remaining_ridge, "Ridge (rate)")
    _scatter_ax(axes[1, 1], y_test, y_pred_remaining_xgb, "XGBoost (rate)")
    plt.tight_layout()
    plt.savefig("scatter_test.png", dpi=100)
    print(f"\nScatter plot saved to scatter_test.png")
    plt.close()

    return results, {"ridge": ridge, "xgb": xgb, "xgb_rate": xgb_rate}


def _eval(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name:20s} | R²={r2:.4f} | MAE={mae:.4f}")
    return {"r2": r2, "mae": mae}


def _scatter_ax(ax, y_true, y_pred, title):
    ax.scatter(y_true, y_pred, alpha=0.1, s=5)
    lims = [0, max(y_true.max(), np.max(y_pred)) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual remaining alarms")
    ax.set_ylabel("Predicted remaining alarms")
    ax.set_title(title)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
