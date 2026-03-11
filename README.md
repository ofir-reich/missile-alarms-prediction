# Missile Alarm Prediction

Predicting how many alarms remain today for a specific city in Israel, given alarms observed so far during Iranian missile/drone attacks.

## Data

- Source: Israeli Home Front Command alarm data (`data/alarms.csv`)
- Filtered to Iran-origin attacks from Feb 28, 2026 onwards
- ~12 attack days, ~750–1400 cities per day, median 2 alarms per city per day

## Approach

For each (city, day, cutoff_hour) triple, we build features and predict the number of remaining alarms for that city after the cutoff.

**Features** (34 total) include:
- **City-level**: alarms so far, minutes since last alarm, historical average, EMA, hit rate, rank percentile
- **Today's context**: global alarm count, events/cities hit, rocket vs drone fraction, wave structure (burst detection, acceleration)
- **Campaign-level**: day number, intensity trend vs prior days
- **Rate baselines**: 24h city rate, intensity-adjusted rate (as features, not just baselines)
- **Interactions**: alarms × hours remaining, intensity × hours remaining

**Two prediction framings**:
- **Direct**: predict remaining alarms
- **Rate-sub**: predict the city's daily total, then subtract alarms already observed — this outperforms direct prediction since it separates "how big is today's attack for this city?" from "how far along are we?"

## Results

Train: Feb 28 – Mar 8 (9 days, 70k samples). Test: Mar 9–11 (3 days, 23k samples).

| Model | R² | MAE |
|---|---|---|
| Naive (predict mean) | -0.29 | 1.35 |
| City avg − observed | -0.69 | 1.25 |
| City prev-day − observed | -0.14 | 1.05 |
| City 24h rate | -0.49 | 1.23 |
| Adjusted rate (intensity × city avg) | 0.21 | 0.88 |
| Ridge (direct) | 0.24 | 0.97 |
| XGBoost (direct) | 0.33 | 0.81 |
| Ridge (rate-sub) | 0.40 | 0.76 |
| **XGBoost (rate-sub)** | **0.43** | **0.73** |

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib xgboost

# Train and evaluate
python main.py

# Predict remaining alarms for today
python predict_today.py
```

## File Structure

```
├── data/alarms.csv        # Raw alarm data
├── src/
│   ├── load_data.py       # Load and filter CSV
│   ├── features.py        # Feature engineering (vectorized)
│   └── model.py           # Model training, evaluation, plots
├── main.py                # Train/eval entry point
└── predict_today.py       # Live prediction for current day
```
