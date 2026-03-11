import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "alarms.csv"
MIN_DATE = "2026-02-28"
DRONE_DESC = "חדירת כלי טיס עוין"
ROCKET_DESC = "ירי רקטות וטילים"


def load_iran_alarms(path=DATA_PATH):
    """Load alarms CSV, filter to Iran origin and Feb 2026+."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["origin"] == "Iran"].copy()
    df = df[df["time"] >= MIN_DATE].copy()

    df["date"] = df["time"].dt.date.astype(str)
    df["hour"] = df["time"].dt.hour
    df["minute_of_day"] = df["time"].dt.hour * 60 + df["time"].dt.minute
    df["is_rocket"] = (df["description"] == ROCKET_DESC).astype(int)

    # Rename cities column for clarity
    df = df.rename(columns={"cities": "city", "id": "event_id"})
    return df.sort_values("time").reset_index(drop=True)


if __name__ == "__main__":
    df = load_iran_alarms()
    print(f"Loaded {len(df)} alarms across {df['date'].nunique()} days, {df['city'].nunique()} cities")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nAlarms per day:\n{df.groupby('date').size()}")
