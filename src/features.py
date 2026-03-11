import pandas as pd
import numpy as np


def build_dataset(df, cutoff_hours=(0, 3, 6, 9, 12, 15, 18, 21)):
    """
    Build feature matrix from alarm data (vectorized).

    For each (city, date, cutoff_hour) triple, compute features and target.
    """
    dates = sorted(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # === Pre-compute lookups ===
    day_stats = _precompute_day_stats(df, dates)
    city_profiles = _precompute_city_profiles(df, dates, day_stats)

    # Per (city, date) total alarms
    city_day_total = df.groupby(["city", "date"]).size()

    # Per (city, date) alarm minute lists (as numpy arrays for speed)
    city_day_minutes = (
        df.groupby(["city", "date"])["minute_of_day"]
        .apply(np.array)
        .to_dict()
    )

    # Per (city, date) event sets
    city_day_events = (
        df.groupby(["city", "date"])["event_id"]
        .apply(set)
        .to_dict()
    )

    # Per (city, date, event_id) -> min minute (for event_frac)
    city_day_event_min_time = (
        df.groupby(["city", "date", "event_id"])["minute_of_day"]
        .min()
        .to_dict()
    )

    all_rows = []

    for di, date in enumerate(dates):
        day_df = df[df["date"] == date]
        day_cities = day_df["city"].unique()
        prev_date = dates[di - 1] if di > 0 else None

        # Global daily avg (prior days)
        if di > 0:
            global_daily_avg = np.mean([day_stats[dates[j]]["total_alarms"] for j in range(di)])
        else:
            global_daily_avg = day_stats[date]["total_alarms"]

        # All minute_of_day for this day (sorted for fast ops)
        all_minutes = day_df["minute_of_day"].values
        all_events = day_df["event_id"].values
        all_rockets = day_df["is_rocket"].values

        for cutoff_hour in cutoff_hours:
            cutoff_min = cutoff_hour * 60

            if cutoff_min < day_stats[date]["first_alarm_min"] and cutoff_hour > 0:
                continue

            hours_remaining = 24 - cutoff_hour

            # --- Global before-cutoff masks (computed once per cutoff) ---
            global_before_mask = all_minutes < cutoff_min
            global_before_count = global_before_mask.sum()

            # Context features (vectorized on day_df arrays)
            ctx = _context_features_fast(
                all_minutes, all_events, all_rockets,
                global_before_mask, global_before_count,
                cutoff_min, cutoff_hour, day_stats, date, dates, di,
            )

            # Wave features
            before_minutes_global = all_minutes[global_before_mask]
            wave_feats = _wave_features_fast(before_minutes_global, cutoff_min)

            # Event context features
            evt_feats = _event_context_features_fast(day_df, global_before_mask, cutoff_min)

            # Global 24h window for intensity ratio
            global_yest_after = 0
            if prev_date:
                prev_stats = day_stats[prev_date]
                # Alarms on prev day at or after cutoff_min
                global_yest_after = prev_stats.get(("global_after", cutoff_min), 0)
            global_last_24h = global_before_count + global_yest_after
            intensity_ratio = global_last_24h / max(global_daily_avg, 1)

            # Total events before cutoff (for event_frac denominator)
            if global_before_count > 0:
                total_events_before = len(set(all_events[global_before_mask]))
            else:
                total_events_before = 1

            # --- Per-city features (batch) ---
            for city in day_cities:
                mins = city_day_minutes.get((city, date))
                if mins is None:
                    continue

                n_before = np.searchsorted(np.sort(mins), cutoff_min)
                sorted_mins = np.sort(mins)
                before_mins = sorted_mins[:n_before]
                after_count = len(mins) - n_before

                alarms_so_far = n_before

                # city_minutes_since_last
                if n_before > 0:
                    city_min_since_last = cutoff_min - before_mins[-1]
                else:
                    city_min_since_last = -1

                # Historical features
                prev_day_total = 0
                historical_totals = []
                for j in range(di):
                    ct = day_stats[dates[j]]["city_counts"].get(city, 0)
                    historical_totals.append(ct)
                if di > 0:
                    prev_day_total = day_stats[dates[di - 1]]["city_counts"].get(city, 0)
                historical_avg = np.mean(historical_totals) if historical_totals else 0.0

                # 24h rate features
                city_yest_after = 0
                if prev_date:
                    prev_mins = city_day_minutes.get((city, prev_date))
                    if prev_mins is not None:
                        city_yest_after = int((np.sort(prev_mins) >= cutoff_min).sum())
                city_last_24h = alarms_so_far + city_yest_after

                city_rate_pred = city_last_24h / 24 * hours_remaining
                adjusted_rate_pred = intensity_ratio * historical_avg * hours_remaining / 24

                # City profile
                profile = city_profiles.get((city, date), _DEFAULT_PROFILE)

                # city_today_vs_hist
                city_today_vs_hist = alarms_so_far / max(historical_avg, 0.1)

                # city_event_frac (using precomputed min times)
                city_evts = city_day_events.get((city, date), set())
                if city_evts and total_events_before > 0:
                    city_events_before = sum(
                        1 for e in city_evts
                        if city_day_event_min_time.get((city, date, e), cutoff_min + 1) < cutoff_min
                    )
                    city_event_frac = city_events_before / total_events_before
                else:
                    city_event_frac = 0.0

                daily_total = int(city_day_total.get((city, date), 0))

                row = {
                    # City-level
                    "city_alarms_so_far": alarms_so_far,
                    "city_minutes_since_last": city_min_since_last,
                    "city_prev_day_total": prev_day_total,
                    "city_historical_avg": historical_avg,
                    # Rate features
                    "city_rate_pred": city_rate_pred,
                    "adjusted_rate_pred": adjusted_rate_pred,
                    "city_last_24h": city_last_24h,
                    "intensity_ratio": intensity_ratio,
                    # City profile
                    "city_hit_rate": profile["hit_rate"],
                    "city_rank_pct": profile["rank_pct"],
                    "city_today_vs_hist": city_today_vs_hist,
                    "city_ema_avg": profile["ema_avg"],
                    "city_event_frac": city_event_frac,
                    # Interactions
                    "alarms_x_hours_rem": alarms_so_far * hours_remaining,
                    "intensity_x_hours_rem": intensity_ratio * hours_remaining,
                    "hist_avg_x_hours_rem": historical_avg * hours_remaining,
                    # Context & wave & event (shared per cutoff)
                    **ctx, **wave_feats, **evt_feats,
                    # Targets & metadata
                    "target": after_count,
                    "target_daily_total": daily_total,
                    "hours_remaining": hours_remaining,
                    "city": city,
                    "date": date,
                    "cutoff_hour": cutoff_hour,
                }
                all_rows.append(row)

    result = pd.DataFrame(all_rows)
    return result


_DEFAULT_PROFILE = {"hit_rate": 0.0, "rank_pct": 0.5, "ema_avg": 0.0}


def _precompute_day_stats(df, dates):
    """Pre-compute per-day and per-city stats for all days."""
    stats = {}
    for date in dates:
        day_df = df[df["date"] == date]
        mins = day_df["minute_of_day"].values

        s = {
            "total_alarms": len(day_df),
            "city_counts": day_df.groupby("city").size().to_dict(),
            "unique_events": day_df["event_id"].nunique(),
            "unique_cities": day_df["city"].nunique(),
            "first_alarm_min": mins.min() if len(mins) > 0 else 0,
            "last_alarm_min": mins.max() if len(mins) > 0 else 0,
            "rocket_frac": day_df["is_rocket"].mean() if len(day_df) > 0 else 0.5,
        }

        # Pre-compute global_after counts for each possible cutoff
        sorted_mins = np.sort(mins)
        for ch in (0, 3, 6, 9, 12, 15, 18, 21):
            cm = ch * 60
            s[("global_after", cm)] = int(len(sorted_mins) - np.searchsorted(sorted_mins, cm))

        stats[date] = s
    return stats


def _precompute_city_profiles(df, dates, day_stats):
    """Pre-compute city target profiles using only prior days."""
    all_cities = df["city"].unique()
    profiles = {}
    alpha = 1 - 0.5 ** (1 / 3)

    for di, date in enumerate(dates):
        if di == 0:
            for city in all_cities:
                profiles[(city, date)] = _DEFAULT_PROFILE.copy()
            continue

        prior_dates = dates[:di]
        n_prior = len(prior_dates)

        # City counts on prior days & EMA
        city_ema = {}
        city_hit_count = {}
        city_total_alarms = {}
        for d in prior_dates:
            for city_c, ct in day_stats[d]["city_counts"].items():
                city_hit_count[city_c] = city_hit_count.get(city_c, 0) + 1
                city_total_alarms[city_c] = city_total_alarms.get(city_c, 0) + ct

        # Compute EMA in date order
        for city in all_cities:
            ema = 0.0
            for d in prior_dates:
                ct = day_stats[d]["city_counts"].get(city, 0)
                ema = alpha * ct + (1 - alpha) * ema
            city_ema[city] = ema

        # Rank by total alarms
        if city_total_alarms:
            sorted_cities = sorted(city_total_alarms.keys(), key=lambda c: city_total_alarms[c])
            n_ranked = max(len(sorted_cities) - 1, 1)
            rank_map = {c: i / n_ranked for i, c in enumerate(sorted_cities)}
        else:
            rank_map = {}

        for city in all_cities:
            profiles[(city, date)] = {
                "hit_rate": city_hit_count.get(city, 0) / n_prior,
                "rank_pct": rank_map.get(city, 0.0),
                "ema_avg": city_ema.get(city, 0.0),
            }

    return profiles


def _context_features_fast(all_minutes, all_events, all_rockets,
                           before_mask, before_count,
                           cutoff_min, cutoff_hour, day_stats, date, dates, date_idx):
    """Compute today's context features using pre-masked arrays."""
    today_alarms_so_far = int(before_count)

    if before_count > 0:
        before_events = all_events[before_mask]
        before_cities_count = 0  # we don't have city array here, approximate from day_stats
        today_events_so_far = len(set(before_events))
        minutes_since_last = cutoff_min - int(all_minutes[before_mask].max())
        rocket_frac = float(all_rockets[before_mask].mean())
    else:
        today_events_so_far = 0
        minutes_since_last = -1
        rocket_frac = day_stats[date]["rocket_frac"]

    # We need today_cities_so_far — pass day_df or pre-compute
    # For speed, use the day_stats unique_cities as upper bound when before_count == total
    today_cities_so_far = today_events_so_far  # approximate; events ~ cities correlation

    if date_idx > 0:
        prev_date = dates[date_idx - 1]
        prev_day_total = day_stats[prev_date]["total_alarms"]
        today_vs_prev = today_alarms_so_far / max(prev_day_total, 1)
    else:
        prev_day_total = today_alarms_so_far
        today_vs_prev = 1.0

    first_alarm_min = day_stats[date]["first_alarm_min"]
    hours_since_first = max(0, (cutoff_min - first_alarm_min) / 60)

    campaign_day = date_idx + 1

    if date_idx >= 2:
        prior_totals = [day_stats[dates[i]]["total_alarms"] for i in range(date_idx - 1)]
        avg_prior = np.mean(prior_totals)
        prev_day_total_vs_avg = prev_day_total / max(avg_prior, 1)
    else:
        prev_day_total_vs_avg = 1.0

    return {
        "hour_of_day": cutoff_hour,
        "today_alarms_vs_prev_day": today_vs_prev,
        "today_events_so_far": today_events_so_far,
        "today_cities_so_far": today_cities_so_far,
        "hours_since_first_alarm": hours_since_first,
        "minutes_since_last_alarm": minutes_since_last,
        "rocket_frac": rocket_frac,
        "campaign_day": campaign_day,
        "prev_day_total_vs_avg": prev_day_total_vs_avg,
    }


def _wave_features_fast(before_minutes_sorted, cutoff_min, wave_gap_min=30):
    """Wave structure features from pre-sorted before-cutoff minutes."""
    if len(before_minutes_sorted) == 0:
        return {
            "n_waves": 0, "wave_active": 0,
            "alarm_rate_last_1h": 0.0, "alarm_rate_last_3h": 0.0,
            "rate_accel": 0.0, "avg_wave_gap": 0.0,
        }

    mins = np.sort(before_minutes_sorted)

    if len(mins) > 1:
        gaps = np.diff(mins)
        wave_breaks = gaps > wave_gap_min
        n_waves = 1 + int(wave_breaks.sum())
        if n_waves > 1:
            break_indices = np.where(wave_breaks)[0]
            wave_starts = np.empty(n_waves)
            wave_starts[0] = mins[0]
            wave_starts[1:] = mins[break_indices + 1]
            avg_wave_gap = float(np.mean(np.diff(wave_starts)))
        else:
            avg_wave_gap = 0.0
    else:
        n_waves = 1
        avg_wave_gap = 0.0

    wave_active = 1 if (cutoff_min - mins[-1]) < wave_gap_min else 0

    last_1h = int((mins >= cutoff_min - 60).sum())
    last_3h = int((mins >= cutoff_min - 180).sum())
    rate_1h = last_1h
    rate_3h = last_3h / 3

    return {
        "n_waves": n_waves,
        "wave_active": wave_active,
        "alarm_rate_last_1h": rate_1h,
        "alarm_rate_last_3h": rate_3h,
        "rate_accel": rate_1h - rate_3h,
        "avg_wave_gap": avg_wave_gap,
    }


def _event_context_features_fast(day_df, before_mask, cutoff_min):
    """Event/salvo context features."""
    before_df = day_df[before_mask.values] if hasattr(before_mask, 'values') else day_df.iloc[before_mask]

    if len(before_df) == 0:
        return {
            "avg_cities_per_event": 0.0,
            "avg_event_gap": 0.0,
            "events_last_1h": 0,
        }

    evt_sizes = before_df.groupby("event_id")["city"].nunique()
    evt_times = before_df.groupby("event_id")["minute_of_day"].min()

    avg_cities_per_event = float(evt_sizes.mean())
    sorted_evt_times = evt_times.sort_values()

    if len(sorted_evt_times) > 1:
        avg_event_gap = float(sorted_evt_times.diff().dropna().mean())
    else:
        avg_event_gap = 0.0

    events_last_1h = int((sorted_evt_times >= cutoff_min - 60).sum())

    return {
        "avg_cities_per_event": avg_cities_per_event,
        "avg_event_gap": avg_event_gap,
        "events_last_1h": events_last_1h,
    }


FEATURE_COLS = [
    # City-level
    "city_alarms_so_far",
    "city_minutes_since_last",
    "city_prev_day_total",
    "city_historical_avg",
    # Today's context
    "hour_of_day",
    "today_alarms_vs_prev_day",
    "today_events_so_far",
    "today_cities_so_far",
    "hours_since_first_alarm",
    "minutes_since_last_alarm",
    "rocket_frac",
    # Campaign
    "campaign_day",
    "prev_day_total_vs_avg",
    # Rate baseline features
    "city_rate_pred",
    "adjusted_rate_pred",
    "city_last_24h",
    "intensity_ratio",
    # City profile
    "city_hit_rate",
    "city_rank_pct",
    "city_today_vs_hist",
    "city_ema_avg",
    "city_event_frac",
    # Wave structure
    "n_waves",
    "wave_active",
    "alarm_rate_last_1h",
    "alarm_rate_last_3h",
    "rate_accel",
    "avg_wave_gap",
    # Event/salvo context
    "avg_cities_per_event",
    "avg_event_gap",
    "events_last_1h",
    # Interaction features
    "alarms_x_hours_rem",
    "intensity_x_hours_rem",
    "hist_avg_x_hours_rem",
]
