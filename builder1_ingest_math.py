"""
AquaSentinel — Builder 1: Ingestion & Math Engine
=================================================

Turns raw DWLR telemetry (6-hourly depth-to-water readings) into the monthly,
station-level early-warning dataset that Builder 2 (ML) and Builder 3
(dashboard) consume. This Builder is fully standalone: it reads CSVs, writes
one CSV, and never imports another Builder's code.

VALIDATION GATEWAY
------------------
  1. Reject readings whose Water_Level is non-numeric, zero, or negative.
  2. Exclude any station missing more than 30% of its EXPECTED readings.
     Expected = (full shared analysis window in hours) / EXPECTED_INTERVAL_HOURS.
     The window is max(Time)-min(Time) across ALL stations — NOT each
     station's own span — so a station that silently went offline is caught.
     We deliberately do NOT auto-infer the interval from the data: random
     missing readings just stretch the median gap and hide the loss.
  3. Inner-join Latitude/Longitude/Block_ID from stations.csv and
     Net_Availability/Official_Category from blocks.csv. These columns must
     survive to the final CSV — the dashboard's map depends on them.

MATH ENGINE (per station, applied in this order)
------------------------------------------------
  1. Rolling 30-day z-score outlier removal with min_periods=3 (a window of
     1 point has undefined std and would wrongly drop that row). Drop
     abs(z) > 3.0.
  2. Resample to one row per calendar month (mean).
  3. Confidence tier from total months of history:
     <12 -> LOW, 12-23 -> MEDIUM, >=24 -> HIGH.
  4. Depth_Decline_Proxy = 12-month diff of monthly Water_Level. The first 12
     months of each station are left NaN ON PURPOSE. Backfilling them (even
     with the station's own later mean) is look-ahead leakage and would render
     as if it were real telemetry on the dashboard.
  5. Estimated_SoE_Proxy_Pct = (Depth_Decline_Proxy / Net_Availability) * 100
     — the multiplier is exactly 100 (see the hand-trace in
     test_builder1_math.py: a 2 m decline / 10 units == 20.0, not 200.0).
  6. Estimated_Category: <=70 Safe, <=90 Semi-Critical, <=100 Critical,
     >100 Over-Exploited — but ONLY when Confidence == 'HIGH', otherwise
     "Insufficient history".
  7. Drift_Flag = True when Confidence == 'HIGH' AND Estimated_Category
     != Official_Category.
  8. Hysteresis alert, in chronological order: ON above 72%, OFF below 68%,
     held in between. State is reset at the START of every station's loop so
     one station's alert never leaks into the next station's rows.

OUTPUT:  processed_math_data.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tunable constants (explained in the docstring above)
# ---------------------------------------------------------------------------
EXPECTED_INTERVAL_HOURS = 6          # 6h readings; set 1 for an hourly feed
MAX_MISSING_FRACTION = 0.30          # stations missing more than this are dropped
OUTLIER_WINDOW = "30D"               # rolling window for the z-score filter
OUTLIER_MIN_PERIODS = 3              # see docstring — never 1
OUTLIER_Z_MAX = 3.0
DECLINE_DIFF_MONTHS = 12             # 12-month change in monthly Water_Level
CONFIDENCE_MEDIUM_MIN = 12           # >=12 months -> MEDIUM
CONFIDENCE_HIGH_MIN = 24             # >=24 months -> HIGH
HYSTERESIS_ON_PCT = 72.0             # alert turns ON above this
HYSTERESIS_OFF_PCT = 68.0            # alert turns OFF below this

CATEGORY_SAFE = 70
CATEGORY_SEMI = 90
CATEGORY_CRITICAL = 100


# ===========================================================================
# 1. VALIDATION GATEWAY
# ===========================================================================
def validate_readings(telemetry):
    """Drop non-numeric, zero, or negative Water_Level readings.

    Returns a cleaned copy and prints what was rejected (so a first-time coder
    can see the gateway working).
    """
    df = telemetry.copy()
    before = len(df)

    # Non-numeric -> NaN, then drop NaN rows (this also catches empty cells).
    df["Water_Level"] = pd.to_numeric(df["Water_Level"], errors="coerce")
    df = df.dropna(subset=["Water_Level"])

    # Zero or negative depth-to-water is physically impossible for our sensors.
    df = df[df["Water_Level"] > 0.0]

    dropped = before - len(df)
    print(f"[validate] Rejected {dropped:,} of {before:,} raw readings "
          f"({dropped / before * 100:.2f}%)")
    return df.reset_index(drop=True)


def expected_reading_count(telemetry, interval_hours=EXPECTED_INTERVAL_HOURS):
    """Expected readings per station over the FULL shared analysis window.

    The window spans max(Time)-min(Time) across ALL stations, not per station.
    """
    span_hours = (telemetry["Time"].max() - telemetry["Time"].min()).total_seconds() / 3600.0
    expected = int(span_hours / interval_hours)
    print(f"[validate] Shared analysis window = {span_hours:.1f} hours "
          f"-> {expected} expected readings per station @ {interval_hours}h")
    return expected


def drop_stations_with_poor_coverage(telemetry, expected_count):
    """Exclude stations that report < 70% of their expected readings."""
    counts = telemetry.groupby("Station_ID").size()
    min_ok = expected_count * (1.0 - MAX_MISSING_FRACTION)
    keep = counts[counts >= min_ok]

    for sid in counts.index.difference(keep.index):
        frac = counts[sid] / expected_count
        print(f"[validate] EXCLUDING station {sid}: only "
              f"{counts[sid]:,}/{expected_count:,} readings "
              f"({frac * 100:.1f}% of expected)")

    return telemetry[telemetry["Station_ID"].isin(keep.index)].copy()


def join_station_and_block_metadata(telemetry, stations, blocks):
    """Inner-join station + block metadata onto the surviving readings.

    Latitude/Longitude/Block_ID/Net_Availability/Official_Category must ride
    along all the way to the output CSV — Builder 3's map reads them.
    """
    stations_meta = stations[["Station_ID", "Block_ID", "Latitude", "Longitude"]]
    blocks_meta = blocks[["Block_ID", "Net_Availability", "Official_Category"]]

    merged = telemetry.merge(stations_meta, on="Station_ID", how="inner")
    merged = merged.merge(blocks_meta, on="Block_ID", how="inner")

    # how="inner" intentionally drops readings whose station/block metadata is
    # missing — the map could not place them anyway.
    print(f"[validate] After metadata joins: {len(merged):,} readings, "
          f"{merged['Station_ID'].nunique()} stations")
    return merged


# ===========================================================================
# 2. MATH ENGINE — pure helpers (each small and testable)
# ===========================================================================
def compute_soe_proxy(decline, availability):
    """Estimated_SoE_Proxy_Pct = (decline / availability) * 100.

    The multiplier is EXACTLY 100. Example (see tests): a 2 m decline against
    10 units of availability == 20.0, never 200.0.
    """
    availability = pd.to_numeric(availability, errors="coerce")
    decline = pd.to_numeric(decline, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(availability > 0, (decline / availability) * 100.0, np.nan)
    return result


def confidence_tier(months):
    """Confidence from total months of history: <12 LOW, 12-23 MEDIUM, >=24 HIGH."""
    if pd.isna(months) or months < CONFIDENCE_MEDIUM_MIN:
        return "LOW"
    if months < CONFIDENCE_HIGH_MIN:
        return "MEDIUM"
    return "HIGH"


def classify_proxy(proxy, confidence):
    """Map a proxy % to a category, gated on HIGH confidence."""
    if pd.isna(proxy):
        return np.nan
    if confidence != "HIGH":
        return "Insufficient history"
    if proxy <= CATEGORY_SAFE:
        return "Safe"
    if proxy <= CATEGORY_SEMI:
        return "Semi-Critical"
    if proxy <= CATEGORY_CRITICAL:
        return "Critical"
    return "Over-Exploited"


def hysteresis_flags(proxy_series):
    """Chronological hysteresis alert.

    State machine: starts OFF. Turns ON when the proxy exceeds 72%, turns OFF
    when it drops below 68%, and HOLDS in between. NaN proxy keeps the
    previous state. This is the one per-station loop where state is carried
    forward — it is freshly reset by the caller for every station.
    """
    flags = []
    active = False
    for val in proxy_series:
        if pd.isna(val):
            flags.append(active)              # no info -> hold state
            continue
        if active and val < HYSTERESIS_OFF_PCT:
            active = False
        if not active and val > HYSTERESIS_ON_PCT:
            active = True
        flags.append(active)
    return flags


# ===========================================================================
# 3. MATH ENGINE — orchestration
# ===========================================================================
def remove_outliers_per_station(telemetry):
    """Rolling 30-day z-score filter, per station, min_periods=3, abs(z) > 3."""
    cleaned = []
    for sid, grp in telemetry.groupby("Station_ID", sort=True):
        g = grp.set_index("Time").sort_index()
        roll_mean = g["Water_Level"].rolling(OUTLIER_WINDOW,
                                             min_periods=OUTLIER_MIN_PERIODS).mean()
        roll_std = g["Water_Level"].rolling(OUTLIER_WINDOW,
                                            min_periods=OUTLIER_MIN_PERIODS).std()
        # If the window is constant (std == 0) it is NOT an outlier — leave z=0.
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(roll_std > 0, (g["Water_Level"] - roll_mean) / roll_std, 0.0)
        g = g[np.abs(z) <= OUTLIER_Z_MAX]
        cleaned.append(g.reset_index())
    return pd.concat(cleaned, ignore_index=True)


def resample_monthly(telemetry):
    """One row per station per calendar month (mean Water_Level).

    Months with no readings are kept as NaN so the calendar stays aligned for
    the 12-month diff and for Builder 2's time-series split.
    """
    out = []
    for sid, grp in telemetry.groupby("Station_ID", sort=True):
        g = grp.set_index("Time").sort_index()
        monthly = g["Water_Level"].resample("MS").mean()          # MS = month start
        meta = g[["Block_ID", "Latitude", "Longitude",
                  "Net_Availability", "Official_Category"]].iloc[0]

        frame = pd.DataFrame({"Time": monthly.index,
                              "Water_Level": monthly.values})
        for col, val in meta.items():
            frame[col] = val
        frame["Station_ID"] = sid
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def compute_math_for_station(frame):
    """Apply the proxy / category / drift / hysteresis math to ONE station.

    `frame` is the monthly-resampled block for a single station, already
    carrying its metadata columns. Returns the enriched frame.
    """
    frame = frame.sort_values("Time").reset_index(drop=True)

    months_of_history = int(frame["Water_Level"].notna().sum())
    confidence = confidence_tier(months_of_history)

    # 12-month change in monthly depth-to-water. First 12 months stay NaN —
    # do NOT backfill them (look-ahead leakage, see module docstring).
    depth_decline = frame["Water_Level"].diff(DECLINE_DIFF_MONTHS)
    frame["Depth_Decline_Proxy"] = depth_decline.round(3)

    # x100 proxy — sanity-checked by test_builder1_math.py.
    proxy = compute_soe_proxy(depth_decline, frame["Net_Availability"])
    frame["Estimated_SoE_Proxy_Pct"] = np.round(proxy, 2)

    frame["Estimated_Category"] = [
        classify_proxy(p, confidence) for p in frame["Estimated_SoE_Proxy_Pct"]
    ]
    frame["Confidence"] = confidence

    # Drift: HIGH confidence + estimated disagrees with the official category.
    drift = (frame["Confidence"] == "HIGH") & (
        frame["Estimated_Category"] != frame["Official_Category"]
    ) & frame["Estimated_Category"].notna()
    frame["Drift_Flag"] = drift

    # Hysteresis alert. `hysteresis_flags` carries state across rows, but that
    # state is local to this call — it cannot leak into the next station.
    frame["Alert_Active"] = hysteresis_flags(frame["Estimated_SoE_Proxy_Pct"])

    return frame


def process_all_stations(monthly):
    """Run the per-station math over every station and combine the results."""
    frames = []
    for sid, grp in monthly.groupby("Station_ID", sort=True):
        frames.append(compute_math_for_station(grp))
    return pd.concat(frames, ignore_index=True)


def run_pipeline(telemetry_path="raw_dwlr_telemetry.csv",
                 stations_path="stations.csv",
                 blocks_path="blocks.csv",
                 interval_hours=EXPECTED_INTERVAL_HOURS,
                 out_path="processed_math_data.csv"):
    """End-to-end Builder 1. Returns the output DataFrame and writes the CSV."""
    # -- inputs -------------------------------------------------------------
    telemetry = pd.read_csv(telemetry_path, parse_dates=["Time"])
    stations = pd.read_csv(stations_path)
    blocks = pd.read_csv(blocks_path)
    print(f"[ingest] Loaded {len(telemetry):,} raw readings from {telemetry_path}")

    # -- validation gateway -------------------------------------------------
    telemetry = validate_readings(telemetry)
    expected = expected_reading_count(telemetry, interval_hours=interval_hours)
    telemetry = drop_stations_with_poor_coverage(telemetry, expected)
    telemetry = join_station_and_block_metadata(telemetry, stations, blocks)

    # -- math engine --------------------------------------------------------
    telemetry = remove_outliers_per_station(telemetry)
    print(f"[math] After 30-day z-score outlier removal: {len(telemetry):,} readings")
    monthly = resample_monthly(telemetry)
    result = process_all_stations(monthly)

    # -- output -------------------------------------------------------------
    cols = [
        "Time", "Station_ID", "Water_Level",
        "Latitude", "Longitude", "Block_ID",
        "Net_Availability", "Official_Category",
        "Depth_Decline_Proxy", "Estimated_SoE_Proxy_Pct",
        "Estimated_Category", "Confidence", "Drift_Flag", "Alert_Active",
    ]
    result = result[cols].sort_values(["Station_ID", "Time"]).reset_index(drop=True)
    result.to_csv(out_path, index=False)

    print(f"[output] Wrote {out_path}: {len(result):,} rows, "
          f"{result['Station_ID'].nunique()} stations")
    print(result["Confidence"].value_counts().to_string())
    return result


def main():
    parser = argparse.ArgumentParser(description="AquaSentinel Builder 1")
    parser.add_argument("--interval-hours", type=int,
                        default=int(os.environ.get("AQUA_INTERVAL_HOURS",
                                                   EXPECTED_INTERVAL_HOURS)),
                        help="Telemetry interval (hours). Default 6, use 1 for hourly feeds.")
    args = parser.parse_args()
    run_pipeline(interval_hours=args.interval_hours)


if __name__ == "__main__":
    sys.exit(main())
