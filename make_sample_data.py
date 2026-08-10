"""
make_sample_data.py — Synthetic demo-data generator for AquaSentinel
====================================================================

This script fabricates the three INPUT CSVs that Builder 1 expects:

    raw_dwlr_telemetry.csv   (Time, Station_ID, Water_Level)
    stations.csv             (Station_ID, Block_ID, Latitude, Longitude)
    blocks.csv               (Block_ID, Net_Availability, Official_Category)

The data is FAKE — created purely so the whole AquaSentinel pipeline can be
run and demoed end-to-end without real government telemetry. It mimics a
6-hourly DWLR feed across Punjab/Haryana (real latitude/longitude, so the
dashboard map looks right) and deliberately contains:

  * seasonal monsoon recharge (water table shallower around September),
  * long-term decline trends (deeper water table = larger Water_Level),
  * sensor noise,
  * a handful of huge spike outliers (to exercise Builder 1's z-score filter),
  * a few zero / negative readings (to exercise the validation gateway),
  * random missing readings (to exercise the expected-coverage rule),
  * one station that is missing ~46% of its readings, so Builder 1 must
    EXCLUDE it (it is genuinely too degraded to trust).

Run:  python make_sample_data.py

Output files are written to the current directory.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
SEED = 2026
START = pd.Timestamp("2023-01-01 00:00:00")   # shared analysis window start
END = pd.Timestamp("2026-06-30 23:00:00")     # shared analysis window end
INTERVAL_HOURS = 6                            # one telemetry reading every 6h

MISSING_FRACTION_NORMAL = 0.04                # 4% random gaps for good stations
MISSING_FRACTION_BAD = 0.46                   # 46% gaps for the "degraded" station
SPIKE_PROB = 0.004                            # chance of a huge sensor spike
BAD_READING_PROB = 0.002                      # chance of a 0 / negative reading


def _build_blocks():
    """One row per administrative block: availability + official GEC category.

    Net_Availability is in arbitrary "availability units"; the estimated
    extraction proxy divides the 12-month depth decline (metres) by it and
    multiplies by 100, so the numbers below were picked to make the demo's
    stations land on a pleasing spread of categories (Safe ... Over-Exploited).
    """
    return pd.DataFrame(
        [
            {"Block_ID": "BLK-01", "Net_Availability": 4.2, "Official_Category": "Over-Exploited"},
            {"Block_ID": "BLK-02", "Net_Availability": 4.8, "Official_Category": "Over-Exploited"},
            {"Block_ID": "BLK-03", "Net_Availability": 5.1, "Official_Category": "Critical"},
            {"Block_ID": "BLK-04", "Net_Availability": 5.6, "Official_Category": "Critical"},
            {"Block_ID": "BLK-05", "Net_Availability": 6.2, "Official_Category": "Semi-Critical"},
            {"Block_ID": "BLK-06", "Net_Availability": 6.8, "Official_Category": "Semi-Critical"},
            {"Block_ID": "BLK-07", "Net_Availability": 7.5, "Official_Category": "Safe"},
            {"Block_ID": "BLK-08", "Net_Availability": 4.5, "Official_Category": "Critical"},
            {"Block_ID": "BLK-09", "Net_Availability": 8.0, "Official_Category": "Safe"},
            {"Block_ID": "BLK-10", "Net_Availability": 4.9, "Official_Category": "Over-Exploited"},
        ]
    )


# One station specification: (Station_ID, Block_ID, lat, lon, base_depth,
# trend_m_per_year, seasonal_amp, noise_std).
# base_depth is the average depth-to-water (m); a POSITIVE trend means the
# water table is dropping (depth-to-water growing), as in real over-extraction.
def _build_stations(blocks, rng):
    """Three DWLR stations per block, scattered a little around the block centre.

    Trend ranges are biased by the block's official category: over-exploited
    blocks fall fastest, safe blocks are flat or gently recovering.
    """
    stations = []

    cat_trend = {
        "Over-Exploited": (4.5, 7.5),
        "Critical": (2.2, 4.2),
        "Semi-Critical": (0.8, 2.0),
        "Safe": (-0.6, 0.3),
    }
    cat_base = {
        "Over-Exploited": (26.0, 38.0),
        "Critical": (18.0, 30.0),
        "Semi-Critical": (12.0, 22.0),
        "Safe": (5.0, 16.0),
    }

    for block in blocks.itertuples():
        t_lo, t_hi = cat_trend[block.Official_Category]
        b_lo, b_hi = cat_base[block.Official_Category]
        for i in range(3):
            sid = f"ST-{block.Block_ID[-2:]}-{i + 1}"   # e.g. ST-01-1
            lat = block.Latitude + rng.uniform(-0.18, 0.18)
            lon = block.Longitude + rng.uniform(-0.18, 0.18)
            base = rng.uniform(b_lo, b_hi)
            trend = rng.uniform(t_lo, t_hi)
            # Within a block, one station deviates clearly from its peers so the
            # dashboard shows at least a couple of Drift_Flag examples.
            if i == 1:
                trend += rng.uniform(0.6, 1.4)
            amp = rng.uniform(1.5, 4.0)
            noise = rng.uniform(0.15, 0.35)
            stations.append(
                {
                    "Station_ID": sid,
                    "Block_ID": block.Block_ID,
                    "Latitude": round(lat, 5),
                    "Longitude": round(lon, 5),
                    "base_depth": base,
                    "trend": trend,
                    "amp": amp,
                    "noise_std": noise,
                }
            )
    return pd.DataFrame(stations)


def _simulate_one_station(station, rng):
    """Create the full 6-hourly telemetry series for a single station.

    depth(t) = base_depth + trend * years_elapsed
               - seasonal_amp * cos(2*pi*(month - 9)/12)   (shallowest in Sept)
               + gaussian noise

    Then we sprinkle in: a missing-data fraction, a few huge spikes, and a few
    zero / negative readings. The caller decides the missing fraction so the
    "degraded" station can be forced past the 30% rule.
    """
    steps = int(((END - START).total_seconds() / 3600) / INTERVAL_HOURS) + 1
    times = pd.date_range(START, periods=steps, freq=f"{INTERVAL_HOURS}h")

    years = (times - START).total_seconds() / (365.25 * 24 * 3600.0)
    month = times.month.astype(float)
    season = station["amp"] * np.cos(2.0 * np.pi * (month - 9.0) / 12.0)

    depth = (
        station["base_depth"]
        + station["trend"] * years
        - season
        + rng.normal(0.0, station["noise_std"], size=steps)
    )
    depth = np.round(np.clip(depth, 0.5, 120.0), 2)   # keep everything positive

    df = pd.DataFrame({"Time": times, "Water_Level": depth})
    df["Station_ID"] = station["Station_ID"]

    # --- random missing readings -------------------------------------------
    missing_mask = rng.random(steps) < station["missing_fraction"]
    df = df[~missing_mask].copy()

    # --- huge sensor spikes (caught by Builder 1's 30-day z-score) ---------
    n = len(df)
    spike_mask = rng.random(n) < SPIKE_PROB
    df.loc[spike_mask, "Water_Level"] += rng.uniform(35.0, 80.0, size=spike_mask.sum())

    # --- zero / negative readings (caught by the validation gateway) -------
    bad_mask = rng.random(n) < BAD_READING_PROB
    df.loc[bad_mask, "Water_Level"] = -rng.uniform(0.0, 3.0, size=bad_mask.sum())

    return df[["Time", "Station_ID", "Water_Level"]]


def main():
    rng = np.random.default_rng(SEED)

    blocks = _build_blocks()
    # Real-ish geography: the blocks are spread across Punjab / Haryana.
    block_centres = {
        "BLK-01": (30.75, 75.55), "BLK-02": (31.05, 75.95), "BLK-03": (30.40, 76.30),
        "BLK-04": (31.35, 76.70), "BLK-05": (30.10, 75.10), "BLK-06": (30.60, 74.80),
        "BLK-07": (31.80, 76.05), "BLK-08": (30.25, 75.85), "BLK-09": (31.60, 75.40),
        "BLK-10": (30.90, 76.20),
    }
    blocks["Latitude"] = blocks["Block_ID"].map(lambda b: block_centres[b][0])
    blocks["Longitude"] = blocks["Block_ID"].map(lambda b: block_centres[b][1])

    stations = _build_stations(blocks, rng)

    # Add one deliberately degraded station: it reports only ~54% of the
    # readings it should, so Builder 1 must exclude it (>30% missing).
    degraded = {
        "Station_ID": "ST-EXCL-1",
        "Block_ID": "BLK-01",
        "Latitude": 30.82,
        "Longitude": 75.62,
        "base_depth": 30.0,
        "trend": 5.0,
        "amp": 2.5,
        "noise_std": 0.30,
    }
    stations = pd.concat([stations, pd.DataFrame([degraded])], ignore_index=True)

    # Attach per-station missing fraction and simulate.
    stations["missing_fraction"] = MISSING_FRACTION_NORMAL
    stations.loc[stations["Station_ID"] == "ST-EXCL-1", "missing_fraction"] = (
        MISSING_FRACTION_BAD
    )

    frames = []
    for row in stations.itertuples():
        spec = {
            "Station_ID": row.Station_ID,
            "base_depth": row.base_depth,
            "trend": row.trend,
            "amp": row.amp,
            "noise_std": row.noise_std,
            "missing_fraction": row.missing_fraction,
        }
        frames.append(_simulate_one_station(pd.Series(spec), rng))

    telemetry = pd.concat(frames, ignore_index=True)
    telemetry = telemetry.sort_values(["Time", "Station_ID"]).reset_index(drop=True)

    # stations.csv — keep only the metadata columns Builder 1 needs.
    stations_out = stations[["Station_ID", "Block_ID", "Latitude", "Longitude"]]
    blocks_out = blocks[["Block_ID", "Net_Availability", "Official_Category"]]

    telemetry.to_csv("raw_dwlr_telemetry.csv", index=False)
    stations_out.to_csv("stations.csv", index=False)
    blocks_out.to_csv("blocks.csv", index=False)

    print(f"Wrote raw_dwlr_telemetry.csv : {len(telemetry):,} rows, "
          f"{telemetry['Station_ID'].nunique()} stations")
    print(f"Wrote stations.csv           : {len(stations_out)} stations")
    print(f"Wrote blocks.csv             : {len(blocks_out)} blocks")
    print("(All data is synthetic demo data — do not use for real decisions.)")


if __name__ == "__main__":
    main()
