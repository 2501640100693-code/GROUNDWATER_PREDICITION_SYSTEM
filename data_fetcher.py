"""
AquaSentinel — data_fetcher.py (ingestion + live-telemetry layer)
=================================================================

Standalone script that handles groundwater data ingestion and simulation.
It sits ABOVE Builder 1 and feeds it: whatever happens here, Builder 1 is
re-run afterwards so `processed_math_data.csv` (the dashboard's single source
of truth) always reflects the latest readings. Like every Builder, it talks to
the rest of the pipeline ONLY through static CSV files.

Two entry points:

  1. fetch_open_government_data(source_path | url)
     Parse an official India-WRIS / Data.gov.in CSV export into the three CSVs
     Builder 1 expects:

         raw_dwlr_telemetry.csv   (Time, Station_ID, Water_Level)
         stations.csv             (Station_ID, Block_ID, Latitude, Longitude)
         blocks.csv               (Block_ID, Net_Availability, Official_Category)

     The real portals have NO stable column names (DWLR dumps use "Station
     Code"/"Water Level (m)", Data.gov.in uses "WELL_ID"/"DEPTH_TO_WATER",
     GEC block datasets use "BLOCK"/"NET_GROUNDWATER_AVAILABILITY", ...). So we
     never hard-code one schema: every column is located by fuzzy alias
     matching (case / space / underscore / parenthesis-insensitive). A file
     that lacks a required field fails loudly and names the missing columns.

  2. simulate_live_telemetry_stream(station_ids, num_days=30)
     Append realistic 6-hourly depth-to-water readings for the coming N days.
     Each station's new series continues smoothly from its last recorded
     reading and models:
       * monsoon recharge cycles  — the water table is shallowest ~September
         (cosine seasonal term, phase-matched to the anchor so there is no
         discontinuity at the append boundary);
       * extraction drawdown      — the station's own long-term falling trend
         (estimated from its recent history by a robust linear fit);
       * sensor noise.
     Fresh stations (no history) get a seeded baseline instead of a single
     isolated spike, so Builder 1 can still classify them.

  3. run_live_update(num_days=30)
     Thin wrapper used by the dashboard's "Fetch Live Telemetry Update"
     button (demo mode) — appends simulated telemetry and re-runs Builder 1.

CLI:
    python data_fetcher.py --simulate [--days 30] [--stations ST-01-1 ST-01-2]
    python data_fetcher.py --fetch path/to/official.csv
    python data_fetcher.py --fetch https://.../dwlr_export.csv
    (add --no-builder1 to skip the Builder 1 re-run)
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# File / process constants
# ---------------------------------------------------------------------------
TELEMETRY_CSV = "raw_dwlr_telemetry.csv"
STATIONS_CSV = "stations.csv"
BLOCKS_CSV = "blocks.csv"
BUILDER1_SCRIPT = "builder1_ingest_math.py"
INTERVAL_HOURS = 6                       # one reading every 6 hours


# ---------------------------------------------------------------------------
# Official-schema column aliases (case / space / underscore-insensitive)
# ---------------------------------------------------------------------------
FIELD_ALIASES = {
    "time": [
        "time", "date", "observation_date", "obs_date", "datetime", "timestamp",
        "date_time", "date/time", "reading_time", "sample_date", "record_date",
        "well_obs_date", "obsdatetime",
    ],
    "station_id": [
        "station_id", "station_code", "station code", "stationcode",
        "well_id", "wellid", "wellid/dugwellid", "dwlr_id", "stationid",
        "station", "well_no", "wellno", "location_code", "station_name_code",
    ],
    "water_level": [
        "water_level", "depth_to_water", "depth_to_water_level", "dtw",
        "depthtowater", "waterlevel", "water_level_mbgl", "water_level_m",
        "water_level_(m)", "waterlevel(m)", "depth_to_water_(m)",
        "depth_to_water_level_m", "depthtowater_m",
    ],
    "latitude": ["latitude", "lat", "well_latitude", "latitude_deg", "lat_deg", "geolat"],
    "longitude": ["longitude", "lon", "long", "well_longitude", "longitude_deg", "lon_deg", "geolon"],
    "block_id": [
        "block", "block_id", "blockid", "block_name", "blockname", "tehsil",
        "taluka", "block_name_en", "admin_block",
    ],
    "availability": [
        "net_availability", "netavailability", "net_groundwater_availability",
        "netgwavailability", "net_groundwater_availability_mcm",
        "annual_availability", "availability", "total_net_groundwater_availability",
        "annual_extractable_groundwater_resource",
    ],
    "category": [
        "official_category", "category", "gec_category", "gecclassification",
        "groundwater_category", "category_gec", "classification",
        "stage_of_development_category",
    ],
}


def _normalise(name):
    """Column-name key: lowercase, drop every non-alphanumeric character."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


_NORM_ALIASES = {
    field: [_normalise(a) for a in aliases]
    for field, aliases in FIELD_ALIASES.items()
}


def _detect_column(df, field):
    """Return the actual column name matching `field`, or None."""
    norm = {_normalise(c): c for c in df.columns}
    for alias in _NORM_ALIASES[field]:
        if alias in norm:
            return norm[alias]
    return None


_DATE_SEP = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


def _infer_dayfirst(sample):
    """Guess dd/mm/yyyy (True) vs mm/dd/yyyy (False) from an ambiguous sample.

    An unambiguously day-first token ("31-12-2024") or month-first token
    ("12-31-2024") forces the right order; ambiguous tokens default to
    day-first because official Indian exports (India-WRIS / Data.gov.in) are
    overwhelmingly dd/mm/yyyy. ISO/4-digit-year-first strings parse correctly
    under either flag.
    """
    m = _DATE_SEP.search(str(sample))
    if m:
        a, b = m.group().replace("/", "-").replace(".", "-").split("-")[:2]
        if a.isdigit() and b.isdigit():
            if int(a) > 12 >= int(b):
                return True
            if int(b) > 12 >= int(a):
                return False
    return True


def _normalise_time_series(df, time_col):
    """Robust datetime parsing that infers the day/month order from the data."""
    vals = df[time_col].astype(str)
    dayfirst = _infer_dayfirst(vals.iloc[0]) if len(vals) else True
    return pd.to_datetime(vals, errors="coerce", dayfirst=dayfirst)


# ===========================================================================
# 1. OFFICIAL DATA FETCH / PARSE
# ===========================================================================
def fetch_open_government_data(source_path=None, url=None,
                               telemetry_out=TELEMETRY_CSV,
                               stations_out=STATIONS_CSV,
                               blocks_out=BLOCKS_CSV,
                               run_builder1=True):
    """Parse an official India-WRIS / Data.gov.in CSV into Builder 1's inputs.

    Pass EXACTLY ONE of `source_path` (local file) or `url` (HTTP(S) CSV).
    The schema is auto-detected by fuzzy column matching, so this one function
    handles DWLR telemetry dumps, state groundwater-yearbook exports and GEC
    block datasets.

    Returns a summary dict:
        {"telemetry_rows", "stations", "blocks", "builder1_ok", "messages"}
    """
    if (source_path is None) == (url is None):
        raise ValueError("Pass exactly one of source_path= or url=.")
    source = url if url else source_path

    try:
        df = pd.read_csv(source)
    except Exception as exc:                                     # noqa: BLE001
        raise RuntimeError(f"Could not read CSV from {source}: {exc}") from exc

    messages = []
    wrote_any = False
    n_stations = 0
    n_blocks = 0

    # -- 1a. telemetry: time + station + water level -------------------------
    time_col = _detect_column(df, "time")
    station_col = _detect_column(df, "station_id")
    wl_col = _detect_column(df, "water_level")

    if time_col and station_col and wl_col:
        telemetry = pd.DataFrame({
            "Time": _normalise_time_series(df, time_col),
            "Station_ID": df[station_col].astype(str).str.strip(),
            "Water_Level": pd.to_numeric(df[wl_col], errors="coerce"),
        }).dropna(subset=["Time", "Station_ID", "Water_Level"])
        telemetry["Water_Level"] = telemetry["Water_Level"].round(2)
        telemetry = (telemetry.sort_values(["Time", "Station_ID"])
                             .reset_index(drop=True))
        if not telemetry.empty:
            telemetry.to_csv(telemetry_out, index=False)
            wrote_any = True
            messages.append(
                f"Wrote {telemetry_out}: {len(telemetry):,} readings, "
                f"{telemetry['Station_ID'].nunique()} stations")
        else:
            messages.append("Skipped telemetry: no valid rows after parsing "
                            "(check the time / station / water-level columns).")
    else:
        missing = [n for n, c in (("time", time_col),
                                  ("station_id", station_col),
                                  ("water_level", wl_col)) if c is None]
        messages.append(f"Skipped telemetry: missing column(s) {missing}")

    # -- 1b. stations: station-level metadata (lat / lon / block) ------------
    lat_col = _detect_column(df, "latitude")
    lon_col = _detect_column(df, "longitude")
    blk_col = _detect_column(df, "block_id")

    if station_col and any(c is not None for c in (lat_col, lon_col, blk_col)):
        stations = pd.DataFrame({
            "Station_ID": df[station_col].astype(str).str.strip()})
        if lat_col:
            stations["Latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
        if lon_col:
            stations["Longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
        if blk_col:
            stations["Block_ID"] = df[blk_col].astype(str).str.strip()
        # Keep the columns Builder 1's inner join expects, even if absent.
        for col in ("Latitude", "Longitude", "Block_ID"):
            if col not in stations.columns:
                stations[col] = np.nan
        stations = (stations.drop_duplicates("Station_ID")
                            .reset_index(drop=True))
        if not stations.empty:
            stations.to_csv(stations_out, index=False)
            n_stations = len(stations)
            wrote_any = True
            messages.append(f"Wrote {stations_out}: {n_stations} stations")
        else:
            messages.append("Skipped stations: no rows.")
    else:
        messages.append("Skipped stations: no station_id + (lat|lon|block) "
                        "columns found.")

    # -- 1c. blocks: GEC block data (availability + official category) -------
    avail_col = _detect_column(df, "availability")
    cat_col = _detect_column(df, "category")

    if blk_col and (avail_col or cat_col):
        blocks = pd.DataFrame({
            "Block_ID": df[blk_col].astype(str).str.strip()})
        if avail_col:
            blocks["Net_Availability"] = pd.to_numeric(df[avail_col],
                                                       errors="coerce")
        if cat_col:
            blocks["Official_Category"] = df[cat_col].astype(str).str.strip()
        for col in ("Net_Availability", "Official_Category"):
            if col not in blocks.columns:
                blocks[col] = np.nan
        blocks = (blocks.dropna(subset=["Net_Availability", "Official_Category"])
                        .drop_duplicates("Block_ID")
                        .sort_values("Block_ID")
                        .reset_index(drop=True))
        if not blocks.empty:
            blocks.to_csv(blocks_out, index=False)
            n_blocks = len(blocks)
            wrote_any = True
            messages.append(f"Wrote {blocks_out}: {n_blocks} blocks")
        else:
            messages.append("Skipped blocks: no rows with both availability "
                            "and category.")
    else:
        messages.append("Skipped blocks: no block + (availability|category) "
                        "columns found.")

    # -- re-run Builder 1 so the dashboard sees the new data -----------------
    builder1_ok = None
    if run_builder1 and wrote_any:
        builder1_ok = _run_builder1()
        messages.append("Builder 1: " + ("OK" if builder1_ok else "FAILED — see stderr above"))

    for msg in messages:
        print(f"[fetch] {msg}")

    return {
        "telemetry_rows": int(len(df)),
        "stations": n_stations,
        "blocks": n_blocks,
        "builder1_ok": builder1_ok,
        "messages": messages,
    }


# ===========================================================================
# 2. LIVE TELEMETRY SIMULATION
# ===========================================================================
def _station_history(telemetry_path):
    """Load existing raw telemetry (or None when the file is absent/corrupt)."""
    if not os.path.exists(telemetry_path):
        return None
    try:
        df = pd.read_csv(telemetry_path, parse_dates=["Time"])
    except Exception:                                            # noqa: BLE001
        return None
    return df.dropna(subset=["Water_Level"]).copy()


def _estimate_station_params(hist):
    """Robust linear trend (m/year) + seasonal amplitude (m) from history.

    Fits on the most recent ~2 years of readings, winzorised so a few sensor
    spikes cannot wreck the slope. Positive trend = water table falling
    (over-extraction drawdown).
    """
    trend, amp = 2.0, 2.5          # sensible defaults for an over-used aquifer
    if hist is None or len(hist) < 12:
        return trend, amp

    h = hist.sort_values("Time").tail(min(len(hist), 730 * 4))   # ~2y @ 6h
    x = (h["Time"] - h["Time"].min()).dt.total_seconds() / (365.25 * 24 * 3600.0)
    y = h["Water_Level"].to_numpy(dtype=float)
    lo, hi = np.nanpercentile(y, [2, 98])
    yc = np.clip(y, lo, hi)
    try:
        slope, _intercept = np.polyfit(x, yc, 1)
    except Exception:                                            # noqa: BLE001
        slope = 2.0
    resid = yc - slope * x
    amp = float(np.nanpercentile(np.abs(resid), 90))
    amp = min(max(amp, 0.5), 6.0)
    trend = min(max(float(slope), -3.0), 12.0)
    return trend, amp


def _simulate_series(anchor_time, anchor_level, trend_m_yr, amp, num_days, rng):
    """6-hourly depth-to-water for the next `num_days`, continuous at the anchor.

    depth(t) = anchor_level
               + trend_m_yr * days / 365.25                 # extraction drawdown
               - amp * (cos(phase(t)) - cos(phase0))        # monsoon recharge
               + sensor noise

    Subtracting cos(phase0) removes the seasonal offset at the append boundary,
    so the very first simulated reading sits exactly on the station's last real
    reading (modulo noise).
    """
    steps = int(num_days * 24 / INTERVAL_HOURS) + 1
    times = pd.date_range(anchor_time, periods=steps, freq=f"{INTERVAL_HOURS}h")
    times = times[times > anchor_time]

    days = (times - anchor_time).total_seconds() / (24 * 3600.0)
    month = times.month.to_numpy(dtype=float)
    phase = 2.0 * np.pi * (month - 9.0) / 12.0
    phase0 = 2.0 * np.pi * (anchor_time.month - 9.0) / 12.0

    depth = (
        anchor_level
        + trend_m_yr * days / 365.25
        - amp * (np.cos(phase) - np.cos(phase0))
        + rng.normal(0.0, 0.20, size=len(times))
    )
    depth = np.round(np.clip(depth, 0.5, 150.0), 2)
    return pd.DataFrame({"Time": times, "Water_Level": depth})


def simulate_live_telemetry_stream(station_ids=None, num_days=30,
                                   telemetry_path=TELEMETRY_CSV,
                                   stations_path=STATIONS_CSV,
                                   run_builder1=True, seed=None):
    """Append N days of realistic 6-hourly depth-to-water readings.

    `station_ids` defaults to every station in stations.csv. Readings are
    appended per station at its own last-read +6h (no timestamp collisions),
    then Builder 1 is re-run so processed_math_data.csv reflects the feed.

    Returns a summary dict:
        {"rows_appended", "stations", "seeded", "window_end", "builder1_ok"}
    """
    if num_days < 1:
        raise ValueError("num_days must be >= 1")

    stations = pd.read_csv(stations_path)
    stations["Station_ID"] = stations["Station_ID"].astype(str).str.strip()

    if station_ids is None:
        station_ids = stations["Station_ID"].tolist()
    station_ids = [str(s).strip() for s in station_ids]

    known = set(stations["Station_ID"])
    unknown = [s for s in station_ids if s not in known]
    if unknown:
        print(f"[simulate] Ignoring unknown Station_ID(s): {unknown}")
        station_ids = [s for s in station_ids if s in known]
    if not station_ids:
        raise ValueError("No valid Station_IDs to simulate (check stations.csv).")

    hist = _station_history(telemetry_path)
    rng = np.random.default_rng(seed)
    new_frames = []
    seeded_any = False

    for sid in station_ids:
        sh = hist[hist["Station_ID"] == sid] if hist is not None else None
        if sh is not None and not sh.empty:
            sh = sh.sort_values("Time")
            anchor_time = sh["Time"].max()
            anchor_level = float(sh["Water_Level"].iloc[-1])
            trend, amp = _estimate_station_params(sh)
        else:
            # A station with no history: seed a plausible short baseline so
            # Builder 1 sees a trend instead of a single isolated reading.
            seeded_any = True
            anchor_time = (pd.Timestamp.now().normalize()
                           - pd.Timedelta(days=num_days))
            anchor_level = float(rng.uniform(15.0, 35.0))
            trend, amp = float(rng.uniform(0.8, 6.0)), float(rng.uniform(1.5, 4.0))

        series = _simulate_series(anchor_time, anchor_level, trend, amp,
                                  num_days, rng)
        series["Station_ID"] = sid
        new_frames.append(series)

    new_data = pd.concat(new_frames, ignore_index=True)
    new_data = (new_data.sort_values(["Time", "Station_ID"])
                        .reset_index(drop=True))

    if hist is None:
        new_data.to_csv(telemetry_path, index=False)
    else:
        combined = pd.concat([hist, new_data], ignore_index=True)
        combined = (combined.drop_duplicates(subset=["Time", "Station_ID"])
                            .sort_values(["Time", "Station_ID"])
                            .reset_index(drop=True))
        combined.to_csv(telemetry_path, index=False)

    builder1_ok = None
    if run_builder1:
        builder1_ok = _run_builder1()

    print(f"[simulate] Appended {len(new_data):,} readings for "
          f"{len(station_ids)} station(s) through {new_data['Time'].max():%Y-%m-%d %H:%M}")
    print(f"[simulate] Builder 1: {'OK' if builder1_ok else 'FAILED — see stderr above'}"
          if builder1_ok is not None else "")

    return {
        "rows_appended": len(new_data),
        "stations": len(station_ids),
        "seeded": seeded_any,
        "window_end": new_data["Time"].max(),
        "builder1_ok": builder1_ok,
    }


def run_live_update(num_days=30, seed=None):
    """Entry point for the dashboard's 'Fetch Live Telemetry Update' button."""
    return simulate_live_telemetry_stream(num_days=num_days,
                                          run_builder1=True, seed=seed)


# ===========================================================================
# 3. BUILDER 1 EXECUTION
# ===========================================================================
def _run_builder1(script=BUILDER1_SCRIPT, interval_hours=None):
    """Run builder1_ingest_math.py as a subprocess; return True on rc == 0.

    A subprocess keeps data_fetcher fully standalone (no Builder imports
    another Builder's code) and matches the file-based architecture.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, script]
    env = None
    if interval_hours:
        env = dict(os.environ, AQUA_INTERVAL_HOURS=str(interval_hours))

    proc = subprocess.run(cmd, cwd=cwd, env=env,
                          capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return False
    return True


# ===========================================================================
# 4. CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="AquaSentinel data fetcher / live-telemetry simulator")
    parser.add_argument("--fetch", metavar="PATH_OR_URL",
                        help="parse an official India-WRIS / Data.gov.in CSV "
                             "(local path or https:// URL) into Builder 1's inputs")
    parser.add_argument("--simulate", action="store_true",
                        help="append simulated live telemetry for the next N days")
    parser.add_argument("--days", type=int, default=30,
                        help="number of days to simulate (default 30)")
    parser.add_argument("--stations", nargs="*", default=None,
                        help="restrict simulation to these Station_IDs "
                             "(default: all stations in stations.csv)")
    parser.add_argument("--no-builder1", action="store_true",
                        help="do not re-run Builder 1 afterwards")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducible simulation")
    args = parser.parse_args()

    if not args.fetch and not args.simulate:
        parser.error("Pass --fetch <path-or-url> and/or --simulate")

    run_b1 = not args.no_builder1
    if args.fetch:
        fetch_open_government_data(source_path=args.fetch,
                                   run_builder1=run_b1)
    if args.simulate:
        simulate_live_telemetry_stream(station_ids=args.stations,
                                       num_days=args.days,
                                       run_builder1=run_b1,
                                       seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
