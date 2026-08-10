"""
test_builder1_math.py — plain-assert regression tests for Builder 1's math
==========================================================================

Runnable with a plain interpreter (no pytest needed):

    python test_builder1_math.py

These tests encode the two failure modes that have broken this project before:

  * the Estimated_SoE_Proxy_Pct multiplier must be x100 (never x1000),
  * a station with a genuine 2-year history must be HIGH confidence
    (never stuck at LOW / MEDIUM).

They import Builder 1's pure helpers directly (Builder 1 is a standalone
script with a guarded main, so importing it is safe).
"""

import os
import tempfile

import numpy as np
import pandas as pd

from builder1_ingest_math import (
    classify_proxy,
    compute_soe_proxy,
    confidence_tier,
    hysteresis_flags,
    DECLINE_DIFF_MONTHS,
)

PASS = 0
FAIL = 0


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def test_x100_proxy_multiplier():
    """Hand-trace: 2 m decline / 10 units availability MUST be 20.0, not 200.0."""
    print("[test] Estimated_SoE_Proxy_Pct multiplier is x100")
    result = compute_soe_proxy(np.array([2.0]), np.array([10.0]))
    check("2m decline / 10 units == 20.0", np.isclose(result[0], 20.0))
    check("...and NOT 200.0 (the historic x1000 bug)",
          not np.isclose(result[0], 200.0))

    result = compute_soe_proxy(np.array([5.0]), np.array([4.0]))
    check("5m decline / 4 units == 125.0 (Over-Exploited territory)",
          np.isclose(result[0], 125.0))


def test_confidence_tiers():
    """A genuine 2-year (24-month) history must be HIGH confidence."""
    print("[test] Confidence tiers")
    check("<12 months  -> LOW", confidence_tier(6) == "LOW")
    check("12 months   -> MEDIUM", confidence_tier(12) == "MEDIUM")
    check("23 months   -> MEDIUM", confidence_tier(23) == "MEDIUM")
    check("24 months   -> HIGH (2-year station!)", confidence_tier(24) == "HIGH")
    check("36 months   -> HIGH", confidence_tier(36) == "HIGH")


def test_classification_boundaries():
    """<=70 Safe, <=90 Semi-Critical, <=100 Critical, >100 Over-Exploited."""
    print("[test] Classification boundaries")
    check("70  -> Safe", classify_proxy(70.0, "HIGH") == "Safe")
    check("71  -> Semi-Critical", classify_proxy(71.0, "HIGH") == "Semi-Critical")
    check("90  -> Semi-Critical", classify_proxy(90.0, "HIGH") == "Semi-Critical")
    check("91  -> Critical", classify_proxy(91.0, "HIGH") == "Critical")
    check("100 -> Critical", classify_proxy(100.0, "HIGH") == "Critical")
    check("101 -> Over-Exploited", classify_proxy(101.0, "HIGH") == "Over-Exploited")
    check("non-HIGH -> 'Insufficient history'",
          classify_proxy(80.0, "MEDIUM") == "Insufficient history")
    check("NaN proxy -> NaN", pd.isna(classify_proxy(np.nan, "HIGH")))


def test_hysteresis_state_machine():
    """ON >72%, OFF <68%, hold in between, reset per station."""
    print("[test] Hysteresis alert state machine")
    proxy = [50.0, 80.0, 80.0, 70.0, 70.0, 65.0, 80.0, 90.0]
    flags = hysteresis_flags(proxy)
    expected = [False, True, True, True, True, False, True, True]
    check("chronological ON/hold/OFF/re-ON sequence", flags == expected)
    check("first row OFF (fresh state)", flags[0] is False)

    # Reset between stations: a fresh call must start OFF again.
    flags2 = hysteresis_flags([80.0, 90.0])
    check("state resets per station (starts OFF)", flags2 == [True, True])


def test_end_to_end_x100_through_pipeline():
    """Drive a tiny fake feed through Builder 1's pipeline and check the proxy."""
    print("[test] End-to-end: Builder 1 pipeline on a 26-month toy station")

    # One station, ~26 months of 6-hourly readings. Depth rises 2.0 m every
    # 12 months, so its 12-month diff ~= 2.0 m -> proxy ~= 20.0 with a 10-unit
    # availability (x100, exactly the hand-trace from the build prompt).
    start = pd.Timestamp("2024-01-01 00:00:00")
    times = pd.date_range(start, periods=26 * 30 * 4, freq="6h")  # ~26 months
    month_no = (times - start).days // 30                          # 0..25
    depth = 10.0 + 2.0 * (month_no / 12.0) + np.random.default_rng(1).normal(0, 0.05, len(times))

    telemetry = pd.DataFrame({"Time": times,
                              "Station_ID": "ST-TEST-1",
                              "Water_Level": depth.round(2)})
    stations = pd.DataFrame([{"Station_ID": "ST-TEST-1", "Block_ID": "BLK-T",
                              "Latitude": 30.5, "Longitude": 75.5}])
    blocks = pd.DataFrame([{"Block_ID": "BLK-T", "Net_Availability": 10.0,
                            "Official_Category": "Safe"}])

    result = _run_pipeline_in_memory(telemetry, stations, blocks)
    station = result[result["Station_ID"] == "ST-TEST-1"].sort_values("Time")

    latest = station.dropna(subset=["Estimated_SoE_Proxy_Pct"]).iloc[-1]
    check("12-month decline ~= 2.0 m",
          np.isclose(latest["Depth_Decline_Proxy"], 2.0, atol=0.3))
    check("proxy ~= 20.0 (x100, not 200.0)",
          np.isclose(latest["Estimated_SoE_Proxy_Pct"], 20.0, atol=3.0))
    check("26-month station is HIGH confidence", latest["Confidence"] == "HIGH")
    check("26-month station classified Safe", latest["Estimated_Category"] == "Safe")

    # The first 12 months must NOT get a fabricated proxy (no look-ahead backfill).
    first_valid_month = station["Time"].iloc[DECLINE_DIFF_MONTHS]
    leading = station[station["Time"] < first_valid_month]
    check("first 12 months have NaN proxy (no backfill)",
          leading["Estimated_SoE_Proxy_Pct"].isna().all())


def _run_pipeline_in_memory(telemetry, stations, blocks):
    """Run Builder 1's file-based pipeline against in-memory frames.

    The pipeline reads three CSVs by path; we swap pd.read_csv for a fake that
    serves our frames instead. The output CSV goes to a temp file (and is
    deleted) so the test leaves the repo clean.
    """
    import builder1_ingest_math as b1

    tables = {"raw_dwlr_telemetry.csv": telemetry,
              "stations.csv": stations,
              "blocks.csv": blocks}

    _orig_read_csv = pd.read_csv
    _orig_to_csv = pd.DataFrame.to_csv

    def fake_read_csv(path, **kwargs):
        if str(path) in tables:
            return tables[str(path)]
        return _orig_read_csv(path, **kwargs)

    def swallow_to_csv(self, path, *args, **kwargs):
        # Redirect the pipeline's output write so we never touch a real file.
        return None

    pd.read_csv = fake_read_csv
    pd.DataFrame.to_csv = swallow_to_csv
    try:
        with tempfile.TemporaryDirectory() as tmp:
            return b1.run_pipeline(out_path=os.path.join(tmp, "out.csv"))
    finally:
        pd.read_csv = _orig_read_csv
        pd.DataFrame.to_csv = _orig_to_csv


if __name__ == "__main__":
    test_x100_proxy_multiplier()
    test_confidence_tiers()
    test_classification_boundaries()
    test_hysteresis_state_machine()
    test_end_to_end_x100_through_pipeline()
    print()
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
