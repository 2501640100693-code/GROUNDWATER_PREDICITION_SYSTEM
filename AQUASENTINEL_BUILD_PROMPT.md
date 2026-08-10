# AquaSentinel — Full Build Prompt for an Autonomous Coding Agent

You are building **AquaSentinel**, a groundwater early-warning dashboard for Smart India
Hackathon 2026, Problem Statement 25068 (Ministry of Jal Shakti). It replaces India's
retrospective, multi-year GEC-2015/IN-GRES groundwater auditing cycle with a continuous
early-warning system driven by real DWLR (Digital Water Level Recorder) telemetry.

Build three independently-runnable components ("Builders") that communicate **only**
through static CSV files — never let one Builder import another's code. Create every file
for real; don't describe code, write it. Ask me before proceeding if any requirement below
is ambiguous for a decision that would be expensive to reverse (e.g. database choice,
folder layout) — otherwise use your best judgment and keep moving.

---

## Global rules that apply to every Builder

- **No unpinned dependencies.** Every `requirements.txt` line must be `package==X.Y.Z`
  with no missing `==`. Double-check this file by re-reading it after you write it — this
  exact formatting mistake (concatenating name and version with no operator) has broken
  this project's deployment twice before.
- **No stdlib packages in requirements.txt** (no `os`, `datetime`, etc. — these ship with
  Python and listing them breaks the Streamlit Cloud container build).
- Prefer clear, commented code over cleverness. This will be read by a first-time coder.

---

## Builder 1 — Ingestion & Math Engine

**Input:** `raw_dwlr_telemetry.csv` (`Time, Station_ID, Water_Level`, where `Water_Level`
is depth-to-water in metres below ground — an *increase* means the water table is
dropping), `stations.csv` (`Station_ID, Block_ID, Latitude, Longitude`), `blocks.csv`
(`Block_ID, Net_Availability, Official_Category`).

**Validation gateway:**
1. Reject non-numeric, zero, or negative `Water_Level` readings.
2. Exclude any station missing more than 30% of its *expected* readings. Expected count =
   full shared analysis window (`max(Time) - min(Time)` across ALL stations, not each
   station's own span) divided by a declared `EXPECTED_INTERVAL_HOURS` constant (default 6,
   configurable to 1 for hourly feeds). Do not auto-infer the interval from the data's own
   median gap — randomly-missing data defeats that approach (median gap just grows to match
   what's left, hiding the loss).
3. Inner-join `Latitude`/`Longitude`/`Block_ID` from `stations.csv` and `Net_Availability`/
   `Official_Category` from `blocks.csv` onto the surviving readings. These columns must
   survive all the way to the final output CSV — Builder 3's map depends on them.

**Math engine (per station):**
1. Remove sensor outliers: rolling 30-day z-score, **`min_periods=3`** (not 1 — a window of
   1 point has undefined std and will incorrectly drop that row), exclude `abs(z) > 3.0`.
2. Resample to one row per station per calendar month (mean).
3. Confidence tier per station based on total months of history: `< 12` → `LOW`,
   `12–23` → `MEDIUM`, `>= 24` → `HIGH`.
4. `Depth_Decline_Proxy` = 12-month diff of monthly `Water_Level`. **Leave the first 12
   months of each station as `NaN` / excluded — do not backfill them with any computed
   average.** Those months are genuinely unknown; inventing a value for them (even from the
   station's own later mean) is look-ahead leakage and will silently render as if it were
   real telemetry on the dashboard, since the chart has no way to distinguish backfilled
   values from real ones.
5. `Estimated_SoE_Proxy_Pct = (Depth_Decline_Proxy / Net_Availability) * 100`. **This is a
   direct percentage — the multiplier is 100, not 1000 or any other constant.** This one
   line has broken this project before; sanity-check it against a hand-calculated example
   before moving on (e.g. a 2-metre decline against a 10-unit Net_Availability should
   compute to exactly 20.0, not 200.0).
6. Classify: `<=70` Safe, `<=90` Semi-Critical, `<=100` Critical, `>100` Over-Exploited —
   but only when `Confidence == 'HIGH'`; otherwise `"Insufficient history"`.
7. `Drift_Flag` = `True` when `Confidence == 'HIGH'` AND `Estimated_Category !=
   Official_Category`.
8. Hysteresis alert, per station, in chronological order: turn ON when the proxy exceeds
   72%, turn OFF when it drops below 68%, hold state between those two thresholds. Reset
   this state at the start of every station's loop — don't let one station's alert state
   leak into the next station's rows.

**Output:** write `processed_math_data.csv` with every column above included.

---

## Builder 2 — Predictive ML Pipeline

**Input:** `processed_math_data.csv`. **Never overwrite this file.** If it's missing when
this script runs, generate a small synthetic stand-in *under a different filename* or in
memory only, print a loud warning that synthetic data is being used, and never silently
write synthetic data over a real file that happens to already exist.

1. Strict chronological split: last 6 months = holdout, everything before = train. No
   random splits — this is time-series data and a random split would leak the future into
   training.
2. Min-Max scale `Estimated_SoE_Proxy_Pct`, fitting the scaler on the **training split
   only**, then transform both splits with that same fitted scaler.
3. Baselines for comparison: a SARIMA model and a plain (non-graph) LSTM. The SARIMA target
   station must be chosen **dynamically** (e.g. `df['Station_ID'].iloc[0]` after sorting) —
   never hardcode a station ID, since it won't exist in real data.
4. Build an adjacency graph connecting every pair of stations that share the same
   `Block_ID` (intra-block only — do not connect stations across different blocks).
5. Core model: a GCN-LSTM using `torch_geometric_temporal`'s `GConvLSTM`. **This must
   actually be recurrent** — thread the hidden and cell state (`h`, `c`) from each
   timestep's output into the next timestep's call as it iterates over the sequence
   dimension, both in the model's `forward()` and in the training loop that calls it.
   Verify this explicitly: if a training loop calls the model independently at each
   timestep with no `h`/`c` passed in from the previous step, the model has no temporal
   memory regardless of what it's named, and the entire justification for using GCN-LSTM
   over a plain per-station LSTM baseline is void. This is the most important architectural
   requirement in this entire Builder — check it twice.
6. Train ~50 epochs, Adam optimizer, MSE loss, chronological train/val (no shuffling).
7. Inference on the 6-month holdout, inverse-transform predictions back to real percentage
   units, clip at 0 minimum (extraction proxy can't be negative).

**Output:** `ml_forecast_results.csv` with `Station_ID, Forecasted_SoE_Proxy_Pct,
Forecast_Horizon_Months`. Note for later: this MVP schema is one row per station at a
constant 6-month horizon (a single point-forecast, not a monthly trajectory) — that's
intentional for now, not a bug to fix here.

---

## Builder 3 — Frontend Dashboard

**Input:** `processed_math_data.csv` + `ml_forecast_results.csv` only — never touch a
database or import Builder 1/2's code directly. Stack: Streamlit + Folium (via
`streamlit-folium`) + Plotly.

1. **One shared "latest status" snapshot** (e.g. `df.sort_values(['Station_ID','Time'])
   .groupby('Station_ID').tail(1)`), used by **both** the sidebar's active-alerts list
   **and** the map's marker coloring. They must never be computed by two different queries
   — that's what caused a station that recovered from a past alert to stay red in the
   sidebar forever while showing blue on the map in an earlier version of this project.
2. Map markers use the **real** `Latitude`/`Longitude` columns carried through from
   Builder 1 — never simulate/fake coordinates when real ones are available in the data.
   If they're genuinely absent from the CSV, show a visible on-screen warning rather than
   silently plotting fake positions.
3. Severity badges use an explicit status-to-color mapping (Safe → green, Semi-Critical →
   amber, Critical → orange, Over-Exploited → red, Insufficient history → gray) rendered
   via injected HTML/CSS. Do not rely on `st.metric`'s `delta`/`delta_color` for this — it
   only colors by numeric sign, so a category string never triggers it correctly and every
   status ends up looking identical.
4. Splice historical and forecast lines with no visual gap: prepend the final historical
   point (as "Time Zero") to the forecast series before plotting, so Plotly draws one
   continuous line rather than two disconnected segments.
5. `st_folium(...)` **must** pass `returned_objects=["last_object_clicked_tooltip"]` — 
   omitting this causes an infinite rerender loop. Also pass `use_container_width=True` so
   the map respects the Streamlit column width instead of overflowing it.
6. Dark theme via `.streamlit/config.toml` (`base="dark"`, high-contrast colors suitable
   for an emergency-monitoring aesthetic — pick your own palette, doesn't need to match any
   prior draft exactly).

**Deploy target:** Streamlit Community Cloud (free). Repo root, no subfolders for `app.py`
or the CSVs. `packages.txt` can be left blank unless you add a C-extension geospatial
library later.

---

## Before you finish

1. Re-read `requirements.txt` line by line and confirm every line has `package==version`
   with no missing `==`.
2. Hand-trace one row of real math through Builder 1's formula in a comment or test — 
   confirm the `× 100` (not `× 1000`) and that a station with a genuine 2-year history
   ends up `HIGH` confidence, not stuck at `LOW`/`MEDIUM`.
3. Confirm Builder 2's model actually threads `h`/`c` state between timesteps — grep your
   own code for where `h` and `c` are passed into the recurrent layer and verify it's not
   always `None`.
4. Confirm the sidebar alert list and the map markers in Builder 3 both read from the same
   snapshot variable, not two separate queries against the full history.

If I only need a **fast demo** (no real database, no real government data, no heavy
`torch_geometric_temporal` install) — say so before you start Builder 2, and swap it for a
lightweight linear-trend forecast (`numpy.polyfit`) that outputs the identical
`ml_forecast_results.csv` schema, so Builder 3 doesn't need to change at all.
