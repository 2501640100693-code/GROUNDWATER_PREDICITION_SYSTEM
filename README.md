# AquaSentinel 💧

A **groundwater early-warning dashboard** for Smart India Hackathon 2026,
Problem Statement 25068 (Ministry of Jal Shakti). AquaSentinel replaces India's
retrospective, multi-year GEC-2015 / IN-GRES groundwater auditing cycle with a
**continuous** early-warning system driven by real DWLR (Digital Water Level
Recorder) telemetry.

```
 raw_dwlr_telemetry.csv      stations.csv      blocks.csv
        │                        │                 │
        ▼                        ▼                 ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Builder 1  builder1_ingest_math.py   (validation + math)│
 └─────────────────────────────────────────────────────────┘
        │   processed_math_data.csv
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Builder 2  builder2_ml_pipeline.py   (SARIMA + LSTM +   │
 │               GCN-LSTM 6-month forecast)                 │
 └─────────────────────────────────────────────────────────┘
        │   ml_forecast_results.csv
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Builder 3  app.py   (Streamlit + Folium + Plotly)       │
 └─────────────────────────────────────────────────────────┘
```

The three Builders communicate **only** through static CSV files — no Builder
imports another Builder's code, and there is no database.

---

## Quick start (local demo)

```bash
# 1. Install dependencies (see requirements.txt — every version is pinned)
pip install -r requirements.txt

# 2. Generate synthetic demo data (only needed because we have no real feed)
python make_sample_data.py

# 3. Builder 1 — validate, clean, and compute the early-warning math
python builder1_ingest_math.py          # -> processed_math_data.csv

# 4. Builder 2 — train the GCN-LSTM and forecast 6 months ahead
python builder2_ml_pipeline.py          # -> ml_forecast_results.csv

# 5. Builder 3 — launch the dashboard
streamlit run app.py
```

There is also a self-test for Builder 1's math (runs with plain Python, no
pytest):

```bash
python test_builder1_math.py
```

---

## What each file does

| File | Role |
|------|------|
| `make_sample_data.py` | Fabricates the three input CSVs (Punjab/Haryana, real lat/lon) so the pipeline runs end-to-end without government data. **Demo only.** |
| `builder1_ingest_math.py` | Validation gateway (rejects bad readings, prunes stations missing >30% of expected readings, inner-joins station/block metadata) + math engine (30-day z-score outliers, monthly resampling, confidence tiers, 12-month decline proxy, `×100` SoE proxy, category, drift flag, hysteresis alert). |
| `test_builder1_math.py` | Regression tests that encode the two historic bugs: the `×100` (not `×1000`) proxy multiplier, and a genuine 2-year station being `HIGH` confidence. |
| `builder2_ml_pipeline.py` | Chronological 6-month holdout split, train-only Min-Max scaling, a SARIMA and a plain-LSTM baseline, an intra-block station graph, and a **genuinely recurrent** GCN-LSTM (`torch_geometric_temporal.GConvLSTM`) whose hidden/cell state is threaded across timesteps. |
| `app.py` | Streamlit dashboard: live map, sidebar alerts, per-station history + spliced 6-month forecast. |
| `.streamlit/config.toml` | Dark "emergency monitoring" theme. |
| `requirements.txt` | Pinned dependencies (every line is `package==version`). |
| `packages.txt` | Blank — no system packages needed for pure-Python deployment. |

---

## Builder 1 — how the early-warning math works

`Water_Level` is **depth-to-water in metres below ground** — an *increase*
means the water table is **dropping**.

1. **Validation gateway** drops non-numeric / zero / negative readings, then
   prunes any station missing more than 30% of its *expected* readings. The
   expected count is the full shared analysis window (`max(Time) − min(Time)`
   across **all** stations) divided by `EXPECTED_INTERVAL_HOURS` (default 6,
   set `--interval-hours 1` for hourly feeds). We do **not** infer the interval
   from the data — random gaps just stretch the median gap and hide the loss.
2. **Outliers**: rolling 30-day z-score with `min_periods=3`, dropping
   `abs(z) > 3`. (A 1-point window has undefined std and would wrongly drop
   that row.)
3. **Monthly resampling** → one row per station per calendar month (mean).
4. **Confidence** from total months of history: `<12` LOW · `12–23` MEDIUM ·
   `≥24` HIGH.
5. **`Depth_Decline_Proxy`** = 12-month diff of monthly `Water_Level`. The
   first 12 months of each station are left `NaN` **on purpose** — backfilling
   them (even from the station's own later mean) is look-ahead leakage and
   would render as if it were real telemetry.
6. **`Estimated_SoE_Proxy_Pct = (Depth_Decline_Proxy / Net_Availability) × 100`**
   — the multiplier is exactly **100**. Hand-trace: a 2 m decline against a
   10-unit availability is `20.0`, never `200.0`.
7. **Category**: `≤70` Safe · `≤90` Semi-Critical · `≤100` Critical ·
   `>100` Over-Exploited — only when `Confidence == HIGH`, else
   `"Insufficient history"`.
8. **`Drift_Flag`** = `HIGH` confidence **and** estimated ≠ official category.
9. **Hysteresis alert**: ON above 72%, OFF below 68%, held in between —
   state is **reset per station**.

> **Why the demo shows only `HIGH` confidence:** the 30%-coverage rule prunes
> any station that doesn't cover ≥70% of the *shared* window. In a multi-year
> window that means every survivor has ≥24 months of history, so the
> LOW/MEDIUM/`Insufficient history` paths are implemented and unit-tested but
> simply don't fire in this synthetic dataset. That is the spec-faithful
> outcome, not a bug.

---

## Builder 2 — the predictive pipeline

- **Split**: strictly chronological — the last 6 months are the holdout, all
  earlier data is training. No random splits (that would leak the future).
- **Scaling**: Min-Max, fit on the **training split only**, then transform
  both splits with that same fitted scaler.
- **Graph**: an edge connects every pair of stations that share a `Block_ID`
  (intra-block only), plus self-loops.
- **Core model**: `torch_geometric_temporal.nn.recurrent.GConvLSTM`. The model
  is **genuinely recurrent** — hidden and cell state `(h, c)` are threaded from
  each timestep's output into the next timestep's call, both inside
  `GCNLSTM.forward()` and in `run_one_epoch()`'s sequence loop (look for the
  `# <-- THREAD` comments). Without that threading the model would have no
  temporal memory and there would be no point to GCN-LSTM over a per-station
  LSTM.
- **Training**: ~50 epochs, Adam, MSE, chronological train/val with no
  shuffling.
- **Inference**: roll the model over history, then forecast recursively
  6 months forward (each prediction feeds back as the next input),
  inverse-transform back to percentage units, and clip at 0 (an extraction
  proxy can't be negative).
- **Baselines** (for comparison, on a dynamically-chosen target station):
  a SARIMA model and a plain, non-graph LSTM.

Output schema (one row per station, constant 6-month horizon — a single
point-forecast, intentional for this MVP):

```
Station_ID, Forecasted_SoE_Proxy_Pct, Forecast_Horizon_Months
```

### ⚠️ Deploying Builder 2 to Streamlit Community Cloud

`torch-geometric-temporal` / `torch-geometric` are **heavy** (PyTorch +
compiled graph kernels). The free Streamlit Cloud build can be slow or fail on
these. For a local demo this repo pins them in `requirements.txt`; if you hit a
Cloud build failure you have two options:

1. Trim the heavy ML deps for the hosted app and run Builder 2 locally
   (the dashboard itself only needs Streamlit, Folium, Plotly, pandas, numpy).
2. Use `packages.txt` for any system-level pieces the build reports missing.

The dashboard (`app.py`) is deliberately independent of PyTorch — it reads only
the two CSVs.

---

## Builder 3 — dashboard details

- **One shared "latest status" snapshot**: `df.sort_values(['Station_ID','Time'])
  .groupby('Station_ID').tail(1)` is computed **once** and used by **both** the
  sidebar alert list **and** the map marker colours. Two different queries
  caused a recovered station to stay red in the sidebar while the map showed it
  blue in an earlier draft.
- **Map markers** use the real `Latitude`/`Longitude` carried through from
  Builder 1. If they are genuinely absent, the app shows a visible on-screen
  warning instead of plotting fake positions.
- **Status badges** are injected HTML/CSS with an explicit status→colour map
  (Safe green · Semi-Critical amber · Critical orange · Over-Exploited red ·
  Insufficient history gray) — `st.metric`'s delta colour can't do this.
- **Splice**: the final historical point is prepended as *Time Zero* to the
  forecast series, so Plotly draws one continuous line with no visual gap.
- **`st_folium(...)`** passes `returned_objects=["last_object_clicked_tooltip"]`
  (required to avoid an infinite rerender loop) and `use_container_width=True`.

---

## Project layout (Streamlit Cloud)

Deploy from the repo root — `app.py` and the CSVs live at the root, no
subfolders. `.streamlit/config.toml` supplies the dark theme automatically.
