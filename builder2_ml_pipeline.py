"""
AquaSentinel — Builder 2: Predictive ML Pipeline
================================================

Consumes Builder 1's `processed_math_data.csv` and produces a 6-month-ahead
forecast of the extraction proxy (`Estimated_SoE_Proxy_Pct`) per station.

  * This script NEVER overwrites processed_math_data.csv. If that file is
    missing it builds a small SYNTHETIC stand-in IN MEMORY (with a loud
    warning) so the pipeline can still be demonstrated — the fake data is
    never written to disk.
  * Strict CHRONOLOGICAL split (last 6 months = holdout, everything before =
    train). No random splits — that would leak the future into training.
  * Min-Max scaling fits ONLY on the training split, then transforms both.
  * Baselines for comparison: a SARIMA model and a plain (non-graph) LSTM,
    both on one dynamically-chosen target station (never hardcoded).
  * Core model: a GCN-LSTM (torch_geometric_temporal's GConvLSTM) where the
    adjacency graph connects stations that share a Block_ID (intra-block).
    CRITICAL: the model is genuinely recurrent — hidden and cell state (h, c)
    are threaded from each timestep into the next, both inside forward() and
    in the training loop. Search this file for "THREAD" to see it.
  * ~50 epochs, Adam, MSE, chronological train/val with no shuffling.
  * Inference runs recursively over the 6-month holdout, predictions are
    inverse-transformed back to percentage units and clipped at 0 (an
    extraction proxy cannot be negative).

OUTPUT:  ml_forecast_results.csv   (Station_ID, Forecasted_SoE_Proxy_Pct,
                                    Forecast_Horizon_Months)
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    raise SystemExit("scikit-learn is required:  pip install scikit-learn==<pinned>")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    raise SystemExit("statsmodels is required:  pip install statsmodels==<pinned>")

try:
    from torch_geometric_temporal.nn.recurrent import GConvLSTM
except ImportError as exc:
    raise SystemExit(
        "torch_geometric_temporal is required. Install it with:\n"
        "    pip install torch-geometric-temporal\n"
        f"(underlying error: {exc})"
    )

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
DATA_PATH = "processed_math_data.csv"     # Builder 1's output — NEVER overwrite
OUT_PATH = "ml_forecast_results.csv"
HORIZON_MONTHS = 6                        # last 6 months = holdout
VAL_MONTHS = 4                            # chronological validation tail inside train
EPOCHS = 50
LEARNING_RATE = 1e-2
HIDDEN_CHANNELS = 32
CHEBYSHEV_K = 2
RANDOM_SEED = 0


# ===========================================================================
# Data loading (with synthetic fallback, never touching the real file)
# ===========================================================================
def make_synthetic_stand_in():
    """Small in-memory stand-in used ONLY when processed_math_data.csv is gone.

    It is returned as a DataFrame and never written to disk — the real file
    (should one appear) is never overwritten.
    """
    print("=" * 72)
    print("  WARNING: processed_math_data.csv not found.")
    print("  Builder 2 is using a SMALL SYNTHETIC STAND-IN built in memory")
    print("  so the pipeline can still be demonstrated.")
    print("  Nothing is written to disk; a real file is never overwritten.")
    print("=" * 72)

    rng = np.random.default_rng(7)
    rows = []
    station_ids = [f"SYN-{i:02d}" for i in range(8)]
    times = pd.date_range("2024-01-01", periods=24, freq="MS")
    for i, sid in enumerate(station_ids):
        avail = 4.0 + 0.6 * i
        trend = 1.2 + 0.55 * i            # 12-month depth decline proxy (m)
        for t in times:
            proxy = (trend / avail) * 100.0
            rows.append({
                "Time": t,
                "Station_ID": sid,
                "Block_ID": f"B{i % 4}",
                "Latitude": 30.0 + 0.1 * i,
                "Longitude": 75.0 + 0.1 * i,
                "Net_Availability": avail,
                "Official_Category": "Safe",
                "Confidence": "HIGH",
                "Estimated_SoE_Proxy_Pct": proxy,
            })
    return pd.DataFrame(rows)


def load_data():
    """Read Builder 1's output, or fall back to the in-memory stand-in."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, parse_dates=["Time"])
        print(f"[data] Loaded {DATA_PATH}: {len(df):,} rows, "
              f"{df['Station_ID'].nunique()} stations")
        return df
    return make_synthetic_stand_in()


# ===========================================================================
# Time-series prep — strictly chronological, no shuffling anywhere
# ===========================================================================
def chronological_split(df):
    """Split so the LAST `horizon` months are the holdout, all prior = train."""
    df = df.dropna(subset=["Estimated_SoE_Proxy_Pct"]).copy()
    df = df.sort_values("Time").reset_index(drop=True)

    months = df["Time"].drop_duplicates().sort_values()
    cut = months.iloc[-HORIZON_MONTHS]            # first month of the holdout
    train_df = df[df["Time"] < cut]
    holdout_df = df[df["Time"] >= cut]
    print(f"[split] Holdout starts {cut.date()}: {len(train_df):,} train rows, "
          f"{len(holdout_df):,} holdout rows")
    return train_df, holdout_df


def build_station_month_matrix(train_df, holdout_df, scaler):
    """Return X_train [N,T], X_holdout [N,H], station_order, all scaled.

    Interior gaps are forward-filled (uses only the past — never the future).
    Leading NaN months (each station's first 12) are dropped, not invented.
    """
    train_pivot = train_df.pivot_table(
        index="Time", columns="Station_ID", values="scaled").ffill()
    train_pivot = train_pivot.dropna(axis=1, how="all")   # drop all-NaN stations
    station_order = list(train_pivot.columns)
    train_pivot = train_pivot.dropna(axis=0)              # drop leading NaN months

    holdout_pivot = holdout_df.pivot_table(
        index="Time", columns="Station_ID", values="scaled").reindex(
            columns=station_order).ffill().dropna(axis=0)

    X_train = torch.tensor(train_pivot.to_numpy().T, dtype=torch.float32)     # [N,T]
    X_holdout = torch.tensor(holdout_pivot.to_numpy().T, dtype=torch.float32) # [N,H]
    print(f"[matrix] {X_train.shape[0]} stations x {X_train.shape[1]} train "
          f"months, {X_holdout.shape[1]} holdout months")
    return X_train, X_holdout, station_order


# ===========================================================================
# Graph construction — intra-block adjacency only
# ===========================================================================
def build_edge_index(train_df, station_order):
    """Connect every station pair sharing a Block_ID; add self-loops.

    Stations from different blocks are never connected.
    """
    block_of = (train_df.drop_duplicates("Station_ID")
                .set_index("Station_ID")["Block_ID"].to_dict())

    edges = []
    by_block = {}
    for sid in station_order:
        by_block.setdefault(block_of[sid], []).append(sid)

    for sid_a in station_order:
        block = block_of[sid_a]
        for sid_b in by_block[block]:
            edges.append((sid_a, sid_b))     # both directions -> undirected

    order = {sid: i for i, sid in enumerate(station_order)}
    edge_list = [[order[a], order[b]] for a, b in edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()   # [2,E]
    print(f"[graph] {len(edges)} directed edges over {len(station_order)} nodes "
          f"(intra-block + self-loops)")
    return edge_index


# ===========================================================================
# Core model — GCN-LSTM that is genuinely recurrent
# ===========================================================================
class GCNLSTM(nn.Module):
    """One GConvLSTM cell (per-timestep) plus a linear read-out head.

    forward() takes the state (h, c) from the previous timestep and returns
    the NEW state. It is the training loop's job to pass it back in — that is
    exactly what gives the model temporal memory.         <-- THREAD (model)
    """

    def __init__(self, in_channels=1, hidden_channels=HIDDEN_CHANNELS, K=CHEBYSHEV_K):
        super().__init__()
        self.gconv_lstm = GConvLSTM(in_channels, hidden_channels, K)
        self.readout = nn.Linear(hidden_channels, in_channels)

    def forward(self, x, edge_index, edge_weight, h=None, c=None):
        # x: [N, in_channels] for ONE timestep.
        # h/c from the PREVIOUS timestep are passed in, the new ones come back.
        h, c = self.gconv_lstm(x, edge_index, edge_weight, h, c)  # <-- THREAD
        pred = self.readout(h)                                     # [N, 1] next value
        return h, c, pred


def run_one_epoch(model, X, edge_index, edge_weight, t_start, t_stop, criterion):
    """Teacher-forced pass over time window [t_start, t_stop).

    Iterates the sequence dimension, threading (h, c) from each step into the
    next. Returns (mean_loss, final_h, final_c).
    """
    h = c = None
    total = torch.tensor(0.0)
    for t in range(t_start, t_stop):
        h, c, pred = model(X[:, t:t + 1], edge_index, edge_weight, h, c)
        total = total + criterion(pred, X[:, t + 1:t + 1])
    mean_loss = total / (t_stop - t_start)
    return mean_loss, h, c


def train_gcnlstm(model, X, edge_index, edge_weight):
    """Chronological train/val, no shuffling, ~50 epochs of Adam + MSE.

    The first (T - VAL_MONTHS) months train; the last VAL_MONTHS form a
    chronological validation tail that continues the SAME hidden/cell state.
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    T = X.shape[1]
    t_inner = max(1, T - VAL_MONTHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        train_loss, h, c = run_one_epoch(model, X, edge_index, edge_weight,
                                         0, t_inner - 1, criterion)
        train_loss.backward()
        optimizer.step()

        # Validation: keep going from the inner loop's final state (detached).
        model.eval()
        with torch.no_grad():
            hv, cv = h.detach(), c.detach()
            val_total = torch.tensor(0.0)
            for t in range(t_inner - 1, T - 1):
                hv, cv, pred = model(X[:, t:t + 1], edge_index, edge_weight, hv, cv)
                val_total = val_total + criterion(pred, X[:, t + 1:t + 1])
            val_loss = val_total / max(1, T - t_inner)

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"  epoch {epoch:3d} | train loss {train_loss.item():.6f} "
                  f"| val loss {val_loss.item():.6f}")


def recursive_forecast(model, X_train, X_holdout, edge_index, edge_weight):
    """Run the model over history, then roll the forecast forward.

    The final (h, c) after the last TRAIN month is fed forward, then each
    predicted month is fed back in as the next input — a genuine multi-step
    forecast, no look-ahead. Returns scaled predictions [N, H].
    """
    model.eval()
    T, H = X_train.shape[1], X_holdout.shape[1]
    with torch.no_grad():
        h = c = None
        for t in range(T):                                 # roll over history
            h, c, _ = model(X_train[:, t:t + 1], edge_index, edge_weight, h, c)

        x_prev = X_train[:, -1:]                           # last observed value
        preds = []
        for _ in range(H):                                 # roll the future
            h, c, pred = model(x_prev, edge_index, edge_weight, h, c)
            preds.append(pred)
            x_prev = pred                                  # feed prediction back
        return torch.cat(preds, dim=1)                     # [N, H]


# ===========================================================================
# Baselines (for comparison only) — target station chosen dynamically
# ===========================================================================
def pick_target_station(train_df):
    """Never hardcode a station ID — real data won't contain it."""
    return sorted(train_df["Station_ID"].unique())[0]


def sarima_baseline(train_df, holdout_df, station_id):
    """SARIMA on the target station's scaled train series; RMSE vs holdout."""
    s = train_df[train_df["Station_ID"] == station_id].sort_values("Time")
    ho = holdout_df[holdout_df["Station_ID"] == station_id].sort_values("Time")
    series = s["scaled"].to_numpy()
    actual = ho["scaled"].to_numpy()[:HORIZON_MONTHS]
    try:
        model = SARIMAX(
            series, order=(1, 0, 1), seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=200)
        fc = np.asarray(fitted.forecast(len(actual)))
        return float(np.sqrt(np.mean((fc - actual) ** 2)))
    except Exception as exc:                                # SARIMA can be brittle
        print(f"  [SARIMA] fit failed for {station_id}: {exc}")
        return float("nan")


class PlainLSTM(nn.Module):
    """A plain, non-graph per-station LSTM used as a baseline."""
    def __init__(self, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden)
        self.fc = nn.Linear(hidden, 1)


def _train_single_lstm(series, epochs=80):
    model = PlainLSTM()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    T = len(series)
    for _ in range(epochs):
        optimizer.zero_grad()
        h = c = None
        total = torch.tensor(0.0)
        for t in range(T - 1):
            x = torch.tensor([[series[t]]], dtype=torch.float32)    # [1,1,1]
            if h is None:
                out, (h, c) = model.lstm(x)
            else:
                out, (h, c) = model.lstm(x, (h, c))
            pred = model.fc(out[-1])
            target = torch.tensor([[series[t + 1]]], dtype=torch.float32)
            total = total + criterion(pred, target)
        (total / max(1, T - 1)).backward()
        optimizer.step()
    return model


def _recursive_single_lstm(model, last_value, horizon):
    model.eval()
    h = c = None
    with torch.no_grad():
        x = torch.tensor([[last_value]], dtype=torch.float32)
        preds = []
        for _ in range(horizon):
            if h is None:
                out, (h, c) = model.lstm(x)
            else:
                out, (h, c) = model.lstm(x, (h, c))
            pred = model.fc(out[-1])
            preds.append(pred.item())
            x = pred
    return np.array(preds)


def lstm_baseline(train_df, holdout_df, station_id):
    """Plain (non-graph) LSTM on the target station; RMSE vs holdout."""
    s = train_df[train_df["Station_ID"] == station_id].sort_values("Time")
    ho = holdout_df[holdout_df["Station_ID"] == station_id].sort_values("Time")
    series = s["scaled"].to_numpy().astype(np.float32)
    actual = ho["scaled"].to_numpy()[:HORIZON_MONTHS]
    model = _train_single_lstm(series)
    fc = _recursive_single_lstm(model, float(series[-1]), len(actual))
    return float(np.sqrt(np.mean((fc - actual) ** 2)))


def persistence_baseline(X_train, X_holdout):
    """Naive 'last observed value repeats forever' baseline, for context."""
    flat = np.repeat(X_train[:, -1:].numpy(), X_holdout.shape[1], axis=1)
    return float(np.sqrt(np.mean((flat - X_holdout.numpy()) ** 2)))


# ===========================================================================
# Main
# ===========================================================================
def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df = load_data()
    train_df, holdout_df = chronological_split(df)

    # --- Min-Max scale: fit on TRAIN ONLY, transform both ------------------
    scaler = MinMaxScaler()
    scaler.fit(train_df[["Estimated_SoE_Proxy_Pct"]])
    train_df["scaled"] = scaler.transform(train_df[["Estimated_SoE_Proxy_Pct"]])
    holdout_df["scaled"] = scaler.transform(holdout_df[["Estimated_SoE_Proxy_Pct"]])
    print(f"[scale] MinMax fitted on train only: "
          f"[{scaler.data_min_[0]:.2f}, {scaler.data_max_[0]:.2f}]")

    X_train, X_holdout, station_order = build_station_month_matrix(
        train_df, holdout_df, scaler)
    edge_index = build_edge_index(train_df, station_order)
    edge_weight = None

    if X_train.shape[1] < VAL_MONTHS + 2:
        raise SystemExit("Not enough training months for the GCN-LSTM "
                         "(need more history in processed_math_data.csv).")

    # --- Baselines ---------------------------------------------------------
    target = pick_target_station(train_df)
    print(f"\n[baseline] target station (chosen dynamically): {target}")
    sarima_rmse = sarima_baseline(train_df, holdout_df, target)
    lstm_rmse = lstm_baseline(train_df, holdout_df, target)
    persist_rmse = persistence_baseline(X_train, X_holdout)
    print(f"  [baseline] SARIMA  RMSE (target, scaled) = {sarima_rmse:.4f}")
    print(f"  [baseline] LSTM    RMSE (target, scaled) = {lstm_rmse:.4f}")
    print(f"  [baseline] Persist RMSE (all,  scaled)   = {persist_rmse:.4f}")

    # --- Core model --------------------------------------------------------
    print(f"\n[model] Training GCN-LSTM for {EPOCHS} epochs "
          f"(Adam, MSE, chronological train/val, no shuffle)")
    model = GCNLSTM()
    train_gcnlstm(model, X_train, edge_index, edge_weight)

    preds_scaled = recursive_forecast(model, X_train, X_holdout,
                                      edge_index, edge_weight)
    gcn_rmse = float(torch.sqrt(
        torch.mean((preds_scaled - X_holdout) ** 2)))
    print(f"[eval] GCN-LSTM RMSE (all stations, scaled) = {gcn_rmse:.4f}")

    # --- Forecast output: last forecast month, back in % units, clip >= 0 ---
    last_forecast = preds_scaled[:, -1].numpy().reshape(-1, 1)          # [N,1]
    forecast_pct = scaler.inverse_transform(last_forecast).ravel()
    forecast_pct = np.clip(forecast_pct, a_min=0.0, a_max=None)         # no negatives

    out = pd.DataFrame({
        "Station_ID": station_order,
        "Forecasted_SoE_Proxy_Pct": np.round(forecast_pct, 2),
        "Forecast_Horizon_Months": HORIZON_MONTHS,
    })
    out.to_csv(OUT_PATH, index=False)
    print(f"\n[output] Wrote {OUT_PATH}: {len(out)} stations, "
          f"{HORIZON_MONTHS}-month horizon")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
