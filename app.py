"""
AquaSentinel — Builder 3: Frontend Dashboard (Streamlit + Folium + Plotly)
=========================================================================

Reads ONLY the two CSVs produced by the other Builders:
    processed_math_data.csv   (Builder 1)
    ml_forecast_results.csv   (Builder 2)
It never touches a database and never imports another Builder's code.

Key implementation notes (each was a real bug in an earlier draft):
  * ONE shared "latest status" snapshot is computed a single time and is used
    by BOTH the sidebar alert list AND the map marker colours. Two different
    queries caused a station that recovered from a past alert to stay red in
    the sidebar forever while the map showed it blue.
  * Map markers use the REAL Latitude/Longitude that Builder 1 carried through.
    If they are genuinely absent we show a visible on-screen warning instead
    of silently plotting fake positions.
  * Severity badges use an explicit status->colour map rendered as injected
    HTML/CSS — st.metric's delta colour only reacts to numeric signs, so it
    can't distinguish category strings.
  * The historical and forecast lines are spliced with no visual gap by
    prepending the final historical point ("Time Zero") to the forecast series.
  * st_folium MUST receive returned_objects=["last_object_clicked_tooltip"]
    (omitting it causes an infinite rerender loop) and use_container_width=True
    so the map respects the column width.

Run:  streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import folium
import plotly.graph_objects as go
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Explicit status -> colour mapping (rendered as HTML/CSS badges)
# ---------------------------------------------------------------------------
STATUS_COLORS = {
    "Safe": "#2ecc71",               # green
    "Semi-Critical": "#f1c40f",      # amber
    "Critical": "#f39c12",           # orange
    "Over-Exploited": "#e74c3c",     # red
    "Insufficient history": "#95a5a6",  # gray
}
NO_DATA_COLOR = "#7f8c8d"

PAGE_TITLE = "AquaSentinel — Groundwater Early-Warning Dashboard"
st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon="💧")


def status_color(category):
    """Colour for a category, with a safe fallback."""
    return STATUS_COLORS.get(category, NO_DATA_COLOR)


def status_badge(category):
    """A small pill badge rendered via inline HTML/CSS.

    We deliberately do NOT use st.metric(delta_color=...) — it only colours by
    the numeric sign of `delta`, so every category string would look identical.
    """
    color = status_color(category)
    return (
        f'<span style="background-color:{color};color:#0a0f1e;padding:2px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.85rem;'
        f'display:inline-block">{category}</span>'
    )


# ---------------------------------------------------------------------------
# Data loading (cached so every Streamlit rerun does not re-read the CSVs)
# ---------------------------------------------------------------------------
@st.cache_data
def load_math_data():
    return pd.read_csv("processed_math_data.csv", parse_dates=["Time"])


@st.cache_data
def load_forecast_data():
    return pd.read_csv("ml_forecast_results.csv")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"# 💧 AquaSentinel")
st.caption(
    "Continuous groundwater early-warning — replaces the retrospective "
    "GEC-2015/IN-GRES audit cycle with live DWLR telemetry-driven alerts "
    "(Smart India Hackathon 2026 · PS 25068 · Ministry of Jal Shakti)."
)

# ---------------------------------------------------------------------------
# Load inputs; stop loudly if Builder 1's output is missing.
# ---------------------------------------------------------------------------
try:
    df = load_math_data()
except FileNotFoundError:
    st.error("`processed_math_data.csv` not found. Run Builder 1 first:\n\n"
             "    python make_sample_data.py\n    python builder1_ingest_math.py")
    st.stop()

try:
    forecast = load_forecast_data()
except FileNotFoundError:
    forecast = pd.DataFrame(columns=[
        "Station_ID", "Forecasted_SoE_Proxy_Pct", "Forecast_Horizon_Months"])
    st.warning("`ml_forecast_results.csv` not found — forecast lines are hidden. "
               "Run Builder 2 first:  python builder2_ml_pipeline.py")

# ===========================================================================
# THE ONE shared "latest status" snapshot.
# Both the sidebar alert list AND the map markers MUST read from this single
# variable — never recompute it as two separate queries against full history.
# ===========================================================================
snapshot = (
    df.sort_values(["Station_ID", "Time"])
      .groupby("Station_ID", as_index=False)
      .tail(1)
)

# ---------------------------------------------------------------------------
# Sidebar — summary + active alerts (built from `snapshot` only)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚨 Active alerts")
    active_alerts = snapshot[snapshot["Alert_Active"] == True]   # noqa: E712
    if active_alerts.empty:
        st.success("No stations are above the hysteresis alert threshold "
                   "(>72% proxy) right now.")
    else:
        st.markdown(
            f"**{len(active_alerts)} station(s)** are above the alert "
            f"threshold:")
        for row in active_alerts.sort_values(
                "Estimated_SoE_Proxy_Pct", ascending=False).itertuples():
            st.markdown(
                f"{status_badge(row.Estimated_Category)} "
                f"**{row.Station_ID}** — {row.Estimated_SoE_Proxy_Pct:.1f}%",
                unsafe_allow_html=True)

    st.markdown("## 🔁 Drift stations")
    drift = snapshot[snapshot["Drift_Flag"] == True]             # noqa: E712
    if drift.empty:
        st.caption("No station differs from its official GEC category.")
    else:
        for row in drift.itertuples():
            st.markdown(
                f"**{row.Station_ID}** · official "
                f"{status_badge(row.Official_Category)} → estimated "
                f"{status_badge(row.Estimated_Category)}",
                unsafe_allow_html=True)

    st.markdown("## 🗂 Filters")
    blocks = snapshot["Block_ID"].dropna().unique()
    selected_block = st.multiselect("Block", sorted(blocks), default=list(blocks))
    selected_status = st.multiselect(
        "Status", list(STATUS_COLORS.keys()),
        default=list(STATUS_COLORS.keys()))

    st.markdown("---")
    st.caption("Data source: synthetic demo telemetry (see README).")

# Filter the snapshot with the sidebar selections — the sidebar list and the
# map keep using the SAME `snapshot` variable, just filtered in one place.
mask = snapshot["Block_ID"].isin(selected_block) & \
       snapshot["Estimated_Category"].isin(selected_status)
filtered = snapshot[mask]

# ---------------------------------------------------------------------------
# KPI row (from the shared snapshot)
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stations monitored", len(snapshot))
c2.metric("Active alerts", int(active_alerts.shape[0]))
c3.metric(
    "Over-Exploited",
    int((snapshot["Estimated_Category"] == "Over-Exploited").sum()))
latest_proxy = snapshot["Estimated_SoE_Proxy_Pct"].dropna()
c4.metric("Mean latest proxy %",
          f"{latest_proxy.mean():.1f}" if len(latest_proxy) else "—")

# ---------------------------------------------------------------------------
# Map — real Latitude/Longitude only; visible warning if they are absent
# ---------------------------------------------------------------------------
st.markdown("## 🗺 Live station map")
has_coords = ("Latitude" in snapshot.columns
              and "Longitude" in snapshot.columns
              and snapshot["Latitude"].notna().any()
              and snapshot["Longitude"].notna().any())

if not has_coords:
    # The spec is explicit: warn visibly, never fake coordinates.
    st.warning(
        "⚠️ No usable Latitude/Longitude found in the data, so the map is "
        "hidden. Run Builder 1 against a feed whose stations.csv includes "
        "real coordinates.")
else:
    center = [filtered["Latitude"].mean(), filtered["Longitude"].mean()]
    m = folium.Map(location=center, zoom_start=7,
                   tiles="CartoDB dark_matter")

    for row in filtered.itertuples():
        color = status_color(row.Estimated_Category)
        tooltip = row.Station_ID   # unique per station; drives the click panel
        popup = folium.Popup(
            f"<b>{row.Station_ID}</b><br>Block {row.Block_ID}<br>"
            f"{status_badge(row.Estimated_Category)}<br>"
            f"Proxy {row.Estimated_SoE_Proxy_Pct:.1f}%<br>"
            f"Confidence {row.Confidence}<br>"
            f"Alert {'🔴 ON' if row.Alert_Active else '⚪ off'}",
            max_width=260)
        folium.CircleMarker(
            location=[row.Latitude, row.Longitude],
            radius=9,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=1.5,
            tooltip=tooltip,
            popup=popup,
        ).add_to(m)

    # MUST pass returned_objects and use_container_width (see module docstring).
    click = st_folium(m,
                      returned_objects=["last_object_clicked_tooltip"],
                      use_container_width=True,
                      key="aqua_map")

    clicked_id = click.get("last_object_clicked_tooltip")
    if clicked_id:
        detail = snapshot[snapshot["Station_ID"] == clicked_id]
        if not detail.empty:
            d = detail.iloc[0]
            st.markdown(
                f"### 📍 {d.Station_ID} — Block {d.Block_ID}"
                f" {status_badge(d.Estimated_Category)}",
                unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Latest depth (m)", f"{d.Water_Level:.2f}")
            k2.metric("Latest proxy %",
                      f"{d.Estimated_SoE_Proxy_Pct:.2f}")
            k3.metric("12-mo decline (m)",
                      f"{d.Depth_Decline_Proxy:.2f}")
            k4.metric("Official category", d.Official_Category)

# ---------------------------------------------------------------------------
# Station time series — history + spliced forecast (no visual gap)
# ---------------------------------------------------------------------------
st.markdown("## 📈 Station time series")
stations_sorted = sorted(snapshot["Station_ID"].unique())
station_id = st.selectbox("Select a station", stations_sorted,
                          index=0 if stations_sorted else None)

hist = (df[df["Station_ID"] == station_id]
        .dropna(subset=["Estimated_SoE_Proxy_Pct"])
        .sort_values("Time"))
fc_row = forecast[forecast["Station_ID"] == station_id]

fig = go.Figure()
if not hist.empty:
    fig.add_trace(go.Scatter(
        x=hist["Time"], y=hist["Estimated_SoE_Proxy_Pct"],
        mode="lines", name="History",
        line=dict(color="#22d3ee", width=2)))

    if not fc_row.empty:
        # Splice: prepend the final historical point ("Time Zero") to the
        # forecast so Plotly draws ONE continuous line, not two segments.
        horizon = int(fc_row["Forecast_Horizon_Months"].iloc[0])
        last_point = hist.iloc[-1]
        time_zero = last_point["Time"]
        forecast_time = time_zero + pd.DateOffset(months=horizon)
        forecast_value = fc_row["Forecasted_SoE_Proxy_Pct"].iloc[0]

        fig.add_trace(go.Scatter(
            x=[time_zero, forecast_time],
            y=[last_point["Estimated_SoE_Proxy_Pct"], forecast_value],
            mode="lines+markers", name=f"{horizon}-month forecast",
            line=dict(color="#f59e0b", dash="dash", width=2.5),
            marker=dict(size=8)))
    else:
        st.caption("No forecast row for this station (insufficient history).")

    # Reference thresholds for the early-warning story.
    for level, color, label in [(70, "#2ecc71", "Safe ≤ 70"),
                                (90, "#f1c40f", "Semi-Critical ≤ 90"),
                                (100, "#f39c12", "Critical ≤ 100"),
                                (72, "#e74c3c", "Hysteresis ON > 72"),
                                (68, "#7f8c8d", "Hysteresis OFF < 68")]:
        fig.add_hline(y=level, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_position="bottom left",
                      annotation_font_color=color)

    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        title=f"{station_id} — extraction proxy % over time",
        xaxis_title="Time", yaxis_title="Estimated_SoE_Proxy_Pct (%)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#e2e8f0"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No proxy history available for this station.")

# ---------------------------------------------------------------------------
# Station status table (rendered with real HTML badges)
# ---------------------------------------------------------------------------
st.markdown("## 📋 Latest station status")
table_rows = []
for row in snapshot.sort_values("Station_ID").itertuples():
    table_rows.append(
        f"<tr>"
        f"<td><b>{row.Station_ID}</b></td>"
        f"<td>{row.Block_ID}</td>"
        f"<td>{status_badge(row.Estimated_Category)}</td>"
        f"<td>{row.Confidence}</td>"
        f"<td>{row.Estimated_SoE_Proxy_Pct:.1f}%</td>"
        f"<td>{'🔴' if row.Alert_Active else '⚪'}</td>"
        f"<td>{'↗' if row.Drift_Flag else '·'}</td>"
        f"</tr>")
st.markdown(
    f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem">
      <thead>
        <tr style="border-bottom:1px solid #333;text-align:left">
          <th>Station</th><th>Block</th><th>Status</th>
          <th>Confidence</th><th>Proxy %</th><th>Alert</th><th>Drift</th>
        </tr>
      </thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
    """,
    unsafe_allow_html=True)
