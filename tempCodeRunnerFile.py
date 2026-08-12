"""
AquaSentinel — Builder 3: Frontend Dashboard (Streamlit + Folium + Plotly)
=========================================================================

Reads ONLY the two CSVs produced by the other Builders:
    processed_math_data.csv   (Builder 1)
    ml_forecast_results.csv   (Builder 2)
It never touches a database and never imports another Builder's code.

Navigation is a 4-tab streamlit-option-menu:
    1. "Live Map"         — KPI row, folium station map, clicked-station detail
    2. "Time Series"      — per-station history + spliced forecast
    3. "Status Table"     — latest per-station status with HTML badges
    4. "Live vs. Audit"   — official GEC-2015 static categories/limits vs live
                            telemetry estimates, plus the "Fetch Live Telemetry
                            Update" button that drives data_fetcher.py and
                            reloads the dashboard without a server restart.

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
  * The "Live vs. Audit" tab fetches via data_fetcher.run_live_update(),
    then st.cache_data.clear() + st.rerun() pick up the fresh
    processed_math_data.csv — no server restart needed.

Run:  streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import folium
import plotly.graph_objects as go
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu

# data_fetcher is the optional ingestion layer behind the "Live vs. Audit"
# fetch button. If it is ever missing, the dashboard stays up and simply
# disables that button instead of crashing.
try:
    import data_fetcher
    DATA_FETCHER_AVAILABLE = True
except Exception:                                            # noqa: BLE001
    DATA_FETCHER_AVAILABLE = False

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

CATEGORY_ORDER = ["Safe", "Semi-Critical", "Critical",
                  "Over-Exploited", "Insufficient history"]

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
    st.caption("Data source: synthetic demo telemetry — the 'Live vs. Audit' "
               "tab can append fresh simulated telemetry via data_fetcher.py.")

# Filter the snapshot with the sidebar selections — the sidebar list and the
# map keep using the SAME `snapshot` variable, just filtered in one place.
mask = snapshot["Block_ID"].isin(selected_block) & \
       snapshot["Estimated_Category"].isin(selected_status)
filtered = snapshot[mask]

# ---------------------------------------------------------------------------
# Navigation — 4 tabs (existing content is preserved under tabs 1-3)
# ---------------------------------------------------------------------------
NAV_STYLES = {
    "container": {"padding": "0!important", "background-color": "#111a2e",
                  "border-radius": "12px"},
    "icon": {"color": "#22d3ee", "font-size": "16px"},
    "nav-link": {"font-size": "15px", "color": "#e2e8f0",
                 "text-align": "center", "margin": "0px",
                 "--hover-color": "#1e293b"},
    "nav-link-selected": {"background-color": "#22d3ee", "color": "#0a0f1e"},
}

nav = option_menu(
    menu_title=None,
    options=["Live Map", "Time Series", "Status Table", "Live vs. Audit"],
    icons=["broadcast", "graph-up-arrow", "list-check", "balance-scale"],
    default_index=0,
    orientation="horizontal",
    styles=NAV_STYLES,
    key="aqua_nav",
)

# ===========================================================================
# TAB 1 — LIVE MAP
# ===========================================================================
if nav == "Live Map":
    # ------------------------------------------------------------------
    # KPI row (from the shared snapshot)
    # ------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stations monitored", len(snapshot))
    c2.metric("Active alerts", int(active_alerts.shape[0]))
    c3.metric(
        "Over-Exploited",
        int((snapshot["Estimated_Category"] == "Over-Exploited").sum()))
    latest_proxy = snapshot["Estimated_SoE_Proxy_Pct"].dropna()
    c4.metric("Mean latest proxy %",
              f"{latest_proxy.mean():.1f}" if len(latest_proxy) else "—")

    # ------------------------------------------------------------------
    # Map — real Latitude/Longitude only; visible warning if they are absent
    # ------------------------------------------------------------------
    st.markdown("## 🗺 Live station map")
    has_coords = ("Latitude" in snapshot.columns
                  and "Longitude" in snapshot.columns
                  and snapshot["Latitude"].notna().any()
                  and snapshot["Longitude"].notna().any())

    if filtered.empty:
        st.warning("No stations match the current sidebar filters.")
    elif not has_coords:
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

        # MUST pass returned_objects and use_container_width (see docstring).
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

# ===========================================================================
# TAB 2 — TIME SERIES
# ===========================================================================
elif nav == "Time Series":
    # ------------------------------------------------------------------
    # Station time series — history + spliced forecast (no visual gap)
    # ------------------------------------------------------------------
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

# ===========================================================================
# TAB 3 — STATUS TABLE
# ===========================================================================
elif nav == "Status Table":
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

# ===========================================================================
# TAB 4 — LIVE VS. AUDIT
# ===========================================================================
else:
    st.markdown("## ⚖️ Live vs. Audit")
    st.caption(
        "Compares the **static GEC-2015 audit** (block-level official category "
        "and annual availability, refreshed every few years) with the **live "
        "DWLR telemetry** that AquaSentinel re-estimates every 6 hours. Where "
        "the two diverge, the station is flagged as drifting.")

    st.markdown(f"**Last telemetry reading:** `{df['Time'].max():%Y-%m-%d %H:%M}` — "
                f"{max((pd.Timestamp.now() - df['Time'].max()).days, 0)} days ago")

    # ------------------------------------------------------------------
    # Fetch Live Telemetry Update button
    # ------------------------------------------------------------------
    btn_col, info_col = st.columns([1.4, 3])
    with btn_col:
        fetch_clicked = st.button(
            "📡 Fetch Live Telemetry Update", type="primary",
            use_container_width=True, disabled=not DATA_FETCHER_AVAILABLE)
    with info_col:
        if DATA_FETCHER_AVAILABLE:
            st.caption(
                "Appends fresh 6-hourly DWLR readings (next 30 days) and re-runs "
                "Builder 1 so the early-warning math reflects the live feed, then "
                "reloads this dashboard — no server restart. Demo mode simulates "
                "the feed; real India-WRIS / Data.gov.in exports flow through the "
                "same `data_fetcher.py` pipeline.")
        else:
            st.warning("`data_fetcher.py` is missing, so live updates are disabled.")

    if fetch_clicked:
        with st.spinner("Fetching live telemetry & recomputing early-warning math …"):
            try:
                res = data_fetcher.run_live_update(num_days=30)
                state = "OK" if res.get("builder1_ok") else "FAILED"
                extra = " (seeded baseline for new stations)" if res.get("seeded") else ""
                st.session_state["aqua_update_msg"] = (
                    f"✅ Live telemetry appended: {res['rows_appended']:,} new "
                    f"readings across {res['stations']} station(s) · Builder 1 "
                    f"{state}{extra}. Reloading dashboard …")
            except Exception as exc:     # noqa: BLE001  surface the failure in-UI
                st.session_state["aqua_update_msg"] = f"⚠️ Update failed: {exc}"
        st.cache_data.clear()
        st.rerun()
    if st.session_state.get("aqua_update_msg"):
        st.success(st.session_state["aqua_update_msg"])

    # ------------------------------------------------------------------
    # Side-by-side metrics: official GEC-2015 vs live telemetry
    # ------------------------------------------------------------------
    official_counts = snapshot["Official_Category"].value_counts()
    live_counts = snapshot["Estimated_Category"].value_counts()
    live_oe = int(live_counts.get("Over-Exploited", 0))
    official_oe = int(official_counts.get("Over-Exploited", 0))
    drift_n = int(snapshot["Drift_Flag"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stations compared", len(snapshot))
    k2.metric("Official Over-Exploited (GEC-2015)", official_oe)
    k3.metric("Live Over-Exploited (telemetry)", live_oe,
              delta=live_oe - official_oe, delta_color="inverse")
    k4.metric("Drift stations", drift_n)

    c_off, c_live = st.columns(2)
    with c_off:
        st.markdown("### 📋 Official GEC-2015 (static audit)")
        for cat in CATEGORY_ORDER:
            n = int(official_counts.get(cat, 0))
            st.markdown(f"{status_badge(cat)} — **{n}** station(s)",
                        unsafe_allow_html=True)
    with c_live:
        st.markdown("### 📡 Live telemetry (estimated)")
        for cat in CATEGORY_ORDER:
            n = int(live_counts.get(cat, 0))
            st.markdown(f"{status_badge(cat)} — **{n}** station(s)",
                        unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Per-station category drift table
    # ------------------------------------------------------------------
    st.markdown("### 🎯 Per-station category drift")
    drift_rows = []
    for row in snapshot.sort_values(["Block_ID", "Station_ID"]).itertuples():
        is_drift = bool(row.Drift_Flag)
        marker = "🟠 drifts" if is_drift else "· same"
        drift_rows.append(
            f"<tr>"
            f"<td><b>{row.Station_ID}</b></td><td>{row.Block_ID}</td>"
            f"<td>{status_badge(row.Official_Category)}</td>"
            f"<td>{status_badge(row.Estimated_Category)}</td>"
            f"<td style='text-align:center'>{marker}</td>"
            f"<td>{row.Estimated_SoE_Proxy_Pct:.1f}%</td>"
            f"<td>{row.Confidence}</td>"
            f"</tr>")
    st.markdown(
        f"""
        <table style="width:100%;border-collapse:collapse;font-size:0.9rem">
          <thead>
            <tr style="border-bottom:1px solid #333;text-align:left">
              <th>Station</th><th>Block</th><th>Official GEC-2015</th>
              <th>Live estimated</th><th>Drift</th><th>Live proxy</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>{''.join(drift_rows)}</tbody>
        </table>
        """,
        unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Official availability limit vs live extraction proxy (Plotly)
    # ------------------------------------------------------------------
    st.markdown("### 📊 Official availability limit vs. live extraction proxy")
    chart_df = (snapshot[snapshot["Estimated_SoE_Proxy_Pct"].notna()]
                .sort_values(["Block_ID", "Station_ID"]).copy())
    if chart_df.empty:
        st.info("No live extraction-proxy values yet — run Builder 1.")
    else:
        proxy_max = float(chart_df["Estimated_SoE_Proxy_Pct"].max())
        av_max = float(chart_df["Net_Availability"].max())

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_df["Station_ID"],
            y=chart_df["Estimated_SoE_Proxy_Pct"],
            name="Live 12-mo extraction proxy (%)",
            marker_color=[status_color(c) for c in chart_df["Estimated_Category"]],
            text=[f"{v:.0f}%" for v in chart_df["Estimated_SoE_Proxy_Pct"]],
            textposition="outside",
            customdata=chart_df["Block_ID"],
            hovertemplate="<b>%{x}</b> · Block %{customdata}<br>"
                          "live proxy %{y:.1f} %<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=chart_df["Station_ID"],
            y=chart_df["Net_Availability"],
            name="Official annual availability limit",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#22d3ee", dash="dot", width=2),
            marker=dict(color="#22d3ee", size=9, symbol="diamond"),
            hovertemplate="<b>%{x}</b><br>official availability "
                          "%{y:.1f} units<extra></extra>"))
        fig.add_hline(
            y=100, line_color="#e74c3c", line_dash="dash",
            annotation_text="proxy = official limit (100%)",
            annotation_position="bottom right",
            annotation_font_color="#e74c3c")
        fig.update_layout(
            height=480, margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Station",
            yaxis=dict(title="Extraction proxy (%)",
                       range=[0, max(160.0, proxy_max * 1.15)]),
            yaxis2=dict(title="Availability (units)", overlaying="y",
                        side="right", range=[0, max(12.0, av_max * 1.2)],
                        showgrid=False),
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            font=dict(color="#e2e8f0"))
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Block-level audit summary
    # ------------------------------------------------------------------
    st.markdown("### 🗂 Block-level audit summary")
    bsum = (snapshot.groupby("Block_ID", as_index=False)
            .agg(Stations=("Station_ID", "count"),
                 Mean_Live_Proxy_Pct=("Estimated_SoE_Proxy_Pct", "mean"),
                 Official_Availability=("Net_Availability", "first"),
                 Official_Category=("Official_Category", "first"),
                 Drift_Stations=("Drift_Flag", "sum"))
            .sort_values("Mean_Live_Proxy_Pct", ascending=False))
    b_rows = []
    for row in bsum.itertuples():
        p = row.Mean_Live_Proxy_Pct
        over = "🔴 over limit" if pd.notna(p) and p > 100 else "· within"
        b_rows.append(
            f"<tr><td><b>{row.Block_ID}</b></td><td>{int(row.Stations)}</td>"
            f"<td>{status_badge(row.Official_Category)}</td>"
            f"<td>{p:.1f}%</td><td>{row.Official_Availability:.1f}</td>"
            f"<td>{int(row.Drift_Stations)}</td><td>{over}</td></tr>")
    st.markdown(
        f"""
        <table style="width:100%;border-collapse:collapse;font-size:0.9rem">
          <thead>
            <tr style="border-bottom:1px solid #333;text-align:left">
              <th>Block</th><th>Stations</th><th>Official category</th>
              <th>Mean live proxy</th><th>Availability</th>
              <th>Drift stations</th><th>Proxy vs limit</th>
            </tr>
          </thead>
          <tbody>{''.join(b_rows)}</tbody>
        </table>
        """,
        unsafe_allow_html=True)
