"""
AquaSentinel — Builder 3: Executive Dashboard (Streamlit + Folium + Plotly)
============================================================================

Reads ONLY the two CSVs produced by the other Builders:
    processed_math_data.csv   (Builder 1)
    ml_forecast_results.csv   (Builder 2)
It never touches a database and never imports another Builder's code.

Navigation — 5 tabs:
    1. Global Command   — KPI row, full-width Folium map, alerts in an expander
    2. Station Analytics— per-station history + spliced GCN-LSTM forecast
    3. Drift Matrix     — searchable/filterable audit table, all stations
    4. Live vs. Audit   — static GEC-2015 audit vs. live telemetry, single-axis
                           toggle between percentage and volumetric views
    5. AI Performance   — reported model evaluation metric + forecast snapshot

SCHEMA NOTE (read before editing): the only columns that exist are the ones
Builder 1/2 actually write:
    processed_math_data.csv -> Time, Station_ID, Water_Level, Latitude,
        Longitude, Block_ID, Net_Availability, Official_Category,
        Depth_Decline_Proxy, Estimated_SoE_Proxy_Pct, Estimated_Category,
        Confidence, Drift_Flag, Alert_Active
    ml_forecast_results.csv -> Station_ID, Forecasted_SoE_Proxy_Pct,
        Forecast_Horizon_Months
There is no State or District column anywhere in the pipeline, and no
holdout/validation column in ml_forecast_results.csv. Do not invent either —
see the sidebar filter and Tab 5 for how that's handled.

Key implementation notes (carried over from the previous draft; unchanged):
  * ONE shared "latest status" snapshot is computed a single time and is used
    by every tab and the map. Two different queries caused a station that
    recovered from a past alert to stay red in one place while showing blue
    in another in an earlier version of this project.
  * Map markers use the REAL Latitude/Longitude that Builder 1 carried
    through. If they are genuinely absent we show a visible on-screen warning
    instead of silently plotting fake positions.
  * st_folium MUST receive returned_objects=["last_object_clicked_tooltip"]
    (omitting it causes an infinite rerender loop) and use_container_width=True.
  * The historical and forecast lines are spliced with no visual gap by
    prepending the final historical point ("Time Zero") to the forecast series.
  * The "Fetch Live Telemetry Update" button calls data_fetcher.run_live_update(),
    then st.cache_data.clear() + st.rerun() pick up the fresh
    processed_math_data.csv — no server restart needed. This logic is
    untouched from the previous draft, only relocated into the sidebar.

Run:  streamlit run app.py
"""

import pandas as pd
import streamlit as st
import folium
import plotly.graph_objects as go
from streamlit_folium import st_folium

# data_fetcher is the optional ingestion layer behind the "Live vs. Audit"
# fetch button. If it is ever missing, the dashboard stays up and simply
# disables that button instead of crashing. UNCHANGED from the previous draft.
try:
    import data_fetcher
    DATA_FETCHER_AVAILABLE = True
except Exception:                                            # noqa: BLE001
    DATA_FETCHER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Explicit status -> colour mapping (rendered as HTML/CSS badges + table styling)
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
# UNCHANGED — same filenames, same parsing, same cache decorator.
# ---------------------------------------------------------------------------
@st.cache_data
def load_math_data():
    return pd.read_csv("processed_math_data.csv", parse_dates=["Time"])


@st.cache_data
def load_forecast_data():
    return pd.read_csv("ml_forecast_results.csv")


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
# THE ONE shared "latest status" snapshot. Every tab and the map read from
# this single variable — never recomputed as separate queries. UNCHANGED.
# ===========================================================================
snapshot = (
    df.sort_values(["Station_ID", "Time"])
      .groupby("Station_ID", as_index=False)
      .tail(1)
)
active_alerts = snapshot[snapshot["Alert_Active"] == True]        # noqa: E712

# ===========================================================================
# SIDEBAR — strict no-data rule: title, global filters, fetch button only.
# No raw tables, no alert lists here — those live inside the tabs instead.
# ===========================================================================
with st.sidebar:
    st.markdown("# 💧 AquaSentinel")
    st.caption("Groundwater Early-Warning Command Center")
    st.markdown("---")

    st.markdown("### 🗂 Global Filters")
    # SCHEMA NOTE: stations.csv/blocks.csv only carry Block_ID today — there is
    # no State or District column anywhere in the pipeline, so only a Block
    # filter is wired up here. Add State/District columns upstream to extend this.
    available_blocks = sorted(snapshot["Block_ID"].dropna().unique())
    selected_blocks = st.multiselect(
        "Block",
        available_blocks,
        default=available_blocks,
        key="sidebar_block_filter",
        help=("Filters every tab and the map by Block_ID. State/District "
              "filters aren't available yet — those columns don't exist in "
              "stations.csv or blocks.csv."))

    st.markdown("---")
    fetch_clicked = st.button(
        "📡 Fetch Live Telemetry Update",
        type="primary",
        use_container_width=True,
        disabled=not DATA_FETCHER_AVAILABLE,
        key="sidebar_fetch_btn",
        help=("Appends fresh 6-hourly DWLR readings and re-runs Builder 1's "
              "math engine, then reloads the dashboard — no server restart. "
              "Disabled because data_fetcher.py isn't available in this "
              "environment." if not DATA_FETCHER_AVAILABLE else
              "Appends fresh 6-hourly DWLR readings and re-runs Builder 1's "
              "math engine, then reloads the dashboard — no server restart."))

    # Fetch logic is UNCHANGED from the previous draft, only relocated here.
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
        st.info(st.session_state["aqua_update_msg"])

# Sidebar-filtered snapshot — every tab below reads from this, not from
# `snapshot` directly, so the Block filter applies dashboard-wide.
filtered = snapshot[snapshot["Block_ID"].isin(selected_blocks)]

# ===========================================================================
# MAIN NAVIGATION — 5 tabs
# ===========================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🗺️ Global Command", "📈 Station Analytics", "🔍 Drift Matrix",
     "⚖️ Live vs. Audit", "🧠 AI Performance"])

# ===========================================================================
# TAB 1 — GLOBAL COMMAND
# ===========================================================================
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Stations", len(filtered))
    k2.metric("Safe", int((filtered["Estimated_Category"] == "Safe").sum()))
    k3.metric("Critical", int((filtered["Estimated_Category"] == "Critical").sum()))
    k4.metric("Over-Exploited",
              int((filtered["Estimated_Category"] == "Over-Exploited").sum()))

    st.markdown("#### 🗺 Live station map")
    has_coords = ("Latitude" in filtered.columns
                  and "Longitude" in filtered.columns
                  and filtered["Latitude"].notna().any()
                  and filtered["Longitude"].notna().any())

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
        m = folium.Map(location=center, zoom_start=7, tiles="CartoDB dark_matter")

        for row in filtered.itertuples():
            color = status_color(row.Estimated_Category)
            popup = folium.Popup(
                f"<b>{row.Station_ID}</b><br>Block {row.Block_ID}<br>"
                f"{status_badge(row.Estimated_Category)}<br>"
                f"Proxy {row.Estimated_SoE_Proxy_Pct:.1f}%<br>"
                f"Confidence {row.Confidence}<br>"
                f"Alert {'🔴 ON' if row.Alert_Active else '⚪ off'}",
                max_width=260)
            folium.CircleMarker(
                location=[row.Latitude, row.Longitude],
                radius=9, color=color, fill=True, fill_color=color,
                fill_opacity=0.85, weight=1.5,
                tooltip=row.Station_ID, popup=popup,
            ).add_to(m)

        click = st_folium(m,
                          returned_objects=["last_object_clicked_tooltip"],
                          use_container_width=True,
                          key="global_command_map")

        clicked_id = click.get("last_object_clicked_tooltip")
        if clicked_id:
            detail = filtered[filtered["Station_ID"] == clicked_id]
            if not detail.empty:
                d = detail.iloc[0]
                st.markdown(
                    f"##### 📍 {d.Station_ID} — Block {d.Block_ID} "
                    f"{status_badge(d.Estimated_Category)}",
                    unsafe_allow_html=True)
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Latest depth (m)", f"{d.Water_Level:.2f}")
                d2.metric("Latest proxy %", f"{d.Estimated_SoE_Proxy_Pct:.2f}"
                          if pd.notna(d.Estimated_SoE_Proxy_Pct) else "—")
                d3.metric("12-mo decline (m)", f"{d.Depth_Decline_Proxy:.2f}"
                          if pd.notna(d.Depth_Decline_Proxy) else "—")
                d4.metric("Official category", d.Official_Category)

    with st.expander("🚨 Active Hysteresis Alerts", expanded=False):
        st.caption("Alert turns ON above 72% and OFF below 68% — this "
                   "hysteresis band stops a station flapping in and out of "
                   "alert state on small fluctuations.")
        tab_alerts = filtered[filtered["Alert_Active"] == True]   # noqa: E712
        if tab_alerts.empty:
            st.success("No stations are currently above the alert threshold.")
        else:
            st.markdown(f"**{len(tab_alerts)} station(s)** currently active:")
            for row in tab_alerts.sort_values(
                    "Estimated_SoE_Proxy_Pct", ascending=False).itertuples():
                st.markdown(
                    f"{status_badge(row.Estimated_Category)} "
                    f"**{row.Station_ID}** (Block {row.Block_ID}) — "
                    f"{row.Estimated_SoE_Proxy_Pct:.1f}%",
                    unsafe_allow_html=True)

# ===========================================================================
# TAB 2 — STATION ANALYTICS
# ===========================================================================
with tab2:
    st.markdown("#### 📈 Station time series", unsafe_allow_html=True)
    stations_sorted = sorted(filtered["Station_ID"].unique())
    station_id = st.selectbox(
        "Select a station", stations_sorted,
        index=0 if stations_sorted else None,
        key="station_analytics_select",
        help="History is Builder 1's monthly extraction proxy. The dashed "
             "segment is Builder 2's GCN-LSTM forecast, spliced onto the "
             "final historical point so the line has no visual gap.")

    if station_id is None:
        st.info("No stations match the current sidebar filters.")
    else:
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
                horizon = int(fc_row["Forecast_Horizon_Months"].iloc[0])
                last_point = hist.iloc[-1]
                time_zero = last_point["Time"]
                forecast_time = time_zero + pd.DateOffset(months=horizon)
                forecast_value = fc_row["Forecasted_SoE_Proxy_Pct"].iloc[0]

                fig.add_trace(go.Scatter(
                    x=[time_zero, forecast_time],
                    y=[last_point["Estimated_SoE_Proxy_Pct"], forecast_value],
                    mode="lines+markers", name=f"{horizon}-month forecast (GCN-LSTM)",
                    line=dict(color="#f59e0b", dash="dash", width=2.5),
                    marker=dict(size=8)))
            else:
                st.caption("No forecast row for this station (insufficient history).")

            for level, color, label in [(70, "#2ecc71", "Safe ≤ 70"),
                                        (90, "#f1c40f", "Semi-Critical ≤ 90"),
                                        (100, "#f39c12", "Critical ≤ 100"),
                                        (72, "#e74c3c", "Hysteresis ON > 72"),
                                        (68, "#7f8c8d", "Hysteresis OFF < 68")]:
                fig.add_hline(y=level, line_dash="dot", line_color=color,
                              annotation_text=label, annotation_position="bottom left",
                              annotation_font_color=color)

            fig.update_layout(
                height=460, margin=dict(l=10, r=10, t=30, b=10),
                title=f"{station_id} — extraction proxy % over time",
                xaxis_title="Time", yaxis_title="Estimated_SoE_Proxy_Pct (%)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                font=dict(color="#e2e8f0"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No proxy history available for this station.")

# ===========================================================================
# TAB 3 — DRIFT MATRIX
# ===========================================================================
with tab3:
    st.markdown("#### 🔍 Full station audit", unsafe_allow_html=True)

    s1, s2 = st.columns([2, 2])
    with s1:
        search_text = st.text_input(
            "Search station or block", "",
            key="drift_search",
            help="Matches against Station_ID and Block_ID.")
    with s2:
        category_filter = st.multiselect(
            "Filter by estimated category", CATEGORY_ORDER,
            default=CATEGORY_ORDER, key="drift_category_filter")

    drift_view = filtered[[
        "Station_ID", "Block_ID", "Official_Category", "Estimated_Category",
        "Drift_Flag", "Estimated_SoE_Proxy_Pct", "Confidence",
    ]].rename(columns={
        "Station_ID": "Station", "Block_ID": "Block",
        "Official_Category": "Official", "Estimated_Category": "Live Estimated",
        "Drift_Flag": "Drift Status", "Estimated_SoE_Proxy_Pct": "Live Proxy %",
    }).sort_values(["Block", "Station"])

    if search_text.strip():
        needle = search_text.strip().lower()
        drift_view = drift_view[
            drift_view["Station"].str.lower().str.contains(needle)
            | drift_view["Block"].str.lower().str.contains(needle)]
    if category_filter:
        drift_view = drift_view[drift_view["Live Estimated"].isin(category_filter)]

    drift_view["Drift Status"] = drift_view["Drift Status"].map(
        {True: "🟠 Drifting", False: "· Matches"})

    def _highlight_category(val):
        color = status_color(val)
        return f"background-color: {color}; color: #0a0f1e; font-weight: 600;"

    styled = drift_view.style.map(_highlight_category, subset=["Official", "Live Estimated"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=560)
    st.caption(f"{len(drift_view)} of {len(filtered)} station(s) shown.")

# ===========================================================================
# TAB 4 — LIVE VS. AUDIT
# ===========================================================================
with tab4:
    st.markdown(
        "#### ⚖️ Live vs. Audit",
        help=("Compares the static GEC-2015 audit (block-level official "
              "category, refreshed every few years) with live DWLR telemetry "
              "that AquaSentinel re-estimates monthly. Where the two diverge, "
              "the station is flagged as drifting."))

    if not filtered.empty:
        st.caption(f"Last telemetry reading: `{df['Time'].max():%Y-%m-%d %H:%M}`")

    official_counts = filtered["Official_Category"].value_counts()
    live_counts = filtered["Estimated_Category"].value_counts()
    live_oe = int(live_counts.get("Over-Exploited", 0))
    official_oe = int(official_counts.get("Over-Exploited", 0))
    drift_n = int(filtered["Drift_Flag"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stations Compared", len(filtered))
    k2.metric("Official Over-Exploited", official_oe)
    k3.metric("Live Over-Exploited", live_oe, delta=live_oe - official_oe,
              delta_color="inverse")
    k4.metric("Active Drifts", drift_n)

    mode = st.radio(
        "Chart mode", ["Extraction Proxy (%)", "Volumetric Units"],
        horizontal=True, key="live_audit_mode_radio",
        help="Both views use a single Y-axis by design — dual-axis bar "
             "charts are easy to misread at a glance.")

    chart_df = (filtered[filtered["Estimated_SoE_Proxy_Pct"].notna()]
                .sort_values(["Block_ID", "Station_ID"]).copy())

    if chart_df.empty:
        st.info("No live extraction-proxy values yet for the current filters.")
    elif mode == "Extraction Proxy (%)":
        proxy_max = float(chart_df["Estimated_SoE_Proxy_Pct"].max())
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_df["Station_ID"], y=chart_df["Estimated_SoE_Proxy_Pct"],
            name="Live 12-mo extraction proxy (%)",
            marker_color=[status_color(c) for c in chart_df["Estimated_Category"]],
            text=[f"{v:.0f}%" for v in chart_df["Estimated_SoE_Proxy_Pct"]],
            textposition="outside",
            customdata=chart_df["Block_ID"],
            hovertemplate="<b>%{x}</b> · Block %{customdata}<br>"
                          "live proxy %{y:.1f}%<extra></extra>"))
        fig.add_hline(y=100, line_color="#e74c3c", line_dash="dash",
                      annotation_text="proxy = official limit (100%)",
                      annotation_position="bottom right",
                      annotation_font_color="#e74c3c")
        fig.update_layout(
            height=480, margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Station",
            yaxis=dict(title="Extraction proxy (%)", range=[0, max(160.0, proxy_max * 1.15)]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, font=dict(color="#e2e8f0"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        # "Volumetric Units" mode: grouped bar, single shared Y-axis — no
        # dual axes. Only real columns are used: Net_Availability is the
        # official annual availability; Depth_Decline_Proxy is the 12-month
        # decline the proxy % is computed from. These are a decline-based
        # proxy for extraction, not an independently audited bcm figure —
        # flagged via the tooltip rather than a paragraph.
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_df["Station_ID"], y=chart_df["Net_Availability"],
            name="Official Annual Availability",
            marker_color="#22d3ee",
            hovertemplate="<b>%{x}</b><br>availability %{y:.2f} units<extra></extra>"))
        fig.add_trace(go.Bar(
            x=chart_df["Station_ID"], y=chart_df["Depth_Decline_Proxy"],
            name="Live Extraction (12-mo decline proxy)",
            marker_color="#f59e0b",
            hovertemplate="<b>%{x}</b><br>decline proxy %{y:.2f} units<extra></extra>"))
        fig.update_layout(
            height=480, margin=dict(l=10, r=10, t=40, b=10),
            barmode="group",
            xaxis_title="Station", yaxis_title="Units (see tooltip)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(color="#e2e8f0"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Decline proxy is Builder 1's 12-month depth-decline figure, "
                   "the same input the % proxy is derived from — not an "
                   "independently metered extraction volume.")

    with st.expander("📋 Block-Level Audit Summary", expanded=True):
        bsum = (filtered.groupby("Block_ID", as_index=False)
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
            p_display = f"{p:.1f}%" if pd.notna(p) else "—"
            b_rows.append(
                f"<tr><td><b>{row.Block_ID}</b></td><td>{int(row.Stations)}</td>"
                f"<td>{status_badge(row.Official_Category)}</td>"
                f"<td>{p_display}</td><td>{row.Official_Availability:.1f}</td>"
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

# ===========================================================================
# TAB 5 — AI PERFORMANCE & VALIDATION
# ===========================================================================
with tab5:
    st.markdown("#### 🧠 AI Performance & Validation", unsafe_allow_html=True)

    m1, _, _ = st.columns(3)
    m1.metric(
        "GCN-LSTM Holdout RMSE", "0.1534",
        help=("Reported evaluation metric from Builder 2's own training run "
              "(scaled Estimated_SoE_Proxy_Pct, 6-month holdout). Not "
              "recomputed here — ml_forecast_results.csv only carries the "
              "final point forecast, not holdout ground-truth, so this "
              "dashboard displays the reported figure rather than deriving it."))

    st.markdown(
        "**Why Spatial Adjacency (GCN) matters:** a plain per-station LSTM "
        "only ever sees one station's own history. The GCN-LSTM instead "
        "connects every pair of stations that share a Block_ID, so it can "
        "learn cross-boundary flow between neighboring monitoring points — "
        "a block-wide drawdown shows up in a station's forecast before its "
        "own telemetry fully reflects it.")

    st.markdown("##### 📊 Current forecast snapshot")
    st.caption("This is Builder 2's point forecast per station, not a "
               "holdout-vs-actual validation chart — ml_forecast_results.csv "
               "doesn't carry holdout ground-truth to plot against.")

    fc_view = forecast[forecast["Station_ID"].isin(filtered["Station_ID"])]
    if fc_view.empty:
        st.info("No forecast rows available for the current filters — run "
                 "Builder 2, or widen the sidebar Block filter.")
    else:
        fc_sorted = fc_view.sort_values("Forecasted_SoE_Proxy_Pct", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fc_sorted["Station_ID"], y=fc_sorted["Forecasted_SoE_Proxy_Pct"],
            marker_color="#f59e0b",
            text=[f"{v:.0f}%" for v in fc_sorted["Forecasted_SoE_Proxy_Pct"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>forecast %{y:.1f}%<extra></extra>"))
        fig.add_hline(y=100, line_color="#e74c3c", line_dash="dash",
                      annotation_text="official limit (100%)",
                      annotation_position="bottom right",
                      annotation_font_color="#e74c3c")
        horizon_label = (f"{int(fc_sorted['Forecast_Horizon_Months'].iloc[0])}-month"
                         if not fc_sorted.empty else "")
        fig.update_layout(
            height=440, margin=dict(l=10, r=10, t=40, b=10),
            title=f"{horizon_label} forward extraction-proxy forecast, by station",
            xaxis_title="Station", yaxis_title="Forecasted proxy (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, font=dict(color="#e2e8f0"))
        st.plotly_chart(fig, use_container_width=True)