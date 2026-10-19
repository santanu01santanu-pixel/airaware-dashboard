import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="AirAware | Air Quality Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
  .stApp { background: #f4f7fb; }
  [data-testid="stSidebar"] { background: #0b1727; }
  [data-testid="stSidebar"] * { color: #f2f6fb !important; }
  .hero {
    background: linear-gradient(120deg, #0b1727 0%, #173c51 58%, #168879 100%);
    padding: 1.6rem 1.8rem; border-radius: 22px; color: white; margin-bottom: 1.2rem;
  }
  .hero h1 { font-size: clamp(2rem, 4vw, 3.2rem); margin: 0; color: white; }
  .hero p { margin: .45rem 0 0; color: #d6e7ef; font-size: 1rem; }
  .demo-note {
    background: #fff5db; border-left: 5px solid #e0a328; padding: .8rem 1rem;
    border-radius: 8px; color: #62440c; margin: .8rem 0 1rem;
  }
  .metric-label { color: #607085; font-size: .88rem; }
  div[data-testid="stMetric"] {
    background: white; border: 1px solid #e2e8f0; border-radius: 15px;
    padding: 1rem 1rem .7rem; box-shadow: 0 3px 12px rgba(15, 23, 42, .035);
  }
  div[data-testid="stMetricLabel"] p { color: #526276 !important; }
  .section-note { color: #627187; font-size: .9rem; }
  footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- Demo data ----------
@st.cache_data
def make_demo_data():
    """Generate synthetic teaching/demo data. This is not observed pollution data."""
    rng = np.random.default_rng(42)
    cities = {
        "Bhubaneswar": (20.2961, 85.8245, 85),
        "Delhi": (28.6139, 77.2090, 180),
        "Mumbai": (19.0760, 72.8777, 95),
        "Kolkata": (22.5726, 88.3639, 120),
        "Chennai": (13.0827, 80.2707, 78),
        "Hyderabad": (17.3850, 78.4867, 88),
        "Bengaluru": (12.9716, 77.5946, 65),
        "Cuttack": (20.4625, 85.8830, 82),
    }
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    rows = []
    for city, (lat, lon, base) in cities.items():
        idx = np.arange(len(dates))
        seasonal = 18 * np.sin(idx * 2 * np.pi / 90) + 8 * np.cos(idx * 2 * np.pi / 30)
        noise = rng.normal(0, 18, len(dates))
        aqi = np.clip(base + seasonal + noise, 15, 420).round(1)
        pm25 = np.clip(aqi * 0.39 + rng.normal(0, 7, len(dates)), 4, 180).round(1)
        pm10 = np.clip(aqi * 0.73 + rng.normal(0, 12, len(dates)), 8, 350).round(1)
        no2 = np.clip(8 + aqi * 0.11 + rng.normal(0, 4, len(dates)), 2, 100).round(1)
        for date, a, p25, p10, n2 in zip(dates, aqi, pm25, pm10, no2):
            rows.append({
                "date": date, "city": city, "latitude": lat, "longitude": lon,
                "aqi_demo": a, "pm25_demo_ugm3": p25, "pm10_demo_ugm3": p10,
                "no2_demo_ugm3": n2,
                "data_source": "SYNTHETIC DEMO DATA — NOT REAL OBSERVATIONS",
            })
    return pd.DataFrame(rows).sort_values(["date", "city"]).reset_index(drop=True)

df = make_demo_data()

def category(value):
    # Broad display bands for demo visualization only; not official CPCB AQI calculation.
    if value <= 50:
        return "Lower demo band"
    if value <= 100:
        return "Moderate demo band"
    if value <= 200:
        return "Elevated demo band"
    if value <= 300:
        return "High demo band"
    return "Very high demo band"

df["demo_band"] = df["aqi_demo"].apply(category)

# ---------- Sidebar filters ----------
st.sidebar.markdown("## 🌿 AirAware")
st.sidebar.caption("AIR QUALITY INTELLIGENCE")
st.sidebar.markdown("---")
all_cities = sorted(df["city"].unique())
selected_cities = st.sidebar.multiselect(
    "Choose cities to compare",
    all_cities,
    default=["Bhubaneswar", "Delhi", "Mumbai", "Kolkata"],
)
if not selected_cities:
    st.sidebar.warning("Select at least one city to display the dashboard.")
    st.stop()

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = pd.Timestamp(min_date), pd.Timestamp(max_date)

metric_choice = st.sidebar.selectbox(
    "Trend metric",
    options=["AQI (demo)", "PM2.5 (demo)", "PM10 (demo)", "NO₂ (demo)"],
)
metric_map = {
    "AQI (demo)": ("aqi_demo", "Illustrative AQI value"),
    "PM2.5 (demo)": ("pm25_demo_ugm3", "Illustrative PM2.5 (µg/m³)"),
    "PM10 (demo)": ("pm10_demo_ugm3", "Illustrative PM10 (µg/m³)"),
    "NO₂ (demo)": ("no2_demo_ugm3", "Illustrative NO₂ (µg/m³)"),
}
metric_col, metric_label = metric_map[metric_choice]

filtered = df[
    df["city"].isin(selected_cities)
    & df["date"].between(start_date, end_date)
].copy()

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>🌿 AirAware</h1>
  <p>Explore pollution patterns. Compare cities. Preview predictive analytics.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="demo-note"><b>DEMONSTRATION MODE:</b> All readings, trends, pollutant values, and model illustrations on this website use generated synthetic data. They are not live readings or real observations and must not be cited as real-world research findings. The category bands shown are illustrative, not the official CPCB AQI calculation.</div>',
    unsafe_allow_html=True,
)

# ---------- KPI cards ----------
latest_date = filtered["date"].max()
latest = filtered[filtered["date"] == latest_date]
overall_avg = filtered[metric_col].mean()
highest_city_row = latest.loc[latest["aqi_demo"].idxmax()]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Cities selected", f"{len(selected_cities)}")
k2.metric("Days in view", f"{filtered['date'].nunique()}")
k3.metric(f"Average {metric_choice}", f"{overall_avg:.1f}" if pd.notna(overall_avg) else "—")
k4.metric("Highest demo AQI", f"{highest_city_row['aqi_demo']:.1f}", highest_city_row["city"])

st.markdown("### Pollution trend")
st.caption("Daily synthetic demo values. Select a pollutant and date range in the sidebar.")
trend = filtered.groupby(["date", "city"], as_index=False)[metric_col].mean()
fig_trend = px.line(
    trend, x="date", y=metric_col, color="city", markers=False,
    labels={"date": "Date", metric_col: metric_label, "city": "City"},
    template="plotly_white",
)
fig_trend.update_layout(
    legend_title_text="", hovermode="x unified", margin=dict(l=10, r=10, t=25, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_trend, use_container_width=True)

left, right = st.columns([1.15, 1])
with left:
    st.markdown("### City comparison")
    comparison = (
        filtered.groupby("city", as_index=False)
        .agg(avg_aqi=("aqi_demo", "mean"), avg_pm25=("pm25_demo_ugm3", "mean"), avg_pm10=("pm10_demo_ugm3", "mean"))
        .sort_values("avg_aqi", ascending=False)
    )
    fig_bar = px.bar(
        comparison, x="city", y="avg_aqi", color="avg_aqi",
        color_continuous_scale=["#35a58b", "#f0c66b", "#d9695d"],
        labels={"city": "City", "avg_aqi": "Average demo AQI"},
        template="plotly_white",
    )
    fig_bar.update_layout(coloraxis_showscale=False, margin=dict(l=5, r=5, t=20, b=5),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_bar, use_container_width=True)
with right:
    st.markdown("### Latest city snapshot")
    latest_table = latest[["city", "aqi_demo", "pm25_demo_ugm3", "pm10_demo_ugm3", "demo_band"]].copy()
    latest_table.columns = ["City", "Demo AQI", "PM2.5 demo", "PM10 demo", "Illustrative band"]
    st.dataframe(latest_table.round(1), use_container_width=True, hide_index=True)

st.markdown("### Geospatial view")
st.caption("Markers show city coordinates. Marker colors/values are synthetic demo AQI values, not current conditions.")
map_data = latest[["city", "latitude", "longitude", "aqi_demo", "pm25_demo_ugm3"]].copy()
if not map_data.empty:
   fig_map = px.scatter_map(
        map_data,
        lat="latitude", lon="longitude", size="aqi_demo", color="aqi_demo",
        hover_name="city", hover_data={"aqi_demo": True, "pm25_demo_ugm3": True, "latitude": False, "longitude": False},
        color_continuous_scale=["#35a58b", "#f0c66b", "#d9695d"],
        zoom=3.4, center={"lat": 21.5, "lon": 82.0}, height=430,
        map_style="open-street-map",
        labels={"aqi_demo": "Demo AQI", "pm25_demo_ugm3": "Demo PM2.5"},
    )
fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar_title="Demo AQI"
    )
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("### Prediction module — preview")
st.caption("This deployable starter does not present fabricated model accuracy as a research result. The real-data model will be connected after the dataset and prediction protocol are validated.")
predict_city = st.selectbox("Choose a city for a demo next-step illustration", selected_cities, key="predict_city")
city_hist = df[df["city"] == predict_city].sort_values("date")
last_value = float(city_hist.iloc[-1]["aqi_demo"])
demo_next = float(np.clip(last_value + np.random.default_rng(7).normal(0, 8), 15, 420))
p1, p2, p3 = st.columns(3)
p1.metric("Last demo value", f"{last_value:.1f}")
p2.metric("Illustrative next value", f"{demo_next:.1f}", delta=f"{demo_next-last_value:+.1f}")
p3.metric("Validated model metrics", "Pending")
st.info("The illustrative next value is randomly generated for interface testing. It is not a trained-model prediction. We will replace it with a trained model evaluated on verified real data.")

with st.expander("View dataset sample and download CSV"):
    st.dataframe(filtered.head(20), use_container_width=True, hide_index=True)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download selected synthetic demo data (CSV)", csv, "airaware_synthetic_demo_data.csv", "text/csv")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6b7b8e;font-size:.85rem'>AirAware · M.Tech project prototype · Synthetic data clearly identified · Not for health or regulatory decisions</div>",
    unsafe_allow_html=True,
)
