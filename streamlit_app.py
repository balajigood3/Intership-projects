"""
Streamlit Dashboard – Predictive Forecasting of Care Load & Placement Demand
UAC Program / HHS
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_prep import load_data, prepare_features
from models import NaivePersistence, MovingAverageForecast, MLForecaster

st.set_page_config(
    page_title="UAC Care Load Forecasting",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #1a365d; margin-bottom: 0.2rem;}
    .sub-header {font-size: 1.1rem; color: #4a5568; margin-bottom: 1.5rem;}
    .metric-card {background: #f7fafc; border-radius: 10px; padding: 1rem; border-left: 4px solid #3182ce;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    df = load_data()
    return df


@st.cache_resource
def load_models():
    models = {}
    try:
        models["rf_care"] = joblib.load(ROOT / "models" / "rf_care_load.joblib")
    except Exception:
        models["rf_care"] = None
    try:
        models["rf_discharge"] = joblib.load(ROOT / "models" / "rf_discharge.joblib")
    except Exception:
        models["rf_discharge"] = None
    return models


def make_forecast(series: pd.Series, model_name: str, horizon: int, last_date: pd.Timestamp):
    """Simple forecast helpers for dashboard"""
    if model_name == "Naive":
        m = NaivePersistence().fit(series)
        return m.predict(horizon)
    elif model_name == "Moving Average (7d)":
        m = MovingAverageForecast(7).fit(series)
        return m.predict(horizon)
    elif model_name == "Random Forest" and "rf_care" in st.session_state.get("models", {}):
        # Use pre-trained if available
        return None  # handled separately
    else:
        # fallback MA
        m = MovingAverageForecast(7).fit(series)
        return m.predict(horizon)


def main():
    st.markdown('<p class="main-header">Predictive Forecasting of Care Load & Placement Demand</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">UAC Program · U.S. Department of Health and Human Services · Unified Mentor Project</p>', unsafe_allow_html=True)

    df = get_data()
    models = load_models()
    st.session_state["models"] = models

    # ---------- Sidebar ----------
    st.sidebar.header("⚙️ Controls")
    horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30, 1)
    model_choice = st.sidebar.selectbox(
        "Primary Model",
        ["Random Forest", "Naive", "Moving Average (7d)", "Compare All"]
    )
    show_ci = st.sidebar.checkbox("Show Confidence Band (approx.)", value=True)
    target_view = st.sidebar.radio("Focus", ["Care Load (Children in HHS)", "Discharge Demand"])

    # ---------- KPI Row ----------
    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Children in HHS Care", f"{int(latest['Children_in_HHS_Care']):,}")
    with col2:
        st.metric("Yesterday Discharges", f"{int(latest['Children_discharged_HHS']):,}")
    with col3:
        avg7 = df["Children_in_HHS_Care"].tail(7).mean()
        st.metric("7-day Avg Care Load", f"{avg7:,.0f}")
    with col4:
        change = latest["Children_in_HHS_Care"] - df.iloc[-8]["Children_in_HHS_Care"]
        st.metric("7-day Change", f"{change:+.0f}")

    st.divider()

    # ---------- Historical Chart ----------
    st.subheader("📈 Historical Care Load & Discharges")
    hist_days = st.slider("History window (days)", 90, 730, 365, key="hist")
    hist = df.tail(hist_days)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist["Date"], y=hist["Children_in_HHS_Care"],
        name="Children in HHS Care", line=dict(color="#2b6cb0", width=2)
    ))
    fig_hist.add_trace(go.Scatter(
        x=hist["Date"], y=hist["Children_discharged_HHS"],
        name="Daily Discharges", line=dict(color="#38a169", width=1.5), yaxis="y2"
    ))
    fig_hist.update_layout(
        height=400,
        yaxis=dict(title="Children in Care"),
        yaxis2=dict(title="Discharges", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ---------- Forecast Section ----------
    st.subheader("🔮 Future Care Load Forecast")

    target_col = "Children_in_HHS_Care" if target_view.startswith("Care") else "Children_discharged_HHS"
    series = df[target_col]

    # Generate future dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # Simple forecasts for dashboard (fast)
    naive_pred = NaivePersistence().fit(series).predict(horizon)
    ma_pred = MovingAverageForecast(7).fit(series).predict(horizon)

    # RF recursive if model loaded
    rf_pred = None
    if models.get("rf_care") and target_col == "Children_in_HHS_Care":
        try:
            rf_model = models["rf_care"]
            # Reset history to actuals
            feat = prepare_features(df, target=target_col)
            rf_model.history = feat[target_col].tolist()
            rf_pred = rf_model.predict(horizon, last_date)
        except Exception as e:
            st.warning(f"RF forecast fallback: {e}")

    if models.get("rf_discharge") and target_col == "Children_discharged_HHS":
        try:
            rf_model = models["rf_discharge"]
            feat = prepare_features(df, target=target_col)
            rf_model.history = feat[target_col].tolist()
            rf_pred = rf_model.predict(horizon, last_date)
        except Exception:
            pass

    # Build plot
    fig_fc = go.Figure()
    # History (last 60 days)
    recent = df.tail(60)
    fig_fc.add_trace(go.Scatter(
        x=recent["Date"], y=recent[target_col],
        name="Historical", line=dict(color="#2b6cb0", width=2)
    ))

    if model_choice in ["Naive", "Compare All"]:
        fig_fc.add_trace(go.Scatter(
            x=future_dates, y=naive_pred,
            name="Naive", line=dict(color="#e53e3e", dash="dot")
        ))
    if model_choice in ["Moving Average (7d)", "Compare All"]:
        fig_fc.add_trace(go.Scatter(
            x=future_dates, y=ma_pred,
            name="Moving Avg 7d", line=dict(color="#dd6b20", dash="dash")
        ))
    if model_choice in ["Random Forest", "Compare All"] and rf_pred is not None:
        fig_fc.add_trace(go.Scatter(
            x=future_dates, y=rf_pred,
            name="Random Forest", line=dict(color="#38a169", width=2.5)
        ))
        if show_ci:
            # Approximate CI using recent residual std
            resid_std = series.diff().dropna().std() * 1.5
            upper = rf_pred + 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1) / 7)
            lower = np.maximum(0, rf_pred - 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1) / 7))
            fig_fc.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself", fillcolor="rgba(56,161,105,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Approx. 95% CI", showlegend=True
            ))

    fig_fc.update_layout(
        height=450,
        title=f"{target_col} – {horizon}-day Forecast",
        xaxis_title="Date",
        yaxis_title="Count",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast table
    with st.expander("📋 Forecast Table"):
        fc_table = pd.DataFrame({"Date": future_dates})
        fc_table["Naive"] = np.round(naive_pred).astype(int)
        fc_table["MovingAvg_7"] = np.round(ma_pred).astype(int)
        if rf_pred is not None:
            fc_table["RandomForest"] = np.round(rf_pred).astype(int)
        st.dataframe(fc_table, use_container_width=True)

    # ---------- Model Comparison (if metrics exist) ----------
    metrics_path = ROOT / "outputs" / "model_metrics.csv"
    if metrics_path.exists():
        st.subheader("📊 Model Performance (Hold-out Test)")
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, use_container_width=True)

        fig_m = px.bar(metrics, x="Model", y=["MAE", "RMSE"], barmode="group",
                       title="Error Metrics Comparison")
        st.plotly_chart(fig_m, use_container_width=True)

    # ---------- Net Pressure / Early Warning ----------
    st.subheader("⚠️ Capacity Stress Indicators")
    df["net"] = df["Children_discharged_HHS"]  # simplified
    recent_net = df.tail(30)
    avg_discharge = recent_net["Children_discharged_HHS"].mean()
    avg_care = recent_net["Children_in_HHS_Care"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("30-day Avg Daily Discharges", f"{avg_discharge:.0f}")
    c2.metric("Current Care Load", f"{int(latest['Children_in_HHS_Care']):,}")
    # Simple stress proxy
    stress = "Low" if latest["Children_in_HHS_Care"] < 5000 else ("Moderate" if latest["Children_in_HHS_Care"] < 10000 else "Elevated")
    c3.metric("Care Load Stress Level", stress)

    st.info("""
    **How to use this dashboard**  
    • Adjust the forecast horizon to plan staffing / bed capacity.  
    • Compare models to understand uncertainty.  
    • Watch the 7-day change and discharge rate for early warning of capacity pressure.  
    • Random Forest generally captures weekly seasonality and recent trends better than pure statistical baselines.
    """)

    st.caption("Data source: HHS / ORR Unaccompanied Alien Children Program (public dataset). Project for educational / research purposes.")


if __name__ == "__main__":
    main()
