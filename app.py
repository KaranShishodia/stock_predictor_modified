"""
app.py
------
Streamlit dashboard for interactive stock analysis and prediction.

Run
---
    streamlit run app.py

Features
--------
- Sidebar controls: ticker, date range, forecast horizon
- Live data download via yfinance
- KPI cards: latest price, daily change, moving averages
- Interactive Plotly charts: price history, MAs, RSI, MACD, Bollinger Bands
- Model loading with graceful fallback
- Back-test predictions vs actual prices
- N-day future price forecast with confidence context
- Model metrics (RMSE, MAE, MAPE, R²) displayed in the UI
- Export predictions to CSV
"""

import os
import io
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from data import fetch_stock_data, add_all_indicators, split_data
from evaluate import compute_metrics
from model import load_model, build_lstm_model
from train import train as train_model

# ────────────────────────────────────────────────────────────────────────────
# Page configuration
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal dark-theme CSS tweak
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 18px 22px;
        border-left: 4px solid;
        margin-bottom: 8px;
    }
    .metric-title { font-size: 13px; color: #aaa; margin-bottom: 4px; }
    .metric-value { font-size: 24px; font-weight: 700; color: #fff; }
    .metric-delta { font-size: 13px; margin-top: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("---")

    POPULAR_TICKERS = ["AAPL", "GOOG", "AMZN", "MSFT", "TSLA", "NVDA", "META", "NFLX", "Custom..."]
    selected = st.selectbox("Stock Ticker", options=POPULAR_TICKERS, index=1)
    if selected == "Custom...":
        ticker = st.text_input("Enter Ticker Symbol", value="", placeholder="e.g. UBER").upper().strip()
        if not ticker:
            st.warning("Please enter a ticker symbol.")
            st.stop()
    else:
        ticker = selected
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start", datetime.date(2015, 1, 1))
    with col_e:
        end_date = st.date_input("End", datetime.date(2024, 1, 1))

    st.markdown("---")
    st.subheader("Model")
    model_path = st.text_input(
        "Model Path (.keras)",
        value=config.MODEL_SAVE_PATH,
        help="Path to a trained .keras model file.",
    )
    future_days = st.slider("Forecast Horizon (days)", min_value=5, max_value=90, value=30)

    st.markdown("---")
    predict_btn = st.button("🚀 Run Prediction", use_container_width=True)
    forecast_btn = st.button("🔮 Forecast Future Prices", use_container_width=True)

    st.markdown("---")
    st.caption("📌 Powered by Keras · yfinance · Streamlit")


# ────────────────────────────────────────────────────────────────────────────
# Title
# ────────────────────────────────────────────────────────────────────────────
st.title("📈 Stock Predictor Pro")
st.markdown("*AI-powered stock price analysis and forecasting dashboard*")
st.markdown("---")


# ────────────────────────────────────────────────────────────────────────────
# Load data
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data(ticker, start, end):
    df = fetch_stock_data(ticker, str(start), str(end))
    return add_all_indicators(df)


with st.spinner(f"Downloading data for **{ticker}**..."):
    try:
        df = get_data(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.stop()

if df.empty:
    st.error("No data returned. Check the ticker symbol and date range.")
    st.stop()

st.success(f"✅ Loaded **{len(df):,}** trading days for **{ticker}**")


# ────────────────────────────────────────────────────────────────────────────
# KPI Cards
# ────────────────────────────────────────────────────────────────────────────
latest   = float(df["Close"].iloc[-1])
prev     = float(df["Close"].iloc[-2])
chg      = latest - prev
chg_pct  = (chg / prev) * 100
ma50     = float(df["SMA_50"].iloc[-1]) if "SMA_50" in df else float(df["Close"].rolling(50).mean().iloc[-1])
ma200    = float(df["SMA_200"].iloc[-1]) if "SMA_200" in df else float(df["Close"].rolling(200).mean().iloc[-1])
rsi_now  = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else None

arrow    = "▲" if chg >= 0 else "▼"
chg_clr  = "#28a745" if chg >= 0 else "#dc3545"

c1, c2, c3, c4, c5 = st.columns(5)

def kpi(col, title, value, delta="", border_color="#0d6efd", delta_color="#aaa"):
    col.markdown(
        f"""
        <div class="metric-card" style="border-color:{border_color}">
          <div class="metric-title">{title}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-delta" style="color:{delta_color}">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

kpi(c1, "Latest Close",  f"${latest:.2f}",  f"{arrow} {abs(chg):.2f} ({chg_pct:.2f}%)", "#0d6efd", chg_clr)
kpi(c2, "50-Day SMA",    f"${ma50:.2f}",     "Moving Average",    "#ffc107", "#aaa")
kpi(c3, "200-Day SMA",   f"${ma200:.2f}",    "Moving Average",    "#dc3545", "#aaa")
kpi(c4, "Data Points",   f"{len(df):,}",     f"{start_date} → {end_date}", "#6f42c1", "#aaa")
if rsi_now is not None:
    rsi_signal = "Overbought ⚠️" if rsi_now > 70 else ("Oversold 🟢" if rsi_now < 30 else "Neutral")
    kpi(c5, "RSI (14)", f"{rsi_now:.1f}", rsi_signal, "#17a2b8", "#aaa")

st.markdown("---")


# ────────────────────────────────────────────────────────────────────────────
# Charts – tabs
# ────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Price & MAs", "📉 RSI", "📈 MACD", "🎯 Bollinger Bands"]
)

# ── Tab 1: Price + Moving Averages
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#0d6efd", width=1.5)))
    for col, clr in zip(
        [c for c in df.columns if c.startswith("SMA_")],
        ["#dc3545", "#fd7e14", "#28a745", "#6f42c1"],
    ):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)))
    fig.update_layout(
        title=f"{ticker} — Close Price & Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: RSI
with tab2:
    if "RSI" in df.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#fd7e14")))
        fig_rsi.add_hline(y=70, line=dict(color="red",   dash="dash"), annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line=dict(color="green", dash="dash"), annotation_text="Oversold (30)")
        fig_rsi.update_layout(title=f"{ticker} — RSI (14)", template="plotly_dark", height=400, yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)
    else:
        st.info("RSI data not available.")

# ── Tab 3: MACD
with tab3:
    if "MACD" in df.columns:
        fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.4, 0.6])
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#0d6efd")), row=1, col=1)
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#0d6efd")), row=2, col=1)
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="#dc3545")), row=2, col=1)
        hist_colors = ["#28a745" if v >= 0 else "#dc3545" for v in df["MACD_Hist"]]
        fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram", marker_color=hist_colors), row=2, col=1)
        fig_macd.update_layout(title=f"{ticker} — MACD", template="plotly_dark", height=550)
        st.plotly_chart(fig_macd, use_container_width=True)
    else:
        st.info("MACD data not available.")

# ── Tab 4: Bollinger Bands
with tab4:
    if "BB_Upper" in df.columns:
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper", line=dict(color="#dc3545", dash="dot")))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower", line=dict(color="#28a745", dash="dot"), fill="tonexty", fillcolor="rgba(13,110,253,0.06)"))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df["BB_Middle"], name="BB Middle", line=dict(color="#fd7e14", dash="dash")))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#0d6efd", width=1.5)))
        fig_bb.update_layout(title=f"{ticker} — Bollinger Bands", template="plotly_dark", height=500)
        st.plotly_chart(fig_bb, use_container_width=True)
    else:
        st.info("Bollinger Band data not available.")


# ────────────────────────────────────────────────────────────────────────────
# Model — retrain whenever the ticker changes
# ────────────────────────────────────────────────────────────────────────────
if "model_cache" not in st.session_state:
    st.session_state.model_cache = {}   # { ticker: {"model": ..., "scaler": ...} }

model = None
scaler = None

cache_key = f"{ticker}_{str(start_date)}_{str(end_date)}"

if cache_key in st.session_state.model_cache:
    model  = st.session_state.model_cache[cache_key]["model"]
    scaler = st.session_state.model_cache[cache_key]["scaler"]
    st.sidebar.success(f"✅ Model ready ({ticker})")
else:
    st.sidebar.info(f"🔄 Training model for **{ticker}**…")
    train_placeholder = st.empty()
    with train_placeholder.container():
        st.info(f"⏳ Training LSTM for **{ticker}** — this takes ~1–2 minutes. Charts are live above.")
        prog = st.progress(0, text="Preparing data…")

    try:
        prog.progress(10, text="Fetching & engineering features…")
        ticker_model_path  = f"models/{ticker}_lstm.keras"
        ticker_scaler_path = f"models/{ticker}_scaler.pkl"

        prog.progress(25, text="Building & training LSTM…")
        result = train_model(
            ticker=ticker,
            start_date=str(start_date),
            end_date=str(end_date),
            model_path=ticker_model_path,
            scaler_path=ticker_scaler_path,
        )
        prog.progress(95, text="Finalising…")

        scaler = result["data"]["scaler"]
        model  = load_model(ticker_model_path)

        st.session_state.model_cache[cache_key] = {"model": model, "scaler": scaler}
        prog.progress(100, text="Done!")
        train_placeholder.empty()
        st.sidebar.success(f"✅ Model trained & ready ({ticker})")

    except Exception as e:
        train_placeholder.empty()
        st.sidebar.error(f"Training failed: {e}")
        st.error(f"❌ Could not train model for **{ticker}**: {e}")


# ────────────────────────────────────────────────────────────────────────────
# Back-test predictions
# ────────────────────────────────────────────────────────────────────────────
if predict_btn:
    if model is None or scaler is None:
        st.warning("No model loaded — cannot run predictions.")
    else:
        st.markdown("---")
        st.subheader("🤖 Back-test: Predicted vs Actual")

        with st.spinner("Running inference on test set..."):
            train_df, test_df = split_data(df)

            overlap = train_df["Close"].tail(config.LOOKBACK_WINDOW)
            test_close = pd.concat([overlap, test_df["Close"]])
            test_scaled = scaler.transform(test_close.values.reshape(-1, 1)).flatten()

            X_test, y_test_scaled = [], []
            for i in range(config.LOOKBACK_WINDOW, len(test_scaled)):
                X_test.append(test_scaled[i - config.LOOKBACK_WINDOW : i])
                y_test_scaled.append(test_scaled[i])

            X_test = np.array(X_test).reshape(-1, config.LOOKBACK_WINDOW, 1)
            y_pred_scaled = model.predict(X_test, verbose=0)
            y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
            y_actual = scaler.inverse_transform(np.array(y_test_scaled).reshape(-1, 1)).flatten()
            pred_dates = test_df.index[-len(y_actual):]

        metrics = compute_metrics(y_actual, y_pred)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE",  f"${metrics['rmse']:.2f}")
        m2.metric("MAE",   f"${metrics['mae']:.2f}")
        m3.metric("MAPE",  f"{metrics['mape']:.2f}%")
        m4.metric("R²",    f"{metrics['r2']:.4f}")

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=y_actual, name="Actual",    line=dict(color="#28a745", width=1.5)))
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=y_pred,   name="Predicted", line=dict(color="#dc3545", width=1.5, dash="dash")))
        fig_pred.update_layout(
            title=f"{ticker} — Actual vs Predicted Close Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            height=500,
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Download button
        pred_df = pd.DataFrame({"Date": pred_dates, "Actual": y_actual, "Predicted": y_pred})
        csv_buf = io.StringIO()
        pred_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Predictions CSV",
            data=csv_buf.getvalue(),
            file_name=f"{ticker}_predictions.csv",
            mime="text/csv",
        )


# ────────────────────────────────────────────────────────────────────────────
# Future forecast
# ────────────────────────────────────────────────────────────────────────────
if forecast_btn:
    if model is None or scaler is None:
        st.warning("No model loaded — cannot forecast.")
    else:
        st.markdown("---")
        st.subheader(f"🔮 {future_days}-Day Price Forecast")

        with st.spinner(f"Forecasting the next {future_days} trading days..."):
            train_df, test_df = split_data(df)

            seed = df["Close"].values[-config.LOOKBACK_WINDOW:]
            window = scaler.transform(seed.reshape(-1, 1)).flatten().tolist()

            preds_scaled = []
            for _ in range(future_days):
                x = np.array(window[-config.LOOKBACK_WINDOW:]).reshape(1, config.LOOKBACK_WINDOW, 1)
                p = model.predict(x, verbose=0)[0, 0]
                preds_scaled.append(p)
                window.append(p)

            preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
            last_date = df.index[-1]
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

        # Chart: last 200 days + forecast
        hist_tail = df["Close"].tail(200)
        fig_fut = go.Figure()
        fig_fut.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="Historical", line=dict(color="#0d6efd")))
        fig_fut.add_trace(go.Scatter(x=future_dates, y=preds, name="Forecast", line=dict(color="#fd7e14", dash="dash", width=2), fill="tozeroy", fillcolor="rgba(253,126,20,0.06)"))
        fig_fut.add_vline(x=str(last_date), line=dict(color="#888", dash="dot"))
        fig_fut.update_layout(
            title=f"{ticker} — {future_days}-Day Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            height=500,
        )
        st.plotly_chart(fig_fut, use_container_width=True)

        fcast_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": preds})
        st.dataframe(fcast_df.set_index("Date").style.format("${:.2f}"), use_container_width=True)

        csv_buf2 = io.StringIO()
        fcast_df.to_csv(csv_buf2, index=False)
        st.download_button(
            "⬇️ Download Forecast CSV",
            data=csv_buf2.getvalue(),
            file_name=f"{ticker}_forecast_{future_days}d.csv",
            mime="text/csv",
        )


# ────────────────────────────────────────────────────────────────────────────
# Raw data viewer
# ────────────────────────────────────────────────────────────────────────────
with st.expander("📋 View Raw Data & Indicators"):
    st.dataframe(df.tail(100).style.format("{:.4f}"), use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🔧 Stock Predictor Pro · Built with Streamlit, Keras & yfinance · For educational purposes only.")
