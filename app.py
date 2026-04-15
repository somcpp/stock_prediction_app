import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import streamlit as st
import os
from datetime import timedelta

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Trend Prediction",
    page_icon="📈",
    layout="wide",
)

# ─── Custom CSS enhancements ──────────────────────────────────────────────────
st.markdown("""
<style>
    /* Title gradient effect */
    h1 {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-align: center;
        font-size: 3rem !important;
    }

    /* Section headers */
    h2 {
        color: #00d2ff !important;
        border-bottom: 1px solid rgba(0, 210, 255, 0.2);
        padding-bottom: 8px;
    }

    h3 {
        color: #7dd3fc !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(0, 210, 255, 0.15);
        border-radius: 12px;
        padding: 16px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5) !important;
        color: #fff !important;
    }

    /* DataFrame rounded corners */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Signal table styling */
    .buy-signal { color: #6bcb77; font-weight: bold; }
    .sell-signal { color: #ff6b6b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ─── Title ────────────────────────────────────────────────────────────────────
st.title("📈 Stock Trend Prediction")
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:1.1rem;'>"
    "Enter a stock ticker to visualize historical trends, LSTM predictions & 2-week forecasts"
    "</p>",
    unsafe_allow_html=True,
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    user_input = st.text_input("Stock Ticker", "AAPL", help="e.g. AAPL, TSLA, GOOG, MSFT")
    period = st.selectbox("Data Period", ["5y", "10y", "max"], index=1)
    forecast_days = st.slider("Forecast Days", min_value=7, max_value=30, value=14, step=1,
                               help="Number of future days to forecast")
    signal_window = st.selectbox("Signal MA Window", [10, 20, 50], index=0,
                                  help="Short MA window used for buy/sell signal generation")
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.85rem;color:#888;'>"
        "Powered by LSTM Deep Learning Model & yfinance"
        "</p>",
        unsafe_allow_html=True,
    )

# ─── Fetch Data ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching stock data…")
def load_data(ticker: str, period: str) -> pd.DataFrame:
    """Download historical stock data via yfinance."""
    data = yf.download(ticker, period=period, auto_adjust=True)
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


df = load_data(user_input, period)

if df.empty:
    st.error(f"❌ No data found for ticker **{user_input}**. Please check the symbol and try again.")
    st.stop()

# ─── Data Description ─────────────────────────────────────────────────────────
st.markdown("## 📊 Data Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Start Date", str(df.index[0].date()))
with col2:
    st.metric("End Date", str(df.index[-1].date()))
with col3:
    st.metric("Total Records", f"{len(df):,}")
with col4:
    latest_close = df['Close'].iloc[-1]
    st.metric("Latest Close", f"${latest_close:.2f}")

with st.expander("📋 View Raw Data Statistics", expanded=False):
    st.dataframe(df.describe().style.format("{:.2f}"), use_container_width=True)

# ─── Closing Price Chart ──────────────────────────────────────────────────────
st.markdown("## 📉 Closing Price Over Time")
fig_close, ax_close = plt.subplots(figsize=(14, 5))
fig_close.patch.set_facecolor('#0d0d1a')
ax_close.set_facecolor('#0d0d1a')
ax_close.plot(df.index, df['Close'], color='#00d2ff', linewidth=1.2, label='Close Price')
ax_close.set_xlabel('Date', color='#aaa', fontsize=11)
ax_close.set_ylabel('Price (USD)', color='#aaa', fontsize=11)
ax_close.tick_params(colors='#888')
ax_close.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='#e0e0e0')
ax_close.grid(color='#333', linestyle='--', linewidth=0.4)
for spine in ax_close.spines.values():
    spine.set_color('#333')
st.pyplot(fig_close)

# ─── Moving Averages ─────────────────────────────────────────────────────────
st.markdown("## 📈 Moving Average Trends")

tab1, tab2, tab3 = st.tabs(["MA100", "MA100 & MA200", "All MAs"])

ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

def styled_chart(ax):
    ax.figure.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    ax.set_xlabel('Date', color='#aaa', fontsize=11)
    ax.set_ylabel('Price (USD)', color='#aaa', fontsize=11)
    ax.tick_params(colors='#888')
    ax.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='#e0e0e0')
    ax.grid(color='#333', linestyle='--', linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_color('#333')

with tab1:
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(df.index, df['Close'], color='#00d2ff', linewidth=1, alpha=0.7, label='Close')
    ax1.plot(df.index, ma100, color='#ff6b6b', linewidth=1.5, label='MA 100')
    styled_chart(ax1)
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(df.index, df['Close'], color='#00d2ff', linewidth=1, alpha=0.7, label='Close')
    ax2.plot(df.index, ma100, color='#ff6b6b', linewidth=1.5, label='MA 100')
    ax2.plot(df.index, ma200, color='#ffd93d', linewidth=1.5, label='MA 200')
    styled_chart(ax2)
    st.pyplot(fig2)

with tab3:
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    ma50 = df['Close'].rolling(50).mean()
    ax3.plot(df.index, df['Close'], color='#00d2ff', linewidth=1, alpha=0.5, label='Close')
    ax3.plot(df.index, ma50, color='#6bcb77', linewidth=1.5, label='MA 50')
    ax3.plot(df.index, ma100, color='#ff6b6b', linewidth=1.5, label='MA 100')
    ax3.plot(df.index, ma200, color='#ffd93d', linewidth=1.5, label='MA 200')
    styled_chart(ax3)
    st.pyplot(fig3)


# ─── Load Model ──────────────────────────────────────────────────────────────
model_path = os.path.join(os.path.dirname(__file__), 'keras_model.h5')
if not os.path.exists(model_path):
    st.error("❌ Model file `keras_model.h5` not found. Please place it in the same directory as this app.")
    st.stop()

from tensorflow.keras.layers import Dense as _Dense, LSTM as _LSTM, Dropout as _Dropout

class Dense(_Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)

class LSTM(_LSTM):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)

class Dropout(_Dropout):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)

try:
    model = load_model(
        model_path,
        custom_objects={'Dense': Dense, 'LSTM': LSTM, 'Dropout': Dropout},
    )
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ─── Prepare Data ────────────────────────────────────────────────────────────
split_idx = int(len(df) * 0.70)
data_training = pd.DataFrame(df['Close'][:split_idx])
data_testing  = pd.DataFrame(df['Close'][split_idx:])

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(data_training)

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test  = np.array(x_test)
y_test  = np.array(y_test)

with st.spinner("Running LSTM prediction…"):
    y_predicted = model.predict(x_test)

scale_factor   = 1 / scaler.scale_[0]
y_predicted_us = y_predicted * scale_factor      # un-scaled
y_test_us      = y_test * scale_factor

# ─── 2-Week Future Forecast ──────────────────────────────────────────────────
with st.spinner(f"Generating {forecast_days}-day future forecast…"):
    # Use the last 100 scaled data-points as seed
    last_100_scaled = input_data[-100:]
    future_input    = last_100_scaled.copy()
    future_preds    = []

    for _ in range(forecast_days):
        seq       = future_input[-100:].reshape(1, 100, 1)
        next_pred = model.predict(seq, verbose=0)[0, 0]
        future_preds.append(next_pred)
        future_input = np.append(future_input, [[next_pred]], axis=0)

    future_preds_us = np.array(future_preds) * scale_factor

# Build future date index (skip weekends for realism)
last_date    = df.index[-1]
future_dates = []
d = last_date
while len(future_dates) < forecast_days:
    d = d + timedelta(days=1)
    if d.weekday() < 5:          # Mon–Fri only
        future_dates.append(d)

# ─── Buy / Sell Signal Generation ────────────────────────────────────────────
# Strategy: short MA of predicted prices crosses over/under a longer MA
def generate_signals(prices: np.ndarray, short_w: int = 5, long_w: int = 20):
    """Return arrays of buy/sell indices based on MA crossover on `prices`."""
    s = pd.Series(prices.flatten())
    ma_short = s.rolling(short_w).mean()
    ma_long  = s.rolling(long_w).mean()

    buy_idx  = []
    sell_idx = []
    position = None   # None = no pos, 'long' = holding

    for i in range(1, len(s)):
        if pd.isna(ma_short.iloc[i]) or pd.isna(ma_long.iloc[i]):
            continue
        prev_short, prev_long = ma_short.iloc[i - 1], ma_long.iloc[i - 1]
        curr_short, curr_long = ma_short.iloc[i], ma_long.iloc[i]
        # Bullish crossover → BUY
        if prev_short <= prev_long and curr_short > curr_long and position != 'long':
            buy_idx.append(i)
            position = 'long'
        # Bearish crossover → SELL
        elif prev_short >= prev_long and curr_short < curr_long and position == 'long':
            sell_idx.append(i)
            position = None

    return buy_idx, sell_idx

# Generate signals on the historical test-set predictions
buy_idx, sell_idx = generate_signals(y_predicted_us, short_w=signal_window, long_w=signal_window * 3)

# Align signals to the real dates in the test window
test_dates = df.index[split_idx + 100 - len(past_100_days):]
# safe guard length
test_dates = test_dates[:len(y_test_us)]

# ─── Prediction Section ──────────────────────────────────────────────────────
st.markdown("## 🤖 LSTM Model Prediction")
st.markdown("### 🔮 Original Price vs Predicted Price + 2-Week Forecast")

fig_pred, ax_pred = plt.subplots(figsize=(16, 7))
fig_pred.patch.set_facecolor('#0d0d1a')
ax_pred.set_facecolor('#0d0d1a')

# Historical actual & predicted
ax_pred.plot(test_dates, y_test_us,        color='#00d2ff', linewidth=1.3,
             label='Actual Price',  alpha=0.9)
ax_pred.plot(test_dates, y_predicted_us,   color='#ff6b6b', linewidth=1.3,
             label='Predicted Price', alpha=0.9)

# Fill between actual and predicted
ax_pred.fill_between(test_dates, y_test_us, y_predicted_us.flatten(),
                     alpha=0.07, color='#ff6b6b')

# ── Future forecast band ──────────────────────────────────────────────────────
ax_pred.plot(future_dates, future_preds_us, color='#ffd93d', linewidth=2,
             linestyle='--', label=f'{forecast_days}-Day Forecast', marker='o',
             markersize=4, markerfacecolor='#ffd93d')

# Shaded forecast region
ax_pred.axvspan(last_date, future_dates[-1], alpha=0.07, color='#ffd93d',
                label='Forecast Zone')

# Vertical separator
ax_pred.axvline(x=last_date, color='#ffd93d', linestyle=':', linewidth=1.5, alpha=0.6)
ax_pred.text(last_date, ax_pred.get_ylim()[0] if ax_pred.get_ylim()[0] != 0 else y_test_us.min(),
             '  Today', color='#ffd93d', fontsize=9, va='bottom')

# ── Buy / Sell markers ────────────────────────────────────────────────────────
if len(buy_idx) > 0:
    valid_buy = [i for i in buy_idx if i < len(test_dates)]
    ax_pred.scatter([test_dates[i] for i in valid_buy],
                    [y_predicted_us[i] for i in valid_buy],
                    color='#6bcb77', marker='^', s=120, zorder=5, label='Buy Signal')

if len(sell_idx) > 0:
    valid_sell = [i for i in sell_idx if i < len(test_dates)]
    ax_pred.scatter([test_dates[i] for i in valid_sell],
                    [y_predicted_us[i] for i in valid_sell],
                    color='#ff4d4d', marker='v', s=120, zorder=5, label='Sell Signal')

# Styling
ax_pred.set_xlabel('Date', color='#aaa', fontsize=12)
ax_pred.set_ylabel('Price (USD)', color='#aaa', fontsize=12)
ax_pred.set_title(f'{user_input} — Actual vs Predicted + {forecast_days}-Day Forecast',
                  color='#e0e0e0', fontsize=14, fontweight='bold')
ax_pred.tick_params(colors='#888')
ax_pred.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax_pred.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax_pred.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax_pred.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='#e0e0e0',
               fontsize=10, loc='upper left')
ax_pred.grid(color='#333', linestyle='--', linewidth=0.4)
for spine in ax_pred.spines.values():
    spine.set_color('#333')

st.pyplot(fig_pred)

# ─── 14-Day Forecast Table ───────────────────────────────────────────────────
st.markdown(f"### 📅 {forecast_days}-Day Price Forecast")

forecast_df = pd.DataFrame({
    "Date":           [d.strftime("%A, %d %b %Y") for d in future_dates],
    "Forecast Price": [f"${p:.2f}" for p in future_preds_us],
    "Change":         ["—"] + [
        f"{'▲' if future_preds_us[i] > future_preds_us[i-1] else '▼'} "
        f"{abs(future_preds_us[i] - future_preds_us[i-1]):.2f}"
        for i in range(1, len(future_preds_us))
    ],
})

st.dataframe(forecast_df, use_container_width=True, hide_index=True)

# ─── Buy / Sell Signal Table ─────────────────────────────────────────────────
st.markdown("### 🟢🔴 Buy & Sell Signal History")

signal_rows = []
valid_buy  = [i for i in buy_idx  if i < len(test_dates)]
valid_sell = [i for i in sell_idx if i < len(test_dates)]

for i in valid_buy:
    signal_rows.append({
        "Date":   test_dates[i].strftime("%d %b %Y"),
        "Signal": "🟢 BUY",
        "Price":  f"${float(y_predicted_us[i]):.2f}",
        "Action": "Consider buying / entering a long position",
    })

for i in valid_sell:
    signal_rows.append({
        "Date":   test_dates[i].strftime("%d %b %Y"),
        "Signal": "🔴 SELL",
        "Price":  f"${float(y_predicted_us[i]):.2f}",
        "Action": "Consider selling / exiting the position",
    })

if signal_rows:
    signals_df = pd.DataFrame(signal_rows).sort_values("Date").reset_index(drop=True)
    st.dataframe(signals_df, use_container_width=True, hide_index=True)
else:
    st.info("No clear buy/sell crossover signals were detected in the test window. "
            "Try reducing the signal MA window in the sidebar.")

# ─── Performance Metrics ─────────────────────────────────────────────────────
st.markdown("## 📐 Model Performance Metrics")

mae  = mean_absolute_error(y_test_us, y_predicted_us)
rmse = np.sqrt(mean_squared_error(y_test_us, y_predicted_us))
mape = np.mean(np.abs((y_test_us - y_predicted_us.flatten()) / y_test_us)) * 100
r2   = r2_score(y_test_us, y_predicted_us)
da   = np.mean(
    np.sign(np.diff(y_test_us)) == np.sign(np.diff(y_predicted_us.flatten()))
) * 100  # Directional Accuracy

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("MAE",  f"${mae:.2f}",  help="Mean Absolute Error — average dollar error")
with m2:
    st.metric("RMSE", f"${rmse:.2f}", help="Root Mean Squared Error — penalises large errors")
with m3:
    st.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
with m4:
    st.metric("R² Score", f"{r2:.4f}", help="Coefficient of determination (1.0 = perfect)")
with m5:
    st.metric("Directional Acc.", f"{da:.1f}%",
              help="% of days where model correctly predicted price direction")

with st.expander("ℹ️ What do these metrics mean?", expanded=False):
    st.markdown("""
| Metric | Description | Good Range |
|--------|-------------|------------|
| **MAE** | Average absolute dollar difference between actual and predicted prices | Lower is better |
| **RMSE** | Similar to MAE but penalises large errors more heavily | Lower is better |
| **MAPE** | Percentage error — useful for comparing across different price scales | < 5% is excellent |
| **R² Score** | How well predictions explain variance in actual prices (1 = perfect) | > 0.90 is good |
| **Directional Accuracy** | How often the model correctly predicts whether price goes up or down | > 55% beats random |
    """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    "<br><p style='text-align:center;color:#555;font-size:0.85rem;'>"
    "⚠️ This is for educational purposes only — not financial advice.<br>"
    "Model trained on historical AAPL data. Predictions on other tickers use the same model weights."
    "</p>",
    unsafe_allow_html=True,
)