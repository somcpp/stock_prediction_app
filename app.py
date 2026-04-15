import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st
import os

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
</style>
""", unsafe_allow_html=True)


# ─── Title ────────────────────────────────────────────────────────────────────
st.title("📈 Stock Trend Prediction")
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:1.1rem;'>"
    "Enter a stock ticker to visualize historical trends and LSTM-based predictions"
    "</p>",
    unsafe_allow_html=True,
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    user_input = st.text_input("Stock Ticker", "AAPL", help="e.g. AAPL, TSLA, GOOG, MSFT")
    period = st.selectbox("Data Period", ["5y", "10y", "max"], index=1)
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
    # Flatten multi-level columns if present
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


# ─── Prediction Section ──────────────────────────────────────────────────────
st.markdown("## 🤖 LSTM Model Prediction")

# Load model — custom wrapper to handle Keras version mismatches
model_path = os.path.join(os.path.dirname(__file__), 'keras_model.h5')
if not os.path.exists(model_path):
    st.error("❌ Model file `keras_model.h5` not found. Please place it in the same directory as this app.")
    st.stop()

# Workaround: model was saved with a newer Keras that adds 'quantization_config'
# to layer configs. The local Keras doesn't recognise it, so we subclass Dense/LSTM/Dropout
# to accept & ignore the extra kwarg.
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

# Prepare data – same pipeline as notebook
# 1) Split into training (70%) and testing (30%)
split_idx = int(len(df) * 0.70)
data_training = pd.DataFrame(df['Close'][:split_idx])
data_testing = pd.DataFrame(df['Close'][split_idx:])

# 2) Scale training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(data_training)  # fit scaler on training data

# 3) Create final_df = last 100 days of training + all testing
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

# 4) Scale the combined data (fit_transform as in notebook)
input_data = scaler.fit_transform(final_df)

# 5) Build x_test / y_test sequences (window = 100)
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

# 6) Predict
with st.spinner("Running LSTM prediction…"):
    y_predicted = model.predict(x_test)

# 7) Inverse scale
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# 8) Plot Original vs Predicted
st.markdown("### 🔮 Original Price vs Predicted Price")

fig_pred, ax_pred = plt.subplots(figsize=(14, 6))
fig_pred.patch.set_facecolor('#0d0d1a')
ax_pred.set_facecolor('#0d0d1a')

ax_pred.plot(y_test, color='#00d2ff', linewidth=1.3, label='Original Price', alpha=0.9)
ax_pred.plot(y_predicted, color='#ff6b6b', linewidth=1.3, label='Predicted Price', alpha=0.9)

ax_pred.fill_between(
    range(len(y_test)),
    y_test,
    y_predicted.flatten(),
    alpha=0.08,
    color='#ff6b6b',
)

ax_pred.set_xlabel('Time', color='#aaa', fontsize=12)
ax_pred.set_ylabel('Price (USD)', color='#aaa', fontsize=12)
ax_pred.set_title(f'{user_input} — Actual vs Predicted', color='#e0e0e0', fontsize=14, fontweight='bold')
ax_pred.tick_params(colors='#888')
ax_pred.legend(
    facecolor='#1a1a2e',
    edgecolor='#333',
    labelcolor='#e0e0e0',
    fontsize=11,
    loc='upper left',
)
ax_pred.grid(color='#333', linestyle='--', linewidth=0.4)
for spine in ax_pred.spines.values():
    spine.set_color('#333')

st.pyplot(fig_pred)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    "<br><p style='text-align:center;color:#555;font-size:0.85rem;'>"
    "⚠️ This is for educational purposes only — not financial advice.<br>"
    "Model trained on historical AAPL data. Predictions on other tickers use the same model weights."
    "</p>",
    unsafe_allow_html=True,
)