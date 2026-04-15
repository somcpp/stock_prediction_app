# 📈 Stock Trend Prediction

A **Streamlit** web application that uses a pre-trained **LSTM (Long Short-Term Memory)** deep learning model to visualize historical stock trends, predict prices, generate buy/sell signals, and forecast future prices for any stock ticker.

---

## 🚀 Demo

> Enter any stock ticker (e.g., `AAPL`, `TSLA`, `GOOG`, `MSFT`) to get instant predictions, moving average charts, and a customizable price forecast.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Data Overview** | Key metrics (start/end date, total records, latest close price) |
| 📉 **Closing Price Chart** | Interactive dark-themed chart of the historical close price |
| 📈 **Moving Averages** | MA50, MA100, and MA200 tabs for trend analysis |
| 🤖 **LSTM Prediction** | Actual vs. predicted prices on the 30% test split |
| 🔮 **Future Forecast** | Auto-regressive N-day price forecast (weekdays only) |
| 🟢🔴 **Buy/Sell Signals** | MA crossover signals plotted on the prediction chart and in a table |
| 📐 **Performance Metrics** | MAE, RMSE, MAPE, R² Score, and Directional Accuracy |

---

## 🗂️ Project Structure

```
stock/
├── app.py              # Main Streamlit application
├── keras_model.h5      # Pre-trained LSTM model (trained on AAPL)
├── LSTM.ipynb          # Jupyter notebook used for model training
├── .streamlit/         # Streamlit configuration
├── venv/               # Python virtual environment
├── .gitignore
└── README.md
```

---

## 🛠️ Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd stock
```

### 2. Create and activate a virtual environment

```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit yfinance tensorflow scikit-learn matplotlib pandas numpy
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## ⚙️ Sidebar Controls

| Control | Description | Default |
|---|---|---|
| **Stock Ticker** | Any valid Yahoo Finance ticker symbol | `AAPL` |
| **Data Period** | Historical data range to download | `10y` |
| **Forecast Days** | Number of future trading days to predict (7–30) | `14` |
| **Signal MA Window** | Short moving average window for buy/sell signal generation | `10` |

---

## 🧠 How the Model Works

1. **Data Download** — Historical OHLCV data is fetched via `yfinance`.
2. **Train/Test Split** — Data is split 70% training / 30% testing.
3. **Scaling** — Closing prices are scaled to `[0, 1]` using `MinMaxScaler`.
4. **Windowing** — Sequences of 100 consecutive days are used as input to the LSTM.
5. **Prediction** — The pre-trained model predicts the next day's normalized price.
6. **Inverse Scaling** — Predictions are converted back to USD for display.
7. **Future Forecast** — The model auto-regressively predicts N days ahead, appending each prediction as the next input (weekends excluded).

### Buy / Sell Signal Strategy

Signals are generated using a **moving average crossover** on the predicted prices:

- 🟢 **BUY** — Short MA crosses **above** long MA (bullish crossover)
- 🔴 **SELL** — Short MA crosses **below** long MA (bearish crossover)

The short window is configurable via the sidebar (`Signal MA Window`). The long window is automatically set to `3×` the short window.

---

## 📐 Performance Metrics

| Metric | Description | Target |
|---|---|---|
| **MAE** | Mean Absolute Error — average dollar error | Lower is better |
| **RMSE** | Root Mean Squared Error — punishes large errors more | Lower is better |
| **MAPE** | Mean Absolute Percentage Error | < 5% is excellent |
| **R² Score** | Variance explained by the model (1.0 = perfect fit) | > 0.90 is good |
| **Directional Accuracy** | % of days with correctly predicted price direction | > 55% beats random |

---

## ⚠️ Disclaimer

> This application is for **educational purposes only** and does **not** constitute financial advice.
> The LSTM model was trained exclusively on historical **AAPL** data. Predictions on other tickers use the same model weights and may be less accurate.

---

## 🧰 Tech Stack

- **[Streamlit](https://streamlit.io/)** — Web UI framework
- **[TensorFlow / Keras](https://www.tensorflow.org/)** — LSTM model inference
- **[yfinance](https://pypi.org/project/yfinance/)** — Stock data API
- **[scikit-learn](https://scikit-learn.org/)** — Data preprocessing & metrics
- **[Matplotlib](https://matplotlib.org/)** — Charts and visualizations
- **[Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/)** — Data manipulation

---

## 📓 Model Training

The LSTM model was built and trained in `LSTM.ipynb`. Key architecture details:

- Input window: **100 days**
- Model type: **Stacked LSTM with Dropout**
- Output: **Next day's closing price (scaled)**
- Training data: **AAPL historical data**
- Optimizer: **Adam**
- Loss: **Mean Squared Error**

To retrain the model on a different ticker, open `LSTM.ipynb`, change the ticker symbol, run all cells, and save the resulting model as `keras_model.h5` in the same directory as `app.py`.
