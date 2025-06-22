## Stock Forecasting Models

Welcome to the **Stock Forecasting Models** repository — an advanced system built using statistical, machine learning, and deep learning models to forecast stock prices. This project includes interactive dashboards, comprehensive evaluation metrics, and multiple forecasting approaches with customizable configurations.

---

## Project Overview

This system enables users to:

- Predict stock prices using ARIMA, SARIMA, Prophet, and Hybrid LSTM models
- Visualize forecasts with interactive candlestick and line charts
- Analyze technical indicators like RSI, MACD, Bollinger Bands
- Compare multiple models with accuracy metrics

---

## Models Implemented

- **ARIMA & SARIMA** (with auto-parameter tuning)
- **Facebook Prophet** (with log scaling + external regressors)
- **Hybrid CNN-LSTM** (enhanced with Conv1D, Bidirectional LSTMs, and regularization)

---

## File Structure

| File                  | Description |
|-----------------------|-------------|
| `app1.py`             | Main Streamlit dashboard with all models integrated |
| `enhanced_forecasting.py` | Modular pipeline for preprocessing, feature engineering, model training, evaluation, and plotting |
| `Prophet.py`          | Lightweight Streamlit app using only Prophet with UI sliders |

---

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/stock-forecasting-models.git
cd stock-forecasting-models
2. Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is missing, see manual installation below.

 Required Libraries
If not using requirements.txt, install manually:

bash
Copy
Edit
pip install streamlit yfinance pandas numpy matplotlib seaborn plotly prophet scikit-learn tensorflow statsmodels ta
▶️ How to Run
Run full dashboard with all models:
bash
Copy
Edit
streamlit run app1.py
Run Prophet-only forecaster:
bash
Copy
Edit
streamlit run Prophet.py
You can change the ticker, forecast horizon, and model settings from the sidebar.

Features
Forecast using 4 modeling approaches
Advanced metrics: RMSE, MAE, MAPE, R², Direction Accuracy
Download predictions as CSV
Visual indicators: RSI, MACD, Bollinger Bands
Deep Learning: Hybrid CNN + LSTM architecture
Interactive Plotly charts
Streamlit-based intuitive UI

