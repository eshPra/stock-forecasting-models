
# 📊 Stock Market Time Series Forecasting

> 🔬 **One-Month Internship Project** | 📈 Powered by Data | 🧠 Built with ML + DL + Python

---

<p align="center">
  <img src="./Stock_Price_Prediction.webp" alt="Project Banner" width="80%">
</p>

---

## 🌟 Project Overview

This project delivers an end-to-end **stock market forecasting system** using advanced **time series models**.  
We integrated both classical and deep learning techniques into a unified platform with **interactive dashboards**.

### 🧪 Models Implemented

- 🔢 **ARIMA** – For stationary series and short-term predictions  
- 🔁 **SARIMA** – For data with seasonal patterns  
- 🔮 **Prophet** – For flexible trend and seasonality modeling  
- 🧬 **LSTM** – For long-term and non-linear deep learning forecasting

---

## 🛠️ Tech Stack

- **Languages & Frameworks**: Python, Streamlit, TensorFlow/Keras  
- **Libraries Used**:
  - 📦 `pandas`, `numpy`, `yfinance`
  - 📊 `matplotlib`, `plotly`, `seaborn`
  - ⚙️ `scikit-learn`, `statsmodels`, `prophet`
  - 🤖 `tensorflow`, `keras`

---

## 📁 Project Structure

```

stock-forecasting-models/
├── arima.py                    # ARIMA model dashboard
├── sarima.py                   # SARIMA model dashboard
├── prophet.py                  # Prophet model dashboard
├── lstm.py                     # LSTM model dashboard
├── Stock\_Price\_Prediction.webp # Project banner image
└── README.md                   # Project documentation

````

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stock-forecasting-models.git
cd stock-forecasting-models
````

### 2. Create and Activate Virtual Environment

```bash
python -m venv stock_env

# On Windows:
stock_env\Scripts\activate

# On macOS/Linux:
source stock_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run a Model

```bash
# For ARIMA
streamlit run arima.py

# For SARIMA
streamlit run sarima.py

# For Prophet
streamlit run prophet.py

# For LSTM
streamlit run lstm.py
```

---

## 🎯 Which Model to Use for Which Stock?

| Stock | Recommended Model | Reason                                     | Suggested Settings/Range                |
| ----- | ----------------- | ------------------------------------------ | --------------------------------------- |
| AAPL  | LSTM              | Long-term trend with non-linear behavior   | Epochs: 100, Seq Len: 60, Units: 128    |
| TSLA  | Prophet           | Sudden jumps + trend shifts                | CPS: 0.3, Forecast: 90 days             |
| MSFT  | SARIMA            | Regular seasonality with trend             | p,d,q: 1,1,1 / P,D,Q,s: 1,0,1,12        |
| AMZN  | ARIMA             | Stable trend, log-stationary series        | Auto ARIMA recommended                  |
| META  | LSTM              | Tech stock with non-linear growth          | Seq Len: 90, Dropout: 0.2, Units: 64    |
| NVDA  | SARIMA            | Strong seasonality patterns                | s: 22 or 12, d=1, seasonal P=1          |
| GOOGL | Prophet           | Trend & outlier resistance needed          | CPS: 0.4, holiday regressors (optional) |
| NFLX  | LSTM              | Fluctuates with momentum and volume trends | Use RSI, MACD, Volatility indicators    |

---

## 🧪 Model Features

### 🔢 ARIMA

* Best for short-term, stationary series
* Quick and interpretable forecasting
* Lightweight and fast execution

### 🔁 SARIMA

* Captures both trend and seasonality
* Auto parameter optimization (optional)
* Includes ADF test for stationarity

### 🔮 Prophet

* Flexible with changepoints and seasonality
* Handles missing data, trend shifts, and outliers
* Weekly/Yearly/Holiday trend modeling

### 🧬 LSTM

* Deep learning with memory-based architecture
* Bidirectional LSTM with dropout regularization
* Includes technical indicators (RSI, MACD, Bollinger Bands)
* Visual feedback: Loss, MAE, Accuracy plots

---

## ⚙️ LSTM Parameter Range (Tunable)

| Parameter        | Typical Range         |
| ---------------- | --------------------- |
| Sequence Length  | 30 - 120              |
| LSTM Units       | 32, 64, 128, 256      |
| Epochs           | 20 - 200              |
| Batch Size       | 16, 32, 64, 128       |
| Dropout Rate     | 0.1 - 0.5             |
| Learning Rate    | 0.001, 0.0005, 0.0001 |
| Validation Split | 0.1 - 0.3             |

---

## 📈 Evaluation Metrics

All models are benchmarked using the following:

* ✔️ **RMSE** – Root Mean Squared Error
* ✔️ **MAE** – Mean Absolute Error
* ✔️ **MAPE** – Mean Absolute Percentage Error
* ✔️ **R² Score** – Coefficient of Determination
* ✔️ **Directional Accuracy** – % of correctly predicted trends

---

## 🧠 Learning Outcomes

* 💡 Implemented classical and deep learning forecasting techniques
* 💡 Learned financial data handling and preprocessing
* 💡 Compared model accuracy using key performance metrics
* 💡 Developed Streamlit-based interactive dashboards
* 💡 Practiced effective collaboration within a team environment

---

## 👨‍💻 Project Contributors

> 🤝 Developed as part of a **One-Month Internship** collaborative project:

* Hamaesh S
* Mrudul Dehankar
* Jeevan Jiji Thomas
* Sahil Khan
* Anoheeta Mukherjee
* Esha Prajapati
* Akshad Viresh Makhana

---

## 📚 References

* [Yahoo Finance API](https://finance.yahoo.com/)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [Facebook Prophet Guide](https://facebook.github.io/prophet/)
* [TensorFlow LSTM Guide](https://www.tensorflow.org/tutorials)
* [Statsmodels Docs](https://www.statsmodels.org/)

---

⭐ *If this helped you learn or saved time, don’t forget to give this repo a star!*
📬 For feedback or collaboration, feel free to connect with any team member via **GitHub** or **LinkedIn**.


