import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objs as go

# Streamlit page config
st.set_page_config(layout="wide")
st.title("📊 Stock Market Forecasting with Prophet")

# User inputs
ticker = st.selectbox("Choose a stock ticker", ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"], index=0)
forecast_days = st.slider("Forecast period (days)", 30, 120, 90, step=10)
cps = st.slider("Changepoint Prior Scale", 0.1, 1.0, 0.3, step=0.1)

@st.cache_data(show_spinner=False)
def load_data(ticker):
    data = yf.download(ticker, start="2015-01-01", end="2023-12-31")
    df = data[['Close', 'Volume']].reset_index()
    df.columns = ['ds', 'y', 'volume']
    df['y'] = np.log(df['y'])
    df['volume'] = np.log(df['volume'] + 1)
    df['ma7'] = df['y'].rolling(window=7, min_periods=1).mean()
    df['ma30'] = df['y'].rolling(window=30, min_periods=1).mean()
    df['ema15'] = df['y'].ewm(span=15, adjust=False).mean()
    return df

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def direction_accuracy(y_true, y_pred):
    true_diff = np.sign(np.diff(y_true))
    pred_diff = np.sign(np.diff(y_pred))
    return np.mean(true_diff == pred_diff) * 100

if st.button("🔮 Run Forecast"):
    df = load_data(ticker)

    # Split into train and test
    train = df[:-forecast_days]
    test = df[-forecast_days:]

    # Build Prophet model
    model = Prophet(
        changepoint_prior_scale=cps,
        seasonality_mode='multiplicative',
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=True
    )

    for reg in ['volume', 'ma7', 'ma30', 'ema15']:
        model.add_regressor(reg)

    model.fit(train)

    # Prepare future dataframe
    future = model.make_future_dataframe(periods=forecast_days)
    for col in ['volume', 'ma7', 'ma30', 'ema15']:
        future[col] = pd.concat([train[col], test[col]]).reset_index(drop=True)

    forecast = model.predict(future)

    # Inverse transform predictions
    forecast['yhat'] = np.exp(forecast['yhat'])
    forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
    y_true = np.exp(test['y'].values)
    y_pred = forecast[-forecast_days:]['yhat'].values

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    dir_acc = direction_accuracy(y_true, y_pred)

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📉 RMSE", f"{rmse:.2f}")
        st.metric("📈 R² Score", f"{r2:.2f}")
    with col2:
        st.metric("📊 MAE", f"{mae:.2f}")
        st.metric("🔁 MAPE (%)", f"{mape:.2f}")
    with col3:
        st.metric("📈 Direction Accuracy", f"{dir_acc:.2f}%")

    # Plotly forecast vs actual
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test['ds'], y=y_true, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=test['ds'], y=y_pred, mode='lines', name='Predicted'))
    fig.add_trace(go.Scatter(
        x=test['ds'], y=forecast[-forecast_days:]['yhat_upper'],
        mode='lines', name='Upper CI', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(
        x=test['ds'], y=forecast[-forecast_days:]['yhat_lower'],
        mode='lines', name='Lower CI', fill='tonexty', line=dict(width=0), showlegend=False))
    fig.update_layout(title=f"{ticker} Forecast vs Actual (cps={cps})", xaxis_title='Date', yaxis_title='Stock Price')
    st.plotly_chart(fig, use_container_width=True)

    # Component plots
    st.subheader("Forecast Components")
    components_fig = model.plot_components(forecast)
    st.pyplot(components_fig)
