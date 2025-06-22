import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Streamlit page config
st.set_page_config(layout="wide")
st.title("📈 Stock Market Forecasting with ARIMA")

# User inputs
ticker = st.selectbox("Choose a stock ticker", ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"], index=0)
forecast_days = st.slider("Forecast period (days)", 30, 120, 90, step=10)

# ARIMA parameters
st.sidebar.header("ARIMA Parameters")
p = st.sidebar.slider("AR order (p)", 0, 5, 1)
d = st.sidebar.slider("Differencing order (d)", 0, 2, 1)
q = st.sidebar.slider("MA order (q)", 0, 5, 1)

@st.cache_data(show_spinner=False)
def load_data(ticker):
    data = yf.download(ticker, start="2015-01-01", end="2023-12-31")
    df = data[['Close']].reset_index()
    df.columns = ['ds', 'y']
    df['y'] = np.log(df['y'])  # Log transformation for stationarity
    return df

def check_stationarity(timeseries):
    """Check stationarity using Augmented Dickey-Fuller test"""
    result = adfuller(timeseries.dropna())
    return {
        'adf_stat': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] <= 0.05
    }

def auto_arima_order(data, max_p=5, max_d=2, max_q=5):
    """Simple auto ARIMA order selection based on AIC"""
    best_aic = float('inf')
    best_order = (1, 1, 1)
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted_model = model.fit()
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                except:
                    continue
    
    return best_order, best_aic

if st.button("🔮 Run ARIMA Forecast"):
    df = load_data(ticker)
    
    # Display stationarity test
    st.subheader("Stationarity Analysis")
    stationarity_result = check_stationarity(df['y'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ADF Statistic", f"{stationarity_result['adf_stat']:.4f}")
    with col2:
        st.metric("P-value", f"{stationarity_result['p_value']:.4f}")
    with col3:
        st.metric("Is Stationary", "Yes" if stationarity_result['is_stationary'] else "No")
    
    # Auto ARIMA option
    if st.checkbox("Use Auto ARIMA (may take time)"):
        with st.spinner("Finding optimal ARIMA parameters..."):
            best_order, best_aic = auto_arima_order(df['y'])
            st.success(f"Best ARIMA order: {best_order} with AIC: {best_aic:.2f}")
            p, d, q = best_order
    
    # Split into train and test
    train = df[:-forecast_days]['y']
    test = df[-forecast_days:]['y']
    
    try:
        # Fit ARIMA model
        with st.spinner("Training ARIMA model..."):
            model = ARIMA(train, order=(p, d, q))
            fitted_model = model.fit()
        
        # Make predictions
        forecast_result = fitted_model.forecast(steps=forecast_days)
        forecast_ci = fitted_model.get_forecast(steps=forecast_days).conf_int()
        
        # Inverse transform predictions (from log scale)
        y_true = np.exp(test.values)
        y_pred = np.exp(forecast_result)
        y_pred_lower = np.exp(forecast_ci.iloc[:, 0])
        y_pred_upper = np.exp(forecast_ci.iloc[:, 1])
        
        # Evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Display metrics
        st.subheader("Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE", f"{rmse:.2f}")
        with col2:
            st.metric("MAE", f"{mae:.2f}")
        with col3:
            st.metric("MAPE", f"{mape:.2f}%")
        with col4:
            st.metric("AIC", f"{fitted_model.aic:.2f}")
        
        # Forecast vs actual plot
        test_dates = df[-forecast_days:]['ds']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_dates, y=y_true, mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test_dates, y=y_pred, mode='lines', name='ARIMA Predicted', line=dict(color='red')))
        fig.add_trace(go.Scatter(
            x=test_dates, y=y_pred_upper,
            mode='lines', name='Upper CI', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(
            x=test_dates, y=y_pred_lower,
            mode='lines', name='Lower CI', fill='tonexty', line=dict(width=0), showlegend=False))
        
        fig.update_layout(
            title=f"{ticker} ARIMA({p},{d},{q}) Forecast vs Actual",
            xaxis_title='Date',
            yaxis_title='Stock Price',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model summary
        st.subheader("Model Summary")
        st.text(str(fitted_model.summary()))
        
        # Residual analysis
        st.subheader("Residual Analysis")
        residuals = fitted_model.resid
        
        fig_resid, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Residuals plot
        ax1.plot(residuals)
        ax1.set_title('Residuals')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Residuals')
        
        # Residuals histogram
        ax2.hist(residuals, bins=30, alpha=0.7)
        ax2.set_title('Residuals Histogram')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        
        # ACF of residuals
        plot_acf(residuals, ax=ax3, lags=20)
        ax3.set_title('ACF of Residuals')
        
        # PACF of residuals
        plot_pacf(residuals, ax=ax4, lags=20)
        ax4.set_title('PACF of Residuals')
        
        plt.tight_layout()
        st.pyplot(fig_resid)
        
    except Exception as e:
        st.error(f"Error fitting ARIMA model: {str(e)}")
        st.info("Try adjusting the ARIMA parameters or using auto ARIMA.")

# Information section
st.sidebar.markdown("---")
st.sidebar.markdown("### About ARIMA")
st.sidebar.markdown("""
**ARIMA** (AutoRegressive Integrated Moving Average) is a popular time series forecasting method.

**Parameters:**
- **p**: Number of lag observations (AR term)
- **d**: Degree of differencing (I term)
- **q**: Size of moving average window (MA term)

**Best for:** Stationary time series with clear patterns.
""")