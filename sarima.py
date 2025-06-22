import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
import itertools
warnings.filterwarnings('ignore')

# Streamlit page config
st.set_page_config(
    page_title="SARIMA Stock Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Stock Market Forecasting with SARIMA")
st.markdown("*Optimized for better performance and reliability*")

# User inputs
col1, col2 = st.columns(2)
with col1:
    ticker = st.selectbox("Choose a stock ticker", 
                         ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX"], 
                         index=0)
with col2:
    forecast_days = st.slider("Forecast period (days)", 10, 60, 30, step=5,
                             help="Reduced range for better performance")

# SARIMA parameters with better defaults
st.sidebar.header("🔧 SARIMA Parameters")
st.sidebar.markdown("*Start with default values for best performance*")

st.sidebar.subheader("Non-seasonal parameters")
p = st.sidebar.slider("AR order (p)", 0, 2, 1, help="Autoregressive order")
d = st.sidebar.slider("Differencing order (d)", 0, 2, 1, help="Degree of differencing")
q = st.sidebar.slider("MA order (q)", 0, 2, 1, help="Moving average order")

st.sidebar.subheader("Seasonal parameters")
P = st.sidebar.slider("Seasonal AR order (P)", 0, 1, 0, help="Seasonal autoregressive order")
D = st.sidebar.slider("Seasonal differencing order (D)", 0, 1, 0, help="Seasonal differencing")
Q = st.sidebar.slider("Seasonal MA order (Q)", 0, 1, 0, help="Seasonal moving average order")
s = st.sidebar.selectbox("Seasonal period (s)", [12, 22, 52], index=1, 
                        help="12=monthly, 22=monthly trading days, 52=weekly")

# Advanced options
with st.sidebar.expander("⚙️ Advanced Options"):
    use_auto_sarima = st.checkbox("Auto parameter selection", 
                                 help="Automatically find best parameters (slower)")
    log_transform = st.checkbox("Log transformation", value=True, 
                               help="Apply log transformation for stationarity")
    train_test_split = st.slider("Train/Test split (%)", 70, 90, 80, 
                                help="Percentage of data for training")

@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def load_data(ticker, start_date="2020-01-01"):
    """Load and preprocess stock data"""
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            st.error(f"No data found for ticker {ticker}")
            return None
            
        df = data[['Close']].reset_index()
        df.columns = ['ds', 'y']
        df = df.dropna()
        
        if len(df) < 100:
            st.warning(f"Limited data available for {ticker}: {len(df)} days")
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def check_stationarity(timeseries):
    """Check stationarity using Augmented Dickey-Fuller test"""
    try:
        result = adfuller(timeseries.dropna(), autolag='AIC')
        return {
            'adf_stat': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] <= 0.05
        }
    except Exception as e:
        st.error(f"Stationarity test failed: {str(e)}")
        return None

def optimized_auto_sarima(data, seasonal_period, max_order=1):
    """Optimized auto SARIMA with limited parameter space"""
    best_aic = float('inf')
    best_order = ((1, 1, 1), (0, 0, 0, seasonal_period))
    
    # Reduced parameter space for speed
    p_values = range(0, max_order + 1)
    d_values = range(0, 2)
    q_values = range(0, max_order + 1)
    
    # Even more limited seasonal parameters
    P_values = [0, 1] if seasonal_period > 1 else [0]
    D_values = [0, 1] if seasonal_period > 1 else [0]
    Q_values = [0, 1] if seasonal_period > 1 else [0]
    
    total_combinations = len(p_values) * len(d_values) * len(q_values) * len(P_values) * len(D_values) * len(Q_values)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    combination_count = 0
    
    for p, d, q in itertools.product(p_values, d_values, q_values):
        for P, D, Q in itertools.product(P_values, D_values, Q_values):
            try:
                combination_count += 1
                progress = combination_count / total_combinations
                progress_bar.progress(progress)
                status_text.text(f"Testing SARIMA({p},{d},{q})x({P},{D},{Q},{seasonal_period})... {combination_count}/{total_combinations}")
                
                model = SARIMAX(data, 
                               order=(p, d, q),
                               seasonal_order=(P, D, Q, seasonal_period),
                               enforce_stationarity=False,
                               enforce_invertibility=False)
                fitted_model = model.fit(disp=False, maxiter=50)  # Limit iterations
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_order = ((p, d, q), (P, D, Q, seasonal_period))
                    
            except Exception:
                continue
    
    progress_bar.empty()
    status_text.empty()
    
    return best_order, best_aic

def fit_sarima_model(train_data, order, seasonal_order):
    """Fit SARIMA model with error handling"""
    try:
        model = SARIMAX(train_data, 
                       order=order,
                       seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        fitted_model = model.fit(disp=False, maxiter=100)
        return fitted_model
    except Exception as e:
        st.error(f"Model fitting failed: {str(e)}")
        return None

def create_forecast_plot(actual_dates, actual_values, pred_dates, pred_values, 
                        pred_lower, pred_upper, ticker, order_info):
    """Create interactive forecast plot"""
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=actual_dates, y=actual_values, 
        mode='lines', name='Actual', 
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=pred_dates, y=pred_values, 
        mode='lines', name='SARIMA Forecast', 
        line=dict(color='#ff7f0e', width=2)
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=pred_dates, y=pred_upper,
        mode='lines', name='Upper CI', 
        line=dict(width=0, color='rgba(255,127,14,0)'), 
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=pred_dates, y=pred_lower,
        mode='lines', name='Lower CI', 
        fill='tonexty', 
        line=dict(width=0, color='rgba(255,127,14,0.3)'), 
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"{ticker} Stock Price Forecast - {order_info}",
        xaxis_title='Date',
        yaxis_title='Stock Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

# Main execution
if st.button("🚀 Run SARIMA Forecast", type="primary"):
    with st.spinner("Loading data..."):
        df = load_data(ticker)
    
    if df is not None:
        # Apply transformations
        original_y = df['y'].copy()
        
        if log_transform:
            df['y'] = np.log(df['y'])
            st.info("✅ Log transformation applied")
        
        # Display basic info
        st.subheader("📈 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Days", len(df))
        with col2:
            st.metric("Date Range", f"{df['ds'].min().strftime('%Y-%m-%d')} to {df['ds'].max().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("Latest Price", f"${original_y.iloc[-1]:.2f}")
        with col4:
            price_change = ((original_y.iloc[-1] - original_y.iloc[-2]) / original_y.iloc[-2]) * 100
            st.metric("Daily Change", f"{price_change:+.2f}%")
        
        # Stationarity test
        stationarity_result = check_stationarity(df['y'])
        
        if stationarity_result:
            st.subheader("📊 Stationarity Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ADF Statistic", f"{stationarity_result['adf_stat']:.4f}")
            with col2:
                st.metric("P-value", f"{stationarity_result['p_value']:.4f}")
            with col3:
                is_stationary = stationarity_result['is_stationary']
                st.metric("Is Stationary", 
                         "✅ Yes" if is_stationary else "❌ No",
                         delta="Good" if is_stationary else "Needs differencing")
        
        # Train-test split
        split_index = int(len(df) * train_test_split / 100)
        train_data = df.iloc[:split_index]['y']
        test_data = df.iloc[split_index:]['y']
        test_dates = df.iloc[split_index:]['ds']
        
        st.info(f"Training on {len(train_data)} days, testing on {len(test_data)} days")
        
        # Auto SARIMA option
        order = (p, d, q)
        seasonal_order = (P, D, Q, s)
        
        if use_auto_sarima:
            with st.spinner("🔍 Finding optimal parameters..."):
                best_order, best_aic = optimized_auto_sarima(train_data, s, max_order=1)
                order, seasonal_order = best_order
                st.success(f"✅ Optimal parameters found: SARIMA{order}x{seasonal_order} (AIC: {best_aic:.2f})")
        
        # Fit model
        with st.spinner("🎯 Training SARIMA model..."):
            fitted_model = fit_sarima_model(train_data, order, seasonal_order)
        
        if fitted_model is not None:
            # Make predictions
            try:
                forecast_steps = len(test_data)
                forecast_result = fitted_model.forecast(steps=forecast_steps)
                forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()
                
                # Transform back if log transformation was used
                if log_transform:
                    y_true = np.exp(test_data.values)
                    y_pred = np.exp(forecast_result.values)
                    y_pred_lower = np.exp(forecast_ci.iloc[:, 0].values)
                    y_pred_upper = np.exp(forecast_ci.iloc[:, 1].values)
                    train_actual = np.exp(train_data.values)
                else:
                    y_true = test_data.values
                    y_pred = forecast_result.values
                    y_pred_lower = forecast_ci.iloc[:, 0].values
                    y_pred_upper = forecast_ci.iloc[:, 1].values
                    train_actual = train_data.values
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
                # Display metrics
                st.subheader("📊 Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSE", f"${rmse:.2f}")
                with col2:
                    st.metric("MAE", f"${mae:.2f}")
                with col3:
                    st.metric("MAPE", f"{mape:.2f}%")
                with col4:
                    st.metric("AIC", f"{fitted_model.aic:.2f}")
                
                # Create forecast plot
                order_info = f"SARIMA{order}x{seasonal_order}"
                fig = create_forecast_plot(
                    test_dates, y_true, test_dates, y_pred,
                    y_pred_lower, y_pred_upper, ticker, order_info
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Historical + forecast view
                st.subheader("📈 Historical Data + Forecast")
                historical_period = min(252, len(df))  # Last year
                hist_data = df.iloc[-historical_period:]
                hist_actual = np.exp(hist_data['y']) if log_transform else hist_data['y']
                
                fig_full = go.Figure()
                fig_full.add_trace(go.Scatter(
                    x=hist_data['ds'], y=hist_actual, 
                    mode='lines', name='Historical', 
                    line=dict(color='#1f77b4', width=2)
                ))
                fig_full.add_trace(go.Scatter(
                    x=test_dates, y=y_pred, 
                    mode='lines', name='Forecast', 
                    line=dict(color='#ff7f0e', width=2)
                ))
                
                fig_full.update_layout(
                    title=f"{ticker} - Full View with Forecast",
                    xaxis_title='Date',
                    yaxis_title='Stock Price ($)',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_full, use_container_width=True)
                
                # Model diagnostics
                with st.expander("🔍 Model Diagnostics"):
                    residuals = fitted_model.resid
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Residuals plot
                        fig_resid = go.Figure()
                        fig_resid.add_trace(go.Scatter(
                            y=residuals, mode='lines', name='Residuals'
                        ))
                        fig_resid.update_layout(
                            title='Residuals Over Time',
                            yaxis_title='Residuals',
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_resid, use_container_width=True)
                    
                    with col2:
                        # Residuals histogram
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(
                            x=residuals, nbinsx=30, name='Residuals'
                        ))
                        fig_hist.update_layout(
                            title='Residuals Distribution',
                            xaxis_title='Residuals',
                            yaxis_title='Frequency',
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                
                # Summary statistics
                with st.expander("📋 Model Summary"):
                    st.text(str(fitted_model.summary()))
                
                st.success("✅ SARIMA forecast completed successfully!")
                
            except Exception as e:
                st.error(f"Forecasting failed: {str(e)}")
                st.info("💡 Try reducing the forecast period or adjusting parameters")

# Information section
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About SARIMA")
st.sidebar.markdown("""
**SARIMA** (Seasonal AutoRegressive Integrated Moving Average) is perfect for time series with seasonal patterns.

**Parameters:**
- **(p,d,q)**: Non-seasonal AR, differencing, MA
- **(P,D,Q,s)**: Seasonal AR, differencing, MA, period

**Tips:**
- Start with default parameters
- Use auto-selection for optimization
- Lower values = faster execution
- Log transformation helps with volatility
""")

st.sidebar.markdown("### 🚀 Performance Tips")
st.sidebar.markdown("""
- Use smaller forecast periods (10-30 days)
- Disable auto-selection for speed
- Try seasonal period = 22 (monthly)
- Check stationarity before modeling
""")