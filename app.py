# # app.py - Stock Market Time Series Analysis and Forecasting (Internship Ready)
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import plotly.graph_objects as go
# import warnings
# warnings.filterwarnings('ignore')

# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, r2_score

# st.set_page_config(page_title="Stock Market Forecasting - AAPL", page_icon="📈", layout="wide")

# @st.cache_data
# def load_aapl_data():
#     data = yf.download("AAPL", start="2020-01-01", progress=False)
#     if data.empty:
#         st.error("Downloaded data is empty.")
#         return pd.DataFrame()
#     data.reset_index(inplace=True)
#     required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
#     if not all(col in data.columns for col in required_cols):
#         st.error("Required columns missing from stock data.")
#         return pd.DataFrame()
#     data.dropna(inplace=True)
#     data = data.sort_values('Date').reset_index(drop=True)
#     data['Date'] = pd.to_datetime(data['Date'])  # Ensure Date is datetime
#     return data

# def calculate_performance_metrics(actual, predicted):
#     try:
#         actual = np.array(actual).flatten()
#         predicted = np.array(predicted).flatten()
#         min_len = min(len(actual), len(predicted))
#         actual = actual[:min_len]
#         predicted = predicted[:min_len]
#         mask = ~(np.isnan(actual) | np.isnan(predicted) | np.isinf(actual) | np.isinf(predicted))
#         actual_clean = actual[mask]
#         predicted_clean = predicted[mask]
#         if len(actual_clean) == 0 or len(predicted_clean) == 0:
#             return np.nan, np.nan, np.nan, np.nan
#         rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
#         mape = np.mean(np.abs((actual_clean - predicted_clean) / np.maximum(np.abs(actual_clean), 1e-8))) * 100
#         r2 = r2_score(actual_clean, predicted_clean)
#         if len(actual_clean) > 1:
#             actual_diff = np.diff(actual_clean)
#             predicted_diff = np.diff(predicted_clean)
#             accuracy = np.mean(np.sign(actual_diff) == np.sign(predicted_diff)) * 100
#         else:
#             accuracy = np.nan
#         return round(rmse, 2), round(mape, 2), round(r2, 4), round(accuracy, 1)
#     except Exception as e:
#         st.warning(f"Error calculating metrics: {str(e)}")
#         return np.nan, np.nan, np.nan, np.nan

# def build_arima_forecast(train, test_len):
#     try:
#         model = ARIMA(train['Close'], order=(5,1,0))
#         fitted = model.fit()
#         forecast = fitted.forecast(steps=test_len)
#         return forecast.tolist()
#     except Exception as e:
#         st.warning(f"ARIMA error: {str(e)}")
#         return [np.nan]*test_len

# def build_sarima_forecast(train, test_len):
#     try:
#         model = SARIMAX(train['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
#         fitted = model.fit(disp=False)
#         forecast = fitted.forecast(steps=test_len)
#         return forecast.tolist()
#     except Exception as e:
#         st.warning(f"SARIMA error: {str(e)}")
#         return [np.nan]*test_len

# def build_lstm_forecast(train, test_len):
#     try:
#         close = train['Close'].values.reshape(-1, 1)
#         scaler = MinMaxScaler()
#         scaled = scaler.fit_transform(close)
#         seq_len = min(60, len(scaled) - 1)
#         if seq_len < 10:
#             return [np.nan]*test_len
#         generator = TimeseriesGenerator(scaled, scaled, length=seq_len, batch_size=32)
#         if len(generator) == 0:
#             return [np.nan]*test_len
#         model = Sequential([
#             LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
#             Dropout(0.2),
#             LSTM(50),
#             Dropout(0.2),
#             Dense(1)
#         ])
#         model.compile(optimizer='adam', loss='mean_squared_error')
#         model.fit(generator, epochs=10, verbose=0)
#         last_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
#         preds = []
#         for _ in range(test_len):
#             pred = model.predict(last_seq, verbose=0)[0][0]
#             preds.append(pred)
#             last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)
#         return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten().tolist()
#     except Exception as e:
#         st.warning(f"LSTM error: {str(e)}")
#         return [np.nan]*test_len

# def plot_forecasts(data, train, test, forecasts):
#     try:
#         fig = go.Figure()
#         # Actual full (faint gray)
#         fig.add_trace(go.Scatter(
#             x=data['Date'], y=data['Close'],
#             name='Actual Price (Full)',
#             line=dict(width=1, color='lightgray', dash='solid'),
#             opacity=0.4,
#             mode='lines'
#         ))
#         # Add split
#         split_date = pd.to_datetime(train['Date'].iloc[-1])
#         if hasattr(split_date, 'to_pydatetime'):
#             split_date = split_date.to_pydatetime()
#         fig.add_vline(x=split_date, line_dash="dash", line_color="gray")
#         # Annotation for split
#         fig.add_annotation(
#             x=split_date,
#             y=max(data['Close']),
#             text="Train/Test Split",
#             showarrow=True,
#             arrowhead=1,
#             yshift=10
#         )
#         colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
#         for i, (model_name, preds) in enumerate(forecasts.items()):
#             if preds is not None and len(preds) == len(test):
#                 # Actual (Test) values for this model
#                 fig.add_trace(go.Scatter(
#                     x=test['Date'], y=test['Close'],
#                     name=f'Actual (Test) - {model_name}',
#                     line=dict(color=colors[i % len(colors)], width=2, dash='solid'),
#                     mode='lines',
#                     opacity=0.7
#                 ))
#                 # Predicted values for this model
#                 fig.add_trace(go.Scatter(
#                     x=test['Date'], y=preds, name=f'Predicted ({model_name})',
#                     line=dict(dash='dot', color=colors[i % len(colors)], width=2),
#                     mode='lines'
#                 ))
#         fig.update_layout(
#             title="Actual vs Predicted AAPL Prices (Test Period)",
#             xaxis_title="Date", yaxis_title="Price ($)", height=600,
#             hovermode='x unified', showlegend=True)
#         st.plotly_chart(fig, use_container_width=True)
#     except Exception as e:
#         st.error(f"Error creating plot: {str(e)}")

# def main():
#     st.title("📈 TIME SERIES ANALYSIS AND FORECASTING FOR STOCK MARKET")
#     st.markdown("**Apple Inc. (AAPL)** stock forecasting using ARIMA, SARIMA, and LSTM models.")

#     data = load_aapl_data()
#     if data.empty:
#         st.stop()
#     st.subheader("📋 AAPL Stock Dataset")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Total Records", len(data))
#         st.metric("Date Range", f"{data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
#     with col2:
#         current_price = float(data['Close'].iloc[-1])
#         avg_price = float(data['Close'].tail(30).mean())
#         st.metric("Current Price", f"${current_price:.2f}")
#         st.metric("30-day Average", f"${avg_price:.2f}")
#     st.dataframe(data.tail(10), use_container_width=True)
#     forecast_days = 30  # Fixed per your project specs
#     st.subheader(f"🚀 Forecast Next {forecast_days} Days")
#     if st.button("Run Forecasting"):
#         with st.spinner("Running forecasting models..."):
#             split = len(data) - forecast_days
#             train, test = data[:split].copy(), data[split:].copy()
#             test_len = len(test)
#             forecasts = {}
#             metrics = []
#             # ARIMA
#             st.text("Running ARIMA...")
#             preds = build_arima_forecast(train, test_len)
#             forecasts["ARIMA"] = preds
#             metrics.append(("ARIMA", *calculate_performance_metrics(test['Close'], preds)))
#             # SARIMA
#             st.text("Running SARIMA...")
#             preds = build_sarima_forecast(train, test_len)
#             forecasts["SARIMA"] = preds
#             metrics.append(("SARIMA", *calculate_performance_metrics(test['Close'], preds)))
#             # LSTM
#             st.text("Running LSTM...")
#             preds = build_lstm_forecast(train, test_len)
#             forecasts["LSTM"] = preds
#             metrics.append(("LSTM", *calculate_performance_metrics(test['Close'], preds)))
#         # Display metrics
#         st.subheader("📊 Model Performance Metrics")
#         st.markdown("""
# - **RMSE**: Root Mean Squared Error (lower is better)  
# - **MAPE**: Mean Absolute Percentage Error (lower is better)  
# - **R² Score**: Coefficient of Determination (closer to 1 is better)  
# - **Accuracy**: % of correct direction predictions
# """)

#         df_metrics = pd.DataFrame(metrics, columns=["Model", "RMSE", "MAPE (%)", "R² Score", "Accuracy (%)"])
#         best_r2_idx = df_metrics["R² Score"].idxmax()
#         best_r2_model = df_metrics.loc[best_r2_idx, "Model"]
#         st.write(f"🏆 Best performing model: **{best_r2_model}** (highest R² Score)")
#         st.dataframe(df_metrics, use_container_width=True)
#         st.subheader("📈 Forecast Plot")
#         plot_forecasts(data, train, test, forecasts)

# if __name__ == "__main__":
#     main()


# app.py - Stock Market Time Series Analysis and Forecasting (Internship Ready)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Stock Market Forecasting - AAPL", page_icon="📈", layout="wide")

@st.cache_data
def load_aapl_data():
    data = yf.download("AAPL", start="2020-01-01", progress=False)
    if data.empty:
        st.error("Downloaded data is empty.")
        return pd.DataFrame()
    data.reset_index(inplace=True)
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_cols):
        st.error("Required columns missing from stock data.")
        return pd.DataFrame()
    data.dropna(inplace=True)
    data = data.sort_values('Date').reset_index(drop=True)
    data['Date'] = pd.to_datetime(data['Date'])  # Ensure Date is datetime
    return data

def calculate_performance_metrics(actual, predicted):
    try:
        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        mask = ~(np.isnan(actual) | np.isnan(predicted) | np.isinf(actual) | np.isinf(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        if len(actual_clean) == 0 or len(predicted_clean) == 0:
            return np.nan, np.nan, np.nan, np.nan
        rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
        mape = np.mean(np.abs((actual_clean - predicted_clean) / np.maximum(np.abs(actual_clean), 1e-8))) * 100
        r2 = r2_score(actual_clean, predicted_clean)
        if len(actual_clean) > 1:
            actual_diff = np.diff(actual_clean)
            predicted_diff = np.diff(predicted_clean)
            accuracy = np.mean(np.sign(actual_diff) == np.sign(predicted_diff)) * 100
        else:
            accuracy = np.nan
        return round(rmse, 2), round(mape, 2), round(r2, 4), round(accuracy, 1)
    except Exception as e:
        st.warning(f"Error calculating metrics: {str(e)}")
        return np.nan, np.nan, np.nan, np.nan

def build_arima_forecast(train, test_len):
    try:
        model = ARIMA(train['Close'], order=(5,1,0))
        fitted = model.fit()
        forecast = fitted.forecast(steps=test_len)
        return forecast.tolist()
    except Exception as e:
        st.warning(f"ARIMA error: {str(e)}")
        return [np.nan]*test_len

def build_sarima_forecast(train, test_len):
    try:
        model = SARIMAX(train['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=test_len)
        return forecast.tolist()
    except Exception as e:
        st.warning(f"SARIMA error: {str(e)}")
        return [np.nan]*test_len

def build_lstm_forecast(train, test_len):
    try:
        close = train['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close)
        seq_len = min(60, len(scaled) - 1)
        if seq_len < 10:
            return [np.nan]*test_len
        generator = TimeseriesGenerator(scaled, scaled, length=seq_len, batch_size=32)
        if len(generator) == 0:
            return [np.nan]*test_len
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(generator, epochs=10, verbose=0)
        last_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
        preds = []
        for _ in range(test_len):
            pred = model.predict(last_seq, verbose=0)[0][0]
            preds.append(pred)
            last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)
        return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten().tolist()
    except Exception as e:
        st.warning(f"LSTM error: {str(e)}")
        return [np.nan]*test_len

# -------------------- IMPROVED PLOTS: Subplots --------------------
def plot_forecasts(data, train, test, forecasts):
    try:
        model_names = list(forecasts.keys())
        n_models = len(model_names)
        fig = make_subplots(
            rows=n_models, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=[f"{name} Forecast vs Actual" for name in model_names]
        )

        for i, model_name in enumerate(model_names):
            preds = forecasts[model_name]
            row = i + 1

            # Show all actual (faint gray)
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Close'],
                name='Actual Price (Full)',
                line=dict(width=1, color='lightgray', dash='solid'),
                opacity=0.33,
                showlegend=(row==1)
            ), row=row, col=1)

            # Highlight test-period actual
            fig.add_trace(go.Scatter(
                x=test['Date'], y=test['Close'],
                name='Actual (Test)',
                line=dict(width=2, color='blue'),
                showlegend=(row==1)
            ), row=row, col=1)

            # Model predictions
            if preds is not None and len(preds) == len(test):
                fig.add_trace(go.Scatter(
                    x=test['Date'],
                    y=preds,
                    name=f'Predicted ({model_name})',
                    line=dict(color='orange', width=2, dash='dot'),
                    showlegend=(row==1)
                ), row=row, col=1)

            # Mark train/test split
            split_date = pd.to_datetime(train['Date'].iloc[-1])
            if hasattr(split_date, 'to_pydatetime'):
                split_date = split_date.to_pydatetime()
            fig.add_vline(
                x=split_date, line_dash="dash", line_color="gray", row=row, col=1
            )
            if row == 1:
                fig.add_annotation(
                    x=split_date,
                    y=max(data['Close']),
                    text="Train/Test Split",
                    showarrow=True,
                    arrowhead=1,
                    yshift=30,
                    font=dict(size=11),
                )

        fig.update_layout(
            title_text="Actual vs Predicted AAPL Prices (Each Model Separate)",
            height=350 * n_models,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode="x",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
# -------------------------------------------------------------------

def main():
    st.title("📈 TIME SERIES ANALYSIS AND FORECASTING FOR STOCK MARKET")
    st.markdown("**Apple Inc. (AAPL)** stock forecasting using ARIMA, SARIMA, and LSTM models.")

    data = load_aapl_data()
    if data.empty:
        st.stop()
    st.subheader("📋 AAPL Stock Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", len(data))
        st.metric("Date Range", f"{data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
    with col2:
        current_price = float(data['Close'].iloc[-1])
        avg_price = float(data['Close'].tail(30).mean())
        st.metric("Current Price", f"${current_price:.2f}")
        st.metric("30-day Average", f"${avg_price:.2f}")
    st.dataframe(data.tail(10), use_container_width=True)
    forecast_days = 30  # Fixed per your project specs
    st.subheader(f"🚀 Forecast Next {forecast_days} Days")
    if st.button("Run Forecasting"):
        with st.spinner("Running forecasting models..."):
            split = len(data) - forecast_days
            train, test = data[:split].copy(), data[split:].copy()
            test_len = len(test)
            forecasts = {}
            metrics = []
            # ARIMA
            st.text("Running ARIMA...")
            preds = build_arima_forecast(train, test_len)
            forecasts["ARIMA"] = preds
            metrics.append(("ARIMA", *calculate_performance_metrics(test['Close'], preds)))
            # SARIMA
            st.text("Running SARIMA...")
            preds = build_sarima_forecast(train, test_len)
            forecasts["SARIMA"] = preds
            metrics.append(("SARIMA", *calculate_performance_metrics(test['Close'], preds)))
            # LSTM
            st.text("Running LSTM...")
            preds = build_lstm_forecast(train, test_len)
            forecasts["LSTM"] = preds
            metrics.append(("LSTM", *calculate_performance_metrics(test['Close'], preds)))
        # Display metrics
        st.subheader("📊 Model Performance Metrics")
        st.markdown("""
- **RMSE**: Root Mean Squared Error (lower is better)  
- **MAPE**: Mean Absolute Percentage Error (lower is better)  
- **R² Score**: Coefficient of Determination (closer to 1 is better)  
- **Accuracy**: % of correct direction predictions
""")

        df_metrics = pd.DataFrame(metrics, columns=["Model", "RMSE", "MAPE (%)", "R² Score", "Accuracy (%)"])
        best_r2_idx = df_metrics["R² Score"].idxmax()
        best_r2_model = df_metrics.loc[best_r2_idx, "Model"]
        st.write(f"🏆 Best performing model: **{best_r2_model}** (highest R² Score)")
        st.dataframe(df_metrics, use_container_width=True)
        st.subheader("📈 Forecast Plot")
        plot_forecasts(data, train, test, forecasts)

if __name__ == "__main__":
    main()