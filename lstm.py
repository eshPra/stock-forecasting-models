import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Input, concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.regularizers import l1_l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Streamlit page config
st.set_page_config(layout="wide", page_title="Advanced LSTM Stock Forecasting")
st.title("🧠 Advanced Stock Market Forecasting with LSTM")

if not TENSORFLOW_AVAILABLE:
    st.error("TensorFlow is required for LSTM model. Please install it using: pip install tensorflow")
    st.stop()

# User inputs
col1, col2 = st.columns(2)
with col1:
    ticker = st.selectbox("Choose a stock ticker", ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX"], index=0)
    forecast_days = st.slider("Forecast period (days)", 30, 120, 60, step=10)

with col2:
    start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))

# LSTM parameters
st.sidebar.header("🔧 Advanced LSTM Parameters")
sequence_length = st.sidebar.slider("Sequence Length (lookback window)", 30, 120, 60, step=10)
lstm_units = st.sidebar.slider("LSTM Units", 32, 256, 128, step=32)
epochs = st.sidebar.slider("Training Epochs", 20, 200, 100, step=10)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.2, step=0.05)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.0005, 0.0001], index=0)

# Advanced options
st.sidebar.header("🚀 Advanced Options")
use_bidirectional = st.sidebar.checkbox("Use Bidirectional LSTM", value=True)
use_technical_indicators = st.sidebar.checkbox("Use Technical Indicators", value=True)
validation_split = st.sidebar.slider("Validation Split", 0.1, 0.3, 0.2, step=0.05)

@st.cache_data(show_spinner=False)
def load_data(ticker, start_date, end_date):
    """Load and preprocess stock data with advanced features"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error(f"No data found for {ticker}")
            return None
            
        df = data[['Close', 'Volume', 'High', 'Low', 'Open']].reset_index()
        df.columns = ['ds', 'close', 'volume', 'high', 'low', 'open']
        
        # Advanced feature engineering
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['volatility'] = (df['high'] - df['low']) / df['close']
        df['price_range'] = (df['high'] - df['low']) / df['open']
        
        # Moving averages
        for window in [7, 14, 21, 30, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Technical indicators
        if use_technical_indicators:
            df['rsi'] = calculate_rsi(df['close'])
            df['macd'], df['macd_signal'] = calculate_macd(df['close'])
            df['bb_upper'], df['bb_lower'], df['bb_middle'] = calculate_bollinger_bands(df['close'])
            df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
            df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])
        
        # Price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Time-based features
        df['day_of_week'] = pd.to_datetime(df['ds']).dt.dayofweek
        df['month'] = pd.to_datetime(df['ds']).dt.month
        df['quarter'] = pd.to_datetime(df['ds']).dt.quarter
        
        # Cyclical encoding
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        df = df.dropna()
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal line"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    middle = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, lower, middle

def calculate_atr(high, low, close, window=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def create_sequences(data, target_col, sequence_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i, target_col])
    return np.array(X), np.array(y)

def build_advanced_lstm_model(input_shape, lstm_units, dropout_rate, use_bidirectional=True):
    """Build advanced LSTM model architecture"""
    inputs = Input(shape=input_shape)
    
    if use_bidirectional:
        # Bidirectional LSTM layers
        lstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(dropout_rate)(lstm1)
        
        lstm2 = Bidirectional(LSTM(lstm_units // 2, return_sequences=True))(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = Dropout(dropout_rate)(lstm2)
        
        lstm3 = Bidirectional(LSTM(lstm_units // 4, return_sequences=False))(lstm2)
        lstm3 = BatchNormalization()(lstm3)
        lstm3 = Dropout(dropout_rate)(lstm3)
    else:
        # Regular LSTM layers
        lstm1 = LSTM(lstm_units, return_sequences=True)(inputs)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(dropout_rate)(lstm1)
        
        lstm2 = LSTM(lstm_units // 2, return_sequences=True)(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = Dropout(dropout_rate)(lstm2)
        
        lstm3 = LSTM(lstm_units // 4, return_sequences=False)(lstm2)
        lstm3 = BatchNormalization()(lstm3)
        lstm3 = Dropout(dropout_rate)(lstm3)
    
    # Dense layers with regularization
    dense1 = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(lstm3)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(dropout_rate)(dense1)
    
    dense2 = Dense(32, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(dropout_rate/2)(dense2)
    
    outputs = Dense(1, activation='linear')(dense2)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='huber',  # More robust to outliers
        metrics=['mae', 'mse']
    )
    return model

def calculate_advanced_metrics(actual, predicted):
    """Calculate comprehensive evaluation metrics"""
    try:
        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()
        
        # Remove any NaN or inf values
        mask = ~(np.isnan(actual) | np.isnan(predicted) | np.isinf(actual) | np.isinf(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return None
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
        mae = mean_absolute_error(actual_clean, predicted_clean)
        mape = np.mean(np.abs((actual_clean - predicted_clean) / np.maximum(np.abs(actual_clean), 1e-8))) * 100
        r2 = r2_score(actual_clean, predicted_clean)
        
        # Direction accuracy
        if len(actual_clean) > 1:
            actual_direction = np.sign(np.diff(actual_clean))
            predicted_direction = np.sign(np.diff(predicted_clean))
            direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
        else:
            direction_accuracy = 0
        
        # Additional metrics
        mse = mean_squared_error(actual_clean, predicted_clean)
        mae_percentage = (mae / np.mean(actual_clean)) * 100
        
        return {
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'MAPE': round(mape, 2),
            'R2': round(r2, 4),
            'Direction_Accuracy': round(direction_accuracy, 2),
            'MSE': round(mse, 4),
            'MAE_Percentage': round(mae_percentage, 2)
        }
    except Exception as e:
        st.warning(f"Error calculating metrics: {str(e)}")
        return None

def create_comprehensive_plots(df, train_predictions, test_predictions, future_predictions, 
                              train_dates, test_dates, future_dates, scaler, target_col_idx):
    """Create comprehensive visualization plots"""
    
    # Inverse transform predictions
    def inverse_transform_predictions(predictions, scaler, target_col_idx, n_features):
        dummy = np.zeros((len(predictions), n_features))
        dummy[:, target_col_idx] = predictions.flatten()
        return scaler.inverse_transform(dummy)[:, target_col_idx]
    
    n_features = scaler.n_features_in_
    
    # Inverse transform all predictions
    train_pred_original = inverse_transform_predictions(train_predictions, scaler, target_col_idx, n_features)
    test_pred_original = inverse_transform_predictions(test_predictions, scaler, target_col_idx, n_features)
    future_pred_original = inverse_transform_predictions(future_predictions, scaler, target_col_idx, n_features)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price Forecast', 'Training Performance', 'Test Performance'),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # Plot 1: Complete forecast
    fig.add_trace(
        go.Scatter(x=df['ds'], y=df['close'], name='Actual Price',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=train_dates, y=train_pred_original, name='Train Predictions',
                  line=dict(color='green', width=2, dash='dot')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=test_dates, y=test_pred_original, name='Test Predictions',
                  line=dict(color='orange', width=2, dash='dot')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=future_dates, y=future_pred_original, name='Future Forecast',
                  line=dict(color='red', width=3, dash='dash')),
        row=1, col=1
    )
    
    # Plot 2: Training performance
    fig.add_trace(
        go.Scatter(x=train_dates, y=df['close'].iloc[sequence_length:len(train_predictions)+sequence_length], 
                  name='Actual (Train)', line=dict(color='blue', width=2)),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=train_dates, y=train_pred_original, name='Predicted (Train)',
                  line=dict(color='green', width=2)),
        row=2, col=1
    )
    
    # Plot 3: Test performance
    fig.add_trace(
        go.Scatter(x=test_dates, y=df['close'].iloc[-len(test_predictions):], 
                  name='Actual (Test)', line=dict(color='blue', width=2)),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=test_dates, y=test_pred_original, name='Predicted (Test)',
                  line=dict(color='orange', width=2)),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title_text=f"Advanced LSTM Forecast for {ticker}",
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=3, col=1)
    
    return fig

if st.button("🔮 Run Advanced LSTM Forecast", type="primary"):
    with st.spinner("Loading data and preparing model..."):
        df = load_data(ticker, start_date, end_date)
        
        if df is None:
            st.stop()
        
        # Feature selection
        if use_technical_indicators:
            feature_cols = ['close', 'volume', 'price_change', 'volatility', 'ma_7', 'ma_30', 
                           'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr', 'momentum_10', 
                           'volume_ratio', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
        else:
            feature_cols = ['close', 'volume', 'price_change', 'volatility', 'ma_7', 'ma_30', 
                           'momentum_10', 'volume_ratio', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
        
        # Remove any columns that don't exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        data = df[feature_cols].values
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Target column index (close price)
        target_col_idx = feature_cols.index('close')
        
        # Split data
        train_size = len(scaled_data) - forecast_days
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - sequence_length:]
        
        # Create sequences
        X_train, y_train = create_sequences(train_data, target_col_idx, sequence_length)
        X_test, y_test = create_sequences(test_data, target_col_idx, sequence_length)
        
        # Display data info
        st.subheader("📊 Data Information")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Training Samples", len(X_train))
        with col3:
            st.metric("Test Samples", len(X_test))
        with col4:
            st.metric("Features", len(feature_cols))
        
        # Build and train model
        st.subheader("🤖 Model Training")
        model = build_advanced_lstm_model(
            (sequence_length, len(feature_cols)), 
            lstm_units, 
            dropout_rate, 
            use_bidirectional
        )
        
        # Display model summary
        with st.expander("Model Architecture"):
            model.summary(print_fn=st.text)
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Advanced callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
            ModelCheckpoint('best_lstm_model.h5', save_best_only=True, monitor='val_loss')
        ]
        
        # Custom callback for progress updates
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f'Epoch {epoch + 1}/{epochs} - Loss: {logs["loss"]:.6f} - Val Loss: {logs["val_loss"]:.6f}')
        
        callbacks.append(StreamlitCallback())
        
        # Train the model
        with st.spinner("Training advanced LSTM model..."):
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=0,
                callbacks=callbacks,
                shuffle=False  # Important for time series
            )
        
        progress_bar.progress(1.0)
        status_text.text("✅ Training completed!")
        
        # Make predictions
        train_predictions = model.predict(X_train, verbose=0)
        test_predictions = model.predict(X_test, verbose=0)
        
        # Future forecasting
        st.subheader("🔮 Future Forecasting")
        last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, len(feature_cols))
        future_predictions = []
        
        for _ in range(forecast_days):
            # Predict next value
            next_pred = model.predict(last_sequence, verbose=0)[0, 0]
            future_predictions.append(next_pred)
            
            # Update sequence for next prediction
            new_row = last_sequence[0, -1, :].copy()
            new_row[target_col_idx] = next_pred
            
            # Shift sequence and add new row
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, :] = new_row
        
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        
        # Calculate metrics
        train_metrics = calculate_advanced_metrics(
            df['close'].iloc[sequence_length:len(train_predictions)+sequence_length], 
            train_predictions
        )
        test_metrics = calculate_advanced_metrics(
            df['close'].iloc[-len(test_predictions):], 
            test_predictions
        )
        
        # Display metrics
        st.subheader("📈 Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Performance**")
            if train_metrics:
                for metric, value in train_metrics.items():
                    st.metric(metric, value)
        
        with col2:
            st.markdown("**Test Performance**")
            if test_metrics:
                for metric, value in test_metrics.items():
                    st.metric(metric, value)
        
        # Create date ranges for plotting
        train_dates = df['ds'].iloc[sequence_length:len(train_predictions)+sequence_length]
        test_dates = df['ds'].iloc[-len(test_predictions):]
        future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), 
                                   periods=forecast_days, freq='D')
        
        # Create comprehensive plots
        st.subheader("📊 Advanced Visualizations")
        fig = create_comprehensive_plots(
            df, train_predictions, test_predictions, future_predictions,
            train_dates, test_dates, future_dates, scaler, target_col_idx
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Training history
        st.subheader("📈 Training History")
        history_df = pd.DataFrame(history.history)
        
        fig_history = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'MAE'))
        
        fig_history.add_trace(
            go.Scatter(y=history_df['loss'], name='Training Loss'),
            row=1, col=1
        )
        fig_history.add_trace(
            go.Scatter(y=history_df['val_loss'], name='Validation Loss'),
            row=1, col=1
        )
        
        fig_history.add_trace(
            go.Scatter(y=history_df['mae'], name='Training MAE'),
            row=1, col=2
        )
        fig_history.add_trace(
            go.Scatter(y=history_df['val_mae'], name='Validation MAE'),
            row=1, col=2
        )
        
        fig_history.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_history, use_container_width=True)
        
        # Future forecast summary
        st.subheader("🔮 Future Forecast Summary")
        future_pred_original = scaler.inverse_transform(
            np.zeros((len(future_predictions), len(feature_cols)))
        )
        future_pred_original[:, target_col_idx] = future_predictions.flatten()
        future_prices = future_pred_original[:, target_col_idx]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${df['close'].iloc[-1]:.2f}")
        with col2:
            st.metric("Predicted End Price", f"${future_prices[-1]:.2f}")
        with col3:
            price_change = ((future_prices[-1] - df['close'].iloc[-1]) / df['close'].iloc[-1]) * 100
            st.metric("Expected Change", f"{price_change:+.2f}%")
        with col4:
            st.metric("Forecast Period", f"{forecast_days} days")
        
        # Download predictions
        st.subheader("📥 Download Results")
        results_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_prices,
            'Current_Price': df['close'].iloc[-1]
        })
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Future Predictions",
            data=csv,
            file_name=f"{ticker}_lstm_forecast_{forecast_days}days.csv",
            mime="text/csv"
        )

# Add some helpful information
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 Model Information
- **LSTM**: Long Short-Term Memory neural network
- **Bidirectional**: Processes sequences in both directions
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR
- **Advanced Features**: Price momentum, volume analysis, time encoding

### 💡 Tips for Better Results
- Increase sequence length for longer patterns
- Use more LSTM units for complex patterns
- Enable technical indicators for better accuracy
- Adjust learning rate if training is unstable
""")