# 📈 Enhanced High-Accuracy Stock Market Forecasting System
# Advanced implementation with improved models targeting >85% accuracy

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import mplfinance as mpf

# Statistical Models
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox

# Prophet with proper error handling
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# Advanced Machine Learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Deep Learning
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional, Conv1D, MaxPooling1D, Flatten, Input, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2

# Technical Analysis
from ta import add_all_ta_features
from ta.utils import dropna

# Utilities
import warnings
import gc
from datetime import datetime, timedelta
from itertools import product
import logging
import joblib
from scipy import stats
from sklearn.metrics import classification_report

# Configure settings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
logging.getLogger('prophet').setLevel(logging.WARNING)
sns.set_palette("husl")

# =============================================================================
# 🔧 ENHANCED UTILITY FUNCTIONS
# =============================================================================

def optimize_memory():
    """Enhanced memory optimization"""
    gc.collect()
    
def add_technical_indicators(df):
    """Add technical indicators without TA-Lib"""
    df = df.copy()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2*df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2*df['Close'].rolling(window=20).std()
    
    return df

def create_advanced_features(df):
    """Create advanced features for better prediction"""
    df = df.copy()
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag) if 'Volume' in df.columns else 0
    
    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'Close_Mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Close_Std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'Close_Min_{window}'] = df['Close'].rolling(window=window).min()
        df[f'Close_Max_{window}'] = df['Close'].rolling(window=window).max()
        df[f'Close_Median_{window}'] = df['Close'].rolling(window=window).median()
    
    # Time-based features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    df['IsQuarterEnd'] = df['Date'].dt.is_quarter_end.astype(int)
    
    # Cyclical encoding for time features
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    return df

def enhanced_stationarity_test(timeseries, title):
    """Enhanced stationarity testing with multiple methods"""
    print(f'\n🔍 Enhanced Stationarity Test for {title}:')
    
    # ADF Test
    adf_result = adfuller(timeseries, autolag='AIC')
    print('ADF Test Results:')
    print(f'ADF Statistic: {adf_result[0]:.6f}')
    print(f'p-value: {adf_result[1]:.6f}')
    
    # KPSS Test
    kpss_result = kpss(timeseries, regression='c')
    print('KPSS Test Results:')
    print(f'KPSS Statistic: {kpss_result[0]:.6f}')
    print(f'p-value: {kpss_result[1]:.6f}')
    
    # Determine stationarity
    is_stationary_adf = adf_result[1] <= 0.05
    is_stationary_kpss = kpss_result[1] > 0.05
    
    if is_stationary_adf and is_stationary_kpss:
        print("✅ Series is STATIONARY (both tests agree)")
        return True
    elif not is_stationary_adf and not is_stationary_kpss:
        print("❌ Series is NON-STATIONARY (both tests agree)")
        return False
    else:
        print("⚠️ Tests disagree - further investigation needed")
        return False

def calculate_advanced_metrics(actual, predicted):
    """Calculate comprehensive evaluation metrics"""
    # Handle any inf or nan values
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        return {
            'RMSE': float('inf'),
            'MAE': float('inf'),
            'MAPE': float('inf'),
            'R2': -float('inf'),
            'Accuracy': 0,
            'Direction_Accuracy': 0
        }
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
    mae = np.mean(np.abs(actual_clean - predicted_clean))
    mape = np.mean(np.abs((actual_clean - predicted_clean) / np.where(actual_clean != 0, actual_clean, 1))) * 100
    r2 = r2_score(actual_clean, predicted_clean)
    
    # Accuracy percentage (within 5% tolerance)
    tolerance = 0.05
    accuracy = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean) <= tolerance) * 100
    
    # Direction accuracy
    actual_direction = np.sign(np.diff(actual_clean))
    predicted_direction = np.sign(np.diff(predicted_clean))
    direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Accuracy': accuracy,
        'Direction_Accuracy': direction_accuracy
    }

# =============================================================================
# 📊 ENHANCED DATA LOADING AND PREPROCESSING
# =============================================================================

def load_comprehensive_data(ticker='AAPL', period='3y'):
    """Load comprehensive stock data with all OHLCV information"""
    print(f"📥 Loading comprehensive data for {ticker}...")
    
    try:
        # Download full OHLCV data
        data = yf.download(ticker, period=period, progress=False)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Clean column names if MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        # Reset index to make Date a column
        data = data.reset_index()
        data = data.dropna()
        
        # Add technical indicators
        data = add_technical_indicators(data)
        
        # Add advanced features
        data = create_advanced_features(data)
        
        # Remove any remaining NaN values
        data = data.dropna()
        
        print(f"✅ Successfully loaded {len(data)} data points with {len(data.columns)} features")
        print(f"📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_advanced_train_test_split(df, test_size=0.15, val_size=0.15):
    """Create train/validation/test splits for time series"""
    n = len(df)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    
    train_data = df[:train_end].copy()
    val_data = df[train_end:val_end].copy()
    test_data = df[val_end:].copy()
    
    print(f"🔄 Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data

# =============================================================================
# 🤖 ENHANCED MODEL IMPLEMENTATIONS
# =============================================================================

class EnhancedARIMA:
    """Enhanced ARIMA with comprehensive parameter optimization"""
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.scaler = StandardScaler()
        
    def auto_arima_advanced(self, train_data, max_p=5, max_d=2, max_q=5):
        """Advanced ARIMA parameter selection with multiple criteria"""
        print("🔍 Advanced ARIMA parameter optimization...")
        
        # Scale the data for better convergence
        scaled_data = self.scaler.fit_transform(train_data[['Close']])
        scaled_series = pd.Series(scaled_data.flatten(), index=train_data.index)
        
        best_aic = float('inf')
        best_bic = float('inf')
        best_params = None
        candidates = []
        
        # Grid search with information criteria
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if p == 0 and d == 0 and q == 0:
                        continue
                        
                    try:
                        model = ARIMA(scaled_series, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        candidates.append({
                            'order': (p, d, q),
                            'aic': fitted_model.aic,
                            'bic': fitted_model.bic,
                            'model': fitted_model
                        })
                        
                    except:
                        continue
        
        if candidates:
            # Select best model based on AIC
            best_candidate = min(candidates, key=lambda x: x['aic'])
            self.best_params = best_candidate['order']
            self.fitted_model = best_candidate['model']
            
            print(f"✅ Best ARIMA parameters: {self.best_params}")
            print(f"📊 AIC: {best_candidate['aic']:.2f}, BIC: {best_candidate['bic']:.2f}")
        
        return self.best_params
    
    def fit(self, train_data):
        """Fit enhanced ARIMA model"""
        self.auto_arima_advanced(train_data)
        return self
    
    def predict(self, steps):
        """Make scaled predictions"""
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet")
        
        forecast = self.fitted_model.forecast(steps=steps)
        
        # Inverse transform
        forecast_reshaped = forecast.values.reshape(-1, 1)
        forecast_original = self.scaler.inverse_transform(forecast_reshaped).flatten()
        
        return forecast_original

class EnhancedSARIMA:
    """Enhanced SARIMA with seasonal pattern detection"""
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.best_seasonal = None
        self.scaler = RobustScaler()
        
    def detect_seasonality(self, data, max_periods=[7, 12, 24, 52]):
        """Detect seasonal patterns in the data"""
        best_seasonal_strength = 0
        best_period = 12
        
        for period in max_periods:
            if len(data) > 2 * period:
                try:
                    decomposition = seasonal_decompose(data, model='additive', period=period)
                    seasonal_strength = np.var(decomposition.seasonal) / np.var(data)
                    
                    if seasonal_strength > best_seasonal_strength:
                        best_seasonal_strength = seasonal_strength
                        best_period = period
                except:
                    continue
        
        print(f"🔍 Detected seasonal period: {best_period} (strength: {best_seasonal_strength:.4f})")
        return best_period
    
    def auto_sarima_advanced(self, train_data):
        """Advanced SARIMA parameter optimization"""
        print("🔍 Advanced SARIMA parameter optimization...")
        
        # Scale data
        scaled_data = self.scaler.fit_transform(train_data[['Close']])
        scaled_series = pd.Series(scaled_data.flatten(), index=train_data.index)
        
        # Detect seasonality
        seasonal_period = self.detect_seasonality(scaled_series)
        
        best_aic = float('inf')
        best_params = None
        best_seasonal = None
        
        # Focused parameter search
        p_values = [0, 1, 2]
        d_values = [0, 1]
        q_values = [0, 1, 2]
        P_values = [0, 1]
        D_values = [0, 1]
        Q_values = [0, 1]
        
        for p, d, q in product(p_values, d_values, q_values):
            for P, D, Q in product(P_values, D_values, Q_values):
                if p == 0 and d == 0 and q == 0 and P == 0 and D == 0 and Q == 0:
                    continue
                    
                try:
                    model = SARIMAX(scaled_series, 
                                  order=(p, d, q),
                                  seasonal_order=(P, D, Q, seasonal_period),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
                    fitted_model = model.fit(disp=False, maxiter=150)
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        best_seasonal = (P, D, Q, seasonal_period)
                        
                except:
                    continue
        
        self.best_params = best_params
        self.best_seasonal = best_seasonal
        
        print(f"✅ Best SARIMA: {best_params}x{best_seasonal} (AIC: {best_aic:.2f})")
        
        return best_params, best_seasonal
    
    def fit(self, train_data):
        """Fit enhanced SARIMA model"""
        self.auto_sarima_advanced(train_data)
        
        scaled_data = self.scaler.transform(train_data[['Close']])
        scaled_series = pd.Series(scaled_data.flatten(), index=train_data.index)
        
        self.model = SARIMAX(scaled_series, 
                           order=self.best_params,
                           seasonal_order=self.best_seasonal,
                           enforce_stationarity=False,
                           enforce_invertibility=False)
        self.fitted_model = self.model.fit(disp=False, maxiter=200)
        
        return self
    
    def predict(self, steps):
        """Make enhanced predictions"""
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet")
        
        forecast = self.fitted_model.forecast(steps=steps)
        
        # Inverse transform
        forecast_reshaped = forecast.values.reshape(-1, 1)
        forecast_original = self.scaler.inverse_transform(forecast_reshaped).flatten()
        
        return forecast_original

class EnhancedProphet:
    """Enhanced Prophet with regressors and log transformation"""
    def __init__(self, changepoint_prior_scale=0.3):
        self.model = None
        self.scaler = None
        self.use_log = True
        self.changepoint_prior_scale = changepoint_prior_scale
        self.regressors = ['volume', 'ma7', 'ma30', 'ema15']
    
    def prepare_features(self, df):
        df = df.copy()
        df['volume'] = np.log(df['Volume'] + 1)
        df['y'] = np.log(df['Close']) if self.use_log else df['Close']
        df['ma7'] = df['y'].rolling(window=7, min_periods=1).mean()
        df['ma30'] = df['y'].rolling(window=30, min_periods=1).mean()
        df['ema15'] = df['y'].ewm(span=15, adjust=False).mean()
        df = df[['Date', 'y'] + self.regressors].dropna()
        df.rename(columns={'Date': 'ds'}, inplace=True)
        return df

    def fit(self, train_data):
        print("🔍 Fitting Enhanced Prophet with regressors...")
        train_df = self.prepare_features(train_data)
        
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode='multiplicative',
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True
        )
        
        for reg in self.regressors:
            self.model.add_regressor(reg)

        self.model.fit(train_df)
        self.train_df = train_df  # Save for future reference
        return self

    def predict(self, steps, last_date):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        full_df = self.train_df.copy()
        last_known_date = full_df['ds'].max()
        future_dates = pd.date_range(start=last_known_date + timedelta(days=1), periods=steps, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Copy last known regressor values for simplicity
        for reg in self.regressors:
            last_value = full_df[reg].iloc[-1]
            future_df[reg] = last_value

        all_future = pd.concat([full_df[['ds'] + self.regressors], future_df], ignore_index=True)
        forecast = self.model.predict(all_future)

        yhat = np.exp(forecast['yhat'].values[-steps:]) if self.use_log else forecast['yhat'].values[-steps:]
        return yhat

class HybridLSTM:
    def __init__(self, lookback=60, feature_columns=None):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lookback = lookback
        self.feature_columns = feature_columns or ['Close']
        
    def prepare_data(self, data):
        """Prepare data with multiple features"""
        feature_data = data[self.feature_columns].values
        scaled_data = self.scaler.fit_transform(feature_data)
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build hybrid CNN-LSTM model"""
        input_layer = Input(shape=input_shape)
        
        # CNN layers
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(0.2)(conv1)
        
        # LSTM layers
        lstm1 = Bidirectional(LSTM(100, return_sequences=True))(conv1)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(0.3)(lstm1)
        
        lstm2 = Bidirectional(LSTM(50, return_sequences=False))(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = Dropout(0.3)(lstm2)
        
        # Dense layers
        dense1 = Dense(50, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(lstm2)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        
        output = Dense(1, activation='linear')(dense1)
        
        model = Model(inputs=input_layer, outputs=output)
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
        
        return model
    
    def fit(self, train_data, val_data=None):
        """Train model"""
        print("Training Enhanced Hybrid CNN-LSTM model...")
        
        X_train, y_train = self.prepare_data(train_data)
        
        X_val, y_val = None, None
        if val_data is not None:
            val_feature_data = val_data[self.feature_columns].values
            scaled_val_data = self.scaler.transform(val_feature_data)
            
            X_val, y_val = [], []
            for i in range(self.lookback, len(scaled_val_data)):
                X_val.append(scaled_val_data[i-self.lookback:i])
                y_val.append(scaled_val_data[i, 0])
            
            X_val, y_val = np.array(X_val), np.array(y_val)
        
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(monitor='val_loss' if val_data else 'loss', 
                         patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if val_data else 'loss', 
                             factor=0.3, patience=10, min_lr=1e-7),
            ModelCheckpoint('best_model.h5', save_best_only=True, 
                          monitor='val_loss' if val_data else 'loss')
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if val_data is not None else None,
            batch_size=32,
            epochs=100,
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )
        
        return self
    
    def predict(self, train_data, steps):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        train_features = train_data[self.feature_columns].values
        scaled_data = self.scaler.transform(train_features)
        
        last_sequence = scaled_data[-self.lookback:].reshape(1, self.lookback, len(self.feature_columns))
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            pred = self.model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(pred)
            
            new_row = current_sequence[0, -1, :].copy()
            new_row[0] = pred
            
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = new_row
        
        predictions = np.array(predictions).reshape(-1, 1)
        dummy_features = np.zeros((len(predictions), len(self.feature_columns)))
        dummy_features[:, 0] = predictions.flatten()
        
        predictions_original = self.scaler.inverse_transform(dummy_features)[:, 0]
        
        return predictions_original

# =============================================================================
# 📊 ADVANCED VISUALIZATION FUNCTIONS
# =============================================================================

def create_candlestick_chart(df, ticker, predictions_dict=None):
    """Create interactive candlestick chart with predictions"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{ticker} - Price & Predictions', 'Volume', 'Technical Indicators'),
        row_weights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add moving averages
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_20'], 
                      name='SMA 20', line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_50'], 
                      name='SMA 50', line=dict(color='blue', width=1)),
            row=1, col=1
        )
    
    # Add predictions if available
    if predictions_dict:
        for model_name, predictions_data in predictions_dict.items():
            if not predictions_data.empty:
                fig.add_trace(
                    go.Scatter(x=predictions_data['Date'], y=predictions_data['Predicted_Close'],
                              mode='lines', name=f'{model_name} Predictions',
                              line=dict(width=2, dash='dot')),
                    row=1, col=1
                )

    # Volume chart
    if 'Volume' in df.columns:
        fig.add_trace(
            go.Bar(x=df['Date'], y=df['Volume'], name='Volume', showlegend=False),
            row=2, col=1
        )
    
    # MACD on third subplot
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MACD'], 
                      name='MACD', line=dict(color='purple', width=1)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MACD_Signal'], 
                      name='MACD Signal', line=dict(color='green', width=1, dash='dot')),
            row=3, col=1
        )
        # MACD Histogram
        colors = ['red' if val < 0 else 'green' for val in df['MACD_Histogram']]
        fig.add_trace(
            go.Bar(x=df['Date'], y=df['MACD_Histogram'], name='MACD Hist',
                   marker_color=colors, showlegend=False),
            row=3, col=1
        )
    
    fig.update_layout(
        title_text=f'Stock Analysis for {ticker}',
        xaxis_rangeslider_visible=False,
        height=900,
        template='plotly_dark'
    )
    
    fig.show()

# =============================================================================
# 🚀 ENHANCED FORECASTING PIPELINE
# =============================================================================

def run_enhanced_forecasting_pipeline(ticker='AAPL', period='3y', forecast_steps=30):
    """Run the complete enhanced stock forecasting pipeline"""
    print(f"🚀 Starting enhanced forecasting pipeline for {ticker}...")
    
    # Load and preprocess data
    df = load_comprehensive_data(ticker, period)
    if df is None or df.empty:
        print("Pipeline aborted due to data loading error.")
        return
    
    train_data, val_data, test_data = create_advanced_train_test_split(df)
    
    # Initialize and train models
    models = {
        'ARIMA': EnhancedARIMA(),
        'SARIMA': EnhancedSARIMA(),
        'Prophet': EnhancedProphet(),
        'Hybrid_LSTM': HybridLSTM(lookback=60, feature_columns=[
            'Close', 'Volume', 'SMA_20', 'RSI', 'MACD', 'BB_Middle', 'Price_Change', 'DayOfWeek_Sin', 'Month_Sin'
        ])
    }
    
    predictions = {}
    
    for name, model in models.items():
        print(f"\n✨ Training {name} model...")
        try:
            if name == 'Prophet':
                model.fit(train_data)
                last_date = train_data['Date'].max()
                preds = model.predict(forecast_steps, last_date)
            elif name == 'Hybrid_LSTM':
                # Ensure validation data is passed for LSTM
                model.fit(train_data, val_data)
                preds = model.predict(train_data, forecast_steps)
            else:
                model.fit(train_data)
                preds = model.predict(forecast_steps)
            
            # Create a DataFrame for predictions
            last_training_date = df['Date'].max()
            forecast_dates = pd.date_range(start=last_training_date + timedelta(days=1), periods=forecast_steps, freq='D')
            
            predictions[name] = pd.DataFrame({
                'Date': forecast_dates,
                'Predicted_Close': preds
            })
            
            print(f"✅ {name} predictions generated.")
            
        except Exception as e:
            print(f"❌ Error training/predicting with {name}: {e}")
            predictions[name] = pd.DataFrame() # Empty dataframe if error
            
    # Combine predictions and plot
    all_predictions_df = df.copy()
    for name, pred_df in predictions.items():
        if not pred_df.empty:
            all_predictions_df = pd.merge(
                all_predictions_df, 
                pred_df.rename(columns={'Predicted_Close': f'{name}_Predicted_Close'}),
                on='Date', how='outer'
            )
            
    # Display performance metrics (for historical predictions)
    print("\n📈 Model Performance (on historical data):")
    for name, model in models.items():
        if not predictions[name].empty: # Only if model successfully ran
            # For simplicity, let's evaluate on the test set if available,
            # or last part of training data if not explicitly set for test
            
            # For Prophet, we get forecast for future, not historical
            # For LSTMs, the prediction function is for future steps
            # So, for performance, we re-predict on historical data if possible
            # or rely on model-specific internal metrics if available.
            # For this simplified example, we'll just show for ARIMA/SARIMA
            
            if name in ['ARIMA', 'SARIMA']:
                try:
                    # Make predictions on test data
                    test_steps = len(test_data)
                    historical_preds_scaled = model.fitted_model.get_forecast(steps=test_steps).predicted_mean
                    historical_preds = model.scaler.inverse_transform(historical_preds_scaled.values.reshape(-1, 1)).flatten()
                    
                    metrics = calculate_advanced_metrics(test_data['Close'].values, historical_preds)
                    print(f"- {name}: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%, R2={metrics['R2']:.2f}, Direction Acc={metrics['Direction_Accuracy']:.2f}%")
                except Exception as e:
                    print(f"- {name}: Error calculating historical metrics - {e}")
            elif name == 'Hybrid_LSTM':
                # This would require re-running prediction on test data which is complex for sequential models.
                # For a quick overview, we'll skip historical metrics for LSTM in this simplified view.
                print(f"- {name}: Historical performance calculation is more complex for this model type. Consider separate evaluation.")
            elif name == 'Prophet':
                 print(f"- {name}: Prophet evaluation typically uses cross-validation on historical data (not implemented in this simplified view).")
    
    # Visualize results including predictions
    print("\n📊 Generating visualization...")
    
    # Filter df to include only historical data for candlestick
    display_df = df.copy()
    
    # Prepare predictions_dict for plotting
    plot_predictions_dict = {}
    for name, pred_df in predictions.items():
        if not pred_df.empty:
            plot_predictions_dict[name] = pred_df
            
    create_candlestick_chart(display_df, ticker, predictions_dict=plot_predictions_dict)
    
    print("\n✅ Forecasting pipeline completed.")

# Example usage
if __name__ == "__main__":
    run_enhanced_forecasting_pipeline(ticker='GOOG', period='2y', forecast_steps=60)