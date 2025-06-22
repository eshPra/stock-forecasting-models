# 📊 Stock Market Time Series Forecasting

> 🔬 **One-Month Internship Project** | 📈 Powered by Data | 🧠 Built with ML + DL + Python

---

## 🌟 Project Highlights

This project focuses on **analyzing and forecasting stock market trends** using a variety of **time series models**.  
We explored both **statistical** and **deep learning** techniques to make accurate stock predictions.

### 🧪 Models Implemented:
- 🔢 **ARIMA** – AutoRegressive Integrated Moving Average  
- 🔁 **SARIMA** – Seasonal ARIMA  
- 🔮 **Prophet** – Facebook's powerful time series model  
- 🧬 **LSTM** – Deep Learning model using Long Short-Term Memory networks

All models are wrapped into **interactive dashboards** using `Streamlit`.

---

## 🛠️ Tech Stack & Tools Used

- **Language & Frameworks:** Python, Streamlit, TensorFlow/Keras  
- **Libraries:**
  - 📦 `pandas`, `numpy`, `yfinance`  
  - 📊 `matplotlib`, `plotly`, `seaborn`  
  - 📈 `scikit-learn`, `statsmodels`, `prophet`  
  - 🤖 `tensorflow`, `keras`  

---

## 📁 Project Structure

```
📦 PROJECT/
├── arima.py        # ARIMA model implementation
├── sarima.py       # SARIMA model implementation
├── prophet.py      # Prophet model implementation
├── lstm.py         # Advanced LSTM model app
└── README.md       # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd Internship
```

### 2. Activate Virtual Environment
```bash
# Activate the existing virtual environment
./stock_env/Scripts/activate
```

### 3. Install Dependencies
```bash
pip install streamlit pandas numpy yfinance matplotlib plotly seaborn scikit-learn statsmodels prophet tensorflow
```

### 4. Run the Applications

#### Main Forecasting App (ARIMA, SARIMA, Prophet)
```bash
streamlit run time_series_forecasting_project/NEW/app.py
```

#### Enhanced Forecasting System
```bash
streamlit run time_series_forecasting_project/NEW/app1.py
```

#### Advanced LSTM Model
```bash
streamlit run time_series_forecasting_project/NEW/PROJECT/lstm.py
```

#### Individual Model Apps
```bash
# ARIMA Model
streamlit run time_series_forecasting_project/NEW/PROJECT/arima.py

# SARIMA Model  
streamlit run time_series_forecasting_project/NEW/PROJECT/sarima.py

# Prophet Model
streamlit run time_series_forecasting_project/NEW/PROJECT/prophet.py
```

---

## 🧪 Virtual Environment Setup

To maintain isolation and reproducibility, this project uses a **Python virtual environment**.

### 🔧 Setup Instructions:

```bash
# Create virtual environment
python -m venv stock_env

# Activate the environment
source stock_env/bin/activate        # On Windows: stock_env\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 📄 Required Libraries (`requirements.txt`)

```txt
streamlit
pandas
numpy
yfinance
matplotlib
plotly
seaborn
scikit-learn
statsmodels
prophet
tensorflow
```

---

## 📊 Features

### 🔢 **ARIMA Model**
- AutoRegressive Integrated Moving Average
- Automatic parameter optimization
- Performance metrics and visualization
- Real-time stock data integration
- Interactive parameter tuning

### 🔁 **SARIMA Model**
- Seasonal ARIMA with seasonal pattern detection
- Advanced parameter optimization
- Comprehensive evaluation metrics
- Seasonal decomposition analysis
- Trend and seasonality identification

### 🔮 **Prophet Model**
- Facebook's time series forecasting tool
- Automatic seasonality detection
- Holiday effects and trend analysis
- Uncertainty quantification
- Custom seasonality patterns

### 🧬 **LSTM Model**
- Advanced deep learning architecture
- Bidirectional LSTM layers
- Technical indicators integration
- Comprehensive feature engineering
- Real-time training progress
- Future forecasting capabilities
- Advanced callbacks and regularization

---

## 📈 Model Performance

All models provide comprehensive evaluation metrics:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R² Score** (Coefficient of Determination)
- **Direction Accuracy** (Trend prediction accuracy)

### 🎯 Performance Comparison:
- **LSTM**: Best for complex patterns and long-term forecasting
- **Prophet**: Excellent for seasonal data and trend analysis
- **SARIMA**: Good for seasonal time series
- **ARIMA**: Effective for stationary time series

---

## 🎯 Key Features

✅ **Interactive Dashboards** - User-friendly Streamlit interfaces  
✅ **Real-time Data** - Live stock data from Yahoo Finance  
✅ **Multiple Models** - Compare different forecasting approaches  
✅ **Advanced Visualizations** - Interactive plots with Plotly  
✅ **Performance Metrics** - Comprehensive model evaluation  
✅ **Future Forecasting** - Predict stock prices for specified periods  
✅ **Download Results** - Export predictions as CSV files  
✅ **Technical Indicators** - RSI, MACD, Bollinger Bands, ATR  
✅ **Feature Engineering** - Advanced data preprocessing  
✅ **Model Comparison** - Side-by-side performance analysis  

---

## 🔧 Advanced Features

### LSTM Model Enhancements:
- **Bidirectional LSTM** for better pattern recognition
- **Batch Normalization** for stable training
- **Advanced Callbacks** (Early Stopping, Learning Rate Reduction)
- **Technical Indicators** (RSI, MACD, Bollinger Bands, ATR)
- **Feature Engineering** (Price momentum, volume analysis, time encoding)
- **Multi-step Forecasting** with confidence intervals
- **Regularization** with L1/L2 penalties

### Visualization Features:
- **Interactive Plots** with Plotly
- **Training History** visualization
- **Performance Comparison** charts
- **Future Forecast** projections
- **Real-time Progress** tracking
- **Confidence Intervals** for predictions
- **Technical Analysis** charts

### Data Processing:
- **Automatic Data Cleaning** and preprocessing
- **Feature Scaling** and normalization
- **Missing Value** handling
- **Outlier Detection** and treatment
- **Time Series** validation

---

## 📱 Usage Examples

### Running Individual Models:

```python
# Example: Running LSTM model
streamlit run lstm.py

# Select stock ticker (AAPL, TSLA, MSFT, etc.)
# Choose forecast period (30-120 days)
# Adjust LSTM parameters in sidebar
# Click "Run Advanced LSTM Forecast"
```

### Model Parameters:

**LSTM Parameters:**
- Sequence Length: 30-120 (lookback window)
- LSTM Units: 32-256 (model complexity)
- Training Epochs: 20-200 (training iterations)
- Batch Size: 16, 32, 64, 128
- Dropout Rate: 0.1-0.5 (regularization)
- Learning Rate: 0.001, 0.0005, 0.0001

**Advanced Options:**
- Bidirectional LSTM
- Technical Indicators
- Validation Split: 0.1-0.3

---

## 🎓 Learning Outcomes

✔️ Hands-on experience with time series modeling  
✔️ Worked with real stock market datasets  
✔️ Built and compared both statistical and deep learning models  
✔️ Created professional dashboards for result visualization  
✔️ Gained collaborative development experience in a team setting  
✔️ Implemented advanced LSTM architectures  
✔️ Integrated technical indicators for better predictions  
✔️ Applied machine learning in financial markets  
✔️ Developed interactive web applications  
✔️ Learned model evaluation and comparison techniques  

---

## 🔍 Model Insights

### When to Use Each Model:

**ARIMA**: 
- Stationary time series
- Linear trends
- Short-term forecasting
- When you need interpretable results

**SARIMA**:
- Seasonal patterns
- Regular cycles in data
- When seasonality is important
- Medium-term forecasting

**Prophet**:
- Strong seasonal patterns
- Holiday effects
- Trend changes
- When you need uncertainty quantification

**LSTM**:
- Complex non-linear patterns
- Long-term dependencies
- High-dimensional features
- When you have large datasets

---

## 👨‍💻 Team Members

🤝 This project is a result of collaborative effort by our internship team:

* Hamaesh S
* Mrudul Dehankar
* Jeevan Jiji Thomas
* Sahil Khan
* Anoheeta Mukherjee
* Esha Prajapati
* Makhana Akshad Viresh

---

## 📌 Final Notes

🔍 This internship project allowed us to explore real-world **financial forecasting** using modern tools and techniques.  
📈 It strengthened our foundations in both **classical time series analysis** and **deep learning**.  
🎯 The project demonstrates practical application of machine learning in financial markets.

### 🚀 Future Enhancements:
- Ensemble methods combining multiple models
- Real-time trading signals
- Portfolio optimization
- Risk assessment models
- Mobile application development

> 💬 *Thank you for visiting our repository — we're proud of what we built together in just one month!*

---

## 📞 Contact

For questions or collaboration opportunities, please reach out to the team members.

### 🤝 Contributing:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📚 References

- [Yahoo Finance API](https://finance.yahoo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Statsmodels Documentation](https://www.statsmodels.org/)

---

**⭐ Star this repository if you found it helpful!**

---

*Last updated: January 2024* 
