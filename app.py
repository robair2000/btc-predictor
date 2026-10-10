import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import yfinance as yf

st.title("📈 Multi-Factor Bitcoin Predictor with Exchange Reserves")
st.write(
    "A machine learning dashboard incorporating BTC price history, S&P 500"
    " macro trends, and Exchange Bitcoin Reserves."
)


# 1. Fetch & Simulate Multi-Asset Data
@st.cache_data
def load_data():
  # Download BTC and S&P 500 data
  df = yf.download(["BTC-USD", "^GSPC"], start="2021-01-01")["Close"]
  df.reset_index(inplace=True)
  df.columns = ["Date", "BTC_Close", "SP500_Close"]
  df = df.dropna()

  # Simulate 'Exchange Coins Held' (Reserve Proxy)
  # Replace this section with a real API call (e.g., CryptoQuant / Glassnode API)
  np.random.seed(42)
  base_reserve = 2_200_000  # Approx. 2.2M BTC held on exchanges
  reserve_fluctuations = np.random.normal(0, 1500, len(df)).cumsum()
  df["Exchange_BTC_Reserve"] = base_reserve + reserve_fluctuations

  return df


with st.spinner("Loading market and exchange balance metrics..."):
  data = load_data()

st.subheader("Raw Multi-Factor Dataset")
st.dataframe(data.tail())

# 2. Feature Engineering
data["BTC_7_MA"] = data["BTC_Close"].rolling(window=7).mean()
data["SP500_7_MA"] = data["SP500_Close"].rolling(window=7).mean()
data["Reserve_7_MA"] = data["Exchange_BTC_Reserve"].rolling(window=7).mean()

# Target: Next day's BTC closing price
data["Target"] = data["BTC_Close"].shift(-1)
data = data.dropna()

# 3. Model Training
X = data[[
    "BTC_7_MA",
    "SP500_Close",
    "SP500_7_MA",
    "Exchange_BTC_Reserve",
    "Reserve_7_MA",
]]
y = data["Target"]

# Sequential split for time series
split_idx = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = xgb.XGBRegressor(
    n_estimators=100, learning_rate=0.05, random_state=42
)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
st.metric(label="Model R² Score on Test Data", value=f"{score:.2f}")

# 4. Predict Next Day
latest_inputs = X.iloc[[-1]]
next_day_pred = model.predict(latest_inputs)[0]

st.subheader("🔮 Price Forecast")
st.success(f"Estimated Next-Day Closing Price: **${next_day_pred:,.2f} USD**")

# Feature Importance Breakdown
st.subheader("📊 Feature Importance (What drives the model?)")
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_,
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))