from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st
import xgboost as xgb
import yfinance as yf
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title='Institutional Crypto Quant Scalper', layout='wide'
)


# --- DATA & INDICATOR FUNCTIES ---
@st.cache_data(ttl=900)
def load_market_data(ticker_symbol):
  df = yf.download(ticker_symbol, period='60d', interval='15m', progress=False)
  if df.empty:
    return None
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
  df = df.reset_index()
  df.columns = [c.lower() for c in df.columns]
  return df


def add_advanced_indicators(df):
  df['sma_50'] = df['close'].rolling(window=50).mean()
  df['mtf_trend_bullish'] = df['close'] > df['sma_50']

  high_low = df['high'] - df['low']
  high_close = np.abs(df['high'] - df['close'].shift())
  low_close = np.abs(df['low'] - df['close'].shift())
  ranges = pd.concat([high_low, high_close, low_close], axis=1)
  true_range = ranges.max(axis=1)
  df['ATR'] = true_range.rolling(14).mean()

  v = df['volume']
  tp = (df['high'] + df['low'] + df['close']) / 3
  df['VWAP_distance'] = (df['close'] - (tp * v).cumsum() / v.cumsum()) / (
      tp * v
  ).cumsum() * v.cumsum()

  ma20 = df['close'].rolling(20).mean()
  std20 = df['close'].rolling(20).std()
  upper = ma20 + (std20 * 2)
  lower = ma20 - (std20 * 2)
  df['BB_pct_b'] = (df['close'] - lower) / (upper - lower)

  df['SMA_20'] = ma20
  df['EMA_9'] = df['close'].ewm(span=9).mean()
  df['EMA_21'] = df['close'].ewm(span=21).mean()

  exp1 = df['close'].ewm(span=12, adjust=False).mean()
  exp2 = df['close'].ewm(span=26, adjust=False).mean()
  df['MACD'] = exp1 - exp2
  df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

  delta = df['close'].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / loss
  df['RSI'] = 100 - (100 / (1 + rs))

  df['taker_buy_ratio'] = 0.5 + (df['close'].pct_change() * 2)
  return df


def fetch_crypto_fear_greed():
  try:
    res = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)
    data = res.json()
    return float(data['data'][0]['value'])
  except:
    return 50.0


def fetch_realtime_price(symbol):
  try:
    res = requests.get(
        f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}',
        timeout=3,
    )
    return float(res.json()['price'])
  except:
    return None


# --- HOOFDTITEL & SIDEBAR ---
st.title('⚡ Institutional Quant Scalping Platform')

st.sidebar.header('⚙️ Model & Strategie Instellingen')
selected_asset = st.sidebar.selectbox(
    'Selecteer Asset', ['LINK-USD', 'BTC-USD'], index=0
)
binance_symbol = 'LINKUSDT' if 'LINK' in selected_asset else 'BTCUSDT'

test_size = st.sidebar.slider(
    'Test / Train Splitsing (%)', 10, 40, 20, step=5
)  # Vaste standaard 20
n_estimators = st.sidebar.slider(
    'XGBoost Bomen (n_estimators)', 50, 300, 150
)  # Vaste standaard 150
max_depth = st.sidebar.slider(
    'Maximale Boomdiepte (max_depth)', 3, 10, 4
)  # Vaste standaard 4

st.sidebar.markdown('---')
st.sidebar.subheader('🤖 Notificatie / Webhook')
webhook_url = st.sidebar.text_input('Discord/Telegram Webhook URL')
send_alert = st.sidebar.button('Stuur Signaal Test')

if send_alert and webhook_url:
  try:
    requests.post(webhook_url, json={'content': 'Test signaal vanuit Quant Bot!'})
    st.sidebar.success('Test notificatie verzonden!')
  except Exception as e:
    st.sidebar.error(f'Fout bij verzenden: {e}')

# --- DATA LADEN & PREPAREREN ---
with st.spinner(f'Laden van {selected_asset} marktdata...'):
  df = load_market_data(selected_asset)
  if df is None or df.empty:
    st.error('Kon geen data laden.')
    st.stop()

  df = add_advanced_indicators(df)
  df['fear_greed'] = fetch_crypto_fear_greed()
  df['Target'] = df['close'].shift(-1)
  df = df.dropna().reset_index(drop=True)

features = [
    'close',
    'open',
    'high',
    'low',
    'volume',
    'taker_buy_ratio',
    'VWAP_distance',
    'BB_pct_b',
    'ATR',
    'SMA_20',
    'EMA_9',
    'EMA_21',
    'MACD',
    'MACD_Signal',
    'RSI',
    'fear_greed',
    'mtf_trend_bullish',
]

X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100.0, shuffle=False
)

# --- MODEL TRAINING ---
model = xgb.XGBRegressor(
    n_estimators=n_estimators, max_depth=max_depth, random_state=42
)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# --- BACKTESTING SIMULATIE ---
test_preds = model.predict(X_test)
test_actuals = y_test.values
test_closes = X_test['close'].values

pred_direction = np.sign(test_preds - test_closes)
actual_direction = np.sign(test_actuals - test_closes)
directional_acc = np.mean(pred_direction == actual_direction) * 100

# --- DASHBOARD UI & RISK MANAGEMENT ---
realtime_price = fetch_realtime_price(binance_symbol)
current_price = realtime_price if realtime_price else float(df['close'].iloc[-1])
current_atr = float(df['ATR'].iloc[-1])
is_mtf_bullish = bool(df['mtf_trend_bullish'].iloc[-1])

# Dynamische ATR-based Stop-Loss & Take-Profit
stop_loss_long = current_price - (1.5 * current_atr)
take_profit_long = current_price + (2.5 * current_atr)

laatste_update = datetime.now().strftime('%H:%M:%S (%d-%m-%Y)')

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(f'Huidige {selected_asset.split("-")[0]} Koers', f'${current_price:,.3f}')
col2.metric('Model Score ($R^2$)', f'{score:.2f}')
col3.metric('Directional Win-Rate', f'{directional_acc:.1f}%')
col4.metric(
    'Daily Trend Filter', '🟢 BULLISH' if is_mtf_bullish else '🔴 BEARISH'
)
col5.metric('Laatste Model Update', laatste_update)

st.markdown('---')

# Risico & Trade setup sectie
st.subheader('🛡️ Dynamisch Risico & ATR Levels')
r_col1, r_col2, r_col3 = st.columns(3)
r_col1.metric('Huidige ATR (Volatiliteit)', f'${current_atr:.4f}')
r_col2.metric('Voorbeeld Stop-Loss (Long)', f'${stop_loss_long:,.3f}')
r_col3.metric('Voorbeeld Take-Profit (Long)', f'${take_profit_long:,.3f}')

st.markdown('---')

# SHAP Explainer sectie
st.subheader('🔍 Model Explainability (SHAP Values)')
try:
  explainer = shap.Explainer(model, X_test)
  shap_values = explainer(X_test)
  fig, ax = plt.subplots(figsize=(10, 4))
  shap.plots.beeswarm(shap_values, max_display=8, show=False)
  st.pyplot(fig)
except Exception as e:
  st.info(
      'SHAP plot kan tijdelijk niet worden gerenderd op deze dataset-omvang.'
  )

st.markdown('---')
st.subheader('📈 Laatste Koersverloop')
st.line_chart(df[['close', 'SMA_20']].tail(100))
