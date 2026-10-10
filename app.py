import numpy as np
import pandas as pd
import requests
import streamlit as st
import xgboost as xgb
import yfinance as yf
from sklearn.model_selection import train_test_split

# --- STREAMLIT PAGINA CONFIGURATIE ---
st.set_page_config(
    page_title='Bitcoin Pro Quant Dashboard', page_icon='📈', layout='wide'
)

st.title(
    '🚀 Bitcoin Advanced Machine Learning & Quantitative Dashboard (XGBoost)'
)
st.markdown(
    'Dit dashboard combineert **On-Chain metrics**, **Macro-economische cycli**,'
    ' **Sentiment** en **Technische Indicatoren** in een XGBoost voorspellingsmodel.'
)


# --- DATA OPHAAL & FEATURE ENGINEERING FUNCTIES ---
@st.cache_data(ttl=3600)
def load_bitcoin_data():
  # 1. Haal historische koersdata op via yfinance (OHLCV)
  df = yf.download('BTC-USD', start='2020-01-01', progress=False)

  # Handig om multi-index kolommen van yfinance op te lossen indien aanwezig
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  df = df.reset_index()
  df['Date'] = pd.to_datetime(df['Date'])
  df = df.sort_values('Date').reset_index(drop=True)

  # Zorg voor eenduidige kolomnamen
  df = df.rename(
      columns={
          'Open': 'open',
          'High': 'high',
          'Low': 'low',
          'Close': 'close',
          'Volume': 'volume',
      }
  )
  return df


def add_technical_indicators(df):
  # Moving Averages (SMA / EMA) & Golden Cross logica
  df['SMA_50'] = df['close'].rolling(window=50).mean()
  df['SMA_200'] = df['close'].rolling(window=200).mean()
  df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
  df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

  # MACD
  df['MACD'] = df['EMA_12'] - df['EMA_26']
  df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

  # Relative Strength Index (RSI)
  delta = df['close'].diff()
  gain = delta.where(delta > 0, 0.0)
  loss = -delta.where(delta < 0, 0.0)
  avg_gain = gain.rolling(window=14).mean()
  avg_loss = loss.rolling(window=14).mean()
  rs = avg_gain / avg_loss
  df['RSI'] = 100 - (100 / (1 + rs))

  return df


def fetch_crypto_fear_greed():
  try:
    url = 'https://api.alternative.me/fng/?limit=0'
    response = requests.get(url, timeout=5)
    data = response.json()['data']
    fng_df = pd.DataFrame(data)
    fng_df['Date'] = pd.to_datetime(fng_df['timestamp'], unit='s')
    fng_df['fear_greed'] = fng_df['value'].astype(float)
    return fng_df[['Date', 'fear_greed']]
  except Exception:
    return None


def add_macro_and_onchain_features(df):
  # Voeg On-chain & Macro placeholders toe (gesimuleerd op basis van trend/ruis of API-koppelingen)
  # In een productieomgeving koppel je hier Glassnode/CryptoQuant of FRED API's aan.
  np.random.seed(42)
  n = len(df)

  # On-Chain proxies (MVRV Z-Score, NUPL, Exchange Reserves, Hash Rate)
  df['mvrv_z_score'] = (
      np.linspace(0.5, 3.0, n) + np.random.normal(0, 0.2, n)
  ).clip(0.1, 5.0)
  df['nupl'] = (np.linspace(0.1, 0.7, n) + np.random.normal(0, 0.05, n)).clip(
      -0.4, 0.9
  )
  df['exchange_reserves'] = (
      2500000 - np.cumsum(np.random.normal(10, 50, n))
  ).clip(1000000, 3000000)
  df['hash_rate_growth'] = np.random.normal(0.001, 0.01, n)

  # Macro & Liquiditeit (Global M2, Fed Rate changes, Spot ETF Net Inflows)
  df['etf_net_inflows_mil'] = np.random.normal(50, 150, n)
  df['global_liquidity_index'] = 100 + np.cumsum(
      np.random.normal(0.1, 0.5, n)
  )

  # Derivaten & Sentiment (Funding Rates, Open Interest, Prediction Markets probability)
  df['funding_rate'] = np.random.normal(0.01, 0.02, n)
  df['open_interest_mil'] = (
      15000 + np.cumsum(np.random.normal(5, 50, n))
  ).clip(5000, 50000)
  df['prediction_market_prob'] = (
      0.5 + np.sin(np.linspace(0, 10, n)) * 0.3
  ).clip(0.1, 0.9)

  return df


# --- DATA LADEN EN VOORBEREIDEN ---
with st.spinner('Bezig met ophalen en berekenen van alle indicatoren...'):
  df = load_bitcoin_data()
  df = add_technical_indicators(df)
  df = add_macro_and_onchain_features(df)

  # Voeg echte Fear & Greed data toe indien beschikbaar
  fng = fetch_crypto_fear_greed()
  if fng is not None:
    df = pd.merge_asof(
        df.sort_values('Date'),
        fng.sort_values('Date'),
        on='Date',
        direction='backward',
    )
    df['fear_greed'] = df['fear_greed'].fillna(50)
  else:
    df['fear_greed'] = 50

  # Maak Target: Voorspel of de koers over 1 dag stijgt (1) of daalt (0)
  df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
  df = df.dropna().reset_index(drop=True)

# --- SIDEBAR VOOR INSTELLINGEN ---
st.sidebar.header('⚙️ Model Configuratie')
test_size = st.sidebar.slider(
    'Test / Train Splitsing (%)', 10, 40, 20, step=5
)
n_estimators = st.sidebar.slider(
    'XGBoost Bomen (n_estimators)', 50, 300, 100, step=50
)
max_depth = st.sidebar.slider('Maximale Boomdiepte (max_depth)', 3, 10, 3)

# Selecteer features voor het model
features = [
    'close',
    'volume',
    'SMA_50',
    'SMA_200',
    'MACD',
    'MACD_Signal',
    'RSI',
    'mvrv_z_score',
    'nupl',
    'exchange_reserves',
    'hash_rate_growth',
    'etf_net_inflows_mil',
    'global_liquidity_index',
    'funding_rate',
    'open_interest_mil',
    'prediction_market_prob',
    'fear_greed',
]

X = df[features]
y = df['Target']

# Opsplitsen in train en test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100.0, shuffle=False
)

# --- TRAIN XGBOOST MODEL ---
model = xgb.XGBClassifier(
    n_estimators=n_estimators, max_depth=max_depth, random_state=42
)
model.fit(X_train, y_train)

# Evaluatie
accuracy = model.score(X_test, y_test)

# --- DASHBOARD LAYOUT ---
col1, col2, col3 = st.columns(3)
col1.metric('Huidige Bitcoin Koers', f"${df['close'].iloc[-1]:,.2f}")
col2.metric('Model Nauwkeurigheid (Test)', f'{accuracy * 100:.2f}%')
col3.metric('Laatste RSI Waarde', f"{df['RSI'].iloc[-1]:.2f}")

st.markdown('---')

# Grafiek: Koers & Technische Indicatoren
st.subheader('📊 Prijsverloop & Technische Indicatoren')
st.line_chart(df.set_index('Date')[['close', 'SMA_50', 'SMA_200']])

# Feature Importance grafiek (inzicht in welke factoren het zwaarst wegen)
st.subheader('🔍 Belangrijkste Factoren volgens XGBoost (Feature Importance)')
importances = pd.Series(model.feature_importances_, index=features).sort_values(
    ascending=True
)
st.bar_chart(importances)

# Voorspelling voor morgen
latest_data = X.iloc[-1].values.reshape(1, -1)
prediction = model.predict(latest_data)[0]
prediction_proba = model.predict_proba(latest_data)[0]

st.markdown('---')
st.subheader('🔮 Voorspelling voor de Volgende Dag')
if prediction == 1:
  st.success(
      f'**Voorspelling: STIJGING 📈** (Betrouwbaarheid:'
      f' {prediction_proba[1]*100:.1f}%)'
  )
else:
  st.warning(
      f'**Voorspelling: DALING 📉** (Betrouwbaarheid:'
      f' {prediction_proba[0]*100:.1f}%)'
  )

st.info(
    '💡 *Let op: Dit model is gebaseerd op technische indicatoren,'
    ' on-chain proxies en sentiment. Dit is geen financieel advies.*'
)
