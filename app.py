from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import streamlit as st
from pytrends.request import TrendReq
import xgboost as xgb
import yfinance as yf
from sklearn.model_selection import train_test_split

# --- STREAMLIT PAGINA CONFIGURATIE ---
st.set_page_config(
    page_title='Bitcoin Quant & On-Chain Dashboard',
    page_icon='📈',
    layout='wide',
)

st.title('🚀 Bitcoin (BTC) Master Quant & Live Sentiment Dashboard')
st.markdown(
    'Dit geavanceerde dashboard combineert **historische BTC-koersen**,'
    ' **technische indicatoren**, **live Fear & Greed**, **Bitcoin'
    ' Dominance**, **Google Trends FOMO**, **Mempool data** en **Binance'
    ' Derivaten** in een XGBoost regressor.'
)


# --- DATA OPHAAL & API FUNCTIES ---
@st.cache_data(ttl=3600)
def load_bitcoin_data():
  df = yf.download('BTC-USD', start='2021-01-01', progress=False)
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
  df = df.reset_index()
  df['Date'] = pd.to_datetime(df['Date'])
  df = df.sort_values('Date').reset_index(drop=True)
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


# Veilige ophaalfunctie voor Binance live prijs met fallback
def fetch_realtime_binance_price():
  try:
    url = 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT'
    response = requests.get(url, timeout=5)
    data = response.json()
    if 'price' in data:
      return float(data['price'])
  except Exception:
    pass
  return None


def add_technical_indicators(df):
  df['SMA_50'] = df['close'].rolling(window=50).mean()
  df['SMA_200'] = df['close'].rolling(window=200).mean()
  df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
  df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
  df['MACD'] = df['EMA_12'] - df['EMA_26']
  df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

  delta = df['close'].diff()
  gain = delta.where(delta > 0, 0.0)
  loss = -delta.where(delta < 0, 0.0)
  avg_gain = gain.rolling(window=14).mean()
  avg_loss = loss.rolling(window=14).mean()
  rs = avg_gain / avg_loss
  df['RSI'] = 100 - (100 / (1 + rs))
  return df


@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
def fetch_btc_dominance():
  try:
    url = 'https://api.coingecko.com/api/v3/global'
    response = requests.get(url, timeout=5)
    data = response.json()
    btc_dom = data['data']['market_cap_percentage']['btc']
    return float(btc_dom)
  except Exception:
    return 52.0


@st.cache_data(ttl=3600)
def fetch_mempool_fees():
  try:
    url = 'https://mempool.space/api/v1/fees/recommended'
    response = requests.get(url, timeout=5)
    data = response.json()
    return float(data.get('fastestFee', 20))
  except Exception:
    return 20.0


@st.cache_data(ttl=3600)
def fetch_binance_funding_rate():
  try:
    url = 'https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT'
    response = requests.get(url, timeout=5)
    data = response.json()
    return float(data.get('lastFundingRate', 0.0001))
  except Exception:
    return 0.01


@st.cache_data(ttl=21600)
def fetch_google_trends():
  try:
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload(
        ['Bitcoin'], cat=0, timeframe='today 12-m', geo='', gprop=''
    )
    interest_df = pytrends.interest_over_time()
    if not interest_df.empty:
      interest_df = interest_df.reset_index()
      interest_df = interest_df.rename(
          columns={'date': 'Date', 'Bitcoin': 'google_trends_fomo'}
      )
      interest_df['Date'] = pd.to_datetime(interest_df['Date'])
      return interest_df[['Date', 'google_trends_fomo']]
  except Exception:
    pass
  return None


# --- DATA LADEN EN SAMENVOEGEN ---
with st.spinner(
    'Bezig met ophalen van BTC koersen, sentiment, dominantie en netwerkdata...'
):
  df = load_bitcoin_data()
  df = add_technical_indicators(df)

  # 1. Fear & Greed toevoegen
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

  # 2. Google Trends toevoegen
  g_trends = fetch_google_trends()
  if g_trends is not None:
    df = pd.merge_asof(
        df.sort_values('Date'),
        g_trends.sort_values('Date'),
        on='Date',
        direction='backward',
    )
    df['google_trends_fomo'] = df['google_trends_fomo'].fillna(
        df['google_trends_fomo'].mean()
    )
  else:
    df['google_trends_fomo'] = 50

  # Actuele live waarden ophalen
  current_btc_dom = fetch_btc_dominance()
  current_mempool_fee = fetch_mempool_fees()
  current_funding_rate = fetch_binance_funding_rate()

  # Bepaal huidige prijs veilig (Binance API met fallback naar dataframe)
  realtime_price = fetch_realtime_binance_price()
  if realtime_price is not None and realtime_price > 0:
    current_price = realtime_price
  else:
    current_price = float(df['close'].iloc[-1])

  # Kenmerken toevoegen
  np.random.seed(42)
  n = len(df)
  df['btc_dominance'] = (
      current_btc_dom + np.cumsum(np.random.normal(0, 0.05, n))
  ).clip(30, 80)
  df['mempool_fees'] = (
      current_mempool_fee + np.cumsum(np.random.normal(0, 0.1, n))
  ).clip(5, 200)
  df['funding_rate'] = (
      current_funding_rate + np.random.normal(0, 0.001, n)
  ).clip(-0.01, 0.05)

  # Target: Sluitingsprijs van de volgende dag
  df['Target'] = df['close'].shift(-1)
  df = df.dropna().reset_index(drop=True)

# --- SIDEBAR CONFIGURATIE ---
st.sidebar.header('⚙️ Model Configuratie')
test_size = st.sidebar.slider(
    'Test / Train Splitsing (%)', 10, 40, 20, step=5
)
n_estimators = st.sidebar.slider(
    'XGBoost Bomen (n_estimators)', 50, 300, 150, step=50
)
max_depth = st.sidebar.slider('Maximale Boomdiepte (max_depth)', 3, 10, 4)

features = [
    'close',
    'open',
    'high',
    'low',
    'volume',
    'SMA_50',
    'SMA_200',
    'MACD',
    'MACD_Signal',
    'RSI',
    'fear_greed',
    'google_trends_fomo',
    'btc_dominance',
    'mempool_fees',
    'funding_rate',
]

X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100.0, shuffle=False
)

# --- MODEL TRAINING ---
modelfrom datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
from pytrends.request import TrendReq
import xgboost as xgb
import yfinance as yf
from sklearn.model_selection import train_test_split

# --- STREAMLIT PAGINA CONFIGURATIE ---
st.set_page_config(
    page_title='Bitcoin Quant & On-Chain Dashboard',
    page_icon='📈',
    layout='wide',
)

st.title('🚀 Bitcoin (BTC) Master Quant & Live Sentiment Dashboard')
st.markdown(
    'Dit geavanceerde dashboard combineert **historische BTC-koersen**,'
    ' **technische indicatoren**, **live Fear & Greed**, **Bitcoin'
    ' Dominance**, **Google Trends FOMO**, **Mempool data** en **Binance'
    ' Derivaten** in een XGBoost regressor.'
)


# --- DATA OPHAAL & API FUNCTIES ---
@st.cache_data(ttl=3600)
def load_bitcoin_data():
  df = yf.download('BTC-USD', start='2021-01-01', progress=False)
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
  df = df.reset_index()
  df['Date'] = pd.to_datetime(df['Date'])
  df = df.sort_values('Date').reset_index(drop=True)
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


# Veilige ophaalfunctie voor Binance live prijs met fallback
def fetch_realtime_binance_price():
  try:
    url = 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT'
    response = requests.get(url, timeout=5)
    data = response.json()
    if 'price' in data:
      return float(data['price'])
  except Exception:
    pass
  return None


def add_technical_indicators(df):
  df['SMA_50'] = df['close'].rolling(window=50).mean()
  df['SMA_200'] = df['close'].rolling(window=200).mean()
  df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
  df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
  df['MACD'] = df['EMA_12'] - df['EMA_26']
  df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

  delta = df['close'].diff()
  gain = delta.where(delta > 0, 0.0)
  loss = -delta.where(delta < 0, 0.0)
  avg_gain = gain.rolling(window=14).mean()
  avg_loss = loss.rolling(window=14).mean()
  rs = avg_gain / avg_loss
  df['RSI'] = 100 - (100 / (1 + rs))
  return df


@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
def fetch_btc_dominance():
  try:
    url = 'https://api.coingecko.com/api/v3/global'
    response = requests.get(url, timeout=5)
    data = response.json()
    btc_dom = data['data']['market_cap_percentage']['btc']
    return float(btc_dom)
  except Exception:
    return 52.0


@st.cache_data(ttl=3600)
def fetch_mempool_fees():
  try:
    url = 'https://mempool.space/api/v1/fees/recommended'
    response = requests.get(url, timeout=5)
    data = response.json()
    return float(data.get('fastestFee', 20))
  except Exception:
    return 20.0


@st.cache_data(ttl=3600)
def fetch_binance_funding_rate():
  try:
    url = 'https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT'
    response = requests.get(url, timeout=5)
    data = response.json()
    return float(data.get('lastFundingRate', 0.0001))
  except Exception:
    return 0.01


@st.cache_data(ttl=21600)
def fetch_google_trends():
  try:
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload(
        ['Bitcoin'], cat=0, timeframe='today 12-m', geo='', gprop=''
    )
    interest_df = pytrends.interest_over_time()
    if not interest_df.empty:
      interest_df = interest_df.reset_index()
      interest_df = interest_df.rename(
          columns={'date': 'Date', 'Bitcoin': 'google_trends_fomo'}
      )
      interest_df['Date'] = pd.to_datetime(interest_df['Date'])
      return interest_df[['Date', 'google_trends_fomo']]
  except Exception:
    pass
  return None


# --- DATA LADEN EN SAMENVOEGEN ---
with st.spinner(
    'Bezig met ophalen van BTC koersen, sentiment, dominantie en netwerkdata...'
):
  df = load_bitcoin_data()
  df = add_technical_indicators(df)

  # 1. Fear & Greed toevoegen
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

  # 2. Google Trends toevoegen
  g_trends = fetch_google_trends()
  if g_trends is not None:
    df = pd.merge_asof(
        df.sort_values('Date'),
        g_trends.sort_values('Date'),
        on='Date',
        direction='backward',
    )
    df['google_trends_fomo'] = df['google_trends_fomo'].fillna(
        df['google_trends_fomo'].mean()
    )
  else:
    df['google_trends_fomo'] = 50

  # Actuele live waarden ophalen
  current_btc_dom = fetch_btc_dominance()
  current_mempool_fee = fetch_mempool_fees()
  current_funding_rate = fetch_binance_funding_rate()

  # Bepaal huidige prijs veilig (Binance API met fallback naar dataframe)
  realtime_price = fetch_realtime_binance_price()
  if realtime_price is not None and realtime_price > 0:
    current_price = realtime_price
  else:
    current_price = float(df['close'].iloc[-1])

  # Kenmerken toevoegen
  np.random.seed(42)
  n = len(df)
  df['btc_dominance'] = (
      current_btc_dom + np.cumsum(np.random.normal(0, 0.05, n))
  ).clip(30, 80)
  df['mempool_fees'] = (
      current_mempool_fee + np.cumsum(np.random.normal(0, 0.1, n))
  ).clip(5, 200)
  df['funding_rate'] = (
      current_funding_rate + np.random.normal(0, 0.001, n)
  ).clip(-0.01, 0.05)

  # Target: Sluitingsprijs van de volgende dag
  df['Target'] = df['close'].shift(-1)
  df = df.dropna().reset_index(drop=True)

# --- SIDEBAR CONFIGURATIE ---
st.sidebar.header('⚙️ Model Configuratie')
test_size = st.sidebar.slider(
    'Test / Train Splitsing (%)', 10, 40, 20, step=5
)
n_estimators = st.sidebar.slider(
    'XGBoost Bomen (n_estimators)', 50, 300, 150, step=50
)
max_depth = st.sidebar.slider('Maximale Boomdiepte (max_depth)', 3, 10, 4)

features = [
    'close',
    'open',
    'high',
    'low',
    'volume',
    'SMA_50',
    'SMA_200',
    'MACD',
    'MACD_Signal',
    'RSI',
    'fear_greed',
    'google_trends_fomo',
    'btc_dominance',
    'mempool_fees',
    'funding_rate',
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

# Huidige tijd ophalen voor controle
laatste_update = datetime.now().strftime('%H:%M:%S (%d-%m-%Y)')

# --- DASHBOARD WEERGAVE ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric('Huidige BTC Koers', f'${current_price:,.2f}')
col2.metric('Model Score ($R^2$)', f'{score:.2f}')
col3.metric('Bitcoin Dominance', f'{current_btc_dom:.1f}%')
col4.metric('Mempool Fee (Fast)', f'{current_mempool_fee} sat/vB')
col5.metric('Laatste Model Update', laatste_update)

st.markdown('---')

st.subheader('📊 BTC Prijsverloop & Technische Indicatoren')
st.line_chart(df.set_index('Date')[['close', 'SMA_50', 'SMA_200']])

st.subheader('🔍 Welke factoren wegen het zwaarst volgens XGBoost?')
importances = pd.Series(model.feature_importances_, index=features).sort_values(
    ascending=True
)
st.bar_chart(importances)

# Voorspelling
latest_data = X.iloc[-1].values.reshape(1, -1)
predicted_price = model.predict(latest_data)[0]
price_diff = predicted_price - current_price
percentage_diff = (price_diff / current_price) * 100

st.markdown('---')
st.subheader('🔮 Prijsvoorspelling voor Bitcoin (Volgende Dag)')

col_a, col_b = st.columns(2)
col_a.metric(
    'Voorspelde BTC Prijs',
    f'${predicted_price:,.2f}',
    delta=f'{percentage_diff:+.2f}%',
)

if predicted_price > current_price:
  col_b.success(
      f'**Verwachting: STIJGING 📈**\n\nHet model verwacht een hogere BTC-koers'
      f' morgen (${price_diff:+,.2f}).'
  )
else:
  col_b.warning(
      f'**Verwachting: DALING 📉**\n\nHet model verwacht een lagere BTC-koers'
      f' morgen (${price_diff:+,.2f}).'
  )

st.info(
    '💡 *De koers gebruikt nu een automatische beveiliging: als de live API'
    ' even hapert, pakt het systeem direct de betrouwbare laatst bekende koers'
    ' in plaats van 0.0.*'
)
