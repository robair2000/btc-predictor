import numpy as np
import pandas as pd
import requests
import streamlit as st
import xgboost as xgb
import yfinance as yf
from sklearn.model_selection import train_test_split

# --- STREAMLIT PAGINA CONFIGURATIE ---
st.set_page_config(
    page_title='Bitcoin Prijsvoorspelling Dashboard', page_icon='📈', layout='wide'
)

st.title('🚀 Bitcoin Machine Learning & Kwantitatief Prijsdashboard')
st.markdown(
    'Dit dashboard voorspelt de **exacte Bitcoin-prijs in dollars** voor de'
    ' volgende dag op basis van echte koersdata, technische indicatoren en'
    ' live sentiment.'
)


# --- DATA OPHAAL & FEATURE ENGINEERING FUNCTIES ---
@st.cache_data(ttl=3600)
def load_bitcoin_data():
  # Haal historische koersdata op via yfinance (OHLCV)
  df = yf.download('BTC-USD', start='2020-01-01', progress=False)

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


def add_technical_indicators(df):
  # Moving Averages (SMA / EMA)
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


# --- DATA LADEN EN VOORBEREIDEN ---
with st.spinner('Bezig met ophalen en berekenen van marktdata...'):
  df = load_bitcoin_data()
  df = add_technical_indicators(df)

  # Voeg echte Fear & Greed data toe
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

  # Maak Target: Voorspel de daadwerkelijke sluitingsprijs van de *volgende* dag (Regressie)
  df['Target'] = df['close'].shift(-1)
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

# Selecteer zuivere features voor het model
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
]

X = df[features]
y = df['Target']

# Opsplitsen in train en test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100.0, shuffle=False
)

# --- TRAIN XGBOOST REGRESSOR MODEL ---
model = xgb.XGBRegressor(
    n_estimators=n_estimators, max_depth=max_depth, random_state=42
)
model.fit(X_train, y_train)

# Model score (R-squared op testset)
score = model.score(X_test, y_test)

# --- DASHBOARD LAYOUT ---
current_price = df['close'].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric('Huidige Bitcoin Koers', f'${current_price:,.2f}')
col2.metric('Model Score ($R^2$ op Testset)', f'{score:.2f}')
col3.metric('Laatste RSI Waarde', f"{df['RSI'].iloc[-1]:.2f}")

st.markdown('---')

# Grafiek: Koers & Technische Indicatoren
st.subheader('📊 Prijsverloop & Moving Averages')
st.line_chart(df.set_index('Date')[['close', 'SMA_50', 'SMA_200']])

# Feature Importance grafiek
st.subheader('🔍 Welke factoren wegen het zwaarst?')
importances = pd.Series(model.feature_importances_, index=features).sort_values(
    ascending=True
)
st.bar_chart(importances)

# Voorspelling voor morgen
latest_data = X.iloc[-1].values.reshape(1, -1)
predicted_price = model.predict(latest_data)[0]
price_diff = predicted_price - current_price
percentage_diff = (price_diff / current_price) * 100

st.markdown('---')
st.subheader('🔮 Prijsvoorspelling voor de Volgende Dag')

col_a, col_b = st.columns(2)
col_a.metric(
    'Voorspelde Prijs',
    f'${predicted_price:,.2f}',
    delta=f'{percentage_diff:+.2f}%',
)

if predicted_price > current_price:
  col_b.success(
      f'**Verwachting: STIJGING 📈**\n\nHet model verwacht dat de koers stijgt'
      f' t.o.v. nu (${price_diff:+,.2f}).'
  )
else:
  col_b.warning(
      f'**Verwachting: DALING 📉**\n\nHet model verwacht dat de koers daalt'
      f' t.o.v. nu (${price_diff:+,.2f}).'
  )

st.info(
    '💡 *Dit model maakt gebruik van echte historische marktdata,'
    ' technische indicatoren en live sentiment zonder gesimuleerde ruis.*'
)
