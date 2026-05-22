import streamlit as st
import pandas as pd
import numpy as np
import ta
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. Load the trained model
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.pkl")

model = load_model()

# ============================================================
# 2. Fetch latest 5‑minute data for the tickers
# ============================================================
TICKERS = ["AAPL","NVDA","TSLA","MSFT","AMZN","AMD","GOOG","AVGO","INTC","MU"]
INTERVAL = "5m"
# Use a period long enough to compute the longest indicator (e.g., 50‑period EMA + 20‑period Bollinger + lag 5)
# 100 bars should be safe, so about 100 * 5 = 500 minutes = ~8.3 hours.
# For safety, request 5 days of 5m data.
LOOKBACK_DAYS = 5

@st.cache_data(ttl=60)   # cache data for 60 seconds
def fetch_data(tickers):
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_DAYS)
    frames = []
    for t in tickers:
        df = yf.download(t, interval=INTERVAL, start=start, end=end,
                         progress=False, auto_adjust=False, prepost=False)
        if df.empty:
            continue
        df.index = df.index.tz_localize(None)
        df = df[['Open','High','Low','Close','Volume']].copy()
        df['Ticker'] = t
        df.reset_index(inplace=True)
        df.rename(columns={'index':'Datetime'}, inplace=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    full['Datetime'] = pd.to_datetime(full['Datetime'])
    full.sort_values(['Ticker','Datetime'], inplace=True)
    return full

# ============================================================
# 3. Feature engineering (exactly as in training)
# ============================================================
def create_features(group):
    group = group.copy()
    # ✅ Reset index to avoid alignment errors from duplicates/gaps
    group.reset_index(drop=True, inplace=True)
    
    # Defensive: ensure 'Close' is a Series (not a DataFrame)
    if isinstance(group['Close'], pd.DataFrame):
        group['Close'] = group['Close'].iloc[:, 0]

    # Basic price features
    group["returns_1"] = group["Close"].pct_change(1)
    group["returns_3"] = group["Close"].pct_change(3)
    group["returns_5"] = group["Close"].pct_change(5)
    group["log_return"] = np.log(group["Close"] / group["Close"].shift(1))

    # EMA features
    for p in [5, 9, 12, 20, 50]:
        group[f"ema_{p}"] = ta.trend.ema_indicator(group["Close"], window=p)
        group[f"ema_dist_{p}"] = (group["Close"] - group[f"ema_{p}"]) / group[f"ema_{p}"]
        group[f"ema_slope_{p}"] = group[f"ema_{p}"].diff()
    group["ema_spread_5_20"] = group["ema_5"] - group["ema_20"]
    group["ema_spread_9_50"] = group["ema_9"] - group["ema_50"]

    # RSI
    group["rsi_14"] = ta.momentum.rsi(group["Close"], window=14)
    group["rsi_7"] = ta.momentum.rsi(group["Close"], window=7)
    group["rsi_change"] = group["rsi_14"].diff()

    # MACD
    macd = ta.trend.MACD(group["Close"])
    group["macd"] = macd.macd()
    group["macd_signal"] = macd.macd_signal()
    group["macd_hist"] = macd.macd_diff()

    # ATR
    group["atr"] = ta.volatility.average_true_range(
        high=group["High"], low=group["Low"], close=group["Close"], window=14
    )
    group["atr_percent"] = group["atr"] / group["Close"]

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=group["Close"], window=20, window_dev=2)
    group["bb_high"] = bb.bollinger_hband()
    group["bb_low"] = bb.bollinger_lband()
    group["bb_mid"] = bb.bollinger_mavg()
    group["bb_width"] = (group["bb_high"] - group["bb_low"]) / group["bb_mid"]
    group["bb_position"] = (group["Close"] - group["bb_low"]) / (group["bb_high"] - group["bb_low"])

    # VWAP (daily reset)
    group["Date"] = group["Datetime"].dt.date
    typical_price = (group["High"] + group["Low"] + group["Close"]) / 3
    cum_vp = (typical_price * group["Volume"]).groupby(group["Date"]).cumsum()
    cum_vol = group["Volume"].groupby(group["Date"]).cumsum()
    group["vwap"] = cum_vp / cum_vol
    group["vwap_distance"] = (group["Close"] - group["vwap"]) / group["vwap"]
    group.drop(columns=["Date"], inplace=True)

    # Volume features
    group["volume_sma_20"] = group["Volume"].rolling(20).mean()
    group["volume_ratio"] = group["Volume"] / group["volume_sma_20"]
    group["volume_change"] = group["Volume"].pct_change()

    # Candle structure
    group["candle_body"] = abs(group["Close"] - group["Open"])
    group["upper_wick"] = group["High"] - np.maximum(group["Open"], group["Close"])
    group["lower_wick"] = np.minimum(group["Open"], group["Close"]) - group["Low"]
    group["full_range"] = group["High"] - group["Low"]
    group["body_ratio"] = group["candle_body"] / (group["full_range"] + 1e-9)

    # Momentum
    group["momentum_3"] = group["Close"] - group["Close"].shift(3)
    group["momentum_5"] = group["Close"] - group["Close"].shift(5)
    group["roc_5"] = ta.momentum.roc(group["Close"], window=5)

    # Stochastic RSI
    stoch = ta.momentum.StochRSIIndicator(close=group["Close"], window=14)
    group["stoch_rsi"] = stoch.stochrsi()

    # Trend flags
    group["above_ema20"] = (group["Close"] > group["ema_20"]).astype(int)
    group["above_vwap"] = (group["Close"] > group["vwap"]).astype(int)
    group["bullish_candle"] = (group["Close"] > group["Open"]).astype(int)

    # Volatility compression
    group["rolling_std_10"] = group["Close"].rolling(10).std()
    group["rolling_std_20"] = group["Close"].rolling(20).std()
    group["volatility_ratio"] = group["rolling_std_10"] / group["rolling_std_20"]

    # Lag features
    for lag in [1, 2, 3, 5]:
        group[f"close_lag_{lag}"] = group["Close"].shift(lag)
        group[f"volume_lag_{lag}"] = group["Volume"].shift(lag)
        group[f"rsi_lag_{lag}"] = group["rsi_14"].shift(lag)

    return group

# ============================================================
# 4. Make predictions and find strongest breakout candidate
# ============================================================
def get_breakout_ticker(df):
    """
    df: raw long‑format data with columns Datetime,Open,High,Low,Close,Volume,Ticker
    Returns a DataFrame with one row per ticker, containing the probability
    of the most confident directional class and the corresponding signal.
    """
    # Engineer features per ticker
    feature_df = df.groupby("Ticker", group_keys=False).apply(create_features)
    # Drop rows with NaN (from rolling windows)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()
    if feature_df.empty:
        return None

    # The model was trained on mapped labels: -1→0, 0→1, 1→2
    # We'll keep the original features (excluding Datetime, Ticker, Target)
    # Target column doesn't exist in new data, so we just exclude Datetime & Ticker
    feature_cols = [c for c in feature_df.columns if c not in ["Datetime","Ticker","Target"]]
    X = feature_df[feature_cols]

    # Predict probabilities (class 0,1,2)
    proba = model.predict_proba(X)
    # proba columns: index 0 = mapped -1 (bearish), index 1 = mapped 0 (neutral), index 2 = mapped 1 (bullish)
    # Directional confidence = max(prob_bear, prob_bull)
    bear_prob = proba[:, 0]
    bull_prob = proba[:, 2]
    neutral_prob = proba[:, 1]
    max_dir_prob = np.maximum(bear_prob, bull_prob)
    direction = np.where(bull_prob > bear_prob, 1, -1)

    # Create result per ticker (take the most recent row for each ticker)
    feature_df["bull_prob"] = bull_prob
    feature_df["bear_prob"] = bear_prob
    feature_df["max_dir_prob"] = max_dir_prob
    feature_df["direction"] = direction

    # Group by Ticker and get the last row (most recent time)
    last_rows = feature_df.groupby("Ticker").last().reset_index()
    return last_rows[["Ticker","bull_prob","bear_prob","max_dir_prob","direction"]]

# ============================================================
# 5. Streamlit UI
# ============================================================
st.set_page_config(page_title="Breakout Scanner", layout="wide")
st.title("📈 5‑Minute Breakout Scanner")
st.markdown("Using a trained XGBoost model to identify which of 10 tickers is most likely to have a directional breakout in the next 15 minutes.")

if st.button("Scan Now"):
    with st.spinner("Fetching latest data and computing predictions..."):
        raw_data = fetch_data(TICKERS)
        if raw_data.empty:
            st.error("No data fetched. Check your tickers and internet connection.")
        else:
            results = get_breakout_ticker(raw_data)
            if results is None or results.empty:
                st.warning("Not enough recent data to compute features. Try again in a few minutes.")
            else:
                # Find the ticker with the highest directional probability
                best = results.loc[results["max_dir_prob"].idxmax()]
                st.success(f"🔝 **{best['Ticker']}** shows the strongest breakout signal!")
                col1, col2, col3 = st.columns(3)
                col1.metric("Signal", "Bullish 📈" if best['direction']==1 else "Bearish 📉")
                col2.metric("Confidence", f"{best['max_dir_prob']:.2%}")
                col3.metric("Bull / Bear Prob", f"{best['bull_prob']:.2%} / {best['bear_prob']:.2%}")

                # Show full table sorted by confidence
                st.subheader("All Scanned Tickers")
                display_df = results.sort_values("max_dir_prob", ascending=False)
                display_df["Signal"] = display_df["direction"].map({1:"Bull", -1:"Bear"})
                display_df["Confidence"] = display_df["max_dir_prob"].apply(lambda x: f"{x:.2%}")
                st.dataframe(display_df[["Ticker","Signal","Confidence","bull_prob","bear_prob"]],
                             use_container_width=True)
