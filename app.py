import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from io import BytesIO
from PIL import Image

# --------------- Page config ---------------
st.set_page_config(page_title="Crypto Price Predictor", page_icon="📈", layout="wide")
st.title("📊 Crypto Price Predictor")
st.caption("Predict the next 7 days of major coins and browse business/economic headlines.")
st.divider()

# --------------- Sidebar ---------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Price Prediction", "Economic News"])

# Supported coins (stable, widely-listed ids)
COINS = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Solana": "solana",
    "Cardano": "cardano",
    "Polkadot": "polkadot",
    "Dogecoin": "dogecoin",
}

# Mappings for alternate providers
SYMBOL_MAP = {  # CryptoCompare symbols
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "cardano": "ADA",
    "polkadot": "DOT",
    "dogecoin": "DOGE",
}
PAPRIKA_ID = {  # CoinPaprika ids
    "bitcoin": "btc-bitcoin",
    "ethereum": "eth-ethereum",
    "solana": "sol-solana",
    "cardano": "ada-cardano",
    "polkadot": "dot-polkadot",
    "dogecoin": "doge-dogecoin",
}

# --------------- Secrets / Keys ---------------
# Add this in Streamlit Cloud -> App -> Settings -> Secrets:
# [api_keys]
# news = "YOUR_NEWSAPI_KEY"
NEWS_API_KEY = st.secrets.get("api_keys", {}).get("news", "")

# --------------- Utilities ---------------
def _fmt_date(ts_seconds: int) -> str:
    return datetime.utcfromtimestamp(ts_seconds).strftime("%Y-%m-%d")

def _with_retries(fn, retries=2, delay=0.8):
    last_err = None
    for _ in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(delay)
    raise last_err

# --------------- Data Providers (with fallbacks) ---------------
@st.cache_data(ttl=600)
def get_crypto_price(coin_id: str):
    """
    Returns ~90 days of daily close prices as:
      [ { 'date': 'YYYY-MM-DD', 'price': float }, ... ]
    Tries providers in order: CryptoCompare -> CoinPaprika -> CoinCap -> CoinGecko
    """

    def fetch_cryptocompare():
        sym = SYMBOL_MAP.get(coin_id)
        if not sym:
            raise ValueError("Symbol not mapped for CryptoCompare.")
        url = "https://min-api.cryptocompare.com/data/v2/histoday"
        params = {"fsym": sym, "tsym": "USD", "limit": 90}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        payload = r.json()
        if payload.get("Response") != "Success":
            raise RuntimeError(payload.get("Message", "CryptoCompare error"))
        rows = payload["Data"]["Data"]
        out = [{"date": _fmt_date(row["time"]), "price": float(row["close"])} for row in rows if row.get("close")]
        if not out:
            raise RuntimeError("Empty data from CryptoCompare.")
        return out

    def fetch_coinpaprika():
        pid = PAPRIKA_ID.get(coin_id)
        if not pid:
            raise ValueError("Paprika id not mapped.")
        end = datetime.utcnow().date()
        start = end - timedelta(days=90)
        url = f"https://api.coinpaprika.com/v1/tickers/{pid}/historical"
        params = {"start": start.isoformat(), "end": end.isoformat(), "interval": "1d"}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()  # list of dicts with 'price' and 'timestamp'
        out = []
        for d in data:
            ts = d.get("timestamp")
            price = d.get("price")
            if ts and (price is not None):
                out.append({"date": ts[:10], "price": float(price)})
        if not out:
            raise RuntimeError("Empty data from CoinPaprika.")
        return out

    def fetch_coincap():
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=90)
        url = f"https://api.coincap.io/v2/assets/{coin_id}/history"
        params = {
            "interval": "d1",
            "start": int(start_dt.timestamp() * 1000),  # ms
            "end": int(end_dt.timestamp() * 1000),      # ms
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
        out = [{"date": d["date"][:10], "price": float(d["priceUsd"])} for d in data if "priceUsd" in d]
        if not out:
            raise RuntimeError("Empty data from CoinCap.")
        return out

    def fetch_coingecko():
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": "90", "interval": "daily"}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        prices = r.json().get("prices", [])
        out = [{"date": datetime.utcfromtimestamp(p[0] / 1000).strftime("%Y-%m-%d"), "price": float(p[1])} for p in prices]
        if not out:
            raise RuntimeError("Empty data from CoinGecko.")
        return out

    providers = [
        ("CryptoCompare", fetch_cryptocompare),
        ("CoinPaprika", fetch_coinpaprika),
        ("CoinCap", fetch_coincap),
        ("CoinGecko", fetch_coingecko),
    ]

    last_err = None
    for name, fn in providers:
        try:
            return fn()
        except Exception as e:
            last_err = e
            st.info(f"ℹ️ Falling back from {name}: {e}")
            continue

    st.error(f"❌ All data providers failed for '{coin_id}'. Last error: {last_err}")
    return []

@st.cache_data(ttl=600)
def get_trending_coins():
    """
    Returns a lightweight trending list using CryptoCompare total volume (public).
    Falls back to the built-in COINS list if the call fails.
    """
    try:
        url = "https://min-api.cryptocompare.com/data/top/totalvolfull"
        params = {"tsym": "USD", "limit": 10}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json().get("Data", [])
        items = []
        for row in data:
            info = row.get("CoinInfo", {})
            name = info.get("FullName")
            symbol = info.get("Name")
            if not name or not symbol:
                continue
            # Try map back to our known ids by symbol (safe subset)
            mapped = None
            for k, v in SYMBOL_MAP.items():
                if v == symbol:
                    mapped = k
                    break
            if mapped:
                items.append({"label": name, "coin_id": mapped})
        # Ensure at least something
        if not items:
            raise RuntimeError("Empty trending list")
        return items
    except Exception:
        # Fallback to our curated set
        return [{"label": k, "coin_id": v} for k, v in COINS.items()]

@st.cache_data(ttl=600)
def get_economic_news():
    """NewsAPI headlines (optional key). Gracefully no-op if missing or request fails."""
    if not NEWS_API_KEY:
        return []
    url = "https://newsapi.org/v2/top-headlines"
    params = {"category": "business", "language": "en", "pageSize": 6, "apiKey": NEWS_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("articles", []) or []
    except Exception:
        return []

# --------------- Model / Plotting ---------------
def predict_next_7_days(data):
    df = pd.DataFrame(data)
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day"]].values
    y = df["price"].values
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X, y)
    last_day = int(df["day"].max())
    preds = []
    for i in range(1, 8):
        preds.append((i, float(model.predict([[last_day + i]])[0])))
    return preds

def backtest_mae(data, test_days=5):
    df = pd.DataFrame(data)
    if df.empty:
        return 0.0, []
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = (df["date"] - df["date"].min()).dt.days
    results = []
    for i in range(1, test_days + 1):
        train_df = df.iloc[:-i]
        if len(train_df) < 2:
            continue
        model = RandomForestRegressor(n_estimators=120, random_state=42)
        model.fit(train_df[["day"]], train_df["price"])
        row = df.iloc[-i]
        pred = float(model.predict([[row["day"]]])[0])
        results.append((row["date"].strftime("%Y-%m-%d"), float(row["price"]), pred))
    if not results:
        return 0.0, []
    mae = float(mean_absolute_error([r[1] for r in results], [r[2] for r in results]))
    return mae, results

def plot_series(data, predictions, title):
    if not data:
        st.warning("No data to plot.")
        return
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    hist_dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    hist_prices = df["price"].tolist()

    last_dt = df["date"].max()
    pred_dates = [(last_dt + timedelta(days=i)).strftime("%Y-%m-%d") for (i, _) in predictions]
    pred_prices = [p[1] for p in predictions]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_dates, y=hist_prices, mode="lines+markers", name="Historical"))
    if pred_dates:
        fig.add_trace(go.Scatter(x=pred_dates, y=pred_prices, mode="lines+markers", name="Predicted", line=dict(dash="dash")))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------- Pages ---------------
if page == "Price Prediction":
    st.header("📈 Crypto Price Prediction")

    # Trending (safe)
    with st.sidebar.expander("🔥 Trending / Popular"):
        trending = get_trending_coins()
        human_labels = [t["label"] for t in trending]
        picks = st.multiselect("Select coins to predict", human_labels, default=human_labels[:2])
        selected_from_trending = [t["coin_id"] for t in trending if t["label"] in picks]

    # Manual picker
    pick_one = st.sidebar.selectbox("Or choose one", list(COINS.keys()))
    selected_ids = set(selected_from_trending + [COINS[pick_one]])

    # Predict each selected coin
    for cid in selected_ids:
        nice_name = cid.replace("-", " ").title()
        st.subheader(f"{nice_name}")
        with st.spinner(f"Fetching data & predicting for {nice_name}…"):
            data = get_crypto_price(cid)
            if not data:
                st.error(f"No data available for {nice_name}.")
                continue
            preds = predict_next_7_days(data)
            mae, back = backtest_mae(data, test_days=5)
            if preds:
                st.success(f"Predicted price for tomorrow: ${preds[0][1]:,.2f}")
            plot_series(data, preds, f"{nice_name} — 7-Day Forecast")
            st.caption(f"MAE (5-day backtest): ${mae:,.2f}")
            if back:
                with st.expander("Backtest (actual vs predicted)"):
                    st.dataframe(pd.DataFrame(back, columns=["Date", "Actual", "Predicted"]))

elif page == "Economic News":
    st.header("🌍 Business & Economic News")
    if not NEWS_API_KEY:
        st.info("Add your NewsAPI key in Streamlit **Secrets** to enable headlines. (Settings → Secrets → `[api_keys] news = \"...\"`)")
    else:
        articles = get_economic_news()
        if not articles:
            st.info("No articles available right now.")
        else:
            for a in articles:
                title = a.get("title") or "Untitled"
                url = a.get("url") or "#"
                src = a.get("source", {}).get("name", "Unknown Source")
                st.markdown(f"**[{title}]({url})**")
                img_url = a.get("urlToImage")
                if img_url:
                    try:
                        resp = requests.get(img_url, timeout=10)
                        img = Image.open(BytesIO(resp.content))
                        if img.format != "GIF":
                            st.image(img, use_column_width=True)
                    except Exception:
                        pass
                st.caption(src)
                st.divider()
