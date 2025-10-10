# app.py
import time
import requests
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from io import BytesIO
from PIL import Image

# ---------------- Page config ----------------
st.set_page_config(page_title="Crypto Price Predictor", page_icon="📈", layout="wide")
st.title("📊 Crypto Price Predictor")
st.caption("Predict the next 7 days of major coins and browse business/economic headlines.")
st.divider()

# ---------------- Sidebar ----------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Price Prediction", "Economic News"])

# Stable coin IDs (work across most providers)
COINS = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Solana": "solana",
    "Cardano": "cardano",
    "Polkadot": "polkadot",
    "Dogecoin": "dogecoin",
}

pick_one = st.sidebar.selectbox("Choose a coin", list(COINS.keys()))
coin_id = COINS[pick_one]

# ---------------- Secrets / Keys ----------------
# In Streamlit Cloud: Manage app → Settings → Secrets → add:
# [api_keys]
# news = "YOUR_NEWSAPI_KEY"
NEWS_API_KEY = st.secrets.get("api_keys", {}).get("news", "")

# ---------------- Helpers ----------------
def _fmt_date(ts_seconds: int) -> str:
    return datetime.utcfromtimestamp(ts_seconds).strftime("%Y-%m-%d")

# ---------------- Data Providers with Diagnostics ----------------
@st.cache_data(ttl=600)
def get_crypto_price(coin_id: str):
    """
    Returns:
      (data, debug_str)
        data: list[{'date':'YYYY-MM-DD','price':float}]
        debug_str: multiline string of what the fetchers did
    Tries: CoinGecko -> CoinPaprika -> CryptoCompare -> CoinCap
    """
    trace_lines = []
    def trace(msg): trace_lines.append(str(msg))

    # ---- 1) CoinGecko (public; no key) ----
    try:
        trace("Trying CoinGecko…")
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": "90", "interval": "daily"}
        headers = {
            "accept": "application/json",
            "user-agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0)"
        }
        r = requests.get(url, params=params, headers=headers, timeout=15)
        trace(f"CoinGecko status: {r.status_code}")
        r.raise_for_status()
        prices = r.json().get("prices", [])
        data = [{"date": _fmt_date(int(p[0] / 1000)), "price": float(p[1])} for p in prices]
        if data:
            trace(f"CoinGecko rows: {len(data)}")
            return data, "\n".join(trace_lines)
        trace("CoinGecko returned empty list.")
    except Exception as e:
        trace(f"CoinGecko error: {e}")

    # ---- 2) CoinPaprika (no key) ----
    PAPRIKA_ID = {
        "bitcoin": "btc-bitcoin",
        "ethereum": "eth-ethereum",
        "solana": "sol-solana",
        "cardano": "ada-cardano",
        "polkadot": "dot-polkadot",
        "dogecoin": "doge-dogecoin",
    }
    try:
        trace("Trying CoinPaprika…")
        pid = PAPRIKA_ID.get(coin_id)
        if pid:
            end = datetime.utcnow().date()
            start = end - timedelta(days=90)
            url = f"https://api.coinpaprika.com/v1/tickers/{pid}/historical"
            params = {"start": start.isoformat(), "end": end.isoformat(), "interval": "1d"}
            r = requests.get(url, params=params, timeout=15)
            trace(f"CoinPaprika status: {r.status_code}")
            r.raise_for_status()
            rows = r.json()
            data = []
            for d in rows:
                ts = d.get("timestamp")
                price = d.get("price")
                if ts and price is not None:
                    data.append({"date": ts[:10], "price": float(price)})
            if data:
                trace(f"CoinPaprika rows: {len(data)}")
                return data, "\n".join(trace_lines)
            trace("CoinPaprika returned empty list.")
        else:
            trace("No CoinPaprika ID mapping for this coin.")
    except Exception as e:
        trace(f"CoinPaprika error: {e}")

    # ---- 3) CryptoCompare (symbol-based; no key) ----
    SYMBOL_MAP = {
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "solana": "SOL",
        "cardano": "ADA",
        "polkadot": "DOT",
        "dogecoin": "DOGE",
    }
    try:
        trace("Trying CryptoCompare…")
        sym = SYMBOL_MAP.get(coin_id)
        if sym:
            url = "https://min-api.cryptocompare.com/data/v2/histoday"
            params = {"fsym": sym, "tsym": "USD", "limit": 90}
            r = requests.get(url, params=params, timeout=15)
            trace(f"CryptoCompare status: {r.status_code}")
            r.raise_for_status()
            payload = r.json()
            if payload.get("Response") == "Success":
                rows = payload["Data"]["Data"]
                data = [{"date": _fmt_date(row["time"]), "price": float(row["close"])} for row in rows if row.get("close")]
                if data:
                    trace(f"CryptoCompare rows: {len(data)}")
                    return data, "\n".join(trace_lines)
                trace("CryptoCompare returned empty list.")
            else:
                trace(f"CryptoCompare says: {payload.get('Message')}")
        else:
            trace("No CryptoCompare symbol mapping for this coin.")
    except Exception as e:
        trace(f"CryptoCompare error: {e}")

    # ---- 4) CoinCap (no key; ms epoch window) ----
    try:
        trace("Trying CoinCap…")
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=90)
        url = f"https://api.coincap.io/v2/assets/{coin_id}/history"
        params = {
            "interval": "d1",
            "start": int(start_dt.timestamp() * 1000),
            "end": int(end_dt.timestamp() * 1000),
        }
        r = requests.get(url, params=params, timeout=15)
        trace(f"CoinCap status: {r.status_code}")
        r.raise_for_status()
        rows = r.json().get("data", [])
        data = [{"date": d["date"][:10], "price": float(d["priceUsd"])} for d in rows if "priceUsd" in d]
        if data:
            trace(f"CoinCap rows: {len(data)}")
            return data, "\n".join(trace_lines)
        trace("CoinCap returned empty list.")
    except Exception as e:
        trace(f"CoinCap error: {e}")

    # Final: nothing worked
    return [], "\n".join(trace_lines)

# ---------------- News ----------------
@st.cache_data(ttl=600)
def get_economic_news():
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

# ---------------- ML + Plot ----------------
def predict_next_7_days(data):
    """
    Autoregressive forecast with lag/rolling features and recursive 7-day prediction.
    """
    df = pd.DataFrame(data).copy()
    if df.empty:
        return []

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["price"] = df["price"].astype(float)

    # Lag features
    for L in [1, 2, 3, 7, 14]:
        df[f"lag{L}"] = df["price"].shift(L)

    # Rolling means (use only past data)
    for W in [3, 7, 14]:
        df[f"roll{W}"] = df["price"].shift(1).rolling(W).mean()

    feature_cols = [c for c in df.columns if c.startswith("lag") or c.startswith("roll")]
    train = df.dropna(subset=feature_cols + ["price"])
    if train.empty:
        return []

    X = train[feature_cols].values
    y = train["price"].values

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    # Recursive forecast: feed each new prediction back in
    history = df[["date", "price"]].copy()
    preds = []
    last_date = history["date"].max()

    for i in range(1, 8):
        row = {}
        for L in [1, 2, 3, 7, 14]:
            row[f"lag{L}"] = history["price"].iloc[-L] if len(history) >= L else history["price"].iloc[-1]
        for W in [3, 7, 14]:
            window = history["price"].iloc[-W:] if len(history) >= W else history["price"]
            row[f"roll{W}"] = float(window.mean())

        x_next = np.array([row[c] for c in feature_cols]).reshape(1, -1)
        y_hat = float(model.predict(x_next)[0])
        preds.append((i, y_hat))

        history = pd.concat(
            [history, pd.DataFrame([{"date": last_date + pd.Timedelta(days=i), "price": y_hat}])],
            ignore_index=True
        )

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

# ---------------- Pages ----------------
if page == "Price Prediction":
    st.header(f"📈 Price Prediction — {pick_one}")
    nice_name = coin_id.replace("-", " ").title()

    with st.spinner(f"Fetching data & predicting for {nice_name}…"):
        data, debug = get_crypto_price(coin_id)

        # Diagnostics panel so you can see exactly what happened
        with st.expander("Diagnostics"):
            st.code(debug or "no trace")

        if not data:
            st.error(f"No data available for {nice_name}. See Diagnostics above.")
        else:
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
        st.info("""
        🔑 Add your NewsAPI key in Streamlit Cloud → **Manage app → Settings → Secrets**
        ```
        [api_keys]
        news = "YOUR_NEWSAPI_KEY"
        ```
        """)
    else:
        articles = get_economic_news()
        if not articles:
            st.info("No recent business or crypto-related news found.")
        else:
            st.write("### 📰 Latest Headlines")
            for a in articles:
                title = a.get("title", "Untitled")
                url = a.get("url", "#")
                source = a.get("source", {}).get("name", "Unknown Source")
                image_url = a.get("urlToImage")
                published = a.get("publishedAt", None)

                # Format date nicely
                published_str = ""
                if published:
                    try:
                        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                        hours_ago = (datetime.utcnow() - dt).total_seconds() / 3600
                        if hours_ago < 24:
                            published_str = f"{int(hours_ago)} hours ago"
                        else:
                            published_str = dt.strftime("%b %d, %Y")
                    except:
                        pass

                # --- Custom "card" style container ---
                st.markdown(
                    f"""
                    <div style="
                        background-color:#f9f9f9;
                        border-radius:12px;
                        box-shadow:0 2px 6px rgba(0,0,0,0.08);
                        padding:18px;
                        margin-bottom:20px;
                    ">
                        <h4 style="margin-bottom:4px;">
                            <a href="{url}" target="_blank" style="text-decoration:none; color:#1a73e8;">{title}</a>
                        </h4>
                        <p style="color:#555; font-size:14px; margin-bottom:6px;">{source} • {published_str}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Image (if available)
                if image_url:
                    st.image(image_url, use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/600x300?text=No+Image+Available", use_container_width=True)

