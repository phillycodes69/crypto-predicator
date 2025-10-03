import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from PIL import Image
from io import BytesIO

# Set up the app page
st.set_page_config(page_title="Crypto Price Predictor", page_icon="📈", layout="wide")

st.title("📊 Crypto Price Predictor")
st.markdown("Predict prices for major cryptocurrencies and explore relevant economic news.")
st.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Price Prediction", "Economic News"])

# Coin options
coin_options = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Dogecoin": "dogecoin",
    "Cardano": "cardano",
    "Solana": "solana",
    "Polkadot": "polkadot"
}
selected_coin = st.sidebar.selectbox("Choose a cryptocurrency", list(coin_options.keys()))
coin_id = coin_options[selected_coin]

# 🔑 Add your NewsAPI key here
NEWS_API_KEY = "your_newsapi_key_here"

@st.cache_data(ttl=600)
def fetch_crypto_data(coin_id, days=90):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return []
    prices = response.json().get("prices", [])
    return [{"date": datetime.datetime.fromtimestamp(p[0] / 1000).strftime("%Y-%m-%d"), "price": p[1]} for p in prices]

def prepare_dataframe(data):
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = (df["date"] - df["date"].min()).dt.days
    return df

def train_predict(df, days_ahead=7):
    X = df[["day"]].values
    y = df["price"].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    last_day = df["day"].max()
    return [(i, model.predict([[last_day + i]])[0]) for i in range(1, days_ahead + 1)]

def calculate_backtest(df, test_days=5):
    results = []
    for i in range(1, test_days + 1):
        train_df = df.iloc[:-i]
        test_row = df.iloc[-i]
        if len(train_df) < 2:
            continue
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train_df[["day"]], train_df["price"])
        pred = model.predict([[test_row["day"]]])[0]
        results.append((test_row["date"].strftime("%Y-%m-%d"), test_row["price"], pred))
    if not results:
        return 0.0, []
    actual = [r[1] for r in results]
    predicted = [r[2] for r in results]
    return mean_absolute_error(actual, predicted), results

def plot_prices(df, predictions):
    hist_dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    hist_prices = df["price"].tolist()
    last_date = df["date"].max()
    pred_dates = [(last_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i, _ in predictions]
    pred_prices = [p[1] for p in predictions]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_dates, y=hist_prices, mode="lines+markers", name="Historical"))
    fig.add_trace(go.Scatter(x=pred_dates, y=pred_prices, mode="lines+markers", name="Predicted", line=dict(dash="dash")))
    fig.update_layout(title="Price Forecast", xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=600)
def fetch_news():
    url = "https://newsapi.org/v2/top-headlines"
    params = {"category": "business", "language": "en", "pageSize": 5, "apiKey": NEWS_API_KEY}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get("articles", [])
    except Exception as e:
        st.error(f"News fetch error: {e}")
        return []

def display_news(articles):
    for article in articles:
        st.markdown(f"**[{article.get('title', 'No Title')}]({article.get('url', '#')})**")
        img_url = article.get("urlToImage")
        if img_url:
            try:
                img_data = requests.get(img_url).content
                img = Image.open(BytesIO(img_data))
                if img.format != "GIF":
                    st.image(img, width=600)
            except:
                st.warning("Image couldn't be loaded.")
        st.caption(article.get("source", {}).get("name", "Unknown Source"))
        st.markdown("---")

# === PAGE LOGIC ===
if page == "Price Prediction":
    st.subheader(f"Price Prediction for {selected_coin}")
    if st.button("Predict"):
        with st.spinner("Fetching data and making predictions..."):
            data = fetch_crypto_data(coin_id)
            if not data:
                st.error("Failed to fetch data.")
            else:
                df = prepare_dataframe(data)
                predictions = train_predict(df)
                mae, backtest = calculate_backtest(df)
                plot_prices(df, predictions)
                st.success(f"Tomorrow's predicted price: ${predictions[0][1]:,.2f}")
                st.markdown(f"**MAE (Mean Absolute Error):** ${mae:.2f}")
                if backtest:
                    with st.expander("Backtest Results"):
                        st.dataframe(pd.DataFrame(backtest, columns=["Date", "Actual", "Predicted"]))

elif page == "Economic News":
    st.subheader("Latest Economic News")
    articles = fetch_news()
    if articles:
        display_news(articles)
    else:
        st.info("No articles available at the moment.")
