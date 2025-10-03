import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Page config
st.set_page_config(page_title="Crypto Price Predictor", page_icon="📈", layout="wide")
st.title("📊 Crypto Price Predictor")
st.markdown("### Predict the next 7–14 days of major coins and view real-world economic news.")
st.markdown("---")

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Price Prediction", "Economic News"])

# Supported coins
coin_names = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Dogecoin": "dogecoin",
    "Cardano": "cardano",
    "Solana": "solana",
    "Polkadot": "polkadot"
}

# Fetch trending coins
@st.cache_data(ttl=600)
def get_trending_coins():
    url = "https://api.coingecko.com/api/v3/search/trending"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("coins", [])
    except:
        return []

trending_coins = get_trending_coins()
trending_coin_names = [coin["item"]["name"] for coin in trending_coins]
selected_trending_coins = st.sidebar.multiselect("Trending Coins", trending_coin_names, default=trending_coin_names[:2])

# Match CoinGecko IDs for selected trending coins
selected_coins = [coin["item"]["id"] for coin in trending_coins if coin["item"]["name"] in selected_trending_coins]

# Regular coin selector
selected_name = st.sidebar.selectbox("Choose a coin", list(coin_names.keys()))
selected_coins.append(coin_names[selected_name])

# News API key (replace with your own if needed)
NEWS_API_KEY = "cd069f7a560a4350b78974c71eedbf53"

@st.cache_data(ttl=600)
def get_economic_news():
    url = "https://newsapi.org/v2/top-headlines"
    params = {"category": "business", "language": "en", "pageSize": 5, "apiKey": NEWS_API_KEY}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get("articles", [])
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ Could not load news: {e}")
        return []

@st.cache_data(ttl=600)
def get_crypto_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": "90", "interval": "daily"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return [{"date": datetime.datetime.fromtimestamp(p[0] / 1000).strftime("%Y-%m-%d"), "price": p[1]} for p in data["prices"]]
    except:
        st.error(f"❌ Error fetching data for {coin_id}")
        return []

def save_data_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def predict_price_from_data(data):
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = (df["date"] - df["date"].min()).dt.days

    X = df[["day"]].values
    y = df["price"].values

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    last_day = df["day"].max()
    return [(i, model.predict(np.array([[last_day + i]]))[0]) for i in range(1, 8)]

def backtest_model(data, test_days=5):
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = (df["date"] - df["date"].min()).dt.days

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    results = []

    for i in range(1, test_days + 1):
        past = df.iloc[:-i]
        if len(past) < 2:
            continue

        model.fit(past[["day"]], past["price"])
        future_day = df.iloc[-i]["day"]
        predicted = model.predict(np.array([[future_day]]))[0]
        actual = df.iloc[-i]["price"]
        results.append((df.iloc[-i]["date"].strftime("%Y-%m-%d"), actual, predicted))

    if not results:
        return 0.0, []

    mae = mean_absolute_error([r[1] for r in results], [r[2] for r in results])
    return mae, results

def plot_prediction(data, predictions):
    dates = [row["date"] for row in data]
    prices = [row["price"] for row in data]
    last_date = datetime.datetime.strptime(dates[-1], "%Y-%m-%d")

    for i, price in predictions:
        dates.append((last_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d"))
        prices.append(price)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[:len(data)], y=prices[:len(data)], mode="lines+markers", name="Historical"))
    fig.add_trace(go.Scatter(x=dates[len(data):], y=prices[len(data):], mode="lines+markers", name="Predicted", line=dict(dash="dash")))

    fig.update_layout(title="Crypto Price Forecast", xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# === Pages ===
if page == "Price Prediction":
    st.header("📈 Crypto Price Prediction")
    for coin in selected_coins:
        st.subheader(f"Predictions for {coin.replace('-', ' ').title()}")
        with st.spinner(f"🔄 Generating prediction for {coin.replace('-', ' ').title()}..."):
            data = get_crypto_price(coin)
            if not data:
                st.error(f"❌ No data available for {coin.title()}.")
                continue

            predicted = predict_price_from_data(data)
            mae, backtest = backtest_model(data)
            st.success(f"Predicted price for tomorrow: ${predicted[0][1]:,.2f}")
            plot_prediction(data, predicted)
            st.write(f"**MAE**: ${mae:,.2f}")

            if backtest:
                with st.expander("See backtest results"):
                    bt_df = pd.DataFrame(backtest, columns=["Date", "Actual", "Predicted"])
                    st.dataframe(bt_df)

elif page == "Economic News":
    st.header("🌍 Economic News That Could Affect Crypto")
    news = get_economic_news()
    if not news:
        st.info("No news articles found.")
    else:
        for article in news:
            if article.get("urlToImage"):
                st.image(article["urlToImage"], width=600)
            st.markdown(f"**[{article['title']}]({article['url']})**")
            st.caption(f"*{article.get('source', {}).get('name', 'Unknown')}*")
            st.markdown("---")
