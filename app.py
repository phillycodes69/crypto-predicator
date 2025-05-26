import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Page setup
st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="📈",
    layout="wide"
)
st.markdown("# 📊 Crypto Price Predictor")
st.markdown("### Predict the next 7 days of major coins and see live economic news.")
st.markdown("---")

# CoinGecko-compatible coin IDs
coin_names = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Dogecoin": "dogecoin",
    "Cardano": "cardano"
}

selected_name = st.selectbox("Choose a coin", list(coin_names.keys()))
coin = coin_names[selected_name]

# News API
NEWS_API_KEY = "your_newsapi_key_here"

@st.cache_data(ttl=600)
def get_economic_news():
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "category": "business",
        "language": "en",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json().get("articles", [])

@st.cache_data(ttl=600)
def get_crypto_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": "90", "interval": "daily"}
    response = requests.get(url, params=params)
    data = response.json()

    price_data = []
    for entry in data["prices"]:
        date = datetime.datetime.fromtimestamp(entry[0] / 1000).strftime("%Y-%m-%d")
        price = entry[1]
        price_data.append({"date": date, "price": price})
    return price_data

def save_data_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def load_data_for_graph(filename):
    df = pd.read_csv(filename)
    return df.to_dict(orient="records")

def predict_price_from_csv(filename):
    df = pd.read_csv(filename)
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = (df["date"] - df["date"].min()).dt.days

    X = df[["day"]].values
    y = df["price"].values

    model = LinearRegression()
    model.fit(X, y)

    future_predictions = []
    last_day = df["day"].max()
    for i in range(1, 8):
        future_day = last_day + i
        predicted_price = model.predict(np.array([[future_day]]))
        future_predictions.append((i, predicted_price[0]))

    return future_predictions

def plot_prediction(data, predictions):
    dates = [row["date"] for row in data]
    prices = [row["price"] for row in data]

    last_date = datetime.datetime.strptime(dates[-1], "%Y-%m-%d")
    for i, price in predictions:
        next_date = (last_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        dates.append(next_date)
        prices.append(price)

    historical_dates = dates[:len(data)]
    historical_prices = prices[:len(data)]
    future_dates = dates[len(data):]
    future_prices = prices[len(data):]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_dates, y=historical_prices, mode="lines+markers", name="Historical", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode="lines+markers", name="Predicted", line=dict(color="red", dash="dash")))

    if future_dates:
        fig.add_trace(go.Scatter(
            x=[future_dates[0]],
            y=[future_prices[0]],
            mode="markers+text",
            name="Next Prediction",
            marker=dict(color="darkred", size=12, symbol="star"),
            text=[f"${future_prices[0]:,.2f}"],
            textposition="top center"
        ))

        fig.add_vrect(
            x0=future_dates[0],
            x1=future_dates[-1],
            fillcolor="rgba(255, 0, 0, 0.05)",
            layer="below",
            line_width=0,
            annotation_text="Prediction Zone",
            annotation_position="top left"
        )

    fig.update_layout(
        title="Crypto Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# === Main app logic ===
if st.button("Predict Tomorrow's Price"):
    with st.spinner("🔄 Fetching data and generating prediction..."):
        try:
            data = get_crypto_price(coin)
            filename = f"{coin}_history.csv"
            save_data_to_csv(data, filename)
            predicted_prices = predict_price_from_csv(filename)
            full_data = load_data_for_graph(filename)

            day, price = predicted_prices[0]
            st.success(f"Predicted price for tomorrow: ${price:,.2f}")

            plot_prediction(full_data, predicted_prices)
st.markdown("## 🌍 Economic News That Could Affect Crypto")
try:
    news = get_economic_news()
    if not news:
        st.info("No news articles found.")
    else:
        for article in news:
            st.markdown(f"**[{article['title']}]({article['url']})**")
            st.caption(f"*Source: {article['source']['name']}*")
            st.markdown("---")
 except Exception as e:
    st.warning("⚠️ Could not load news articles.")

            st.markdown("## 🌍 Economic News That Could Affect Crypto")
            try:
                news = get_economic_news()
                for article in news:
                    st.markdown(f"**[{article['title']}]({article['url']})**")
                    st.caption(f"*Source: {article['source']['name']}*")
                    st.markdown("---")
            except Exception as e:
                st.warning("⚠️ Could not load news articles.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
