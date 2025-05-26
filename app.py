import streamlit as st
import requests

NEWS_API_KEY = "cd069f7a560a4350b78974c71eedbf53"  # Replace this with your real NewsAPI key

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

import datetime
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="📈",
    layout="wide"
)
@st.cache_data(ttl=600)
def get_crypto_price(coin_id="bitcoin"):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": "365", "interval": "daily"}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise ValueError(f"Error fetching data for '{coin_id}'. Check the coin name.")

    data = response.json()

    if 'prices' not in data:
        raise ValueError(f"API response for '{coin_id}' did not contain price data.")

    formatted_data = []
    for i in range(len(data['prices'])):
        timestamp = data['prices'][i][0] / 1000
        date = datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')
        price = data['prices'][i][1]
        volume = data['total_volumes'][i][1]
        formatted_data.append({"date": date, "price": price, "volume": volume})
    return formatted_data

def save_data_to_csv(data, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["date", "price", "volume"])
        if not file_exists:
            writer.writeheader()
        for row in data:
            if not row_exists(row["date"], filename):
                writer.writerow(row)

def row_exists(date, filename):
    if not os.path.isfile(filename):
        return False
    with open(filename, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["date"] == date:
                return True
    return False

def predict_price_from_csv(filename):
    df = pd.read_csv(filename)
    df["date_obj"] = pd.to_datetime(df["date"])
    df["days"] = (df["date_obj"] - df["date_obj"].min()).dt.days
    X = df[["days", "volume"]].values
    y = df["price"].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_predictions = []
    last_day = df["days"].max()
    last_volume = df["volume"].iloc[-1]  # Use current volume for simplicity

    for i in range(1, 8):  # 7 future days
        future_day = last_day + i
        predicted_price = model.predict([[future_day, last_volume]])
        future_predictions.append((future_day, predicted_price[0]))

    return future_predictions

def load_data_for_graph(filename):
    full_data = []
    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_data.append({"date": row["date"], "price": float(row["price"])})
    return full_data


import plotly.graph_objects as go

def plot_prediction(data, predictions):
    dates = [row["date"] for row in data]
    prices = [row["price"] for row in data]

    # Extend with predicted dates and prices
    last_date = datetime.datetime.strptime(dates[-1], "%Y-%m-%d")
    for i, price in enumerate(predictions):
        next_date = (last_date + datetime.timedelta(days=i + 1)).strftime("%Y-%m-%d")
        dates.append(next_date)
        prices.append(price)

    # Split data
    historical_dates = dates[:len(data)]
    historical_prices = prices[:len(data)]
    future_dates = dates[len(data):]
    future_prices = prices[len(data):]

    fig = go.Figure()

    # Historical trace
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_prices,
        mode="lines+markers",
        name="Historical",
        line=dict(color="blue")
    ))

    # Prediction trace
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_prices,
        mode="lines+markers",
        name="Predicted",
        line=dict(color="red", dash="dash")
    ))

    # 🔵 Highlight the first predicted point
    if future_dates:
        fig.add_trace(go.Scatter(
            x=[future_dates[0]],
            y=[future_prices[0]],
            mode="markers+text",
            name="Next Prediction",
            marker=dict(color="darkred", size=12, symbol="star"),
            text=[f"${future_prices[0]:,.2f}"],
            textposition="top center",
            showlegend=True
        ))

        # Optional: shaded area for prediction range
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





st.markdown("# 🪙 Crypto Price Predictor")
st.markdown("#### Predict the next 7 days of major coins and see live economic news.")
st.markdown("---")

coin_names = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Dogecoin": "dogecoin",
    "Cardano": "cardano"
}

selected_name = st.selectbox("Choose a coin", list(coin_names.keys()))
coin = coin_names[selected_name]  # CoinGecko-compatible ID


st.markdown("---")  # Horizontal line to separate sections
if st.button("Predict Tomorrow's Price"):
    with st.spinner("🔄 Fetching data and generating prediction..."):
        try:
            st.info("Fetching data...")
            data = get_crypto_price(coin)
            filename = f"{coin}_history.csv"
            save_data_to_csv(data, filename)
            predicted_prices = predict_price_from_csv(filename)
            st.success(f"Predicted price for tomorrow: ${predicted_prices[0][1]:,.2f}")
            full_data = load_data_for_graph(filename)
            plot_prediction(full_data, predicted_prices)

            # ✅ Economic news section
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


