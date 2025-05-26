import streamlit as st
import requests
import datetime
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

def plot_prediction(data, predictions):
    dates = [row["date"] for row in data]
    prices = [row["price"] for row in data]

    last_date = datetime.datetime.strptime(dates[-1], "%Y-%m-%d")
    for i, price in enumerate(predictions):
        next_date = (last_date + datetime.timedelta(days=i+1)).strftime("%Y-%m-%d")
        dates.append(next_date)
        prices.append(price)

    plt.figure(figsize=(12, 6))
    plt.plot(dates[:len(data)], prices[:len(data)], label="Historical", marker='o')
    plt.plot(dates[len(data):], prices[len(data):], label="Predicted", linestyle='--', marker='x', color='red')
    plt.title("Crypto Price with 7-Day Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

st.markdown("# 🪙 Crypto Price Predictor")
st.markdown("#### Select a coin below to get a 1-day price prediction:")

coin = st.selectbox("🔍 Choose a coin", ["bitcoin", "ethereum", "dogecoin", "cardano"])

st.markdown("---")  # Horizontal line to separate sections

if st.button("🚀 Predict Tomorrow's Price"):

    try:
        st.info(f"Fetching data for {coin}...")
        data = get_crypto_price(coin)
        filename = f"{coin}_history.csv"
        save_data_to_csv(data, filename)
        predicted_prices = predict_price_from_csv(filename)
        st.success(f"Predicted {coin.upper()} price for tomorrow: ${predicted_prices[0]:,.2f}")
        full_data = load_data_for_graph(filename)
        plot_prediction(full_data, predicted_prices)
    except Exception as e:
        st.error(f"❌ Something went wrong:\n\n{e}")

