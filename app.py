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
    params = {"vs_currency": "usd", "days": "7", "interval": "daily"}
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
    next_day = df["days"].max() + 1
    next_volume = df["volume"].iloc[-1]
    prediction = model.predict([[next_day, next_volume]])
    return prediction[0]

def load_data_for_graph(filename):
    full_data = []
    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_data.append({"date": row["date"], "price": float(row["price"])})
    return full_data

def plot_prediction(data, predicted_price):
    dates = [row["date"] for row in data]
    prices = [row["price"] for row in data]
    last_date = datetime.datetime.strptime(dates[-1], "%Y-%m-%d")
    next_date = (last_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    dates.append(next_date)
    prices.append(predicted_price)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates[:-1], prices[:-1], label="Historical Prices", marker='o')
    ax.plot(dates[-2:], prices[-2:], label="Predicted Price", marker='x', linestyle='--', color='red')
    ax.set_title("Crypto Price and Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    plt.xticks(rotation=45)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

st.title("📈 Crypto Price Predictor")
coin = st.selectbox("Choose a cryptocurrency", ["bitcoin", "ethereum", "dogecoin", "cardano"])

if st.button("Predict Tomorrow's Price"):
    try:
        st.info(f"Fetching data for {coin}...")
        data = get_crypto_price(coin)
        filename = f"{coin}_history.csv"
        save_data_to_csv(data, filename)
        predicted_price = predict_price_from_csv(filename)
        st.success(f"Predicted {coin.upper()} price for tomorrow: ${predicted_price:,.2f}")
        full_data = load_data_for_graph(filename)
        plot_prediction(full_data, predicted_price)
    except Exception as e:
        st.error(f"❌ Something went wrong:\n\n{e}")

