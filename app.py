import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import io
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from vaderSentiment.varderSentiment import SentimentIntesityAnalyzer

# Page setup
st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.markdown("# 📊 Crypto Price Predictor")
st.markdown("### Predict the next 7–14 days of major coins and view real-world economic news.")
st.markdown("---")

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Price Prediction", "Economic News"])

# Coin selector in sidebar
coin_names = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Dogecoin": "dogecoin",
    "Cardano": "cardano"
}
selected_name = st.sidebar.selectbox("Choose a coin", list(coin_names.keys()))
coin = coin_names[selected_name]

# News API key
NEWS_API_KEY = "cd069f7a560a4350b78974c71eedbf53"


# === Sentiment Analysis ===
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    return score['compound']  # -1 (very negative) to +1 (very positive)

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

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    future_predictions = []
    last_day = df["day"].max()
    for i in range(1, 8):
        future_day = last_day + i
        predicted_price = model.predict(np.array([[future_day]]))
        future_predictions.append((i, predicted_price[0]))

    return future_predictions

def backtest_model(filename, test_days=5):
    df = pd.read_csv(filename)
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = (df["date"] - df["date"].min()).dt.days

    X = df[["day"]].values
    y = df["price"].values

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    backtest_results = []

    for i in range(1, test_days + 1):
        past_df = df.iloc[:-i]
        if len(past_df) < 2:
            continue

        future_actual = df.iloc[-i]
        past_X = past_df[["day"]].values
        past_y = past_df["price"].values

        model.fit(past_X, past_y)
        future_day = future_actual["day"]
        predicted = model.predict(np.array([[future_day]]))[0]
        actual = future_actual["price"]

        backtest_results.append((
            future_actual["date"].strftime("%Y-%m-%d"),
            actual,
            predicted
        ))

    if not backtest_results:
        return 0.0, []

    mae = mean_absolute_error(
        [r[1] for r in backtest_results],
        [r[2] for r in backtest_results]
    )

    return mae, backtest_results

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

# === Page Views ===
if page == "Price Prediction":
    st.header("📈 Crypto Price Prediction")
    if st.button("Predict Tomorrow's Price"):
        with st.spinner("🔄 Fetching data and generating prediction..."):
            try:
                data = get_crypto_price(coin)
                filename = f"{coin}_history.csv"
                save_data_to_csv(data, filename)
                predicted_prices = predict_price_from_csv(filename)
                full_data = load_data_for_graph(filename)
                mae, backtest_results = backtest_model(filename)

                day, price = predicted_prices[0]
                st.success(f"Predicted price for tomorrow: ${predicted_prices[0][1]:,.2f}")
                plot_prediction(full_data, predicted_prices)

                st.write(f"**MAE** (Mean Absolute Error): ${mae:,.2f}")

                if backtest_results:
                    with st.expander("See actual vs predicted"):
                        backtest_df = pd.DataFrame(backtest_results, columns=["Date", "Actual Price", "Predicted Price"])
                        st.dataframe(backtest_df)

                        csv = backtest_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Backtest Results as CSV",
                            data=csv,
                            file_name="backtest_results.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"❌ Error: {e}")

elif page == "Economic News":
    st.header("🌍 Economic News That Could Affect Crypto")
    try:
        news = get_economic_news()
        if not news:
            st.info("No news articles found from the API.")
        else:
            for article in news:
                st.image(article.get("urlToImage"), width=600)
                st.markdown(f"**[{article['title']}]({article['url']})**")
                st.caption(f"*{article.get('source', {}).get('name', 'Unknown Source')}*")
                st.markdown("---")
    except Exception as e:
        st.warning("⚠️ Could not load news articles.")
