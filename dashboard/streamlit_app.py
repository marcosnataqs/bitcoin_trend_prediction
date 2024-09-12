import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import plotly.graph_objects as go
import numpy as np


def get_prediction():
    # Get the latest Bitcoin data
    btc = yf.Ticker("BTC-USD")
    latest_data = btc.history(period="1d")

    # Prepare the data for prediction
    data = {
        "date": latest_data.index[0].strftime("%Y-%m-%d"),
        "open": latest_data["Open"].iloc[0],
        "high": latest_data["High"].iloc[0],
        "low": latest_data["Low"].iloc[0],
        "close": latest_data["Close"].iloc[0],
        "volume": latest_data["Volume"].iloc[0],
        # Add other features here
    }

    # Convert int64 to regular Python int
    data = {k: int(v) if isinstance(v, np.int64) else v for k, v in data.items()}

    # Make prediction request
    response = requests.post("https://bitcoin-trend-prediction.onrender.com/predict", json=data)
    if response.status_code == 200:
        return response.json()["prediction"], response.json()["probability"]
    else:
        return None, None


def get_bitcoin_data():
    btc = yf.Ticker("BTC-USD")
    data = btc.history(period="1mo")
    return get_last_7_days(data)


def get_last_7_days(data):
    end_date = data.index[-1]
    start_date = end_date - pd.Timedelta(days=7)
    return data.loc[start_date:end_date]


def get_fear_greed_index():
    url = "https://api.alternative.me/fng/?limit=7"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()["data"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="s")
        df = df.sort_values("timestamp")
        return df
    else:
        return None


st.title("💰 Bitcoin Trend Prediction Dashboard")

# Section 1: ML Model Prediction
st.header("Today's Bitcoin Trend Prediction")
prediction, probability = get_prediction()

if prediction is not None:
    trend = "Up" if prediction == 1 else "Down"
    icon = "⬆️" if prediction == 1 else "⬇️"
    st.subheader(f"Predicted Trend: {trend} {icon}")
    st.write(f"Probability: {probability:.2f}")
else:
    st.write("Unable to fetch prediction at the moment.")

# Section 2: Bitcoin Price Chart
st.header("Bitcoin Price (Last 7 Days)")
btc_data = get_bitcoin_data()

if not btc_data.empty:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=btc_data.index,
            open=btc_data["Open"],
            high=btc_data["High"],
            low=btc_data["Low"],
            close=btc_data["Close"],
            name="Bitcoin Price",
        )
    )

    # Add colored markers for trend
    for i in range(1, len(btc_data)):
        color = (
            "green"
            if btc_data["Close"].iloc[i] > btc_data["Close"].iloc[i - 1]
            else "red"
        )
        fig.add_trace(
            go.Scatter(
                x=[btc_data.index[i]],
                y=[btc_data["Close"].iloc[i]],
                mode="markers",
                marker=dict(color=color, size=10),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Bitcoin Price (Last 7 Days)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
    )
    st.plotly_chart(fig)
else:
    st.write("Unable to fetch Bitcoin data at the moment.")

# Section 3: Fear and Greed Index
st.header("Fear and Greed Index (Last 7 Days)")
fng_data = get_fear_greed_index()

if fng_data is not None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fng_data["timestamp"],
            y=fng_data["value"],
            mode="lines+markers",
            name="Fear and Greed Index",
        )
    )
    fig.update_layout(
        title="Fear and Greed Index (Last 7 Days)",
        xaxis_title="Date",
        yaxis_title="Index Value",
    )
    st.plotly_chart(fig)

    # Display the current Fear and Greed Index value and classification
    current_fng = fng_data.iloc[-1]
    st.subheader(
        f"Current Fear and Greed Index: {current_fng['value']} ({current_fng['value_classification']})"
    )
else:
    st.write("Unable to fetch Fear and Greed Index data at the moment.")
