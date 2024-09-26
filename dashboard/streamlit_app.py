import streamlit as st
import yfinance as yf
import requests
import plotly.graph_objects as go
import numpy as np
import os

from dotenv import load_dotenv

from dashboard_utils import get_fear_greed_index, get_bitcoin_data, get_from_adls

load_dotenv()


def get_prediction() -> tuple[int, float]:
    # Get the latest Bitcoin data
    btc = yf.Ticker("BTC-USD")
    latest_data = btc.history(period="1d")
    data_load = get_from_adls(os.environ["CONTAINER_NAME"])

    # Prepare the data for prediction
    data = {
        "date": latest_data.index[0].strftime("%Y-%m-%d"),
        "low": data_load["low"].values.tolist(),
        "close": data_load["close"].values.tolist(),
        "sentiment": data_load["sentiment"].values.tolist(),
        "neg_sentiment": data_load["neg_sentiment"].values.tolist(),
        "close_ratio_2": data_load["close_ratio_2"].values.tolist(),
        "edit_2": data_load["edit_2"].values.tolist(),
        "close_ratio_7": data_load["close_ratio_7"].values.tolist(),
        "close_ratio_365": data_load["close_ratio_365"].values.tolist(),
        "edit_365": data_load["edit_365"].values.tolist(),
    }

    # print(data)

    # Convert int64 to regular Python int
    data = {k: int(v) if isinstance(v, np.int64) else v for k, v in data.items()}

    # Make prediction request
    response = requests.post(
        "https://bitcoin-trend-prediction.onrender.com/predict", json=data
    )
    if response.status_code == 200:
        return response.json()["prediction"], response.json()["probability"]
    else:
        return None, None


st.title("💰 Bitcoin Trend Prediction")

# Section 1: ML Model Prediction
st.header("Today's Bitcoin Trend Prediction")
prediction, probability = get_prediction()

if prediction is not None:
    trend = "Up" if prediction == 1 else "Down"
    icon = "🚀" if prediction == 1 else "⬇️"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"<h1 style='text-align: center;'>{icon}</h1>",
            unsafe_allow_html=True,
        )
    with col2:
        st.subheader(f"Predicted Trend: {trend}")
        st.progress(probability)
        st.text(f"Model Probability Metric: {probability:.2%}")
        st.text("Above 55% means trend is to go up otherwise down")
else:
    st.error("Unable to fetch prediction at the moment.")

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
