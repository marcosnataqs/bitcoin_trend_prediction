import requests
import yfinance as yf
import pandas as pd

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

def get_bitcoin_data():
    btc = yf.Ticker("BTC-USD")
    data = btc.history(period="1mo")
    return get_last_7_days(data)


def get_last_7_days(data):
    end_date = data.index[-1]
    start_date = end_date - pd.Timedelta(days=7)
    return data.loc[start_date:end_date]