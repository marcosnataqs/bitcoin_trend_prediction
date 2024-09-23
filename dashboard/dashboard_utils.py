import requests
import yfinance as yf
import pandas as pd
import joblib
import os

from azure.storage.blob import BlobClient
from dotenv import load_dotenv
from io import StringIO
from typing import List

load_dotenv()


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


## Azure Data Consumption
def connect_to_adls(container_name: str, blob_name: str):
    connection_string = os.environ["CONNECTION_STRING_DL"]
    blob = BlobClient.from_connection_string(
        conn_str=connection_string,
        container_name=container_name,
        blob_name=blob_name
    )
    return blob

def get_from_adls(container_name: str):
    blob_name = 'raw/bitcoin/btc_sent.csv'
    blob = connect_to_adls(container_name, blob_name)
    blob_data = blob.download_blob()
    blob_content = blob_data.readall()
    csv_data = blob_content.decode('utf-8')
    data = pd.read_csv(StringIO(csv_data), index_col=0, parse_dates=[0])
    print('Download from conformed sucessful')
    return apply_scaler_to_dataframe(data.tail(1))

def apply_scaler_to_dataframe(df):
  """Applies a pre-trained scaler to a DataFrame using an environment variable.
  Args:
    df: The DataFrame to apply the scaler to.
  Returns:
    The scaled DataFrame.
  """
  predictors: List[str] = ['low', 'close', 'sentiment', 'neg_sentiment', 'close_ratio_2', 'edit_2',
       'close_ratio_7', 'close_ratio_365', 'edit_365']
  scaler_path = 'scaler.joblib'
  scaler = joblib.load(scaler_path)
  scaled_data = scaler.transform(df[predictors])
  scaled_df = pd.DataFrame(scaled_data, columns=df[predictors].columns)
  return scaled_df


if __name__ == "__main__":
    #df = get_data()

    data = get_from_adls("bronze")
    print(data)
    print(type(data))
