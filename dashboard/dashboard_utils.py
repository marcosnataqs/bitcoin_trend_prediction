import requests
import yfinance as yf
import pandas as pd
import mwclient
import time
from datetime import datetime
from transformers import pipeline
from statistics import mean

sentiment_pipeline = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
data_inicio = datetime.strptime("2018-01-01", "%Y-%m-%d")
# data_inicio = datetime.strptime(str((date.today() - timedelta(days=30))), '%Y-%m-%d')


def extract_btc(data_inicio: datetime) -> pd.DataFrame:
    ticker = yf.Ticker("BTC-USD")
    btc = ticker.history(start=data_inicio)
    return btc


def format_base(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index).tz_localize(None)
    del df["Dividends"]
    del df["Stock Splits"]
    df.columns = [c.lower() for c in df.columns]
    return df


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


def extract_reviews() -> list:
    site = mwclient.Site("en.wikipedia.org")
    page = site.pages["Bitcoin"]
    revs = list(page.revisions(start=data_inicio, dir="newer"))
    revs = sorted(revs, key=lambda rev: rev["timestamp"])
    return revs


def find_sentiment(text):
    sent = sentiment_pipeline([text[:250]])[0]
    score = sent["score"]
    if sent["label"] == "NEGATIVE":
        score *= -1
    return score


def format_edits() -> dict:
    edits = {}
    revs = extract_reviews()
    for rev in revs:
        date = time.strftime("%Y-%m-%d", rev["timestamp"])
        if date not in edits:
            edits[date] = dict(sentiments=list(), edit_count=0)

        edits[date]["edit_count"] += 1
        comment = rev.get("comment", "")
        edits[date]["sentiments"].append(find_sentiment(comment))
    return edits


def clean_sentiment_base(sentiment_edits: dict) -> dict:
    edits = sentiment_edits
    for key in edits:
        if len(edits[key]["sentiments"]) > 0:
            edits[key]["sentiment"] = mean(edits[key]["sentiments"])
            edits[key]["neg_sentiment"] = len(
                [s for s in edits[key]["sentiments"] if s < 0]
            ) / len(edits[key]["sentiments"])
        else:
            edits[key]["sentiment"] = 0
            edits[key]["neg_sentiment"] = 0

        del edits[key]["sentiments"]
    return edits


def create_edits_df() -> pd.DataFrame:
    edits = clean_sentiment_base(format_edits())
    edits_df = pd.DataFrame.from_dict(edits, orient="index")
    edits_df.index = pd.to_datetime(edits_df.index)
    return edits_df


def improve_edits_df(edits_df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range(start=data_inicio, end=datetime.today())
    edits_df = edits_df.reindex(dates, fill_value=0)
    edits_df["edit_count"] = edits_df["edit_count"].shift(1)
    edits_df["sentiment"] = edits_df["sentiment"].shift(1)
    edits_df["neg_sentiment"] = edits_df["neg_sentiment"].shift(1)
    # edits_df['edit_2'] = edits_df["edit_count"].rolling(2, min_periods=1).mean()
    # edits_df['edit_365'] = edits_df["edit_count"].rolling(365, min_periods=1).mean()
    edits_df = edits_df.dropna()
    rolling_edits = edits_df.rolling(30, min_periods=30).mean()
    rolling_edits = rolling_edits.dropna()
    return rolling_edits


def get_sentiment_df() -> pd.DataFrame:
    edits_df = create_edits_df()
    improved_df = improve_edits_df(edits_df)
    return improved_df


def merge_dfs() -> pd.DataFrame:
    btc = format_base(extract_btc(data_inicio))
    df_sentiment = get_sentiment_df()
    data = btc.merge(df_sentiment, left_index=True, right_index=True)
    return data


def trends_col(df: pd.DataFrame) -> pd.DataFrame:
    horizons = [2, 7, 365]

    for horizon in horizons:
        rolling_averages = df.rolling(horizon, min_periods=1).mean()

        ratio_column = f"close_ratio_{horizon}"
        df[ratio_column] = df["close"] / rolling_averages["close"]

        edit_column = f"edit_{horizon}"
        df[edit_column] = rolling_averages["edit_count"]

    return df


def get_data() -> pd.DataFrame:
    df = merge_dfs()
    df = trends_col(df)
    return df.tail(1)


if __name__ == "__main__":
    df = get_data()
