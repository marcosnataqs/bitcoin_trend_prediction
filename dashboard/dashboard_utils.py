import requests
import yfinance as yf
import pandas as pd
import mwclient
import time
from datetime import datetime
from transformers import pipeline
from statistics import mean

sentiment_pipeline = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
data_inicio = datetime.strptime('2018-01-01', '%Y-%m-%d')

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

def clean_sentiment_base(sentiment_edits:dict) -> dict:
    edits = sentiment_edits
    for key in edits:
        if len(edits[key]["sentiments"]) > 0:
            edits[key]["sentiment"] = mean(edits[key]["sentiments"])
            edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentiments"] if s < 0]) / len(edits[key]["sentiments"])
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


if __name__ == '__main__':
    edits_df = create_edits_df()
    print(edits_df.head())


