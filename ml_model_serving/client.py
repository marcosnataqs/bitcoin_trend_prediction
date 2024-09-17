import requests

# Production URL => https://bitcoin-trend-prediction.onrender.com
BASE_URL = "http://127.0.0.1:8000"


def predict_multiple_days(data_list) -> list:
    predictions = []
    for data in data_list:
        response = requests.post(f"{BASE_URL}/predict", json=data)
        if response.status_code == 200:
            predictions.append(
                (
                    data["date"],
                    response.json()["prediction"],
                    response.json()["probability"],
                )
            )
        else:
            print(f"Error for data: {data}")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
    return predictions


# Sample data for multiple days
sample_data_list = [
    {
        "date": "2024-09-09",
        "low": 54598.43359375,
        "close": 57019.53515625,
        "sentiment": -0.008963787555694587,
        "neg_sentiment": 0.06666666666666668,
        "close_ratio_2": 1.0194702959257298,
        "edit_2": 0.13333333333333333,
        "close_ratio_7": 1.0194735813495144,
        "close_ratio_365": 1.0933944881675497,
        "edit_365": 1.9114155251141551
    },
    {
        "date": "2024-09-10",
        "low": 56419.4140625,
        "close": 57648.7109375,
        "sentiment": -0.008963787555694587,
        "neg_sentiment": 0.06666666666666668,
        "close_ratio_2": 1.0054869225150231,
        "edit_2": 0.13333333333333333,
        "close_ratio_7": 1.0301500697745714,
        "close_ratio_365": 1.1035759554915516,
        "edit_365": 1.9102283105022833        
    },
]

predictions = predict_multiple_days(sample_data_list)
print("Predictions for multiple days:")
for date, prediction, probability in predictions:
    print(f"Date: {date}, Prediction: {prediction}, Probability: {probability}")
