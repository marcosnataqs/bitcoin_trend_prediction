import requests


def predict_multiple_days(data_list) -> list:
    predictions = []
    for data in data_list:
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
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
        "open": 54851.88671875,
        "high": 58041.125,
        "low": 54598.43359375,
        "close": 57019.53515625,
        "volume": 34618096173,
        "close_ratio_2": 1.0194702959257298,
        "edit_2": 0.13333333333333333,
        "trend_2": 1.0,
        "close_ratio_7": 1.0194735813495144,
        "edit_7": 0.1476190476190476,
        "trend_7": 0.5714285714285714,
        "close_ratio_30": 0.965696930610218,
        "edit_30": 0.28,
        "trend_30": 0.5,
        "close_ratio_60": 0.9322505139296448,
        "edit_60": 0.2833333333333333,
        "trend_60": 0.5,
        "close_ratio_365": 1.0933944881675497,
        "edit_365": 1.9114155251141551,
        "trend_365": 0.5232876712328767,
        "edit_count_y": 0.13333333333333333,
        "sentiment_y": -0.008963787555694587,
        "neg_sentiment_y": 0.06666666666666668,
        "fng_index_y": 26,
    },
    {
        "date": "2024-09-10",
        "open": 57020.09765625,
        "high": 58029.9765625,
        "low": 56419.4140625,
        "close": 57648.7109375,
        "volume": 28857630507,
        "close_ratio_2": 1.0054869225150231,
        "edit_2": 0.13333333333333333,
        "trend_2": 1.0,
        "close_ratio_7": 1.0301500697745714,
        "edit_7": 0.14285714285714285,
        "trend_7": 0.7142857142857143,
        "close_ratio_30": 0.9769433655086518,
        "edit_30": 0.2711111111111111,
        "trend_30": 0.5333333333333333,
        "close_ratio_60": 0.9426017366450857,
        "edit_60": 0.2827777777777778,
        "trend_60": 0.5,
        "close_ratio_365": 1.1035759554915516,
        "edit_365": 1.9102283105022833,
        "trend_365": 0.5260273972602739,
        "edit_count_y": 0.13333333333333333,
        "sentiment_y": -0.008963787555694587,
        "neg_sentiment_y": 0.06666666666666668,
        "fng_index_y": 33,
    },
]

predictions = predict_multiple_days(sample_data_list)
print("Predictions for multiple days:")
for date, prediction, probability in predictions:
    print(f"Date: {date}, Prediction: {prediction}, Probability: {probability}")
