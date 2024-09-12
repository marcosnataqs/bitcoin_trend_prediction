# LitServe => https://github.com/Lightning-AI/litserve

import os
import joblib
import numpy as np
import litserve as ls
from dotenv import load_dotenv

load_dotenv()


class BTC_Trend_Prediction_API(ls.LitAPI):
    def setup(self, device):
        model_path = os.environ.get("MODEL_PATH")
        if model_path is None:
            raise ValueError(
                "MODEL_PATH environment variable is not set. Please set it to the path of your model file."
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Please check the MODEL_PATH environment variable."
            )

        self.model = joblib.load(model_path)
        self.feature_names = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_ratio_2",
            "edit_2",
            "trend_2",
            "close_ratio_7",
            "edit_7",
            "trend_7",
            "close_ratio_30",
            "edit_30",
            "trend_30",
            "close_ratio_60",
            "edit_60",
            "trend_60",
            "close_ratio_365",
            "edit_365",
            "trend_365",
            "edit_count_y",
            "sentiment_y",
            "neg_sentiment_y",
            "fng_index_y",
        ]
        self.default_values = {feature: 0 for feature in self.feature_names}

    def decode_request(self, request):
        features = {k: v for k, v in request.items() if k in self.feature_names}
        return features

    def predict(self, x):
        input_data = self.prepare_input(x)
        probabilities = self.model.predict_proba(input_data)
        class_1_prob = probabilities[:, 1]  # Probability for class 1
        prediction = int(class_1_prob >= 0.6)  # Apply threshold
        return prediction, class_1_prob

    def prepare_input(self, features):
        input_array = np.array([features.get(f, 0) for f in self.feature_names]).reshape(1, -1)
        return input_array

    def encode_response(self, output):
        prediction, probability = output
        return {"prediction": prediction, "probability": float(probability)}


if __name__ == "__main__":
    api = BTC_Trend_Prediction_API()
    server = ls.LitServer(api)
    server.run(port=8000, generate_client_file=False)
