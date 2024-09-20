# LitServe => https://github.com/Lightning-AI/litserve

import os
import joblib
import numpy as np
import litserve as ls
from sklearn.preprocessing import StandardScaler
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
            "low",
            "close",
            "sentiment",
            "neg_sentiment",
            "close_ratio_2",
            "edit_2",
            "close_ratio_7",
            "close_ratio_365",
            "edit_365"
        ]

    def decode_request(self, request):
        features = {k: v for k, v in request.items() if k in self.feature_names}
        return features

    def predict(self, x):
        input_data = self.prepare_input(x)
        probabilities = self.model.predict_proba(input_data)
        class_1_prob = probabilities[:, 1]  # Probability for class 1
        prediction = int(class_1_prob >= 0.55)  # Apply threshold
        return prediction, class_1_prob

    def prepare_input(self, features):
        std_scaler = StandardScaler()
        input_array = np.array([features.get(f, 0) for f in self.feature_names]).reshape(1, -1)
        input_array = std_scaler.fit_transform(input_array)
        return input_array

    def encode_response(self, output):
        prediction, probability = output
        return {"prediction": prediction, "probability": float(probability)}


if __name__ == "__main__":
    api = BTC_Trend_Prediction_API()
    server = ls.LitServer(api)
    server.run(port=8000, generate_client_file=False)
