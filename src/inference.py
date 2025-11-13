import pandas as pd
import joblib
from pathlib import Path


def load_model():
    model_path = Path(__file__).resolve().parents[1] / "models" / "random_forest_model.pkl"
    model = joblib.load(model_path)
    return model


def predict_single(input_dict):
    model = load_model()
    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]
    return prediction


if __name__ == "__main__":
    sample = {
        "CRIM": 0.03,
        "ZN": 18.0,
        "INDUS": 2.31,
        "CHAS": 0,
        "NOX": 0.538,
        "RM": 6.5,
        "AGE": 65.2,
        "DIS": 4.1,
        "RAD": 1,
        "TAX": 296,
        "PTRATIO": 15.3,
        "B": 396.9,
        "LSTAT": 5.0
    }

    pred = predict_single(sample)
    print(f"Predicted MEDV: {pred:.2f}")