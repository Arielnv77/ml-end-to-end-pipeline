import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data():
    data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "BostonHousing.csv"
    df = pd.read_csv(data_path)
    return df


def train_model(df):
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print("Training completed.")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return pipeline


def save_model(model):
    output_path = Path(__file__).resolve().parents[1] / "models" / "random_forest_model.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved at: {output_path}")


if __name__ == "__main__":
    df = load_data()
    model = train_model(df)
    save_model(model)