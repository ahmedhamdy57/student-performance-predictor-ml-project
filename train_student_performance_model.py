"""
train.py
Trains a Linear Regression model on the Student Performance dataset (Kaggle)
and saves the trained model + metadata for use in the Streamlit app.

Dataset: nikhil7280/student-performance-multiple-linear-regression
"""

import os
import glob
import json

import pandas as pd
import kagglehub
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_dataset() -> pd.DataFrame:
    """Download the dataset via kagglehub and load it into a DataFrame."""
    path = kagglehub.dataset_download(
        "nikhil7280/student-performance-multiple-linear-regression"
    )
    print("Path to dataset files:", path)

    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {path}")

    df = pd.read_csv(csv_files[0])
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def preprocess(df: pd.DataFrame):
    """Clean column names and encode the categorical feature."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Encode Yes/No -> 1/0
    df["Extracurricular Activities"] = (
        df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    )

    feature_cols = [
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced",
    ]
    target_col = "Performance Index"

    X = df[feature_cols]
    y = df[target_col]

    return X, y, feature_cols


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "r2": round(r2_score(y_test, y_pred), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "rmse": round(mean_squared_error(y_test, y_pred) ** 0.5, 4),
    }

    print("Evaluation metrics on test set:")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v}")

    return model, metrics


def main():
    df = load_dataset()
    X, y, feature_cols = preprocess(df)
    model, metrics = train_and_evaluate(X, y)

    # Save model
    joblib.dump(model, "model.pkl")
    print("Saved trained model to model.pkl")

    # Save metadata (feature columns, metrics, coefficients) for the app / README
    metadata = {
        "feature_cols": feature_cols,
        "target_col": "Performance Index",
        "metrics": metrics,
        "coefficients": dict(zip(feature_cols, model.coef_.tolist())),
        "intercept": model.intercept_,
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata to model_metadata.json")


if __name__ == "__main__":
    main()
