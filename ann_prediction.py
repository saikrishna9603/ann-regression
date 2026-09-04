import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
DATA_PATH = "creditcard.csv"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    features = data.drop(columns=["Amount", "Class"])
    target = data["Amount"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    x_train_scaled = feature_scaler.fit_transform(x_train)
    x_test_scaled = feature_scaler.transform(x_test)
    y_train_scaled = target_scaler.fit_transform(
        pd.Series(np.log1p(y_train), index=y_train.index).to_numpy().reshape(-1, 1)
    ).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model.fit(x_train_scaled, y_train_scaled)

    predicted_log_amount = target_scaler.inverse_transform(
        model.predict(x_test_scaled).reshape(-1, 1)
    ).ravel()
    predictions = pd.Series(np.expm1(predicted_log_amount), index=y_test.index).clip(lower=0)

    metrics = {
        "dataset_rows": len(data),
        "target": "Amount",
        "features": features.shape[1],
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "iterations": model.n_iter_,
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r_squared": r2_score(y_test, predictions),
    }
    with open("ann_results.json", "w", encoding="utf-8") as results_file:
        json.dump(metrics, results_file, indent=2)

    print(f"Dataset rows: {len(data):,}")
    print(f"Regression target: Amount")
    print(f"Features used: {features.shape[1]}")
    print(f"Training rows used: {len(x_train):,}")
    print(f"Test rows: {len(x_test):,}")
    print(f"ANN iterations: {model.n_iter_}")
    print(f"MAE: ${mean_absolute_error(y_test, predictions):.2f}")
    print(f"RMSE: ${mean_squared_error(y_test, predictions) ** 0.5:.2f}")
    print(f"R-squared: {r2_score(y_test, predictions):.4f}")


if __name__ == "__main__":
    main()