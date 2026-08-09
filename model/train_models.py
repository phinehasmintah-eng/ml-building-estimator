"""
train_models.py

Trains and evaluates multiple ML regression models on the synthesized
construction dataset, for two targets:
  1) total_cost_ghs        (cost estimation)
  2) total_cement_bags     (material quantity estimation, representative
                             of the quantity-prediction task)

Pipeline:
  - One-hot encode categorical features, scale numeric features
  - 70/15/15 train/validation/test split
  - 10-fold cross-validation on the training set for model selection
  - Final evaluation on the held-out test set
  - Metrics: MAE, RMSE, R^2, MAPE (mean absolute percentage error, used to
    report an "accuracy" figure comparable to the assessor's requested
    error-rate framing)
  - Saves the best model + preprocessing pipeline with joblib for use by app.py

Run: python train_models.py
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

CATEGORICAL = ["structural_type", "wall_material", "region", "finish_quality"]
NUMERIC = ["floor_area_m2", "storeys"]
TARGETS = ["total_cost_ghs", "total_cement_bags"]

MODELS = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
}


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
    }


def run_for_target(df, target):
    print(f"\n===== TARGET: {target} =====")
    X = df[NUMERIC + CATEGORICAL]
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    results = {}
    fitted_pipelines = {}
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for name, model in MODELS.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")

        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        val_metrics = evaluate(y_val, val_pred)

        results[name] = {
            "cv_r2_mean": float(np.mean(cv_scores)),
            "cv_r2_std": float(np.std(cv_scores)),
            "val_metrics": val_metrics,
        }
        fitted_pipelines[name] = pipe

        print(f"{name}: 10-fold CV R2 = {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f}) | "
              f"Val MAE={val_metrics['MAE']:.2f} RMSE={val_metrics['RMSE']:.2f} "
              f"R2={val_metrics['R2']:.4f} MAPE={val_metrics['MAPE_%']:.2f}%")

    # pick best by validation R2
    best_name = max(results, key=lambda n: results[n]["val_metrics"]["R2"])
    best_pipe = fitted_pipelines[best_name]

    test_pred = best_pipe.predict(X_test)
    test_metrics = evaluate(y_test, test_pred)
    print(f"\nBest model for {target}: {best_name}")
    print(f"Held-out TEST metrics: {test_metrics}")

    results["best_model"] = best_name
    results["test_metrics"] = test_metrics

    return best_pipe, results


def main():
    df = pd.read_csv("../data/construction_dataset.csv")

    all_results = {}
    for target in TARGETS:
        best_pipe, results = run_for_target(df, target)
        joblib.dump(best_pipe, f"model_{target}.joblib")
        all_results[target] = results

    with open("evaluation_report.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved trained pipelines (*.joblib) and evaluation_report.json")


if __name__ == "__main__":
    main()
