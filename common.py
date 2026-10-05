"""
Shared code for both methods, so that they use exactly the same data,
features, train/validation/test split, threshold choice and metrics.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

CSV_PATH = Path(__file__).parent / "junadata" / "trains.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RANDOM_SEED = 42

NUMERIC_FEATURES = ["departureDelay", "travelTime", "numStops", "hourSin", "hourCos"]
CATEGORICAL_FEATURES = ["trainType", "weekday", "originStation", "destinationStation"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Stations with fewer trains than this in the training set are grouped as "infrequent"
MIN_STATION_COUNT = 50


def load_data():
    """Read trains.csv and add the derived features. Returns the full table."""
    df = pd.read_csv(CSV_PATH)

    # Hour as sin/cos so that 23:00 and 00:00 are close to each other
    df["hourSin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hourCos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday"] = df["weekday"].astype(str)  # treat as a category, not a number

    print(f"{len(df)} trains, {df['delayed'].mean():.1%} delayed")
    return df


def split_data(df):
    """Random 70 / 15 / 15 split with a fixed seed."""
    X, y = df[FEATURES], df["delayed"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED)
    print(f"train {len(X_train)}, validation {len(X_val)}, test {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def choose_threshold(y_val, val_proba):
    """
    Only 13 % of trains are delayed, so a model rarely gives P(delayed) > 0.5
    and the default threshold 0.5 finds almost no delays. We keep the model's
    probabilities as they are and pick the threshold that maximises F1
    (balance of precision and recall) on the validation set.
    """
    thresholds = np.round(np.arange(0.05, 0.91, 0.01), 2)
    scores = pd.DataFrame({
        "threshold": thresholds,
        "precision": [precision_score(y_val, val_proba >= t, zero_division=0) for t in thresholds],
        "recall": [recall_score(y_val, val_proba >= t) for t in thresholds],
        "F1": [f1_score(y_val, val_proba >= t) for t in thresholds],
    })
    best_threshold = scores.loc[scores["F1"].idxmax(), "threshold"]
    print(f"best threshold = {best_threshold:.2f}")
    return best_threshold, scores


def evaluate(model, name, X_part, y_part, threshold):
    """Print the metrics for one data set and return them as a dict."""
    proba = model.predict_proba(X_part)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "log loss": log_loss(y_part, proba),
        "ROC AUC": roc_auc_score(y_part, proba),
        "accuracy": accuracy_score(y_part, pred),
        "precision": precision_score(y_part, pred, zero_division=0),
        "recall": recall_score(y_part, pred),
        "F1": f1_score(y_part, pred),
    }
    print(f"\n=== {name} (threshold {threshold:.2f}) ===")
    for k, v in metrics.items():
        print(f"{k + ':':<11}{v:.4f}")
    print("confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_part, pred))
    print(classification_report(y_part, pred, digits=3))
    return metrics


def save_test_predictions(method_name, X_test, y_test, test_proba, threshold):
    """Save test-set predictions so that compare_methods.py can compare the methods."""
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({
        "departureDelay": X_test["departureDelay"].to_numpy(),
        "delayed": y_test.to_numpy(),
        "proba": test_proba,
        "threshold": threshold,
    }).to_csv(RESULTS_DIR / f"{method_name}_test_predictions.csv", index=False)
