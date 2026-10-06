"""
Compares method 1 (logistic regression) and method 2 (random forest) on the
test set. Run both method scripts first; they save their test predictions
to results/.

Usage:
    python compare_methods.py
"""

import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)

import plots
from common import RESULTS_DIR

METHODS = {
    "Logistic regression": "method1",
    "Random forest": "method2",
}


def metrics(y, pred, proba = None):
    return {
        "log loss": log_loss(y, proba) if proba is not None else None,
        "ROC AUC": roc_auc_score(y, proba) if proba is not None else None,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division = 0),
        "recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
    }


predictions = {name: pd.read_csv(RESULTS_DIR / f"{file}_test_predictions.csv")
               for name, file in METHODS.items()}

# Both methods must have been evaluated on the same test trains
first = next(iter(predictions.values()))
for p in predictions.values():
    assert p["delayed"].equals(first["delayed"]), "methods used different test sets"
y_test = first["delayed"]

rows = {}
for name, p in predictions.items():
    threshold = p["threshold"].iloc[0]
    rows[f"{name} (threshold {threshold:.2f})"] = metrics(y_test, p["proba"] >= threshold, p["proba"])

# Simple baselines for comparison
rows["Baseline: always on time"] = metrics(y_test, pd.Series(0, index = y_test.index))
rows["Baseline: departure delay >= 5 min"] = metrics(y_test, first["departureDelay"] >= 5)

table = pd.DataFrame(rows).T
pd.set_option("display.width", 200)
print(f"Test set: {len(y_test)} trains, {y_test.mean():.1%} delayed\n")
print(table.round(3).to_string(na_rep = "–"))

table.round(4).to_csv(RESULTS_DIR / "comparison_test.csv")
plots.roc_curves(y_test, {name: p["proba"] for name, p in predictions.items()},
                 "ROC curves on the test set", "comparison_roc.png")
print(f"\nTable saved to {RESULTS_DIR / 'comparison_test.csv'}")
print(f"Figure saved to {plots.FIG_DIR / 'comparison_roc.png'}")
