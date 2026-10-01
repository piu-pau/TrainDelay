import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import plots

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV_PATH = Path(__file__).parent / "junadata" / "trains.csv"
RANDOM_SEED = 42

NUMERIC_FEATURES = ["departureDelay", "travelTime", "numStops", "hourSin", "hourCos"]
CATEGORICAL_FEATURES = ["trainType", "weekday", "originStation", "destinationStation"]

# Stations with fewer trains than this in the training set are grouped as "infrequent"
MIN_STATION_COUNT = 50


# --- 1. Load data and build features ---------------------------------------

df = pd.read_csv(CSV_PATH)

# Hour as sin/cos so that 23:00 and 00:00 are close to each other
df["hourSin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hourCos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["weekday"] = df["weekday"].astype(str)  # treat as a category, not a number

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df["delayed"]
print(f"{len(df)} trains, {y.mean():.1%} delayed")


# --- 2. Split 70 / 15 / 15 --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED)
print(f"train {len(X_train)}, validation {len(X_val)}, test {len(X_test)}")


# --- 3. Model: scaling + one-hot encoding + logistic regression -------------

def build_model(C):
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=MIN_STATION_COUNT), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])


# --- 4. Choose regularisation strength C with the validation set -----------

C_values = [0.001, 0.01, 0.1, 1, 10, 100]
train_losses, val_losses, models = [], [], []

for C in C_values:
    model = build_model(C).fit(X_train, y_train)
    train_losses.append(log_loss(y_train, model.predict_proba(X_train)[:, 1]))
    val_losses.append(log_loss(y_val, model.predict_proba(X_val)[:, 1]))
    models.append(model)
    print(f"C={C:<6} train loss {train_losses[-1]:.4f}  validation loss {val_losses[-1]:.4f}")

best = int(np.argmin(val_losses))
best_C, best_model = C_values[best], models[best]
print(f"best C = {best_C}")


# --- 5. Choose the decision threshold with the validation set --------------
# Only 13 % of trains are delayed, so the model rarely gives P(delayed) > 0.5
# and the default threshold 0.5 finds almost no delays. We keep the model's
# probabilities as they are and instead pick the threshold that maximises F1
# (balance of precision and recall) on the validation set.

thresholds = np.round(np.arange(0.05, 0.91, 0.01), 2)
val_proba = best_model.predict_proba(X_val)[:, 1]
threshold_scores = pd.DataFrame({
    "threshold": thresholds,
    "precision": [precision_score(y_val, val_proba >= t, zero_division=0) for t in thresholds],
    "recall": [recall_score(y_val, val_proba >= t) for t in thresholds],
    "F1": [f1_score(y_val, val_proba >= t) for t in thresholds],
})
best_threshold = threshold_scores.loc[threshold_scores["F1"].idxmax(), "threshold"]
print(f"best threshold = {best_threshold:.2f}")


# --- 6. Evaluate the chosen model -------------------------------------------

def evaluate(name, X_part, y_part, threshold):
    proba = best_model.predict_proba(X_part)[:, 1]
    pred = (proba >= threshold).astype(int)
    print(f"\n=== {name} (threshold {threshold:.2f}) ===")
    print(f"log loss: {log_loss(y_part, proba):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_part, proba):.4f}")
    print(f"accuracy: {accuracy_score(y_part, pred):.4f}")
    print(f"F1:       {f1_score(y_part, pred):.4f}")
    print("confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_part, pred))
    print(classification_report(y_part, pred, digits=3))

evaluate("Training", X_train, y_train, best_threshold)
evaluate("Validation", X_val, y_val, best_threshold)
evaluate("Test", X_test, y_test, best_threshold)

# For comparison: the default threshold 0.5 on the test set
evaluate("Test", X_test, y_test, 0.5)

# Baseline for comparison: always predict "not delayed"
print(f"baseline accuracy (always on time), test: {1 - y_test.mean():.4f}")


# --- 7. Coefficients (features are standardised, so sizes are comparable) --

names = best_model.named_steps["preprocess"].get_feature_names_out()
coefs = best_model.named_steps["clf"].coef_[0]
print("\nCoefficients:")
for n, c in sorted(zip(names, coefs), key=lambda t: -abs(t[1])):
    print(f"  {n:<40}{c:+.3f}")


# --- 8. Figures (drawing code is in plots.py) -------------------------------

plots.label_distribution(y, "method1_label_distribution.png")
plots.probability_curves(
    best_model, df.loc[X_train.index], df.loc[X_val.index],
    NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    f"Logistic regression (C={best_C}): predicted probability of delay",
    "method1_probability_curves.png")
plots.loss_vs_parameter(C_values, train_losses, val_losses,
                        "C (inverse regularisation strength)",
                        "Logistic regression: loss vs C", "method1_loss_vs_C.png")
plots.threshold_scores(threshold_scores, best_threshold,
                       "Logistic regression: choosing the threshold (validation set)",
                       "method1_threshold.png")
plots.confusion_matrix_plot(y_test, best_model.predict_proba(X_test)[:, 1] >= best_threshold,
                            f"Confusion matrix (test set, threshold {best_threshold:.2f})",
                            "method1_confusion_matrix.png")
plots.feature_weights([n.split("__")[1] for n in names], coefs,
                      "Coefficient  (> 0 → more likely delayed)",
                      "Logistic regression: 20 largest coefficients",
                      "method1_coefficients.png", top_n=20)

print(f"\nFigures saved to {plots.FIG_DIR}")
plt.show()
