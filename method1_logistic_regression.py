"""
Method 1: logistic regression for predicting whether a train arrives
>= 5 min late at its final station.

Data, features, split and metrics are shared with method 2 (see common.py).
Test predictions are saved to results/ for compare_methods.py.

Usage:
    python method1_logistic_regression.py
"""

import matplotlib.pyplot as plt

import common
import plots
from common import CATEGORICAL_FEATURES, FEATURES, MIN_STATION_COUNT, NUMERIC_FEATURES

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Load data and split 70 / 15 / 15 (same as in method 2)

df = common.load_data()
X_train, X_val, X_test, y_train, y_val, y_test = common.split_data(df)


# Model: scaling + converting categorial values into binary + logistic regression
# Numeric features are standardised so that their coefficients are comparable

def build_model(C):
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown = "infrequent_if_exist",
                              min_frequency = MIN_STATION_COUNT), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(C = C, max_iter = 2000)),
    ])


# Choose regularization strength C with the validation set

C_values = [0.001, 0.01, 0.1, 1, 10, 100]
results = []  # (C, train loss, validation loss, model)

for C in C_values:
    model = build_model(C).fit(X_train, y_train)
    train_loss = log_loss(y_train, model.predict_proba(X_train)[:, 1])
    val_loss = log_loss(y_val, model.predict_proba(X_val)[:, 1])
    results.append((C, train_loss, val_loss, model))
    print(f"C={C:<6} train loss {train_loss:.4f}  validation loss {val_loss:.4f}")

best_C, _, _, best_model = min(results, key = lambda r: r[2])
print(f"best C = {best_C}")


# Choose the decision threshold with the validation set

best_threshold, threshold_scores = common.choose_threshold(
    y_val, best_model.predict_proba(X_val)[:, 1])


# Evaluate the chosen model

common.evaluate(best_model, "Training", X_train, y_train, best_threshold)
common.evaluate(best_model, "Validation", X_val, y_val, best_threshold)
common.evaluate(best_model, "Test", X_test, y_test, best_threshold)

# For comparison: the default threshold 0.5 on the test set
common.evaluate(best_model, "Test", X_test, y_test, 0.5)

test_proba = best_model.predict_proba(X_test)[:, 1]
common.save_test_predictions("method1", X_test, y_test, test_proba, best_threshold)


# Features are standardized so the sizes are comparable
names = [n.split("__")[1] for n in best_model.named_steps["preprocess"].get_feature_names_out()]
coefs = best_model.named_steps["clf"].coef_[0]
print("\n20 largest coefficients:")
for n, v in sorted(zip(names, coefs), key = lambda t: -abs(t[1]))[:20]:
    print(f"  {n:<40}{v:+.3f}")


# Figures (drawing code is in plots.py)

plots.label_distribution(df["delayed"], "label_distribution.png")

plots.loss_vs_parameter([r[0] for r in results],
                        [r[1] for r in results], [r[2] for r in results],
                        "C (inverse regularisation strength)",
                        "Logistic regression: loss vs C",
                        "method1_loss_vs_C.png", log_x = True)
plots.probability_curves(
    best_model, df.loc[X_train.index], df.loc[X_val.index], FEATURES,
    f"Logistic regression (C={best_C}): predicted probability of delay",
    "method1_probability_curves.png")
plots.threshold_scores(threshold_scores, best_threshold,
                       "Logistic regression: choosing the threshold (validation set)",
                       "method1_threshold.png")
plots.confusion_matrix_plot(y_test, test_proba >= best_threshold,
                            f"Confusion matrix (test set, threshold {best_threshold:.2f})",
                            "method1_confusion_matrix.png")
plots.feature_weights(names, coefs, "Coefficient  (> 0 → more likely delayed)",
                      "Logistic regression: 20 largest coefficients",
                      "method1_coefficients.png", top_n = 20)

print(f"\nFigures saved to {plots.FIG_DIR}")
plt.show()
