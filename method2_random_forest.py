"""
Method 2: random forest for predicting whether a train arrives
>= 5 min late at its final station.

Data, features, split and metrics are shared with method 1 (see common.py).
Test predictions are saved to results/ for compare_methods.py.

Usage:
    python method2_random_forest.py
"""

import matplotlib.pyplot as plt

import common
import plots
from common import CATEGORICAL_FEATURES, FEATURES, MIN_STATION_COUNT, NUMERIC_FEATURES, RANDOM_SEED

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --- 1. Load data and split 70 / 15 / 15 (shared with method 1) -------------

df = common.load_data()
X_train, X_val, X_test, y_train, y_val, y_test = common.split_data(df)


# --- 2. Model: one-hot encoding + random forest ------------------------------
# Trees split on one feature at a time, so numeric features need no scaling.

N_TREES = 200

def build_model(max_depth, min_samples_leaf):
    preprocess = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=MIN_STATION_COUNT), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(n_estimators=N_TREES, max_depth=max_depth,
                                       min_samples_leaf=min_samples_leaf,
                                       n_jobs=-1, random_state=RANDOM_SEED)),
    ])


# --- 3. Choose tree depth and leaf size with the validation set -------------
# Deeper trees and smaller leaves fit the training data more closely;
# the validation loss shows where this starts to overfit.

depths = [4, 8, 12, 16, 20, 25]
leaf_sizes = [1, 10, 30]
results = []  # (depth, leaf size, train loss, validation loss, model)

for leaf in leaf_sizes:
    for depth in depths:
        model = build_model(depth, leaf).fit(X_train, y_train)
        train_loss = log_loss(y_train, model.predict_proba(X_train)[:, 1])
        val_loss = log_loss(y_val, model.predict_proba(X_val)[:, 1])
        results.append((depth, leaf, train_loss, val_loss, model))
        print(f"max_depth={depth:<3} min_samples_leaf={leaf:<3} "
              f"train loss {train_loss:.4f}  validation loss {val_loss:.4f}")

best_depth, best_leaf, _, _, best_model = min(results, key=lambda r: r[3])
print(f"best max_depth = {best_depth}, min_samples_leaf = {best_leaf}")


# --- 4. Choose the decision threshold with the validation set --------------

best_threshold, threshold_scores = common.choose_threshold(
    y_val, best_model.predict_proba(X_val)[:, 1])


# --- 5. Evaluate the chosen model -------------------------------------------

common.evaluate(best_model, "Training", X_train, y_train, best_threshold)
common.evaluate(best_model, "Validation", X_val, y_val, best_threshold)
common.evaluate(best_model, "Test", X_test, y_test, best_threshold)

# For comparison: the default threshold 0.5 on the test set
common.evaluate(best_model, "Test", X_test, y_test, 0.5)

test_proba = best_model.predict_proba(X_test)[:, 1]
common.save_test_predictions("method2", X_test, y_test, test_proba, best_threshold)


# --- 6. Feature importances -------------------------------------------------
# How much each feature reduces impurity across the forest (always >= 0,
# so unlike logistic regression coefficients they show no direction).

names = [n.split("__")[1] for n in best_model.named_steps["preprocess"].get_feature_names_out()]
importances = best_model.named_steps["clf"].feature_importances_
print("\n20 most important features:")
for n, v in sorted(zip(names, importances), key=lambda t: -abs(t[1]))[:20]:
    print(f"  {n:<40}{v:.3f}")


# --- 7. Figures (drawing code is in plots.py) -------------------------------

best_leaf_results = [r for r in results if r[1] == best_leaf]
plots.loss_vs_parameter([r[0] for r in best_leaf_results],
                        [r[2] for r in best_leaf_results], [r[3] for r in best_leaf_results],
                        "max_depth (maximum tree depth)",
                        f"Random forest: loss vs depth (min_samples_leaf={best_leaf})",
                        "method2_loss_vs_depth.png", log_x=False)
plots.probability_curves(
    best_model, df.loc[X_train.index], df.loc[X_val.index], FEATURES,
    f"Random forest (max_depth={best_depth}, min_samples_leaf={best_leaf}): "
    "predicted probability of delay",
    "method2_probability_curves.png")
plots.threshold_scores(threshold_scores, best_threshold,
                       "Random forest: choosing the threshold (validation set)",
                       "method2_threshold.png")
plots.confusion_matrix_plot(y_test, test_proba >= best_threshold,
                            f"Confusion matrix (test set, threshold {best_threshold:.2f})",
                            "method2_confusion_matrix.png")
plots.feature_weights(names, importances, "Importance (mean decrease in impurity)",
                      "Random forest: 20 most important features",
                      "method2_feature_importance.png", top_n=20)

print(f"\nFigures saved to {plots.FIG_DIR}")
plt.show()
