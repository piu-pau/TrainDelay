# Figures for the train delay models. Used by the method scripts, e.g.

    # import plots
    # plots.label_distribution(y, "method1_label_distribution.png")

# All figures are saved to the figures/ folder.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score, roc_curve

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok = True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

sns.set_theme(style = "whitegrid", context = "notebook", font_scale = 0.9)


def _save(fig, filename):
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename)


def label_distribution(y, filename):
    """Bar chart of how many trains are on time (0) vs delayed (1)."""
    label_df = pd.DataFrame({"label": y.map({0: "0 = on time", 1: "1 = delayed"})})
    fig, ax = plt.subplots(figsize = (6, 4))
    sns.countplot(data = label_df, x = "label", hue = "label", order = ["0 = on time", "1 = delayed"],
                  palette = [BLUE, ORANGE], legend = False, width = 0.6, saturation = 1, ax = ax)
    for container in ax.containers:
        ax.bar_label(container, fmt = lambda c: f"{c:.0f}  ({c / len(y):.1%})")
    ax.set_ylim(0, y.value_counts().max() * 1.12)
    ax.set_xlabel("")
    ax.set_ylabel("Number of trains")
    ax.set_title("Distribution of the label (delayed = arrival delay ≥ 5 min)")
    _save(fig, filename)


def probability_curves(model, df_train, df_val, feature_columns, title, filename,
                       seed = 42):
    """
    Model's P(delayed) as a function of departure delay, travel time and hour,
    with the other features fixed at typical values (median / most common;
    for the stations the most common route).

    df_train, df_val: rows of the full data table (with "hour" and "delayed")
    feature_columns:  the columns the model was trained on
    """
    typical = {
        "departureDelay": df_train["departureDelay"].median(),
        "travelTime": df_train["travelTime"].median(),
        "numStops": df_train["numStops"].median(),
        "hour": df_train["hour"].median(),
        "trainType": df_train["trainType"].mode()[0],
        "weekday": df_train["weekday"].mode()[0],
    }
    route = df_train.groupby(["originStation", "destinationStation"]).size().idxmax()
    typical["originStation"], typical["destinationStation"] = route

    def curve(feature, grid):
        rows = pd.DataFrame({k: [v] * len(grid) for k, v in typical.items()})
        rows[feature] = grid
        rows["hourSin"] = np.sin(2 * np.pi * rows["hour"] / 24)
        rows["hourCos"] = np.cos(2 * np.pi * rows["hour"] / 24)
        return model.predict_proba(rows[feature_columns])[:, 1]

    def fixed_text(feature):
        parts = []
        for k, v in typical.items():
            if k == feature:
                continue
            if k == "weekday":
                v = WEEKDAYS[int(v)]
            elif isinstance(v, float):
                v = f"{v:g}"
            parts.append(f"{k}={v}")
        return "\n".join(", ".join(parts[i:i + 3]) for i in range(0, len(parts), 3))

    panels = [
        ("departureDelay", np.linspace(-2, 30, 200), "Departure delay (min)"),
        ("travelTime", np.linspace(10, 1000, 200), "Travel time (min)"),
        ("hour", np.linspace(0, 24, 200), "Scheduled departure hour (Finnish time)"),
    ]

    # Sample of training + validation points, scattered around 0 and 1 so they are visible
    trval_all = pd.concat([df_train, df_val])
    rng = np.random.default_rng(seed)
    trval = trval_all.sample(1500, random_state = seed).copy()
    trval["delayedJitter"] = trval["delayed"] + rng.uniform(-0.04, 0.04, len(trval))

    trval_loss = log_loss(trval_all["delayed"], model.predict_proba(trval_all[feature_columns])[:, 1])

    fig, axes = plt.subplots(1, 3, figsize = (15, 4.5))
    for ax, (feature, grid, xlabel) in zip(axes, panels):
        in_range = trval[trval[feature].between(grid[0], grid[-1])]
        sns.scatterplot(data = in_range, x = feature, y = "delayedJitter", s = 12, alpha = 0.3,
                        color = BLUE, edgecolor = None, label = "Train/Val (actual label)", ax = ax)
        sns.lineplot(x = grid, y = curve(feature, grid), color = ORANGE, linewidth = 2,
                     label = "Model P(delayed)", ax = ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("P(delayed)")
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"Fixed: {fixed_text(feature)}\nTR/VAL log loss={trval_loss:.3f}", fontsize = 9)
        ax.legend(loc = "center right", fontsize = 8)
    fig.suptitle(title)
    _save(fig, filename)


def loss_vs_parameter(values, train_losses, val_losses, xlabel, title, filename, log_x = True):
    """Training and validation loss for each value of a hyperparameter."""
    loss_df = pd.DataFrame({
        "value": list(values) * 2,
        "log loss": list(train_losses) + list(val_losses),
        "set": ["Training"] * len(values) + ["Validation"] * len(values),
    })
    fig, ax = plt.subplots(figsize = (6, 4))
    sns.lineplot(data = loss_df, x = "value", y = "log loss", hue = "set", palette = [BLUE, ORANGE],
                 marker = "o", linewidth = 2, ax = ax)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Log loss")
    ax.set_title(title)
    ax.legend(title = None)
    _save(fig, filename)


def threshold_scores(scores, best_threshold, title, filename):
    """Precision, recall and F1 for each decision threshold (columns of `scores`)."""
    long_df = scores.melt(id_vars = "threshold", var_name = "metric", value_name = "score")
    fig, ax = plt.subplots(figsize = (6, 4))
    sns.lineplot(data = long_df, x = "threshold", y = "score", hue = "metric",
                 palette = {"precision": BLUE, "recall": ORANGE, "F1": "#1baf7a"},
                 linewidth = 2, ax = ax)
    ax.axvline(best_threshold, color = "#898781", linestyle = "--", linewidth = 1)
    ax.text(best_threshold + 0.01, 0.95, f"chosen {best_threshold:.2f}", color = "#52514e",
            va = "top", fontsize = 9)
    ax.set_xlabel("Decision threshold for P(delayed)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(title = None)
    _save(fig, filename)


def confusion_matrix_plot(y_true, y_pred, title, filename):
    """2x2 heatmap of actual vs predicted labels."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize = (5, 4.2))
    sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues", cbar = False, square = True,
                xticklabels = ["on time", "delayed"], yticklabels = ["on time", "delayed"],
                annot_kws = {"size": 12}, ax = ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    _save(fig, filename)


def feature_weights(names, values, xlabel, title, filename, top_n = None):
    """
    Horizontal bars per feature (e.g. coefficients or feature importances).
    If top_n is given, only the top_n largest in absolute value are shown.
    """
    w_df = pd.DataFrame({"feature": names, "value": values})
    if top_n is not None:
        w_df = w_df.loc[w_df["value"].abs().nlargest(top_n).index]
    w_df = w_df.sort_values("value", ascending = False)
    if (w_df["value"] >= 0).all():
        colors = [BLUE] * len(w_df)  # importances: no direction, one colour
    else:
        colors = [ORANGE if v > 0 else BLUE for v in w_df["value"]]
    fig, ax = plt.subplots(figsize = (6, 1.5 + 0.25 * len(w_df)))
    sns.barplot(data = w_df, x = "value", y = "feature", hue = "feature", palette = colors,
                legend = False, saturation = 1, ax = ax)
    ax.axvline(0, color = "#898781", linewidth = 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_title(title)
    _save(fig, filename)


def roc_curves(y_true, probas, title, filename):
    """ROC curves of several models. probas: dict {model name: P(delayed)}."""
    fig, ax = plt.subplots(figsize = (5.5, 5))
    for (name, proba), color in zip(probas.items(), [BLUE, ORANGE]):
        fpr, tpr, _ = roc_curve(y_true, proba)
        sns.lineplot(x = fpr, y = tpr, color = color, linewidth = 2, errorbar = None,
                     label = f"{name} (AUC {roc_auc_score(y_true, proba):.3f})", ax = ax)
    ax.plot([0, 1], [0, 1], color = "#898781", linestyle = "--", linewidth = 1,
            label = "Random guessing (AUC 0.5)")
    ax.set_xlabel("False positive rate (on-time trains flagged as delayed)")
    ax.set_ylabel("True positive rate (delays found)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc = "lower right", fontsize = 9)
    _save(fig, filename)
