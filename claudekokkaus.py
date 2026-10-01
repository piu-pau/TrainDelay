"""
Stage 2: Junien myöhästymisen ennustaminen – logistinen regressio vs. satunnaismetsä.

Aja repon juuressa:  python stage2_models.py
Lukee esikäsitellyn datan kansiosta junadata/filtered-data4/ ja tulostaa
tulostaulukot sekä tallentaa kuvat kansioon figures/.
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss, confusion_matrix, roc_curve)

SEED = 42
DATA_GLOB = "junadata/filtered-data4/*/*.json"
os.makedirs("figures", exist_ok=True)

# ---------- 1. Datan lataus ja piirteet ----------
rows = []
for f in sorted(glob.glob(DATA_GLOB)):
    with open(f, encoding="utf-8") as fh:
        for t in json.load(fh):
            o, d = t["origin"], t["destination"]
            rows.append(dict(departureDate=t["departureDate"], trainType=t["trainType"],
                             numStops=t["numStops"], sched_dep=o["scheduledTime"],
                             sched_arr=d["scheduledTime"], dep_delay=o["differenceInMinutes"],
                             arr_delay=d["differenceInMinutes"]))
df = pd.DataFrame(rows)
n_raw = len(df)
df = df.dropna(subset=["dep_delay", "arr_delay"])      # puuttuvat toteumat
df = df[df.trainType != "P"]                           # vain 1 havainto
print(f"Datapisteitä: {n_raw} -> {len(df)} siivouksen jälkeen")

dep_utc = pd.to_datetime(df.sched_dep, utc=True)
dep_local = dep_utc.dt.tz_convert("Europe/Helsinki")   # aikataulut ovat UTC-ajassa
df["duration"] = (pd.to_datetime(df.sched_arr, utc=True) - dep_utc).dt.total_seconds() / 60
df["hour"] = dep_local.dt.hour
df["weekday"] = dep_local.dt.dayofweek
df["y"] = (df.arr_delay >= 5).astype(int)             # nimiö: >= 5 min myöhässä
print(f"Myöhässä olevien osuus: {df.y.mean():.3f}")

NUM = ["dep_delay", "duration", "numStops"]
CAT = ["trainType", "weekday", "hour"]
X, y, groups = df[NUM + CAT], df.y.values, df.departureDate.values

# ---------- 2. Jako 70/15/15, ryhmät = lähtöpäivä ----------
tr, tmp = next(GroupShuffleSplit(1, test_size=0.30, random_state=SEED).split(X, y, groups))
v_i, t_i = next(GroupShuffleSplit(1, test_size=0.50, random_state=SEED).split(X.iloc[tmp], y[tmp], groups[tmp]))
va, te = tmp[v_i], tmp[t_i]
for name, idx in [("Koulutus", tr), ("Validointi", va), ("Testi", te)]:
    print(f"{name:10s}: {len(idx):6d} junaa, {len(set(groups[idx])):3d} päivää, myöhässä {y[idx].mean():.3f}")

def preprocessor():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])

def evaluate(model, idx):
    p = model.predict_proba(X.iloc[idx])[:, 1]
    yh = (p >= 0.5).astype(int)
    # Painotettu logistinen häviö: sama tasapainotus kuin koulutuksessa
    # (class_weight="balanced"), jolloin arvaus p=0.5 antaa häviön ln 2 ≈ 0.693.
    w = compute_sample_weight("balanced", y[idx])
    return dict(logloss=log_loss(y[idx], p, sample_weight=w), auc=roc_auc_score(y[idx], p),
                acc=accuracy_score(y[idx], yh), prec=precision_score(y[idx], yh, zero_division=0),
                rec=recall_score(y[idx], yh), f1=f1_score(y[idx], yh))

# ---------- 3. Hyperparametrien valinta validointijoukolla ----------
candidates = {}
for C in [0.01, 0.1, 1, 10]:
    candidates[f"LogReg C={C}"] = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
for depth in [4, 8, 12, None]:
    for leaf in [1, 20, 50]:
        candidates[f"RF depth={depth} leaf={leaf}"] = RandomForestClassifier(
            n_estimators=300, max_depth=depth, min_samples_leaf=leaf,
            class_weight="balanced", n_jobs=-1, random_state=SEED)

results, fitted = [], {}
for name, clf in candidates.items():
    m = Pipeline([("pre", preprocessor()), ("clf", clf)]).fit(X.iloc[tr], y[tr])
    fitted[name] = m
    trm, vam = evaluate(m, tr), evaluate(m, va)
    results.append(dict(model=name, train_logloss=trm["logloss"], val_logloss=vam["logloss"],
                        train_auc=trm["auc"], val_auc=vam["auc"], val_f1=vam["f1"]))
res = pd.DataFrame(results)
pd.set_option("display.width", 200)
print("\n=== Validointitulokset ===")
print(res.round(3).to_string(index=False))

best_lr = res[res.model.str.startswith("LogReg")].sort_values("val_auc").iloc[-1].model
best_rf = res[res.model.str.startswith("RF")].sort_values("val_auc").iloc[-1].model
print(f"\nParas LR: {best_lr}\nParas RF: {best_rf}")

# ---------- 4. Parhaiden mallien vertailu ----------
summary = []
for name in [best_lr, best_rf]:
    for split, idx in [("koulutus", tr), ("validointi", va)]:
        summary.append(dict(malli=name, joukko=split, **evaluate(fitted[name], idx)))
# yksinkertainen sääntö vertailukohdaksi: myöhässä jos lähti >= 5 min myöhässä
for split, idx in [("koulutus", tr), ("validointi", va)]:
    yh = (X.iloc[idx].dep_delay >= 5).astype(int)
    summary.append(dict(malli="Sääntö: lähtö >= 5 min", joukko=split, logloss=np.nan, auc=np.nan,
                        acc=accuracy_score(y[idx], yh), prec=precision_score(y[idx], yh),
                        rec=recall_score(y[idx], yh), f1=f1_score(y[idx], yh)))
print("\n=== Vertailu (kynnys 0.5) ===")
print(pd.DataFrame(summary).round(3).to_string(index=False))

# ---------- 5. Lopullinen malli testijoukolla ----------
final_name = best_rf if res.set_index("model").val_auc[best_rf] > res.set_index("model").val_auc[best_lr] else best_lr
final = fitted[final_name]
tm = evaluate(final, te)
print(f"\n=== Valittu malli: {final_name} – TESTI ===")
print({k: round(v, 3) for k, v in tm.items()})
print("Sekaannusmatriisi [[TN FP] [FN TP]]:")
print(confusion_matrix(y[te], (final.predict_proba(X.iloc[te])[:, 1] >= 0.5).astype(int)))
yh = (X.iloc[te].dep_delay >= 5).astype(int)
print(f"Sääntö testillä: acc={accuracy_score(y[te], yh):.3f} f1={f1_score(y[te], yh):.3f}")

# ---------- 6. Kuvat ----------
plt.figure(figsize=(5, 5))
for name in [best_lr, best_rf]:
    p = fitted[name].predict_proba(X.iloc[va])[:, 1]
    fpr, tpr, _ = roc_curve(y[va], p)
    plt.plot(fpr, tpr, label=f"{name.split()[0]} (AUC {roc_auc_score(y[va], p):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=0.8)
plt.xlabel("Väärien positiivisten osuus"); plt.ylabel("Oikeiden positiivisten osuus")
plt.title("ROC-käyrät validointijoukolla"); plt.legend(); plt.tight_layout()
plt.savefig("figures/roc_validation.png", dpi=150); plt.close()

feat_names = final.named_steps["pre"].get_feature_names_out()
imp = pd.Series(final.named_steps["clf"].feature_importances_, index=feat_names) \
      if hasattr(final.named_steps["clf"], "feature_importances_") else \
      pd.Series(np.abs(final.named_steps["clf"].coef_[0]), index=feat_names)
imp.index = imp.index.str.replace("num__", "").str.replace("cat__", "")
fig, ax = plt.subplots(figsize=(6, 4.5))
imp.sort_values().tail(12).plot.barh(ax=ax)
plt.title("Tärkeimmät piirteet (satunnaismetsä)"); plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150); plt.close()
print("\nTärkeimmät piirteet:\n", imp.sort_values(ascending=False).head(8).round(3))
