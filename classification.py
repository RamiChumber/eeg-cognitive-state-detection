"""
Classification — EEG Cognitive State Detection

Two evaluation schemes on the frequency-domain (band power) feature set:

  1. Within-session — Stratified Shuffle Split (5 folds)
     Randomly splits windows across all pilots; optimistic upper bound due to
     autocorrelation between adjacent windows assigned to train and test.

  2. Pilot-out — Leave-One-Pilot-Out Cross-Validation
     Trains on all pilots except one, tests on the held-out pilot.
     Realistic estimate of operational cross-subject performance.

Inputs:  data/transformed_train.csv       (band power features, 18-pilot)
         data/anova_band_features.csv     (ANOVA-ranked feature importance)
Outputs: results/within_session_results.csv
         results/pilot_out_results.csv
         figures/fold_analysis.png
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, f1_score,
                              accuracy_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

PHYS_COLS = ["ecg", "r", "gsr"]
BANDS     = ["delta", "theta", "alpha", "beta", "gamma"]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_sets(band_df, anova_df, all_feature_cols):
    """Return the four ANOVA-filtered feature sets used across both CV schemes."""
    return {
        "liberal":    [f for f in anova_df.loc[anova_df["eta2"] > 0.005, "feature"] if f in all_feature_cols],
        "medium":     [f for f in anova_df.loc[anova_df["eta2"] > 0.01,  "feature"] if f in all_feature_cols],
        "strict":     [f for f in anova_df.loc[anova_df["eta2"] > 0.02,  "feature"] if f in all_feature_cols],
        "beta_gamma": [f for f in all_feature_cols
                       if (f.endswith("_beta") or f.endswith("_gamma"))
                       or any(f.startswith(p + "_") or f == p for p in PHYS_COLS)],
    }


def normalise_to_baseline(df, feature_cols,
                           baseline_event="A",
                           group_cols=["pilot", "experiment"],
                           event_col="event"):
    """
    Z-score each feature relative to that pilot-experiment's baseline (event A).
    Reduces between-pilot absolute differences without leaking cross-pilot info.
    """
    df           = df.copy().reset_index(drop=True)
    feature_cols = [f for f in feature_cols if f in df.columns]
    feat_idx     = [df.columns.get_loc(c) for c in feature_cols]

    for (pilot, exp), group in df.groupby(group_cols):
        row_idx  = group.index.tolist()
        baseline = group.loc[group[event_col] == baseline_event, feature_cols]
        if len(baseline) < 5:
            baseline = group[feature_cols]
        bl_mean = baseline.mean().values
        bl_std  = baseline.std().values
        bl_std  = np.where(~np.isfinite(bl_std) | (bl_std < 1e-8), 1.0, bl_std)
        normalised = (group[feature_cols].values - bl_mean) / bl_std
        normalised = np.where(np.isfinite(normalised), normalised, 0.0)
        df.iloc[row_idx, feat_idx] = normalised

    return df


def make_pipeline(estimator):
    return Pipeline([
        ("scaler",  StandardScaler()),
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("model",   estimator)
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 1. WITHIN-SESSION STRATIFIED SHUFFLE SPLIT
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("WITHIN-SESSION STRATIFIED SHUFFLE SPLIT (5 folds)")
print("=" * 60)
print("NOTE: Scores are optimistic — adjacent windows appear in both train/test.")
print("      Pilot-out CV (Section 2) is the more reliable evaluation.\n")

band_df  = pd.read_csv("data/transformed_train.csv")
anova_df = pd.read_csv("data/anova_band_features.csv")

eeg_band_cols = [
    c for c in band_df.columns
    if c.startswith("eeg_") and any(c.endswith(f"_{b}") for b in BANDS)
]
phys_feat_cols = [
    c for c in band_df.columns
    if any(c.startswith(p + "_") or c == p for p in PHYS_COLS)
]
all_feature_cols = eeg_band_cols + phys_feat_cols
zero_var = [f for f in all_feature_cols if band_df[f].std() == 0]
all_feature_cols = [f for f in all_feature_cols if f not in zero_var]
print(f"Features: {len(all_feature_cols)}")

feat = build_feature_sets(band_df, anova_df, all_feature_cols)

label_map     = {label: i for i, label in enumerate(sorted(band_df["event"].unique()))}
inv_label_map = {v: k for k, v in label_map.items()}
band_df["label"] = band_df["event"].map(label_map)
class_names   = [inv_label_map[i] for i in range(len(label_map))]
print(f"Classes: {label_map}\n")

print("Applying baseline normalisation...")
band_df_norm = normalise_to_baseline(band_df, all_feature_cols)
band_df_norm["label"] = band_df["label"].values

SSS_MODELS = [
    ("LightGBM",
     lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                        num_leaves=31, min_child_samples=20, colsample_bytree=0.8,
                        subsample=0.8, class_weight="balanced", random_state=42, verbose=-1),
     feat["liberal"]),
    ("XGBoost",
     xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                       colsample_bytree=0.8, subsample=0.8, eval_metric="mlogloss",
                       random_state=42, verbosity=0),
     feat["liberal"]),
    ("Random Forest",
     RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                            max_features="sqrt", class_weight="balanced",
                            random_state=42, n_jobs=-1),
     feat["liberal"]),
    ("SVM (Linear)",
     make_pipeline(LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)),
     feat["strict"]),
    ("KNN",
     make_pipeline(KNeighborsClassifier(n_neighbors=11, weights="distance",
                                        metric="euclidean", n_jobs=-1)),
     feat["strict"]),
    ("MLP",
     make_pipeline(MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=0.01,
                                 learning_rate="adaptive", max_iter=200,
                                 early_stopping=True, validation_fraction=0.1, random_state=42)),
     feat["strict"]),
]

N_SPLITS = 5
sss      = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.3, random_state=42)
X_all    = band_df_norm[all_feature_cols]
y_all    = band_df_norm["label"].values

print(f"{'Model':<22}  {'Mean Acc':>8}  {'Mean Prec':>9}  {'Mean Rec':>8}  {'Mean F1':>7}  {'Std F1':>6}")
print("-" * 70)

sss_summary = []
for name, model, feat_cols in SSS_MODELS:
    feat_cols = [f for f in feat_cols if f in band_df_norm.columns]
    X         = band_df_norm[feat_cols].values
    fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []

    for train_idx, test_idx in sss.split(X_all, y_all):
        fold_model = clone(model)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        if name == "XGBoost":
            sw = compute_sample_weight("balanced", y_train)
            fold_model.fit(X_train, y_train, sample_weight=sw)
        else:
            fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)
        fold_acc.append(accuracy_score(y_test, y_pred))
        fold_prec.append(precision_score(y_test, y_pred, average="macro", zero_division=0))
        fold_rec.append(recall_score(y_test, y_pred,    average="macro", zero_division=0))
        fold_f1.append(f1_score(y_test, y_pred,         average="macro", zero_division=0))

    row = {"Model": name,
           "Mean Acc.":  round(np.mean(fold_acc),  3),
           "Mean Prec.": round(np.mean(fold_prec), 3),
           "Mean Rec.":  round(np.mean(fold_rec),  3),
           "Mean F1":    round(np.mean(fold_f1),   3),
           "Std F1":     round(np.std(fold_f1),    3)}
    sss_summary.append(row)
    print(f"  {name:<22}  {row['Mean Acc.']:>8.3f}  {row['Mean Prec.']:>9.3f}  "
          f"{row['Mean Rec.']:>8.3f}  {row['Mean F1']:>7.3f}  {row['Std F1']:>6.3f}")

sss_df = pd.DataFrame(sss_summary).sort_values("Mean F1", ascending=False)
sss_df.to_csv("results/within_session_results.csv", index=False)
print(f"\nSaved: results/within_session_results.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 2. PILOT-OUT (LEAVE-ONE-PILOT-OUT) CV (18-pilot dataset)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PILOT-OUT CROSS-VALIDATION (LEAVE-ONE-PILOT-OUT)")
print("=" * 60)

LOPO_MODELS = [
    ("LightGBM",
     lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                        num_leaves=31, min_child_samples=20, colsample_bytree=0.8,
                        subsample=0.8, class_weight="balanced", random_state=42, verbose=-1),
     feat["liberal"]),
    ("LightGBM (β/γ)",
     lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                        num_leaves=31, min_child_samples=20, colsample_bytree=0.8,
                        subsample=0.8, class_weight="balanced", random_state=42, verbose=-1),
     feat["beta_gamma"]),
    ("XGBoost",
     xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                       colsample_bytree=0.8, subsample=0.8, eval_metric="mlogloss",
                       random_state=42, verbosity=0),
     feat["liberal"]),
    ("XGBoost (β/γ)",
     xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                       colsample_bytree=0.8, subsample=0.8, eval_metric="mlogloss",
                       random_state=42, verbosity=0),
     feat["beta_gamma"]),
    ("Random Forest",
     RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                            max_features="sqrt", class_weight="balanced",
                            random_state=42, n_jobs=-1),
     feat["liberal"]),
    ("SVM (Linear)",
     Pipeline([("scaler", StandardScaler()),
               ("svm", LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42))]),
     feat["strict"]),
    ("SVM (Linear β/γ)",
     Pipeline([("scaler", StandardScaler()),
               ("svm", LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42))]),
     feat["beta_gamma"]),
    ("KNN",
     Pipeline([("scaler", StandardScaler()),
               ("knn", KNeighborsClassifier(n_neighbors=11, weights="distance",
                                            metric="euclidean", n_jobs=-1))]),
     feat["strict"]),
    ("MLP",
     Pipeline([("scaler", StandardScaler()),
               ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=0.01,
                                     learning_rate="adaptive", max_iter=200,
                                     early_stopping=True, validation_fraction=0.1, random_state=42))]),
     feat["medium"]),
]
LOPO_MODELS = [(name, model, [f for f in fc if f in band_df.columns])
               for name, model, fc in LOPO_MODELS]

pilots  = sorted(band_df["pilot"].unique())
results = {name: [] for name, _, _ in LOPO_MODELS}

for held_out in pilots:
    fold_start = time.time()
    print(f"\nFold: held-out pilot {held_out}  ({pilots.index(held_out)+1}/{len(pilots)})")

    train_raw = band_df[band_df["pilot"] != held_out]
    test_raw  = band_df[band_df["pilot"] == held_out]

    train_df = normalise_to_baseline(train_raw, all_feature_cols)
    test_df  = normalise_to_baseline(test_raw,  all_feature_cols)
    train_df["label"] = train_raw["event"].map(label_map).values
    test_df["label"]  = test_raw["event"].map(label_map).values

    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    for name, model, feat_cols in LOPO_MODELS:
        fold_model = clone(model)
        X_train    = train_df[feat_cols].values
        X_test     = test_df[feat_cols].values
        if "XGBoost" in name:
            sw = compute_sample_weight("balanced", y_train)
            fold_model.fit(X_train, y_train, sample_weight=sw)
        else:
            fold_model.fit(X_train, y_train)
        y_pred   = fold_model.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        results[name].append({
            "pilot":    held_out,
            "macro_f1": macro_f1,
            "report":   classification_report(y_test, y_pred, target_names=class_names,
                                              output_dict=True, zero_division=0)
        })
        print(f"  {name:<22} macro F1 = {macro_f1:.3f}")
    print(f"  Fold time: {time.time() - fold_start:.0f}s")

# Summary
print(f"\n{'='*60}")
print(f"{'Model':<22}  {'Mean F1':>8}  {'Std':>6}  {'Min':>6}  {'Max':>6}")
print("-" * 55)

lopo_summary = []
for name, _, _ in LOPO_MODELS:
    f1s = [r["macro_f1"] for r in results[name]]
    row = {"model": name, "mean_f1": np.mean(f1s), "std_f1": np.std(f1s),
           "min_f1": np.min(f1s), "max_f1": np.max(f1s)}
    lopo_summary.append(row)
    print(f"  {name:<22}  {row['mean_f1']:>8.3f}  {row['std_f1']:>6.3f}  "
          f"{row['min_f1']:>6.3f}  {row['max_f1']:>6.3f}")

print(f"\nPer-class F1 (mean across folds):")
print(f"  {'Model':<22}", end="")
for cls in class_names:
    print(f"  {cls:>8}", end="")
print()
for name, _, _ in LOPO_MODELS:
    print(f"  {name:<22}", end="")
    for cls in class_names:
        f1s = [r["report"][cls]["f1-score"] for r in results[name]]
        print(f"  {np.mean(f1s):>8.3f}", end="")
    print()

lopo_df = pd.DataFrame(lopo_summary).sort_values("mean_f1", ascending=False)
lopo_df.to_csv("results/pilot_out_results.csv", index=False)
print(f"\nSaved: results/pilot_out_results.csv")

# Per-fold plot (top 5 models)
fig, ax = plt.subplots(figsize=(12, 5))
for name in lopo_df.head(5)["model"].tolist():
    f1s = [r["macro_f1"] for r in results[name]]
    ax.plot(range(len(pilots)), f1s, marker="o", label=name, linewidth=1.5)
ax.set_xticks(range(len(pilots)))
ax.set_xticklabels([f"Pilot {p}" for p in pilots], rotation=30, ha="right")
ax.set_ylabel("Macro F1")
ax.set_title("Pilot-out CV — macro F1 per fold (top 5 models)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fold_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: figures/fold_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. CALIBRATION CURVE — LightGBM (liberal features)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("CALIBRATION CURVE — LightGBM")
print("(% of test-pilot data added to training before evaluation)\n")

CALIBRATION_FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.30]
cal_results   = {frac: [] for frac in CALIBRATION_FRACTIONS}
feat_cols_cal = [f for f in feat["liberal"] if f in band_df.columns]

for held_out in pilots:
    train_raw  = band_df[band_df["pilot"] != held_out]
    test_pilot = band_df[band_df["pilot"] == held_out].copy().sort_values(["experiment", "time"])

    train_norm        = normalise_to_baseline(train_raw, feat_cols_cal)
    train_norm["label"] = train_raw["event"].map(label_map).values

    for frac in CALIBRATION_FRACTIONS:
        n_cal     = int(len(test_pilot) * frac)
        cal_part  = test_pilot.iloc[:n_cal]
        eval_part = test_pilot.iloc[n_cal:]

        if len(eval_part) == 0:
            continue
        if len(np.unique(eval_part["event"].map(label_map).values)) < 2:
            continue

        if n_cal > 0:
            eval_norm = normalise_to_baseline(
                pd.concat([cal_part, eval_part]), feat_cols_cal
            ).iloc[n_cal:]
            eval_norm["label"] = eval_part["event"].map(label_map).values
            cal_norm            = normalise_to_baseline(cal_part, feat_cols_cal)
            cal_norm["label"]   = cal_part["event"].map(label_map).values
            X_tr = np.vstack([train_norm[feat_cols_cal].values, cal_norm[feat_cols_cal].values])
            y_tr = np.concatenate([train_norm["label"].values, cal_norm["label"].values])
        else:
            eval_norm        = normalise_to_baseline(eval_part, feat_cols_cal)
            eval_norm["label"] = eval_part["event"].map(label_map).values
            X_tr = train_norm[feat_cols_cal].values
            y_tr = train_norm["label"].values

        X_te  = eval_norm[feat_cols_cal].values
        y_te  = eval_norm["label"].values
        sw    = compute_sample_weight("balanced", y_tr)
        model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                                    class_weight="balanced", random_state=42, verbose=-1)
        model.fit(X_tr, y_tr, sample_weight=sw)
        cal_results[frac].append(f1_score(y_te, model.predict(X_te),
                                          average="macro", zero_division=0))

print(f"  {'Calibration %':<16}  {'Mean F1':>8}  {'Std':>6}")
print("  " + "-" * 35)
for frac in CALIBRATION_FRACTIONS:
    f1s = cal_results[frac]
    if f1s:
        print(f"  {int(frac*100):>3}%{'':12}  {np.mean(f1s):>8.3f}  {np.std(f1s):>6.3f}")
