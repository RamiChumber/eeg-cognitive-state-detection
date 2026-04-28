"""
Statistical Analysis — Feature Separability via One-Way ANOVA

Computes ANOVA η² (effect size) for both raw time-domain and frequency-domain
features across the 18-pilot dataset.

Inputs:  data/train_original.csv          (18-pilot raw)
         data/large_transformed_train.csv  (18-pilot transformed)
Outputs: results/anova_raw_18pilots.csv
         results/anova_transformed_18pilots.csv
         figures/anova_raw_18pilots.png
         figures/anova_transformed_18pilots.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

RAW_PATH         = "data/train_original.csv"
TRANSFORMED_PATH = "data/large_transformed_train.csv"

RAW_FEATURES = [
    "eeg_fp1","eeg_f7","eeg_f8","eeg_t4","eeg_t6","eeg_t5","eeg_t3","eeg_fp2",
    "eeg_o1","eeg_p3","eeg_pz","eeg_f3","eeg_fz","eeg_f4","eeg_c4","eeg_p4",
    "eeg_poz","eeg_c3","eeg_cz","eeg_o2","ecg","r","gsr"
]
PHYS_COLS = ["ecg", "r", "gsr"]
BANDS     = ["delta", "theta", "alpha", "beta", "gamma"]


def anova_eta2(df, feature_cols, target_col="event"):
    """One-way ANOVA per feature; returns DataFrame ranked by eta-squared."""
    classes = df[target_col].unique()
    rows    = []
    for feat in feature_cols:
        groups = [df.loc[df[target_col] == c, feat].dropna().values for c in classes]
        if any(len(g) == 0 or np.std(g) == 0 for g in groups):
            rows.append({"feature": feat, "F": np.nan, "p": np.nan, "eta2": 0.0})
            continue
        f_stat, p_val = stats.f_oneway(*groups)
        f_stat = float(np.nan_to_num(f_stat, nan=0.0, posinf=0.0))
        p_val  = float(np.nan_to_num(p_val,  nan=1.0, posinf=1.0))
        grand_mean = df[feat].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total   = ((df[feat] - grand_mean) ** 2).sum()
        eta2       = float(ss_between / ss_total) if ss_total > 0 else 0.0
        rows.append({"feature": feat, "F": f_stat, "p": p_val, "eta2": eta2})
    return pd.DataFrame(rows).sort_values("eta2", ascending=False).reset_index(drop=True)


def plot_anova(anova_df, title, save_path, top_n=40):
    plot_data = anova_df.head(top_n)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(plot_data)), plot_data["eta2"],
           color="#1D9E75", alpha=0.8, edgecolor="none")
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(plot_data["feature"], rotation=45, ha="right", fontsize=7)
    ax.axhline(0.06, color="red",     linestyle="--", linewidth=0.9, label="η²=0.06 (medium)")
    ax.axhline(0.14, color="darkred", linestyle="--", linewidth=0.9, label="η²=0.14 (large)")
    ax.set_ylabel("η²")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. RAW TIME-DOMAIN ANOVA (18 pilots)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading raw dataset (18-pilot)...")
raw_df = pd.read_csv(RAW_PATH)
raw_df["pilot"] = 100 * raw_df["seat"] + raw_df["crew"]
print(f"  Rows: {len(raw_df):,}  |  Pilots: {raw_df['pilot'].nunique()}  |  "
      f"Classes: {sorted(raw_df['event'].unique())}")

# Subsample for speed — stratify by event to preserve class proportions
sample_raw = (raw_df.groupby("event", group_keys=False)
                    .apply(lambda x: x.sample(min(len(x), 25000), random_state=42))
                    .reset_index(drop=True))
print(f"Subsampled to {len(sample_raw):,} rows for ANOVA\n")

print("Running ANOVA on raw features...")
anova_raw = anova_eta2(sample_raw, RAW_FEATURES)
print("\nTop 10 raw features by η²:")
print(anova_raw.head(10)[["feature", "eta2", "F", "p"]].to_string(index=False))

anova_raw.to_csv("results/anova_raw_18pilots.csv", index=False)
plot_anova(anova_raw,
           title="Feature separability — η² (ANOVA, raw time-domain, 18 pilots)",
           save_path="figures/anova_raw_18pilots.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FREQUENCY-DOMAIN ANOVA (18 pilots)
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading transformed dataset (18-pilot)...")
band_df = pd.read_csv(TRANSFORMED_PATH)
print(f"  Rows: {len(band_df):,}  |  Pilots: {band_df['pilot'].nunique()}")

eeg_band_cols = [
    c for c in band_df.columns
    if c.startswith("eeg_") and any(c.endswith(f"_{b}") for b in BANDS)
]
phys_feat_cols = [
    c for c in band_df.columns
    if any(c.startswith(p + "_") or c == p for p in PHYS_COLS)
]
all_transformed_cols = eeg_band_cols + phys_feat_cols

zero_var = [f for f in all_transformed_cols if band_df[f].std() == 0]
if zero_var:
    print(f"Dropping {len(zero_var)} zero-variance features")
all_transformed_cols = [f for f in all_transformed_cols if f not in zero_var]

print(f"Running ANOVA on {len(all_transformed_cols)} transformed features...")
anova_band = anova_eta2(band_df, all_transformed_cols)
print("\nTop 20 transformed features by η²:")
print(anova_band.head(20)[["feature", "eta2"]].to_string(index=False))

anova_band.to_csv("results/anova_transformed_18pilots.csv", index=False)
plot_anova(anova_band,
           title="Feature separability — η² (ANOVA, frequency-domain, 18 pilots)",
           save_path="figures/anova_transformed_18pilots.png",
           top_n=50)

# ══════════════════════════════════════════════════════════════════════════════
# 3. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("SUMMARY")
print("=" * 55)
print(f"\nRaw time-domain (18 pilots):")
print(f"  η² > 0.01: {(anova_raw['eta2'] > 0.01).sum()} / {len(anova_raw)}")
print(f"  η² > 0.06: {(anova_raw['eta2'] > 0.06).sum()} / {len(anova_raw)}")
print(f"  Top: {anova_raw.iloc[0]['feature']} (η²={anova_raw.iloc[0]['eta2']:.4f})")
print(f"\nFrequency-domain (18 pilots):")
print(f"  η² > 0.01: {(anova_band['eta2'] > 0.01).sum()} / {len(anova_band)}")
print(f"  η² > 0.06: {(anova_band['eta2'] > 0.06).sum()} / {len(anova_band)}")
print(f"  Top: {anova_band.iloc[0]['feature']} (η²={anova_band.iloc[0]['eta2']:.4f})")
