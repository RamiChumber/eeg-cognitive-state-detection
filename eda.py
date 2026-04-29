"""
Exploratory Data Analysis — EEG Cognitive State Detection

Section 1: Raw time-domain data — class balance, signal distributions,
           ANOVA feature separability, temporal traces, inter-channel correlations
Section 2: Per-pilot cognitive state activation over time
Section 3: Frequency-domain data — pilot variability, pairwise pilot distances,
           pilot identity vs cognitive state variance

Inputs:  data/train.csv                 (raw 18-pilot dataset)
         data/transformed_train.csv     (band power features, 18-pilot)
         data/anova_band_features.csv   (ANOVA rankings — from feature_engineering.py)
Outputs: figures/*.png
         data/pilot_variability.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform

os.makedirs("figures", exist_ok=True)
os.makedirs("data",    exist_ok=True)

features_n = [
    "eeg_fp1","eeg_f7","eeg_f8","eeg_t4","eeg_t6","eeg_t5","eeg_t3","eeg_fp2",
    "eeg_o1","eeg_p3","eeg_pz","eeg_f3","eeg_fz","eeg_f4","eeg_c4","eeg_p4",
    "eeg_poz","eeg_c3","eeg_cz","eeg_o2","ecg","r","gsr"
]
PHYS_COLS = ["ecg", "r", "gsr"]
BANDS     = ["delta", "theta", "alpha", "beta", "gamma"]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: RAW TIME-DOMAIN EDA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: Raw time-domain EDA")
print("=" * 60)

df = pd.read_csv("data/train.csv")
df["pilot"] = 100 * df["seat"] + df["crew"]

# 1a. Class balance
print("\n=== Target distribution ===")
print(df["event"].value_counts(normalize=True).mul(100).round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["event"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="k")
axes[0].set_title("Class counts")
axes[0].set_xlabel("Event label")
pd.crosstab(df["experiment"], df["event"], normalize="index").mul(100)\
  .plot(kind="bar", ax=axes[1], colormap="Set2")
axes[1].set_title("Class share per experiment (%)")
axes[1].set_xlabel("Experiment")
plt.tight_layout()
plt.savefig("figures/class_distribution.png", dpi=150)
plt.show()

# 1b. Per-class distributions: box plots
n_cols = 4
n_rows = int(np.ceil(len(features_n) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3))
axes = axes.flatten()
for i, feat in enumerate(features_n):
    df.boxplot(column=feat, by="event", ax=axes[i], notch=True)
    axes[i].set_title(feat, fontsize=9)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("EEG feature distributions by event class", y=1.02)
plt.tight_layout()
plt.savefig("figures/eeg_boxplots.png", dpi=150)
plt.show()

# 1c. Class-mean heatmap (z-scored)
class_means   = df.groupby("event")[features_n].mean()
class_means_z = (class_means - class_means.mean()) / class_means.std()
plt.figure(figsize=(14, 4))
sns.heatmap(class_means_z, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.4, cbar_kws={"label": "z-score"})
plt.title("Z-scored class-mean EEG activation")
plt.tight_layout()
plt.savefig("figures/class_mean_heatmap.png", dpi=150)
plt.show()

# 1d. Feature separability: one-way ANOVA + eta²
groups = [df.loc[df["event"] == c, features_n] for c in df["event"].unique()]
anova_rows = []
for feat in features_n:
    f, p = stats.f_oneway(*[g[feat].dropna() for g in groups])
    grand_mean = df[feat].mean()
    ss_between = sum(len(g) * (g[feat].mean() - grand_mean)**2 for g in groups)
    ss_total   = ((df[feat] - grand_mean)**2).sum()
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    anova_rows.append({"feature": feat, "F": f, "p": p, "eta2": eta2})

anova_df = pd.DataFrame(anova_rows).sort_values("eta2", ascending=False)
print("\n=== ANOVA separability (top 10) ===")
print(anova_df.head(10).to_string(index=False))

plt.figure(figsize=(10, 5))
anova_df.set_index("feature")["eta2"].plot(kind="bar", color="teal", edgecolor="k")
plt.axhline(0.06, color="r",       linestyle="--", label="η²=0.06 (medium effect)")
plt.axhline(0.14, color="darkred", linestyle="--", label="η²=0.14 (large effect)")
plt.title("Feature separability — η² (ANOVA effect size, raw features)")
plt.ylabel("η²")
plt.legend()
plt.tight_layout()
plt.savefig("figures/feature_separability_raw.png", dpi=150)
plt.show()

# 1e. Temporal signal traces for top-4 discriminative features
top_feats = anova_df.head(4)["feature"].tolist()
fig, axes = plt.subplots(len(top_feats), 1,
                         figsize=(14, 2.5 * len(top_feats)), sharex=True)
for ax, feat in zip(axes, top_feats):
    for evt, grp in df.groupby("event"):
        ax.plot(grp["time"].values[:500], grp[feat].values[:500],
                alpha=0.6, linewidth=0.8, label=f"event {evt}")
    ax.set_ylabel(feat, fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
axes[-1].set_xlabel("Time")
plt.suptitle("Top-4 discriminative EEG channels (first 500 samples per class)")
plt.tight_layout()
plt.savefig("figures/temporal_traces.png", dpi=150)
plt.show()

# 1f. Inter-channel correlation per class
eeg_only = [f for f in features_n if f.startswith("eeg")]
classes  = sorted(df["event"].unique())
fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4.5))
for ax, cls in zip(axes, classes):
    corr = df.loc[df["event"] == cls, eeg_only].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.2,
                cbar=cls == classes[-1], xticklabels=False,
                yticklabels=[c.replace("eeg_", "") for c in eeg_only])
    ax.set_title(f"Event {cls}")
plt.suptitle("EEG inter-channel correlation per class")
plt.tight_layout()
plt.savefig("figures/correlation_matrices.png", dpi=150)
plt.show()

print("\n=== Class imbalance ratio ===")
counts   = df["event"].value_counts()
majority = counts.max()
for cls, cnt in counts.items():
    print(f"  event {cls}: {cnt:>5}  ratio {majority/cnt:.1f}:1")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PER-PILOT COGNITIVE STATE ACTIVATION OVER TIME
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: Per-pilot state activation over time")
print("=" * 60)

ACTIVE_EVENT = {"CA": "C", "DA": "D", "SS": "B"}
EXP_COLORS   = {"CA": "#1D9E75", "DA": "#378ADD", "SS": "#D85A30"}
pilots       = sorted(df["pilot"].unique())
experiments  = ["CA", "DA", "SS"]

fig, axes = plt.subplots(
    nrows=len(pilots), ncols=len(experiments),
    figsize=(16, 2.8 * len(pilots)), sharey="row"
)

for row_idx, pilot in enumerate(pilots):
    for col_idx, exp in enumerate(experiments):
        ax     = axes[row_idx, col_idx]
        subset = df[(df["pilot"] == pilot) & (df["experiment"] == exp)].sort_values("time")

        if subset.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="grey", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        t         = subset["time"].values
        is_active = (subset["event"] == ACTIVE_EVENT[exp]).astype(int).values

        ax.fill_between(t, is_active, step="post",
                        color=EXP_COLORS[exp], alpha=0.75, linewidth=0)
        ax.fill_between(t, is_active, 1, step="post",
                        color="#E0E0E0", alpha=0.5, linewidth=0)

        transitions = np.where(np.diff(is_active) != 0)[0]
        for tr in transitions:
            ax.axvline(t[tr], color="black", linewidth=0.4, alpha=0.4)

        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["baseline", "active"], fontsize=7)
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(axis="x", labelsize=7)
        frac = is_active.mean()
        ax.set_title(f"{exp} — pilot {pilot}\n{frac*100:.1f}% in active state",
                     fontsize=8, pad=3)
        if col_idx == 0:
            ax.set_ylabel(f"Pilot {pilot}", fontsize=8)
        if row_idx == len(pilots) - 1:
            ax.set_xlabel("time", fontsize=8)

legend_handles = [
    mpatches.Patch(color=EXP_COLORS[e], alpha=0.75, label=f"{e} active state")
    for e in experiments
] + [mpatches.Patch(color="#E0E0E0", alpha=0.8, label="baseline")]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=len(experiments) + 1, fontsize=9,
           bbox_to_anchor=(0.5, -0.01), frameon=False)
fig.suptitle("Cognitive state activation over time\n(per pilot per experiment)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("figures/state_over_time.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: FREQUENCY-DOMAIN DATA — PILOT VARIABILITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: Pilot variability in transformed feature space")
print("=" * 60)

band_df = pd.read_csv("data/transformed_train.csv")

eeg_band_cols = [
    c for c in band_df.columns
    if c.startswith("eeg_") and any(c.endswith(f"_{b}") for b in BANDS)
]
phys_feat_cols = [
    c for c in band_df.columns
    if any(c.startswith(p + "_") or c == p for p in PHYS_COLS)
]
all_feature_cols = eeg_band_cols + phys_feat_cols
pilots           = sorted(band_df["pilot"].unique())

# 3a. Coefficient of variation across pilots in baseline windows
baseline_df          = band_df[band_df["event"] == "A"]
pilot_baseline_means = baseline_df.groupby("pilot")[all_feature_cols].mean()

cv_rows = []
for feat in all_feature_cols:
    vals = pilot_baseline_means[feat].values
    mean = np.mean(vals)
    std  = np.std(vals)
    cv_rows.append({"feature": feat,
                    "cv": std / abs(mean) if mean != 0 else np.nan,
                    "mean": mean, "std": std})

cv_df = pd.DataFrame(cv_rows).sort_values("cv", ascending=False)
print("\n=== Coefficient of variation (CV) across pilots — baseline windows ===")
print(cv_df.head(20).to_string(index=False))
cv_df.to_csv("data/pilot_variability.csv", index=False)

top_variable = cv_df.head(6)["feature"].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, feat in enumerate(top_variable):
    for pilot in pilots:
        axes[i].hist(baseline_df[baseline_df["pilot"] == pilot][feat].values,
                     bins=40, alpha=0.5, label=f"Pilot {pilot}", density=True)
    axes[i].set_title(
        f"{feat}\nCV={cv_df.loc[cv_df['feature']==feat,'cv'].values[0]:.2f}", fontsize=9
    )
    axes[i].set_xlabel("Band power", fontsize=8)
    axes[i].set_ylabel("Density", fontsize=8)
    axes[i].legend(fontsize=7)
plt.suptitle("Baseline distributions per pilot — top 6 most variable features", fontsize=11)
plt.tight_layout()
plt.savefig("figures/pilot_baseline_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# 3b. Pairwise pilot distance matrix
pilot_vectors = pilot_baseline_means[all_feature_cols].values
dist_matrix   = squareform(pdist(pilot_vectors, metric="euclidean"))
dist_df       = pd.DataFrame(dist_matrix,
                              index=[f"Pilot {p}" for p in pilots],
                              columns=[f"Pilot {p}" for p in pilots])
print("\n=== Pairwise pilot distance (baseline feature space) ===")
print(dist_df.round(2).to_string())

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(dist_df, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Euclidean distance"})
ax.set_title("Pairwise pilot distance\n(baseline feature space)")
plt.tight_layout()
plt.savefig("figures/pilot_distance_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# 3c. Pilot η² vs class η² — which source of variance dominates per feature?
print("\n=== Pilot effect size (η²) ===")
pilot_eta2_rows = []
for feat in all_feature_cols:
    groups = [band_df.loc[band_df["pilot"] == p, feat].dropna().values for p in pilots]
    if any(len(g) == 0 or np.std(g) == 0 for g in groups):
        pilot_eta2_rows.append({"feature": feat, "pilot_eta2": 0.0})
        continue
    grand_mean = band_df[feat].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = ((band_df[feat] - grand_mean) ** 2).sum()
    pilot_eta2_rows.append({"feature": feat,
                             "pilot_eta2": float(ss_between / ss_total) if ss_total > 0 else 0.0})

pilot_eta2_df = pd.DataFrame(pilot_eta2_rows).sort_values("pilot_eta2", ascending=False)

anova_band_df = pd.read_csv("data/anova_band_features.csv")
comparison_df = pilot_eta2_df.merge(
    anova_band_df[["feature", "eta2"]].rename(columns={"eta2": "class_eta2"}),
    on="feature"
).sort_values("pilot_eta2", ascending=False)
comparison_df["pilot_dominance"] = (
    comparison_df["pilot_eta2"] / (comparison_df["class_eta2"] + 1e-6)
)

print("\n=== Pilot η² vs Class η² (top 20 pilot-dominated features) ===")
print(comparison_df.head(20)[
    ["feature", "pilot_eta2", "class_eta2", "pilot_dominance"]
].to_string(index=False))

top_feats = comparison_df.head(20)["feature"].tolist()
plot_df   = comparison_df[comparison_df["feature"].isin(top_feats)].copy()
x, width  = np.arange(len(top_feats)), 0.35

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - width/2, plot_df["pilot_eta2"], width, label="Pilot η²",  color="#D85A30", alpha=0.8)
ax.bar(x + width/2, plot_df["class_eta2"], width, label="Class η²",  color="#1D9E75", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(top_feats, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("η²")
ax.set_title("Pilot identity vs cognitive state — variance explained per feature\n"
             "(orange = explained by pilot, green = explained by cognitive state)")
ax.legend()
ax.axhline(0.06, color="grey", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.savefig("figures/pilot_vs_class_variance.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n=== Summary ===")
print(f"Mean pilot η²:  {comparison_df['pilot_eta2'].mean():.4f}")
print(f"Mean class η²:  {comparison_df['class_eta2'].mean():.4f}")
print(f"Features where pilot η² > class η²: "
      f"{(comparison_df['pilot_eta2'] > comparison_df['class_eta2']).sum()} / {len(comparison_df)}")
print(f"Mean pilot dominance ratio: {comparison_df['pilot_dominance'].mean():.1f}x")
