"""
Feature Engineering — EEG Cognitive State Detection

Transforms raw time-domain EEG + physiological signals into frequency-domain
band power features using Welch's PSD method, then validates the output.

Input:  data/train_original.csv          (raw, 18-pilot dataset)
Output: data/large_transformed_train.csv  (band power features)
        data/anova_band_features.csv      (ANOVA-ranked feature importance)
        figures/welch_verification.png    (diagnostic plot)
"""

import os
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal, stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

os.makedirs("data",    exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
RAW_PATH         = "data/train_original.csv"
TRANSFORMED_PATH = "data/large_transformed_train.csv"
ANOVA_OUT        = "data/anova_band_features.csv"

WINDOW_SIZE = 256
STEP_SIZE   = 64

features_n = [
    "eeg_fp1","eeg_f7","eeg_f8","eeg_t4","eeg_t6","eeg_t5","eeg_t3","eeg_fp2",
    "eeg_o1","eeg_p3","eeg_pz","eeg_f3","eeg_fz","eeg_f4","eeg_c4","eeg_p4",
    "eeg_poz","eeg_c3","eeg_cz","eeg_o2","ecg","r","gsr"
]
PHYS_COLS = ["ecg", "r", "gsr"]
EEG_COLS  = [f for f in features_n if f.startswith("eeg")]

BANDS = {
    "delta": (1,   4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
def estimate_fs(df, time_col="time", pilot_col="pilot", exp_col="experiment"):
    """Infer sampling frequency from median timestep within one pilot/experiment block."""
    sample = df[(df[pilot_col] == df[pilot_col].iloc[0]) &
                (df[exp_col]   == df[exp_col].iloc[0])
               ].sort_values(time_col)
    median_dt = sample[time_col].diff().median()
    fs = 1.0 / median_dt
    print(f"Estimated sampling frequency: {fs:.1f} Hz")
    return fs


def compute_features(df, eeg_cols, phys_cols, fs,
                     group_cols=["pilot", "experiment"],
                     time_col="time", label_col="event",
                     window_size=256, step_size=64):
    """
    Sliding-window feature extraction:
      EEG channels  → 5 band power values each (Welch PSD)
      Phys signals  → mean + std + delta (change from previous window)
    """
    records = []
    centre  = window_size // 2

    for (pilot, exp), group in df.groupby(group_cols):
        group   = group.sort_values(time_col)
        eeg_sig = group[eeg_cols].values
        phy_sig = group[phys_cols].values
        times   = group[time_col].values
        labels  = group[label_col].values

        prev_phys_means = None

        for start in range(0, len(group) - window_size + 1, step_size):
            row = {
                "pilot":      pilot,
                "experiment": exp,
                "time":       times[start + centre],
                "event":      labels[start + centre],
            }

            eeg_win = eeg_sig[start : start + window_size]
            for ch_idx, ch in enumerate(eeg_cols):
                freqs, psd = scipy_signal.welch(
                    eeg_win[:, ch_idx], fs=fs,
                    nperseg=min(window_size, 128), noverlap=64
                )
                for band, (lo, hi) in BANDS.items():
                    mask = (freqs >= lo) & (freqs < hi)
                    row[f"{ch}_{band}"] = (
                        np.trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
                    )

            phy_win = phy_sig[start : start + window_size]
            curr_phys_means = phy_win.mean(axis=0)
            for p_idx, pc in enumerate(phys_cols):
                row[f"{pc}_mean"]  = curr_phys_means[p_idx]
                row[f"{pc}_std"]   = phy_win[:, p_idx].std()
                row[f"{pc}_delta"] = (
                    curr_phys_means[p_idx] - prev_phys_means[p_idx]
                    if prev_phys_means is not None else 0.0
                )
            prev_phys_means = curr_phys_means
            records.append(row)

    return pd.DataFrame(records)


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


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════
print("Loading raw dataset...")
df = pd.read_csv(RAW_PATH)
df["pilot"] = 100 * df["seat"] + df["crew"]
print(f"  Rows: {len(df):,}  |  Pilots: {df['pilot'].nunique()}  |  Classes: {sorted(df['event'].unique())}")

fs = estimate_fs(df)

print("\nComputing band power features (this may take several minutes)...")
band_df = compute_features(df, EEG_COLS, PHYS_COLS, fs,
                           window_size=WINDOW_SIZE, step_size=STEP_SIZE)
print(f"Windows generated: {len(band_df):,}")

band_df.to_csv(TRANSFORMED_PATH, index=False)
print(f"Saved: {TRANSFORMED_PATH}")

eeg_band_cols    = [c for c in band_df.columns if any(f"_{b}" in c for b in BANDS)]
phys_feat_cols   = [c for c in band_df.columns if any(c.startswith(p) for p in PHYS_COLS)]
all_feature_cols = eeg_band_cols + phys_feat_cols

print(f"\nEEG band features:  {len(eeg_band_cols)}")
print(f"Phys features:      {len(phys_feat_cols)}")
print(f"Total features:     {len(all_feature_cols)}")

# ANOVA feature ranking (used downstream by eda.py and classification.py)
print("\nRanking features by ANOVA η²...")
anova_df = anova_eta2(band_df, all_feature_cols)
print("Top 20 features:")
print(anova_df.head(20)[["feature", "eta2"]].to_string(index=False))

for name, thresh in [("strict (η²>0.02)", 0.02), ("medium (η²>0.01)", 0.01), ("liberal (η²>0.005)", 0.005)]:
    print(f"  {name}: {(anova_df['eta2'] > thresh).sum()} features")

anova_df.to_csv(ANOVA_OUT, index=False)
print(f"Saved: {ANOVA_OUT}")

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DIAGNOSTIC CHECKS")
print("=" * 60)

# Check 1: Zero-variance features (delta band at nperseg=128)
zero_var = [c for c in all_feature_cols if band_df[c].std() == 0]
print(f"\nCheck 1 — Zero-variance features: {len(zero_var)}")
if zero_var:
    nperseg  = 128
    freq_res = fs / nperseg
    n_bins   = int((BANDS["delta"][1] - BANDS["delta"][0]) / freq_res)
    print(f"  Welch freq resolution at nperseg={nperseg}: {freq_res:.2f} Hz/bin")
    print(f"  Delta band gets {n_bins} bin(s) — "
          f"{'sufficient' if n_bins >= 2 else 'insufficient (expected at this nperseg)'}")

# Check 2: Outlier windows (>5σ)
n_affected = sum(
    1 for feat in eeg_band_cols
    if (band_df[feat] > band_df[feat].mean() + 5 * band_df[feat].std()).any()
)
print(f"\nCheck 2 — Features with outlier windows (>5σ): {n_affected}")

# Check 3: Near-zero total EEG power
total_power          = band_df[eeg_band_cols].sum(axis=1)
zero_power_threshold = total_power.quantile(0.001)
n_zero_power         = (total_power < zero_power_threshold).sum()
print(f"\nCheck 3 — Near-zero power windows: {n_zero_power} ({100*n_zero_power/len(band_df):.2f}%)")

# Check 4: Dominant band at baseline (alpha expected for posterior channels)
baseline_df          = band_df[band_df["event"] == "A"]
dominant_band_counts = {b: 0 for b in BANDS}
for ch in EEG_COLS:
    band_means = {b: baseline_df[f"{ch}_{b}"].mean() for b in BANDS if f"{ch}_{b}" in baseline_df.columns}
    if band_means:
        dominant_band_counts[max(band_means, key=band_means.get)] += 1
print("\nCheck 4 — Dominant band per channel at baseline (alpha expected):")
for band, count in dominant_band_counts.items():
    print(f"  {band:<8} {'█' * count} ({count} channels)")

# Visual: raw signal and PSD for first pilot/experiment
pilot_id     = df["pilot"].iloc[0]
exp_id       = df["experiment"].iloc[0]
sample_pilot = (df[(df["pilot"] == pilot_id) & (df["experiment"] == exp_id)]
                .sort_values("time").head(512))

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for i, ch in enumerate(EEG_COLS[:3]):
    axes[0, i].plot(sample_pilot["time"].values, sample_pilot[ch].values,
                    linewidth=0.6, color="#1D9E75")
    axes[0, i].set_title(f"Raw: {ch}", fontsize=9)
    axes[0, i].set_xlabel("time", fontsize=8)
    axes[0, i].set_ylabel("amplitude", fontsize=8)

    sig = sample_pilot[ch].values.astype(float)
    freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=min(len(sig), 128), noverlap=64)
    axes[1, i].semilogy(freqs, psd, linewidth=0.8, color="#D85A30")
    axes[1, i].set_xlim(0, 60)
    for band, (lo, hi) in BANDS.items():
        axes[1, i].axvspan(lo, hi, alpha=0.08, label=band)
    axes[1, i].set_title(f"PSD: {ch}", fontsize=9)
    axes[1, i].set_xlabel("frequency (Hz)", fontsize=8)
    axes[1, i].set_ylabel("power", fontsize=8)
    if i == 0:
        axes[1, i].legend(fontsize=6, loc="upper right")

plt.suptitle(f"Raw signal and Welch PSD — pilot {pilot_id}, first 512 samples", fontsize=11)
plt.tight_layout()
plt.savefig("figures/welch_verification.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: figures/welch_verification.png")

print(f"\nSUMMARY")
print(f"  Zero-variance features:  {len(zero_var)}")
print(f"  Features with outliers:  {n_affected}")
print(f"  Near-zero power windows: {n_zero_power}")
