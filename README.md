# EEG-Based Cognitive State Detection

This project detects pilot cognitive state from EEG and physiological signals recorded during non-flight cognitive experiments. Raw EEG is transformed into frequency-domain band power features, which are used to train and evaluate several classifiers under both within-session and cross-subject evaluation schemes.

## Report

This project was done as a part of the course AAE4009 DATA SCIENCE AND DATA-DRIVEN OPTIMIZATION IN AIRLINE AND AIRPORT OPERATIONS at the Hong Kong Polytechnic University. A technical report detailing the project can be found at:

[Technical Report (PDF)](report.pdf)

Note that the results in the report differ from the ones in main.ipynb, as the SSS-results are on a smaller, 4-pilot dataset. The report posits that the frequency domain transformation led to better results than performing classification on the time domain features. This was true for the smaller dataset, but isn't the case for the full 18-pilot dataset (the inverse is actually true), which is a noteworthy result.

## Dataset

EEG (20 channels at 256 Hz), ECG, respiration (R), and GSR signals from 18 pilots across three experiments:

| Experiment | Task | Active event label |
|---|---|---|
| CA | Controlled Attention | C |
| DA | Divided Attention | D |
| SS | Sustained Surveillance | B |

Event `A` is the resting baseline in all experiments. Data is sourced from the [Reducing Commercial Aviation Fatalities](https://www.kaggle.com/competitions/reducing-commercial-aviation-fatalities/data), a Booz Allen Hamilton competition dataset from 2018. It is not included in this repository — place the following files in `data/` before running any scripts:

| File | Description |
|---|---|
| `train_original.csv` | Raw 18-pilot time-domain dataset |
| `large_transformed_train.csv` | Pre-computed band power features (output of `feature_engineering.py`) |

## Project Structure

```
├── feature_engineering.py   # Welch PSD transformation + diagnostic validation
├── eda.py                   # Exploratory analysis (raw data, per-pilot state timelines, pilot variability)
├── statistical_analysis.py  # ANOVA feature separability (raw + freq-domain, 18-pilot)
├── classification.py        # Within-session SSS + pilot-out cross-validation + calibration curve
├── main.ipynb               # Full pipeline notebook with rendered outputs
├── data/                    # Raw and transformed datasets (gitignored)
├── results/                 # Output result CSVs (gitignored)
└── figures/                 # Output plots (gitignored)
```

## Pipeline

**1. Feature Engineering** (`feature_engineering.py`)

Applies a sliding window (256 samples, 75% overlap) Welch PSD to extract delta/theta/alpha/beta/gamma band powers for each EEG channel, plus mean, std, and delta aggregates for physiological signals. Runs diagnostic checks on the output (zero-variance features, outliers, dominant band validation).

**2. Exploratory Data Analysis** (`eda.py`)

- Class balance and per-class EEG distributions
- Cognitive state activation timelines per pilot and experiment
- Between-pilot variability in the frequency domain (coefficient of variation, pairwise Euclidean distances, pilot η² vs class η²)

**3. Statistical Analysis** (`statistical_analysis.py`)

One-way ANOVA η² ranks feature separability for raw and frequency-domain features across the full 18-pilot dataset.

**4. Classification** (`classification.py`)

Evaluates LightGBM, XGBoost, Random Forest, SVM (Linear), KNN, and MLP under two schemes on the 18-pilot dataset:

- **Within-session SSS**: 5-fold stratified shuffle split — optimistic upper bound due to autocorrelation between adjacent overlapping windows.
- **Pilot-out CV**: Leave-one-pilot-out — realistic cross-subject performance. Also includes a calibration curve showing how much held-out pilot data is needed to close the generalisation gap.

## Running the Scripts

Run in order (each script's output feeds the next):

```bash
python feature_engineering.py   # → data/large_transformed_train.csv, data/anova_band_features.csv
python eda.py                   # → figures/*.png  (requires data/anova_band_features.csv)
python statistical_analysis.py  # → results/anova_*.csv, figures/anova_*.png
python classification.py        # → results/*_results.csv, figures/fold_analysis.png
```

Or open `main.ipynb` for a classification-only notebook that covers SSS and pilot-out CV on both the frequency-domain and raw datasets, enabling direct comparison between evaluation schemes and feature representations.

## Dependencies

```
numpy pandas scipy matplotlib seaborn scikit-learn lightgbm xgboost
```

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn lightgbm xgboost
```

## Future Work

For the scope of this project (time frame and computational resources), hyperparameters were treated as fixed based on reasonable defaults. Future work should focus on tuning these through cross-validation (e.g., nested cross-validation), as well as considering employing deep-learning models on the raw EEG amplitude signals.