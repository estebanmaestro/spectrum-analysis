# Improved ON/OFF Scoring System — Results (Updated)

## 1. Dataset

| Property | Previous | Updated |
|----------|----------|---------|
| Labeled spectrums | 230 | **272** |
| ON / OFF | 160 / 70 | **188 / 84** |
| Sites | 33 | **38** |
| Sites with both ON & OFF | ~25 | **28** |
| Raw waveform match | 230/242 | **272/272** |

New sites added: `Harq_ST1_PT`, `Harq_ST2_PT`, `Harq_ST3_PT`, `Prairie_State_2`, `NC2`, `GAPAC_TG5`, `Mosaic_Riverview`, `Millenium_GT1_step`. Updated labels on `Towantic_STG`, `GAPAC_TG3`, `GAPAC_TG4`, and others.

Cross-validation: 5-fold StratifiedGroupKFold (site-aware).

## 2. Existing 6-Feature Median-Vote System (Baseline)

**Features**: `spectral_slope_norm`, `lr_B2_B4`, `clr_B2`, `clr_B4`, `alr_B4`, `lr_B3_B4`

| Metric | Value |
|--------|-------|
| CV Accuracy | 0.792 +/- 0.077 |
| F1 | 0.853 |
| Precision | 0.826 |
| Recall | 0.883 |
| AUC | 0.794 |

Confusion matrix (OOF):

|  | Pred OFF | Pred ON |
|--|----------|---------|
| **True OFF** | 49 | 35 |
| **True ON** | 22 | 166 |

## 3. New Features Engineered

48 new band-level features computed from raw spectrum waveforms (same families as before):
- Per-band spectral slopes (log-log), curvature, entropy, flatness, CV
- Per-band crest factor, peak-to-mean, rolloff
- Inter-band transitions and slope differences
- Energy concentration, normalised centroid

**52 features** significant at BH-corrected p < 0.05 (out of 77 tested).

### Top 10 Features by Effect Size

| Rank | Feature | Source | Cohen's d | AUC |
|------|---------|--------|-----------|-----|
| 1 | `ssn_B1` | new | +0.856 | 0.697 |
| 2 | `lr_B2_B4` | original | +0.841 | 0.745 |
| 3 | `slope_diff_B1_B2` | new | -0.757 | 0.687 |
| 4 | `rolloff_norm_B3` | new | -0.743 | 0.679 |
| 5 | `spectral_spread` | original | -0.715 | 0.662 |
| 6 | `spectral_slope_norm` | original | -0.700 | 0.791 |
| 7 | `curvature_B2` | new | +0.691 | 0.708 |
| 8 | `lr_B3_B4` | original | +0.665 | 0.679 |
| 9 | `slope_loglog_B3` | new | -0.642 | 0.648 |
| 10 | `ilr_3` | original | +0.832 | 0.740 |

## 4. Recommended 9-Feature Scoring System

Feature-count optimisation (3–25 features, median thresholds, Cohen's d weights) found **9 features** maximise cross-validated accuracy:

| # | Feature | Direction | Weight (|d|) | Description |
|---|---------|-----------|-------------|-------------|
| 1 | `ssn_B1` | higher → ON | 0.856 | Normalised spectral slope in B1 (30–300 kHz) |
| 2 | `lr_B2_B4` | higher → ON | 0.841 | Log-ratio of B2 to B4 band power |
| 3 | `slope_diff_B1_B2` | lower → ON | 0.757 | Slope change from B1 to B2 band |
| 4 | `rolloff_norm_B3` | lower → ON | 0.743 | Normalised 85%-energy rolloff in B3 |
| 5 | `spectral_spread` | lower → ON | 0.715 | Width of power distribution |
| 6 | `spectral_slope_norm` | lower → ON | 0.700 | Global normalised spectral slope |
| 7 | `curvature_B2` | higher → ON | 0.691 | Spectral curvature in B2 (300 kHz–3 MHz) |
| 8 | `lr_B3_B4` | higher → ON | 0.665 | Log-ratio of B3 to B4 band power |
| 9 | `slope_loglog_B3` | lower → ON | 0.642 | Power-law exponent in B3 (3–30 MHz) |

**Scoring rule**: For each feature, compare against the training-fold median. If the value is on the "ON side" (direction column), the vote is weighted by |Cohen's d|. Votes are summed and normalised to [0, 1]. Threshold: weighted_score_norm >= 0.35 → ON.

5 of 9 features are **new band-level features** (`ssn_B1`, `slope_diff_B1_B2`, `rolloff_norm_B3`, `curvature_B2`, `slope_loglog_B3`).

## 5. Performance Comparison

### Cross-Validated Results (5-fold, site-aware)

| System | Accuracy | F1 | Precision | Recall | AUC |
|--------|----------|-----|-----------|--------|-----|
| Original (6-feat median) | 0.792 +/- 0.077 | 0.853 | 0.826 | 0.883 | 0.794 |
| **Recommended (9-feat hybrid)** | **0.827 +/- 0.080** | **0.883** | — | — | **0.826** |

### Recommended System Confusion Matrix (OOF)

|  | Pred OFF | Pred ON |
|--|----------|---------|
| **True OFF** | 48 | 36 |
| **True ON** | 11 | 177 |

### Improvement Summary

| Metric | Original | Recommended | Change |
|--------|----------|-------------|--------|
| CV Accuracy | 79.2% | **82.7%** | **+3.5%** |
| F1-score | 0.853 | **0.883** | **+0.030** |
| AUC | 0.794 | **0.826** | **+0.032** |
| False Negatives | 22 | **11** | **-50%** |

### In-Sample Metrics (all 272 spectrums, decision_criteria.csv)

| Metric | Value |
|--------|-------|
| Accuracy | 0.805 |
| F1 | 0.856 |
| Precision | 0.873 |
| Recall | 0.840 |

Confusion matrix:

|  | Pred OFF | Pred ON |
|--|----------|---------|
| **True OFF** | 61 | 23 |
| **True ON** | 30 | 158 |

## 6. Per-Site Analysis (Existing System, Sites with Both Classes)

Sites where the system performs well (accuracy >= 0.80):
- Har_CT1, HARQCT1, HARQCT2, HARQCT3, VAN1, Van_GT1, Van_GT2, SQN2, OPPD_NC1

Sites that remain challenging (accuracy < 0.50):
- Decatur, Harq_ST1, WBN2, Harq_ST3

These difficult sites have spectra whose ON patterns deviate from the global average (e.g., Decatur's extreme B3 hump, Harq's PT-specific patterns).

## 7. Feature Count Optimisation

| N features | CV Accuracy | F1 | AUC |
|-----------|-------------|-----|-----|
| 3 | 0.724 | 0.804 | 0.685 |
| 5 | 0.739 | 0.831 | 0.760 |
| 7 | 0.798 | 0.868 | 0.818 |
| **9** | **0.827** | **0.883** | **0.826** |
| 12 | 0.739 | 0.818 | 0.822 |
| 15 | 0.765 | 0.838 | 0.828 |
| 25 | 0.765 | 0.835 | 0.846 |

Peak accuracy at 9 features. More features increase AUC (ranking quality) but decrease accuracy (threshold generalisation degrades with small per-site samples).

## 8. Files Generated

| File | Description |
|------|-------------|
| `src/improved_on_off_scoring.ipynb` | Full analysis notebook (source) |
| `src/improved_on_off_scoring_executed.ipynb` | Executed notebook with all outputs |
| `data/results_clean.csv` | Input data (old vote columns removed) |
| `data/decision_criteria.csv` | 272 spectrums with 9 features, votes, scores, predictions |
| `data/scoring_parameters.csv` | All 25 non-redundant feature parameters |
| `data/feature_association_stats.csv` | All 77 features with statistical tests |
| `data/improved_scoring_results.csv` | Per-spectrum detailed results |
| `src/_scoring_figs/*.png` | 9 analysis figures |
