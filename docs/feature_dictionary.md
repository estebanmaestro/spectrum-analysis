# Feature Dictionary — ON/OFF Spectrum Analysis

**Version:** 2.0  
**Date:** April 2026  
**Original features:** 52 (27 scale-dependent + 25 scale-invariant)  
**Compositional log-metric features:** 20 (all scale-invariant)  
**Total:** 72

---

## Input Data

Each spectrum consists of two arrays:

- **freqs** — frequency values in Hz
- **powers_uv** — amplitude values in µV

All features are derived from the **power array** `pwr = powers_uv²` (units: µV²) and the frequency array.

---

## Frequency Band Definitions

| Band | Label | Low Bound | High Bound | Description |
|------|-------|-----------|------------|-------------|
| B1 | Low frequency | 30 kHz | 300 kHz | Sub-broadcast band |
| B2 | Mid-low frequency | 300 kHz | 3 MHz | MF/HF transition |
| B3 | Mid-high frequency | 3 MHz | 30 MHz | HF band |
| B4 | High frequency | 30 MHz | 100 MHz | VHF band |

---

## Feature Groups

### 1. Global Power Statistics

Aggregate statistics computed across the full spectrum.

| # | Feature | Formula | Units | Scale-Inv | Description |
|---|---------|---------|-------|-----------|-------------|
| 1 | `total_power` | `Σ pwr` | µV² | No | Total integrated power across all frequency bins. Primary indicator of overall emission level. |
| 2 | `mean_power` | `mean(pwr)` | µV² | No | Average power per frequency bin. |
| 3 | `max_power` | `max(pwr)` | µV² | No | Peak power value in any single frequency bin. |
| 4 | `std_power` | `std(pwr)` | µV² | No | Standard deviation of power across bins. Higher values indicate a more uneven spectral distribution. |

### 2. Band Power Statistics

For each of the four frequency bands (B1–B4), four statistics are computed on the power values within that band. This produces 4 × 4 = 16 features.

| # | Feature | Formula | Units | Scale-Inv | Description |
|---|---------|---------|-------|-----------|-------------|
| 5 | `B1_power` | `Σ pwr[B1]` | µV² | No | Total power in B1 (30–300 kHz). |
| 6 | `B1_mean` | `mean(pwr[B1])` | µV² | No | Mean power per bin in B1. |
| 7 | `B1_max` | `max(pwr[B1])` | µV² | No | Peak power bin in B1. |
| 8 | `B1_std` | `std(pwr[B1])` | µV² | No | Power variability within B1. |
| 9 | `B2_power` | `Σ pwr[B2]` | µV² | No | Total power in B2 (300 kHz – 3 MHz). |
| 10 | `B2_mean` | `mean(pwr[B2])` | µV² | No | Mean power per bin in B2. |
| 11 | `B2_max` | `max(pwr[B2])` | µV² | No | Peak power bin in B2. |
| 12 | `B2_std` | `std(pwr[B2])` | µV² | No | Power variability within B2. |
| 13 | `B3_power` | `Σ pwr[B3]` | µV² | No | Total power in B3 (3–30 MHz). |
| 14 | `B3_mean` | `mean(pwr[B3])` | µV² | No | Mean power per bin in B3. |
| 15 | `B3_max` | `max(pwr[B3])` | µV² | No | Peak power bin in B3. |
| 16 | `B3_std` | `std(pwr[B3])` | µV² | No | Power variability within B3. |
| 17 | `B4_power` | `Σ pwr[B4]` | µV² | No | Total power in B4 (30–100 MHz). |
| 18 | `B4_mean` | `mean(pwr[B4])` | µV² | No | Mean power per bin in B4. |
| 19 | `B4_max` | `max(pwr[B4])` | µV² | No | Peak power bin in B4. |
| 20 | `B4_std` | `std(pwr[B4])` | µV² | No | Power variability within B4. |

### 3. Band Power Fractions

Fraction of total power contained in each band. These are dimensionless ratios that are insensitive to absolute power level, making them comparable across sites with different baseline emissions.

| # | Feature | Formula | Units | Scale-Inv | Description |
|---|---------|---------|-------|-----------|-------------|
| 21 | `B1_frac` | `B1_power / total_power` | — | Yes | Proportion of total power in the low-frequency band (30–300 kHz). |
| 22 | `B2_frac` | `B2_power / total_power` | — | Yes | Proportion of total power in the mid-low band (300 kHz – 3 MHz). |
| 23 | `B3_frac` | `B3_power / total_power` | — | Yes | Proportion of total power in the mid-high band (3–30 MHz). |
| 24 | `B4_frac` | `B4_power / total_power` | — | Yes | Proportion of total power in the high-frequency band (30–100 MHz). |

### 4. Log-Transformed Powers

Logarithmic transformation compresses the dynamic range of power values and makes distributions more symmetric. Useful for methods that assume normality or for gap-based clustering.

| # | Feature | Formula | Units | Scale-Inv | Description |
|---|---------|---------|-------|-----------|-------------|
| 25 | `log_total` | `log(1 + total_power)` | log(µV²) | No | Log-compressed total power. Used by the PowerGap clustering method to detect ON/OFF separation. |
| 26 | `B1_log` | `log(1 + B1_power)` | log(µV²) | No | Log-compressed B1 band power. |
| 27 | `B2_log` | `log(1 + B2_power)` | log(µV²) | No | Log-compressed B2 band power. Top-ranked feature in the supervised study (consensus rank #1). |
| 28 | `B3_log` | `log(1 + B3_power)` | log(µV²) | No | Log-compressed B3 band power. |
| 29 | `B4_log` | `log(1 + B4_power)` | log(µV²) | No | Log-compressed B4 band power. |

### 5. Band Power Ratios

Ratios between bands capture the relative energy distribution. They are dimensionless and scale-invariant — if all power levels double, the ratios remain unchanged. Useful for identifying characteristic spectral shapes independent of absolute emission levels.

| # | Feature | Formula | Units | Scale-Inv | Description |
|---|---------|---------|-------|-----------|-------------|
| 30 | `B2_B1_ratio` | `B2_power / B1_power` | — | Yes | Energy ratio of mid-low to low band. Values > 1 indicate more energy in the 300 kHz – 3 MHz range than below 300 kHz. |
| 31 | `B3_B1_ratio` | `B3_power / B1_power` | — | Yes | Energy ratio of mid-high to low band. |
| 32 | `B4_B1_ratio` | `B4_power / B1_power` | — | Yes | Energy ratio of high to low band. |
| 33 | `B4_B2_ratio` | `B4_power / B2_power` | — | Yes | Energy ratio of high to mid-low band. |
| 34 | `B3_B2_ratio` | `B3_power / B2_power` | — | Yes | Energy ratio of mid-high to mid-low band. |
| 35 | `hi_lo_ratio` | `(B3 + B4) / (B1 + B2)` | — | Yes | Ratio of upper-half to lower-half spectral energy. Summarizes the overall spectral balance in a single number. |
| 36 | `log_B2_B1` | `log(1 + B2_B1_ratio)` | — | Yes | Log-compressed B2/B1 ratio. Reduces the effect of extreme ratio values. |
| 37 | `log_B4_B1` | `log(1 + B4_B1_ratio)` | — | Yes | Log-compressed B4/B1 ratio. |
| 38 | `log_hi_lo` | `log(1 + hi_lo_ratio)` | — | Yes | Log-compressed high-to-low ratio. |

### 6. Spectral Shape Descriptors

Statistical moments and information-theoretic measures of the power distribution treated as a probability distribution (`normp = pwr / total_power`). These describe the shape of the spectrum independent of its magnitude.

| # | Feature | Formula | Units | Scale-Inv | Description |
|---|---------|---------|-------|-----------|-------------|
| 39 | `spectral_centroid` | `Σ(i × normp[i])` | bin index | Yes | Weighted-mean frequency index — the "center of mass" of the spectrum. Lower values mean energy is concentrated at lower frequencies. |
| 40 | `spectral_spread` | `√(Σ((i − centroid)² × normp))` | bin index | Yes | Standard deviation of the spectral distribution around the centroid. Larger values indicate energy spread across a wider frequency range. |
| 41 | `spectral_skew` | `scipy.stats.skew(pwr)` | — | Yes | Asymmetry of the power distribution. Positive skew means a long tail toward high-power bins (a few bins dominate). |
| 42 | `spectral_kurtosis` | `scipy.stats.kurtosis(pwr)` | — | Yes | Peakedness of the power distribution. High kurtosis means energy is concentrated in a few sharp peaks rather than spread evenly. |
| 43 | `spectral_entropy` | `H(normp) / log(N)` | — | Yes | Normalized Shannon entropy (0 to 1). A value near 1.0 means power is uniformly distributed across frequencies (flat spectrum). A value near 0 means power is concentrated in very few bins (peaked spectrum). |
| 44 | `spectral_flatness` | `exp(mean(log(pwr))) / mean(pwr)` | — | Yes | Wiener entropy — ratio of geometric mean to arithmetic mean of power. Values near 1.0 indicate a flat/white-noise-like spectrum. Values near 0 indicate a spectrum with prominent peaks. Also known as tonality coefficient. |

### 7. Peak and Slope Characteristics

Features that describe the dominant frequency, spectral tilt, and energy roll-off.

| # | Feature | Formula | Units | Scale-Inv | Description |
|---|---------|---------|-------|-----------|-------------|
| 45 | `peak_freq_mhz` | `freqs[argmax(pwr)] / 10⁶` | MHz | Yes | Frequency (in MHz) of the bin with the highest power. Identifies the dominant emission frequency. |
| 46 | `peak_to_mean` | `max(pwr) / mean(pwr)` | — | Yes | Ratio of the peak power to the mean power. High values indicate a single dominant narrowband emission; low values indicate broadband energy. |
| 47 | `spectral_slope` | `polyfit(i, log(1+pwr), 1)[0]` | log(µV²)/bin | Yes | Slope of a linear fit to the log-power vs. frequency-bin-index. Captures the rate at which power decays across the spectrum. More negative values = steeper roll-off from low to high frequencies. Running equipment typically changes this tilt. Consensus importance rank #2 in the supervised study. |
| 48 | `rolloff_freq_mhz` | `freq where cumsum(pwr) ≥ 0.85 × total` | MHz | Yes | The frequency (in MHz) below which 85% of the total spectral energy is contained. A low rolloff frequency means most energy is concentrated at low frequencies; a higher value indicates significant high-frequency content. |

### 8. Amplitude Shape

Features computed on the raw amplitude (µV) rather than power (µV²), characterizing the waveform's peakiness and variability.

| # | Feature | Formula | Units | Scale-Inv | Description |
|---|---------|---------|-------|-----------|-------------|
| 49 | `crest_factor` | `max(amplitude) / RMS(amplitude)` | — | Yes | Ratio of peak amplitude to RMS amplitude. High crest factor indicates sharp transient peaks in the spectrum. Low values indicate a more uniform amplitude envelope. |
| 50 | `amplitude_cv` | `std(amplitude) / mean(amplitude)` | — | Yes | Coefficient of variation of the amplitude. Dimensionless measure of relative variability. Higher values mean the spectrum has more contrast between peaks and valleys. Selected by RFECV as one of the 5 optimal features. |

### 9. Cross-Band Dynamics

Features that capture trends across the four frequency bands, summarizing how energy changes from B1 through B4.

| # | Feature | Formula | Units | Scale-Inv | Description |
|---|---------|---------|-------|-----------|-------------|
| 51 | `band_gradient` | `mean(diff(log(1 + Bx_power)))` | — | Yes | Average successive difference in log-band-power across B1 → B2 → B3 → B4. Positive values mean energy increases toward higher bands; negative values mean it decreases. Captures the overall direction of the spectral envelope. |
| 52 | `band_range` | `max(log(1+Bx)) − min(log(1+Bx))` | — | Yes | Dynamic range of log-band-powers across the four bands. Large values indicate strong contrast between the most and least energetic bands; small values indicate uniform energy distribution. |

---

## Compositional Log-Metric Features (Groups 10–14)

These 20 features were introduced in the compositional log-metric analysis to address a fundamental issue: band powers `(B1, B2, B3, B4)` are **compositional data** — parts of a whole (`total_power`). Standard linear models assume unconstrained Euclidean space, but compositional data lives on a simplex. The features below apply proper log-ratio transformations from compositional data analysis (Aitchison, 1986) to produce geometrically correct, fully scale-invariant representations.

All use `log(x)` (natural logarithm) rather than `log(1+x)`, which avoids the asymmetric compression of `log1p` when one band is much smaller than another.

**Notation**: `log_Bx = log(Bx_power + eps)`, `log_T = log(total_power + eps)`, `eps = 10⁻¹⁰`

### 10. Proper Pairwise Log-Ratios — Family A (6 features)

The log of the ratio between every unique pair of bands. Because `log(Bi/Bj) = log(Bi) - log(Bj)`, these are symmetric: `lr_B1_B2 = -lr_B2_B1`. Unlike the existing `log1p(ratio)` features (Group 5), proper log-ratios preserve full information when one band dominates another. Head-to-head analysis shows identical ROC-AUC but 2–13x larger Cohen's d effect sizes compared to existing ratio features, meaning better-separated class distributions for linear classifiers.

| # | Feature | Formula | Units | Description |
|---|---------|---------|-------|-------------|
| 53 | `lr_B1_B2` | `log(B1) - log(B2)` | — | Log energy ratio of low to mid-low band. Negative values indicate B2 carries more power than B1. |
| 54 | `lr_B1_B3` | `log(B1) - log(B3)` | — | Log energy ratio of low to mid-high band. |
| 55 | `lr_B1_B4` | `log(B1) - log(B4)` | — | Log energy ratio of low to high band. Equivalent to `band_gradient_proper` in direction. AUC = 0.695. |
| 56 | `lr_B2_B3` | `log(B2) - log(B3)` | — | Log energy ratio of mid-low to mid-high band. Captures the mid-spectrum tilt. AUC = 0.648. |
| 57 | `lr_B2_B4` | `log(B2) - log(B4)` | — | Log energy ratio of mid-low to high band. **Top-ranked Family A feature** (AUC = 0.752, Cohen's d = 0.893). Captures how much more energy the mid-low band carries relative to VHF. |
| 58 | `lr_B3_B4` | `log(B3) - log(B4)` | — | Log energy ratio of mid-high to high band. AUC = 0.693. |

### 11. Additive Log-Ratios (ALR) — Family B (4 features)

Each feature measures `log(total / Bx)`, i.e., how much of the total energy is NOT in band Bx, expressed in log-space. This is the Additive Log-Ratio (ALR) transformation, equivalent to `-log(Bx_frac)`. Compared to raw band fractions (Group 3), the log transform makes differences at small fractions (e.g., B4_frac = 0.002 vs 0.02) much more visible to linear models.

| # | Feature | Formula | Units | Description |
|---|---------|---------|-------|-------------|
| 59 | `alr_B1` | `log(total) - log(B1)` | — | Log complement of B1. Small values mean B1 dominates total power. AUC = 0.533. |
| 60 | `alr_B2` | `log(total) - log(B2)` | — | Log complement of B2. Small values mean B2 dominates. AUC = 0.572. |
| 61 | `alr_B3` | `log(total) - log(B3)` | — | Log complement of B3. AUC = 0.569. |
| 62 | `alr_B4` | `log(total) - log(B4)` | — | Log complement of B4. **Top-ranked Family B feature** (AUC = 0.701, Cohen's d = 0.779). Large values indicate B4 contributes little to total power. |

### 12. Centered Log-Ratios (CLR) — Family C (4 features)

The standard Aitchison transform for compositional data. Each band's log-power is centered by subtracting the geometric mean of all four bands' log-powers. The CLR coordinates sum to zero by construction. This removes absolute scale while preserving the full relative structure between bands. Any pairwise log-ratio can be reconstructed as a linear combination of CLR coordinates.

| # | Feature | Formula | Units | Description |
|---|---------|---------|-------|-------------|
| 63 | `clr_B1` | `log(B1) - mean(log(B1..B4))` | — | B1 power relative to the geometric mean of all bands. Positive = B1 is above average. AUC = 0.607. |
| 64 | `clr_B2` | `log(B2) - mean(log(B1..B4))` | — | B2 power relative to geometric mean. **Top-ranked Family C feature** (AUC = 0.746, Cohen's d = 0.820). Indicates B2 carries disproportionate energy compared to the average band. |
| 65 | `clr_B3` | `log(B3) - mean(log(B1..B4))` | — | B3 power relative to geometric mean. AUC = 0.569. |
| 66 | `clr_B4` | `log(B4) - mean(log(B1..B4))` | — | B4 power relative to geometric mean. AUC = 0.746, Cohen's d = 0.887. Strongly negative values (B4 well below average) are characteristic of ON state. |

### 13. Corrected Cross-Band Features — Family D (3 features)

Improved versions of existing cross-band features (Group 9) plus a truly scale-invariant spectral slope. Family D has the highest mean AUC (0.702) of all new families.

| # | Feature | Formula | Units | Description |
|---|---------|---------|-------|-------------|
| 67 | `band_gradient_proper` | `mean(diff(log(Bx)))` | — | Average successive difference in log-band-power using `log` instead of `log1p`. Equivalent to `lr_B1_B4 / 3`. Positive = energy increases toward higher bands. AUC = 0.695. |
| 68 | `band_range_proper` | `max(log(Bx)) - min(log(Bx))` | — | Dynamic range across bands in proper log-space. Measures spectral contrast between the strongest and weakest band. AUC = 0.634. |
| 69 | `spectral_slope_norm` | `polyfit(i, log(normp), 1)[0]` | 1/bin | Slope of a linear fit to `log(pwr / total)` vs frequency bin index. Unlike the original `spectral_slope` which fits `log(1 + pwr)` and retains absolute power dependence, this version operates on the normalized spectrum and is **truly scale-invariant**. AUC = 0.779, Cohen's d = 0.819. Ranked #2 overall among all new features. |

### 14. Isometric Log-Ratios (ILR) — Family E (3 features)

Three orthonormal coordinates derived from a sequential binary partition of the four bands. The ILR transform maps the 4-part composition into 3 unconstrained Euclidean dimensions — the mathematically correct representation for use with standard multivariate methods. Each coordinate contrasts a group of bands against the next.

| # | Feature | Formula | Units | Description |
|---|---------|---------|-------|-------------|
| 70 | `ilr_1` | `√(1/2) × (log(B1) - log(B2))` | — | Contrast between B1 and B2 (low vs mid-low). Positive = B1 dominant. AUC = 0.580. |
| 71 | `ilr_2` | `√(2/3) × (mean(log(B1,B2)) - log(B3))` | — | Contrast between the average of {B1, B2} and B3. Captures whether mid-high energy deviates from the low-frequency average. AUC = 0.607. |
| 72 | `ilr_3` | `√(3/4) × (mean(log(B1,B2,B3)) - log(B4))` | — | Contrast between the average of {B1, B2, B3} and B4. **Top-ranked Family E feature** (AUC = 0.746, Cohen's d = 0.887). Strongly positive values (B4 much weaker than the rest) characterize the ON state. |

---

## Scale-Invariant vs. Scale-Dependent

**Scale-invariant features** (45 of 72) are ratios, fractions, or shape descriptors that remain unchanged if the absolute power level is scaled up or down. They generalize better across sites that have different baseline emission levels or sensor calibrations. All 20 compositional log-metric features (Groups 10–14) are scale-invariant by construction.

**Scale-dependent features** (27 of 72) are raw power values or their log transforms. They carry information about absolute emission magnitude, which is relevant within a single site but may vary across sites for reasons unrelated to ON/OFF state (e.g., sensor distance, equipment model).

### Scale-Invariant Feature List (Original 25)

`B1_frac`, `B2_frac`, `B3_frac`, `B4_frac`, `B2_B1_ratio`, `B3_B1_ratio`, `B4_B1_ratio`, `B4_B2_ratio`, `B3_B2_ratio`, `hi_lo_ratio`, `log_B2_B1`, `log_B4_B1`, `log_hi_lo`, `spectral_centroid`, `spectral_spread`, `spectral_skew`, `spectral_kurtosis`, `spectral_entropy`, `spectral_flatness`, `peak_freq_mhz`, `peak_to_mean`, `spectral_slope`, `rolloff_freq_mhz`, `crest_factor`, `amplitude_cv`, `band_gradient`, `band_range`

### Compositional Log-Metric Features (20, all scale-invariant)

`lr_B1_B2`, `lr_B1_B3`, `lr_B1_B4`, `lr_B2_B3`, `lr_B2_B4`, `lr_B3_B4`, `alr_B1`, `alr_B2`, `alr_B3`, `alr_B4`, `clr_B1`, `clr_B2`, `clr_B3`, `clr_B4`, `band_gradient_proper`, `band_range_proper`, `spectral_slope_norm`, `ilr_1`, `ilr_2`, `ilr_3`

---

## Key Findings from Supervised Study (Original 52 Features)

| Rank | Feature | Univariate ROC-AUC | Consensus Importance Rank | RFECV Selected |
|------|---------|-------------------|--------------------------|----------------|
| 1 | `B2_log` | 0.888 | 1 | Yes |
| 2 | `spectral_slope` | 0.787 | 2 | No |
| 3 | `spectral_entropy` | — | 3 | No |
| 4 | `B3_power` | 0.749 | 4 | Yes |
| 5 | `spectral_kurtosis` | — | 5 | No |
| 6 | `B1_power` | 0.792 | — | Yes |
| 7 | `B1_log` | 0.792 | — | Yes |
| 8 | `amplitude_cv` | — | 12 | Yes |

The **RFECV-optimal subset** (5 features: `B1_power`, `B3_power`, `B1_log`, `B2_log`, `amplitude_cv`) achieves F1 = 0.856 — outperforming all 52 features (F1 = 0.822) — indicating that the remaining 47 features add noise rather than signal for ON/OFF classification.

---

## Key Findings from Compositional Log-Metric Analysis

### Top New Features by ROC-AUC

| Rank | Feature | Family | ROC-AUC | Cohen's d | MW p (FDR-adj) |
|------|---------|--------|---------|-----------|----------------|
| 1 | `spectral_slope_norm` | D | 0.779 | 0.819 | 8.3 × 10⁻¹¹ |
| 2 | `lr_B2_B4` | A | 0.752 | 0.893 | 4.0 × 10⁻⁹ |
| 3 | `clr_B2` | C | 0.746 | 0.820 | 6.0 × 10⁻⁹ |
| 4 | `clr_B4` | C | 0.746 | 0.887 | 6.0 × 10⁻⁹ |
| 5 | `ilr_3` | E | 0.746 | 0.887 | 6.0 × 10⁻⁹ |
| 6 | `alr_B4` | B | 0.701 | 0.779 | 2.5 × 10⁻⁶ |

### Head-to-Head: Log-Ratio vs Existing Features

ROC-AUC is identical (log is a monotonic transform, so rank order is preserved), but **Cohen's d is 2–13x larger** for proper log-ratios. This means class distributions are much better separated in log-space, which benefits linear classifiers:

| Pair | AUC | New d | Existing d | d improvement |
|------|-----|-------|-----------|---------------|
| `lr_B1_B2` vs `B2_B1_ratio` | 0.580 | 0.287 | 0.022 | 13x |
| `alr_B4` vs `B4_frac` | 0.701 | 0.779 | 0.246 | 3.2x |
| `alr_B2` vs `B2_frac` | 0.572 | 0.247 | 0.073 | 3.4x |
| `lr_B1_B4` vs `B4_B1_ratio` | 0.695 | 0.631 | 0.334 | 1.9x |

### Family-Level Performance Summary

| Family | Description | Mean AUC | Mean Cohen's d | Features |
|--------|-------------|----------|----------------|----------|
| D | Corrected cross-band | 0.702 | 0.647 | 3 |
| A | Pairwise log-ratios | 0.657 | 0.540 | 6 |
| C | CLR (Aitchison) | 0.655 | 0.523 | 4 |
| E | ILR (orthonormal) | 0.644 | 0.497 | 3 |
| Existing | Original ratio/frac features | 0.646 | 0.271 | 16 |
| B | ALR | 0.594 | 0.360 | 4 |

**Key takeaway**: The new compositional features achieve ~2x the mean Cohen's d of existing features at comparable AUC. The B2 and B4 bands consistently emerge as the most discriminative for ON/OFF separation, regardless of the transformation used.
