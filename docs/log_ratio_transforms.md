# Log-Ratio Transforms — LR, ALR, CLR

**Context**: Band powers (B1, B2, B3, B4) are compositional data — parts of a whole (total_power). Standard linear models assume unconstrained Euclidean space, but compositions live on a simplex. Log-ratio transforms project them into proper Euclidean space while removing absolute power scale dependence.

**Input variables**:

- `B1_power` = total power in 30–300 kHz
- `B2_power` = total power in 300 kHz – 3 MHz
- `B3_power` = total power in 3–30 MHz
- `B4_power` = total power in 30–100 MHz
- `total_power` = power across all frequency bins
- `eps = 10⁻¹⁰` (added before log to avoid log(0))

---

## LR — Pairwise Log-Ratios

The natural logarithm of the ratio between two bands' powers.

### Calculation

```
log_Bx = log(Bx_power + eps)

lr_Bi_Bj = log_Bi - log_Bj = log(Bi / Bj)
```

### All 6 features

| Feature | Calculation | Equivalent to |
|---------|-------------|---------------|
| `lr_B1_B2` | `log(B1) - log(B2)` | `log(B1 / B2)` |
| `lr_B1_B3` | `log(B1) - log(B3)` | `log(B1 / B3)` |
| `lr_B1_B4` | `log(B1) - log(B4)` | `log(B1 / B4)` |
| `lr_B2_B3` | `log(B2) - log(B3)` | `log(B2 / B3)` |
| `lr_B2_B4` | `log(B2) - log(B4)` | `log(B2 / B4)` |
| `lr_B3_B4` | `log(B3) - log(B4)` | `log(B3 / B4)` |

### Interpretation

- Value of **0** means equal energy in both bands.
- **Positive** means the first band has more energy; **negative** means the second does.
- **Symmetric**: `lr_B1_B2 = -lr_B2_B1`, unlike the existing `log(1 + B2/B1)` which compresses asymmetrically.

### Example

If B2 has 1000x more power than B4: `lr_B2_B4 = log(1000) = 6.9`.
If they are equal: `lr_B2_B4 = 0`.
If B4 has 10x more: `lr_B2_B4 = log(0.1) = -2.3`.

---

## ALR — Additive Log-Ratios

The logarithm of the ratio of total power to a single band's power. Measures how much of the total energy is **outside** that band.

### Calculation

```
log_total = log(total_power + eps)
log_Bx    = log(Bx_power + eps)

alr_Bx = log_total - log_Bx = log(total / Bx) = -log(Bx_frac)
```

### All 4 features

| Feature | Calculation | Equivalent to |
|---------|-------------|---------------|
| `alr_B1` | `log(total) - log(B1)` | `-log(B1_frac)` |
| `alr_B2` | `log(total) - log(B2)` | `-log(B2_frac)` |
| `alr_B3` | `log(total) - log(B3)` | `-log(B3_frac)` |
| `alr_B4` | `log(total) - log(B4)` | `-log(B4_frac)` |

### Interpretation

- **Small** ALR values mean the band dominates total power.
- **Large** ALR values mean the band contributes very little.
- ALR stretches out differences at small fractions: 0.1% vs 1% becomes as visible as 10% vs 100%.

### Example

If B4 contains 1% of total power (B4_frac = 0.01): `alr_B4 = -log(0.01) = 4.6`.
If B2 contains 60% (B2_frac = 0.6): `alr_B2 = -log(0.6) = 0.51`.

### Why ALR differs from raw band fractions

The difference between B4_frac = 0.01 and B4_frac = 0.001 is only 0.009 in linear space — nearly invisible to a linear classifier. In ALR space, it becomes 4.6 vs 6.9 — a gap of 2.3, easily separable. This explains why `alr_B4` has Cohen's d = 0.779 while `B4_frac` has d = 0.246 at identical AUC.

---

## CLR — Centered Log-Ratios

Each band's log-power, centered by subtracting the geometric mean of all four bands' log-powers. The standard Aitchison (1986) transformation for compositional data.

### Calculation

```
log_Bx = log(Bx_power + eps)   for each band

geo_mean = (log_B1 + log_B2 + log_B3 + log_B4) / 4

clr_Bx = log_Bx - geo_mean
```

### All 4 features

| Feature | Calculation |
|---------|-------------|
| `clr_B1` | `log(B1) - geo_mean` |
| `clr_B2` | `log(B2) - geo_mean` |
| `clr_B3` | `log(B3) - geo_mean` |
| `clr_B4` | `log(B4) - geo_mean` |

### Mathematical properties

1. **Sum to zero**: `clr_B1 + clr_B2 + clr_B3 + clr_B4 = 0` always. One is fully determined by the other three.
2. **Scale-invariant**: Multiplying all band powers by a constant k (e.g., different sensor gain) shifts the geometric mean by log(k), which cancels out. CLR values are unchanged.
3. **Reconstructs any LR**: `lr_Bi_Bj = clr_Bi - clr_Bj`. Any pairwise log-ratio is a simple difference of CLR coordinates.

### Interpretation

A CLR value tells how that band compares to the average band (geometric mean in log-space):

- `clr_Bx > 0`: band carries **more** power than the geometric mean
- `clr_Bx < 0`: band carries **less** power than the geometric mean
- `clr_Bx = 0`: band is exactly at the average level

### Worked example

Band powers: B1 = 100, B2 = 10000, B3 = 500, B4 = 10.

```
geo_mean = (log(100) + log(10000) + log(500) + log(10)) / 4
         = (4.6 + 9.2 + 6.2 + 2.3) / 4
         = 5.58
```

| Band | log(Bx) | CLR | Meaning |
|------|---------|-----|---------|
| B1 | 4.6 | -0.98 | Below average |
| B2 | 9.2 | +3.63 | Well above average |
| B3 | 6.2 | +0.63 | Slightly above |
| B4 | 2.3 | -3.28 | Well below average |

Sum check: -0.98 + 3.63 + 0.63 + (-3.28) = 0.0

---

## Relationship Between the Three Transforms

```
LR  (pairwise)  = difference of any two log(Bx)
ALR (additive)  = log(total) minus one log(Bx)     — uses total as reference
CLR (centered)  = log(Bx) minus geometric mean      — uses geometric mean as reference
```

- Any **LR** can be computed from **CLR**: `lr_Bi_Bj = clr_Bi - clr_Bj`
- **ALR** uses an external reference (`total_power`), while **CLR** uses an internal reference (geometric mean of the 4 bands only)
- **CLR** is the most general: given 4 CLR values, you can reconstruct all 6 LRs and all 4 ALRs

---

## Statistical Performance vs Existing Features

| Transform | Top Feature | AUC | Cohen's d | Existing Equivalent | Existing d | d improvement |
|-----------|-------------|-----|-----------|--------------------|-----------:|:-------------:|
| LR | `lr_B2_B4` | 0.752 | 0.893 | `B4_B2_ratio` | 0.167 | 5.3x |
| LR | `lr_B1_B2` | 0.580 | 0.287 | `B2_B1_ratio` | 0.022 | 13x |
| ALR | `alr_B4` | 0.701 | 0.779 | `B4_frac` | 0.246 | 3.2x |
| ALR | `alr_B2` | 0.572 | 0.247 | `B2_frac` | 0.073 | 3.4x |
| CLR | `clr_B2` | 0.746 | 0.820 | — | — | — |
| CLR | `clr_B4` | 0.746 | 0.887 | — | — | — |

ROC-AUC is identical between new and existing paired features (log is monotonic, so rank order is preserved). However, Cohen's d is 2–13x larger, meaning ON/OFF class distributions are much better separated in log-space — directly benefiting linear classifiers.
