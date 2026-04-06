# Running State Detection — Reduced Handoff

## What This Is

Unsupervised clustering to detect generator ON/OFF state from HFCT EMI spectrums. Per-site only — each site's model sees only its own data.

## Results Summary

### Reference Sites (labeled, per-site unsupervised)

| Site | Samples | Best Method | F1 |
|------|---------|-------------|-----|
| Vandolah | 10 (5 ON / 5 OFF) | KMeans + Band Powers | **1.000** |
| Harquahala | 12 (6 ON / 6 OFF) | DBSCAN + Band Powers | **0.909** |

### AEP Turk (2 unlabeled spectrums)

With only N=2, clustering is trivial. Validation uses **feature-direction consistency**: does the sign of (S1 − S2) match the sign of (ON − OFF) at each reference site?

| Hypothesis | Harquahala | Vandolah | Average |
|-----------|-----------|---------|---------|
| **A: S1=ON, S2=OFF** | 69% (11/16) | **100%** (16/16) | **84%** |
| B: S1=OFF, S2=ON | 31% | 0% | 16% |

**Verdict: S1 = ON, S2 = OFF** (84% confidence). Key evidence:
- S1 total power is **265× higher** than S2 (54.9M µV² vs 207K µV²)
- S1 concentrates 98% of power in B1 (low-frequency) — matching ON pattern at both reference sites
- S2 has more energy distributed across upper bands — matching OFF pattern

### Power Computation

All spectrum values are in **µV (microvolts)**. Power features use **V²** (physically correct). This affects absolute magnitudes but not discrimination direction.

## Files in This Package

| File | What it is |
|------|-----------|
| `REDUCED_HANDOFF.md` | This document |
| `_validate_aep_turk.py` | AEP Turk validation script (V² corrected, generates all results + HTML report) |
| `_run_unsupervised_v3.py` | Full unsupervised pipeline for labeled sites (production candidate) |
| `UC-Unsupervised-OnOff-Clustering-v3.ipynb` | Development notebook for the v3 pipeline |
| `_aep_turk_validation_report.html` | Standalone HTML report with all AEP Turk results, tables, and figures |
| `Unsupervised_Clustering_Report_v3.html` | Standalone HTML report for the full multi-site clustering analysis |

### Data (in `On_Off_data/`)

- 10 Vandolah `.zip` files (Van1_Gen_{on|off}_*.zip)
- 12 Harquahala `.zip` files (Harq{CT|ST}{1-3}_Gen_{on|off}_*.zip)
- 2 AEP Turk `.csv` files (AEP_Turk_Gen_Spectrum_{1,2}.csv)

Each zip contains `ChartData.xlsx` (freq Hz, amplitude µV). CSVs have 2 header rows then freq,µV pairs.

## How to Re-run

```bash
# Full labeled-sites clustering analysis
python _run_unsupervised_v3.py

# AEP Turk per-site validation (V² corrected)
python _validate_aep_turk.py
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `openpyxl`.

## What's Next

- **Apply V² correction to `_run_unsupervised_v3.py`** — only `_validate_aep_turk.py` uses V² currently.
- **Get ground truth for AEP Turk** — operator confirmation of ON/OFF for the 2 spectrums.
- **Collect more AEP Turk data** — N≥10 enables true per-site unsupervised clustering.
