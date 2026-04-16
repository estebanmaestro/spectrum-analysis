# Reduced ON/OFF — EMI Running-State Detection

Detects generator **ON/OFF running state** from HFCT EMI spectrums using a weighted 5-feature scoring system and unsupervised clustering.

## Quick start — using the scorer

```python
from on_off_scorer import OnOffScorer

scorer = OnOffScorer()                          # global medians, threshold 0.35
result = scorer.score(freqs_hz, amplitudes_uv)  # numpy arrays, same length

result['on_status']       # True (ON) / False (OFF)
result['score']           # float in [0, 1] — Running State Index
result['is_flat_line']    # True if spectrum is flagged as flat-line
result['votes']           # per-feature binary votes
result['features']        # per-feature computed values
result['warnings']        # [] if clean, otherwise validation issues
```

**Inputs** must be NumPy arrays of the same length:

- `freqs_hz` — frequency values in Hz (expected: 8 000 bins, 30 kHz – 100 MHz)
- `amplitudes_uv` — amplitude values in µV

The scorer validates that the input spectrum matches the training format (8 000-bin grid, 30 kHz – 100 MHz). Mismatches produce warnings because the default medians — especially `spectral_spread` — were calibrated on that exact grid.

### Site-specific medians

When enough labeled data is available for a site (≥ 10 spectrums per class), override the global medians with site-specific values:

```python
scorer = OnOffScorer(
    medians={
        'ssn_B1': -0.08,
        'lr_B2_B4': 2.1,
        'spectral_spread': 690.0,
        'rolloff_norm_B3': 0.55,
        'curvature_B2': 0.18,
    },
    threshold=0.40,
)
```

### Batch scoring

```python
spectrums = [(freqs1, amps1), (freqs2, amps2), ...]
results = scorer.score_batch(spectrums)
```

## What's in this repo

### Source (`src/`)

| File | Description |
|------|-------------|
| `on_off_scorer.py` | Production scorer — weighted 5-feature voting with input validation |
| `_supervised_study.py` | Supervised feature selection study (272 spectrums, 52 + 20 features) |
| `_log_metric_analysis.py` | Compositional log-ratio feature engineering (LR, ALR, CLR, ILR families) |
| `_export_significant_features.py` | Exports the statistically significant features with association stats |
| `_export_consolidated.py` | Consolidates feature metadata and analysis results |
| `_validate_new_sites.py` | Validation pipeline for newly added sites |
| `_supervised_dashboard.py` | Interactive Dash dashboard for supervised study exploration |
| `_dashboard.py` | Interactive Dash dashboard for unsupervised clustering results |

### Analysis scripts (root)

| File | Description |
|------|-------------|
| `_run_unsupervised_v3.py` | Full unsupervised clustering pipeline for labeled sites (multi-method) |
| `_validate_aep_turk.py` | AEP Turk per-site validation (N=2 spectrum directional analysis) |

### Documentation (`docs/`)

| File | Description |
|------|-------------|
| `On-Off integration requirements.md` | When to use the weighted scorer vs clustering, scenario playbook by data size/balance |
| `feature_dictionary.md` | All 72 features (52 original + 20 compositional log-metrics) with formulas, AUC, Cohen's d |
| `improved_scoring_results.md` | Scoring system results — comparison of feature subsets and threshold optimisation |
| `log_ratio_transforms.md` | Mathematical reference for LR, ALR, CLR, ILR transforms on band powers |

### Data (`data/`)

Intermediate CSV outputs: feature statistics, scoring parameters, labeled results, and consolidated analysis. Not raw spectrums — those live in `On_Off_data/`.

### Raw spectrums (`On_Off_data/`)

Instrument exports (`.zip` with `ChartData.xlsx`, or `.csv`) for Vandolah, Harquahala, AEP Turk, and newer sites.

## Dependencies

```
numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl, plotly, dash
```

Install with `uv sync` or `pip install -r` from `pyproject.toml`.
