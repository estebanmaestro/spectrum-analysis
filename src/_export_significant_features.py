"""
Export consolidated CSV of spectrums with all statistically significant features.
Includes original features (MW p < 0.05) and new compositional log-metric
features (BH-adjusted p < 0.05).

Usage:
    python src/_export_significant_features.py
"""
import os, pickle, glob
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pointbiserialr, ks_2samp
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-10
BANDS = ['B1', 'B2', 'B3', 'B4']

print("=" * 80)
print("  Export: Spectrums with Significant Features")
print("=" * 80)

# ── Load ──
with open(os.path.join(ROOT, '_results_bundle.pkl'), 'rb') as f:
    bundle = pickle.load(f)
with open(os.path.join(ROOT, '_supervised_study_bundle.pkl'), 'rb') as f:
    study = pickle.load(f)
with open(os.path.join(ROOT, '_log_metric_analysis_bundle.pkl'), 'rb') as f:
    log_analysis = pickle.load(f)

data = bundle['data']
raw_spectrums = bundle['raw_spectrums']
labels_path = sorted(glob.glob(os.path.join(ROOT, 'running_state_labels_*_reviewed.csv')))[-1]
labels = pd.read_csv(labels_path)
labels['y'] = (labels['running_state'] == 'ON').astype(int)
merged = data.merge(labels[['site', 'spectrum_id', 'y', 'source']],
                    on=['site', 'spectrum_id'], how='inner')

meta_cols = ['spectrum_id', 'asset', 'state', 'site', 'running']
feature_cols = [c for c in data.columns if c not in meta_cols]

# ── Identify significant original features (MW p < 0.05) ──
univar = study['univar_df']
sig_original = univar[univar['mw_p'] < 0.05]['feature'].tolist()

# ── Identify significant log-metric features (BH-adj p < 0.05) ──
log_results = log_analysis['results']
sig_log_new = log_results[(log_results['mw_p_adj'] < 0.05) &
                           (log_results['is_new'])]['feature'].tolist()

print(f"\n  Significant original features: {len(sig_original)}")
print(f"  Significant new log-metric features: {len(sig_log_new)}")
print(f"  Total significant features: {len(sig_original) + len(sig_log_new)}")

# ── Compute new log-metric features for all labeled samples ──
log_B = {bn: np.log(merged[f'{bn}_power'].values.astype(float) + EPS)
         for bn in BANDS}
log_total = np.log(merged['total_power'].values.astype(float) + EPS)

new_feat_vals = {}

for i, bi in enumerate(BANDS):
    for bj in BANDS[i + 1:]:
        col = f'lr_{bi}_{bj}'
        if col in sig_log_new:
            new_feat_vals[col] = log_B[bi] - log_B[bj]

for bn in BANDS:
    col = f'alr_{bn}'
    if col in sig_log_new:
        new_feat_vals[col] = log_total - log_B[bn]

geo_mean = np.mean([log_B[bn] for bn in BANDS], axis=0)
for bn in BANDS:
    col = f'clr_{bn}'
    if col in sig_log_new:
        new_feat_vals[col] = log_B[bn] - geo_mean

log_bands = np.column_stack([log_B[bn] for bn in BANDS])
if 'band_gradient_proper' in sig_log_new:
    new_feat_vals['band_gradient_proper'] = np.mean(np.diff(log_bands, axis=1), axis=1)
if 'band_range_proper' in sig_log_new:
    new_feat_vals['band_range_proper'] = log_bands.max(axis=1) - log_bands.min(axis=1)

if 'spectral_slope_norm' in sig_log_new:
    N = len(merged)
    slope_vals = np.full(N, np.nan)
    for idx in range(N):
        key = (merged.iloc[idx]['site'], merged.iloc[idx]['spectrum_id'])
        if key in raw_spectrums:
            freqs, amps = raw_spectrums[key]
            pwr = amps.astype(float) ** 2
            normp = pwr / (pwr.sum() + EPS)
            freq_idx = np.arange(len(pwr))
            slope, _ = np.polyfit(freq_idx, np.log(normp + EPS), 1)
            slope_vals[idx] = slope
    new_feat_vals['spectral_slope_norm'] = slope_vals

if 'ilr_1' in sig_log_new:
    new_feat_vals['ilr_1'] = np.sqrt(0.5) * (log_B['B1'] - log_B['B2'])
if 'ilr_2' in sig_log_new:
    new_feat_vals['ilr_2'] = np.sqrt(2/3) * ((log_B['B1'] + log_B['B2']) / 2 - log_B['B3'])
if 'ilr_3' in sig_log_new:
    new_feat_vals['ilr_3'] = np.sqrt(3/4) * ((log_B['B1'] + log_B['B2'] + log_B['B3']) / 3 - log_B['B4'])

# ── Build output DataFrame ──
out = pd.DataFrame()

out['site'] = merged['site'].values
out['spectrum_id'] = merged['spectrum_id'].values
out['org'] = [s[:3] for s in merged['site'].values]
out['running_state'] = merged['y'].map({1: 'ON', 0: 'OFF'}).values
out['label_source'] = merged['source'].values

# Original significant features, ordered by AUC
orig_ordered = univar[univar['feature'].isin(sig_original)].sort_values(
    'roc_auc', ascending=False)['feature'].tolist()
for col in orig_ordered:
    out[col] = merged[col].values

# New log-metric significant features, ordered by AUC
log_ordered = log_results[(log_results['feature'].isin(sig_log_new))].sort_values(
    'roc_auc', ascending=False)['feature'].tolist()
for col in log_ordered:
    out[col] = new_feat_vals[col]

# ── Compute per-feature stats and append as a metadata header ──
y = merged['y'].values
stats_rows = []

for col in orig_ordered + log_ordered:
    vals = out[col].values.astype(float)
    valid = ~np.isnan(vals)
    v, yv = vals[valid], y[valid]
    v_on, v_off = v[yv == 1], v[yv == 0]

    _, mw_p = mannwhitneyu(v_on, v_off, alternative='two-sided')
    pooled_std = np.sqrt((v_on.std()**2 + v_off.std()**2) / 2)
    d = abs(v_on.mean() - v_off.mean()) / (pooled_std + EPS)
    try:
        auc = roc_auc_score(yv, v)
        auc = max(auc, 1 - auc)
    except ValueError:
        auc = 0.5
    r, _ = pointbiserialr(yv, v)
    ks_stat, ks_p = ks_2samp(v_on, v_off)

    is_new = col in sig_log_new
    family = 'compositional' if is_new else 'original'

    stats_rows.append({
        'feature': col,
        'family': family,
        'roc_auc': round(auc, 4),
        'cohens_d': round(d, 3),
        'mw_p': mw_p,
        'ks_stat': round(ks_stat, 4),
        'pb_corr': round(abs(r), 4),
        'mean_ON': round(float(v_on.mean()), 6),
        'mean_OFF': round(float(v_off.mean()), 6),
        'std_ON': round(float(v_on.std()), 6),
        'std_OFF': round(float(v_off.std()), 6),
    })

stats_df = pd.DataFrame(stats_rows)

# ── Save ──
out_path = os.path.join(ROOT, 'significant_features_dataset.csv')
out.to_csv(out_path, index=False)
print(f"\n  Spectrum dataset: {out_path}")
print(f"  {len(out)} rows x {len(out.columns)} columns")
print(f"  Columns: 5 metadata + {len(orig_ordered)} original + {len(log_ordered)} compositional")

stats_path = os.path.join(ROOT, 'significant_features_stats.csv')
stats_df.to_csv(stats_path, index=False)
print(f"\n  Feature stats: {stats_path}")
print(f"  {len(stats_df)} features x {len(stats_df.columns)} columns")

print(f"\n  Feature summary (by AUC):")
print(f"  {'#':<4} {'Feature':<28} {'Family':<14} {'AUC':>7} {'d':>7} {'MW p':>11} {'|r|':>6}")
print("  " + "-" * 80)
for i, r in stats_df.iterrows():
    p_str = f"{r['mw_p']:.2e}" if r['mw_p'] < 0.01 else f"{r['mw_p']:.4f}"
    print(f"  {i+1:<4} {r['feature']:<28} {r['family']:<14} {r['roc_auc']:>7.4f} "
          f"{r['cohens_d']:>7.3f} {p_str:>11} {r['pb_corr']:>6.4f}")

print("\n  Done.")
