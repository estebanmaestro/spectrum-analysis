"""
Compositional Log-Metric Analysis
==================================
Computes new log-family features (proper pairwise log-ratios, ALR, CLR, ILR,
corrected cross-band) from existing band powers, then runs a full battery of
statistical association tests against ON/OFF labels.

Usage:
    python src/_log_metric_analysis.py
"""
import os, pickle, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, pointbiserialr, ks_2samp
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(ROOT, '_results_bundle.pkl')
FIG_DIR = os.path.join(ROOT, '_supervised_figs')
os.makedirs(FIG_DIR, exist_ok=True)

LABELS_PATTERN = os.path.join(ROOT, 'running_state_labels_*_reviewed.csv')
BANDS = ['B1', 'B2', 'B3', 'B4']
EPS = 1e-10

PALETTE = {
    'A': '#1D4ED8', 'B': '#047857', 'C': '#7C3AED',
    'D': '#DC2626', 'E': '#B45309', 'existing': '#64748b',
}

print("=" * 80)
print("  Compositional Log-Metric Analysis")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════
# [1/5] DATA LOADING
# ═══════════════════════════════════════════════════════════════
print("\n[1/5] Loading data...")

with open(BUNDLE_PATH, 'rb') as f:
    bundle = pickle.load(f)

data = bundle['data']
raw_spectrums = bundle['raw_spectrums']

labels_path = sorted(glob.glob(LABELS_PATTERN))[-1]
labels = pd.read_csv(labels_path)
labels['y'] = (labels['running_state'] == 'ON').astype(int)

merged = data.merge(labels[['site', 'spectrum_id', 'y', 'source']],
                    on=['site', 'spectrum_id'], how='inner')

y = merged['y'].values
sites = merged['site'].values
N = len(merged)
print(f"  {N} labeled samples ({int(y.sum())} ON / {N - int(y.sum())} OFF)")

# ═══════════════════════════════════════════════════════════════
# [2/5] COMPUTE NEW LOG-FAMILY FEATURES
# ═══════════════════════════════════════════════════════════════
print("\n[2/5] Computing log-family features...")

log_B = {}
for bn in BANDS:
    log_B[bn] = np.log(merged[f'{bn}_power'].values.astype(float) + EPS)
log_total = np.log(merged['total_power'].values.astype(float) + EPS)

new_feats = pd.DataFrame(index=merged.index)
family_map = {}

# ── Family A: Proper Pairwise Log-Ratios (6 features) ──
for i, bi in enumerate(BANDS):
    for bj in BANDS[i + 1:]:
        col = f'lr_{bi}_{bj}'
        new_feats[col] = log_B[bi] - log_B[bj]
        family_map[col] = 'A'
print(f"  Family A (pairwise log-ratios): {sum(1 for v in family_map.values() if v == 'A')} features")

# ── Family B: ALR — Additive Log-Ratios (4 features) ──
for bn in BANDS:
    col = f'alr_{bn}'
    new_feats[col] = log_total - log_B[bn]
    family_map[col] = 'B'
print(f"  Family B (ALR): {sum(1 for v in family_map.values() if v == 'B')} features")

# ── Family C: CLR — Centered Log-Ratios (4 features) ──
geo_mean = np.mean([log_B[bn] for bn in BANDS], axis=0)
for bn in BANDS:
    col = f'clr_{bn}'
    new_feats[col] = log_B[bn] - geo_mean
    family_map[col] = 'C'
print(f"  Family C (CLR): {sum(1 for v in family_map.values() if v == 'C')} features")

# ── Family D: Corrected Cross-Band Features (3 features) ──
log_bands = np.column_stack([log_B[bn] for bn in BANDS])
new_feats['band_gradient_proper'] = np.mean(np.diff(log_bands, axis=1), axis=1)
family_map['band_gradient_proper'] = 'D'

new_feats['band_range_proper'] = log_bands.max(axis=1) - log_bands.min(axis=1)
family_map['band_range_proper'] = 'D'

slope_norm_vals = np.full(N, np.nan)
for idx_row in range(N):
    site = merged.iloc[idx_row]['site']
    sid = merged.iloc[idx_row]['spectrum_id']
    key = (site, sid)
    if key in raw_spectrums:
        freqs, amps = raw_spectrums[key]
        pwr = amps.astype(float) ** 2
        total = pwr.sum()
        normp = pwr / (total + EPS)
        log_normp = np.log(normp + EPS)
        freq_idx = np.arange(len(pwr))
        slope, _ = np.polyfit(freq_idx, log_normp, 1)
        slope_norm_vals[idx_row] = slope
new_feats['spectral_slope_norm'] = slope_norm_vals
family_map['spectral_slope_norm'] = 'D'
print(f"  Family D (corrected cross-band): {sum(1 for v in family_map.values() if v == 'D')} features")

# ── Family E: ILR — Isometric Log-Ratios (3 features) ──
new_feats['ilr_1'] = np.sqrt(1.0 / 2) * (log_B['B1'] - log_B['B2'])
new_feats['ilr_2'] = np.sqrt(2.0 / 3) * (
    (log_B['B1'] + log_B['B2']) / 2 - log_B['B3'])
new_feats['ilr_3'] = np.sqrt(3.0 / 4) * (
    (log_B['B1'] + log_B['B2'] + log_B['B3']) / 3 - log_B['B4'])
for c in ['ilr_1', 'ilr_2', 'ilr_3']:
    family_map[c] = 'E'
print(f"  Family E (ILR): {sum(1 for v in family_map.values() if v == 'E')} features")

new_feat_names = list(new_feats.columns)
print(f"  Total new features: {len(new_feat_names)}")

# ── Existing reference features for comparison ──
existing_ref = [
    'B1_frac', 'B2_frac', 'B3_frac', 'B4_frac',
    'log_B2_B1', 'log_B4_B1', 'log_hi_lo',
    'B2_B1_ratio', 'B3_B1_ratio', 'B4_B1_ratio', 'B4_B2_ratio',
    'B3_B2_ratio', 'hi_lo_ratio',
    'band_gradient', 'band_range', 'spectral_slope',
]
existing_ref = [c for c in existing_ref if c in merged.columns]

# ═══════════════════════════════════════════════════════════════
# [3/5] STATISTICAL ASSOCIATION TESTS
# ═══════════════════════════════════════════════════════════════
print("\n[3/5] Running statistical association tests...")

all_test_cols = new_feat_names + existing_ref
results_rows = []

for col in all_test_cols:
    if col in new_feats.columns:
        vals = new_feats[col].values.astype(float)
        fam = family_map.get(col, '?')
        is_new = True
    else:
        vals = merged[col].values.astype(float)
        fam = 'existing'
        is_new = False

    valid = ~np.isnan(vals)
    v = vals[valid]
    yv = y[valid]
    v_on = v[yv == 1]
    v_off = v[yv == 0]

    mw_stat, mw_p = mannwhitneyu(v_on, v_off, alternative='two-sided')

    pooled_std = np.sqrt((v_on.std() ** 2 + v_off.std() ** 2) / 2)
    d = abs(v_on.mean() - v_off.mean()) / (pooled_std + EPS)

    try:
        auc = roc_auc_score(yv, v)
        auc = max(auc, 1 - auc)
    except ValueError:
        auc = 0.5

    r, r_p = pointbiserialr(yv, v)
    ks_stat, ks_p = ks_2samp(v_on, v_off)

    results_rows.append({
        'feature': col, 'family': fam, 'is_new': is_new,
        'roc_auc': round(auc, 4), 'cohens_d': round(d, 3),
        'mw_p': mw_p, 'pb_corr': round(abs(r), 4),
        'ks_stat': round(ks_stat, 4), 'ks_p': ks_p,
    })

results = pd.DataFrame(results_rows)

_, mw_adj, _, _ = multipletests(results['mw_p'].values, method='fdr_bh')
results['mw_p_adj'] = mw_adj
_, ks_adj, _, _ = multipletests(results['ks_p'].values, method='fdr_bh')
results['ks_p_adj'] = ks_adj

results = results.sort_values('roc_auc', ascending=False).reset_index(drop=True)
results['rank'] = range(1, len(results) + 1)

print(f"\n  {'Rk':<4} {'Feature':<26} {'Fam':>4} {'AUC':>7} {'d':>7} "
      f"{'MW p(adj)':>11} {'KS stat':>8} {'|r|':>6}")
print("  " + "-" * 78)
for _, r in results.head(25).iterrows():
    p_str = f"{r['mw_p_adj']:.2e}" if r['mw_p_adj'] < 0.01 else f"{r['mw_p_adj']:.4f}"
    new_tag = "*" if r['is_new'] else " "
    print(f"  {r['rank']:<4} {r['feature']:<26} {r['family']:>4} {r['roc_auc']:>7.4f} "
          f"{r['cohens_d']:>7.3f} {p_str:>11} {r['ks_stat']:>8.4f} {r['pb_corr']:>6.4f}{new_tag}")

n_sig = (results['mw_p_adj'] < 0.05).sum()
print(f"\n  Features significant at FDR < 0.05: {n_sig}/{len(results)}")

# ═══════════════════════════════════════════════════════════════
# [4/5] HEAD-TO-HEAD COMPARISONS
# ═══════════════════════════════════════════════════════════════
print("\n[4/5] Head-to-head comparisons...")

h2h_pairs = [
    ('lr_B2_B1', 'log_B2_B1', 'log(B2/B1) vs log1p(B2/B1)'),
    ('lr_B4_B1', 'log_B4_B1', 'log(B4/B1) vs log1p(B4/B1)'),
    ('lr_B1_B2', 'B2_B1_ratio', 'log(B1/B2) vs B2/B1 ratio'),
    ('lr_B1_B4', 'B4_B1_ratio', 'log(B1/B4) vs B4/B1 ratio'),
    ('alr_B1', 'B1_frac', 'ALR(B1) vs B1_frac'),
    ('alr_B2', 'B2_frac', 'ALR(B2) vs B2_frac'),
    ('alr_B3', 'B3_frac', 'ALR(B3) vs B3_frac'),
    ('alr_B4', 'B4_frac', 'ALR(B4) vs B4_frac'),
    ('band_gradient_proper', 'band_gradient', 'gradient(log) vs gradient(log1p)'),
    ('band_range_proper', 'band_range', 'range(log) vs range(log1p)'),
    ('spectral_slope_norm', 'spectral_slope', 'slope(normp) vs slope(pwr)'),
]

res_idx = results.set_index('feature')
print(f"\n  {'Comparison':<38} {'New AUC':>8} {'Old AUC':>8} {'Delta':>7} "
      f"{'New d':>7} {'Old d':>7}")
print("  " + "-" * 78)
h2h_rows = []
for new_col, old_col, desc in h2h_pairs:
    if new_col not in res_idx.index or old_col not in res_idx.index:
        continue
    new_auc = res_idx.loc[new_col, 'roc_auc']
    old_auc = res_idx.loc[old_col, 'roc_auc']
    new_d = res_idx.loc[new_col, 'cohens_d']
    old_d = res_idx.loc[old_col, 'cohens_d']
    delta = new_auc - old_auc
    sign = "+" if delta >= 0 else ""
    print(f"  {desc:<38} {new_auc:>8.4f} {old_auc:>8.4f} {sign}{delta:>6.4f} "
          f"{new_d:>7.3f} {old_d:>7.3f}")
    h2h_rows.append({'new': new_col, 'old': old_col, 'desc': desc,
                      'new_auc': new_auc, 'old_auc': old_auc, 'delta_auc': delta,
                      'new_d': new_d, 'old_d': old_d})

h2h_df = pd.DataFrame(h2h_rows)

# Family-level summary
print("\n  Family-level summary:")
print(f"  {'Family':<12} {'Mean AUC':>9} {'Mean d':>8} {'Count':>6}")
print("  " + "-" * 38)
for fam in ['A', 'B', 'C', 'D', 'E', 'existing']:
    subset = results[results['family'] == fam]
    if subset.empty:
        continue
    label = f"Fam {fam}" if fam != 'existing' else "Existing"
    print(f"  {label:<12} {subset['roc_auc'].mean():>9.4f} "
          f"{subset['cohens_d'].mean():>8.3f} {len(subset):>6}")

# ═══════════════════════════════════════════════════════════════
# [5/5] FIGURES
# ═══════════════════════════════════════════════════════════════
print("\n[5/5] Generating figures...")

# ── Fig 1: All new features ranked by ROC-AUC ──
new_results = results[results['is_new']].copy()
top_n = min(20, len(new_results))
top = new_results.head(top_n).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 10))
colors = [PALETTE.get(family_map.get(f, ''), '#999') for f in top['feature']]
ax.barh(range(top_n), top['roc_auc'].values, color=colors, alpha=0.88,
        edgecolor='white')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top['feature'].values, fontsize=8)
ax.set_xlabel('ROC-AUC (single feature)')
ax.set_title('New Log-Family Features — Ranked by ROC-AUC', fontsize=12,
             fontweight='bold')
ax.axvline(0.5, color='gray', ls='--', alpha=0.4)
handles = [plt.Rectangle((0, 0), 1, 1, fc=PALETTE[k],
           label=f'Family {k}') for k in ['A', 'B', 'C', 'D', 'E']]
ax.legend(handles=handles, fontsize=8, loc='lower right')
ax.grid(axis='x', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_log_families_ranking.png'),
            dpi=180, bbox_inches='tight')
plt.close()

# ── Fig 2: Head-to-head comparison ──
if len(h2h_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(h2h_df))
    width = 0.35
    ax.bar(x - width / 2, h2h_df['new_auc'], width, label='New (proper log)',
           color=PALETTE['A'], alpha=0.88, edgecolor='white')
    ax.bar(x + width / 2, h2h_df['old_auc'], width, label='Existing (log1p / ratio)',
           color=PALETTE['existing'], alpha=0.88, edgecolor='white')
    for i, delta in enumerate(h2h_df['delta_auc']):
        sign = "+" if delta >= 0 else ""
        clr = '#047857' if delta >= 0 else '#DC2626'
        ax.text(i, max(h2h_df.iloc[i]['new_auc'], h2h_df.iloc[i]['old_auc']) + 0.01,
                f"{sign}{delta:.3f}", ha='center', fontsize=7, color=clr,
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(h2h_df['desc'], fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Head-to-Head: New Log-Metrics vs Existing Features', fontsize=12,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig_log_families_comparison.png'),
                dpi=180, bbox_inches='tight')
    plt.close()

# ── Fig 3: Correlation heatmap of all new + existing related ──
all_vals = pd.DataFrame(index=merged.index)
for col in new_feat_names:
    all_vals[col] = new_feats[col].values
for col in existing_ref:
    all_vals[col] = merged[col].values

corr = all_vals.corr()
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': 5},
            xticklabels=corr.columns, yticklabels=corr.columns,
            ax=ax, linewidths=0.3, linecolor='white')
ax.set_title('Correlation: New Log-Families + Existing Reference Features',
             fontsize=12, fontweight='bold')
ax.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_log_families_correlation.png'),
            dpi=180, bbox_inches='tight')
plt.close()

# ── Fig 4: Violin plots of top-8 new features by ON/OFF ──
top8 = new_results.head(8)['feature'].tolist()
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax_idx, feat in enumerate(top8):
    ax = axes[ax_idx // 4][ax_idx % 4]
    vals = new_feats[feat].values.astype(float)
    df_plot = pd.DataFrame({'value': vals, 'label': np.where(y == 1, 'ON', 'OFF')})
    sns.violinplot(data=df_plot, x='label', y='value', ax=ax, inner='box',
                   palette={'ON': '#DC2626', 'OFF': '#2563EB'},
                   order=['OFF', 'ON'], alpha=0.8)
    auc_val = res_idx.loc[feat, 'roc_auc'] if feat in res_idx.index else 0
    ax.set_title(f'{feat}\nAUC={auc_val:.3f}', fontsize=9, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(axis='y', alpha=0.2)
fig.suptitle('Top-8 New Log Features — ON vs OFF Distribution', fontsize=12,
             fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(FIG_DIR, 'fig_log_families_distributions.png'),
            dpi=180, bbox_inches='tight')
plt.close()

print("  Figures saved:")
for fn in ['fig_log_families_ranking.png', 'fig_log_families_comparison.png',
           'fig_log_families_correlation.png', 'fig_log_families_distributions.png']:
    print(f"    {fn}")

# ═══════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════
print("\n  Exporting results...")

# Append new features to consolidated_analysis.csv
consolidated_path = os.path.join(ROOT, 'consolidated_analysis.csv')
if os.path.exists(consolidated_path):
    cons = pd.read_csv(consolidated_path)
    for col in new_feat_names:
        key_col = f'log_{col}'
        vals = new_feats[col].values
        if len(vals) == len(cons):
            cons[key_col] = vals
    try:
        cons.to_csv(consolidated_path, index=False)
        print(f"  Updated {consolidated_path} ({len(cons.columns)} columns)")
    except PermissionError:
        alt_path = os.path.join(ROOT, 'consolidated_analysis_v2.csv')
        cons.to_csv(alt_path, index=False)
        print(f"  WARNING: {consolidated_path} locked, saved to {alt_path}")

# Save analysis bundle
analysis_bundle = {
    'new_features': new_feats,
    'family_map': family_map,
    'results': results,
    'h2h_df': h2h_df,
    'existing_ref_cols': existing_ref,
}
bundle_path = os.path.join(ROOT, '_log_metric_analysis_bundle.pkl')
with open(bundle_path, 'wb') as f:
    pickle.dump(analysis_bundle, f)
print(f"  Saved {bundle_path}")

print("\n  Done.")
