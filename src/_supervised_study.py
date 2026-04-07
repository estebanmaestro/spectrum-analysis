"""
Supervised Feature Importance and Classification Study
=======================================================
Uses manually-reviewed ON/OFF labels as ground truth to:
  1. Identify the most discriminative features (univariate + model-based)
  2. Train regularized classifiers with site-aware cross-validation
  3. Select the optimal feature subset via consensus ranking

Inputs:
  - _results_bundle.pkl        (feature matrix from the clustering pipeline)
  - running_state_labels_*_reviewed.csv  (manually verified ON/OFF labels)

Outputs:
  - Console report
  - Figures in _supervised_figs/
  - _supervised_study_bundle.pkl

Usage:
    python src/_supervised_study.py
"""
import os, pickle, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, pointbiserialr
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import (StratifiedGroupKFold, cross_validate,
                                     GridSearchCV)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score, make_scorer)
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(ROOT, '_results_bundle.pkl')
FIG_DIR = os.path.join(ROOT, '_supervised_figs')
os.makedirs(FIG_DIR, exist_ok=True)

LABELS_PATTERN = os.path.join(ROOT, 'running_state_labels_*_reviewed.csv')

SCALE_INVARIANT_KEYWORDS = [
    'frac', 'ratio', 'hi_lo', 'spectral', 'entropy', 'flatness', 'slope',
    'rolloff', 'crest', 'peak_to_mean', 'peak_freq', 'gradient', 'range',
    'amplitude_cv',
]

# ═══════════════════════════════════════════════════════════════
# [1/5] DATA LOADING AND JOIN
# ═══════════════════════════════════════════════════════════════
print("=" * 80)
print("  Supervised Feature Importance and Classification Study")
print("=" * 80)

print("\n[1/5] Loading data and joining labels...")

with open(BUNDLE_PATH, 'rb') as f:
    bundle = pickle.load(f)

data = bundle['data']
meta_cols = ['spectrum_id', 'asset', 'state', 'site', 'running']
feature_cols = [c for c in data.columns if c not in meta_cols]
scale_invariant_cols = [c for c in feature_cols
                        if any(k in c for k in SCALE_INVARIANT_KEYWORDS)]

labels_files = sorted(glob.glob(LABELS_PATTERN))
if not labels_files:
    raise FileNotFoundError(f"No reviewed labels file matching {LABELS_PATTERN}")
labels_path = labels_files[-1]
print(f"  Bundle: {len(data)} spectrums, {len(feature_cols)} features")
print(f"  Labels: {labels_path}")

labels = pd.read_csv(labels_path)
labels['y'] = (labels['running_state'] == 'ON').astype(int)

merged = data.merge(labels[['site', 'spectrum_id', 'y', 'source']],
                    on=['site', 'spectrum_id'], how='inner')

X = merged[feature_cols].values.astype(float)
y = merged['y'].values
sites = merged['site'].values
spectrum_ids = merged['spectrum_id'].values

n_on = int(y.sum())
n_off = len(y) - n_on
n_sites = len(np.unique(sites))
print(f"  Joined: {len(merged)} labeled samples ({n_on} ON / {n_off} OFF)")
print(f"  Sites: {n_sites}")
print(f"  Manual overrides: {(merged['source'] == 'manual').sum()}")

unmatched_labels = len(labels) - len(merged)
if unmatched_labels > 0:
    print(f"  WARNING: {unmatched_labels} reviewed labels could not be matched")

# ═══════════════════════════════════════════════════════════════
# [2/5] UNIVARIATE FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n[2/5] Univariate feature analysis...")

scaler = RobustScaler()
Xs = scaler.fit_transform(X)

univar_rows = []
for j, feat in enumerate(feature_cols):
    vals_on = Xs[y == 1, j]
    vals_off = Xs[y == 0, j]

    stat, p_val = mannwhitneyu(vals_on, vals_off, alternative='two-sided')

    pooled_std = np.sqrt((vals_on.std()**2 + vals_off.std()**2) / 2)
    d = abs(vals_on.mean() - vals_off.mean()) / (pooled_std + 1e-10)

    try:
        auc = roc_auc_score(y, Xs[:, j])
        auc = max(auc, 1 - auc)
    except ValueError:
        auc = 0.5

    r, r_p = pointbiserialr(y, Xs[:, j])

    is_si = feat in scale_invariant_cols
    univar_rows.append({
        'feature': feat,
        'roc_auc': round(auc, 4),
        'cohens_d': round(d, 3),
        'mw_p': p_val,
        'pb_corr': round(abs(r), 4),
        'scale_invariant': is_si,
    })

univar_df = pd.DataFrame(univar_rows).sort_values('roc_auc', ascending=False)
univar_df = univar_df.reset_index(drop=True)
univar_df['rank'] = range(1, len(univar_df) + 1)

print("\n  TOP-20 Features by ROC-AUC:")
print(f"  {'Rank':<5} {'Feature':<25} {'AUC':>7} {'Cohen d':>9} {'MW p':>10} {'SI':>4}")
print("  " + "-" * 64)
for _, r in univar_df.head(20).iterrows():
    si = "Y" if r['scale_invariant'] else ""
    p_str = f"{r['mw_p']:.2e}" if r['mw_p'] < 0.01 else f"{r['mw_p']:.4f}"
    print(f"  {r['rank']:<5} {r['feature']:<25} {r['roc_auc']:>7.4f} "
          f"{r['cohens_d']:>9.3f} {p_str:>10} {si:>4}")

# ── Fig: Univariate ranking ──
fig, ax = plt.subplots(figsize=(10, 12))
top_n = min(30, len(univar_df))
top = univar_df.head(top_n).iloc[::-1]
colors = ['#047857' if si else '#1D4ED8'
          for si in top['scale_invariant'].values]
ax.barh(range(top_n), top['roc_auc'].values, color=colors, alpha=0.85,
        edgecolor='white')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top['feature'].values, fontsize=7)
ax.set_xlabel('ROC-AUC (single feature)')
ax.set_title('Univariate Feature Ranking — ROC-AUC', fontsize=12,
             fontweight='bold')
ax.axvline(0.5, color='gray', ls='--', alpha=0.4)
ax.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, fc='#047857', label='Scale-Invariant'),
    plt.Rectangle((0, 0), 1, 1, fc='#1D4ED8', label='Scale-Dependent'),
], fontsize=8, loc='lower right')
ax.grid(axis='x', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_univariate_ranking.png'),
            dpi=180, bbox_inches='tight')
plt.close()

# ── Fig: Correlation heatmap of top-20 features ──
top20_feats = univar_df.head(20)['feature'].tolist()
top20_idx = [feature_cols.index(f) for f in top20_feats]
corr = np.corrcoef(Xs[:, top20_idx].T)
fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(corr, xticklabels=top20_feats, yticklabels=top20_feats,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1, annot=True,
            fmt='.2f', annot_kws={'size': 5.5}, ax=ax,
            linewidths=0.3, linecolor='white')
ax.set_title('Feature Correlation — Top 20 by ROC-AUC', fontsize=12,
             fontweight='bold')
ax.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_feature_correlation.png'),
            dpi=180, bbox_inches='tight')
plt.close()

print("  Figures saved: fig_univariate_ranking.png, fig_feature_correlation.png")

# ═══════════════════════════════════════════════════════════════
# [3/5] SUPERVISED MODELS WITH SITE-AWARE CV
# ═══════════════════════════════════════════════════════════════
print("\n[3/5] Training supervised models (site-aware CV)...")

site_labels = merged['site'].values
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'roc_auc': 'roc_auc',
}


def _best_C(estimator_cls, X_s, y_s, groups, cv_obj, **extra_params):
    """Quick inner CV to pick the best C from C_GRID."""
    best_c, best_score = 1.0, -1
    for c in C_GRID:
        est = estimator_cls(C=c, max_iter=5000, class_weight='balanced',
                            **extra_params)
        try:
            scores = cross_validate(est, X_s, y_s, groups=groups, cv=cv_obj,
                                    scoring='f1', error_score=0.0)
            mean_f1 = scores['test_score'].mean()
            if mean_f1 > best_score:
                best_score = mean_f1
                best_c = c
        except Exception:
            continue
    return best_c


def _best_alpha(X_s, y_s, groups, cv_obj):
    best_a, best_score = 1.0, -1
    for a in ALPHA_GRID:
        est = RidgeClassifier(alpha=a, class_weight='balanced')
        try:
            scores = cross_validate(est, X_s, y_s, groups=groups, cv=cv_obj,
                                    scoring='f1', error_score=0.0)
            mean_f1 = scores['test_score'].mean()
            if mean_f1 > best_score:
                best_score = mean_f1
                best_a = a
        except Exception:
            continue
    return best_a


print("  Tuning hyperparameters...")
best_c_l1 = _best_C(LogisticRegression, Xs, y, site_labels, cv,
                     penalty='l1', solver='liblinear')
best_c_l2 = _best_C(LogisticRegression, Xs, y, site_labels, cv,
                     penalty='l2', solver='lbfgs')
best_c_en = _best_C(LogisticRegression, Xs, y, site_labels, cv,
                     penalty='elasticnet', solver='saga', l1_ratio=0.5)
best_c_lsvm = _best_C(SVC, Xs, y, site_labels, cv,
                       kernel='linear', probability=True)
best_c_rbf = _best_C(SVC, Xs, y, site_labels, cv,
                      kernel='rbf', probability=True)
best_alpha = _best_alpha(Xs, y, site_labels, cv)

MODELS = {
    'LogReg L1': LogisticRegression(
        penalty='l1', C=best_c_l1, solver='liblinear',
        class_weight='balanced', max_iter=5000),
    'LogReg L2': LogisticRegression(
        penalty='l2', C=best_c_l2, solver='lbfgs',
        class_weight='balanced', max_iter=5000),
    'LogReg ElasticNet': LogisticRegression(
        penalty='elasticnet', C=best_c_en, solver='saga',
        l1_ratio=0.5, class_weight='balanced', max_iter=5000),
    'Linear SVM': SVC(
        kernel='linear', C=best_c_lsvm, class_weight='balanced',
        probability=True, max_iter=10000),
    'RBF SVM': SVC(
        kernel='rbf', C=best_c_rbf, class_weight='balanced',
        probability=True, max_iter=10000),
    'Ridge Classifier': RidgeClassifier(
        alpha=best_alpha, class_weight='balanced'),
}

print("  Best hyperparameters:")
print(f"    LogReg L1        C={best_c_l1}")
print(f"    LogReg L2        C={best_c_l2}")
print(f"    LogReg EN        C={best_c_en}")
print(f"    Linear SVM       C={best_c_lsvm}")
print(f"    RBF SVM          C={best_c_rbf}")
print(f"    Ridge Classifier alpha={best_alpha}")

print("\n  Running 5-fold site-grouped CV...")
model_results = {}
for name, est in MODELS.items():
    cv_scoring = scoring.copy()
    if isinstance(est, RidgeClassifier):
        cv_scoring.pop('roc_auc', None)

    scores = cross_validate(est, Xs, y, groups=site_labels, cv=cv,
                            scoring=cv_scoring, error_score=0.0)
    result = {}
    for metric in ['accuracy', 'f1', 'precision', 'recall']:
        vals = scores[f'test_{metric}']
        result[metric] = (vals.mean(), vals.std())

    if 'test_roc_auc' in scores:
        vals = scores['test_roc_auc']
        result['roc_auc'] = (vals.mean(), vals.std())
    else:
        result['roc_auc'] = (np.nan, np.nan)

    model_results[name] = result

print(f"\n  {'Model':<22} {'Acc':>12} {'F1':>12} {'Prec':>12} "
      f"{'Recall':>12} {'AUC':>12}")
print("  " + "-" * 82)
for name, res in model_results.items():
    acc = f"{res['accuracy'][0]:.3f}±{res['accuracy'][1]:.3f}"
    f1 = f"{res['f1'][0]:.3f}±{res['f1'][1]:.3f}"
    prec = f"{res['precision'][0]:.3f}±{res['precision'][1]:.3f}"
    rec = f"{res['recall'][0]:.3f}±{res['recall'][1]:.3f}"
    auc_m, auc_s = res['roc_auc']
    auc_str = f"{auc_m:.3f}±{auc_s:.3f}" if not np.isnan(auc_m) else "N/A"
    print(f"  {name:<22} {acc:>12} {f1:>12} {prec:>12} {rec:>12} {auc_str:>12}")

# ── Fig: Model comparison ──
fig, ax = plt.subplots(figsize=(12, 6))
metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall']
x_pos = np.arange(len(MODELS))
width = 0.18
metric_colors = {'accuracy': '#1D4ED8', 'f1': '#047857',
                 'precision': '#7C3AED', 'recall': '#DC2626'}

for i, metric in enumerate(metrics_to_plot):
    means = [model_results[m][metric][0] for m in MODELS]
    stds = [model_results[m][metric][1] for m in MODELS]
    ax.bar(x_pos + i * width, means, width, yerr=stds,
           label=metric.capitalize(), color=metric_colors[metric],
           alpha=0.85, edgecolor='white', capsize=3)

ax.set_xticks(x_pos + 1.5 * width)
ax.set_xticklabels(MODELS.keys(), fontsize=8, rotation=15, ha='right')
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — Site-Aware 5-Fold CV', fontsize=12,
             fontweight='bold')
ax.legend(fontsize=8)
ax.axhline(0.9, color='gray', ls='--', alpha=0.3)
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_model_comparison.png'),
            dpi=180, bbox_inches='tight')
plt.close()
print("  Figure saved: fig_model_comparison.png")

# ═══════════════════════════════════════════════════════════════
# [4/5] FEATURE IMPORTANCE AND SELECTION
# ═══════════════════════════════════════════════════════════════
print("\n[4/5] Feature importance and selection...")

# ── 4a. L1 coefficients ──
print("  [4a] L1 coefficient analysis...")
lr_l1 = LogisticRegression(penalty='l1', C=best_c_l1, solver='liblinear',
                           class_weight='balanced', max_iter=5000)
lr_l1.fit(Xs, y)
l1_coefs = np.abs(lr_l1.coef_[0])
l1_rank = np.argsort(-l1_coefs)
l1_nonzero = [feature_cols[i] for i in range(len(feature_cols))
              if l1_coefs[i] > 1e-6]
print(f"    Non-zero features ({len(l1_nonzero)}/{len(feature_cols)}): "
      f"{', '.join(l1_nonzero[:10])}{'...' if len(l1_nonzero) > 10 else ''}")

# ── 4b. Permutation importance ──
print("  [4b] Permutation importance (best F1 model)...")
best_model_name = max(model_results,
                      key=lambda m: model_results[m]['f1'][0])
best_model = MODELS[best_model_name]
best_model.fit(Xs, y)
perm_result = permutation_importance(best_model, Xs, y, n_repeats=30,
                                     random_state=42, scoring='f1')
perm_imp = perm_result.importances_mean
perm_rank = np.argsort(-perm_imp)
print(f"    Best model for permutation: {best_model_name}")
print(f"    Top-5 by permutation: "
      f"{', '.join(feature_cols[i] for i in perm_rank[:5])}")

# ── 4c. RFECV ──
print("  [4c] Recursive Feature Elimination with CV (RFECV)...")
rfecv_est = LogisticRegression(penalty='l1', C=best_c_l1,
                               solver='liblinear',
                               class_weight='balanced', max_iter=5000)
rfecv = RFECV(estimator=rfecv_est, step=1, cv=cv, scoring='f1',
              min_features_to_select=3, n_jobs=-1)
rfecv.fit(Xs, y, groups=site_labels)
rfecv_selected = [feature_cols[i] for i in range(len(feature_cols))
                  if rfecv.support_[i]]
rfecv_scores = rfecv.cv_results_['mean_test_score']
print(f"    Optimal feature count: {rfecv.n_features_}")
print(f"    Selected features: {', '.join(rfecv_selected)}")

# ── Consensus ranking ──
rank_l1 = np.zeros(len(feature_cols))
rank_l1[l1_rank] = np.arange(len(feature_cols))

rank_perm = np.zeros(len(feature_cols))
rank_perm[perm_rank] = np.arange(len(feature_cols))

rfecv_ranking = rfecv.ranking_
rank_rfecv = rfecv_ranking - 1.0

consensus = (rank_l1 + rank_perm + rank_rfecv) / 3.0
consensus_order = np.argsort(consensus)

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'l1_coef': l1_coefs,
    'l1_rank': rank_l1.astype(int),
    'perm_imp': perm_imp,
    'perm_rank': rank_perm.astype(int),
    'rfecv_rank': rank_rfecv.astype(int),
    'consensus_rank': consensus,
    'scale_invariant': [f in scale_invariant_cols for f in feature_cols],
    'rfecv_selected': rfecv.support_,
}).sort_values('consensus_rank')
importance_df = importance_df.reset_index(drop=True)
importance_df.index = range(1, len(importance_df) + 1)

print("\n  Consensus Feature Ranking (top 20):")
print(f"  {'#':<4} {'Feature':<25} {'L1|coef|':>9} {'L1 rk':>6} "
      f"{'Perm':>7} {'Pm rk':>6} {'RFE rk':>7} {'Cons':>6} {'SI':>3} {'Sel':>4}")
print("  " + "-" * 88)
for idx, r in importance_df.head(20).iterrows():
    si = "Y" if r['scale_invariant'] else ""
    sel = "*" if r['rfecv_selected'] else ""
    print(f"  {idx:<4} {r['feature']:<25} {r['l1_coef']:>9.4f} {r['l1_rank']:>6} "
          f"{r['perm_imp']:>7.4f} {r['perm_rank']:>6} {r['rfecv_rank']:>7} "
          f"{r['consensus_rank']:>6.1f} {si:>3} {sel:>4}")

# ── Fig: Feature importance bar chart ──
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
top15 = importance_df.head(15)

for ax, (col, title) in zip(axes, [
    ('l1_coef', 'L1 Coefficient (abs)'),
    ('perm_imp', 'Permutation Importance'),
    ('consensus_rank', 'Consensus Rank (lower=better)'),
]):
    vals = top15[col].values
    names = top15['feature'].values
    si_flags = top15['scale_invariant'].values
    if col == 'consensus_rank':
        order = np.argsort(vals)
        vals = vals[order]
        names = names[order]
        si_flags = si_flags[order]
    else:
        order = np.argsort(-vals)
        vals = vals[order]
        names = names[order]
        si_flags = si_flags[order]
    colors = ['#047857' if s else '#1D4ED8' for s in si_flags]
    ax.barh(range(len(vals)), vals, color=colors, alpha=0.85,
            edgecolor='white')
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)
    ax.invert_yaxis()

fig.suptitle('Feature Importance — Top 15 by Consensus', fontsize=12,
             fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(FIG_DIR, 'fig_feature_importance.png'),
            dpi=180, bbox_inches='tight')
plt.close()

# ── Fig: RFECV curve ──
fig, ax = plt.subplots(figsize=(10, 5))
n_features_range = range(rfecv.min_features_to_select,
                         len(feature_cols) + 1)
ax.plot(n_features_range, rfecv_scores, marker='o', markersize=3,
        color='#1D4ED8', linewidth=1.5)
ax.axvline(rfecv.n_features_, color='#DC2626', ls='--', alpha=0.7,
           label=f'Optimal: {rfecv.n_features_} features')
ax.fill_between(n_features_range,
                rfecv_scores - rfecv.cv_results_['std_test_score'],
                rfecv_scores + rfecv.cv_results_['std_test_score'],
                alpha=0.15, color='#1D4ED8')
ax.set_xlabel('Number of Features')
ax.set_ylabel('Cross-Validated F1')
ax.set_title('RFECV — CV Score vs Number of Features', fontsize=12,
             fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_rfecv_curve.png'),
            dpi=180, bbox_inches='tight')
plt.close()
print("  Figures saved: fig_feature_importance.png, fig_rfecv_curve.png")

# ═══════════════════════════════════════════════════════════════
# [5/5] SUMMARY AND OUTPUT
# ═══════════════════════════════════════════════════════════════
print("\n[5/5] Summary and final evaluation...")

# Re-train best model on RFECV-selected features
rfecv_mask = rfecv.support_
Xs_sel = Xs[:, rfecv_mask]

sel_results = {}
for name, est in MODELS.items():
    cv_scoring_sel = scoring.copy()
    if isinstance(est, RidgeClassifier):
        cv_scoring_sel.pop('roc_auc', None)
    scores = cross_validate(est, Xs_sel, y, groups=site_labels, cv=cv,
                            scoring=cv_scoring_sel, error_score=0.0)
    result = {}
    for metric in ['accuracy', 'f1', 'precision', 'recall']:
        vals = scores[f'test_{metric}']
        result[metric] = (vals.mean(), vals.std())
    if 'test_roc_auc' in scores:
        vals = scores['test_roc_auc']
        result['roc_auc'] = (vals.mean(), vals.std())
    else:
        result['roc_auc'] = (np.nan, np.nan)
    sel_results[name] = result

# Scale-invariant only
si_mask = np.array([f in scale_invariant_cols for f in feature_cols])
Xs_si = Xs[:, si_mask]

si_results = {}
for name, est in MODELS.items():
    cv_scoring_si = scoring.copy()
    if isinstance(est, RidgeClassifier):
        cv_scoring_si.pop('roc_auc', None)
    scores = cross_validate(est, Xs_si, y, groups=site_labels, cv=cv,
                            scoring=cv_scoring_si, error_score=0.0)
    result = {}
    for metric in ['accuracy', 'f1', 'precision', 'recall']:
        vals = scores[f'test_{metric}']
        result[metric] = (vals.mean(), vals.std())
    if 'test_roc_auc' in scores:
        vals = scores['test_roc_auc']
        result['roc_auc'] = (vals.mean(), vals.std())
    else:
        result['roc_auc'] = (np.nan, np.nan)
    si_results[name] = result

print("\n" + "=" * 80)
print("  SUMMARY")
print("=" * 80)

best_f1_all = max(model_results, key=lambda m: model_results[m]['f1'][0])
best_f1_sel = max(sel_results, key=lambda m: sel_results[m]['f1'][0])
best_f1_si = max(si_results, key=lambda m: si_results[m]['f1'][0])

print(f"\n  Best model (all {len(feature_cols)} features):  "
      f"{best_f1_all}  F1={model_results[best_f1_all]['f1'][0]:.3f}")
print(f"  Best model (RFECV {rfecv.n_features_} features): "
      f"{best_f1_sel}  F1={sel_results[best_f1_sel]['f1'][0]:.3f}")
print(f"  Best model (scale-invariant {si_mask.sum()} features): "
      f"{best_f1_si}  F1={si_results[best_f1_si]['f1'][0]:.3f}")

print(f"\n  RFECV-selected features ({rfecv.n_features_}):")
for feat in rfecv_selected:
    si = " (SI)" if feat in scale_invariant_cols else ""
    print(f"    - {feat}{si}")

# ── Fig: Feature subset comparison ──
fig, ax = plt.subplots(figsize=(12, 6))
subset_names = [f'All ({len(feature_cols)})',
                f'RFECV ({rfecv.n_features_})',
                f'Scale-Inv ({si_mask.sum()})']
all_result_sets = [model_results, sel_results, si_results]
x_pos = np.arange(len(MODELS))
width = 0.25
subset_colors = ['#1D4ED8', '#047857', '#7C3AED']

for si_idx, (sname, rset) in enumerate(zip(subset_names, all_result_sets)):
    f1_means = [rset[m]['f1'][0] for m in MODELS]
    f1_stds = [rset[m]['f1'][1] for m in MODELS]
    ax.bar(x_pos + si_idx * width, f1_means, width, yerr=f1_stds,
           label=sname, color=subset_colors[si_idx], alpha=0.85,
           edgecolor='white', capsize=3)

ax.set_xticks(x_pos + width)
ax.set_xticklabels(MODELS.keys(), fontsize=8, rotation=15, ha='right')
ax.set_ylim(0, 1.1)
ax.set_ylabel('F1 Score')
ax.set_title('Feature Subset Comparison — F1 (Site-Aware CV)', fontsize=12,
             fontweight='bold')
ax.legend(fontsize=9)
ax.axhline(0.9, color='gray', ls='--', alpha=0.3)
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_subset_comparison.png'),
            dpi=180, bbox_inches='tight')
plt.close()
print("  Figure saved: fig_subset_comparison.png")

# Save bundle
study_bundle = {
    'merged_data': merged,
    'feature_cols': feature_cols,
    'scale_invariant_cols': scale_invariant_cols,
    'univar_df': univar_df,
    'model_results_all': model_results,
    'model_results_selected': sel_results,
    'model_results_si': si_results,
    'importance_df': importance_df,
    'rfecv_selected': rfecv_selected,
    'rfecv_n_features': rfecv.n_features_,
    'rfecv_scores': rfecv_scores,
    'best_hyperparams': {
        'c_l1': best_c_l1, 'c_l2': best_c_l2, 'c_en': best_c_en,
        'c_lsvm': best_c_lsvm, 'c_rbf': best_c_rbf, 'alpha': best_alpha,
    },
}
bundle_out = os.path.join(ROOT, '_supervised_study_bundle.pkl')
with open(bundle_out, 'wb') as f:
    pickle.dump(study_bundle, f)
print(f"  Bundle saved: {bundle_out}")

print(f"\n  All figures saved to: {FIG_DIR}/")
print("\n  Done.")
