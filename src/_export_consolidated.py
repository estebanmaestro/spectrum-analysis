"""
Export consolidated analysis CSV
================================
One row per labeled spectrum, containing:
  - Metadata: site, spectrum_id, org, true label, label source
  - All 52 computed features
  - CV fold assignment
  - Out-of-fold predictions and probabilities from every model x feature-set
  - Univariate stats per feature (appended as header-level metadata)

Usage:
    python src/_export_consolidated.py
"""
import os, pickle, glob, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load data ──
with open(os.path.join(ROOT, '_results_bundle.pkl'), 'rb') as f:
    bundle = pickle.load(f)

with open(os.path.join(ROOT, '_supervised_study_bundle.pkl'), 'rb') as f:
    study = pickle.load(f)

data = bundle['data']
meta_cols = ['spectrum_id', 'asset', 'state', 'site', 'running']
feature_cols = [c for c in data.columns if c not in meta_cols]

labels_path = sorted(glob.glob(os.path.join(ROOT, 'running_state_labels_*_reviewed.csv')))[-1]
labels = pd.read_csv(labels_path)
labels['y'] = (labels['running_state'] == 'ON').astype(int)

merged = data.merge(labels[['site', 'spectrum_id', 'y', 'source']],
                    on=['site', 'spectrum_id'], how='inner')

X = merged[feature_cols].values.astype(float)
y = merged['y'].values
sites = merged['site'].values

scaler = RobustScaler()
Xs = scaler.fit_transform(X)

hp = study['best_hyperparams']
rfecv_selected = study['rfecv_selected']
scale_inv_cols = study['scale_invariant_cols']

rfecv_mask = np.array([f in rfecv_selected for f in feature_cols])
si_mask = np.array([f in scale_inv_cols for f in feature_cols])

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# ── Assign CV fold per sample ──
fold_ids = np.full(len(y), -1, dtype=int)
for fold_i, (_, test_idx) in enumerate(cv.split(Xs, y, groups=sites)):
    fold_ids[test_idx] = fold_i

# ── Model definitions (same as study) ──
MODELS = {
    'LogReg_L1': LogisticRegression(
        penalty='l1', C=hp['c_l1'], solver='liblinear',
        class_weight='balanced', max_iter=5000),
    'LogReg_L2': LogisticRegression(
        penalty='l2', C=hp['c_l2'], solver='lbfgs',
        class_weight='balanced', max_iter=5000),
    'LogReg_EN': LogisticRegression(
        penalty='elasticnet', C=hp['c_en'], solver='saga',
        l1_ratio=0.5, class_weight='balanced', max_iter=5000),
    'LinearSVM': SVC(
        kernel='linear', C=hp['c_lsvm'], class_weight='balanced',
        probability=True, max_iter=10000),
    'RBF_SVM': SVC(
        kernel='rbf', C=hp['c_rbf'], class_weight='balanced',
        probability=True, max_iter=10000),
    'Ridge': RidgeClassifier(
        alpha=hp['alpha'], class_weight='balanced'),
}

FEATURE_SETS = {
    'all': (Xs, 'All 52 features'),
    'rfecv': (Xs[:, rfecv_mask], f'RFECV {rfecv_mask.sum()} features'),
    'si': (Xs[:, si_mask], f'Scale-invariant {si_mask.sum()} features'),
}

# ── Build output DataFrame ──
out = pd.DataFrame()
out['site'] = merged['site'].values
out['spectrum_id'] = merged['spectrum_id'].values
out['org'] = [s[:3] for s in merged['site'].values]
out['true_label'] = merged['y'].map({1: 'ON', 0: 'OFF'}).values
out['label_source'] = merged['source'].values
out['cv_fold'] = fold_ids

for col in feature_cols:
    out[col] = merged[col].values

# ── Cross-validated predictions for each model x feature set ──
print("Generating out-of-fold predictions...")
for fs_key, (X_fs, fs_desc) in FEATURE_SETS.items():
    print(f"  Feature set: {fs_desc}")
    for model_name, est in MODELS.items():
        col_prefix = f"{model_name}__{fs_key}"

        pred = cross_val_predict(est, X_fs, y, groups=sites, cv=cv, method='predict')
        out[f'{col_prefix}__pred'] = np.where(pred == 1, 'ON', 'OFF')
        out[f'{col_prefix}__correct'] = (pred == y).astype(int)

        if hasattr(est, 'predict_proba'):
            proba = cross_val_predict(est, X_fs, y, groups=sites, cv=cv,
                                      method='predict_proba')
            out[f'{col_prefix}__prob_ON'] = np.round(proba[:, 1], 4)
        elif hasattr(est, 'decision_function'):
            dec = cross_val_predict(est, X_fs, y, groups=sites, cv=cv,
                                    method='decision_function')
            out[f'{col_prefix}__decision'] = np.round(dec, 4)

        print(f"    {model_name}: acc={np.mean(pred == y):.3f}")

# ── Add univariate stats as columns per feature ──
univar = study['univar_df'].set_index('feature')
importance = study['importance_df'].set_index('feature')

feat_meta_rows = []
for feat in feature_cols:
    row = {'feature': feat}
    if feat in univar.index:
        row['univar_roc_auc'] = univar.loc[feat, 'roc_auc']
        row['univar_cohens_d'] = univar.loc[feat, 'cohens_d']
        row['univar_mw_p'] = univar.loc[feat, 'mw_p']
        row['univar_pb_corr'] = univar.loc[feat, 'pb_corr']
    if feat in importance.index:
        row['imp_l1_coef'] = importance.loc[feat, 'l1_coef']
        row['imp_perm'] = importance.loc[feat, 'perm_imp']
        row['imp_consensus_rank'] = importance.loc[feat, 'consensus_rank']
        row['imp_rfecv_selected'] = importance.loc[feat, 'rfecv_selected']
    row['is_scale_invariant'] = feat in scale_inv_cols
    feat_meta_rows.append(row)

feat_meta_df = pd.DataFrame(feat_meta_rows)

# ── Save ──
out_path = os.path.join(ROOT, 'consolidated_analysis.csv')
out.to_csv(out_path, index=False)
print(f"\nSpectrum-level results: {out_path}")
print(f"  {len(out)} rows x {len(out.columns)} columns")

feat_meta_path = os.path.join(ROOT, 'consolidated_feature_metadata.csv')
feat_meta_df.to_csv(feat_meta_path, index=False)
print(f"\nFeature metadata: {feat_meta_path}")
print(f"  {len(feat_meta_df)} features x {len(feat_meta_df.columns)} columns")

pred_cols = [c for c in out.columns if c.endswith('__pred')]
print(f"\nPrediction columns ({len(pred_cols)}):")
for c in pred_cols:
    acc = (out[c] == out['true_label']).mean()
    print(f"  {c}: acc={acc:.3f}")

print("\nDone.")
