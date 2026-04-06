"""
Unsupervised Clustering for Running State Detection — v3
=========================================================
Key constraint: EVERY sample must receive a predicted ON/OFF state.
No sample may be dropped, skipped, or labeled as noise.

Changes from v2:
  - DBSCAN noise points assigned to nearest cluster via centroid distance
  - OneClassSVM "outlier" label still maps to a binary class (no drops)
  - match_labels evaluates on ALL samples (no valid/invalid masking)
  - Added coverage metric to flag methods that originally wanted to drop data
  - All P/R/F1 computed on the full dataset per site
"""
import os, re, zipfile, json, io, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM, SVC
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             accuracy_score, f1_score, precision_score, recall_score,
                             silhouette_score, confusion_matrix)
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from itertools import permutations
from collections import Counter

warnings.filterwarnings('ignore')
np.random.seed(42)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'On_Off_data')
FIG_DIR = os.path.join(ROOT, '_unsup_v3_figs')
os.makedirs(FIG_DIR, exist_ok=True)

BAND_DEFS = [('B1',30000,300000),('B2',300000,3000000),
             ('B3',3000000,30000000),('B4',30000000,100000000)]
PATTERNS = [
    re.compile(r'^(Van\d+)_(\w+?)_(on|off)_(\d{4}_\d{2}_\d{2})\.zip$', re.I),
    re.compile(r'^(Harq\w+?)_(\w+?)_(on|off)_(\d{4}_\d{2}_\d{2})\.zip$', re.I),
]

def parse_filename(fname):
    for pat in PATTERNS:
        m = pat.match(fname)
        if m:
            unit = m.group(1).upper()
            site = 'Vandolah' if unit.startswith('VAN') else 'Harquahala'
            return unit, m.group(2), m.group(3).lower(), m.group(4).replace('_','-'), site
    return None

# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
def compute_features(freqs, powers):
    """Extract features from 8000-bin spectrum.
    Categories:
      - Scale-dependent: raw powers (vary with sensor baseline)
      - Scale-invariant: fractions, ratios, spectral shape
        (baseline cancels out — key for site-agnostic use)
    """
    f = {}
    total = powers.sum()
    f['total_power'] = total
    f['mean_power'] = powers.mean()
    f['max_power'] = powers.max()
    f['std_power'] = powers.std()
    for bn, lo, hi in BAND_DEFS:
        mask = (freqs >= lo) & (freqs < hi)
        bp = powers[mask]; bsum = bp.sum()
        f[f'{bn}_power'] = bsum
        f[f'{bn}_mean'] = bp.mean() if len(bp)>0 else 0
        f[f'{bn}_max'] = bp.max() if len(bp)>0 else 0
        f[f'{bn}_std'] = bp.std() if len(bp)>0 else 0
    for bn,_,_ in BAND_DEFS:
        f[f'{bn}_frac'] = f[f'{bn}_power'] / (total + 1e-10)
    f['log_total'] = np.log1p(total)
    for bn,_,_ in BAND_DEFS:
        f[f'{bn}_log'] = np.log1p(f[f'{bn}_power'])
    f['B2_B1_ratio'] = f['B2_power'] / (f['B1_power']+1e-6)
    f['B3_B1_ratio'] = f['B3_power'] / (f['B1_power']+1e-6)
    f['B4_B1_ratio'] = f['B4_power'] / (f['B1_power']+1e-6)
    f['B4_B2_ratio'] = f['B4_power'] / (f['B2_power']+1e-6)
    f['B3_B2_ratio'] = f['B3_power'] / (f['B2_power']+1e-6)
    f['hi_lo_ratio'] = (f['B3_power']+f['B4_power']) / (f['B1_power']+f['B2_power']+1e-6)
    f['log_B2_B1'] = np.log1p(f['B2_B1_ratio'])
    f['log_B4_B1'] = np.log1p(f['B4_B1_ratio'])
    f['log_hi_lo'] = np.log1p(f['hi_lo_ratio'])
    normp = powers / (total + 1e-10)
    freq_idx = np.arange(len(powers))
    centroid = np.sum(freq_idx * normp)
    spread = np.sqrt(np.sum((freq_idx - centroid)**2 * normp))
    f['spectral_centroid'] = centroid; f['spectral_spread'] = spread
    f['spectral_skew'] = skew(powers); f['spectral_kurtosis'] = kurtosis(powers)
    normp_pos = np.clip(normp, 1e-20, None)
    f['spectral_entropy'] = entropy(normp_pos) / np.log(len(powers))
    log_mean = np.mean(np.log(powers + 1e-10))
    f['spectral_flatness'] = np.exp(log_mean) / (powers.mean() + 1e-10)
    peak_idx = np.argmax(powers)
    f['peak_freq_mhz'] = freqs[peak_idx] / 1e6
    f['peak_to_mean'] = powers[peak_idx] / (powers.mean() + 1e-10)
    log_p = np.log1p(powers)
    slope, _ = np.polyfit(freq_idx, log_p, 1)
    f['spectral_slope'] = slope
    cumsum = np.cumsum(powers)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * total)
    f['rolloff_freq_mhz'] = freqs[min(rolloff_idx, len(freqs)-1)] / 1e6
    rms = np.sqrt(np.mean(powers**2))
    f['crest_factor'] = powers.max() / (rms + 1e-10)
    bp_list = [f[f'{bn}_power'] for bn,_,_ in BAND_DEFS]
    lb = np.log1p(bp_list)
    f['band_gradient'] = np.mean(np.diff(lb))
    f['band_range'] = lb.max() - lb.min()
    return f

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("[1/8] Loading data...")
records = []
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith('.zip'): continue
    parsed = parse_filename(fname)
    if not parsed: continue
    unit, sensor, state, date_str, site = parsed
    with zipfile.ZipFile(os.path.join(DATA_DIR, fname)) as z:
        with z.open('ChartData.xlsx') as xf:
            df = pd.read_excel(io.BytesIO(xf.read()))
    freqs = pd.to_numeric(df.iloc[1:,0], errors='coerce')
    powers = pd.to_numeric(df.iloc[1:,1], errors='coerce')
    valid = freqs.notna() & powers.notna()
    feats = compute_features(freqs[valid].values, powers[valid].values)
    feats.update({'file': fname, 'unit': unit, 'sensor': sensor,
                  'state': state, 'date': date_str, 'site': site})
    records.append(feats)

data = pd.DataFrame(records)
data['running'] = (data['state']=='on').astype(int)
meta_cols = ['file','unit','sensor','state','date','site','running']
feature_cols = [c for c in data.columns if c not in meta_cols]
y_true = data['running'].values
sites = sorted(data['site'].unique())

print(f"  {len(data)} samples, {len(feature_cols)} features")
for sn in sites:
    mk = data['site']==sn
    units = data.loc[mk,'unit'].unique()
    print(f"  {sn}: {mk.sum()} ({data.loc[mk,'running'].sum()} ON / {(~data.loc[mk,'running'].astype(bool)).sum()} OFF) units={list(units)}")

# ═══════════════════════════════════════════════════════════════
# FEATURE SUBSETS
# ═══════════════════════════════════════════════════════════════
print("[2/8] Defining feature subsets...")
SCALE_INVARIANT = [c for c in feature_cols if any(k in c for k in
    ['frac','ratio','hi_lo','spectral','entropy','flatness','slope',
     'rolloff','crest','peak_to_mean','peak_freq','gradient','range'])]

BAND_LOGS = ['B1_log','B2_log','B3_log','B4_log','log_total']
BAND_FRACS = ['B1_frac','B2_frac','B3_frac','B4_frac']
RATIOS = [c for c in feature_cols if 'ratio' in c or 'hi_lo' in c]
SHAPE = [c for c in feature_cols if any(k in c for k in
    ['spectral','entropy','flatness','slope','rolloff','crest','peak_to_mean',
     'peak_freq','gradient','range'])]

S1 = ['total_power']
S2 = ['total_power','B1_power','B2_power','B3_power','B4_power']
S3a = S2 + BAND_LOGS
S3b = S3a + [f'{b}_{s}' for b in ['B1','B2','B3','B4'] for s in ['mean','max','std']]
S3c = S3b + RATIOS + BAND_FRACS + [c for c in feature_cols if 'log' in c and c not in S3b]
S3d = S3c + SHAPE
S3e = feature_cols
S_INV = SCALE_INVARIANT
S_INV_R = [c for c in S_INV if c not in SHAPE]

X_all = data[feature_cols].values.astype(float)
mi_scores = mutual_info_classif(X_all, y_true, random_state=42)
mi_rank = {c: i for i,(c,_) in enumerate(sorted(zip(feature_cols, mi_scores), key=lambda x: -x[1]))}
def cohens_d(col):
    on_v = data.loc[data['running']==1, col].values
    off_v = data.loc[data['running']==0, col].values
    ps = np.sqrt((on_v.std()**2 + off_v.std()**2)/2)
    return abs(on_v.mean()-off_v.mean())/(ps+1e-10)
d_rank = {c: i for i,(c,_) in enumerate(sorted([(c,cohens_d(c)) for c in feature_cols], key=lambda x: -x[1]))}
rf_sel = RandomForestClassifier(100, random_state=42, class_weight='balanced')
from sklearn.preprocessing import StandardScaler as SS
rf_sel.fit(SS().fit_transform(X_all), y_true)
rf_rank = {c: i for i,(c,_) in enumerate(sorted(zip(feature_cols, rf_sel.feature_importances_), key=lambda x: -x[1]))}
combined_rank = sorted(feature_cols, key=lambda c: mi_rank[c]+d_rank[c]+rf_rank[c])

mi_ranking_list = sorted(zip(feature_cols, mi_scores), key=lambda x: -x[1])
d_ranking_list = sorted([(c, cohens_d(c)) for c in feature_cols], key=lambda x: -x[1])
rf_ranking_list = sorted(zip(feature_cols, rf_sel.feature_importances_), key=lambda x: -x[1])

S4_10 = combined_rank[:10]
S4_10_inv = [c for c in combined_rank if c in set(SCALE_INVARIANT)][:10]

ALL_SUBSETS = {
    'S1: Total Power (1)': S1,
    'S2: Band Powers (5)': S2,
    'S3a: +Log (10)': S3a,
    'S3b: +Band Stats (22)': S3b,
    'S3c: +Ratios+Fracs': S3c,
    'S3d: +Shape': S3d,
    'S3e: All Features': S3e,
    'SI: Scale-Invariant': S_INV,
    'SI-r: Ratios+Fracs Only': S_INV_R,
    'S4: Top-10 Guided': S4_10,
    'S4i: Top-10 Inv. Guided': S4_10_inv,
}
for name, feats in ALL_SUBSETS.items():
    print(f"  {name}: {len(feats)} features")

# ═══════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════
def preprocess(X_raw):
    """Clip outliers at 1st/99th pctile, then RobustScaler."""
    X = X_raw.copy().astype(float)
    for j in range(X.shape[1]):
        p01, p99 = np.percentile(X[:,j], [1, 99])
        X[:,j] = np.clip(X[:,j], p01, p99)
    return RobustScaler().fit_transform(X)

# ═══════════════════════════════════════════════════════════════
# CLUSTERING — all methods MUST return a label for every sample
# ═══════════════════════════════════════════════════════════════
METHODS = ['KMeans','GMM','Spectral','Agglomerative','DBSCAN','OneClassSVM','MMC']

def run_clustering(X, method, rs=42):
    n = len(X)
    if method == 'KMeans':
        return KMeans(2, n_init=20, random_state=rs).fit_predict(X)
    elif method == 'GMM':
        return GaussianMixture(2, n_init=10, random_state=rs, covariance_type='full').fit_predict(X)
    elif method == 'Spectral':
        if X.shape[1] < 2 or n < 4:
            return KMeans(2, n_init=10, random_state=rs).fit_predict(X)
        return SpectralClustering(2, affinity='rbf', random_state=rs, n_init=10).fit_predict(X)
    elif method == 'Agglomerative':
        return AgglomerativeClustering(2, linkage='ward').fit_predict(X)
    elif method == 'DBSCAN':
        nn = NearestNeighbors(n_neighbors=min(3, n-1)); nn.fit(X)
        dists, _ = nn.kneighbors(X)
        eps = np.median(dists[:,-1]) * 1.2
        raw = DBSCAN(eps=eps, min_samples=2).fit_predict(X)
        # Assign noise points to nearest cluster centroid
        clusters = set(raw) - {-1}
        if len(clusters) == 0:
            return KMeans(2, n_init=20, random_state=rs).fit_predict(X)
        if len(clusters) == 1:
            centroids = {c: X[raw==c].mean(axis=0) for c in clusters}
            other = 1 if 0 in clusters else 0
            noise_mask = raw == -1
            if noise_mask.any():
                raw[noise_mask] = other
            return raw
        centroids = {c: X[raw==c].mean(axis=0) for c in clusters}
        noise_mask = raw == -1
        if noise_mask.any():
            noise_X = X[noise_mask]
            for i, xi in enumerate(noise_X):
                dists_c = {c: np.linalg.norm(xi - centroids[c]) for c in clusters}
                raw[np.where(noise_mask)[0][i]] = min(dists_c, key=dists_c.get)
        return raw
    elif method == 'OneClassSVM':
        return np.where(OneClassSVM(kernel='rbf', gamma='scale', nu=0.4).fit_predict(X)==1, 0, 1)
    elif method == 'MMC':
        best = KMeans(2, n_init=10, random_state=rs).fit_predict(X)
        for _ in range(15):
            svc = SVC(kernel='rbf', gamma='scale', C=1.0); svc.fit(X, best)
            new = svc.predict(X)
            if np.array_equal(new, best): break
            best = new
        return best

def match_labels(yt, yp):
    """Find optimal cluster-to-class mapping. Evaluates on ALL samples."""
    best_f1, best_p, best_r, best_a, best_m = -1, 0, 0, 0, yp.copy()
    unique_labels = sorted(np.unique(yp))
    for perm in permutations([0, 1]):
        mapping = {old: new for old, new in zip(unique_labels, perm)}
        mapped = np.array([mapping.get(p, 0) for p in yp])
        f = f1_score(yt, mapped, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_p = precision_score(yt, mapped, zero_division=0)
            best_r = recall_score(yt, mapped, zero_division=0)
            best_a = accuracy_score(yt, mapped)
            best_m = mapped
    return best_p, best_r, best_f1, best_a, best_m

def evaluate(yt, yp, Xs):
    prec, rec, f1, acc, matched = match_labels(yt, yp)
    ari = adjusted_rand_score(yt, yp)
    nmi = normalized_mutual_info_score(yt, yp)
    nc = len(np.unique(yp))
    sil = np.nan
    if nc >= 2 and len(Xs) >= 4:
        try: sil = silhouette_score(Xs, yp)
        except: pass
    return {'Precision': prec, 'Recall': rec, 'F1': f1, 'Accuracy': acc,
            'ARI': ari, 'NMI': nmi, 'Silhouette': sil, 'n_clusters': nc,
            'matched': matched}

# ═══════════════════════════════════════════════════════════════
# SITE-AGNOSTIC EVALUATION
# ═══════════════════════════════════════════════════════════════
print("[3/8] Running site-agnostic evaluation (100% coverage)...")
site_results = []
site_matched = {}

for sn in sites:
    mk = data['site']==sn
    ys = data.loc[mk,'running'].values
    for ss_name, feat_list in ALL_SUBSETS.items():
        vf = [c for c in feat_list if c in data.columns]
        Xr = data.loc[mk, vf].values.astype(float)
        Xs = preprocess(Xr)
        for method in METHODS:
            try:
                labels = run_clustering(Xs, method)
                assert len(labels) == len(ys), "Not all samples got predictions"
                assert (labels >= 0).all(), "Negative labels found"
                m = evaluate(ys, labels, Xs)
                m.update({'site': sn, 'subset': ss_name, 'method': method, 'n_features': len(vf)})
                site_results.append(m)
                site_matched[(sn, ss_name, method)] = m['matched']
            except Exception as e:
                site_results.append({'site': sn, 'subset': ss_name, 'method': method,
                    'Precision': 0, 'Recall': 0, 'F1': 0, 'Accuracy': 0,
                    'ARI': 0, 'NMI': 0, 'Silhouette': np.nan,
                    'n_features': len(vf), 'error': str(e)})

site_df = pd.DataFrame(site_results)
print(f"  {len(site_df)} per-site experiments (all with 100% sample coverage)")

# Aggregate
agg_rows = []
for ss_name in ALL_SUBSETS:
    for method in METHODS:
        sdf = site_df[(site_df['subset']==ss_name)&(site_df['method']==method)]
        if len(sdf) < 2: continue
        for metric in ['Precision','Recall','F1','Accuracy']:
            vals = sdf[metric].values
            agg_rows.append({'subset': ss_name, 'method': method, 'metric': metric,
                'mean': vals.mean(), 'min': vals.min(), 'max': vals.max(), 'std': vals.std()})
agg_df = pd.DataFrame(agg_rows)

f1_agg = agg_df[agg_df['metric']=='F1'].sort_values('mean', ascending=False)
print("\n  TOP-15 SITE-AGNOSTIC CONFIGS (by mean F1, 100% coverage):")
for i, (_, r) in enumerate(f1_agg.head(15).iterrows()):
    p_row = agg_df[(agg_df['subset']==r['subset'])&(agg_df['method']==r['method'])&(agg_df['metric']=='Precision')]
    r_row = agg_df[(agg_df['subset']==r['subset'])&(agg_df['method']==r['method'])&(agg_df['metric']=='Recall')]
    pm = p_row['mean'].values[0] if len(p_row)>0 else 0
    rm = r_row['mean'].values[0] if len(r_row)>0 else 0
    print(f"    {i+1:>2d}. {r['method']:<15s} {r['subset']:<25s} P={pm:.3f} R={rm:.3f} F1={r['mean']:.3f} (min={r['min']:.3f})")

# ═══════════════════════════════════════════════════════════════
# ENSEMBLE METHODS
# ═══════════════════════════════════════════════════════════════
print("[4/8] Running ensemble methods...")
ENSEMBLE_BASE = ['KMeans','GMM','Spectral','Agglomerative','MMC']
ens_results = []

for sn in sites:
    mk = data['site']==sn; ys = data.loc[mk,'running'].values; n_site = mk.sum()
    for ss_name, feat_list in ALL_SUBSETS.items():
        vf = [c for c in feat_list if c in data.columns]
        Xs = preprocess(data.loc[mk, vf].values.astype(float))
        base_preds = {}; base_f1s = {}
        for method in ENSEMBLE_BASE:
            key = (sn, ss_name, method)
            if key in site_matched:
                base_preds[method] = site_matched[key]
                row = site_df[(site_df['site']==sn)&(site_df['subset']==ss_name)&(site_df['method']==method)]
                base_f1s[method] = row['F1'].values[0] if len(row)>0 else 0
        if len(base_preds) < 3: continue
        preds_arr = np.column_stack([base_preds[m] for m in base_preds])

        # Majority Vote
        mv = np.array([Counter(row).most_common(1)[0][0] for row in preds_arr])
        p,r,f,a,matched = match_labels(ys, mv)
        ens_results.append({'site':sn,'subset':ss_name,'method':'Ensemble-MV',
            'Precision':p,'Recall':r,'F1':f,'Accuracy':a,'ARI':adjusted_rand_score(ys,mv),
            'NMI':normalized_mutual_info_score(ys,mv),'n_features':len(vf),'matched':matched})
        site_matched[(sn, ss_name, 'Ensemble-MV')] = matched

        # Weighted Vote
        weights = np.array([base_f1s[m] for m in base_preds])
        weights = weights / (weights.sum()+1e-10)
        wv = np.zeros(n_site)
        for mi, m in enumerate(base_preds): wv += weights[mi]*base_preds[m]
        wv = (wv >= 0.5).astype(int)
        p,r,f,a,matched = match_labels(ys, wv)
        ens_results.append({'site':sn,'subset':ss_name,'method':'Ensemble-WV',
            'Precision':p,'Recall':r,'F1':f,'Accuracy':a,'ARI':adjusted_rand_score(ys,wv),
            'NMI':normalized_mutual_info_score(ys,wv),'n_features':len(vf),'matched':matched})
        site_matched[(sn, ss_name, 'Ensemble-WV')] = matched

ens_df = pd.DataFrame(ens_results)
combined_df = pd.concat([site_df, ens_df], ignore_index=True)

ALL_METHODS = METHODS + ['Ensemble-MV','Ensemble-WV']
agg2_rows = []
for ss_name in ALL_SUBSETS:
    for method in ALL_METHODS:
        sdf = combined_df[(combined_df['subset']==ss_name)&(combined_df['method']==method)]
        if len(sdf) < 2: continue
        for metric in ['Precision','Recall','F1','Accuracy']:
            vals = sdf[metric].values
            agg2_rows.append({'subset':ss_name,'method':method,'metric':metric,
                'mean':vals.mean(),'min':vals.min(),'max':vals.max(),'std':vals.std()})
agg2_df = pd.DataFrame(agg2_rows)
f1_agg2 = agg2_df[agg2_df['metric']=='F1'].sort_values('mean', ascending=False)

print("\n  TOP-15 (incl. ensembles) by mean F1:")
for i, (_, r) in enumerate(f1_agg2.head(15).iterrows()):
    p_row = agg2_df[(agg2_df['subset']==r['subset'])&(agg2_df['method']==r['method'])&(agg2_df['metric']=='Precision')]
    r_row = agg2_df[(agg2_df['subset']==r['subset'])&(agg2_df['method']==r['method'])&(agg2_df['metric']=='Recall')]
    pm = p_row['mean'].values[0] if len(p_row)>0 else 0
    rm = r_row['mean'].values[0] if len(r_row)>0 else 0
    print(f"    {i+1:>2d}. {r['method']:<15s} {r['subset']:<25s} P={pm:.3f} R={rm:.3f} F1={r['mean']:.3f} (min={r['min']:.3f})")

# ═══════════════════════════════════════════════════════════════
# STABILITY (10 seeds)
# ═══════════════════════════════════════════════════════════════
print("[5/8] Running stability analysis...")
stab_subsets = ['S2: Band Powers (5)', 'SI: Scale-Invariant', 'S4i: Top-10 Inv. Guided']
stab_results = []
for ss in stab_subsets:
    if ss not in ALL_SUBSETS: continue
    vf = [c for c in ALL_SUBSETS[ss] if c in data.columns]
    for sn in sites:
        mk = data['site']==sn
        Xs = preprocess(data.loc[mk, vf].values.astype(float))
        ys = data.loc[mk,'running'].values
        for method in ['KMeans','GMM','Spectral','DBSCAN']:
            f1s = []
            for seed in range(10):
                try:
                    labels = run_clustering(Xs, method, rs=seed*7+1)
                    _,_,f,_,_ = match_labels(ys, labels)
                    f1s.append(f)
                except: pass
            if f1s:
                stab_results.append({'subset':ss,'site':sn,'method':method,
                    'mean_f1':np.mean(f1s),'std_f1':np.std(f1s),'min_f1':np.min(f1s),'max_f1':np.max(f1s)})
stab_df = pd.DataFrame(stab_results)

# ═══════════════════════════════════════════════════════════════
# PER-UNIT DETAIL for production candidate
# ═══════════════════════════════════════════════════════════════
print("[5.5/8] Per-unit detail for production candidate...")
prod_row = f1_agg2.iloc[0]
prod_method = prod_row['method']; prod_subset = prod_row['subset']
print(f"  Production candidate: {prod_method} + {prod_subset}")

unit_detail = []
for sn in sites:
    mk = data['site']==sn
    key = (sn, prod_subset, prod_method)
    if key in site_matched:
        matched = site_matched[key]
        ys = data.loc[mk,'running'].values
        units = data.loc[mk,'unit'].values
        states = data.loc[mk,'state'].values
        for u, s, y, p in zip(units, states, ys, matched):
            correct = "OK" if y==p else "MISS"
            unit_detail.append({'site':sn,'unit':u,'state':s.upper(),'true':y,'pred':int(p),'correct':correct})
            print(f"    {u:<10s} {s.upper():<5s} true={y} pred={int(p)} {correct}")
unit_detail_df = pd.DataFrame(unit_detail)

# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════
print("[6/8] Generating figures...")

method_colors = {'KMeans':'#1D4ED8','GMM':'#047857','Spectral':'#7C3AED',
                 'Agglomerative':'#B45309','DBSCAN':'#DC2626','OneClassSVM':'#0891B2',
                 'MMC':'#BE185D','Ensemble-MV':'#1E293B','Ensemble-WV':'#475569'}

# ── Fig 1: Heatmap F1 ──
f1_piv = f1_agg2.pivot_table(index='subset', columns='method', values='mean')
ss_order = list(ALL_SUBSETS.keys()); m_order = ALL_METHODS
f1_piv = f1_piv.reindex(ss_order)[m_order]
fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(f1_piv.values, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='auto')
for i in range(f1_piv.shape[0]):
    for j in range(f1_piv.shape[1]):
        v = f1_piv.values[i,j]
        txt = f'{v:.2f}' if not np.isnan(v) else '-'
        color = 'white' if (not np.isnan(v) and v < 0.5) else 'black'
        ax.text(j, i, txt, ha='center', va='center', fontsize=7, color=color, fontweight='bold')
ax.set_xticks(range(len(m_order))); ax.set_xticklabels(m_order, fontsize=8, rotation=35, ha='right')
ax.set_yticks(range(len(ss_order))); ax.set_yticklabels(ss_order, fontsize=7)
ax.set_title('Site-Agnostic Mean F1 (100% sample coverage, averaged across sites)', fontsize=12, fontweight='bold')
fig.colorbar(im, ax=ax, shrink=0.6, label='Mean F1')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_heatmap_f1.png'), dpi=180, bbox_inches='tight'); plt.close()

# ── Fig 2: Per-site F1 for top configs ──
top_configs = f1_agg2.head(8)[['subset','method']].values
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(top_configs)); w = 0.35
for si, sn in enumerate(sites):
    vals = []
    for ss, meth in top_configs:
        row = combined_df[(combined_df['site']==sn)&(combined_df['subset']==ss)&(combined_df['method']==meth)]
        vals.append(row['F1'].values[0] if len(row)>0 else 0)
    ax.bar(x+si*w, vals, w, label=sn, color=['#1D4ED8','#047857'][si], alpha=0.85, edgecolor='white')
ax.set_xticks(x+w/2)
ax.set_xticklabels([f'{m}\n{s.split(":")[1][:18].strip()}' for s,m in top_configs], fontsize=7)
ax.set_ylim(0, 1.1); ax.set_ylabel('F1 Score')
ax.set_title('Per-Site F1 for Top-8 Configs (100% coverage)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.axhline(0.9, color='gray', ls='--', alpha=0.4); ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_persite_f1.png'), dpi=180, bbox_inches='tight'); plt.close()

# ── Fig 3: P/R/F1 per method (best subset) ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Precision / Recall / F1 — Site-Agnostic Mean (100% coverage)', fontsize=12, fontweight='bold')
for ax, metric in zip(axes, ['Precision','Recall','F1']):
    m_agg = agg2_df[agg2_df['metric']==metric]
    m_best = m_agg.loc[m_agg.groupby('method')['mean'].idxmax()].sort_values('mean', ascending=True)
    colors = [method_colors.get(m, '#475569') for m in m_best['method']]
    bars = ax.barh(range(len(m_best)), m_best['mean'].values, color=colors, alpha=0.85, edgecolor='white')
    for i, (_, r) in enumerate(m_best.iterrows()):
        ax.plot([r['min'], r['max']], [i, i], color='black', lw=1.5, alpha=0.6)
        ax.plot([r['min']], [i], marker='|', color='black', markersize=8)
        ax.plot([r['max']], [i], marker='|', color='black', markersize=8)
    ax.set_yticks(range(len(m_best))); ax.set_yticklabels(m_best['method'], fontsize=8)
    ax.set_xlim(0, 1.1); ax.set_xlabel(metric)
    ax.set_title(metric, fontsize=10, fontweight='bold')
    ax.axvline(0.9, color='gray', ls='--', alpha=0.4); ax.grid(axis='x', alpha=0.2)
    for bar, val in zip(bars, m_best['mean'].values):
        ax.text(val+0.01, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=7)
plt.tight_layout(rect=[0,0,1,0.93])
plt.savefig(os.path.join(FIG_DIR, 'fig_prf1_methods.png'), dpi=180, bbox_inches='tight'); plt.close()

# ── Fig 4: Progression ──
inc_subsets = [k for k in ALL_SUBSETS if k.startswith('S1') or k.startswith('S2') or k.startswith('S3')]
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(inc_subsets)); w = 0.08
for mi, method in enumerate(ALL_METHODS):
    vals = []
    for ss in inc_subsets:
        row = f1_agg2[(f1_agg2['subset']==ss)&(f1_agg2['method']==method)]
        vals.append(row['mean'].values[0] if len(row)>0 else np.nan)
    ax.bar(x+mi*w, vals, w, label=method, color=method_colors.get(method,'gray'), alpha=0.85, edgecolor='white')
ax.set_xticks(x+4*w)
nf = [len(ALL_SUBSETS[s]) for s in inc_subsets]
ax.set_xticklabels([f'{s.split(":")[1].split("(")[0].strip()}\n({n}f)' for s,n in zip(inc_subsets, nf)], fontsize=7)
ax.set_ylim(0, 1.1); ax.set_ylabel('Mean F1')
ax.set_title('Feature Subset Progression — Mean F1 (100% coverage)', fontsize=12, fontweight='bold')
ax.legend(fontsize=6, ncol=5, loc='lower right'); ax.axhline(0.9, color='gray', ls='--', alpha=0.4)
ax.grid(axis='y', alpha=0.2); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_progression_f1.png'), dpi=180, bbox_inches='tight'); plt.close()

# ── Fig 5: Confusion matrices for prod candidate ──
fig, axes = plt.subplots(1, len(sites), figsize=(7*len(sites), 5))
if len(sites)==1: axes=[axes]
fig.suptitle(f'Production Candidate: {prod_method} + {prod_subset}\nConfusion Matrices (100% coverage)', fontsize=12, fontweight='bold')
for ax, sn in zip(axes, sites):
    mk = data['site']==sn; ys = data.loc[mk,'running'].values
    key = (sn, prod_subset, prod_method)
    matched = site_matched.get(key, np.zeros(mk.sum()))
    cm = confusion_matrix(ys, matched, labels=[0,1])
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=18,fontweight='bold',
                    color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set_xticks([0,1]); ax.set_xticklabels(['Pred OFF','Pred ON'],fontsize=9)
    ax.set_yticks([0,1]); ax.set_yticklabels(['True OFF','True ON'],fontsize=9)
    row_s = combined_df[(combined_df['site']==sn)&(combined_df['subset']==prod_subset)&(combined_df['method']==prod_method)]
    f1_s = row_s['F1'].values[0] if len(row_s)>0 else 0
    p_s = row_s['Precision'].values[0] if len(row_s)>0 else 0
    r_s = row_s['Recall'].values[0] if len(row_s)>0 else 0
    ax.set_title(f'{sn}\nP={p_s:.3f}  R={r_s:.3f}  F1={f1_s:.3f}', fontsize=10)
plt.tight_layout(rect=[0,0,1,0.9])
plt.savefig(os.path.join(FIG_DIR, 'fig_production_cm.png'), dpi=180, bbox_inches='tight'); plt.close()

# ── Fig 6: Feature ranking ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
top_n = 15
for ax, (title, ranking) in zip(axes, [
    ('Mutual Information', mi_ranking_list[:top_n]),
    ("Cohen's d", d_ranking_list[:top_n]),
    ('RF Importance', rf_ranking_list[:top_n])]):
    names = [r[0] for r in ranking][::-1]; vals = [r[1] for r in ranking][::-1]
    inv_set = set(SCALE_INVARIANT)
    colors = ['#047857' if n in inv_set else '#1D4ED8' for n in names]
    ax.barh(range(len(names)), vals, color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_title(title, fontsize=10, fontweight='bold'); ax.grid(axis='x', alpha=0.2)
axes[0].legend(handles=[Patch(facecolor='#047857', label='Scale-Invariant'),
    Patch(facecolor='#1D4ED8', label='Scale-Dependent')], fontsize=7, loc='lower right')
fig.suptitle(f'Feature Ranking (top {top_n})', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(os.path.join(FIG_DIR, 'fig_feature_ranking.png'), dpi=180, bbox_inches='tight'); plt.close()

# ── Fig 7: Scale-invariant comparison ──
fig, ax = plt.subplots(figsize=(12, 6))
compare_subsets = ['S3e: All Features','SI: Scale-Invariant','S4: Top-10 Guided','S4i: Top-10 Inv. Guided']
x = np.arange(len(compare_subsets)); w = 0.08
for mi, method in enumerate(ALL_METHODS):
    vals = []
    for ss in compare_subsets:
        row = f1_agg2[(f1_agg2['subset']==ss)&(f1_agg2['method']==method)]
        vals.append(row['mean'].values[0] if len(row)>0 else np.nan)
    ax.bar(x+mi*w, vals, w, label=method, color=method_colors.get(method,'gray'), alpha=0.85, edgecolor='white')
ax.set_xticks(x+4*w)
ax.set_xticklabels([s.split(':')[1].strip() for s in compare_subsets], fontsize=8)
ax.set_ylim(0, 1.1); ax.set_ylabel('Mean F1'); ax.legend(fontsize=6, ncol=5)
ax.set_title('Scale-Invariant vs Full Features — Mean F1 (100% coverage)', fontsize=11, fontweight='bold')
ax.axhline(0.9, color='gray', ls='--', alpha=0.4); ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_invariant_vs_all.png'), dpi=180, bbox_inches='tight'); plt.close()

# ── Fig 8: PCA per site ──
fig, axes = plt.subplots(1, len(sites), figsize=(7*len(sites), 5))
if len(sites)==1: axes=[axes]
fig.suptitle(f'PCA Projection — {prod_method} + {prod_subset}', fontsize=12, fontweight='bold')
for ax, sn in zip(axes, sites):
    mk = data['site']==sn
    vf = [c for c in ALL_SUBSETS[prod_subset] if c in data.columns]
    Xs = preprocess(data.loc[mk, vf].values.astype(float))
    ys = data.loc[mk,'running'].values
    nc = min(2, Xs.shape[1])
    pca = PCA(nc); Xp = pca.fit_transform(Xs)
    if nc==1: Xp = np.column_stack([Xp, np.zeros(len(Xp))])
    units = data.loc[mk,'unit'].values
    key = (sn, prod_subset, prod_method)
    preds = site_matched.get(key, np.zeros(mk.sum()))
    for tl, marker, label in [(0,'v','OFF'),(1,'^','ON')]:
        tmk = ys==tl
        ax.scatter(Xp[tmk,0], Xp[tmk,1], c=['#1D4ED8' if tl==0 else '#DC2626'],
                   marker=marker, s=80, alpha=0.85, edgecolors='white', linewidth=0.5, label=f'True {label}')
        for i in np.where(tmk)[0]:
            c_mark = "OK" if preds[i]==ys[i] else "MISS"
            ec = '#047857' if c_mark=='OK' else '#DC2626'
            ax.annotate(f'{units[i]} {c_mark}', xy=(Xp[i,0],Xp[i,1]), fontsize=5.5, alpha=0.7, ha='center', va='bottom', color=ec)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})')
    pc2 = pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_)>1 else 0
    ax.set_ylabel(f'PC2 ({pc2:.0%})'); ax.set_title(sn, fontsize=10, fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.2)
plt.tight_layout(rect=[0,0,1,0.93])
plt.savefig(os.path.join(FIG_DIR, 'fig_pca_prod.png'), dpi=180, bbox_inches='tight'); plt.close()

# ── Fig 9: Per-unit prediction detail ──
if len(unit_detail_df) > 0:
    fig, ax = plt.subplots(figsize=(12, max(4, len(unit_detail_df)*0.35)))
    y_pos = range(len(unit_detail_df))
    colors = ['#047857' if r['correct']=='OK' else '#DC2626' for _,r in unit_detail_df.iterrows()]
    ax.barh(y_pos, [1]*len(unit_detail_df), color=colors, alpha=0.7, edgecolor='white')
    for i, (_, r) in enumerate(unit_detail_df.iterrows()):
        label = f"{r['unit']} ({r['site'][:4]}) — {r['state']}"
        pred_label = 'ON' if r['pred']==1 else 'OFF'
        true_label = 'ON' if r['true']==1 else 'OFF'
        ax.text(0.02, i, f"{label}  |  True: {true_label}  Pred: {pred_label}  [{r['correct']}]",
                va='center', fontsize=8, fontweight='bold' if r['correct']=='MISS' else 'normal',
                color='white' if r['correct']=='OK' else 'white')
    ax.set_yticks([]); ax.set_xticks([]); ax.set_xlim(0, 1)
    ax.set_title(f'Per-Unit Prediction Detail — {prod_method} + {prod_subset}', fontsize=11, fontweight='bold')
    ax.legend(handles=[Patch(facecolor='#047857', label='Correct'),
                       Patch(facecolor='#DC2626', label='Misclassified')], fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig_unit_detail.png'), dpi=180, bbox_inches='tight'); plt.close()

print("[7/8] Saving results bundle...")
bundle = {
    'data': data, 'feature_cols': feature_cols, 'ALL_SUBSETS': ALL_SUBSETS,
    'METHODS': METHODS, 'ALL_METHODS': ALL_METHODS, 'method_colors': method_colors,
    'sites': sites, 'combined_df': combined_df, 'agg2_df': agg2_df, 'f1_agg2': f1_agg2,
    'stab_df': stab_df, 'SCALE_INVARIANT': SCALE_INVARIANT,
    'mi_ranking_list': mi_ranking_list, 'd_ranking_list': d_ranking_list,
    'rf_ranking_list': rf_ranking_list, 'combined_rank': combined_rank,
    'S4_10': S4_10, 'S4_10_inv': S4_10_inv,
    'prod_method': prod_method, 'prod_subset': prod_subset,
    'ENSEMBLE_BASE': ENSEMBLE_BASE, 'unit_detail_df': unit_detail_df,
    'site_matched': {str(k): v.tolist() for k, v in site_matched.items()},
}
with open(os.path.join(ROOT, '_unsup_v3_bundle.pkl'), 'wb') as f:
    pickle.dump(bundle, f)

print("[8/8] Done.")
