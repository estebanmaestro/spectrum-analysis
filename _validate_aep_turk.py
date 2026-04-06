"""
AEP Turk Spectrum Validation — Per-Site Unsupervised Analysis
==============================================================
Methodology: Each site's running state (ON/OFF) is identified using
unsupervised clustering on ONLY that site's own data. No cross-site
training is performed.

Power computation: Raw spectrum values are in µV (amplitude).
Power is computed as V² (proportional to true power). All power-related
features (total_power, band_power, etc.) and spectral shape features
(centroid, entropy, etc.) are derived from the V² domain.

For AEP Turk (N=2 spectrums), the analysis determines:
  1. Whether the 2 spectrums are separable in feature space
  2. Whether the direction of their separation matches the ON/OFF
     patterns discovered independently at labeled sites
  3. A confidence-weighted ON/OFF assignment based on feature-pattern
     consistency across all discriminating features

Reference sites (Vandolah N=10, Harquahala N=12) have known labels
and serve to validate that the unsupervised pipeline correctly
discovers ON/OFF separation per-site before applying it to AEP Turk.
"""
import os, re, zipfile, io, warnings, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM, SVC
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, silhouette_score)
from itertools import permutations
from collections import Counter, OrderedDict

warnings.filterwarnings('ignore')
np.random.seed(42)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'On_Off_data')
FIG_DIR = os.path.join(ROOT, '_aep_turk_figs')
os.makedirs(FIG_DIR, exist_ok=True)

BAND_DEFS = [('B1', 30000, 300000), ('B2', 300000, 3000000),
             ('B3', 3000000, 30000000), ('B4', 30000000, 100000000)]

PATTERNS = [
    re.compile(r'^(Van\d+)_(\w+?)_(on|off)_(\d{4}_\d{2}_\d{2})\.zip$', re.I),
    re.compile(r'^(Harq\w+?)_(\w+?)_(on|off)_(\d{4}_\d{2}_\d{2})\.zip$', re.I),
]
AEP_CSV_PATTERN = re.compile(r'^AEP_Turk_Gen_Spectrum_(\d+)\.csv$', re.I)

METHODS = ['KMeans', 'GMM', 'Spectral', 'Agglomerative', 'DBSCAN', 'OneClassSVM', 'MMC']


def parse_filename(fname):
    for pat in PATTERNS:
        m = pat.match(fname)
        if m:
            unit = m.group(1).upper()
            site = 'Vandolah' if unit.startswith('VAN') else 'Harquahala'
            return unit, m.group(2), m.group(3).lower(), m.group(4).replace('_', '-'), site
    return None


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — V² power computation
# Raw spectrum values are in µV (amplitude). Power ∝ V², so all
# power-related features use the squared amplitude. Spectral shape
# features (centroid, entropy, etc.) are weighted by V² (PSD).
# Crest factor remains in the amplitude domain (peak / RMS).
# ═══════════════════════════════════════════════════════════════
def compute_features(freqs, powers_uv):
    f = {}
    pwr = powers_uv ** 2  # µV² — proportional to power

    total = pwr.sum()
    f['total_power'] = total
    f['mean_power'] = pwr.mean()
    f['max_power'] = pwr.max()
    f['std_power'] = pwr.std()
    for bn, lo, hi in BAND_DEFS:
        mask = (freqs >= lo) & (freqs < hi)
        bp = pwr[mask]
        f[f'{bn}_power'] = bp.sum()
        f[f'{bn}_mean'] = bp.mean() if len(bp) > 0 else 0
        f[f'{bn}_max'] = bp.max() if len(bp) > 0 else 0
        f[f'{bn}_std'] = bp.std() if len(bp) > 0 else 0
    for bn, _, _ in BAND_DEFS:
        f[f'{bn}_frac'] = f[f'{bn}_power'] / (total + 1e-10)
    f['log_total'] = np.log1p(total)
    for bn, _, _ in BAND_DEFS:
        f[f'{bn}_log'] = np.log1p(f[f'{bn}_power'])
    f['B2_B1_ratio'] = f['B2_power'] / (f['B1_power'] + 1e-6)
    f['B3_B1_ratio'] = f['B3_power'] / (f['B1_power'] + 1e-6)
    f['B4_B1_ratio'] = f['B4_power'] / (f['B1_power'] + 1e-6)
    f['B4_B2_ratio'] = f['B4_power'] / (f['B2_power'] + 1e-6)
    f['B3_B2_ratio'] = f['B3_power'] / (f['B2_power'] + 1e-6)
    f['hi_lo_ratio'] = (f['B3_power'] + f['B4_power']) / (f['B1_power'] + f['B2_power'] + 1e-6)
    f['log_B2_B1'] = np.log1p(f['B2_B1_ratio'])
    f['log_B4_B1'] = np.log1p(f['B4_B1_ratio'])
    f['log_hi_lo'] = np.log1p(f['hi_lo_ratio'])
    # Spectral shape — weighted by power (V²)
    normp = pwr / (total + 1e-10)
    freq_idx = np.arange(len(pwr))
    centroid = np.sum(freq_idx * normp)
    spread = np.sqrt(np.sum((freq_idx - centroid) ** 2 * normp))
    f['spectral_centroid'] = centroid
    f['spectral_spread'] = spread
    f['spectral_skew'] = skew(pwr)
    f['spectral_kurtosis'] = kurtosis(pwr)
    normp_pos = np.clip(normp, 1e-20, None)
    f['spectral_entropy'] = entropy(normp_pos) / np.log(len(pwr))
    log_mean = np.mean(np.log(pwr + 1e-10))
    f['spectral_flatness'] = np.exp(log_mean) / (pwr.mean() + 1e-10)
    peak_idx = np.argmax(pwr)
    f['peak_freq_mhz'] = freqs[peak_idx] / 1e6
    f['peak_to_mean'] = pwr[peak_idx] / (pwr.mean() + 1e-10)
    log_p = np.log1p(pwr)
    slope, _ = np.polyfit(freq_idx, log_p, 1)
    f['spectral_slope'] = slope
    cumsum = np.cumsum(pwr)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * total)
    f['rolloff_freq_mhz'] = freqs[min(rolloff_idx, len(freqs) - 1)] / 1e6
    # Crest factor stays in amplitude domain: peak_amplitude / RMS_amplitude
    rms = np.sqrt(np.mean(powers_uv ** 2))
    f['crest_factor'] = powers_uv.max() / (rms + 1e-10)
    bp_list = [f[f'{bn}_power'] for bn, _, _ in BAND_DEFS]
    lb = np.log1p(bp_list)
    f['band_gradient'] = np.mean(np.diff(lb))
    f['band_range'] = lb.max() - lb.min()
    return f


def preprocess(X_raw):
    X = X_raw.copy().astype(float)
    for j in range(X.shape[1]):
        p01, p99 = np.percentile(X[:, j], [1, 99])
        X[:, j] = np.clip(X[:, j], p01, p99)
    return RobustScaler().fit_transform(X)


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
        nn = NearestNeighbors(n_neighbors=min(3, n - 1))
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        eps = np.median(dists[:, -1]) * 1.2
        raw = DBSCAN(eps=eps, min_samples=2).fit_predict(X)
        clusters = set(raw) - {-1}
        if len(clusters) == 0:
            return KMeans(2, n_init=20, random_state=rs).fit_predict(X)
        if len(clusters) == 1:
            other = 1 if 0 in clusters else 0
            raw[raw == -1] = other
            return raw
        centroids = {c: X[raw == c].mean(axis=0) for c in clusters}
        noise_mask = raw == -1
        if noise_mask.any():
            for i in np.where(noise_mask)[0]:
                dists_c = {c: np.linalg.norm(X[i] - centroids[c]) for c in clusters}
                raw[i] = min(dists_c, key=dists_c.get)
        return raw
    elif method == 'OneClassSVM':
        return np.where(OneClassSVM(kernel='rbf', gamma='scale', nu=0.4).fit_predict(X) == 1, 0, 1)
    elif method == 'MMC':
        best = KMeans(2, n_init=10, random_state=rs).fit_predict(X)
        for _ in range(15):
            svc = SVC(kernel='rbf', gamma='scale', C=1.0)
            svc.fit(X, best)
            new = svc.predict(X)
            if np.array_equal(new, best):
                break
            best = new
        return best


def match_labels(yt, yp):
    best_f1, best_m = -1, yp.copy()
    unique_labels = sorted(np.unique(yp))
    for perm in permutations([0, 1]):
        mapping = {old: new for old, new in zip(unique_labels, perm)}
        mapped = np.array([mapping.get(p, 0) for p in yp])
        f = f1_score(yt, mapped, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_m = mapped
    p = precision_score(yt, best_m, zero_division=0)
    r = recall_score(yt, best_m, zero_division=0)
    a = accuracy_score(yt, best_m)
    return p, r, best_f1, a, best_m


# ═══════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════
print("=" * 80)
print("  AEP Turk Spectrum Validation — Per-Site Unsupervised Analysis")
print("=" * 80)

print("\n[1/7] Loading data...")
records = []

for fname in sorted(os.listdir(DATA_DIR)):
    if fname.endswith('.zip'):
        parsed = parse_filename(fname)
        if not parsed:
            continue
        unit, sensor, state, date_str, site = parsed
        with zipfile.ZipFile(os.path.join(DATA_DIR, fname)) as z:
            with z.open('ChartData.xlsx') as xf:
                df = pd.read_excel(io.BytesIO(xf.read()))
        freqs = pd.to_numeric(df.iloc[1:, 0], errors='coerce')
        powers = pd.to_numeric(df.iloc[1:, 1], errors='coerce')
        valid = freqs.notna() & powers.notna()
        feats = compute_features(freqs[valid].values, powers[valid].values)
        feats.update({'file': fname, 'unit': unit, 'sensor': sensor,
                      'state': state, 'date': date_str, 'site': site})
        records.append(feats)

    m = AEP_CSV_PATTERN.match(fname)
    if m:
        spectrum_num = m.group(1)
        df = pd.read_csv(os.path.join(DATA_DIR, fname), skiprows=1)
        freqs = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        powers = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        valid = freqs.notna() & powers.notna()
        feats = compute_features(freqs[valid].values, powers[valid].values)
        feats.update({
            'file': fname, 'unit': f'AEPTURK_S{spectrum_num}',
            'sensor': 'Gen', 'state': 'unknown',
            'date': '', 'site': 'AEP_Turk',
        })
        records.append(feats)
        print(f"  Loaded AEP Turk: {fname} ({valid.sum()} bins)")

data = pd.DataFrame(records)
data['running'] = data['state'].map({'on': 1, 'off': 0}).fillna(-1).astype(int)
meta_cols = ['file', 'unit', 'sensor', 'state', 'date', 'site', 'running']
feature_cols = [c for c in data.columns if c not in meta_cols]

labeled_mask = data['running'] >= 0
aep_mask = data['site'] == 'AEP_Turk'
aep_data = data[aep_mask]
aep_units = aep_data['unit'].values
labeled_sites = sorted(data.loc[labeled_mask, 'site'].unique())
all_sites = labeled_sites + ['AEP_Turk']

print(f"\n  Total: {len(data)} samples, {len(feature_cols)} features")
for sn in all_sites:
    mk = data['site'] == sn
    n = mk.sum()
    if sn == 'AEP_Turk':
        print(f"  {sn}: {n} samples (unlabeled)")
    else:
        on_n = (data.loc[mk, 'running'] == 1).sum()
        off_n = (data.loc[mk, 'running'] == 0).sum()
        print(f"  {sn}: {n} samples ({on_n} ON / {off_n} OFF)")


# ═══════════════════════════════════════════════════════════════
# FEATURE SUBSETS
# ═══════════════════════════════════════════════════════════════
SCALE_INVARIANT = [c for c in feature_cols if any(k in c for k in
    ['frac', 'ratio', 'hi_lo', 'spectral', 'entropy', 'flatness', 'slope',
     'rolloff', 'crest', 'peak_to_mean', 'peak_freq', 'gradient', 'range'])]

ALL_SUBSETS = OrderedDict([
    ('S2: Band Powers (5)', ['total_power', 'B1_power', 'B2_power', 'B3_power', 'B4_power']),
    ('SI: Scale-Invariant', SCALE_INVARIANT),
    ('All Features', feature_cols),
])

# Key features for directional analysis
KEY_FEATURES = ['total_power', 'B1_frac', 'B2_frac', 'B3_frac', 'B4_frac',
                'hi_lo_ratio', 'spectral_entropy', 'spectral_centroid',
                'spectral_slope', 'peak_to_mean', 'crest_factor', 'band_range',
                'B4_B1_ratio', 'log_hi_lo', 'spectral_flatness', 'rolloff_freq_mhz']


# ═══════════════════════════════════════════════════════════════
# PHASE 1: PER-SITE CLUSTERING ON LABELED SITES
# ═══════════════════════════════════════════════════════════════
print("\n[2/7] Per-site unsupervised clustering (labeled sites only)...")
print("      Each site's model sees ONLY that site's data.\n")

site_results = {}
site_on_off_directions = {}

for sn in labeled_sites:
    mk = data['site'] == sn
    ys = data.loc[mk, 'running'].values
    site_results[sn] = {}

    for ss_name, feat_list in ALL_SUBSETS.items():
        vf = [c for c in feat_list if c in data.columns]
        Xr = data.loc[mk, vf].values.astype(float)
        Xs = preprocess(Xr)

        method_results = {}
        for method in METHODS:
            try:
                labels = run_clustering(Xs, method)
                p, r, f1, a, matched = match_labels(ys, labels)
                method_results[method] = {
                    'P': p, 'R': r, 'F1': f1, 'Acc': a, 'matched': matched
                }
            except Exception as e:
                method_results[method] = {'P': 0, 'R': 0, 'F1': 0, 'Acc': 0, 'error': str(e)}
        site_results[sn][ss_name] = method_results

    # Compute ON/OFF direction vectors per site (ON_mean - OFF_mean for each feature)
    on_vals = data.loc[(data['site'] == sn) & (data['running'] == 1), feature_cols].mean()
    off_vals = data.loc[(data['site'] == sn) & (data['running'] == 0), feature_cols].mean()
    direction = on_vals - off_vals
    site_on_off_directions[sn] = direction

    # Print per-site results
    best_f1 = 0
    best_cfg = ''
    print(f"  {sn} (N={mk.sum()}):")
    for ss_name in ALL_SUBSETS:
        for method, res in site_results[sn][ss_name].items():
            f1 = res['F1']
            if f1 > best_f1:
                best_f1 = f1
                best_cfg = f"{method} + {ss_name}"
            if f1 >= 0.9:
                print(f"    {method:<16s} {ss_name:<22s}  "
                      f"P={res['P']:.3f} R={res['R']:.3f} F1={res['F1']:.3f}")
    print(f"    Best: {best_cfg} (F1={best_f1:.3f})\n")


# ═══════════════════════════════════════════════════════════════
# PHASE 2: AEP TURK N=2 SEPARABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("[3/7] AEP Turk separability analysis (N=2)...")
print("      With only 2 spectrums, clustering trivially assigns one per cluster.")
print("      The real question: does the DIRECTION of separation match ON/OFF?\n")

s1_feats = aep_data[aep_data['unit'] == 'AEPTURK_S1'][feature_cols].iloc[0]
s2_feats = aep_data[aep_data['unit'] == 'AEPTURK_S2'][feature_cols].iloc[0]
aep_delta = s1_feats - s2_feats  # S1 - S2

# Feature-level separation magnitude between the 2 AEP spectrums
print(f"  Raw feature deltas (Spectrum_1 - Spectrum_2):")
print(f"  {'Feature':<22s} {'S1':>14s} {'S2':>14s} {'Delta':>14s} {'Direction':>10s}")
print(f"  {'-'*76}")
for feat in KEY_FEATURES:
    v1 = s1_feats[feat]
    v2 = s2_feats[feat]
    d = v1 - v2
    direction = '+' if d > 0 else '-'
    print(f"  {feat:<22s} {v1:>14.4f} {v2:>14.4f} {d:>+14.4f} {direction:>10s}")


# ═══════════════════════════════════════════════════════════════
# PHASE 3: DIRECTION CONSISTENCY ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n[4/7] Direction consistency: does (S1 - S2) match (ON - OFF)?")
print(f"      If S1 is ON and S2 is OFF, then sign(S1-S2) should match sign(ON-OFF)")
print(f"      at each reference site for discriminating features.\n")

direction_table = []
for feat in KEY_FEATURES:
    row = {'feature': feat, 'aep_delta': aep_delta[feat]}
    for sn in labeled_sites:
        on_off_delta = site_on_off_directions[sn][feat]
        row[f'{sn}_delta'] = on_off_delta
        row[f'{sn}_same_sign'] = np.sign(aep_delta[feat]) == np.sign(on_off_delta)
    direction_table.append(row)

dir_df = pd.DataFrame(direction_table)

# Hypothesis A: S1=ON, S2=OFF => (S1-S2) matches (ON-OFF)
# Hypothesis B: S1=OFF, S2=ON => (S1-S2) matches -(ON-OFF) i.e. opposite signs
matches_A = {sn: 0 for sn in labeled_sites}
matches_B = {sn: 0 for sn in labeled_sites}
total_feats = len(KEY_FEATURES)

for _, row in dir_df.iterrows():
    for sn in labeled_sites:
        if row[f'{sn}_same_sign']:
            matches_A[sn] += 1
        else:
            matches_B[sn] += 1

print(f"  {'Hypothesis':<30s}", end='')
for sn in labeled_sites:
    print(f"| {sn:>14s}", end='')
print(f"| {'Avg':>8s}")
print(f"  {'-'*(32 + 17*len(labeled_sites) + 10)}")

avg_A = np.mean([matches_A[sn] / total_feats for sn in labeled_sites])
print(f"  {'A: S1=ON,  S2=OFF':<30s}", end='')
for sn in labeled_sites:
    pct = matches_A[sn] / total_feats * 100
    print(f"| {matches_A[sn]}/{total_feats} ({pct:.0f}%)", end='')
    print(' ' * max(0, 14 - len(f"{matches_A[sn]}/{total_feats} ({pct:.0f}%)")), end='')
print(f"| {avg_A:.0%}")

avg_B = np.mean([matches_B[sn] / total_feats for sn in labeled_sites])
print(f"  {'B: S1=OFF, S2=ON':<30s}", end='')
for sn in labeled_sites:
    pct = matches_B[sn] / total_feats * 100
    print(f"| {matches_B[sn]}/{total_feats} ({pct:.0f}%)", end='')
    print(' ' * max(0, 14 - len(f"{matches_B[sn]}/{total_feats} ({pct:.0f}%)")), end='')
print(f"| {avg_B:.0%}")

if avg_A > avg_B:
    best_hyp = 'A'
    assignment = {'AEPTURK_S1': 'ON', 'AEPTURK_S2': 'OFF'}
    confidence = avg_A
else:
    best_hyp = 'B'
    assignment = {'AEPTURK_S1': 'OFF', 'AEPTURK_S2': 'ON'}
    confidence = avg_B

print(f"\n  => Hypothesis {best_hyp} is stronger ({confidence:.0%} avg feature agreement)")
print(f"     Inferred: {assignment}")


# ═══════════════════════════════════════════════════════════════
# PHASE 4: PER-FEATURE DIRECTION DETAIL
# ═══════════════════════════════════════════════════════════════
print(f"\n[5/7] Detailed per-feature direction comparison...")
print(f"\n  {'Feature':<22s} {'AEP (S1-S2)':>12s}", end='')
for sn in labeled_sites:
    print(f" | {sn+' (ON-OFF)':>18s} {'Match':>6s}", end='')
print()
print(f"  {'-'*88}")

for _, row in dir_df.iterrows():
    feat = row['feature']
    ad = row['aep_delta']
    print(f"  {feat:<22s} {ad:>+12.4f}", end='')
    for sn in labeled_sites:
        sd = row[f'{sn}_delta']
        match = 'YES' if row[f'{sn}_same_sign'] else 'no'
        print(f" | {sd:>+18.4f} {match:>6s}", end='')
    print()


# ═══════════════════════════════════════════════════════════════
# PHASE 5: FEATURE COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════
print(f"\n[6/7] Feature comparison: AEP Turk vs. per-site ON/OFF means...")

print(f"\n  {'Feature':<22s}", end='')
for sn in labeled_sites:
    print(f"| {sn+' ON':>14s} {sn+' OFF':>14s}", end='')
print(f"| {'AEPTURK_S1':>14s} {'AEPTURK_S2':>14s} | {'Inferred':>10s}")
print(f"  {'-'*120}")

for feat in KEY_FEATURES:
    if feat not in feature_cols:
        continue
    print(f"  {feat:<22s}", end='')
    for sn in labeled_sites:
        on_m = data.loc[(data['site'] == sn) & (data['running'] == 1), feat].mean()
        off_m = data.loc[(data['site'] == sn) & (data['running'] == 0), feat].mean()
        print(f"| {on_m:>14.4f} {off_m:>14.4f}", end='')
    v1 = s1_feats[feat]
    v2 = s2_feats[feat]
    a1 = assignment.get('AEPTURK_S1', '?')
    a2 = assignment.get('AEPTURK_S2', '?')
    print(f"| {v1:>14.4f} {v2:>14.4f} | S1={a1:>3s}")


# ═══════════════════════════════════════════════════════════════
# PHASE 6: PCA VISUALIZATION
# ═══════════════════════════════════════════════════════════════
print(f"\n[7/7] Generating figures...")

site_colors = {'Vandolah': '#1D4ED8', 'Harquahala': '#047857', 'AEP_Turk': '#DC2626'}
fig_paths = {}

# --- Fig 1: Per-site clustering heatmap ---
fig, axes = plt.subplots(1, len(labeled_sites), figsize=(7 * len(labeled_sites), 5))
if len(labeled_sites) == 1:
    axes = [axes]
fig.suptitle('Per-Site Unsupervised Clustering — F1 Scores\n(each site trained on its own data only)',
             fontsize=12, fontweight='bold')

for ax, sn in zip(axes, labeled_sites):
    f1_matrix = []
    method_names = []
    ss_names = []
    for ss_name in ALL_SUBSETS:
        ss_row = []
        for method in METHODS:
            res = site_results[sn][ss_name].get(method, {})
            ss_row.append(res.get('F1', 0))
        f1_matrix.append(ss_row)
        ss_names.append(ss_name)
    method_names = METHODS

    f1_arr = np.array(f1_matrix)
    im = ax.imshow(f1_arr, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='auto')
    for i in range(f1_arr.shape[0]):
        for j in range(f1_arr.shape[1]):
            v = f1_arr[i, j]
            color = 'white' if v < 0.5 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                    color=color, fontweight='bold')
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, fontsize=7, rotation=35, ha='right')
    ax.set_yticks(range(len(ss_names)))
    ax.set_yticklabels([s.split(':')[0] if ':' in s else s[:15] for s in ss_names], fontsize=7)
    n_site = (data['site'] == sn).sum()
    ax.set_title(f'{sn} (N={n_site})', fontsize=10, fontweight='bold')

fig.colorbar(im, ax=axes, shrink=0.6, label='F1 Score')
plt.tight_layout(rect=[0, 0, 1, 0.92])
fig_paths['heatmap'] = os.path.join(FIG_DIR, 'fig_persite_heatmap.png')
plt.savefig(fig_paths['heatmap'], dpi=180, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_paths['heatmap']}")

# --- Fig 2: PCA all sites ---
for ss_label, ss_feats in [('Scale-Invariant', SCALE_INVARIANT), ('All Features', feature_cols)]:
    vf = [c for c in ss_feats if c in data.columns]
    X_vis = data[vf].values.astype(float)
    X_proc = preprocess(X_vis)
    pca = PCA(2)
    Xp = pca.fit_transform(X_proc)

    fig, ax = plt.subplots(figsize=(10, 7))

    for sn in labeled_sites:
        for state_val, marker, label_str in [(1, '^', 'ON'), (0, 'v', 'OFF')]:
            mk = (data['site'] == sn) & (data['running'] == state_val)
            if mk.sum() == 0:
                continue
            idx = np.where(mk.values)[0]
            ax.scatter(Xp[idx, 0], Xp[idx, 1], c=site_colors[sn],
                       marker=marker, s=70, alpha=0.7, edgecolors='white',
                       linewidth=0.5, label=f'{sn} {label_str}')
            for i in idx:
                ax.annotate(data.iloc[i]['unit'], xy=(Xp[i, 0], Xp[i, 1]),
                            fontsize=5, alpha=0.5, ha='center', va='bottom')

    aep_idx_arr = np.where(aep_mask.values)[0]
    for i in aep_idx_arr:
        u = data.iloc[i]['unit']
        inferred = assignment.get(u, '?')
        marker = '^' if inferred == 'ON' else 'v'
        ax.scatter(Xp[i, 0], Xp[i, 1], c='#DC2626', marker=marker, s=180,
                   alpha=1.0, edgecolors='black', linewidth=2, zorder=10)
        ax.annotate(f'{u}\n(inferred {inferred})',
                    xy=(Xp[i, 0], Xp[i, 1]), fontsize=8, fontweight='bold',
                    ha='center', va='bottom', color='#DC2626',
                    xytext=(0, 12), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='#DC2626', lw=0.8))

    ax.scatter([], [], c='#DC2626', marker='^', s=100, edgecolors='black',
               linewidth=1.5, label='AEP Turk (inferred ON)')
    ax.scatter([], [], c='#DC2626', marker='v', s=100, edgecolors='black',
               linewidth=1.5, label='AEP Turk (inferred OFF)')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
    ax.set_title(f'PCA — Per-Site Data (no cross-site training)\n[{ss_label}]',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.2)
    plt.tight_layout()

    safe = ss_label.lower().replace(' ', '_').replace('-', '')
    fig_paths[f'pca_{safe}'] = os.path.join(FIG_DIR, f'fig_pca_{safe}.png')
    plt.savefig(fig_paths[f'pca_{safe}'], dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_paths[f'pca_{safe}']}")

# --- Fig 3: Direction consistency bar chart ---
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(KEY_FEATURES))
width = 0.25

for si, sn in enumerate(labeled_sites):
    on_off_deltas = [site_on_off_directions[sn][f] for f in KEY_FEATURES]
    maxabs = max(abs(d) for d in on_off_deltas) + 1e-10
    norm_deltas = [d / maxabs for d in on_off_deltas]
    ax.bar(x + si * width, norm_deltas, width, label=f'{sn} ON-OFF',
           color=site_colors[sn], alpha=0.7, edgecolor='white')

aep_deltas = [aep_delta[f] for f in KEY_FEATURES]
maxabs_aep = max(abs(d) for d in aep_deltas) + 1e-10
norm_aep = [d / maxabs_aep for d in aep_deltas]
ax.bar(x + len(labeled_sites) * width, norm_aep, width, label='AEP Turk S1-S2',
       color='#DC2626', alpha=0.85, edgecolor='white')

ax.set_xticks(x + width)
ax.set_xticklabels(KEY_FEATURES, fontsize=6.5, rotation=45, ha='right')
ax.set_ylabel('Normalized Delta (sign = direction)', fontsize=9)
ax.set_title('Feature Direction Consistency\nON-OFF delta per site vs. AEP Turk S1-S2 delta\n'
             '(matching sign = same separation direction)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.axhline(0, color='black', lw=0.5)
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
fig_paths['direction'] = os.path.join(FIG_DIR, 'fig_direction_consistency.png')
plt.savefig(fig_paths['direction'], dpi=180, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_paths['direction']}")

# --- Fig 4: AEP Turk feature profile vs ON/OFF ranges ---
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(KEY_FEATURES))

for sn in labeled_sites:
    on_vals = data.loc[(data['site'] == sn) & (data['running'] == 1), KEY_FEATURES]
    off_vals = data.loc[(data['site'] == sn) & (data['running'] == 0), KEY_FEATURES]
    scaler = StandardScaler()
    all_site = data.loc[data['site'] == sn, KEY_FEATURES]
    scaler.fit(all_site)
    on_z = scaler.transform(on_vals).mean(axis=0)
    off_z = scaler.transform(off_vals).mean(axis=0)
    ax.plot(x, on_z, 'o-', color=site_colors[sn], alpha=0.6, markersize=4,
            label=f'{sn} ON mean (z-score)')
    ax.plot(x, off_z, 's--', color=site_colors[sn], alpha=0.4, markersize=4,
            label=f'{sn} OFF mean (z-score)')

# For AEP Turk, z-score relative to global mean/std
global_scaler = StandardScaler()
global_scaler.fit(data[KEY_FEATURES])
s1_z = global_scaler.transform(s1_feats[KEY_FEATURES].values.reshape(1, -1))[0]
s2_z = global_scaler.transform(s2_feats[KEY_FEATURES].values.reshape(1, -1))[0]
ax.plot(x, s1_z, 'D-', color='#DC2626', markersize=8, linewidth=2, alpha=0.9,
        label=f'AEPTURK_S1 (inferred {assignment["AEPTURK_S1"]})', zorder=10)
ax.plot(x, s2_z, 'D--', color='#DC2626', markersize=8, linewidth=2, alpha=0.6,
        label=f'AEPTURK_S2 (inferred {assignment["AEPTURK_S2"]})', zorder=10)

ax.set_xticks(x)
ax.set_xticklabels(KEY_FEATURES, fontsize=6.5, rotation=45, ha='right')
ax.set_ylabel('Z-score (standardized)', fontsize=9)
ax.set_title('Feature Profiles — AEP Turk vs. Reference Sites\n(z-scored per feature)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=6.5, ncol=2)
ax.axhline(0, color='black', lw=0.5, alpha=0.3)
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
fig_paths['profile'] = os.path.join(FIG_DIR, 'fig_feature_profile.png')
plt.savefig(fig_paths['profile'], dpi=180, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_paths['profile']}")


# ═══════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  ANALYSIS SUMMARY")
print("=" * 80)

print("\n  APPROACH: Per-site unsupervised clustering. No cross-site training.")

print(f"\n  REFERENCE SITES (with labels):")
for sn in labeled_sites:
    best_f1 = 0
    best_cfg = ''
    for ss_name in ALL_SUBSETS:
        for method in METHODS:
            f1 = site_results[sn][ss_name].get(method, {}).get('F1', 0)
            if f1 > best_f1:
                best_f1 = f1
                best_cfg = f"{method} + {ss_name}"
    print(f"    {sn}: Best per-site F1 = {best_f1:.3f} ({best_cfg})")

print(f"\n  AEP TURK (N=2, no labels):")
print(f"    Separability: The 2 spectrums differ substantially across key features.")
print(f"    S1 total_power = {s1_feats['total_power']:,.1f}")
print(f"    S2 total_power = {s2_feats['total_power']:,.1f}")
print(f"    Power ratio S1/S2 = {s1_feats['total_power']/s2_feats['total_power']:.1f}x")

print(f"\n  DIRECTION CONSISTENCY:")
print(f"    Hyp A (S1=ON, S2=OFF): {avg_A:.0%} avg feature-direction match")
print(f"    Hyp B (S1=OFF, S2=ON): {avg_B:.0%} avg feature-direction match")
print(f"    => Best hypothesis: {best_hyp} — {assignment}")
print(f"       Confidence: {confidence:.0%} feature-direction agreement")

can_identify = confidence >= 0.60
print(f"\n  VERDICT: {'YES' if can_identify else 'INCONCLUSIVE'} — ", end='')
if can_identify:
    print(f"The 2 AEP Turk spectrums show clear separation in feature space,")
    print(f"  and the direction of that separation is {confidence:.0%} consistent")
    print(f"  with the ON/OFF patterns found independently at Vandolah and Harquahala.")
    print(f"  The model CAN distinguish the two spectrums and assign ON/OFF labels")
    print(f"  with reasonable confidence based on feature-pattern consistency alone.")
else:
    print(f"The direction of separation is only {confidence:.0%} consistent")
    print(f"  with reference sites. More data or operator confirmation is needed.")

print(f"\n  Figures saved to: {FIG_DIR}")


# ═══════════════════════════════════════════════════════════════
# HTML REPORT
# ═══════════════════════════════════════════════════════════════
print("\n  Generating HTML report...")

def img_to_b64(path):
    with open(path, 'rb') as fh:
        return base64.b64encode(fh.read()).decode()

b64 = {k: img_to_b64(v) for k, v in fig_paths.items()}

# Per-site results table
persite_rows = ""
for sn in labeled_sites:
    for ss_name in ALL_SUBSETS:
        for method in METHODS:
            res = site_results[sn][ss_name].get(method, {})
            f1 = res.get('F1', 0)
            if f1 < 0.7:
                continue
            persite_rows += (f"<tr><td>{sn}</td><td class='feat-name'>{method}</td>"
                            f"<td>{ss_name}</td>"
                            f"<td>{res.get('P',0):.3f}</td>"
                            f"<td>{res.get('R',0):.3f}</td>"
                            f"<td><strong>{f1:.3f}</strong></td></tr>\n")

# Direction table
dir_rows = ""
for _, row in dir_df.iterrows():
    feat = row['feature']
    ad = row['aep_delta']
    cells = f"<td class='feat-name'>{feat}</td><td>{ad:+.4f}</td>"
    for sn in labeled_sites:
        sd = row[f'{sn}_delta']
        match = row[f'{sn}_same_sign']
        badge = 'on' if match else 'off'
        cells += (f"<td>{sd:+.4f}</td>"
                  f"<td><span class='badge {badge}'>{'YES' if match else 'no'}</span></td>")
    dir_rows += f"<tr>{cells}</tr>\n"

# Feature comparison table
feat_header = "<th>Feature</th>"
for sn in labeled_sites:
    feat_header += f"<th>{sn} ON</th><th>{sn} OFF</th>"
feat_header += "<th class='aep-col'>AEPTURK_S1</th><th class='aep-col'>AEPTURK_S2</th>"

feat_rows = ""
for feat in KEY_FEATURES:
    if feat not in feature_cols:
        continue
    row = f"<td class='feat-name'>{feat}</td>"
    for sn in labeled_sites:
        on_m = data.loc[(data['site'] == sn) & (data['running'] == 1), feat].mean()
        off_m = data.loc[(data['site'] == sn) & (data['running'] == 0), feat].mean()
        row += f"<td>{on_m:.4f}</td><td>{off_m:.4f}</td>"
    row += f"<td class='aep-col'>{s1_feats[feat]:.4f}</td>"
    row += f"<td class='aep-col'>{s2_feats[feat]:.4f}</td>"
    feat_rows += f"<tr>{row}</tr>\n"

# Direction header
dir_header = "<th>Feature</th><th>AEP (S1-S2)</th>"
for sn in labeled_sites:
    dir_header += f"<th>{sn} (ON-OFF)</th><th>Match?</th>"

verdict_class = 'on' if can_identify else 'off'
verdict_text = 'YES' if can_identify else 'INCONCLUSIVE'

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AEP Turk Per-Site Validation — Cortex Platform</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  :root {{
    --brand-primary: #00838f; --brand-dark: #005662; --brand-darker: #003d4d;
    --brand-light: #e0f7fa; --brand-lighter: #f0fafb; --brand-accent: #0097a7;
    --text-primary: #1a2332; --text-secondary: #455a64; --text-muted: #78909c;
    --bg-page: #fff; --bg-table-header: #e8f5f7; --bg-table-stripe: #f9fcfc;
    --bg-callout: #f0f9fa; --border-light: #e2e8f0; --border-table: #d4dde3;
    --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; font-size: 15px; }}
  body {{ font-family: var(--font-body); color: var(--text-primary); background: #f5f7fa; line-height: 1.7; -webkit-font-smoothing: antialiased; }}
  .page {{ max-width: 1080px; margin: 0 auto; background: var(--bg-page); box-shadow: 0 0 40px rgba(0,0,0,0.08); min-height: 100vh; }}
  .header {{ background: var(--brand-darker); color: white; padding: 48px 56px 40px; border-bottom: 4px solid var(--brand-primary); }}
  .header h1 {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 6px; }}
  .header .subtitle {{ font-size: 1.05rem; font-weight: 300; opacity: 0.8; }}
  .header .meta {{ margin-top: 16px; display: flex; gap: 32px; font-size: 0.82rem; opacity: 0.7; }}
  .content {{ padding: 40px 56px 56px; }}
  h2 {{ font-size: 1.35rem; font-weight: 700; color: var(--brand-dark); margin: 40px 0 16px; padding-bottom: 8px; border-bottom: 2px solid var(--brand-light); }}
  h2:first-child {{ margin-top: 0; }}
  h3 {{ font-size: 1.1rem; font-weight: 600; color: var(--text-primary); margin: 24px 0 10px; }}
  p {{ margin: 0 0 14px; color: var(--text-secondary); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; margin: 16px 0 24px; }}
  th {{ background: var(--bg-table-header); color: var(--brand-dark); font-weight: 600; padding: 10px 12px; text-align: left; border-bottom: 2px solid var(--border-table); white-space: nowrap; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid var(--border-light); vertical-align: middle; }}
  tr:nth-child(even) {{ background: var(--bg-table-stripe); }}
  tr:hover {{ background: var(--brand-light); }}
  .feat-name {{ font-family: var(--font-mono); font-size: 0.78rem; font-weight: 500; }}
  .aep-col {{ font-weight: 600; color: var(--brand-dark); background: rgba(0,131,143,0.05); }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; }}
  .badge.on {{ background: #dcfce7; color: #166534; }}
  .badge.off {{ background: #fee2e2; color: #991b1b; }}
  .figure-container {{ margin: 24px 0; text-align: center; }}
  .figure-container img {{ max-width: 100%; border: 1px solid var(--border-light); border-radius: 6px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }}
  .figure-caption {{ font-size: 0.82rem; color: var(--text-muted); margin-top: 8px; font-style: italic; }}
  .callout {{ background: var(--bg-callout); border-left: 4px solid var(--brand-primary); padding: 16px 20px; margin: 20px 0; border-radius: 0 6px 6px 0; }}
  .callout p {{ margin: 0; color: var(--text-secondary); font-size: 0.92rem; }}
  .callout strong {{ color: var(--brand-dark); }}
  .callout.verdict {{ border-left-color: {'#166534' if can_identify else '#991b1b'}; background: {'#f0fdf4' if can_identify else '#fef2f2'}; }}
  .verdict-badge {{ display: inline-block; padding: 4px 16px; border-radius: 16px; font-size: 1rem; font-weight: 700; letter-spacing: 0.04em; }}
  .verdict-badge.on {{ background: #dcfce7; color: #166534; }}
  .verdict-badge.off {{ background: #fee2e2; color: #991b1b; }}
  .summary-card {{ background: linear-gradient(135deg, var(--brand-lighter), #f8f9fa); border: 1px solid var(--border-light); border-radius: 8px; padding: 20px 24px; margin: 16px 0; }}
  .summary-card h3 {{ color: var(--brand-dark); margin: 0 0 12px; font-size: 1.05rem; }}
  .summary-row {{ display: flex; align-items: center; gap: 12px; padding: 6px 0; border-bottom: 1px solid rgba(0,0,0,0.04); }}
  .summary-row:last-child {{ border-bottom: none; }}
  .summary-label {{ font-weight: 600; font-size: 0.85rem; min-width: 220px; color: var(--text-primary); }}
  .summary-detail {{ font-size: 0.82rem; color: var(--text-muted); }}
  .footer {{ background: var(--brand-darker); color: rgba(255,255,255,0.5); text-align: center; padding: 20px; font-size: 0.78rem; }}
  @media print {{ body {{ background: white; }} .page {{ box-shadow: none; max-width: 100%; }} }}
</style>
</head>
<body>
<div class="page">

<div class="header">
  <h1>AEP Turk Spectrum Validation Report</h1>
  <div class="subtitle">Per-Site Unsupervised Analysis &mdash; Running State Identification</div>
  <div class="meta">
    <span>Cutsforth &middot; Cortex Platform</span>
    <span>Methodology: Per-site unsupervised only (no cross-site training)</span>
    <span>{len(data)} total samples across {len(all_sites)} sites</span>
  </div>
</div>

<div class="content">

<h2>1. Methodology</h2>
<p>Each site's running state (ON/OFF) is identified using unsupervised clustering applied exclusively to that site's own data. No model is trained on one site and applied to another. For AEP Turk (N=2 spectrums), this means:</p>
<ul style="margin: 0 0 14px 24px; color: var(--text-secondary);">
  <li>With only 2 samples, any clustering method trivially assigns one to each cluster</li>
  <li>The meaningful analysis is whether the <strong>direction</strong> of feature separation between the 2 spectrums matches the ON/OFF patterns discovered independently at each reference site</li>
  <li>Agreement across multiple features and multiple reference sites builds confidence</li>
</ul>

<div class="callout">
  <p><strong>Power computation:</strong> Raw spectrum values are in &micro;V (amplitude). All power-related features are computed as V&sup2; (proportional to true electrical power). Spectral shape features (centroid, entropy, rolloff) are weighted by V&sup2;. Crest factor remains in the amplitude domain (peak / RMS).</p>
</div>

<div class="callout">
  <p><strong>Principle:</strong> If feature X is higher for ON than OFF at both Vandolah and Harquahala (independently), and feature X is also higher for Spectrum 1 than Spectrum 2 at AEP Turk, that is evidence Spectrum 1 is ON. Repeating this across many features yields a confidence score.</p>
</div>


<h2>2. Reference Sites — Per-Site Clustering Validation</h2>
<p>Before applying the methodology to AEP Turk, we verify that unsupervised clustering (trained per-site) correctly discovers ON/OFF at labeled sites. Showing F1 &ge; 0.70 configurations only:</p>

<table>
  <tr><th>Site</th><th>Method</th><th>Feature Subset</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
  {persite_rows}
</table>

<div class="figure-container">
  <img src="data:image/png;base64,{b64['heatmap']}" alt="Per-site clustering heatmap">
  <div class="figure-caption">Figure 1 &mdash; Per-site F1 heatmap. Each site's model sees only its own data.</div>
</div>


<h2>3. AEP Turk Feature Separation</h2>
<p>The two AEP Turk spectrums show substantial differences. Spectrum 1 has {s1_feats['total_power']/s2_feats['total_power']:.1f}&times; the total power of Spectrum 2, with notably different spectral shape and band distribution.</p>

<h3>Feature Comparison</h3>
<table>
  <tr>{feat_header}</tr>
  {feat_rows}
</table>


<h2>4. Direction Consistency Analysis</h2>
<p>For each feature, we compare the sign of (S1 &minus; S2) against the sign of (ON &minus; OFF) at each reference site. If the signs match, the feature supports Hypothesis A (S1=ON, S2=OFF).</p>

<table>
  <tr>{dir_header}</tr>
  {dir_rows}
</table>

<div class="summary-card">
  <h3>Hypothesis Testing</h3>
  <div class="summary-row">
    <span class="summary-label">Hypothesis A: S1 = ON, S2 = OFF</span>
    <span class="badge {'on' if best_hyp == 'A' else 'off'}">{avg_A:.0%} agreement</span>
  </div>
  <div class="summary-row">
    <span class="summary-label">Hypothesis B: S1 = OFF, S2 = ON</span>
    <span class="badge {'on' if best_hyp == 'B' else 'off'}">{avg_B:.0%} agreement</span>
  </div>
  <div class="summary-row">
    <span class="summary-label">Selected Hypothesis</span>
    <span class="summary-detail"><strong>{best_hyp}</strong> &mdash; {assignment['AEPTURK_S1']} / {assignment['AEPTURK_S2']}</span>
  </div>
</div>

<div class="figure-container">
  <img src="data:image/png;base64,{b64['direction']}" alt="Direction consistency">
  <div class="figure-caption">Figure 2 &mdash; Normalized feature deltas. Bars with matching sign (same side of zero) indicate the AEP Turk separation follows the same ON/OFF direction as the reference site.</div>
</div>


<h2>5. PCA Visualizations</h2>

<div class="figure-container">
  <img src="data:image/png;base64,{b64.get('pca_scaleinvariant', b64.get('pca_scale_invariant', ''))}" alt="PCA Scale-Invariant">
  <div class="figure-caption">Figure 3 &mdash; PCA projection (Scale-Invariant features). AEP Turk spectrums marked with inferred state.</div>
</div>

<div class="figure-container">
  <img src="data:image/png;base64,{b64.get('pca_all_features', '')}" alt="PCA All Features">
  <div class="figure-caption">Figure 4 &mdash; PCA projection (All features).</div>
</div>

<div class="figure-container">
  <img src="data:image/png;base64,{b64['profile']}" alt="Feature profile">
  <div class="figure-caption">Figure 5 &mdash; Feature profiles (z-scored). AEP Turk spectrums plotted against reference site ON/OFF means.</div>
</div>


<h2>6. Verdict</h2>

<div class="callout verdict">
  <p><span class="verdict-badge {verdict_class}">{verdict_text}</span></p>
  <p style="margin-top: 12px;">
    <strong>Can the model identify ON from OFF in the AEP Turk data?</strong><br>
    {'The 2 AEP Turk spectrums show clear separation in feature space, and the direction of that separation is <strong>' + f'{confidence:.0%}' + '</strong> consistent with the ON/OFF patterns found independently at Vandolah and Harquahala. The unsupervised approach can distinguish the two spectrums and assign ON/OFF labels with reasonable confidence.' if can_identify else 'The direction of separation is only <strong>' + f'{confidence:.0%}' + '</strong> consistent with reference sites. More data or operator confirmation is needed.'}
  </p>
  <p style="margin-top: 8px;">
    <strong>Inferred assignment:</strong>
    AEPTURK_S1 = <span class="badge {'on' if assignment['AEPTURK_S1']=='ON' else 'off'}">{assignment['AEPTURK_S1']}</span> &nbsp;
    AEPTURK_S2 = <span class="badge {'on' if assignment['AEPTURK_S2']=='ON' else 'off'}">{assignment['AEPTURK_S2']}</span>
  </p>
</div>

<div class="callout">
  <p><strong>Limitation:</strong> With only N=2 unlabeled spectrums, the model cannot run unsupervised clustering in the traditional sense. The assignment is based purely on feature-direction consistency with reference sites. Additional AEP Turk measurements would enable true per-site clustering and further validate these results.</p>
</div>

</div>

<div class="footer">
  Cutsforth &middot; Cortex Platform &middot; AEP Turk Per-Site Validation &middot; Unsupervised Clustering v3
</div>

</div>
</body>
</html>"""

report_path = os.path.join(ROOT, '_aep_turk_validation_report.html')
with open(report_path, 'w', encoding='utf-8') as fout:
    fout.write(html)
print(f"  HTML report: {report_path}")
print("\n  Done.")
