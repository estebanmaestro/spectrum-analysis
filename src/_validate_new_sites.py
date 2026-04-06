"""
Multi-Site ON/OFF Unsupervised Analysis (v3 — Independent Per-Site/Asset)
=========================================================================
Each physical unit/asset is clustered INDEPENDENTLY:
  1. Zip-based sites split by unit: VAN1, HARQCT1, HARQCT2, ...
  2. XLSM sites split by asset: Towantic_GT1, Towantic_GT2, Towantic_STG, ...
  3. Best method selected by intrinsic silhouette score (no cross-site reference)
  4. ON/OFF assigned by total_power: cluster with higher mean total_power = ON
  5. PowerGap method (1D log-power gap split) handles flat-line ON spectrums
     that confuse shape-based clustering

Well-tested sites: Decatur, Har_CT1, VAN1.

Power computation: V² (squared µV amplitude).
GSU spectrum columns are excluded from XLSM files.
Results are saved as a pickle bundle for the interactive dashboard.
"""
import os, re, zipfile, io, warnings, pickle
import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM, SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples, f1_score, precision_score, recall_score, accuracy_score
from itertools import permutations
from collections import OrderedDict

warnings.filterwarnings('ignore')
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'On_Off_data')

BAND_DEFS = [('B1', 30000, 300000), ('B2', 300000, 3000000),
             ('B3', 3000000, 30000000), ('B4', 30000000, 100000000)]

METHODS = ['KMeans', 'GMM', 'Spectral', 'Agglomerative', 'DBSCAN', 'OneClassSVM', 'MMC']

ZIP_PATTERNS = [
    re.compile(r'^(Van\d+)_(\w+?)_(on|off)_(\d{4}_\d{2}_\d{2})\.zip$', re.I),
    re.compile(r'^(Harq\w+?)_(\w+?)_(on|off)_(\d{4}_\d{2}_\d{2})\.zip$', re.I),
]
AEP_CSV_PATTERN = re.compile(r'^AEP_Turk_Gen_Spectrum_(\d+)\.csv$', re.I)

XLSM_SITES = OrderedDict([
    ('Towantic', 'Towantic Spectrum Compare Gen NG.xlsm'),
    ('Har_CT1', 'Har CT 1 GEN Spectrum Compare.xlsm'),
    ('EMSA_GT2', 'EMSA Spectrum Compare Towatic GT2.xlsm'),
    ('EMSA_GT1', 'EMSA Spectrum Compare Towatic GT1.xlsm'),
    ('Decatur', 'Decatur_EMI Spectrum Compare.xlsm'),
])

NEW_BATCH_DIR = os.path.join(DATA_DIR, 'new_batch')
GAPAC_TG_RE = re.compile(r'^(TG\d+)')

NEW_BATCH_SITES = OrderedDict([
    ('Prairie_State_2', '2024_02_15_Prairie State 2 EMSA Gen Compare.xlsm'),
    ('Middletown_GT', '2025_09_Middletown_GT_NGT_Power_Spectrum_Compare.xlsm'),
    ('Middletown_ST', '2025_09_Middletown_ST_NGT_Power_Spectrum_Compare.xlsm'),
    ('Van_GT1', '2025_10_15_Van_GT1_Gen_Spectrum Compare.xlsm'),
    ('Van_GT2', '2025_10_15_Van_GT2_Gen_Spectrum Compare.xlsm'),
    ('Van_GT3', '2025_10_15_Van_GT3_Gen_Spectrum Compare.xlsm'),
    ('Van_GT4', '2025_10_15_Van_GT4_Gen_Spectrum Compare.xlsm'),
    ('GAPAC', '2025_10_24_GAPAC_BM_EMI Spectrum Compare TG3_TG4_TG5.xlsm'),
    ('Harq_ST1', '2025_12_30_Harq_ST1_GEN_Spectrum Comparison.xlsm'),
    ('Millenium_GT1_step', 'EMSA Spectrum Compare GT1 Millenium step change.xlsm'),
    ('Millenium_GT1', 'EMSA Spectrum Compare GT1 Millenium.xlsm'),
    ('Millenium_ST1', 'EMSA Spectrum Compare ST1 Millenium.xlsm'),
    ('Mosaic_Riverview', 'Mosaic_Riverview_RVTG1_Gen_Spectrum_Compare_10_05_2025.xlsm'),
    ('NC2', 'NC2 EMI Gen Neu Spectrum Compare.xlsm'),
    ('OPPD_NC1', 'OPPD NC1 2025_11_14_GEN_Spectrum_Compare.xlsx'),
    ('SQN2', 'SQN2 EMI Spectrum Compare 2025_06_02.xlsm'),
    ('WBN2', 'WBN Unit 2 Generator EMI Spectrum Compare.xlsm'),
])


def compute_features(freqs, powers_uv):
    f = {}
    pwr = powers_uv ** 2
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
    rms = np.sqrt(np.mean(powers_uv ** 2))
    f['crest_factor'] = powers_uv.max() / (rms + 1e-10)
    f['amplitude_cv'] = np.std(powers_uv) / (np.mean(powers_uv) + 1e-10)
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
    for perm in permutations([0, 1]):
        mapping = {old: new for old, new in zip(sorted(np.unique(yp)), perm)}
        mapped = np.array([mapping.get(p, 0) for p in yp])
        f = f1_score(yt, mapped, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_m = mapped
    return (precision_score(yt, best_m, zero_division=0),
            recall_score(yt, best_m, zero_division=0),
            best_f1, accuracy_score(yt, best_m), best_m)


def extract_asset(spectrum_id):
    """Extract asset identifier from spectrum column header."""
    parts = spectrum_id.strip().split()
    if len(parts) >= 2:
        return ' '.join(parts[1:])
    return 'default'


def _should_exclude_col(col):
    """Exclude GSU and PT measurement columns."""
    words = str(col).upper().split()
    return any(w in ('GSU', 'PT') for w in words)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_labeled_zips(raw_store):
    records = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith('.zip'):
            continue
        for pat in ZIP_PATTERNS:
            m = pat.match(fname)
            if m:
                unit = m.group(1).upper()
                site = unit
                state = m.group(3).lower()
                date_str = m.group(4).replace('_', '-')
                sid = f'{unit}_{state}_{date_str}'
                with zipfile.ZipFile(os.path.join(DATA_DIR, fname)) as z:
                    with z.open('ChartData.xlsx') as xf:
                        df = pd.read_excel(io.BytesIO(xf.read()))
                freqs = pd.to_numeric(df.iloc[1:, 0], errors='coerce')
                powers = pd.to_numeric(df.iloc[1:, 1], errors='coerce')
                valid = freqs.notna() & powers.notna()
                raw_store[(site, sid)] = (freqs[valid].values, powers[valid].values)
                feats = compute_features(freqs[valid].values, powers[valid].values)
                feats.update({'spectrum_id': sid, 'asset': unit,
                              'state': state, 'site': site})
                records.append(feats)
                break
    return records


def load_aep_turk(raw_store):
    records = []
    for fname in sorted(os.listdir(DATA_DIR)):
        m = AEP_CSV_PATTERN.match(fname)
        if not m:
            continue
        spectrum_num = m.group(1)
        df = pd.read_csv(os.path.join(DATA_DIR, fname), skiprows=1)
        freqs = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        powers = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        valid = freqs.notna() & powers.notna()
        sid = f'AEP_Turk_S{spectrum_num}'
        raw_store[('AEP_Turk', sid)] = (freqs[valid].values, powers[valid].values)
        feats = compute_features(freqs[valid].values, powers[valid].values)
        feats.update({'spectrum_id': sid, 'asset': 'AEP_Turk',
                      'state': 'unknown', 'site': 'AEP_Turk'})
        records.append(feats)
    return records


def load_xlsm_site(site_name, filename, raw_store):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_excel(path, sheet_name='EMSASweep')
    freqs = df['Hz'].values.astype(float)
    records = []
    for col in df.columns:
        if col in ('Hz', 'MHz'):
            continue
        if df[col].isna().all():
            continue
        if _should_exclude_col(col):
            continue
        if site_name.startswith('EMSA') and 'bkr' in str(col).lower():
            continue
        powers_uv = pd.to_numeric(df[col], errors='coerce').values
        valid = np.isfinite(freqs) & np.isfinite(powers_uv)
        asset = extract_asset(col)
        effective_site = f"{site_name}_{asset}" if asset != 'default' else site_name
        raw_store[(effective_site, col)] = (freqs[valid], powers_uv[valid])
        feats = compute_features(freqs[valid], powers_uv[valid])
        feats.update({'spectrum_id': col, 'asset': asset,
                      'state': 'unknown', 'site': effective_site})
        records.append(feats)
    return records


def load_new_batch_site(site_name, filename, raw_store):
    """Load spectrums from new_batch XLSM/XLSX files.
    GAPAC is split by TG prefix; all other files stay as one site.
    """
    path = os.path.join(NEW_BATCH_DIR, filename)
    df = pd.read_excel(path, sheet_name='EMSASweep')
    freqs = df['Hz'].values.astype(float)
    records = []
    for col in df.columns:
        if col in ('Hz', 'MHz'):
            continue
        if df[col].isna().all():
            continue
        if _should_exclude_col(col):
            continue
        powers_uv = pd.to_numeric(df[col], errors='coerce').values
        valid = np.isfinite(freqs) & np.isfinite(powers_uv)
        if valid.sum() == 0:
            continue
        # GAPAC: split by TG prefix (TG3, TG4, TG5)
        if site_name == 'GAPAC':
            m = GAPAC_TG_RE.match(str(col))
            tg = m.group(1) if m else 'unknown'
            effective_site = f"GAPAC_{tg}"
            asset = tg
        else:
            effective_site = site_name
            asset = site_name
        raw_store[(effective_site, col)] = (freqs[valid], powers_uv[valid])
        feats = compute_features(freqs[valid], powers_uv[valid])
        feats.update({'spectrum_id': col, 'asset': asset,
                      'state': 'unknown', 'site': effective_site})
        records.append(feats)
    return records


# ═══════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("=" * 80)
print("  Multi-Site ON/OFF Analysis (v2 — Independent Per-Site)")
print("=" * 80)

print("\n[1/4] Loading data...")
raw_spectrums = {}
all_records = []

labeled_zip_records = load_labeled_zips(raw_spectrums)
all_records.extend(labeled_zip_records)
zip_sites = sorted(set(r['site'] for r in labeled_zip_records))
print(f"  Loaded {len(labeled_zip_records)} labeled zip spectrums: {', '.join(zip_sites)}")

aep_records = load_aep_turk(raw_spectrums)
all_records.extend(aep_records)
if aep_records:
    print(f"  Loaded {len(aep_records)} AEP Turk spectrums")

for site_name, filename in XLSM_SITES.items():
    site_records = load_xlsm_site(site_name, filename, raw_spectrums)
    all_records.extend(site_records)
    effective_sites = sorted(set(r['site'] for r in site_records))
    if len(effective_sites) > 1:
        parts = ', '.join(f"{es}({sum(1 for r in site_records if r['site'] == es)})"
                          for es in effective_sites)
        print(f"  Loaded {len(site_records)} spectrums from {site_name}: {parts}")
    else:
        print(f"  Loaded {len(site_records)} spectrums for {effective_sites[0]}")

print("\n  --- new_batch ---")
for site_name, filename in NEW_BATCH_SITES.items():
    fpath = os.path.join(NEW_BATCH_DIR, filename)
    if not os.path.exists(fpath):
        print(f"  SKIP (not found): {filename}")
        continue
    site_records = load_new_batch_site(site_name, filename, raw_spectrums)
    all_records.extend(site_records)
    effective_sites = sorted(set(r['site'] for r in site_records))
    if len(effective_sites) > 1:
        parts = ', '.join(f"{es}({sum(1 for r in site_records if r['site'] == es)})"
                          for es in effective_sites)
        print(f"  Loaded {len(site_records)} spectrums from {site_name}: {parts}")
    elif site_records:
        print(f"  Loaded {len(site_records)} spectrums for {effective_sites[0]}")

data = pd.DataFrame(all_records)
data['running'] = data['state'].map({'on': 1, 'off': 0}).fillna(-1).astype(int)
meta_cols = ['spectrum_id', 'asset', 'state', 'site', 'running']
feature_cols = [c for c in data.columns if c not in meta_cols]

all_site_names = sorted(data['site'].unique())
print(f"\n  Total: {len(data)} samples, {len(feature_cols)} features, {len(all_site_names)} sites")
for sn in all_site_names:
    mk = data['site'] == sn
    n = mk.sum()
    assets = sorted(data.loc[mk, 'asset'].unique())
    print(f"  {sn}: {n} spectrums, assets={assets}")

# ═══════════════════════════════════════════════════════════════
# FEATURE SUBSETS (consistent across all sites)
# ═══════════════════════════════════════════════════════════════
SCALE_INVARIANT = [c for c in feature_cols if any(k in c for k in
    ['frac', 'ratio', 'hi_lo', 'spectral', 'entropy', 'flatness', 'slope',
     'rolloff', 'crest', 'peak_to_mean', 'peak_freq', 'gradient', 'range',
     'amplitude_cv'])]

ALL_SUBSETS = OrderedDict([
    ('Band Powers', ['total_power', 'B1_power', 'B2_power', 'B3_power', 'B4_power']),
    ('Scale-Invariant', SCALE_INVARIANT),
    ('All Features', feature_cols),
])

# ═══════════════════════════════════════════════════════════════
# [2/4] PER-SITE CLUSTERING — fully independent
# ═══════════════════════════════════════════════════════════════
print("\n[2/4] Per-site unsupervised clustering (independent)...\n")

site_assignments = {}

for sn in all_site_names:
    mk = data['site'] == sn
    site_df = data.loc[mk]
    n_site = mk.sum()
    has_labels = (site_df['running'] >= 0).all()
    log_tp = np.log1p(site_df['total_power'].values.astype(float))

    best_sil = -2
    best_labels = None
    best_cfg = ''
    best_Xs = None

    for ss_name, feat_list in ALL_SUBSETS.items():
        vf = [c for c in feat_list if c in data.columns]
        Xr = data.loc[mk, vf].values.astype(float)
        Xs = preprocess(Xr)

        for method in METHODS:
            try:
                labels = run_clustering(Xs, method)
                unique_labels = np.unique(labels)
                if len(unique_labels) < 2:
                    continue
                if n_site >= 3:
                    sil = silhouette_score(Xs, labels)
                else:
                    sil = 0.0
                if sil > best_sil:
                    best_sil = sil
                    best_labels = labels
                    best_cfg = f"{method} + {ss_name}"
                    best_Xs = Xs
            except Exception:
                continue

    # PowerGap: 1D split at largest gap in log(total_power).
    pg_result = None
    if n_site >= 2:
        sorted_idx = np.argsort(log_tp)
        sorted_lp = log_tp[sorted_idx]
        gaps = np.diff(sorted_lp)
        gap_pos = int(np.argmax(gaps))
        pg_labels = np.zeros(n_site, dtype=int)
        for i in range(gap_pos + 1, len(sorted_idx)):
            pg_labels[sorted_idx[i]] = 1
        if len(np.unique(pg_labels)) == 2:
            pg_sil = silhouette_score(
                log_tp.reshape(-1, 1), pg_labels) if n_site >= 3 else 0.5
            pg_result = (pg_labels.copy(), pg_sil)
            if pg_sil > best_sil:
                best_sil = pg_sil
                best_labels = pg_labels
                best_cfg = "PowerGap (log_total_power)"
                best_Xs = log_tp.reshape(-1, 1)

    if best_labels is None:
        Xf = preprocess(data.loc[mk, ['total_power']].values.astype(float))
        best_labels = KMeans(2, n_init=20, random_state=42).fit_predict(Xf)
        best_cfg = "KMeans + total_power (fallback)"
        best_sil = 0.0
        best_Xs = Xf

    best_labels = np.asarray(best_labels).ravel()
    cluster_ids = sorted(np.unique(best_labels))
    cluster_power = {}
    for cid in cluster_ids:
        cidx = np.where(best_labels == cid)[0]
        cluster_power[cid] = data.loc[mk, 'total_power'].iloc[cidx].mean()
    on_cluster = max(cluster_power, key=cluster_power.get)

    # Power-coherence validation: if the winning method's ON/OFF assignment
    # overlaps significantly in power space, fall back to PowerGap.
    if len(cluster_ids) == 2 and pg_result is not None:
        on_pw = site_df['total_power'].values[best_labels == on_cluster]
        off_pw = site_df['total_power'].values[best_labels != on_cluster]
        if len(on_pw) > 0 and len(off_pw) > 0 and off_pw.max() > on_pw.min():
            n_overlap = int((on_pw < off_pw.max()).sum())
            if n_overlap / len(on_pw) > 0.15:
                pg_lbl, pg_s = pg_result
                best_labels = pg_lbl
                best_sil = pg_s
                best_cfg = "PowerGap (coherence override)"
                best_Xs = log_tp.reshape(-1, 1)
                cluster_ids = sorted(np.unique(best_labels))
                cluster_power = {}
                for cid in cluster_ids:
                    cidx = np.where(best_labels == cid)[0]
                    cluster_power[cid] = data.loc[mk, 'total_power'].iloc[cidx].mean()
                on_cluster = max(cluster_power, key=cluster_power.get)

    assignment = {}
    for i in range(n_site):
        sid = site_df.iloc[i]['spectrum_id']
        assignment[sid] = 'ON' if best_labels[i] == on_cluster else 'OFF'

    # --- Separability Score (0–100) ---
    # Actual gap between the weakest ON and the strongest OFF in
    # log(total_power).  Directly measures: "is there clear daylight
    # between the two groups, or do they intermix?"
    #   actual_gap ≈ 4  →  ~55x power ratio, zero overlap   →  100
    #   actual_gap ≈ 0  →  clusters barely touch / overlap   →    0
    on_mask = np.array([assignment[site_df.iloc[i]['spectrum_id']] == 'ON'
                        for i in range(n_site)])
    on_log = log_tp[on_mask]
    off_log = log_tp[~on_mask]
    n_on = int(on_mask.sum())
    n_off = n_site - n_on

    if n_on >= 1 and n_off >= 1:
        actual_gap = float(on_log.min() - off_log.max())
        clear_gap = max(actual_gap, 0.0)
    else:
        actual_gap = 0.0
        clear_gap = 0.0
    separability = round(min(clear_gap / 4.0, 1.0) * 100, 1)

    confidence = {
        'separability': separability,
        'actual_gap': round(actual_gap, 2),
        'n_on': n_on,
        'n_off': n_off,
    }

    # --- Per-sample confidence (numeric 0–100) ---
    # Based on power margin: how far the sample is from the decision boundary
    # in log-power, normalized by the half-gap between cluster means.
    if len(on_log) > 0 and len(off_log) > 0:
        boundary = (on_log.mean() + off_log.mean()) / 2.0
        half_gap = abs(on_log.mean() - off_log.mean()) / 2.0
    else:
        boundary = log_tp.mean()
        half_gap = 1.0

    sample_confidence = {}
    for i in range(n_site):
        sid = site_df.iloc[i]['spectrum_id']
        margin = abs(log_tp[i] - boundary) / (half_gap + 1e-10)
        score = round(min(margin, 2.0) / 2.0 * 100, 1)
        sample_confidence[sid] = {
            'power_margin': round(float(margin), 2),
            'score': score,
        }

    f1_val = None
    if has_labels:
        ys = site_df['running'].values
        pred = np.array([1 if assignment[sid] == 'ON' else 0
                         for sid in site_df['spectrum_id'].values])
        _, _, f1_val, _, _ = match_labels(ys, pred)

    site_assignments[sn] = {
        'assignment': assignment,
        'silhouette': best_sil,
        'config': best_cfg,
        'labels': best_labels,
        'f1': f1_val,
        'confidence': confidence,
        'sample_confidence': sample_confidence,
    }

    on_c = sum(1 for v in assignment.values() if v == 'ON')
    off_c = sum(1 for v in assignment.values() if v == 'OFF')
    f1_str = f", F1={f1_val:.3f}" if f1_val is not None else ""
    print(f"  {sn} (N={n_site}): {best_cfg}, sil={best_sil:.3f}{f1_str}")
    print(f"    => {on_c} ON / {off_c} OFF  "
          f"[Separability: {separability}]")

# Write inferred state back into the dataframe
for sn in all_site_names:
    mk = data['site'] == sn
    for idx in data.loc[mk].index:
        sid = data.loc[idx, 'spectrum_id']
        inferred = site_assignments[sn]['assignment'][sid]
        if data.loc[idx, 'state'] == 'unknown':
            data.loc[idx, 'state'] = inferred.lower()
            data.loc[idx, 'running'] = 1 if inferred == 'ON' else 0

# ═══════════════════════════════════════════════════════════════
# [3/4] CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  SUMMARY — Independent Per-Site Analysis")
print("=" * 80)
for sn in all_site_names:
    sa = site_assignments[sn]
    conf = sa['confidence']
    mk = data['site'] == sn
    n_s = mk.sum()
    on_c = sum(1 for v in sa['assignment'].values() if v == 'ON')
    off_c = sum(1 for v in sa['assignment'].values() if v == 'OFF')
    f1_str = f"  F1={sa['f1']:.3f}" if sa['f1'] is not None else ""
    sep = conf['separability']
    print(f"\n  {sn} (N={n_s}): {sa['config']}  sil={sa['silhouette']:.3f}{f1_str}")
    print(f"    {on_c} ON / {off_c} OFF  [Separability: {sep}]")
    for sid, state in sorted(sa['assignment'].items()):
        sc = sa['sample_confidence'].get(sid, {})
        sc_score = sc.get('score', 0)
        print(f"      {sid:<35s} => {state:<4s} (score: {sc_score:>5.1f})")

# ═══════════════════════════════════════════════════════════════
# [4/4] SAVE RESULTS BUNDLE
# ═══════════════════════════════════════════════════════════════
print("\n[4/4] Saving results bundle...")

bundle = {
    'data': data,
    'feature_cols': feature_cols,
    'raw_spectrums': raw_spectrums,
    'site_assignments': site_assignments,
    'all_site_names': all_site_names,
}
bundle_path = os.path.join(ROOT, '_results_bundle.pkl')
with open(bundle_path, 'wb') as f:
    pickle.dump(bundle, f)
print(f"  Saved: {bundle_path}")
print("\n  Done. Run 'python src/_dashboard.py' to launch the interactive UI.")
