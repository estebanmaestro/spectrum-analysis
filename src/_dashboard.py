"""
Interactive ON/OFF Spectrum Dashboard
=====================================
Two-tab Dash/Plotly app:
  Tab 1 — Clustering Analysis: per-site unsupervised clustering with manual overrides.
  Tab 2 — Scoring Analysis: reads decision_criteria.csv, graphs spectrums colored
          by true label or 5-feature voting prediction, with manual override + export.

Usage:
    python src/_dashboard.py
    Then open http://127.0.0.1:8050 in a browser.
"""
import os, pickle
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import (Dash, html, dcc, dash_table, callback,
                  Output, Input, State, no_update)
from sklearn.metrics import silhouette_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(ROOT, '_results_bundle.pkl')
DECISION_CSV = os.path.join(ROOT, 'data', 'decision_criteria.csv')

with open(BUNDLE_PATH, 'rb') as f:
    bundle = pickle.load(f)

data = bundle['data']
raw_spectrums = bundle['raw_spectrums']
site_assignments = bundle['site_assignments']
all_site_names = bundle['all_site_names']


def _load_decision_criteria():
    df = pd.read_csv(DECISION_CSV)
    df['spectrum_id'] = df['spectrum_id'].astype(str).str.strip()
    df['site'] = df['site'].astype(str).str.strip()
    return df


dc_data = _load_decision_criteria()
dc_sites = sorted(dc_data['site'].unique())


def _build_dc_key_map():
    """Build mapping from CSV (site, spectrum_id) → actual raw_spectrums key.
    Handles truncated suffixes and whitespace differences.
    """
    import re
    _ws = re.compile(r'\s+')
    key_map = {}
    site_buckets = {}
    for k in raw_spectrums:
        site_buckets.setdefault(k[0], []).append(k)

    for _, row in dc_data.iterrows():
        csv_site, csv_sid = row['site'], row['spectrum_id']
        csv_key = (csv_site, csv_sid)
        if csv_key in raw_spectrums:
            key_map[csv_key] = csv_key
            continue
        candidates = site_buckets.get(csv_site, [])
        norm_sid = _ws.sub(' ', csv_sid)
        matched = False
        for cand in candidates:
            cand_sid = cand[1]
            if cand_sid.startswith(csv_sid):
                key_map[csv_key] = cand
                matched = True
                break
            if _ws.sub(' ', cand_sid) == norm_sid:
                key_map[csv_key] = cand
                matched = True
                break
            if _ws.sub(' ', cand_sid).startswith(norm_sid):
                key_map[csv_key] = cand
                matched = True
                break
        if not matched:
            key_map[csv_key] = None
    return key_map


dc_key_map = _build_dc_key_map()
_n_resolved = sum(1 for v in dc_key_map.values() if v is not None)
print(f"  Decision criteria: {len(dc_data)} rows, {_n_resolved} matched to raw spectrums")

CLUSTER_DEFAULT = 'default'
CLUSTER_TOTAL_POWER = 'total_power'
CLUSTER_BAND_LIMITED = 'band_limited'

CLUSTER_OPTIONS = [
    {'label': 'Default (saved pipeline)', 'value': CLUSTER_DEFAULT},
    {'label': 'Total power — largest gap in log-power', 'value': CLUSTER_TOTAL_POWER},
    {'label': 'Band power — <1 MHz + ≥30 MHz (largest gap)', 'value': CLUSTER_BAND_LIMITED},
]

# ═══════════════════════════════════════════════════════════════
# CLUSTERING HELPERS (Tab 1)
# ═══════════════════════════════════════════════════════════════

def _powergap_binary_labels(log_vals):
    log_vals = np.asarray(log_vals, dtype=float)
    n = len(log_vals)
    if n < 2:
        return None
    sorted_idx = np.argsort(log_vals)
    sorted_lp = log_vals[sorted_idx]
    gaps = np.diff(sorted_lp)
    gap_pos = int(np.argmax(gaps))
    labels = np.zeros(n, dtype=int)
    for i in range(gap_pos + 1, n):
        labels[sorted_idx[i]] = 1
    if len(np.unique(labels)) < 2:
        return None
    return labels


def _assignment_from_labels(site_df, labels, linear_power):
    linear_power = np.asarray(linear_power, dtype=float)
    labels = np.asarray(labels).ravel()
    cluster_ids = sorted(np.unique(labels))
    cluster_power = {cid: linear_power[labels == cid].mean() for cid in cluster_ids}
    on_cluster = max(cluster_power, key=cluster_power.get)
    assignment = {}
    for i, sid in enumerate(site_df['spectrum_id'].values):
        assignment[sid] = 'ON' if labels[i] == on_cluster else 'OFF'
    return assignment


def _band_limited_power_uv2(freqs_hz, amps_uv):
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    pwr = np.asarray(amps_uv, dtype=float) ** 2
    mask = (freqs_hz < 1e6) | (freqs_hz >= 30e6)
    return float(pwr[mask].sum())


def _band_powers_for_site(site, site_df):
    out = []
    for sid in site_df['spectrum_id'].values:
        key = (site, sid)
        if key not in raw_spectrums:
            out.append(0.0)
            continue
        fh, av = raw_spectrums[key]
        out.append(_band_limited_power_uv2(fh, av))
    return np.array(out, dtype=float)


def _confidence_from_log_metric(site_df, assignment, log_metric):
    log_tp = np.asarray(log_metric, dtype=float)
    n_site = len(site_df)
    sids = site_df['spectrum_id'].values
    on_mask = np.array([assignment[sid] == 'ON' for sid in sids])
    on_log = log_tp[on_mask]
    off_log = log_tp[~on_mask]
    n_on, n_off = int(on_mask.sum()), n_site - int(on_mask.sum())
    if n_on >= 1 and n_off >= 1:
        actual_gap = float(on_log.min() - off_log.max())
        clear_gap = max(actual_gap, 0.0)
    else:
        actual_gap, clear_gap = 0.0, 0.0
    separability = round(min(clear_gap / 4.0, 1.0) * 100, 1)
    confidence = {'separability': separability, 'actual_gap': round(actual_gap, 2),
                  'n_on': n_on, 'n_off': n_off}
    if len(on_log) > 0 and len(off_log) > 0:
        boundary = (on_log.mean() + off_log.mean()) / 2.0
        half_gap = abs(on_log.mean() - off_log.mean()) / 2.0
    else:
        boundary, half_gap = log_tp.mean(), 1.0
    sample_confidence = {}
    for i in range(n_site):
        margin = abs(log_tp[i] - boundary) / (half_gap + 1e-10)
        sample_confidence[sids[i]] = {
            'power_margin': round(float(margin), 2),
            'score': round(min(margin, 2.0) / 2.0 * 100, 1),
        }
    return confidence, sample_confidence


def _silhouette_1d(log_metric, labels):
    arr = np.asarray(log_metric, dtype=float).reshape(-1, 1)
    lbl = np.asarray(labels).ravel()
    if len(lbl) < 3 or len(np.unique(lbl)) < 2:
        return 0.0
    try:
        return float(silhouette_score(arr, lbl))
    except Exception:
        return 0.0


def _dashboard_cluster_view(site, mode):
    if site not in site_assignments:
        return None
    sa = site_assignments[site]
    mk = data['site'] == site
    site_df = data.loc[mk]
    if len(site_df) == 0:
        return None
    if mode == CLUSTER_DEFAULT:
        return {'assignment': sa['assignment'], 'config': sa['config'],
                'silhouette': sa['silhouette'],
                'confidence': sa.get('confidence', {}),
                'sample_confidence': sa.get('sample_confidence', {})}
    if mode in (CLUSTER_TOTAL_POWER, CLUSTER_BAND_LIMITED):
        if mode == CLUSTER_TOTAL_POWER:
            linear = site_df['total_power'].values.astype(float)
            cfg = 'Total power (PowerGap)'
        else:
            linear = _band_powers_for_site(site, site_df)
            cfg = 'Band power <1 MHz + ≥30 MHz (PowerGap)'
        log_m = np.log1p(np.maximum(linear, 0.0))
        labels = _powergap_binary_labels(log_m)
        if labels is None:
            labels = np.zeros(len(site_df), dtype=int)
            labels[len(labels) // 2:] = 1
        assignment = _assignment_from_labels(site_df, labels, linear)
        conf, sc = _confidence_from_log_metric(site_df, assignment, log_m)
        sil = _silhouette_1d(log_m, labels)
        return {'assignment': assignment, 'config': cfg, 'silhouette': sil,
                'confidence': conf, 'sample_confidence': sc}
    return None


def _views_for_sites(sites, mode):
    return {s: v for s in sites if (v := _dashboard_cluster_view(s, mode)) is not None}


# ═══════════════════════════════════════════════════════════════
# PALETTES & COLOR HELPERS
# ═══════════════════════════════════════════════════════════════

SITE_PALETTE = [
    '#1D4ED8', '#047857', '#7C3AED', '#DC2626', '#B45309',
    '#0891B2', '#BE185D', '#D97706', '#059669', '#6D28D9',
    '#0369A1', '#A21CAF', '#C2410C', '#15803D', '#1E40AF',
    '#9333EA', '#B91C1C', '#0E7490', '#A16207', '#4338CA',
    '#DB2777', '#065F46', '#7E22CE', '#EA580C', '#1D4ED8',
    '#047857', '#7C3AED', '#DC2626', '#B45309', '#0891B2',
]

COLOR_BY_STATE = 'running_state'
COLOR_BY_SITE = 'site'
COLOR_BY_ORG = 'org'

COLOR_BY_OPTIONS = [
    {'label': 'Running State (ON/OFF)', 'value': COLOR_BY_STATE},
    {'label': 'Site', 'value': COLOR_BY_SITE},
    {'label': 'Organization (first 3 letters)', 'value': COLOR_BY_ORG},
]

_all_orgs = sorted(set(s[:3] for s in all_site_names))
ORG_PALETTE = SITE_PALETTE[:15]


def _org_color(site_name):
    org = site_name[:3]
    try:
        idx = _all_orgs.index(org)
    except ValueError:
        idx = 0
    return ORG_PALETTE[idx % len(ORG_PALETTE)]


def _site_color(site, selected_sites):
    try:
        idx = list(selected_sites).index(site)
    except ValueError:
        idx = 0
    return SITE_PALETTE[idx % len(SITE_PALETTE)]


def _resolve_color(site, inferred, sites, color_by, is_override=False):
    if color_by == COLOR_BY_STATE:
        if is_override:
            return '#F87171' if inferred == 'ON' else '#93C5FD'
        return '#DC2626' if inferred == 'ON' else '#2563EB'
    if color_by == COLOR_BY_ORG:
        return _org_color(site)
    return _site_color(site, sites)


def _on_off_color(state, muted=False):
    if muted:
        return '#F87171' if state == 'ON' else '#93C5FD'
    return '#DC2626' if state == 'ON' else '#2563EB'


# ═══════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "ON/OFF Spectrum Analysis"

first_site = all_site_names[0]


def _normalize_sites(sites):
    if sites is None:
        return [first_site]
    if isinstance(sites, str):
        return [sites]
    return list(sites) if sites else [first_site]


_LABEL_STYLE = {
    'fontWeight': '600', 'fontSize': '0.85rem', 'marginBottom': '6px',
    'display': 'block', 'color': '#1a2332',
}

_TABLE_COLUMNS = [
    {'name': 'Site', 'id': 'site', 'editable': False},
    {'name': 'Spectrum ID', 'id': 'spectrum_id', 'editable': False},
    {'name': 'Cluster State', 'id': 'cluster_state', 'editable': False},
    {'name': 'Manual Override', 'id': 'manual_override', 'presentation': 'dropdown'},
    {'name': 'Score', 'id': 'score', 'type': 'numeric', 'editable': False},
    {'name': 'Total Power (µV²)', 'id': 'total_power', 'editable': False},
    {'name': 'Method', 'id': 'method', 'editable': False},
]

_SC_TABLE_COLUMNS = [
    {'name': 'Site', 'id': 'site', 'editable': False},
    {'name': 'Spectrum ID', 'id': 'spectrum_id', 'editable': False},
    {'name': 'True Label', 'id': 'running_state', 'editable': False},
    {'name': 'Prediction', 'id': 'prediction', 'editable': False},
    {'name': 'Manual Override', 'id': 'manual_override', 'presentation': 'dropdown'},
    {'name': 'Weighted Score', 'id': 'weighted_score_norm', 'type': 'numeric',
     'editable': False},
    {'name': 'Total Votes', 'id': 'total_votes', 'type': 'numeric', 'editable': False},
    {'name': 'Label Source', 'id': 'label_source', 'editable': False},
]

_OVERRIDE_DROPDOWN = {
    'manual_override': {
        'options': [{'label': '', 'value': ''},
                    {'label': 'ON', 'value': 'ON'},
                    {'label': 'OFF', 'value': 'OFF'}],
    },
}

_TABLE_STYLE_COND = [
    {'if': {'column_id': 'cluster_state', 'filter_query': '{cluster_state} = "ON"'},
     'backgroundColor': '#dcfce7', 'color': '#166534', 'fontWeight': '700'},
    {'if': {'column_id': 'cluster_state', 'filter_query': '{cluster_state} = "OFF"'},
     'backgroundColor': '#fee2e2', 'color': '#991b1b', 'fontWeight': '700'},
    {'if': {'column_id': 'manual_override', 'filter_query': '{manual_override} = "ON"'},
     'backgroundColor': '#dcfce7', 'color': '#166534', 'fontWeight': '700'},
    {'if': {'column_id': 'manual_override', 'filter_query': '{manual_override} = "OFF"'},
     'backgroundColor': '#fee2e2', 'color': '#991b1b', 'fontWeight': '700'},
    {'if': {'column_id': 'score', 'filter_query': '{score} >= 70'},
     'backgroundColor': '#d1fae5', 'color': '#065f46'},
    {'if': {'column_id': 'score', 'filter_query': '{score} < 40'},
     'backgroundColor': '#fecaca', 'color': '#991b1b'},
]

_SC_TABLE_STYLE_COND = [
    {'if': {'column_id': 'running_state', 'filter_query': '{running_state} = "ON"'},
     'backgroundColor': '#dcfce7', 'color': '#166534', 'fontWeight': '700'},
    {'if': {'column_id': 'running_state', 'filter_query': '{running_state} = "OFF"'},
     'backgroundColor': '#fee2e2', 'color': '#991b1b', 'fontWeight': '700'},
    {'if': {'column_id': 'prediction', 'filter_query': '{prediction} = "ON"'},
     'backgroundColor': '#dcfce7', 'color': '#166534', 'fontWeight': '700'},
    {'if': {'column_id': 'prediction', 'filter_query': '{prediction} = "OFF"'},
     'backgroundColor': '#fee2e2', 'color': '#991b1b', 'fontWeight': '700'},
    {'if': {'column_id': 'manual_override', 'filter_query': '{manual_override} = "ON"'},
     'backgroundColor': '#dcfce7', 'color': '#166534', 'fontWeight': '700'},
    {'if': {'column_id': 'manual_override', 'filter_query': '{manual_override} = "OFF"'},
     'backgroundColor': '#fee2e2', 'color': '#991b1b', 'fontWeight': '700'},
]


def _make_card(title, value, color='#1a2332'):
    return html.Div(style={
        'background': 'white', 'borderRadius': '8px', 'padding': '16px 24px',
        'boxShadow': '0 1px 8px rgba(0,0,0,0.06)', 'border': '1px solid #e2e8f0',
        'flex': '1', 'minWidth': '140px',
    }, children=[
        html.Div(title, style={'fontSize': '0.75rem', 'color': '#78909c',
                                'fontWeight': '600', 'textTransform': 'uppercase',
                                'letterSpacing': '0.04em'}),
        html.Div(value, style={'fontSize': '1.3rem', 'fontWeight': '700',
                                'color': color, 'marginTop': '4px'}),
    ])


_CARD_STYLE = {
    'background': 'white', 'borderRadius': '8px', 'padding': '20px',
    'boxShadow': '0 1px 8px rgba(0,0,0,0.06)',
    'border': '1px solid #e2e8f0', 'marginBottom': '24px',
}


# ═══════════════════════════════════════════════════════════════
# TAB 1 LAYOUT — Clustering Analysis
# ═══════════════════════════════════════════════════════════════

def _tab1_layout():
    return html.Div(style={'padding': '24px 0'}, children=[
        html.Div(style={
            'display': 'flex', 'gap': '20px', 'marginBottom': '20px',
            'flexWrap': 'wrap', 'alignItems': 'flex-start',
            'justifyContent': 'space-between',
        }, children=[
            html.Div(style={'flex': '1 1 320px', 'minWidth': '260px', 'maxWidth': '520px'},
                     children=[
                html.Label("Sites (multi-select)", style=_LABEL_STYLE),
                dcc.Dropdown(id='site-filter',
                             options=[{'label': s, 'value': s} for s in all_site_names],
                             value=[first_site], multi=True,
                             placeholder="Select one or more sites...",
                             style={'fontSize': '0.85rem'}),
            ]),
            html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '16px',
                             'alignItems': 'flex-start', 'flex': '0 1 auto'}, children=[
                html.Div(style={'minWidth': '160px', 'width': '180px'}, children=[
                    html.Label("Running State", style=_LABEL_STYLE),
                    dcc.Dropdown(id='state-filter',
                                 options=[{'label': 'All', 'value': 'all'},
                                          {'label': 'ON', 'value': 'ON'},
                                          {'label': 'OFF', 'value': 'OFF'}],
                                 value='all', clearable=False, style={'fontSize': '0.85rem'}),
                ]),
                html.Div(style={'minWidth': '220px', 'width': '280px'}, children=[
                    html.Label("Clustering view", style=_LABEL_STYLE),
                    dcc.Dropdown(id='clustering-mode', options=CLUSTER_OPTIONS,
                                 value=CLUSTER_DEFAULT, clearable=False,
                                 style={'fontSize': '0.82rem'}),
                ]),
                html.Div(style={'minWidth': '200px', 'width': '240px'}, children=[
                    html.Label("Color by", style=_LABEL_STYLE),
                    dcc.Dropdown(id='color-by', options=COLOR_BY_OPTIONS,
                                 value=COLOR_BY_STATE, clearable=False,
                                 style={'fontSize': '0.85rem'}),
                ]),
                html.Div(id='separability-slot', style={'minWidth': '132px', 'flexShrink': 0}),
            ]),
        ]),
        html.Div(id='info-cards', style={
            'display': 'flex', 'gap': '16px', 'marginBottom': '24px', 'flexWrap': 'wrap'}),
        html.Div(style=_CARD_STYLE, children=[
            dcc.Graph(id='spectrum-chart', style={'height': '560px'})]),
        html.Div(style=_CARD_STYLE, children=[
            html.Label("Select Spectrums to Display", style={
                'fontWeight': '600', 'fontSize': '0.9rem', 'marginBottom': '10px',
                'display': 'block', 'color': '#1a2332'}),
            dcc.Checklist(id='spectrum-selector', inline=True,
                          style={'fontSize': '0.82rem', 'display': 'flex',
                                 'flexWrap': 'wrap', 'gap': '6px 16px'},
                          inputStyle={'marginRight': '4px'}),
        ]),
        html.Div(style={**_CARD_STYLE, 'marginBottom': '0'}, children=[
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between',
                             'alignItems': 'center', 'marginBottom': '12px'}, children=[
                html.H3("Spectrum Assignments", style={
                    'margin': '0', 'fontSize': '1rem', 'color': '#1a2332'}),
                html.Button("Export Running States", id='export-btn', n_clicks=0,
                            style={'background': '#005662', 'color': 'white',
                                   'border': 'none', 'borderRadius': '6px',
                                   'padding': '8px 20px', 'fontSize': '0.85rem',
                                   'fontWeight': '600', 'cursor': 'pointer'}),
            ]),
            html.Div(style={'fontSize': '0.78rem', 'color': '#64748b', 'marginBottom': '12px'},
                     children="Use the Manual Override column to correct labels. "
                              "Export merges cluster results with your overrides."),
            dash_table.DataTable(
                id='assignment-datatable', columns=_TABLE_COLUMNS, data=[],
                editable=True, dropdown=_OVERRIDE_DROPDOWN, page_size=50,
                sort_action='native', filter_action='native',
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '8px 14px',
                            'fontSize': '0.82rem',
                            'fontFamily': "'Inter', -apple-system, sans-serif"},
                style_header={'background': '#e8f5f7', 'color': '#005662',
                              'fontWeight': '600', 'borderBottom': '2px solid #d4dde3',
                              'whiteSpace': 'nowrap'},
                style_data_conditional=_TABLE_STYLE_COND),
        ]),
        dcc.Store(id='manual-overrides', storage_type='session'),
        dcc.Download(id='download-labels'),
    ])


# ═══════════════════════════════════════════════════════════════
# TAB 2 LAYOUT — Scoring Analysis
# ═══════════════════════════════════════════════════════════════

def _tab2_layout():
    return html.Div(style={'padding': '24px 0'}, children=[
        html.Div(style={
            'display': 'flex', 'gap': '20px', 'marginBottom': '20px',
            'flexWrap': 'wrap', 'alignItems': 'flex-start',
        }, children=[
            html.Div(style={'flex': '1 1 320px', 'minWidth': '260px', 'maxWidth': '520px'},
                     children=[
                html.Label("Sites (multi-select)", style=_LABEL_STYLE),
                dcc.Dropdown(id='sc-site-filter',
                             options=[{'label': s, 'value': s} for s in dc_sites],
                             value=[dc_sites[0]] if dc_sites else [],
                             multi=True, placeholder="Select sites...",
                             style={'fontSize': '0.85rem'}),
            ]),
            html.Div(style={'minWidth': '200px', 'width': '240px'}, children=[
                html.Label("Color spectrums by", style=_LABEL_STYLE),
                dcc.Dropdown(id='sc-color-mode',
                             options=[{'label': 'True Label', 'value': 'true_label'},
                                      {'label': 'Prediction', 'value': 'prediction'}],
                             value='true_label', clearable=False,
                             style={'fontSize': '0.85rem'}),
            ]),
            html.Div(style={'minWidth': '160px', 'width': '180px'}, children=[
                html.Label("Running State", style=_LABEL_STYLE),
                dcc.Dropdown(id='sc-state-filter',
                             options=[{'label': 'All', 'value': 'all'},
                                      {'label': 'ON', 'value': 'ON'},
                                      {'label': 'OFF', 'value': 'OFF'}],
                             value='all', clearable=False,
                             style={'fontSize': '0.85rem'}),
            ]),
        ]),
        html.Div(id='sc-info-cards', style={
            'display': 'flex', 'gap': '16px', 'marginBottom': '24px', 'flexWrap': 'wrap'}),
        html.Div(style=_CARD_STYLE, children=[
            dcc.Graph(id='sc-spectrum-chart', style={'height': '560px'})]),
        html.Div(style=_CARD_STYLE, children=[
            html.Label("Select Spectrums to Display", style={
                'fontWeight': '600', 'fontSize': '0.9rem', 'marginBottom': '10px',
                'display': 'block', 'color': '#1a2332'}),
            dcc.Checklist(id='sc-spectrum-selector', inline=True,
                          style={'fontSize': '0.82rem', 'display': 'flex',
                                 'flexWrap': 'wrap', 'gap': '6px 16px'},
                          inputStyle={'marginRight': '4px'}),
        ]),
        html.Div(style={**_CARD_STYLE, 'marginBottom': '0'}, children=[
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between',
                             'alignItems': 'center', 'marginBottom': '12px'}, children=[
                html.H3("Scoring Assignments", style={
                    'margin': '0', 'fontSize': '1rem', 'color': '#1a2332'}),
                html.Button("Export Running States", id='sc-export-btn', n_clicks=0,
                            style={'background': '#005662', 'color': 'white',
                                   'border': 'none', 'borderRadius': '6px',
                                   'padding': '8px 20px', 'fontSize': '0.85rem',
                                   'fontWeight': '600', 'cursor': 'pointer'}),
            ]),
            html.Div(style={'fontSize': '0.78rem', 'color': '#64748b', 'marginBottom': '12px'},
                     children="Use the Manual Override column to correct labels. "
                              "Export merges true labels with your overrides."),
            dash_table.DataTable(
                id='sc-assignment-datatable', columns=_SC_TABLE_COLUMNS, data=[],
                editable=True, dropdown=_OVERRIDE_DROPDOWN, page_size=50,
                sort_action='native', filter_action='native',
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '8px 14px',
                            'fontSize': '0.82rem',
                            'fontFamily': "'Inter', -apple-system, sans-serif"},
                style_header={'background': '#e8f5f7', 'color': '#005662',
                              'fontWeight': '600', 'borderBottom': '2px solid #d4dde3',
                              'whiteSpace': 'nowrap'},
                style_data_conditional=_SC_TABLE_STYLE_COND),
        ]),
        dcc.Store(id='sc-manual-overrides', storage_type='session'),
        dcc.Download(id='sc-download-labels'),
    ])


# ═══════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════

def build_layout():
    return html.Div(style={
        'fontFamily': "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
        'background': '#f5f7fa', 'minHeight': '100vh', 'padding': '0',
    }, children=[
        html.Div(style={'background': '#003d4d', 'color': 'white', 'padding': '28px 40px',
                         'borderBottom': '4px solid #00838f'}, children=[
            html.H1("ON/OFF Spectrum Analysis", style={
                'margin': '0 0 4px', 'fontSize': '1.6rem', 'fontWeight': '700'}),
            html.Div("Per-site unsupervised clustering & 5-feature scoring analysis",
                      style={'opacity': '0.7', 'fontSize': '0.9rem'}),
        ]),
        html.Div(style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '24px 32px'},
                 children=[
            dcc.Tabs(id='main-tabs', value='tab-clustering', children=[
                dcc.Tab(label='Clustering Analysis', value='tab-clustering',
                        style={'fontWeight': '600'}, selected_style={'fontWeight': '700'}),
                dcc.Tab(label='Scoring Analysis', value='tab-scoring',
                        style={'fontWeight': '600'}, selected_style={'fontWeight': '700'}),
            ]),
            html.Div(id='tab-content'),
        ]),
    ])


app.layout = build_layout()


@callback(Output('tab-content', 'children'), Input('main-tabs', 'value'))
def render_tab(tab):
    if tab == 'tab-scoring':
        return _tab2_layout()
    return _tab1_layout()


# ═══════════════════════════════════════════════════════════════
# TAB 1 CALLBACKS — Clustering Analysis
# ═══════════════════════════════════════════════════════════════

@callback(
    Output('spectrum-selector', 'options'),
    Output('spectrum-selector', 'value'),
    Input('site-filter', 'value'),
    Input('state-filter', 'value'),
    Input('clustering-mode', 'value'),
    Input('color-by', 'value'),
)
def update_spectrum_selector(sites, state_val, cluster_mode, color_by):
    sites = _normalize_sites(sites)
    mode = cluster_mode or CLUSTER_DEFAULT
    color_by = color_by or COLOR_BY_STATE
    views = _views_for_sites(sites, mode)
    options, all_values = [], []
    for site in sites:
        if site not in views:
            continue
        view = views[site]
        assignment = view['assignment']
        multi = len(sites) > 1
        mk = data['site'] == site
        filtered = data.loc[mk]
        if state_val and state_val != 'all':
            keep = [i for i in filtered.index
                    if assignment.get(filtered.loc[i, 'spectrum_id'], '') == state_val]
            filtered = filtered.loc[keep]
        for _, row in filtered.iterrows():
            sid = row['spectrum_id']
            inferred = assignment.get(sid, '?')
            tag = f"{site}::{sid}"
            prefix = f"[{site}] " if multi else ""
            color = _resolve_color(site, inferred, sites, color_by)
            options.append({'label': html.Span([
                html.Span(f"{prefix}[{inferred}] ", style={'color': color, 'fontWeight': '700'}),
                sid]), 'value': tag})
            all_values.append(tag)
    return options, all_values


@callback(Output('separability-slot', 'children'),
          Input('site-filter', 'value'), Input('clustering-mode', 'value'))
def update_separability_slot(sites, cluster_mode):
    sites = _normalize_sites(sites)
    mode = cluster_mode or CLUSTER_DEFAULT
    if len(sites) != 1:
        return []
    view = _dashboard_cluster_view(sites[0], mode)
    if view is None:
        return []
    sep = view.get('confidence', {}).get('separability', 0)
    color = '#047857' if sep >= 70 else ('#B45309' if sep >= 40 else '#DC2626')
    return [_make_card("Separability", f"{sep}", color)]


@callback(Output('info-cards', 'children'),
          Input('site-filter', 'value'), Input('clustering-mode', 'value'))
def update_info_cards(sites, cluster_mode):
    sites = _normalize_sites(sites)
    mode = cluster_mode or CLUSTER_DEFAULT
    views = _views_for_sites(sites, mode)
    total_n = total_on = total_off = 0
    for site in sites:
        if site not in views:
            continue
        total_n += (data['site'] == site).sum()
        total_on += sum(1 for v in views[site]['assignment'].values() if v == 'ON')
        total_off += sum(1 for v in views[site]['assignment'].values() if v == 'OFF')
    site_label = ', '.join(sites) if len(sites) <= 3 else f"{len(sites)} sites"
    cards = [_make_card("Sites", site_label, '#005662'),
             _make_card("Spectrums", str(total_n)),
             _make_card("ON", str(total_on), '#DC2626'),
             _make_card("OFF", str(total_off), '#2563EB')]
    if len(sites) == 1 and sites[0] in views:
        v = views[sites[0]]
        cards.append(_make_card("Method", v['config'], '#475569'))
        cards.append(_make_card("Silhouette", f"{v['silhouette']:.3f}", '#7C3AED'))
    return cards


@callback(
    Output('spectrum-chart', 'figure'),
    Input('site-filter', 'value'), Input('spectrum-selector', 'value'),
    Input('clustering-mode', 'value'), Input('color-by', 'value'),
    Input('manual-overrides', 'data'),
)
def update_spectrum_chart(sites, selected_tags, cluster_mode, color_by, overrides):
    sites = _normalize_sites(sites)
    mode = cluster_mode or CLUSTER_DEFAULT
    color_by = color_by or COLOR_BY_STATE
    overrides = overrides or {}
    views = _views_for_sites(sites, mode)
    fig = go.Figure()
    if not selected_tags:
        fig.update_layout(title="No spectrums selected",
                          xaxis_title="Frequency (MHz)", yaxis_title="Amplitude (µV)")
        return fig
    total_on = total_off = 0
    for tag in sorted(selected_tags):
        if '::' not in tag:
            continue
        site, sid = tag.split('::', 1)
        if site not in views:
            continue
        key = (site, sid)
        if key not in raw_spectrums:
            continue
        cluster_state = views[site]['assignment'].get(sid, '?')
        manual = overrides.get(tag, '')
        effective = manual if manual else cluster_state
        is_on = effective == 'ON'
        total_on += is_on
        total_off += not is_on
        freqs_hz, amps_uv = raw_spectrums[key]
        color = _resolve_color(site, effective, sites, color_by, is_override=bool(manual))
        suffix = f" ({effective}*)" if manual else f" ({effective})"
        fig.add_trace(go.Scattergl(
            x=freqs_hz / 1e6, y=amps_uv, mode='lines',
            name=f"{site} | {sid}{suffix}",
            line=dict(color=color, width=1.2, dash='solid' if is_on else 'dash'),
            opacity=0.85,
            hovertemplate=(f"<b>{site}</b> — {sid} ({effective})<br>"
                           "Freq: %{x:.3f} MHz<br>Amp: %{y:.2f} µV<extra></extra>")))
    title_sites = ', '.join(sites) if len(sites) <= 3 else f"{len(sites)} sites"
    conf_suffix = ""
    if len(sites) == 1 and sites[0] in views:
        sep = views[sites[0]].get('confidence', {}).get('separability', 0)
        conf_suffix = f" — Separability: {sep}/100"
    fig.update_layout(
        title=dict(text=f"{title_sites} — {total_on + total_off} spectrums "
                        f"({total_on} ON, {total_off} OFF){conf_suffix}", font=dict(size=15)),
        xaxis=dict(title="Frequency (MHz)", type='log', gridcolor='#eee', showgrid=True),
        yaxis=dict(title="Amplitude (µV)", type='log', gridcolor='#eee', showgrid=True),
        template='plotly_white',
        legend=dict(font=dict(size=9), orientation='v', bgcolor='rgba(255,255,255,0.85)'),
        hovermode='x unified', margin=dict(l=60, r=20, t=50, b=50))
    return fig


@callback(
    Output('assignment-datatable', 'data'),
    Input('site-filter', 'value'), Input('state-filter', 'value'),
    Input('clustering-mode', 'value'), State('manual-overrides', 'data'),
)
def update_assignment_table(sites, state_val, cluster_mode, overrides):
    sites = _normalize_sites(sites)
    mode = cluster_mode or CLUSTER_DEFAULT
    views = _views_for_sites(sites, mode)
    overrides = overrides or {}
    table_data = []
    for site in sites:
        if site not in views:
            continue
        view = views[site]
        assignment = view['assignment']
        sc_map = view.get('sample_confidence', {})
        mk = data['site'] == site
        filtered = data.loc[mk]
        if state_val and state_val != 'all':
            keep = [i for i in filtered.index
                    if assignment.get(filtered.loc[i, 'spectrum_id'], '') == state_val]
            filtered = filtered.loc[keep]
        for _, row in filtered.sort_values('spectrum_id').iterrows():
            sid = row['spectrum_id']
            table_data.append({
                'site': site, 'spectrum_id': sid,
                'cluster_state': assignment.get(sid, '?'),
                'manual_override': overrides.get(f"{site}::{sid}", ''),
                'score': sc_map.get(sid, {}).get('score', 0),
                'total_power': f"{row['total_power']:,.0f}",
                'method': view['config']})
    return table_data


@callback(Output('manual-overrides', 'data'),
          Input('assignment-datatable', 'data'),
          State('manual-overrides', 'data'), prevent_initial_call=True)
def save_manual_overrides(table_data, current):
    if not table_data:
        return current or {}
    overrides = dict(current or {})
    for row in table_data:
        key = f"{row['site']}::{row['spectrum_id']}"
        manual = row.get('manual_override', '')
        if manual:
            overrides[key] = manual
        elif key in overrides:
            del overrides[key]
    return overrides


@callback(Output('download-labels', 'data'),
          Input('export-btn', 'n_clicks'),
          State('manual-overrides', 'data'),
          State('clustering-mode', 'value'), prevent_initial_call=True)
def export_labels(n_clicks, overrides, cluster_mode):
    if not n_clicks:
        return no_update
    overrides = overrides or {}
    mode = cluster_mode or CLUSTER_DEFAULT
    rows = []
    for site in all_site_names:
        view = _dashboard_cluster_view(site, mode)
        if view is None:
            continue
        assignment = view['assignment']
        for _, row in data.loc[data['site'] == site].iterrows():
            sid = row['spectrum_id']
            manual = overrides.get(f"{site}::{sid}", '')
            rows.append({'site': site, 'spectrum_id': sid,
                         'running_state': manual if manual else assignment.get(sid, '?'),
                         'source': 'manual' if manual else 'cluster'})
    ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    return dcc.send_data_frame(pd.DataFrame(rows).to_csv,
                               f'running_state_labels_{ts}.csv', index=False)


# ═══════════════════════════════════════════════════════════════
# TAB 2 CALLBACKS — Scoring Analysis
# ═══════════════════════════════════════════════════════════════

def _sc_normalize_sites(sites):
    if not sites:
        return [dc_sites[0]] if dc_sites else []
    if isinstance(sites, str):
        return [sites]
    return list(sites)


def _sc_filtered(sites, state_val, color_mode):
    """Return filtered dc_data rows for the given sites and state filter."""
    df = dc_data[dc_data['site'].isin(sites)].copy()
    col = 'running_state' if color_mode == 'true_label' else 'prediction'
    if state_val and state_val != 'all':
        df = df[df[col] == state_val]
    return df


@callback(
    Output('sc-spectrum-selector', 'options'),
    Output('sc-spectrum-selector', 'value'),
    Input('sc-site-filter', 'value'),
    Input('sc-state-filter', 'value'),
    Input('sc-color-mode', 'value'),
)
def sc_update_selector(sites, state_val, color_mode):
    sites = _sc_normalize_sites(sites)
    color_mode = color_mode or 'true_label'
    df = _sc_filtered(sites, state_val, color_mode)
    options, values = [], []
    for _, row in df.iterrows():
        site, sid = row['site'], row['spectrum_id']
        col = 'running_state' if color_mode == 'true_label' else 'prediction'
        state = row[col]
        tag = f"{site}::{sid}"
        color = _on_off_color(state)
        multi = len(sites) > 1
        prefix = f"[{site}] " if multi else ""
        options.append({'label': html.Span([
            html.Span(f"{prefix}[{state}] ", style={'color': color, 'fontWeight': '700'}),
            sid]), 'value': tag})
        values.append(tag)
    return options, values


@callback(Output('sc-info-cards', 'children'),
          Input('sc-site-filter', 'value'), Input('sc-color-mode', 'value'))
def sc_update_info_cards(sites, color_mode):
    sites = _sc_normalize_sites(sites)
    color_mode = color_mode or 'true_label'
    df = dc_data[dc_data['site'].isin(sites)]
    col = 'running_state' if color_mode == 'true_label' else 'prediction'
    n_on = int((df[col] == 'ON').sum())
    n_off = int((df[col] == 'OFF').sum())
    site_label = ', '.join(sites) if len(sites) <= 3 else f"{len(sites)} sites"
    label_kind = "True Label" if color_mode == 'true_label' else "Prediction"
    return [
        _make_card("Sites", site_label, '#005662'),
        _make_card("Spectrums", str(len(df))),
        _make_card(f"ON ({label_kind})", str(n_on), '#DC2626'),
        _make_card(f"OFF ({label_kind})", str(n_off), '#2563EB'),
    ]


@callback(
    Output('sc-spectrum-chart', 'figure'),
    Input('sc-site-filter', 'value'), Input('sc-spectrum-selector', 'value'),
    Input('sc-color-mode', 'value'), Input('sc-manual-overrides', 'data'),
)
def sc_update_chart(sites, selected_tags, color_mode, overrides):
    sites = _sc_normalize_sites(sites)
    color_mode = color_mode or 'true_label'
    overrides = overrides or {}
    fig = go.Figure()
    if not selected_tags:
        fig.update_layout(title="No spectrums selected",
                          xaxis_title="Frequency (MHz)", yaxis_title="Amplitude (µV)")
        return fig
    dc_lookup = {(r['site'], r['spectrum_id']): dict(r) for _, r in dc_data.iterrows()}
    total_on = total_off = 0
    for tag in sorted(selected_tags):
        if '::' not in tag:
            continue
        site, sid = tag.split('::', 1)
        csv_key = (site, sid)
        raw_key = dc_key_map.get(csv_key)
        if raw_key is None or raw_key not in raw_spectrums:
            continue
        row = dc_lookup.get(csv_key, {})
        col = 'running_state' if color_mode == 'true_label' else 'prediction'
        csv_state = row.get(col, '?')
        manual = overrides.get(tag, '')
        effective = manual if manual else csv_state
        is_on = effective == 'ON'
        total_on += is_on
        total_off += not is_on
        freqs_hz, amps_uv = raw_spectrums[raw_key]
        color = _on_off_color(effective, muted=bool(manual))
        suffix = f" ({effective}*)" if manual else f" ({effective})"
        fig.add_trace(go.Scattergl(
            x=freqs_hz / 1e6, y=amps_uv, mode='lines',
            name=f"{site} | {sid}{suffix}",
            line=dict(color=color, width=1.2, dash='solid' if is_on else 'dash'),
            opacity=0.85,
            hovertemplate=(f"<b>{site}</b> — {sid} ({effective})<br>"
                           "Freq: %{x:.3f} MHz<br>Amp: %{y:.2f} µV<extra></extra>")))
    title_sites = ', '.join(sites) if len(sites) <= 3 else f"{len(sites)} sites"
    label_kind = "True Label" if color_mode == 'true_label' else "Prediction"
    fig.update_layout(
        title=dict(text=f"{title_sites} — {total_on + total_off} spectrums "
                        f"({total_on} ON, {total_off} OFF) — by {label_kind}",
                   font=dict(size=15)),
        xaxis=dict(title="Frequency (MHz)", type='log', gridcolor='#eee', showgrid=True),
        yaxis=dict(title="Amplitude (µV)", type='log', gridcolor='#eee', showgrid=True),
        template='plotly_white',
        legend=dict(font=dict(size=9), orientation='v', bgcolor='rgba(255,255,255,0.85)'),
        hovermode='x unified', margin=dict(l=60, r=20, t=50, b=50))
    return fig


@callback(
    Output('sc-assignment-datatable', 'data'),
    Input('sc-site-filter', 'value'), Input('sc-state-filter', 'value'),
    Input('sc-color-mode', 'value'), State('sc-manual-overrides', 'data'),
)
def sc_update_table(sites, state_val, color_mode, overrides):
    sites = _sc_normalize_sites(sites)
    color_mode = color_mode or 'true_label'
    overrides = overrides or {}
    df = _sc_filtered(sites, state_val, color_mode)
    table_data = []
    for _, row in df.sort_values(['site', 'spectrum_id']).iterrows():
        site, sid = row['site'], row['spectrum_id']
        key = f"{site}::{sid}"
        table_data.append({
            'site': site, 'spectrum_id': sid,
            'running_state': row['running_state'],
            'prediction': row['prediction'],
            'manual_override': overrides.get(key, ''),
            'weighted_score_norm': round(float(row.get('weighted_score_norm', 0)), 3),
            'total_votes': int(row.get('total_votes', 0)),
            'label_source': row.get('label_source', ''),
        })
    return table_data


@callback(Output('sc-manual-overrides', 'data'),
          Input('sc-assignment-datatable', 'data'),
          State('sc-manual-overrides', 'data'), prevent_initial_call=True)
def sc_save_overrides(table_data, current):
    if not table_data:
        return current or {}
    overrides = dict(current or {})
    for row in table_data:
        key = f"{row['site']}::{row['spectrum_id']}"
        manual = row.get('manual_override', '')
        if manual:
            overrides[key] = manual
        elif key in overrides:
            del overrides[key]
    return overrides


@callback(Output('sc-download-labels', 'data'),
          Input('sc-export-btn', 'n_clicks'),
          State('sc-manual-overrides', 'data'), prevent_initial_call=True)
def sc_export_labels(n_clicks, overrides):
    if not n_clicks:
        return no_update
    overrides = overrides or {}
    rows = []
    for _, row in dc_data.iterrows():
        site, sid = row['site'], row['spectrum_id']
        manual = overrides.get(f"{site}::{sid}", '')
        final = manual if manual else row['running_state']
        rows.append({
            'site': site, 'spectrum_id': sid,
            'running_state': final,
            'label_source': 'manual' if manual else row.get('label_source', 'cluster'),
            'prediction': row['prediction'],
            'weighted_score_norm': row.get('weighted_score_norm', 0),
            'total_votes': row.get('total_votes', 0),
        })
    ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    return dcc.send_data_frame(pd.DataFrame(rows).to_csv,
                               f'decision_criteria_labels_{ts}.csv', index=False)


if __name__ == '__main__':
    print(f"\n  Starting dashboard at http://127.0.0.1:8050")
    print(f"  Press Ctrl+C to stop.\n")
    app.run(debug=False, port=8050)
