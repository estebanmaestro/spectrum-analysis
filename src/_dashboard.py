"""
Interactive ON/OFF Spectrum Dashboard
=====================================
Dash/Plotly app for exploring per-site ON/OFF clustering results.
Supports multi-site selection with per-site color differentiation.

Clustering mode (per site): default bundle result, PowerGap on total power,
or PowerGap on summed µV² in f < 1 MHz plus f ≥ 30 MHz.

Usage:
    python src/_dashboard.py
    Then open http://127.0.0.1:8050 in a browser.
"""
import os, pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input
from sklearn.metrics import silhouette_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(ROOT, '_results_bundle.pkl')

with open(BUNDLE_PATH, 'rb') as f:
    bundle = pickle.load(f)

data = bundle['data']
raw_spectrums = bundle['raw_spectrums']
site_assignments = bundle['site_assignments']
all_site_names = bundle['all_site_names']

CLUSTER_DEFAULT = 'default'
CLUSTER_TOTAL_POWER = 'total_power'
CLUSTER_BAND_LIMITED = 'band_limited'

CLUSTER_OPTIONS = [
    {'label': 'Default (saved pipeline)', 'value': CLUSTER_DEFAULT},
    {'label': 'Total power — largest gap in log-power', 'value': CLUSTER_TOTAL_POWER},
    {
        'label': 'Band power — <1 MHz + ≥30 MHz (largest gap)',
        'value': CLUSTER_BAND_LIMITED,
    },
]


def _powergap_binary_labels(log_vals):
    """Largest-gap split on sorted log-scale values; returns 0/1 labels or None."""
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
    """Higher mean linear power cluster → ON."""
    linear_power = np.asarray(linear_power, dtype=float)
    labels = np.asarray(labels).ravel()
    cluster_ids = sorted(np.unique(labels))
    cluster_power = {
        cid: linear_power[labels == cid].mean() for cid in cluster_ids
    }
    on_cluster = max(cluster_power, key=cluster_power.get)
    assignment = {}
    for i, sid in enumerate(site_df['spectrum_id'].values):
        assignment[sid] = 'ON' if labels[i] == on_cluster else 'OFF'
    return assignment


def _band_limited_power_uv2(freqs_hz, amps_uv):
    """Sum of squared amplitude (µV²) for f < 1 MHz and f ≥ 30 MHz."""
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    amps_uv = np.asarray(amps_uv, dtype=float)
    pwr = amps_uv ** 2
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
    """Separability 0–100 and per-sample scores (same logic as validation script)."""
    log_tp = np.asarray(log_metric, dtype=float)
    n_site = len(site_df)
    sids = site_df['spectrum_id'].values
    on_mask = np.array([assignment[sid] == 'ON' for sid in sids])
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

    if len(on_log) > 0 and len(off_log) > 0:
        boundary = (on_log.mean() + off_log.mean()) / 2.0
        half_gap = abs(on_log.mean() - off_log.mean()) / 2.0
    else:
        boundary = log_tp.mean()
        half_gap = 1.0

    sample_confidence = {}
    for i in range(n_site):
        sid = sids[i]
        margin = abs(log_tp[i] - boundary) / (half_gap + 1e-10)
        score = round(min(margin, 2.0) / 2.0 * 100, 1)
        sample_confidence[sid] = {
            'power_margin': round(float(margin), 2),
            'score': score,
        }
    return confidence, sample_confidence


def _silhouette_1d(log_metric, labels):
    log_metric = np.asarray(log_metric, dtype=float).reshape(-1, 1)
    labels = np.asarray(labels).ravel()
    n = len(labels)
    if n < 3 or len(np.unique(labels)) < 2:
        return 0.0
    try:
        return float(silhouette_score(log_metric, labels))
    except Exception:
        return 0.0


def _dashboard_cluster_view(site, mode):
    """
    Returns dict: assignment, config, silhouette, confidence, sample_confidence.
    For default mode, pulls from bundle (silhouette/confidence from saved run).
    """
    if site not in site_assignments:
        return None
    sa = site_assignments[site]
    mk = data['site'] == site
    site_df = data.loc[mk]
    if len(site_df) == 0:
        return None

    if mode == CLUSTER_DEFAULT:
        return {
            'assignment': sa['assignment'],
            'config': sa['config'],
            'silhouette': sa['silhouette'],
            'confidence': sa.get('confidence', {}),
            'sample_confidence': sa.get('sample_confidence', {}),
        }

    if mode == CLUSTER_TOTAL_POWER:
        linear = site_df['total_power'].values.astype(float)
        log_m = np.log1p(linear)
        labels = _powergap_binary_labels(log_m)
        if labels is None:
            labels = np.zeros(len(site_df), dtype=int)
            labels[len(labels) // 2:] = 1
        assignment = _assignment_from_labels(site_df, labels, linear)
        conf, sc = _confidence_from_log_metric(site_df, assignment, log_m)
        sil = _silhouette_1d(log_m, labels)
        return {
            'assignment': assignment,
            'config': 'Total power (PowerGap)',
            'silhouette': sil,
            'confidence': conf,
            'sample_confidence': sc,
        }

    if mode == CLUSTER_BAND_LIMITED:
        linear = _band_powers_for_site(site, site_df)
        log_m = np.log1p(np.maximum(linear, 0.0))
        labels = _powergap_binary_labels(log_m)
        if labels is None:
            labels = np.zeros(len(site_df), dtype=int)
            labels[len(labels) // 2:] = 1
        assignment = _assignment_from_labels(site_df, labels, linear)
        conf, sc = _confidence_from_log_metric(site_df, assignment, log_m)
        sil = _silhouette_1d(log_m, labels)
        return {
            'assignment': assignment,
            'config': 'Band power <1 MHz + ≥30 MHz (PowerGap)',
            'silhouette': sil,
            'confidence': conf,
            'sample_confidence': sc,
        }

    return None


def _views_for_sites(sites, mode):
    """Per-site cluster views; missing sites omitted."""
    out = {}
    for site in sites:
        v = _dashboard_cluster_view(site, mode)
        if v is not None:
            out[site] = v
    return out

SITE_PALETTE = [
    '#1D4ED8', '#047857', '#7C3AED', '#DC2626', '#B45309',
    '#0891B2', '#BE185D', '#D97706', '#059669', '#6D28D9',
    '#0369A1', '#A21CAF', '#C2410C', '#15803D', '#1E40AF',
    '#9333EA', '#B91C1C', '#0E7490', '#A16207', '#4338CA',
    '#DB2777', '#065F46', '#7E22CE', '#EA580C', '#1D4ED8',
    '#047857', '#7C3AED', '#DC2626', '#B45309', '#0891B2',
]

app = Dash(__name__)
app.title = "ON/OFF Spectrum Analysis"

first_site = all_site_names[0]


def _site_color(site, selected_sites):
    try:
        idx = list(selected_sites).index(site)
    except ValueError:
        idx = 0
    return SITE_PALETTE[idx % len(SITE_PALETTE)]


def _normalize_sites(sites):
    if sites is None:
        return [first_site]
    if isinstance(sites, str):
        return [sites]
    if len(sites) == 0:
        return [first_site]
    return list(sites)


def build_layout():
    return html.Div(style={
        'fontFamily': "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
        'background': '#f5f7fa', 'minHeight': '100vh', 'padding': '0',
    }, children=[
        html.Div(style={
            'background': '#003d4d', 'color': 'white', 'padding': '28px 40px',
            'borderBottom': '4px solid #00838f',
        }, children=[
            html.H1("ON/OFF Spectrum Analysis", style={
                'margin': '0 0 4px', 'fontSize': '1.6rem', 'fontWeight': '700',
            }),
            html.Div("Per-site unsupervised clustering — independent analysis per group",
                      style={'opacity': '0.7', 'fontSize': '0.9rem'}),
        ]),

        html.Div(style={
            'maxWidth': '1400px', 'margin': '0 auto', 'padding': '24px 32px',
        }, children=[
            html.Div(style={
                'display': 'flex', 'gap': '20px', 'marginBottom': '20px',
                'flexWrap': 'wrap', 'alignItems': 'flex-start',
                'justifyContent': 'space-between',
            }, children=[
                html.Div(style={'flex': '1 1 320px', 'minWidth': '260px', 'maxWidth': '520px'},
                         children=[
                    html.Label("Sites (multi-select)", style={
                        'fontWeight': '600', 'fontSize': '0.85rem', 'marginBottom': '6px',
                        'display': 'block', 'color': '#1a2332',
                    }),
                    dcc.Dropdown(
                        id='site-filter',
                        options=[{'label': s, 'value': s} for s in all_site_names],
                        value=[first_site],
                        multi=True,
                        placeholder="Select one or more sites...",
                        style={'fontSize': '0.85rem'},
                    ),
                ]),
                html.Div(style={
                    'display': 'flex', 'flexWrap': 'wrap', 'gap': '16px',
                    'alignItems': 'flex-start', 'flex': '0 1 auto',
                }, children=[
                    html.Div(style={'minWidth': '160px', 'width': '180px'}, children=[
                        html.Label("Running State", style={
                            'fontWeight': '600', 'fontSize': '0.85rem', 'marginBottom': '6px',
                            'display': 'block', 'color': '#1a2332',
                        }),
                        dcc.Dropdown(
                            id='state-filter',
                            options=[
                                {'label': 'All', 'value': 'all'},
                                {'label': 'ON', 'value': 'ON'},
                                {'label': 'OFF', 'value': 'OFF'},
                            ],
                            value='all',
                            clearable=False,
                            style={'fontSize': '0.85rem'},
                        ),
                    ]),
                    html.Div(style={'minWidth': '220px', 'width': '280px'}, children=[
                        html.Label("Clustering view", style={
                            'fontWeight': '600', 'fontSize': '0.85rem', 'marginBottom': '6px',
                            'display': 'block', 'color': '#1a2332',
                        }),
                        dcc.Dropdown(
                            id='clustering-mode',
                            options=CLUSTER_OPTIONS,
                            value=CLUSTER_DEFAULT,
                            clearable=False,
                            style={'fontSize': '0.82rem'},
                        ),
                    ]),
                    html.Div(id='separability-slot', style={
                        'minWidth': '132px', 'flexShrink': 0,
                    }),
                ]),
            ]),

            html.Div(id='info-cards', style={
                'display': 'flex', 'gap': '16px', 'marginBottom': '24px',
                'flexWrap': 'wrap',
            }),

            html.Div(style={
                'background': 'white', 'borderRadius': '8px', 'padding': '20px',
                'boxShadow': '0 1px 8px rgba(0,0,0,0.06)',
                'border': '1px solid #e2e8f0', 'marginBottom': '24px',
            }, children=[
                dcc.Graph(id='spectrum-chart', style={'height': '560px'}),
            ]),

            html.Div(style={
                'background': 'white', 'borderRadius': '8px', 'padding': '20px',
                'boxShadow': '0 1px 8px rgba(0,0,0,0.06)',
                'border': '1px solid #e2e8f0', 'marginBottom': '24px',
            }, children=[
                html.Label("Select Spectrums to Display", style={
                    'fontWeight': '600', 'fontSize': '0.9rem', 'marginBottom': '10px',
                    'display': 'block', 'color': '#1a2332',
                }),
                dcc.Checklist(
                    id='spectrum-selector',
                    inline=True,
                    style={'fontSize': '0.82rem', 'display': 'flex',
                           'flexWrap': 'wrap', 'gap': '6px 16px'},
                    inputStyle={'marginRight': '4px'},
                ),
            ]),

            html.Div(style={
                'background': 'white', 'borderRadius': '8px', 'padding': '20px',
                'boxShadow': '0 1px 8px rgba(0,0,0,0.06)',
                'border': '1px solid #e2e8f0',
            }, children=[
                html.H3("Spectrum Assignments", style={
                    'margin': '0 0 12px', 'fontSize': '1rem', 'color': '#1a2332',
                }),
                html.Div(id='assignment-table'),
            ]),
        ]),
    ])


app.layout = build_layout()


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


@callback(
    Output('spectrum-selector', 'options'),
    Output('spectrum-selector', 'value'),
    Input('site-filter', 'value'),
    Input('state-filter', 'value'),
    Input('clustering-mode', 'value'),
)
def update_spectrum_selector(sites, state_val, cluster_mode):
    sites = _normalize_sites(sites)
    mode = cluster_mode or CLUSTER_DEFAULT
    views = _views_for_sites(sites, mode)
    options = []
    all_values = []

    for site in sites:
        if site not in views:
            continue
        view = views[site]
        assignment = view['assignment']
        color = _site_color(site, sites)
        multi = len(sites) > 1
        mk = data['site'] == site
        filtered = data.loc[mk]
        if state_val and state_val != 'all':
            keep = [
                idx for idx in filtered.index
                if assignment.get(filtered.loc[idx, 'spectrum_id'], '') == state_val
            ]
            filtered = filtered.loc[keep]

        for _, row in filtered.iterrows():
            sid = row['spectrum_id']
            inferred = assignment.get(sid, '?')
            tag = f"{site}::{sid}"
            prefix = f"[{site}] " if multi else ""
            options.append({
                'label': html.Span([
                    html.Span(f"{prefix}[{inferred}] ", style={
                        'color': color if multi else (
                            '#DC2626' if inferred == 'ON' else '#2563EB'),
                        'fontWeight': '700',
                    }),
                    sid,
                ]),
                'value': tag,
            })
            all_values.append(tag)

    return options, all_values


@callback(
    Output('separability-slot', 'children'),
    Input('site-filter', 'value'),
    Input('clustering-mode', 'value'),
)
def update_separability_slot(sites, cluster_mode):
    sites = _normalize_sites(sites)
    mode = cluster_mode or CLUSTER_DEFAULT
    if len(sites) != 1:
        return []
    view = _dashboard_cluster_view(sites[0], mode)
    if view is None:
        return []
    conf = view.get('confidence', {})
    sep = conf.get('separability', 0)
    if sep >= 70:
        sep_color = '#047857'
    elif sep >= 40:
        sep_color = '#B45309'
    else:
        sep_color = '#DC2626'
    return [_make_card("Separability", f"{sep}", sep_color)]


@callback(
    Output('info-cards', 'children'),
    Input('site-filter', 'value'),
    Input('clustering-mode', 'value'),
)
def update_info_cards(sites, cluster_mode):
    sites = _normalize_sites(sites)
    mode = cluster_mode or CLUSTER_DEFAULT
    views = _views_for_sites(sites, mode)
    total_n = 0
    total_on = 0
    total_off = 0

    for site in sites:
        if site not in views:
            continue
        view = views[site]
        total_n += (data['site'] == site).sum()
        total_on += sum(1 for v in view['assignment'].values() if v == 'ON')
        total_off += sum(1 for v in view['assignment'].values() if v == 'OFF')

    site_label = ', '.join(sites) if len(sites) <= 3 else f"{len(sites)} sites"
    cards = [
        _make_card("Sites", site_label, '#005662'),
        _make_card("Spectrums", str(total_n)),
        _make_card("ON", str(total_on), '#DC2626'),
        _make_card("OFF", str(total_off), '#2563EB'),
    ]
    if len(sites) == 1 and sites[0] in views:
        v = views[sites[0]]
        cards.append(_make_card("Method", v['config'], '#475569'))
        cards.append(_make_card("Silhouette", f"{v['silhouette']:.3f}", '#7C3AED'))
    return cards


@callback(
    Output('spectrum-chart', 'figure'),
    Input('site-filter', 'value'),
    Input('spectrum-selector', 'value'),
    Input('clustering-mode', 'value'),
)
def update_spectrum_chart(sites, selected_tags, cluster_mode):
    sites = _normalize_sites(sites)
    mode = cluster_mode or CLUSTER_DEFAULT
    views = _views_for_sites(sites, mode)
    fig = go.Figure()

    if not selected_tags:
        fig.update_layout(
            title="No spectrums selected",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Amplitude (µV)",
        )
        return fig

    multi = len(sites) > 1
    total_on = 0
    total_off = 0

    for tag in sorted(selected_tags):
        if '::' not in tag:
            continue
        site, sid = tag.split('::', 1)
        if site not in views:
            continue
        view = views[site]
        assignment = view['assignment']
        key = (site, sid)
        if key not in raw_spectrums:
            continue

        freqs_hz, amps_uv = raw_spectrums[key]
        freq_mhz = freqs_hz / 1e6
        inferred = assignment.get(sid, '?')
        is_on = inferred == 'ON'

        if is_on:
            total_on += 1
        else:
            total_off += 1

        dash_style = 'solid' if is_on else 'dash'
        color = _site_color(site, sites) if multi else (
            '#DC2626' if is_on else '#2563EB')
        prefix = f"{site} | " if multi else ""

        fig.add_trace(go.Scattergl(
            x=freq_mhz, y=amps_uv,
            mode='lines',
            name=f"{prefix}{sid} ({inferred})",
            line=dict(color=color, width=1.2, dash=dash_style),
            opacity=0.85,
            hovertemplate=(
                f"<b>{site}</b> — {sid} ({inferred})<br>"
                "Freq: %{x:.3f} MHz<br>"
                "Amp: %{y:.2f} µV<extra></extra>"
            ),
        ))

    title_sites = ', '.join(sites) if len(sites) <= 3 else f"{len(sites)} sites"
    total_shown = total_on + total_off
    conf_suffix = ""
    if len(sites) == 1 and sites[0] in views:
        c = views[sites[0]].get('confidence', {})
        if c:
            conf_suffix = f" — Separability: {c.get('separability', 0)}/100"
    fig.update_layout(
        title=dict(
            text=f"{title_sites} — {total_shown} spectrums "
                 f"({total_on} ON, {total_off} OFF){conf_suffix}",
            font=dict(size=15),
        ),
        xaxis=dict(title="Frequency (MHz)", type='log',
                   gridcolor='#eee', showgrid=True),
        yaxis=dict(title="Amplitude (µV)", type='log',
                   gridcolor='#eee', showgrid=True),
        template='plotly_white',
        legend=dict(font=dict(size=9), orientation='v',
                    bgcolor='rgba(255,255,255,0.85)'),
        hovermode='x unified',
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


@callback(
    Output('assignment-table', 'children'),
    Input('site-filter', 'value'),
    Input('state-filter', 'value'),
    Input('clustering-mode', 'value'),
)
def update_assignment_table(sites, state_val, cluster_mode):
    sites = _normalize_sites(sites)
    mode = cluster_mode or CLUSTER_DEFAULT
    views = _views_for_sites(sites, mode)
    multi = len(sites) > 1

    header_cells = []
    if multi:
        header_cells.append(html.Th("Site", style=_th_style()))
    header_cells.extend([
        html.Th("Spectrum ID", style=_th_style()),
        html.Th("State", style=_th_style()),
        html.Th("Score", style=_th_style()),
        html.Th("Total Power (µV²)", style=_th_style()),
        html.Th("Method", style=_th_style()),
    ])
    rows = [html.Tr(header_cells)]

    for site in sites:
        if site not in views:
            continue
        view = views[site]
        assignment = view['assignment']
        conf = view.get('confidence', {})
        sep = conf.get('separability', 0)
        color = _site_color(site, sites)

        if sep < 40:
            warn_text = (f"{site}: Separability {sep}/100 — "
                         "weak separation, may be a single state")
            rows.append(html.Tr([html.Td(
                warn_text,
                colSpan=6 if multi else 5,
                style={**_td_style(), 'background': '#fef3c7',
                       'color': '#92400e', 'fontWeight': '600',
                       'fontSize': '0.78rem', 'textAlign': 'center'},
            )]))

        mk = data['site'] == site
        filtered = data.loc[mk]
        if state_val and state_val != 'all':
            keep = [
                idx for idx in filtered.index
                if assignment.get(filtered.loc[idx, 'spectrum_id'], '') == state_val
            ]
            filtered = filtered.loc[keep]
        filtered = filtered.sort_values('spectrum_id')

        sc_map = view.get('sample_confidence', {})
        for _, row in filtered.iterrows():
            sid = row['spectrum_id']
            inferred = assignment.get(sid, '?')
            badge_style = {
                'display': 'inline-block', 'padding': '2px 12px',
                'borderRadius': '12px', 'fontSize': '0.75rem',
                'fontWeight': '700', 'letterSpacing': '0.03em',
            }
            if inferred == 'ON':
                badge_style.update({'background': '#dcfce7', 'color': '#166534'})
            else:
                badge_style.update({'background': '#fee2e2', 'color': '#991b1b'})

            sc = sc_map.get(sid, {})
            sc_score = sc.get('score', 0)
            if sc_score >= 70:
                sc_bg, sc_fg = '#d1fae5', '#065f46'
            elif sc_score >= 40:
                sc_bg, sc_fg = '#fef9c3', '#854d0e'
            else:
                sc_bg, sc_fg = '#fecaca', '#991b1b'
            sc_badge = {
                'display': 'inline-block', 'padding': '2px 10px',
                'borderRadius': '12px', 'fontSize': '0.72rem',
                'fontWeight': '600',
                'background': sc_bg, 'color': sc_fg,
            }

            cells = []
            if multi:
                cells.append(html.Td(site, style={
                    **_td_style(), 'color': color, 'fontWeight': '600'}))
            cells.extend([
                html.Td(sid, style=_td_style()),
                html.Td(html.Span(inferred, style=badge_style), style=_td_style()),
                html.Td(html.Span(f"{sc_score}", style=sc_badge),
                         style=_td_style()),
                html.Td(f"{row['total_power']:,.0f}", style=_td_style()),
                html.Td(view['config'], style={
                    **_td_style(), 'fontSize': '0.75rem', 'color': '#78909c'}),
            ])
            rows.append(html.Tr(cells))

    return html.Table(rows, style={
        'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '0.82rem',
    })


def _th_style():
    return {
        'background': '#e8f5f7', 'color': '#005662', 'fontWeight': '600',
        'padding': '10px 14px', 'textAlign': 'left',
        'borderBottom': '2px solid #d4dde3', 'whiteSpace': 'nowrap',
    }


def _td_style():
    return {
        'padding': '8px 14px', 'borderBottom': '1px solid #e2e8f0',
    }


if __name__ == '__main__':
    print(f"\n  Starting dashboard at http://127.0.0.1:8050")
    print(f"  Press Ctrl+C to stop.\n")
    app.run(debug=False, port=8050)
