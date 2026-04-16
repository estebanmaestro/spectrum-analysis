"""
Supervised Study Results Dashboard
===================================
Interactive Dash/Plotly app for exploring the results of the supervised
feature importance and classification study.

Usage:
    python src/_supervised_dashboard.py
    Then open http://127.0.0.1:8051 in a browser.
"""
import os, pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, dash_table, callback, Output, Input

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(ROOT, '_supervised_study_bundle.pkl')

with open(BUNDLE_PATH, 'rb') as f:
    B = pickle.load(f)

univar_df = B['univar_df']
importance_df = B['importance_df']
feature_cols = B['feature_cols']
scale_inv_cols = B['scale_invariant_cols']
rfecv_selected = B['rfecv_selected']
rfecv_n = int(B['rfecv_n_features'])
rfecv_scores = B['rfecv_scores']
model_all = B['model_results_all']
model_sel = B['model_results_selected']
model_si = B['model_results_si']
hyperparams = B['best_hyperparams']
merged = B['merged_data']

MODEL_NAMES = list(model_all.keys())
METRICS = ['accuracy', 'f1', 'precision', 'recall']
N_FEATURES = len(feature_cols)
N_SI = len(scale_inv_cols)
N_SAMPLES = len(merged)
N_ON = int((merged['y'] == 1).sum())
N_OFF = N_SAMPLES - N_ON

ACCENT = '#003d4d'
PALETTE = {
    'blue': '#1D4ED8', 'green': '#047857', 'purple': '#7C3AED',
    'red': '#DC2626', 'amber': '#B45309', 'teal': '#0891B2',
    'pink': '#BE185D',
}
METRIC_COLORS = {
    'accuracy': PALETTE['blue'], 'f1': PALETTE['green'],
    'precision': PALETTE['purple'], 'recall': PALETTE['red'],
}

# ═══════════════════════════════════════════════════════════════
# PRE-BUILT FIGURES
# ═══════════════════════════════════════════════════════════════

def _fig_univariate(top_n=30):
    top = univar_df.head(top_n).copy()
    top['color'] = top['scale_invariant'].map(
        {True: PALETTE['green'], False: PALETTE['blue']})
    top['type'] = top['scale_invariant'].map(
        {True: 'Scale-Invariant', False: 'Scale-Dependent'})
    fig = go.Figure()
    for si_val, label in [(False, 'Scale-Dependent'), (True, 'Scale-Invariant')]:
        subset = top[top['scale_invariant'] == si_val]
        if subset.empty:
            continue
        fig.add_trace(go.Bar(
            y=subset['feature'], x=subset['roc_auc'],
            orientation='h', name=label,
            marker_color=PALETTE['green'] if si_val else PALETTE['blue'],
            opacity=0.88,
        ))
    fig.update_layout(
        title='Univariate Feature Ranking by ROC-AUC',
        xaxis_title='ROC-AUC (single feature)',
        yaxis=dict(autorange='reversed', tickfont=dict(size=9)),
        template='plotly_white', barmode='overlay',
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        margin=dict(l=160, r=20, t=60, b=40),
        height=max(400, top_n * 22),
    )
    fig.add_vline(x=0.5, line_dash='dash', line_color='gray', opacity=0.4)
    return fig


def _fig_correlation():
    top20 = univar_df.head(20)['feature'].tolist()
    idx = [feature_cols.index(f) for f in top20]
    meta = ['spectrum_id', 'asset', 'state', 'site', 'running', 'y', 'source']
    feat_data = merged[[c for c in merged.columns if c not in meta]]
    vals = feat_data.iloc[:, idx].values.astype(float)
    corr = np.corrcoef(vals.T)
    fig = go.Figure(data=go.Heatmap(
        z=corr, x=top20, y=top20,
        colorscale='RdBu_r', zmin=-1, zmax=1, zmid=0,
        text=np.round(corr, 2), texttemplate='%{text:.2f}',
        textfont=dict(size=8),
    ))
    fig.update_layout(
        title='Feature Correlation Heatmap (Top 20 by ROC-AUC)',
        template='plotly_white', height=650,
        xaxis=dict(tickfont=dict(size=8), tickangle=45),
        yaxis=dict(tickfont=dict(size=8)),
        margin=dict(l=140, b=120, t=50, r=20),
    )
    return fig


def _fig_model_comparison(results_dict, subtitle='All Features'):
    fig = go.Figure()
    x = np.arange(len(MODEL_NAMES))
    width = 0.18
    for i, metric in enumerate(METRICS):
        means = [results_dict[m][metric][0] for m in MODEL_NAMES]
        stds = [results_dict[m][metric][1] for m in MODEL_NAMES]
        fig.add_trace(go.Bar(
            x=[n + i * width for n in x], y=means,
            error_y=dict(type='data', array=stds, visible=True),
            name=metric.capitalize(), marker_color=METRIC_COLORS[metric],
            opacity=0.88, width=width,
        ))
    fig.update_layout(
        title=f'Model Comparison — {subtitle}',
        xaxis=dict(
            tickvals=[n + 1.5 * width for n in x],
            ticktext=MODEL_NAMES, tickfont=dict(size=9),
        ),
        yaxis=dict(title='Score', range=[0, 1.12]),
        template='plotly_white', barmode='group',
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        margin=dict(l=50, r=20, t=60, b=80), height=420,
    )
    fig.add_hline(y=0.9, line_dash='dash', line_color='gray', opacity=0.3)
    return fig


def _fig_subset_comparison():
    subsets = [
        (f'All ({N_FEATURES})', model_all),
        (f'RFECV ({rfecv_n})', model_sel),
        (f'Scale-Inv ({N_SI})', model_si),
    ]
    colors = [PALETTE['blue'], PALETTE['green'], PALETTE['purple']]
    fig = go.Figure()
    x = np.arange(len(MODEL_NAMES))
    width = 0.25
    for si, (label, rset) in enumerate(subsets):
        means = [rset[m]['f1'][0] for m in MODEL_NAMES]
        stds = [rset[m]['f1'][1] for m in MODEL_NAMES]
        fig.add_trace(go.Bar(
            x=[n + si * width for n in x], y=means,
            error_y=dict(type='data', array=stds, visible=True),
            name=label, marker_color=colors[si], opacity=0.88, width=width,
        ))
    fig.update_layout(
        title='Feature Subset Comparison — F1 Score',
        xaxis=dict(
            tickvals=[n + width for n in x],
            ticktext=MODEL_NAMES, tickfont=dict(size=9),
        ),
        yaxis=dict(title='F1 Score', range=[0, 1.12]),
        template='plotly_white', barmode='group',
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        margin=dict(l=50, r=20, t=60, b=80), height=420,
    )
    fig.add_hline(y=0.9, line_dash='dash', line_color='gray', opacity=0.3)
    return fig


def _fig_importance(top_n=15):
    top = importance_df.head(top_n).copy()
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=['L1 Coefficient (abs)',
                                        'Permutation Importance',
                                        'Consensus Rank (lower = better)'],
                        horizontal_spacing=0.12)
    for col_idx, (col, ascending) in enumerate([
        ('l1_coef', False), ('perm_imp', False), ('consensus_rank', True),
    ], start=1):
        sorted_top = top.sort_values(col, ascending=ascending)
        colors = [PALETTE['green'] if s else PALETTE['blue']
                  for s in sorted_top['scale_invariant']]
        fig.add_trace(go.Bar(
            y=sorted_top['feature'], x=sorted_top[col],
            orientation='h', marker_color=colors, opacity=0.88,
            showlegend=False,
        ), row=1, col=col_idx)
        fig.update_yaxes(tickfont=dict(size=8), row=1, col=col_idx)

    fig.update_layout(
        title='Feature Importance — Top 15 by Consensus',
        template='plotly_white', height=500,
        margin=dict(l=140, r=20, t=70, b=30),
    )
    return fig


def _fig_rfecv():
    n_range = list(range(3, 3 + len(rfecv_scores)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_range, y=rfecv_scores, mode='lines+markers',
        marker=dict(size=4, color=PALETTE['blue']),
        line=dict(color=PALETTE['blue'], width=2),
        name='CV F1',
    ))
    fig.add_vline(x=rfecv_n, line_dash='dash', line_color=PALETTE['red'],
                  opacity=0.7, annotation_text=f'Optimal: {rfecv_n}',
                  annotation_position='top right')
    fig.update_layout(
        title='RFECV — CV F1 Score vs Number of Features',
        xaxis_title='Number of Features',
        yaxis_title='Cross-Validated F1',
        template='plotly_white', height=380,
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# HELPER: STYLED COMPONENTS
# ═══════════════════════════════════════════════════════════════

_CARD_STYLE = {
    'background': 'white', 'borderRadius': '8px', 'padding': '16px 24px',
    'boxShadow': '0 1px 8px rgba(0,0,0,0.06)', 'border': '1px solid #e2e8f0',
    'flex': '1', 'minWidth': '140px',
}
_SECTION_STYLE = {
    'background': 'white', 'borderRadius': '8px', 'padding': '24px',
    'boxShadow': '0 1px 8px rgba(0,0,0,0.06)', 'border': '1px solid #e2e8f0',
    'marginBottom': '24px',
}
_LABEL_STYLE = {
    'fontWeight': '600', 'fontSize': '0.75rem', 'color': '#78909c',
    'textTransform': 'uppercase', 'letterSpacing': '0.04em',
}


def _metric_card(title, value, color='#1a2332'):
    return html.Div(style=_CARD_STYLE, children=[
        html.Div(title, style=_LABEL_STYLE),
        html.Div(value, style={
            'fontSize': '1.3rem', 'fontWeight': '700',
            'color': color, 'marginTop': '4px',
        }),
    ])


def _section_title(text):
    return html.H2(text, style={
        'margin': '0 0 16px', 'fontSize': '1.1rem',
        'fontWeight': '700', 'color': ACCENT,
        'borderBottom': f'2px solid {ACCENT}',
        'paddingBottom': '8px',
    })


# ═══════════════════════════════════════════════════════════════
# DATA TABLES
# ═══════════════════════════════════════════════════════════════

univar_table_data = univar_df.head(30).to_dict('records')
for row in univar_table_data:
    row['mw_p'] = f"{row['mw_p']:.2e}" if row['mw_p'] < 0.01 else f"{row['mw_p']:.4f}"
    row['scale_invariant'] = 'Yes' if row['scale_invariant'] else ''

_UNIVAR_COLS = [
    {'name': '#', 'id': 'rank'},
    {'name': 'Feature', 'id': 'feature'},
    {'name': 'ROC-AUC', 'id': 'roc_auc', 'type': 'numeric'},
    {'name': "Cohen's d", 'id': 'cohens_d', 'type': 'numeric'},
    {'name': 'MW p-value', 'id': 'mw_p'},
    {'name': 'Point-Biserial r', 'id': 'pb_corr', 'type': 'numeric'},
    {'name': 'Scale Inv.', 'id': 'scale_invariant'},
]

importance_table_data = importance_df.head(20).reset_index(drop=True).copy()
importance_table_data['consensus_rank'] = importance_table_data['consensus_rank'].round(1)
importance_table_data['l1_coef'] = importance_table_data['l1_coef'].round(4)
importance_table_data['perm_imp'] = importance_table_data['perm_imp'].round(4)
importance_table_data['scale_invariant'] = importance_table_data['scale_invariant'].map(
    {True: 'Yes', False: ''})
importance_table_data['rfecv_selected'] = importance_table_data['rfecv_selected'].map(
    {True: '*', False: ''})
importance_table_data = importance_table_data.to_dict('records')

_IMP_COLS = [
    {'name': 'Feature', 'id': 'feature'},
    {'name': '|L1 coef|', 'id': 'l1_coef', 'type': 'numeric'},
    {'name': 'L1 Rank', 'id': 'l1_rank', 'type': 'numeric'},
    {'name': 'Perm Imp', 'id': 'perm_imp', 'type': 'numeric'},
    {'name': 'Perm Rank', 'id': 'perm_rank', 'type': 'numeric'},
    {'name': 'RFECV Rank', 'id': 'rfecv_rank', 'type': 'numeric'},
    {'name': 'Consensus', 'id': 'consensus_rank', 'type': 'numeric'},
    {'name': 'SI', 'id': 'scale_invariant'},
    {'name': 'RFECV Sel.', 'id': 'rfecv_selected'},
]


def _model_table(results_dict):
    rows = []
    for name in MODEL_NAMES:
        r = results_dict[name]
        row = {'model': name}
        for m in METRICS:
            row[m] = f"{r[m][0]:.3f} +/- {r[m][1]:.3f}"
        auc_m, auc_s = r['roc_auc']
        row['roc_auc'] = f"{auc_m:.3f} +/- {auc_s:.3f}" if not np.isnan(auc_m) else 'N/A'
        rows.append(row)
    return rows

_MODEL_COLS = [
    {'name': 'Model', 'id': 'model'},
    {'name': 'Accuracy', 'id': 'accuracy'},
    {'name': 'F1', 'id': 'f1'},
    {'name': 'Precision', 'id': 'precision'},
    {'name': 'Recall', 'id': 'recall'},
    {'name': 'ROC-AUC', 'id': 'roc_auc'},
]

_TABLE_CELL_STYLE = {
    'textAlign': 'left', 'padding': '8px 14px', 'fontSize': '0.82rem',
    'fontFamily': "'Inter', -apple-system, sans-serif",
}
_TABLE_HEADER_STYLE = {
    'background': '#e8f5f7', 'color': '#005662', 'fontWeight': '600',
    'borderBottom': '2px solid #d4dde3', 'whiteSpace': 'nowrap',
}

best_all_name = max(model_all, key=lambda m: model_all[m]['f1'][0])
best_sel_name = max(model_sel, key=lambda m: model_sel[m]['f1'][0])
best_si_name = max(model_si, key=lambda m: model_si[m]['f1'][0])

# ═══════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════

app = Dash(__name__)
app.title = "Supervised Study Results"

app.layout = html.Div(style={
    'fontFamily': "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    'background': '#f5f7fa', 'minHeight': '100vh',
}, children=[
    # Header
    html.Div(style={
        'background': ACCENT, 'color': 'white', 'padding': '28px 40px',
        'borderBottom': '4px solid #00838f',
    }, children=[
        html.H1("Supervised Feature Study — Results", style={
            'margin': '0 0 4px', 'fontSize': '1.6rem', 'fontWeight': '700',
        }),
        html.Div("ON/OFF class separation analysis — feature importance, "
                  "model comparison, and optimal subset selection",
                  style={'opacity': '0.7', 'fontSize': '0.9rem'}),
    ]),

    # Content
    html.Div(style={
        'maxWidth': '1500px', 'margin': '0 auto', 'padding': '24px 32px',
    }, children=[
        # ── Summary cards ──
        html.Div(style={
            'display': 'flex', 'gap': '16px', 'marginBottom': '24px',
            'flexWrap': 'wrap',
        }, children=[
            _metric_card("Samples", str(N_SAMPLES)),
            _metric_card("ON / OFF", f"{N_ON} / {N_OFF}", PALETTE['red']),
            _metric_card("Total Features", str(N_FEATURES)),
            _metric_card("RFECV Selected", str(rfecv_n), PALETTE['green']),
            _metric_card("Best F1 (all)", f"{model_all[best_all_name]['f1'][0]:.3f}",
                         PALETTE['blue']),
            _metric_card("Best F1 (RFECV)", f"{model_sel[best_sel_name]['f1'][0]:.3f}",
                         PALETTE['green']),
        ]),

        # ── Tab selector ──
        dcc.Tabs(id='main-tabs', value='tab-overview', children=[
            dcc.Tab(label='Overview', value='tab-overview'),
            dcc.Tab(label='Univariate Analysis', value='tab-univariate'),
            dcc.Tab(label='Model Comparison', value='tab-models'),
            dcc.Tab(label='Feature Importance', value='tab-importance'),
            dcc.Tab(label='Feature Subset Comparison', value='tab-subsets'),
        ], style={'marginBottom': '24px'},
           colors={
               'border': '#d4dde3',
               'primary': ACCENT,
               'background': '#f0f3f5',
           }),

        html.Div(id='tab-content'),
    ]),
])


# ═══════════════════════════════════════════════════════════════
# TAB CONTENT CALLBACK
# ═══════════════════════════════════════════════════════════════

@callback(Output('tab-content', 'children'), Input('main-tabs', 'value'))
def render_tab(tab):
    if tab == 'tab-overview':
        return _tab_overview()
    if tab == 'tab-univariate':
        return _tab_univariate()
    if tab == 'tab-models':
        return _tab_models()
    if tab == 'tab-importance':
        return _tab_importance()
    if tab == 'tab-subsets':
        return _tab_subsets()
    return html.Div("Select a tab.")


def _tab_overview():
    rfecv_list = ', '.join(
        f"{f} (SI)" if f in scale_inv_cols else f for f in rfecv_selected)
    return html.Div([
        html.Div(style=_SECTION_STYLE, children=[
            _section_title("Study Summary"),
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr',
                            'gap': '20px'}, children=[
                html.Div([
                    html.H4("Dataset", style={'color': ACCENT, 'marginTop': 0}),
                    html.Ul(style={'lineHeight': '1.8', 'fontSize': '0.9rem'}, children=[
                        html.Li(f"{N_SAMPLES} labeled samples ({N_ON} ON / {N_OFF} OFF)"),
                        html.Li(f"{len(merged['site'].unique())} unique sites"),
                        html.Li(f"{N_FEATURES} features ({N_SI} scale-invariant)"),
                        html.Li(f"{(merged['source'] == 'manual').sum()} manual overrides"),
                    ]),
                    html.H4("Cross-Validation", style={'color': ACCENT}),
                    html.Ul(style={'lineHeight': '1.8', 'fontSize': '0.9rem'}, children=[
                        html.Li("StratifiedGroupKFold (5 folds, grouped by site)"),
                        html.Li("class_weight='balanced' on all models"),
                        html.Li("RobustScaler preprocessing"),
                    ]),
                ]),
                html.Div([
                    html.H4("Best Results", style={'color': ACCENT, 'marginTop': 0}),
                    html.Ul(style={'lineHeight': '1.8', 'fontSize': '0.9rem'}, children=[
                        html.Li([
                            html.Strong("All features: "),
                            f"{best_all_name} — "
                            f"F1 = {model_all[best_all_name]['f1'][0]:.3f}",
                        ]),
                        html.Li([
                            html.Strong(f"RFECV ({rfecv_n} features): "),
                            f"{best_sel_name} — "
                            f"F1 = {model_sel[best_sel_name]['f1'][0]:.3f}",
                        ]),
                        html.Li([
                            html.Strong(f"Scale-invariant ({N_SI}): "),
                            f"{best_si_name} — "
                            f"F1 = {model_si[best_si_name]['f1'][0]:.3f}",
                        ]),
                    ]),
                    html.H4("RFECV-Selected Features", style={'color': ACCENT}),
                    html.P(rfecv_list, style={'fontSize': '0.9rem', 'lineHeight': '1.6'}),
                    html.H4("Tuned Hyperparameters", style={'color': ACCENT}),
                    html.Ul(style={'lineHeight': '1.8', 'fontSize': '0.9rem'}, children=[
                        html.Li(f"LogReg L1 C={hyperparams['c_l1']}"),
                        html.Li(f"LogReg L2 C={hyperparams['c_l2']}"),
                        html.Li(f"LogReg EN C={hyperparams['c_en']}"),
                        html.Li(f"Linear SVM C={hyperparams['c_lsvm']}"),
                        html.Li(f"RBF SVM C={hyperparams['c_rbf']}"),
                        html.Li(f"Ridge alpha={hyperparams['alpha']}"),
                    ]),
                ]),
            ]),
        ]),
        html.Div(style=_SECTION_STYLE, children=[
            _section_title("Quick Glance — Model F1 by Feature Subset"),
            dcc.Graph(figure=_fig_subset_comparison(), config={'displayModeBar': False}),
        ]),
    ])


def _tab_univariate():
    return html.Div([
        html.Div(style=_SECTION_STYLE, children=[
            _section_title("Univariate Feature Ranking"),
            html.P("Each feature scored independently against the ON/OFF target. "
                   "Green bars = scale-invariant features.",
                   style={'fontSize': '0.85rem', 'color': '#64748b', 'marginBottom': '16px'}),
            dcc.Graph(figure=_fig_univariate(), config={'displayModeBar': False}),
        ]),
        html.Div(style=_SECTION_STYLE, children=[
            _section_title("Univariate Scores Table"),
            dash_table.DataTable(
                columns=_UNIVAR_COLS, data=univar_table_data,
                sort_action='native', filter_action='native',
                page_size=30,
                style_cell=_TABLE_CELL_STYLE,
                style_header=_TABLE_HEADER_STYLE,
                style_data_conditional=[
                    {'if': {'column_id': 'roc_auc', 'filter_query': '{roc_auc} >= 0.8'},
                     'backgroundColor': '#d1fae5', 'color': '#065f46', 'fontWeight': '700'},
                    {'if': {'column_id': 'scale_invariant',
                            'filter_query': '{scale_invariant} = "Yes"'},
                     'backgroundColor': '#ecfdf5', 'color': '#047857'},
                ],
            ),
        ]),
        html.Div(style=_SECTION_STYLE, children=[
            _section_title("Correlation Heatmap (Top 20)"),
            html.P("Reveals redundancy clusters among the best univariate features.",
                   style={'fontSize': '0.85rem', 'color': '#64748b', 'marginBottom': '16px'}),
            dcc.Graph(figure=_fig_correlation(), config={'displayModeBar': False}),
        ]),
    ])


def _tab_models():
    return html.Div([
        html.Div(style=_SECTION_STYLE, children=[
            _section_title(f"Model Performance — All {N_FEATURES} Features"),
            dcc.Graph(figure=_fig_model_comparison(model_all, f'All {N_FEATURES} Features'),
                      config={'displayModeBar': False}),
        ]),
        html.Div(style=_SECTION_STYLE, children=[
            _section_title("Detailed Metrics — All Features"),
            dash_table.DataTable(
                columns=_MODEL_COLS, data=_model_table(model_all),
                style_cell=_TABLE_CELL_STYLE, style_header=_TABLE_HEADER_STYLE,
            ),
        ]),
        html.Div(style=_SECTION_STYLE, children=[
            _section_title(f"Model Performance — RFECV {rfecv_n} Features"),
            dcc.Graph(figure=_fig_model_comparison(model_sel, f'RFECV {rfecv_n} Features'),
                      config={'displayModeBar': False}),
        ]),
        html.Div(style=_SECTION_STYLE, children=[
            _section_title(f"Detailed Metrics — RFECV {rfecv_n} Features"),
            dash_table.DataTable(
                columns=_MODEL_COLS, data=_model_table(model_sel),
                style_cell=_TABLE_CELL_STYLE, style_header=_TABLE_HEADER_STYLE,
            ),
        ]),
        html.Div(style=_SECTION_STYLE, children=[
            _section_title(f"Model Performance — Scale-Invariant {N_SI} Features"),
            dcc.Graph(figure=_fig_model_comparison(model_si,
                                                   f'Scale-Invariant {N_SI} Features'),
                      config={'displayModeBar': False}),
        ]),
        html.Div(style=_SECTION_STYLE, children=[
            _section_title(f"Detailed Metrics — Scale-Invariant {N_SI} Features"),
            dash_table.DataTable(
                columns=_MODEL_COLS, data=_model_table(model_si),
                style_cell=_TABLE_CELL_STYLE, style_header=_TABLE_HEADER_STYLE,
            ),
        ]),
    ])


def _tab_importance():
    return html.Div([
        html.Div(style=_SECTION_STYLE, children=[
            _section_title("Feature Importance — Three Methods"),
            html.P("L1 sparsity, permutation importance, and RFECV provide "
                   "complementary views. Consensus ranking averages all three.",
                   style={'fontSize': '0.85rem', 'color': '#64748b', 'marginBottom': '16px'}),
            dcc.Graph(figure=_fig_importance(), config={'displayModeBar': False}),
        ]),
        html.Div(style=_SECTION_STYLE, children=[
            _section_title("Consensus Ranking Table"),
            dash_table.DataTable(
                columns=_IMP_COLS, data=importance_table_data,
                sort_action='native', filter_action='native',
                page_size=20,
                style_cell=_TABLE_CELL_STYLE, style_header=_TABLE_HEADER_STYLE,
                style_data_conditional=[
                    {'if': {'column_id': 'rfecv_selected',
                            'filter_query': '{rfecv_selected} = "*"'},
                     'backgroundColor': '#d1fae5', 'color': '#065f46',
                     'fontWeight': '700'},
                    {'if': {'column_id': 'scale_invariant',
                            'filter_query': '{scale_invariant} = "Yes"'},
                     'backgroundColor': '#ecfdf5', 'color': '#047857'},
                ],
            ),
        ]),
        html.Div(style=_SECTION_STYLE, children=[
            _section_title("RFECV Curve"),
            html.P(f"Optimal number of features: {rfecv_n}. "
                   f"Selected: {', '.join(rfecv_selected)}.",
                   style={'fontSize': '0.85rem', 'color': '#64748b', 'marginBottom': '16px'}),
            dcc.Graph(figure=_fig_rfecv(), config={'displayModeBar': False}),
        ]),
    ])


def _tab_subsets():
    return html.Div([
        html.Div(style=_SECTION_STYLE, children=[
            _section_title("Feature Subset Comparison — F1 Score"),
            html.P("Compares F1 across all models using three feature sets: "
                   "all features, RFECV-selected, and scale-invariant only.",
                   style={'fontSize': '0.85rem', 'color': '#64748b', 'marginBottom': '16px'}),
            dcc.Graph(figure=_fig_subset_comparison(), config={'displayModeBar': False}),
        ]),
        html.Div(style={**_SECTION_STYLE,
                         'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr',
                         'gap': '20px'}, children=[
            html.Div([
                html.H4(f"All {N_FEATURES} Features", style={'color': PALETTE['blue']}),
                html.P(f"Best: {best_all_name}", style={'fontWeight': '700'}),
                html.P(f"F1 = {model_all[best_all_name]['f1'][0]:.3f} "
                       f"+/- {model_all[best_all_name]['f1'][1]:.3f}"),
            ]),
            html.Div([
                html.H4(f"RFECV {rfecv_n} Features", style={'color': PALETTE['green']}),
                html.P(f"Best: {best_sel_name}", style={'fontWeight': '700'}),
                html.P(f"F1 = {model_sel[best_sel_name]['f1'][0]:.3f} "
                       f"+/- {model_sel[best_sel_name]['f1'][1]:.3f}"),
                html.P(', '.join(rfecv_selected),
                       style={'fontSize': '0.82rem', 'color': '#64748b'}),
            ]),
            html.Div([
                html.H4(f"Scale-Invariant {N_SI}", style={'color': PALETTE['purple']}),
                html.P(f"Best: {best_si_name}", style={'fontWeight': '700'}),
                html.P(f"F1 = {model_si[best_si_name]['f1'][0]:.3f} "
                       f"+/- {model_si[best_si_name]['f1'][1]:.3f}"),
            ]),
        ]),
    ])


if __name__ == '__main__':
    print(f"\n  Supervised Study Dashboard at http://127.0.0.1:8051")
    print(f"  Press Ctrl+C to stop.\n")
    app.run(debug=False, port=8051)
