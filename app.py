import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoS₂ HER Predictor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.03em; }

.method-badge {
    display: inline-flex; align-items: center; gap: 10px;
    padding: 12px 20px; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1em; font-weight: 600;
    border-left: 5px solid; margin-bottom: 8px;
}
.score-bar-wrap { margin: 6px 0 2px 0; }
.score-bar-bg {
    background: rgba(255,255,255,0.08); border-radius: 2px;
    height: 8px; width: 100%; overflow: hidden;
}
.score-bar-fill { height: 8px; border-radius: 2px; transition: width 0.4s; }
.descriptor-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 6px; padding: 14px 16px; margin-bottom: 8px;
}
.descriptor-card .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72em; color: #888; text-transform: uppercase; letter-spacing: 0.08em;
}
.descriptor-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5em; font-weight: 600; margin: 2px 0;
}
.descriptor-card .note { font-size: 0.78em; color: #aaa; }
.ref-chip {
    display: inline-block; background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 3px; padding: 1px 7px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72em; color: #aaa;
    margin: 2px;
}
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75em; text-transform: uppercase;
    letter-spacing: 0.12em; color: #666;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 6px; margin: 20px 0 12px 0;
}
.stMetric label { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78em !important; }
.stMetric [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = {
        'sample':      ['MoS-T600','MoS-T700','MoS-T800',
                        'MoS-N5','MoS-N10','MoS-N20','MoS-N30','MoS-N50',
                        'MoS-M2.0','MoS-M2.5','MoS-M3.0','MoS-M6.0','MoS-M8.0','MoS-M9.0'],
        'series':      ['T','T','T','N','N','N','N','N','M','M','M','M','M','M'],
        'temp':        [600,700,800,800,800,800,800,800,800,800,800,800,800,800],
        'cycles':      [50,50,50,5,10,20,30,50,50,50,50,50,50,50],
        's_thick':     [9.0,9.0,9.0,3.0,3.0,3.0,3.0,3.0,2.0,2.5,3.0,6.0,8.0,9.0],
        'layer_n':     [12,14,18,2,5,9,13,20,20,20,20,20,20,20],
        'mo_s_ratio':  [0.49,0.48,0.46,0.62,0.56,0.52,0.50,0.47,0.82,0.76,0.65,0.52,0.48,0.46],
        'raman':       [2.41,2.34,2.29,1.01,1.63,1.85,1.78,1.99,1.70,1.97,1.99,2.05,2.24,2.29],
        'resistivity': [15.98,16.52,19.26,7.75,8.99,11.08,11.40,12.45,9.01,9.50,12.45,15.09,17.14,19.26],
        'ecsa':        [6.7,6.5,3.5,4.5,8.0,6.5,6.3,6.5,4.3,6.3,6.5,9.2,4.7,3.5],
        'loading':     [24.7,24.7,24.7,1.9,3.7,7.4,11.1,18.5,17.5,18.0,18.5,21.6,23.7,24.7],
        'eta':         [-0.46,-0.48,-0.58,-0.43,-0.33,-0.39,-0.35,-0.35,-0.58,-0.49,-0.35,-0.35,-0.52,-0.58],
        'tafel':       [136,257,297,161,80,105,93,114,484,253,114,91,223,297],
        'rct':         [98.4,113.0,193.3,136.5,52.8,76.9,59.0,64.0,161.2,104.5,64.0,45.5,124.5,193.3],
        'tof_ecsa':    [5.7,5.2,5.7,9.9,13.0,11.4,9.9,8.3,6.2,4.6,8.3,6.7,5.1,5.7],
        'tof_mass':    [1.6,1.4,0.8,22.9,24.9,9.9,5.5,2.9,1.6,1.6,2.9,2.9,1.0,0.8],
    }
    return pd.DataFrame(data)

df = load_data()

TARGETS = {
    'eta':         ('Overpotential η', 'V',             'max'),
    'tafel':       ('Tafel slope',      'mV/dec',        'min'),
    'rct':         ('Rct',              'Ω·cm²',         'min'),
    'raman':       ('Raman A₁g/E₂g',   '',              'min'),
    'resistivity': ('Resistivity',      'Ω·cm',          'min'),
    'tof_ecsa':    ('TOF (ECSA)',        'nmol/cm²/s',    'max'),
    'tof_mass':    ('TOF (mass)',        'nmol/µg/s',     'max'),
}

FEATURES = ['layer_n', 'mo_s_ratio', 'ecsa']
FEATURE_LABELS = {
    'layer_n':    'Layer #',
    'mo_s_ratio': 'Mo/S ratio',
    'ecsa':       'ECSA (cm²)',
}
FEATURE_RANGES = {
    'layer_n':    (1, 20),
    'mo_s_ratio': (0.45, 0.90),
    'ecsa':       (2.0, 12.0),
}

SERIES_COLORS  = {'T': '#4E9AF1', 'N': '#2DCE89', 'M': '#F5A623'}
SERIES_LABELS  = {'T': 'T-series (Temp)', 'N': 'N-series (Cycles)', 'M': 'M-series (S-thick)'}

METHOD_COLORS  = {
    'mbe':  '#2DCE89',
    'both': '#F5A623',
    'cvd':  '#4E9AF1',
}


# ── GP Models ─────────────────────────────────────────────────────────────────
@st.cache_resource
def train_models():
    X = df[FEATURES].values.astype(float)
    n = X.shape[1]
    gp_models, gp_scores, sx_dict, sy_dict, loo_stds_dict = {}, {}, {}, {}, {}
    rf_models, rf_scores, rf_imps = {}, {}, {}
    loo = LeaveOneOut()

    for key in TARGETS:
        y = df[key].values.astype(float)
        sx = StandardScaler().fit(X)
        sy = StandardScaler().fit(y.reshape(-1,1))

        kernel = (C(1.0,(1e-3,1e3))
                  * Matern(length_scale=[1.0]*n,
                           length_scale_bounds=[(0.01,100)]*n, nu=2.5)
                  + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5,10)))

        loo_means, loo_stds_list = [], []
        for tr, te in loo.split(X):
            sx_l = StandardScaler().fit(X[tr])
            sy_l = StandardScaler().fit(y[tr].reshape(-1,1))
            gp_l = GaussianProcessRegressor(
                kernel=C(1.0,(1e-3,1e3))*Matern(length_scale=[1.0]*n,
                         length_scale_bounds=[(0.01,100)]*n,nu=2.5)
                       +WhiteKernel(0.1,(1e-5,10)),
                n_restarts_optimizer=5, normalize_y=False, alpha=1e-6)
            gp_l.fit(sx_l.transform(X[tr]),
                     sy_l.transform(y[tr].reshape(-1,1)).ravel())
            m_s, std_s = gp_l.predict(sx_l.transform(X[te]), return_std=True)
            loo_means.append(sy_l.inverse_transform(m_s.reshape(-1,1)).ravel()[0])
            loo_stds_list.append(std_s[0]*sy_l.scale_[0])

        loo_means = np.array(loo_means)
        avg_err   = np.mean(np.abs(y - loo_means))
        avg_std   = np.mean(loo_stds_list)
        calib     = avg_err/avg_std if avg_std > 0 else 1.0

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                      normalize_y=False, alpha=1e-6)
        gp.fit(sx.transform(X), sy.transform(y.reshape(-1,1)).ravel())

        gp_models[key]     = gp
        gp_scores[key]     = {'r2':r2_score(y,loo_means),
                              'mae':mean_absolute_error(y,loo_means),
                              'loo_preds':loo_means, 'calib':calib}
        sx_dict[key]       = sx
        sy_dict[key]       = sy
        loo_stds_dict[key] = np.array(loo_stds_list)

        # RF
        rf = RandomForestRegressor(n_estimators=300, max_depth=4,
                                   min_samples_leaf=2, random_state=42)
        preds = np.zeros(len(y))
        for tr,te in loo.split(X):
            rf.fit(X[tr],y[tr]); preds[te]=rf.predict(X[te])
        rf.fit(X,y)
        rf_models[key] = rf
        rf_scores[key] = {'r2':r2_score(y,preds),'mae':mean_absolute_error(y,preds),'loo_preds':preds}
        rf_imps[key]   = rf.feature_importances_

    return gp_models, gp_scores, sx_dict, sy_dict, loo_stds_dict, rf_models, rf_scores, rf_imps

with st.spinner("Training GP + RF models… (first load only)"):
    gp_models, gp_scores, sx_dict, sy_dict, loo_stds_dict, \
    rf_models, rf_scores, rf_imps = train_models()


def gp_predict(key, ln, msr, ecsa_v):
    X_new = np.array([[ln, msr, ecsa_v]])
    sx = sx_dict[key]; sy = sy_dict[key]; gp = gp_models[key]
    m_s, std_s = gp.predict(sx.transform(X_new), return_std=True)
    mean  = sy.inverse_transform(m_s.reshape(-1,1)).ravel()[0]
    std   = std_s[0]*sy.scale_[0]*gp_scores[key]['calib']
    return mean, mean-1.96*std, mean+1.96*std, std

def predict_all(ln, msr, ecsa_v):
    return {k: gp_predict(k, ln, msr, ecsa_v)[0] for k in TARGETS}


# ── CVD/MBE Scoring ───────────────────────────────────────────────────────────
def score_method(layer_n, mo_s_ratio, ecsa_v, rct_v=None):
    """
    Returns (label, color_key, score, max_score, reasons_list).
    Each reason is a dict: {criterion, points, max_points, ref, detail}.
    """
    reasons = []
    total   = 0
    MAX     = 8   # 3 + 3 + 1 + 1

    # Layer #
    if layer_n <= 3:
        pts = 3
        detail = f"≤3L → onset −0.10 V vs RHE, k⁰ ~250 cm/s (vs 1.5 cm/s at 3L — 167× advantage)"
        refs   = ["Manyepedza 2022", "Choudhury §2.3"]
    elif layer_n <= 6:
        pts = 2
        detail = f"4–6L few-layer: near-optimal zone (Jeon N10 ~5L is best N-series sample)"
        refs   = ["Manyepedza 2022", "Choudhury §2.2"]
    elif layer_n <= 12:
        pts = 1
        detail = f"7–12L multi-layer: CVD nucleation density unstable — MBE preferred"
        refs   = ["Choudhury §3.1"]
    else:
        pts = 0
        detail = f"≥13L thick film: CVD viable when Mo/S is near-stoichiometric"
        refs   = ["Choudhury §2.2", "Jeon T-series"]
    total += pts
    reasons.append({'criterion':'Layer #','points':pts,'max':3,'refs':refs,'detail':detail})

    # Mo/S ratio
    if mo_s_ratio > 0.72:
        pts = 3
        detail = f"Highly Mo-rich: XANES shows residual Mo⁰ peaks — CVD cannot reach this regime"
        refs   = ["Sherwood 2024", "Choudhury §2.1"]
    elif mo_s_ratio > 0.58:
        pts = 2
        detail = f"Mo⁰/MoS₂ coexistence zone: CVD S-overpressure pushes toward stoichiometric"
        refs   = ["Choudhury §2.1"]
    elif mo_s_ratio >= 0.50:
        pts = 1
        detail = f"Slightly S-deficient: CVD overshoots toward S/Mo=2.2"
        refs   = ["Sherwood 2024"]
    else:
        pts = 0
        detail = f"Near-stoichiometric (S/Mo≥2.0): CVD S-rich atmosphere sufficient"
        refs   = ["Manyepedza 2022 XPS", "Choudhury §2.2"]
    total += pts
    reasons.append({'criterion':'Mo/S ratio','points':pts,'max':3,'refs':refs,'detail':detail})

    # ECSA
    if ecsa_v >= 8.0:
        pts = 1
        detail = f"ECSA ≥8.0 cm²: MBE wafer-scale uniformity maximises accessible edge sites"
        refs   = ["Jeon 2026 (N10: 8.0, M6.0: 9.2)"]
    else:
        pts = 0
        detail = f"ECSA <8.0 cm²: no additional constraint on method"
        refs   = []
    total += pts
    reasons.append({'criterion':'ECSA','points':pts,'max':1,'refs':refs,'detail':detail})

    # Rct
    rct_use = rct_v if rct_v is not None else gp_predict('rct', layer_n, mo_s_ratio, ecsa_v)[0]
    if rct_use < 55:
        pts = 1
        detail = f"Rct={rct_use:.0f} Ω·cm² <55: needs metallic Mo⁰ domains — MBE flux control required"
        refs   = ["Jeon 2026 (N10: 52.8, M6.0: 45.5)"]
    else:
        pts = 0
        detail = f"Rct={rct_use:.0f} Ω·cm² ≥55: no additional constraint"
        refs   = []
    total += pts
    reasons.append({'criterion':'Rct','points':pts,'max':1,'refs':refs,'detail':detail})

    # Decision
    if total >= 3:
        label    = "🔬 Physical Method (MBE)"
        col_key  = 'mbe'
    elif total >= 1:
        label    = "⚗️ Both viable — MBE preferred"
        col_key  = 'both'
    else:
        label    = "🧪 Chemical Method (CVD/PVT)"
        col_key  = 'cvd'

    return label, col_key, total, MAX, reasons


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗️ MoS₂ HER Predictor")
    st.markdown("<div style='font-size:0.78em;color:#666;margin-bottom:16px;'>"
                "Jeon et al. ACS Nano 2026 · 13-paper framework</div>",
                unsafe_allow_html=True)

    st.markdown('<div class="section-header">KEY DESCRIPTORS</div>', unsafe_allow_html=True)
    st.caption("Move sliders → predictor updates in real time")

    layer_n    = st.slider("Layer #", 1, 20, 5, 1,
        help="⚠ Estimated from XRD Scherrer (002) ÷ 0.615 nm/layer (AFM: Manyepedza 2022).\n"
             "≤3L: onset −0.10 V vs RHE, k⁰ ~250 cm/s.")
    mo_s_ratio = st.slider("Mo/S atomic ratio", 0.45, 0.90, 0.56, 0.01,
        help="⚠ Estimated from XANES/EXAFS (Jeon 2026).\n"
             "Stoichiometric = 0.455 (S/Mo=2.2). Mo-rich limit = 0.893.")
    ecsa_val   = st.slider("ECSA (cm²)", 2.0, 12.0, 8.0, 0.5,
        help="✅ Measured in Jeon 2026. Range: 3.5–9.2 cm².")

    # Closest match
    df_dist = df.copy()
    df_dist['dist'] = df.apply(lambda r: np.sqrt(
        ((r.layer_n    - layer_n)    / 18)    **2 +
        ((r.mo_s_ratio - mo_s_ratio) / 0.36)  **2 +
        ((r.ecsa       - ecsa_val)   / 6.0)   **2), axis=1)
    best_match = df_dist.nsmallest(1,'dist').iloc[0]
    dist_val   = df_dist['dist'].min()

    if dist_val < 0.15:
        st.success(f"✓ Closest sample: **{best_match['sample']}**")
    elif dist_val < 0.40:
        st.info(f"≈ Nearest: **{best_match['sample']}** (interpolating)")
    else:
        st.warning(f"⚠ Extrapolating — nearest: **{best_match['sample']}**")

    # Method badge
    m_label, m_col_key, m_score, m_max, m_reasons = score_method(layer_n, mo_s_ratio, ecsa_val)
    m_color = METHOD_COLORS[m_col_key]
    pct     = int(m_score / m_max * 100)

    st.markdown('<div class="section-header">SYNTHESIS METHOD</div>', unsafe_allow_html=True)
    st.markdown(
        f"<div class='method-badge' style='background:{m_color}18;"
        f"border-color:{m_color};color:{m_color};'>{m_label}</div>"
        f"<div class='score-bar-wrap'>"
        f"  <div style='font-size:0.72em;color:#666;font-family:IBM Plex Mono,monospace;"
        f"margin-bottom:3px;'>MBE score: {m_score}/{m_max}</div>"
        f"  <div class='score-bar-bg'>"
        f"    <div class='score-bar-fill' style='width:{pct}%;background:{m_color};'></div>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True)

    with st.expander("Scoring breakdown", expanded=False):
        for r in m_reasons:
            dot_color = m_color if r['points'] > 0 else '#444'
            st.markdown(
                f"**{r['criterion']}**: {r['points']}/{r['max']} pts  \n"
                f"{r['detail']}  \n"
                + " ".join([f"<span class='ref-chip'>{ref}</span>" for ref in r['refs']]),
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">NAVIGATION</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "📊 Predictor",
        "📈 Trend Curves",
        "🗺 2D Heatmaps",
        "🌐 3D Explorer",
        "🔄 Inverse Predictor",
        "🧮 Feature Importance",
        "📚 Theoretical Basis",
        "ℹ️ About",
    ], label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Predictor":
    st.markdown("# MoS₂ HER Predictor")
    st.markdown(f"<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
                f"Gaussian Process · Jeon et al. ACS Nano 2026 · 14 MBE samples · 1M KOH</div>",
                unsafe_allow_html=True)

    # Method banner
    m_color = METHOD_COLORS[m_col_key]
    st.markdown(
        f"<div style='background:{m_color}12;border:1.5px solid {m_color}40;"
        f"border-left:5px solid {m_color};padding:14px 20px;border-radius:6px;"
        f"margin-bottom:20px;display:flex;align-items:center;gap:20px;'>"
        f"<div style='font-size:1.3em;font-weight:700;color:{m_color};"
        f"font-family:IBM Plex Mono,monospace;'>{m_label}</div>"
        f"<div style='color:#888;font-size:0.85em;'>Score {m_score}/{m_max} · "
        f"Layer# {layer_n} · Mo/S {mo_s_ratio:.2f} · ECSA {ecsa_val:.1f} cm²</div>"
        f"</div>",
        unsafe_allow_html=True)

    # Predict
    if dist_val < 0.05:
        vals   = {k: best_match[k] for k in TARGETS}
        source = f"Experimental data — {best_match['sample']} (Jeon 2026 Table 1)"
        gp_ci  = None
    else:
        vals   = predict_all(layer_n, mo_s_ratio, ecsa_val)
        source = "GP prediction (calibrated 95% credible interval)"
        gp_ci  = {k: dict(zip(['mean','lower','upper','std'], gp_predict(k, layer_n, mo_s_ratio, ecsa_val)))
                  for k in TARGETS}

    st.caption(f"Source: {source}")

    # ── Key descriptor cards ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">KEY DESCRIPTORS</div>', unsafe_allow_html=True)
    kc1, kc2, kc3 = st.columns(3)
    desc_cards = [
        (kc1, "Layer #", f"{layer_n}", "capas",
         "🟢 Optimal ≤3L · k⁰×167 (Manyepedza 2022)" if layer_n<=3
         else ("🟡 Few-layer" if layer_n<=6 else "🔴 Thick film"),
         "⚠ XRD Scherrer estimate"),
        (kc2, "Mo/S ratio", f"{mo_s_ratio:.2f}", "",
         "🟢 Mo⁰/MoS₂ coexistence (optimal)" if 0.55<=mo_s_ratio<=0.72
         else ("🟡 Near-stoich" if mo_s_ratio<0.55 else "🔴 Highly Mo-rich"),
         "⚠ XANES/EXAFS estimate"),
        (kc3, "ECSA", f"{ecsa_val:.1f}", "cm²",
         "🟢 High — edge sites accessible" if ecsa_val>=7 else "🟡 Moderate",
         "✅ Measured, Jeon 2026"),
    ]
    for col, label, val, unit, status, note in desc_cards:
        with col:
            st.markdown(
                f"<div class='descriptor-card'>"
                f"<div class='label'>{label}</div>"
                f"<div class='value'>{val} <span style='font-size:0.6em;color:#888;'>{unit}</span></div>"
                f"<div class='note'>{status}</div>"
                f"<div class='note' style='margin-top:4px;color:#555;'>{note}</div>"
                f"</div>", unsafe_allow_html=True)

    # ── Performance metrics ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">PREDICTED PERFORMANCE METRICS</div>',
                unsafe_allow_html=True)
    cols = st.columns(4)
    metrics_order = ['eta','tafel','rct','tof_ecsa','tof_mass','raman','resistivity']
    thresholds = {
        'eta':(-0.38,-0.50),'tafel':(110,200),'rct':(70,130),
        'raman':(1.8,2.2),'resistivity':(12,17),'tof_ecsa':(9,6),'tof_mass':(5,2)
    }
    for i, key in enumerate(metrics_order):
        name, unit, better = TARGETS[key]
        v = vals[key]
        col = cols[i % 4]
        if key in thresholds:
            g, b = thresholds[key]
            color = "normal" if (v>=g if better=='max' else v<=g) \
                    else ("off" if (v<=b if better=='max' else v>=b) else "inverse")
        else:
            color = "normal"
        fmt = f"{v:.2f}" if abs(v)<100 else f"{v:.0f}"
        if gp_ci:
            std = gp_ci[key]['std']
            col.metric(name, f"{fmt} {unit}",
                       delta=f"±{std:.2f}" if abs(std)<100 else f"±{std:.0f}",
                       delta_color="off")
        else:
            col.metric(name, f"{fmt} {unit}")

    # ── Radar chart ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">PERFORMANCE PROFILE</div>', unsafe_allow_html=True)
    # Normalise each metric 0→1 (best=1, worst=0) using dataset min/max
    radar_keys  = ['eta','tafel','rct','tof_ecsa','tof_mass','raman','resistivity']
    radar_names = ['η (overpot.)','Tafel','Rct','TOF(ECSA)','TOF(mass)','Raman','Resistivity']
    normed = []
    for key in radar_keys:
        _, _, better = TARGETS[key]
        col_vals = df[key].values
        vmin, vmax = col_vals.min(), col_vals.max()
        v = vals[key]
        n = (v - vmin) / (vmax - vmin + 1e-9)
        normed.append(n if better=='max' else 1-n)

    # Also add best sample and closest experimental for comparison
    normed_best, normed_closest = [], []
    best_exp = df.loc[df['eta'].idxmax()]         # best η overall (least negative)
    # actually best = least negative η
    best_exp = df.loc[df['eta'].idxmax()]
    closest_row = best_match
    for key in radar_keys:
        _, _, better = TARGETS[key]
        vmin = df[key].min(); vmax = df[key].max()
        bv = (best_exp[key]-vmin)/(vmax-vmin+1e-9)
        normed_best.append(bv if better=='max' else 1-bv)
        cv = (closest_row[key]-vmin)/(vmax-vmin+1e-9)
        normed_closest.append(cv if better=='max' else 1-cv)

    fig_radar = go.Figure()
    cats = radar_names + [radar_names[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=normed+[normed[0]], theta=cats,
        fill='toself', name='Your prediction',
        fillcolor=f"{m_color}30", line=dict(color=m_color, width=2)))
    fig_radar.add_trace(go.Scatterpolar(
        r=normed_closest+[normed_closest[0]], theta=cats,
        fill='toself', name=f'Closest: {best_match["sample"]}',
        fillcolor='rgba(255,255,255,0.04)', line=dict(color='#aaa', width=1.5, dash='dot')))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=9)),
                   angularaxis=dict(tickfont=dict(size=10))),
        showlegend=True, height=380,
        legend=dict(orientation='h', yanchor='bottom', y=-0.25),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=60, l=40, r=40))
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Closest samples table ────────────────────────────────────────────────
    st.markdown('<div class="section-header">3 CLOSEST EXPERIMENTAL SAMPLES</div>',
                unsafe_allow_html=True)
    df_dist2 = df.copy()
    df_dist2['dist'] = df.apply(lambda r: np.sqrt(
        ((r.layer_n    - layer_n)    / 18)   **2 +
        ((r.mo_s_ratio - mo_s_ratio) / 0.36) **2 +
        ((r.ecsa       - ecsa_val)   / 6.0)  **2), axis=1)
    closest = df_dist2.nsmallest(3,'dist')
    show_cols = ['sample','series','layer_n','mo_s_ratio','ecsa',
                 'eta','tafel','rct','tof_ecsa','tof_mass']
    st.dataframe(closest[show_cols].reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TREND CURVES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trend Curves":
    st.markdown("# Trend Curves")
    st.markdown("<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
                "How each descriptor drives performance — GP mean + 95% CI vs experimental data</div>",
                unsafe_allow_html=True)

    tc1, tc2 = st.columns([1,2])
    with tc1:
        target_tc = st.selectbox("Performance metric",
            options=list(TARGETS.keys()),
            format_func=lambda k: f"{TARGETS[k][0]} ({TARGETS[k][1]})")
    with tc2:
        feat_tc = st.selectbox("Descriptor to vary",
            options=FEATURES,
            format_func=lambda k: FEATURE_LABELS[k])

    name_tc, unit_tc, better_tc = TARGETS[target_tc]

    defaults = {'layer_n': layer_n, 'mo_s_ratio': mo_s_ratio, 'ecsa': ecsa_val}

    lo, hi = FEATURE_RANGES[feat_tc]
    x_range = np.linspace(lo, hi, 80)
    y_means, y_lows, y_highs = [], [], []
    for xv in x_range:
        row = {f: (xv if f == feat_tc else defaults[f]) for f in FEATURES}
        m, lo_, hi_, _ = gp_predict(target_tc, row['layer_n'], row['mo_s_ratio'], row['ecsa'])
        y_means.append(m); y_lows.append(lo_); y_highs.append(hi_)
    y_means = np.array(y_means)
    y_lows  = np.array(y_lows)
    y_highs = np.array(y_highs)

    # Experimental range mask
    exp_lo, exp_hi = df[feat_tc].min(), df[feat_tc].max()
    in_range = (x_range >= exp_lo) & (x_range <= exp_hi)

    fig_tc = go.Figure()

    # CI band
    fig_tc.add_trace(go.Scatter(
        x=np.concatenate([x_range, x_range[::-1]]),
        y=np.concatenate([y_highs, y_lows[::-1]]),
        fill='toself', fillcolor='rgba(78,154,241,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name='95% CI', showlegend=True))

    # GP mean — full range (dashed, extrapolation)
    fig_tc.add_trace(go.Scatter(x=x_range, y=y_means, mode='lines',
        line=dict(color='rgba(78,154,241,0.35)', width=1.5, dash='dot'),
        name='GP mean (extrapolation)', showlegend=True))

    # GP mean — interpolation range (solid)
    x_in = x_range[in_range]; y_in = y_means[in_range]
    fig_tc.add_trace(go.Scatter(x=x_in, y=y_in, mode='lines',
        line=dict(color='#4E9AF1', width=3),
        name='GP mean (interpolation)', showlegend=True))

    # Experimental data points, coloured by series
    exp_x = df[feat_tc].values; exp_y = df[target_tc].values
    for ser, scolor in SERIES_COLORS.items():
        mask = df['series'] == ser
        fig_tc.add_trace(go.Scatter(
            x=exp_x[mask], y=exp_y[mask], mode='markers',
            name=SERIES_LABELS[ser],
            marker=dict(size=11, color=scolor, line=dict(width=1.5, color='white')),
            text=df['sample'][mask],
            hovertemplate='<b>%{text}</b><br>'+FEATURE_LABELS[feat_tc]+'=%{x:.2f}'
                          '<br>'+name_tc+'=%{y:.3f} '+unit_tc+'<extra></extra>'))

    # Vertical line at current slider value
    cur_val = defaults[feat_tc]
    fig_tc.add_vline(x=cur_val, line_width=1.5, line_dash="dash",
                     line_color=METHOD_COLORS[m_col_key],
                     annotation_text=f"Current: {cur_val:.2f}",
                     annotation_font_color=METHOD_COLORS[m_col_key])

    # Shaded experimental range
    fig_tc.add_vrect(x0=exp_lo, x1=exp_hi,
                     fillcolor="rgba(255,255,255,0.03)", line_width=0,
                     annotation_text="Experimental range",
                     annotation_position="top left",
                     annotation_font_size=10, annotation_font_color="#555")

    fig_tc.update_layout(
        title=f"{name_tc} vs {FEATURE_LABELS[feat_tc]}"
              f"  |  other descriptors fixed at slider values",
        xaxis_title=FEATURE_LABELS[feat_tc],
        yaxis_title=f"{name_tc} ({unit_tc})",
        height=480,
        legend=dict(orientation='h', yanchor='bottom', y=-0.35),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    fig_tc.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.12)')
    fig_tc.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.12)')
    st.plotly_chart(fig_tc, use_container_width=True)

    # ── All 3 descriptors in a row ────────────────────────────────────────────
    st.markdown('<div class="section-header">ALL 3 DESCRIPTORS — OVERVIEW</div>',
                unsafe_allow_html=True)
    cols3 = st.columns(3)
    for fi, feat in enumerate(FEATURES):
        lo_f, hi_f = FEATURE_RANGES[feat]
        xr = np.linspace(lo_f, hi_f, 60)
        ym = []
        for xv in xr:
            row = {f: (xv if f == feat else defaults[f]) for f in FEATURES}
            m,_,_,_ = gp_predict(target_tc, row['layer_n'], row['mo_s_ratio'], row['ecsa'])
            ym.append(m)
        ym = np.array(ym)
        exp_xf = df[feat].values; exp_yf = df[target_tc].values
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=xr, y=ym, mode='lines',
            line=dict(color='#4E9AF1', width=2), showlegend=False))
        for ser, scolor in SERIES_COLORS.items():
            mask = df['series'] == ser
            fig_s.add_trace(go.Scatter(
                x=exp_xf[mask], y=exp_yf[mask], mode='markers',
                marker=dict(size=8, color=scolor, line=dict(width=1, color='white')),
                name=ser, showlegend=(fi==0), text=df['sample'][mask],
                hovertemplate='<b>%{text}</b><br>%{x:.2f} → %{y:.3f}<extra></extra>'))
        cur = defaults[feat]
        fig_s.add_vline(x=cur, line_dash='dash', line_color=METHOD_COLORS[m_col_key],
                        line_width=1.2)
        fig_s.update_layout(
            title=dict(text=FEATURE_LABELS[feat], font=dict(size=12)),
            xaxis_title=FEATURE_LABELS[feat], yaxis_title=f"{name_tc} ({unit_tc})",
            height=260, margin=dict(t=40,b=40,l=40,r=10),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        fig_s.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.10)')
        fig_s.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.10)')
        cols3[fi].plotly_chart(fig_s, use_container_width=True)

    st.caption(
        "Solid blue line = GP mean within experimental range. "
        "Dashed = extrapolation. Colored dots = experimental data (T/N/M series). "
        "Vertical dashed line = current slider value.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 2D HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺 2D Heatmaps":
    st.markdown("# 2D Heatmaps")
    st.markdown("<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
                "GP-predicted performance over pairs of descriptors. "
                "Third descriptor fixed at sidebar slider value.</div>",
                unsafe_allow_html=True)

    hc1, hc2 = st.columns(2)
    with hc1:
        target_hm = st.selectbox("Performance metric",
            options=list(TARGETS.keys()),
            format_func=lambda k: f"{TARGETS[k][0]} ({TARGETS[k][1]})")
    with hc2:
        axis_pair = st.selectbox("Axes (X × Y)", [
            "Layer# × Mo/S  (ECSA fixed)",
            "Layer# × ECSA  (Mo/S fixed)",
            "Mo/S × ECSA   (Layer# fixed)",
        ])

    name_hm, unit_hm, better_hm = TARGETS[target_hm]
    N = 40  # grid resolution

    defaults_hm = {'layer_n': layer_n, 'mo_s_ratio': mo_s_ratio, 'ecsa': ecsa_val}

    if axis_pair.startswith("Layer# × Mo/S"):
        xf, yf, fixed_f = 'layer_n', 'mo_s_ratio', 'ecsa'
        xlabel, ylabel   = 'Layer #', 'Mo/S ratio'
    elif axis_pair.startswith("Layer# × ECSA"):
        xf, yf, fixed_f = 'layer_n', 'ecsa', 'mo_s_ratio'
        xlabel, ylabel   = 'Layer #', 'ECSA (cm²)'
    else:
        xf, yf, fixed_f = 'mo_s_ratio', 'ecsa', 'layer_n'
        xlabel, ylabel   = 'Mo/S ratio', 'ECSA (cm²)'

    xlo, xhi = FEATURE_RANGES[xf]
    ylo, yhi = FEATURE_RANGES[yf]
    xgrid = np.linspace(xlo, xhi, N)
    ygrid = np.linspace(ylo, yhi, N)

    Z = np.zeros((N, N))
    for i, yv in enumerate(ygrid):
        for j, xv in enumerate(xgrid):
            row = {xf: xv, yf: yv, fixed_f: defaults_hm[fixed_f]}
            m,_,_,_ = gp_predict(target_hm,
                                  row['layer_n'], row['mo_s_ratio'], row['ecsa'])
            Z[i, j] = m

    # Color scale: green=better, red=worse
    cs = 'RdYlGn' if better_hm == 'max' else 'RdYlGn_r'

    fig_hm = go.Figure(data=go.Heatmap(
        z=Z, x=xgrid, y=ygrid,
        colorscale=cs,
        colorbar=dict(title=f"{name_hm}<br>({unit_hm})", titleside='right'),
        hoverongaps=False,
        hovertemplate=f'{xlabel}=%{{x:.2f}}<br>{ylabel}=%{{y:.2f}}'
                      f'<br>{name_hm}=%{{z:.3f}} {unit_hm}<extra></extra>'))

    # Overlay experimental data
    exp_x_hm = df[xf].values; exp_y_hm = df[yf].values; exp_z_hm = df[target_hm].values
    for ser, scolor in SERIES_COLORS.items():
        mask = df['series'] == ser
        fig_hm.add_trace(go.Scatter(
            x=exp_x_hm[mask], y=exp_y_hm[mask], mode='markers+text',
            marker=dict(size=12, color=scolor, symbol='circle',
                        line=dict(width=2, color='white')),
            text=df['sample'][mask],
            textposition='top center',
            textfont=dict(size=9, color='white'),
            name=SERIES_LABELS[ser],
            hovertemplate='<b>%{text}</b><br>'+xlabel+'=%{x:.2f}'
                          '<br>'+ylabel+'=%{y:.2f}'
                          '<br>'+name_hm+f'=%{{customdata:.3f}} {unit_hm}<extra></extra>',
            customdata=exp_z_hm[mask]))

    # Current slider position
    fig_hm.add_trace(go.Scatter(
        x=[defaults_hm[xf]], y=[defaults_hm[yf]], mode='markers',
        marker=dict(size=16, color=METHOD_COLORS[m_col_key], symbol='star',
                    line=dict(width=2, color='white')),
        name='Your position', showlegend=True))

    fig_hm.update_layout(
        title=f"{name_hm} — {xlabel} × {ylabel}  |  {FEATURE_LABELS[fixed_f]} fixed at {defaults_hm[fixed_f]:.2f}",
        xaxis_title=xlabel, yaxis_title=ylabel,
        height=540,
        legend=dict(orientation='h', yanchor='bottom', y=-0.22),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── CVD/MBE method map ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">CVD vs MBE METHOD MAP</div>',
                unsafe_allow_html=True)
    st.caption("Score ≥3 = MBE · Score 1–2 = Both (MBE preferred) · Score 0 = CVD. "
               "Only Layer# × Mo/S pair shown (strongest drivers of method choice).")

    NM = 50
    ln_grid  = np.linspace(1, 20, NM)
    msr_grid = np.linspace(0.45, 0.90, NM)
    Zm = np.zeros((NM, NM))
    for i, msr_v in enumerate(msr_grid):
        for j, ln_v in enumerate(ln_grid):
            _, _, sc, _, _ = score_method(ln_v, msr_v, ecsa_val)
            Zm[i, j] = sc

    fig_map = go.Figure(data=go.Heatmap(
        z=Zm, x=ln_grid, y=msr_grid,
        colorscale=[
            [0.0,  '#4E9AF155'],   # 0 = CVD
            [0.375,'#4E9AF1'],
            [0.375,'#F5A623'],     # 1-2 = Both
            [0.75, '#F5A623'],
            [0.75, '#2DCE89'],     # 3+ = MBE
            [1.0,  '#2DCE89'],
        ],
        zmin=0, zmax=4,
        colorbar=dict(title='MBE score', tickvals=[0,1,2,3,4]),
        hovertemplate='Layer#=%{x:.0f}<br>Mo/S=%{y:.2f}<br>Score=%{z:.0f}<extra></extra>'))

    # Overlay real samples
    for ser, scolor in SERIES_COLORS.items():
        mask = df['series'] == ser
        _, _, sc_ser, _, _ = score_method(0,0,0)  # placeholder
        fig_map.add_trace(go.Scatter(
            x=df['layer_n'][mask], y=df['mo_s_ratio'][mask], mode='markers+text',
            marker=dict(size=11, color='white', line=dict(width=2, color=scolor)),
            text=df['sample'][mask], textposition='top center',
            textfont=dict(size=8, color='white'),
            name=SERIES_LABELS[ser]))

    # Current position
    fig_map.add_trace(go.Scatter(
        x=[layer_n], y=[mo_s_ratio], mode='markers',
        marker=dict(size=16, color=METHOD_COLORS[m_col_key], symbol='star',
                    line=dict(width=2, color='white')),
        name='Your position'))

    # Decision boundary lines
    fig_map.add_hline(y=0.58, line_dash='dot', line_color='white', line_width=1,
                      annotation_text="Mo/S=0.58 (MBE preferred)", annotation_font_color='white')
    fig_map.add_hline(y=0.72, line_dash='dot', line_color='#2DCE89', line_width=1,
                      annotation_text="Mo/S=0.72 (MBE required)", annotation_font_color='#2DCE89')
    fig_map.add_vline(x=3, line_dash='dot', line_color='#2DCE89', line_width=1,
                      annotation_text="Layer#=3", annotation_font_color='#2DCE89')
    fig_map.add_vline(x=12, line_dash='dot', line_color='white', line_width=1,
                      annotation_text="Layer#=12", annotation_font_color='white')

    fig_map.update_layout(
        title=f"CVD vs MBE decision map — Layer# × Mo/S  |  ECSA fixed at {ecsa_val:.1f} cm²",
        xaxis_title='Layer #', yaxis_title='Mo/S ratio',
        height=480,
        legend=dict(orientation='h', yanchor='bottom', y=-0.25),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_map, use_container_width=True)

    st.caption(
        "🟢 Green = MBE required (score ≥3) · 🟡 Amber = Both viable (score 1–2) · "
        "🔵 Blue = CVD sufficient (score 0). "
        "White dots = experimental Jeon samples. ★ = your current slider position.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 3D EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌐 3D Explorer":
    st.markdown("# 3D Descriptor Space Explorer")
    st.markdown("<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
                "Layer# × Mo/S × ECSA — colour = selected performance metric. "
                "Rotate, zoom, hover for details.</div>",
                unsafe_allow_html=True)

    t3c1, t3c2 = st.columns(2)
    with t3c1:
        target_3d = st.selectbox("Color metric",
            options=list(TARGETS.keys()),
            format_func=lambda k: f"{TARGETS[k][0]} ({TARGETS[k][1]})")
    with t3c2:
        show_surf = st.checkbox("Show GP surface slice (Mo/S fixed at slider)", value=True)

    name_3d, unit_3d, better_3d = TARGETS[target_3d]

    fig_3d = go.Figure()

    # ── GP surface slice (Layer# × ECSA at fixed Mo/S) ───────────────────────
    if show_surf:
        N3 = 25
        ln3  = np.linspace(1, 20, N3)
        ec3  = np.linspace(2, 12, N3)
        Zs   = np.zeros((N3, N3))
        for i, ev in enumerate(ec3):
            for j, lv in enumerate(ln3):
                Zs[i,j] = gp_predict(target_3d, lv, mo_s_ratio, ev)[0]
        fig_3d.add_trace(go.Surface(
            x=ln3, y=ec3, z=Zs,
            colorscale='RdYlGn' if better_3d=='max' else 'RdYlGn_r',
            opacity=0.55, showscale=False,
            name=f'GP surface (Mo/S={mo_s_ratio:.2f})',
            hovertemplate='Layer#=%{x:.1f}<br>ECSA=%{y:.1f}<br>'+name_3d+'=%{z:.3f}<extra></extra>'))

    # ── Experimental data points ──────────────────────────────────────────────
    for ser, scolor in SERIES_COLORS.items():
        mask  = df['series'] == ser
        sub   = df[mask]
        zvals = sub[target_3d].values
        fig_3d.add_trace(go.Scatter3d(
            x=sub['layer_n'], y=sub['ecsa'], z=sub['mo_s_ratio'],
            mode='markers+text',
            marker=dict(size=8, color=zvals,
                        colorscale='RdYlGn' if better_3d=='max' else 'RdYlGn_r',
                        cmin=df[target_3d].min(), cmax=df[target_3d].max(),
                        line=dict(width=2, color='white')),
            text=sub['sample'],
            name=SERIES_LABELS[ser],
            hovertemplate='<b>%{text}</b><br>Layer#=%{x}<br>ECSA=%{y:.1f}'
                          '<br>Mo/S=%{z:.2f}<br>'+name_3d+f'=%{{marker.color:.3f}} {unit_3d}'
                          '<extra></extra>'))

    # ── Current position ──────────────────────────────────────────────────────
    cur_pred = gp_predict(target_3d, layer_n, mo_s_ratio, ecsa_val)[0]
    fig_3d.add_trace(go.Scatter3d(
        x=[layer_n], y=[ecsa_val], z=[mo_s_ratio],
        mode='markers',
        marker=dict(size=14, color=METHOD_COLORS[m_col_key], symbol='diamond',
                    line=dict(width=3, color='white')),
        name=f'Your position ({cur_pred:.3f} {unit_3d})'))

    fig_3d.update_layout(
        scene=dict(
            xaxis_title='Layer #',
            yaxis_title='ECSA (cm²)',
            zaxis_title='Mo/S ratio',
            xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(128,128,128,0.2)'),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(128,128,128,0.2)'),
        ),
        title=f"{name_3d} ({unit_3d}) in Layer# × ECSA × Mo/S space",
        height=620,
        legend=dict(orientation='h', yanchor='bottom', y=-0.12),
        paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_3d, use_container_width=True)

    st.info(
        f"**Your position:** Layer# {layer_n} · Mo/S {mo_s_ratio:.2f} · ECSA {ecsa_val:.1f} cm²  "
        f"→ GP predicts **{name_3d} = {cur_pred:.3f} {unit_3d}**  |  Method: **{m_label}**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INVERSE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Inverse Predictor":
    st.markdown("# Inverse Predictor")
    st.markdown("<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
                "Set target HER properties → find closest experimental match → get synthesis method.</div>",
                unsafe_allow_html=True)

    st.markdown('<div class="section-header">TARGET PERFORMANCE</div>', unsafe_allow_html=True)
    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1: t_eta   = st.slider("Target η (V)", -0.60, -0.25, -0.35, 0.01)
    with ic2: t_tafel = st.slider("Target Tafel (mV/dec)", 60, 300, 100, 5)
    with ic3: t_ecsa  = st.slider("Target ECSA (cm²)", 2.0, 12.0, 7.0, 0.5)
    with ic4: t_rct   = st.slider("Target Rct (Ω·cm²)", 20.0, 200.0, 60.0, 5.0)

    df_inv = df.copy()
    df_inv['perf_score'] = df_inv.apply(lambda r: np.sqrt(
        ((r.eta    - t_eta)   / 0.30) **2 +
        ((r.tafel  - t_tafel) / 250)  **2 +
        ((r.ecsa   - t_ecsa)  / 8)    **2 +
        ((r.rct    - t_rct)   / 180)  **2), axis=1)
    candidates = df_inv.nsmallest(3, 'perf_score')
    best_inv   = candidates.iloc[0]

    st.markdown('<div class="section-header">CLOSEST EXPERIMENTAL MATCHES</div>',
                unsafe_allow_html=True)
    show_inv = candidates[['sample','series','layer_n','mo_s_ratio','ecsa',
                            'eta','tafel','rct','tof_ecsa']].reset_index(drop=True)
    st.dataframe(show_inv, use_container_width=True)

    st.markdown('<div class="section-header">SYNTHESIS RECOMMENDATION</div>',
                unsafe_allow_html=True)
    inv_label, inv_col, inv_score, inv_max, inv_reasons = score_method(
        best_inv['layer_n'], best_inv['mo_s_ratio'],
        best_inv['ecsa'],    best_inv['rct'])
    inv_color = METHOD_COLORS[inv_col]

    st.markdown(
        f"<div style='background:{inv_color}12;border:2px solid {inv_color}40;"
        f"border-left:5px solid {inv_color};padding:16px 20px;border-radius:6px;'>"
        f"<div style='font-size:1.4em;font-weight:700;color:{inv_color};"
        f"font-family:IBM Plex Mono,monospace;'>{inv_label}</div>"
        f"<div style='color:#888;margin-top:6px;'>Best match: <b>{best_inv['sample']}</b> · "
        f"η={best_inv.eta:.2f} V · Tafel={best_inv.tafel:.0f} mV/dec · "
        f"ECSA={best_inv.ecsa:.1f} cm² · Rct={best_inv.rct:.1f} Ω·cm²</div>"
        f"<div style='margin-top:10px;'><b>Score: {inv_score}/{inv_max}</b></div>"
        f"</div>", unsafe_allow_html=True)

    st.markdown("**Scoring breakdown:**")
    for r in inv_reasons:
        st.markdown(f"• **{r['criterion']}** ({r['points']}/{r['max']} pts): {r['detail']}")

    st.markdown('<div class="section-header">TARGET SYNTHESIS PARAMETERS</div>',
                unsafe_allow_html=True)
    param_df = pd.DataFrame({
        'Parameter': ['Annealing temp','Deposition cycles','S-layer thickness','Layer #','Mo/S ratio'],
        'Value':     [f"{best_inv['temp']:.0f} °C", f"{best_inv['cycles']:.0f}",
                      f"{best_inv['s_thick']:.1f} Å", f"{best_inv['layer_n']:.0f}",
                      f"{best_inv['mo_s_ratio']:.2f}"],
        'Note':      ['Higher T → crystalline but fewer edge sites',
                      '~1 MoS₂ layer per 5 cycles (QCM calibrated, Jeon)',
                      'Key lever for Mo/S ratio and S-vacancy density',
                      '≤3L: MBE required (k⁰ ×167, Manyepedza 2022)',
                      '>0.58: Mo+MoS₂ coexistence — MBE Mo-flux control']
    })
    st.dataframe(param_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧮 Feature Importance":
    st.markdown("# Feature Importance")
    st.markdown("<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
                "Random Forest LOO feature importance. "
                "GP used for predictions; RF shown here for interpretability.</div>",
                unsafe_allow_html=True)

    st.markdown('<div class="section-header">MODEL PERFORMANCE — LOO CV</div>',
                unsafe_allow_html=True)
    perf_rows = []
    for k in TARGETS:
        n, u, _ = TARGETS[k]
        perf_rows.append({'Property':n,'Unit':u,
                          'GP R²':round(gp_scores[k]['r2'],3),
                          'GP MAE':round(gp_scores[k]['mae'],3),
                          'RF R²':round(rf_scores[k]['r2'],3),
                          'RF MAE':round(rf_scores[k]['mae'],3)})
    st.dataframe(pd.DataFrame(perf_rows), use_container_width=True)
    st.caption("⚠ n=14 — LOO scores have high variance. GP is primary predictor; RF here for importance only.")

    fi_colors = {'layer_n':'#9B59B6','mo_s_ratio':'#E84040','ecsa':'#2DCE89'}
    fi_names  = {'layer_n':'Layer #','mo_s_ratio':'Mo/S ratio','ecsa':'ECSA'}

    imp_target = st.selectbox("Property for importance",
        options=list(TARGETS.keys()),
        format_func=lambda k: f"{TARGETS[k][0]} ({TARGETS[k][1]})")

    imps = rf_imps[imp_target]
    fig_fi = go.Figure(go.Bar(
        x=[fi_names[f] for f in FEATURES],
        y=imps,
        marker_color=[fi_colors[f] for f in FEATURES],
        text=[f"{v:.3f}" for v in imps],
        textposition='outside'))
    fig_fi.update_layout(
        title=f"Feature importance — {TARGETS[imp_target][0]}",
        yaxis_title='Relative importance', yaxis_range=[0, max(imps)*1.25],
        height=320, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_fi, use_container_width=True)

    # Heatmap of all importances
    heat = np.array([[rf_imps[k][i] for i in range(3)] for k in TARGETS])
    heat_df = pd.DataFrame(heat,
                           index=[TARGETS[k][0] for k in TARGETS],
                           columns=[fi_names[f] for f in FEATURES])
    fig_heat = px.imshow(heat_df, text_auto=".2f", aspect="auto",
                         color_continuous_scale='Greens', zmin=0, zmax=1,
                         title="Feature importance matrix (all targets)")
    fig_heat.update_layout(height=360)
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: THEORETICAL BASIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Theoretical Basis":
    st.markdown("# Theoretical Framework")
    st.markdown("<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
                "13 papers integrated — DFT, electrochemistry, surface science, epitaxy.</div>",
                unsafe_allow_html=True)

    papers = [
        ("1 · Hanslin, Jónsson & Akola — PCCP 2023 (DFT)",
         "Exposed Mo edge sites: activation barriers <0.5 eV at 0 V vs SHE. Low Raman A₁g/E₂g → more Mo exposed → more active. Theoretical overpotential Mo₀ edge ≈0.25 V. S-bound H: Heyrovsky barriers >1.5 eV (kinetically dead). T>900°C → grain coalescence → η >−0.7 V expected."),
        ("2 · Li, Qin & Voiry — ACS Nano 2019",
         "S vacancies follow non-linear optimum: too few → inert basal plane, optimal → max TOF, too many → structural collapse. Direct analog to Jeon M-series: M3.0–M6.0 is the optimal S-deficiency window."),
        ("3 · Geng et al. — Nature Communications 2016",
         "1T metallic phase: ρ=0.48 Ω·cm, Rct≈1 Ω, η=−175 mV. Mixed Mo+MoS₂ domains in N10 and M3–M6 replicate this via metallic Mo pathways."),
        ("4 · Muhyuddin et al. — J. Energy Chemistry 2023 (review)",
         "Tafel <60 mV/dec → Heyrovsky RDS. 60–120 → Volmer–Heyrovsky. >120 → Volmer RDS (slow H₂O dissociation in alkaline KOH). N10 (80 mV/dec) is the only Jeon sample in Volmer–Heyrovsky regime."),
        ("5 · Jeon et al. — ACS Nano 2026 (experimental base)",
         "14 MBE samples in 1M KOH. Global optimum: MoS-N10 (η=−0.33V, Tafel=80, ECSA=8.0 cm², Rct=52.8). T↑ → resistivity↑, ECSA↓, Tafel↑. Cycles↑ → resistivity↑ monotonically. S-thickness non-linear optimum at 3–6 Å."),
        ("6 · Zhu et al. — Nature Communications 2019",
         "2H–2H and 2H–1T grain boundaries are active HER sites. ΔGH* = −0.13 eV for 2H–1T boundaries (≈ Pt at −0.18 eV). More boundaries → better performance. Tafel 73–75 mV/dec, stable >200 h."),
        ("7 · Yang et al. — RSC Advances 2023 (DFT)",
         "Defect + strain synergy: Vs, VMoS3, VMoS6 are active sites. Without strain: ΔGH*>0.22 eV. With 1% tensile: VMoS3 → ΔGH*≈0 eV. MBE on Si: ~1–2% tensile mismatch activates vacancy sites."),
        ("8 · Integrated mechanistic picture",
         "MoS-N10 optimum from convergence of 4 mechanisms: (1) intermediate thickness → ρ=8.99 Ω·cm; (2) MBE strain + Vs → ΔGH*≈0 eV; (3) high 2H–1T boundary density → ΔGH*≈−0.13 eV; (4) Mo conductive domains → Rct=52.8 Ω·cm². Synergy essential — no single mechanism explains the optimum alone."),
        ("9 · Tsai, Li, Park et al. — Nature Communications 2017",
         "EC desulfurization generates S-vacancies at ≥−1.0 V vs RHE. Optimal concentration 12.5–15.6% surface atoms (ΔGH*≈0 eV). MBE M2.0–M3.0 sits in this optimal window by design."),
        ("10 · Li, Qin, Ries & Voiry — ACS 2019 (Stage 1/2 framework)",
         "Stage 1 (<~20% S-vacancies): isolated point defects, moderate HER improvement. Stage 2 (>~50%): large undercoordinated Mo regions, TOF ~2 s⁻¹ at −160 mV in 0.1M KOH. M-series spans Stage 1→2 transition."),
        ("11 · Sherwood et al. — ACS Nano 2024 (XPS phase fingerprinting)",
         "XPS 4-peak model distinguishes 2H MoS₂, 1T MoS₂, MoS₂₋ₓ. Key: stoichiometric 2H = S/Mo=2.2–2.5 (Mo/S≈0.40–0.455). Ar⁺ bombardment drives S/Mo→1.1 (Mo/S→0.91) — upper limit. Validates Mo/S slider range (0.45–0.90)."),
        ("12 · Choudhury, Zhang, Al Balushi & Redwing — Penn State Review (CVD vs MBE)",
         "CVD: S/Mo vapor ratio varies with substrate position — cannot independently control stoichiometry + layer number + crystallinity. MBE: independent Mo flux (e-beam) and S flux (effusion cell + QCM), RHEED monitoring. CVD cannot reproduce <5L reliably. MBE: layer count set by deposition cycles. Conclusion: 4 key descriptors only independently tunable by MBE."),
        ("13 · Manyepedza et al. — J. Phys. Chem. C 2022 ★",
         "Impact electrochemistry (single NPs). Onset: 1–3 TL → −0.10 V vs RHE (GC confirmed). Electrodep.: −0.29 V. Bulk NPs: −0.49 V. k⁰: 1 TL→250 cm/s, 3 TL→1.5 cm/s (167× factor). AFM: 0.615 nm/TL (same value used for layer_n estimation in Jeon). XPS: S/Mo=2.2 → Mo/S=0.455 (stoichiometric endpoint). Faradaic efficiency 45–48%."),
    ]
    for title, body in papers:
        with st.expander(title):
            st.write(body)

    st.markdown('<div class="section-header">KEY DESCRIPTORS SUMMARY</div>',
                unsafe_allow_html=True)
    desc_df = pd.DataFrame({
        'Descriptor':    ['Layer # ⚠','Mo/S ratio ⚠','ECSA ✅','Raman A₁g/E₂g ✅',
                          'Resistivity ✅','Rct ✅','ΔGH* (DFT)'],
        'Measures':      ['Film thickness → edge/basal ratio + k⁰',
                          '2H MoS₂ ↔ MoS₂₋ₓ ↔ Mo⁰/MoS₂ phase',
                          'Electrochemically active surface area',
                          'Mo vs S edge site exposure',
                          'Electronic conductivity (Mo⁰ domains)',
                          'Interfacial charge transfer resistance',
                          'H adsorption free energy'],
        'Optimal':       ['≤3L → onset −0.10 V, k⁰~250 cm/s',
                          '0.55–0.72 (Mo⁰/MoS₂ coexistence)',
                          '>7 cm²','<1.8',
                          '<12 Ω·cm','<70 Ω·cm²','≈ 0 eV'],
        'Source':        ['⚠ XRD Scherrer / Manyepedza 2022 AFM',
                          '⚠ Scale: Sherwood 2024 + Manyepedza XPS',
                          '✅ Jeon 2026 Table 1','✅ Jeon 2026',
                          '✅ Jeon 2026','✅ Jeon 2026','DFT: Hanslin/Yang'],
    })
    st.dataframe(desc_df, use_container_width=True)
    st.caption("⚠ = estimated descriptor · ✅ = directly measured in Jeon et al. 2026")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("# About")
    st.markdown("""
**MoS₂ HER Trend Predictor** — theory-guided GP prediction for MBE-grown MoS₂ in 1M KOH.

### Experimental base
**Jeon et al., ACS Nano 2026** — 14 MBE samples on Si.
- **T-series**: Annealing temperature 600–800°C (cycles=50, S=9.0 Å)
- **N-series**: Deposition cycles 5–50 (800°C, S=3.0 Å)
- **M-series**: S-layer thickness 2.0–9.0 Å (800°C, cycles=50)

### Machine learning
| Component | Detail |
|---|---|
| Primary model | Gaussian Process (Matérn ν=2.5, ARD, calibrated 95% CI) |
| Secondary model | Random Forest (300 trees) — feature importance only |
| Validation | Leave-One-Out cross-validation (n=14) |
| Features | Layer #, Mo/S ratio, ECSA (3 key descriptors) |
| Targets | η, Tafel, Rct, Raman ratio, resistivity, TOF×2 |

### CVD vs MBE scoring
| Criterion | Max pts | Basis |
|---|---|---|
| Layer # | 3 | k⁰ kinetics (Manyepedza 2022) |
| Mo/S ratio | 3 | Phase control (Choudhury review) |
| ECSA | 1 | Edge site uniformity (Jeon 2026) |
| Rct | 1 | Mo⁰ domain requirement (Jeon 2026) |

Score ≥3 → MBE · Score 1–2 → Both (MBE preferred) · Score 0 → CVD

### New in this version
- **Trend Curves page**: GP mean + 95% CI vs each descriptor, all 3 in one view
- **2D Heatmaps page**: GP surface over descriptor pairs + CVD/MBE decision map
- **3D Explorer page**: Layer# × Mo/S × ECSA space, rotatable, GP surface slice
- **Scoring breakdown**: transparent per-criterion points with references
- **Radar chart**: performance profile vs closest experimental sample

⚠ With only 14 training samples, all predictions carry high uncertainty.
This tool is for **trend analysis and mechanistic understanding**, not precise numerical prediction.
    """)
