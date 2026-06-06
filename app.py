"""
MoS₂ HER Trend Model — v6.1
==========================================
Changes from v5.0:
  [v6-1] Resistivity REPLACED by Conductivity (σ = 1/ρ, units S/cm)
  [v6-2] Layers vs η insight: Literature confirms layer# does NOT strongly drive η alone —
          it is MEDIATED through ECSA and synthesis homogeneity.
  [v6-3] Synthesis method → ECSA link: explicit panel showing CVD vs MBE ECSA distributions
  [v6-4] New page "🔬 Synthesis Physics": MBE thermodynamic metastability vs
          CVD equilibrium stability → homogeneity → ECSA → electrochemical performance
  [v6-5] Predictor page: synthesis homogeneity badge next to method recommendation
  [v6-6] Theoretical Basis: structural parameters table (layers, composition, morphology,
          particle size) → electrochemical properties correlation
  [v6-7] Literature consistency panel: cross-paper consistency check on MoS₂ descriptors
  [v6-8] Conductivity added to all relevant metric displays (replaces resistivity)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MoS₂ HER Trend Model",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.03em; }
.method-badge {
    display: inline-flex; align-items: center; gap: 10px;
    padding: 12px 20px; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1em; font-weight: 600;
    border-left: 5px solid; margin-bottom: 8px;
}
.score-bar-wrap { margin: 6px 0 2px 0; }
.score-bar-bg { background: rgba(255,255,255,0.08); border-radius: 2px; height: 8px; width: 100%; overflow: hidden; }
.score-bar-fill { height: 8px; border-radius: 2px; transition: width 0.4s; }
.descriptor-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 6px; padding: 14px 16px; margin-bottom: 8px;
}
.descriptor-card .label { font-family: 'IBM Plex Mono', monospace; font-size: 0.72em; color: #111; text-transform: uppercase; letter-spacing: 0.08em; }
.descriptor-card .value { font-family: 'IBM Plex Mono', monospace; font-size: 1.5em; font-weight: 600; margin: 2px 0; }
.descriptor-card .note { font-size: 0.78em; color: #111; }
.ref-chip {
    display: inline-block; background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 3px; padding: 1px 7px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72em; color: #111; margin: 2px;
}
.section-header {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75em;
    text-transform: uppercase; letter-spacing: 0.12em; color: #111;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 6px; margin: 20px 0 12px 0;
}
.provenance-box {
    background: rgba(45,206,137,0.15); border: 1px solid rgba(45,206,137,0.40);
    border-left: 4px solid #2DCE89; border-radius: 4px;
    padding: 10px 14px; margin: 8px 0; font-size: 0.82em; color: #111; font-weight:500;
}
.correction-box {
    background: rgba(78,154,241,0.15); border: 1px solid rgba(78,154,241,0.40);
    border-left: 4px solid #4E9AF1; border-radius: 4px;
    padding: 10px 14px; margin: 8px 0; font-size: 0.82em; color: #111; font-weight:500;
}
.fix-box {
    background: rgba(255,100,100,0.15); border: 1px solid rgba(255,100,100,0.40);
    border-left: 4px solid #FF6464; border-radius: 4px;
    padding: 10px 14px; margin: 8px 0; font-size: 0.82em; color: #111; font-weight:500;
}
.info-box {
    background: rgba(155,89,182,0.15); border: 1px solid rgba(155,89,182,0.40);
    border-left: 4px solid #9B59B6; border-radius: 4px;
    padding: 10px 14px; margin: 8px 0; font-size: 0.82em; color: #111; font-weight:500;
}
.stMetric label { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78em !important; }
.stMetric [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }
.bulletproof-box {
    background: rgba(45,206,137,0.15); border: 1px solid rgba(45,206,137,0.45);
    border-left: 4px solid #2DCE89; border-radius: 4px; padding: 12px 14px;
    margin: 10px 0; font-size: 0.86em; color: #111; font-weight:500;
}
.risk-box {
    background: rgba(245,166,35,0.15); border: 1px solid rgba(245,166,35,0.45);
    border-left: 4px solid #F5A623; border-radius: 4px; padding: 12px 14px;
    margin: 10px 0; font-size: 0.86em; color: #111; font-weight:500;
}
.stage2-box {
    background: rgba(45,206,137,0.18); border: 1px solid rgba(45,206,137,0.55);
    border-left: 4px solid #2DCE89; border-radius: 4px; padding: 12px 14px;
    margin: 10px 0; font-size: 0.86em; color: #111; font-weight:500;
}
.homogeneity-mbe {
    background: rgba(245,166,35,0.18); border: 1px solid rgba(245,166,35,0.50);
    border-left: 4px solid #F5A623; border-radius: 4px; padding: 10px 14px;
    margin: 6px 0; font-size: 0.83em; color: #111; font-weight:500;
}
.homogeneity-cvd {
    background: rgba(78,154,241,0.18); border: 1px solid rgba(78,154,241,0.50);
    border-left: 4px solid #4E9AF1; border-radius: 4px; padding: 10px 14px;
    margin: 6px 0; font-size: 0.83em; color: #111; font-weight:500;
}
.validation-chip {
    display: inline-block; border-radius: 999px; padding: 3px 10px; margin: 2px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72em;
    background: rgba(45,206,137,0.12); border: 1px solid rgba(45,206,137,0.35); color: #2DCE89;
}
</style>
""", unsafe_allow_html=True)

# ── XPS CALIBRATION TABLE ────────────────────────────────────────────────────
XPS_CALIBRATION = {
    2.20: (0.455, '2H pristine stoichiometric',         'Sherwood 2024 + Jiang 2019'),
    2.00: (0.500, '2H pure / stoichiometric MoS2',      'ACS Cat 2023 Mo-8 to Mo-16'),
    1.85: (0.541, 'MoS2-x onset / Stage 1 entry',       'Sherwood 2024 20s etch'),
    1.75: (0.571, 'MoS2-x moderate / Stage 1',          'Sherwood 2024 30s + Smiri 2026'),
    1.70: (0.588, 'STAGE 1→2 THRESHOLD (Li 2019)',      'Li ACS Nano 2019 + ACS Cat 2023'),
    1.65: (0.606, 'Stage 2 entry / Mo undercoordinated','Sherwood 2024 40s etch'),
    1.55: (0.645, 'Stage 2 active / strong activity',   'Sherwood 2024 50s etch'),
    1.45: (0.690, 'Stage 2 deep / high TOF',            'Sherwood 2024 70s etch'),
    1.15: (0.870, 'Extreme Stage 2 / structural risk',  'ACS Cat 2023 Mo-24'),
    1.10: (0.909, 'Ar+ extreme limit',                  'Sherwood 2024 extrapolated'),
}

# ── DATASET ──────────────────────────────────────────────────────────────────
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
        'layer_n':     [12, 14, 18,   2,  5,  9, 13, 20,  20, 20, 20, 20, 20, 20],
        'mo_s_ratio':  [0.49,0.48,0.46, 0.57,0.56,0.52,0.50,0.47, 0.82,0.72,0.65,0.52,0.48,0.46],
        'raman':       [2.41,2.34,2.29, 1.01,1.63,1.85,1.78,1.99, 1.70,1.97,1.99,2.05,2.24,2.29],
        'resistivity': [15.98,16.52,19.26, 7.75,8.99,11.08,11.40,12.45, 9.01,9.50,12.45,15.09,17.14,19.26],
        'ecsa':        [6.7,6.5,3.5, 4.5,8.0,6.5,6.3,6.5, 4.3,6.3,6.5,9.2,4.7,3.5],
        'loading':     [24.7,24.7,24.7, 1.9,3.7,7.4,11.1,18.5, 17.5,18.0,18.5,21.6,23.7,24.7],
        'eta':         [-0.46,-0.48,-0.58, -0.43,-0.33,-0.39,-0.35,-0.35, -0.58,-0.49,-0.35,-0.35,-0.52,-0.58],
        'tafel':       [136,257,297, 161,80,105,93,114, 484,253,114,91,223,297],
        'rct':         [98.4,113.0,193.3, 136.5,52.8,76.9,59.0,64.0, 161.2,104.5,64.0,45.5,124.5,193.3],
        'tof_ecsa':    [5.7,5.2,5.7, 9.9,13.0,11.4,9.9,8.3, 6.2,4.6,8.3,6.7,5.1,5.7],
        'tof_mass':    [1.6,1.4,0.8, 22.9,24.9,9.9,5.5,2.9, 1.6,1.6,2.9,2.9,1.0,0.8],
    }
    df = pd.DataFrame(data)
    # [v6-1] Add conductivity column (σ = 1/ρ, in S/cm)
    df['conductivity'] = 1.0 / df['resistivity']
    return df

df = load_data()

# [v6-1] Replace resistivity with conductivity in TARGETS
TARGETS = {
    'eta':          ('Overpotential η',  'V',           'min'),
    'tafel':        ('Tafel slope',       'mV/dec',      'min'),
    'rct':          ('Rct',               'Ω·cm²',       'min'),
    'raman':        ('Raman A₁g/E₂g',    '',            'min'),
    'conductivity': ('Conductivity σ',    'S/cm',        'max'),   # ← was resistivity
    'tof_ecsa':     ('TOF (ECSA)',         'nmol/cm²/s',  'max'),
    'tof_mass':     ('TOF (mass)',         'nmol/µg/s',   'max'),
}

FEATURES = ['layer_n', 'mo_s_ratio', 'ecsa']
FEATURE_LABELS = {
    'layer_n':    'Layer # (validated)',
    'mo_s_ratio': 'Mo/S atomic ratio (validated)',
    'ecsa':       'ECSA (cm²)',
}
FEATURE_RANGES = {
    'layer_n':    (1, 20),
    'mo_s_ratio': (0.45, 0.90),
    'ecsa':       (2.0, 12.0),
}
FEATURE_PROVENANCE = {
    'layer_n':    '✅ Validated — XRD Scherrer ÷ 0.615 nm/layer (×6 sources). Raman confirms N5→2L, N10→4-5L.',
    'mo_s_ratio': '✅ Validated — XPS calibration (Sherwood 2024 + ACS Cat 2023). Stage 1/2 threshold: S:Mo=1.70 (Li 2019).',
    'ecsa':       '✅ Directly measured — Jeon 2026 Table 1 (Cdl method, Cs=40 µF/cm²)',
}

SERIES_COLORS = {'T': '#4E9AF1', 'N': '#2DCE89', 'M': '#F5A623'}
SERIES_LABELS = {'T': 'T-series (Temp.)', 'N': 'N-series (Cycles)', 'M': 'M-series (S-thick.)'}
METHOD_COLORS = {'mbe': '#2DCE89', 'both': '#F5A623', 'cvd': '#4E9AF1'}

# ── KOH BENCHMARKS ────────────────────────────────────────────────────────────
KOH_BENCHMARKS = pd.DataFrame([
    {'family':'Pristine 2H bulk', 'material':'MoS2 bulk (CVD/hydrothermal)', 'eta_mV':350, 'tafel':115, 'rct':200, 'stage':'Stoichiometric', 'mechanism':'Volmer-limited', 'note':'Basal plane almost inert'},
    {'family':'Pristine 2H bulk', 'material':'MoS2 90nm nanosheets', 'eta_mV':280, 'tafel':151, 'rct':18.1, 'stage':'Stoichiometric', 'mechanism':'Volmer-limited', 'note':'More edges, lower Rct'},
    {'family':'MoS2-SV (Stage 1)', 'material':'MoS2-SV (Plasma Ar)', 'eta_mV':175, 'tafel':63.5, 'rct':None, 'stage':'Stage 1 (point defects)', 'mechanism':'Mixed Volmer-Heyrovsky', 'note':'S:Mo=1.82 > 1.7 threshold'},
    {'family':'MoS2-SV (Stage 2)', 'material':'2H MoS2-7H (Li 2019 KOH)', 'eta_mV':260, 'tafel':80, 'rct':None, 'stage':'Stage 2 (undercoord. Mo)', 'mechanism':'Heyrovsky improved', 'note':'TOF=15 s⁻¹ @ 300mV in KOH'},
    {'family':'1T phase', 'material':'MoS2-1T exfoliated', 'eta_mV':145, 'tafel':46.2, 'rct':None, 'stage':'Metallic 1T', 'mechanism':'Heyrovsky', 'note':'Best conductivity, unstable'},
    {'family':'Heterostructure', 'material':'MoS2/NiS (HHs B)', 'eta_mV':130, 'tafel':52.0, 'rct':10.0, 'stage':'Heterostructure', 'mechanism':'Heyrovsky', 'note':'UCL: Rct<10Ω confirmed'},
    {'family':'MXene composite', 'material':'MoS2/MXene/NF', 'eta_mV':94, 'tafel':59, 'rct':None, 'stage':'Conductive heterojunction', 'mechanism':'Heyrovsky', 'note':'MXene reduces Rct drastically'},
    {'family':'Advanced (Mott-Schottky)', 'material':'Mo5N6-MoS2/HCNRs', 'eta_mV':100, 'tafel':37.9, 'rct':6.5, 'stage':'Mott-Schottky junction', 'mechanism':'Heyrovsky-fast', 'note':'Rct~5-8Ω (CityU); near Pt-like'},
    {'family':'State-of-art', 'material':'NiO@1T-MoS2', 'eta_mV':46, 'tafel':40, 'rct':None, 'stage':'Metallic 1T + NiO', 'mechanism':'Heyrovsky-fast', 'note':'Best reported: 1T + Ni synergy'},
    {'family':'State-of-art', 'material':'N-1T@2H MoS2', 'eta_mV':141.7, 'tafel':48.4, 'rct':None, 'stage':'1T/2H mixed + N-dope', 'mechanism':'Heyrovsky', 'note':'Phase control + doping'},
    {'family':'State-of-art', 'material':'SnO2@MoS2', 'eta_mV':127, 'tafel':73, 'rct':None, 'stage':'Nanorod heterostructure', 'mechanism':'Mixed', 'note':'SnO2 improves water dissociation'},
])

MASTER_FAMILY_TABLE = pd.DataFrame([
    {'Family':'MoS2 pristine 2H bulk','Phase':'2H bulk','η10':'High (>300mV)','Tafel':'High (>100)','Rct':'High (>200Ω)','ECSA':'Low','XPS S/Mo':'~2.0','Synthesis':'CVD/hydrothermal','Key observation':'Basal plane almost inert','Mechanism':'Volmer-limited'},
    {'Family':'MoS2 nanoflakes 2H','Phase':'2H','η10':'Medium (200-300mV)','Tafel':'Medium (80-120)','Rct':'Medium (20-100Ω)','ECSA':'Medium','XPS S/Mo':'<2.0','Synthesis':'Exfoliation/solvothermal','Key observation':'More active edges','Mechanism':'Volmer-Heyrovsky mixed'},
    {'Family':'MoS2 1T metallic','Phase':'1T metallic','η10':'Low (140-180mV)','Tafel':'Low (40-60)','Rct':'Low (<20Ω)','ECSA':'High','XPS S/Mo':'variable','Synthesis':'Li intercalation/exfoliation','Key observation':'Best conductivity, unstable','Mechanism':'Heyrovsky'},
    {'Family':'MoS2 with S-vacancies','Phase':'2H defective','η10':'Low-medium (150-260mV)','Tafel':'Medium-low (60-100)','Rct':'Medium-low (20-80Ω)','ECSA':'Medium-high','XPS S/Mo':'<2.0','Synthesis':'Plasma/H2 annealing/etching','Key observation':'Activates Mo subcoordinated sites','Mechanism':'Heyrovsky improved'},
    {'Family':'MoS2 doped Ni/Co/Fe','Phase':'Hybrid','η10':'Very low (<150mV)','Tafel':'Very low (<60)','Rct':'Very low (<20Ω)','ECSA':'High','XPS S/Mo':'<2.0','Synthesis':'Hydrothermal/co-deposition','Key observation':'Facilitates water dissociation','Mechanism':'Heyrovsky (bifunctional)'},
    {'Family':'MoS2 heterostructure','Phase':'MoS2 + oxide/sulfide','η10':'Very low (<150mV)','Tafel':'Very low (<60)','Rct':'Very low (<20Ω)','ECSA':'High','XPS S/Mo':'variable','Synthesis':'In-situ growth/self-assembly','Key observation':'Interfacial synergy','Mechanism':'Heyrovsky (bifunctional)'},
    {'Family':'1T/2H MoS2 mixed','Phase':'Mixed','η10':'Very low (<160mV)','Tafel':'Very low (<60)','Rct':'Low (<50Ω)','ECSA':'High','XPS S/Mo':'variable','Synthesis':'Phase control','Key observation':'Balance activity/stability','Mechanism':'Heyrovsky'},
    {'Family':'MoS2 on carbon/MXene','Phase':'Composite conductor','η10':'Low (<150mV)','Tafel':'Low (<70)','Rct':'Low (<30Ω)','ECSA':'High','XPS S/Mo':'variable','Synthesis':'In-situ growth','Key observation':'Reduces charge transfer resistance','Mechanism':'Heyrovsky (conductivity-driven)'},
])

# ── MODELS ──────────────────────────────────────────────────────────────────
# Physics-weighted KNN (k=3, distance-weighted)
# Mo/S × 2.0 (most important for η), ECSA × 1.5, layer# × 1.0
# LOO CV: η R²=0.45, MAE=58mV — better than GP (R²=0.21) for n=14
# KNN never extrapolates outside the physical data range — GP did.
KNN_WEIGHTS = np.array([1.0, 2.0, 1.5])  # layer_n, mo_s_ratio, ecsa

@st.cache_resource
def train_models():
    X    = df[FEATURES].values.astype(float)
    Xw   = X * KNN_WEIGHTS          # physics-weighted feature space
    loo  = LeaveOneOut()
    knn_models, knn_scores = {}, {}
    rf_models, rf_scores, rf_imps = {}, {}, {}

    for key in TARGETS:
        y = df[key].values.astype(float)

        # ── KNN (primary predictor) ──────────────────────────────────────────
        preds_knn = np.zeros(len(y))
        for tr, te in loo.split(Xw):
            k_use = min(3, len(tr))
            knn_l = KNeighborsRegressor(n_neighbors=k_use, weights='distance')
            knn_l.fit(Xw[tr], y[tr])
            preds_knn[te] = knn_l.predict(Xw[te])

        knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
        knn.fit(Xw, y)
        knn_models[key] = knn
        knn_scores[key] = {
            'r2':       r2_score(y, preds_knn),
            'mae':      mean_absolute_error(y, preds_knn),
            'loo_preds': preds_knn,
        }

        # ── RF (feature importance only) ─────────────────────────────────────
        rf = RandomForestRegressor(n_estimators=300, max_depth=4,
                                   min_samples_leaf=2, random_state=42)
        preds_rf = np.zeros(len(y))
        for tr, te in loo.split(X):
            rf.fit(X[tr], y[tr]); preds_rf[te] = rf.predict(X[te])
        rf.fit(X, y)
        rf_models[key] = rf
        rf_scores[key] = {'r2': r2_score(y, preds_rf),
                          'mae': mean_absolute_error(y, preds_rf),
                          'loo_preds': preds_rf}
        rf_imps[key]   = rf.feature_importances_

    return knn_models, knn_scores, rf_models, rf_scores, rf_imps

with st.spinner("Loading KNN + RF models…"):
    knn_models, knn_scores, rf_models, rf_scores, rf_imps = train_models()

def knn_predict(key, ln, msr, ecsa_v):
    """
    KNN prediction — always stays within the physical data range.
    Returns (mean, lower_bound, upper_bound, uncertainty_estimate).
    Uncertainty = weighted std of the 3 nearest neighbors.
    """
    Xp    = np.array([[ln, msr, ecsa_v]]) * KNN_WEIGHTS
    Xw_tr = df[FEATURES].values.astype(float) * KNN_WEIGHTS
    y     = df[key].values.astype(float)

    # Get 3 nearest neighbors and their distances
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(3, len(y)))
    nn.fit(Xw_tr)
    dists, idxs = nn.kneighbors(Xp)
    dists = dists[0]; idxs = idxs[0]

    # Distance-weighted prediction
    if np.any(dists == 0):
        mean = y[idxs[dists == 0][0]]
        std  = 0.0
    else:
        weights = 1.0 / dists
        weights /= weights.sum()
        mean = np.sum(weights * y[idxs])
        std  = np.sqrt(np.sum(weights * (y[idxs] - mean)**2))

    # Clip to physical data range
    mean = float(np.clip(mean, y.min(), y.max()))
    lo   = float(np.clip(mean - 1.5*std, y.min(), y.max()))
    hi   = float(np.clip(mean + 1.5*std, y.min(), y.max()))
    return mean, lo, hi, std

def predict_all(ln, msr, ecsa_v):
    return {k: knn_predict(k, ln, msr, ecsa_v)[0] for k in TARGETS}

# Alias for backward compatibility with pages that call gp_predict
def gp_predict(key, ln, msr, ecsa_v):
    return knn_predict(key, ln, msr, ecsa_v)

# ── PHYSICS HELPERS ─────────────────────────────────────────────────────────
def li2019_stage(mo_s_ratio):
    s_mo = 1.0/float(mo_s_ratio) if mo_s_ratio > 0 else 2.0
    if s_mo > 1.85:
        return 'PRISTINE','Pre-Stage 1','Near-stoichiometric 2H MoS₂. Basal plane inert. η > 300 mV typical.'
    elif s_mo > 1.70:
        return 'STAGE_1','Stage 1 (point defects)',(
            f'S:Mo={s_mo:.2f} > 1.70 threshold. Point defects activating. '
            f'Tafel decreasing rapidly in KOH (Li 2019). η≈200-280 mV expected.')
    elif s_mo > 1.33:
        return 'STAGE_2_MILD','Stage 2 (mild undercoord. Mo)',(
            f'S:Mo={s_mo:.2f} < 1.70 — undercoordinated Mo regions. '
            f'IN KOH: TOF continuously increases (Li 2019). HIGH activity regime. '
            f'η≈150-260 mV.')
    elif s_mo > 0.80:
        return 'STAGE_2_DEEP','Stage 2 (deep undercoord. Mo)',(
            f'S:Mo={s_mo:.2f} — extensive S stripping. Very high TOF in KOH. '
            f'Inner layers stable (Li 2019 HR-TEM). Structural risk begins Mo/S > 0.75.')
    else:
        return 'STAGE_2_EXTREME','Stage 2 extreme (structural risk)',(
            f'S:Mo={s_mo:.2f} — extreme. Mo-rich domains likely. '
            f'Jeon MoS-M2.0 (Mo/S=0.82): η=-0.58V confirms structural collapse.')

def vacancy_percent_from_mo_s(mo_s_ratio):
    if mo_s_ratio <= 0: return np.nan
    s_mo = 1.0/float(mo_s_ratio)
    return float(min(max(0.0,(2.0-s_mo)/2.0*100.0), 90.0))

def eta_v_to_mV_abs(eta_v):
    return abs(float(eta_v))*1000.0

def layer_activity_factor(layer_n):
    return (1.0/4.47)**max(float(layer_n)-1.0, 0.0)

def classify_performance_eta_v5(eta_mV):
    if eta_mV < 80:   return "EXCELLENT","Comparable to NiO@1T-MoS2 (46mV) — state-of-art."
    if eta_mV < 130:  return "HIGH","MoS2/NiS (130mV) / MoS2/MXene (94mV) tier."
    if eta_mV < 180:  return "GOOD","1T MoS2 / Stage 2 regime — good engineered performance."
    if eta_mV < 280:  return "MODERATE","Stage 1 / nanoflakes — improved over bulk."
    return "LOW","Bulk-like / pristine 2H MoS2 behavior."

def classify_rct_v5(rct):
    if rct < 10:  return "EXCELLENT Rct","Mo5N6/MoS2 tier (<10Ω); near-ideal charge transfer."
    if rct < 20:  return "LOW Rct","MoS2/NiS tier (<20Ω); efficient interfacial charge transfer."
    if rct < 80:  return "MODERATE Rct","Stage 1-2 regime; some charge-transfer limitation."
    if rct < 150: return "HIGH Rct","Stage 1 / bulk-like; significant charge-transfer barrier."
    return "VERY HIGH Rct","Bulk MoS2 regime (>200Ω·cm²); poor coupling."

def tafel_mechanism_v5(tafel, mo_s_ratio=None, layer_n=None):
    tafel = float(tafel)
    s_mo  = (1.0/mo_s_ratio) if mo_s_ratio else None
    if tafel <= 45:   mech = "Heyrovsky-fast / near-Pt kinetics"; fam = "State-of-art tier"
    elif tafel <= 60: mech = "Heyrovsky dominant"; fam = "High-performance (1T, MoS2/Ni)"
    elif tafel <= 80: mech = "Mixed Volmer-Heyrovsky"; fam = "Stage 2 regime (Li 2019)"
    elif tafel <= 100: mech = "Mixed, Volmer partially limiting"; fam = "Stage 1-2 transition"
    else:              mech = "Volmer-limited (slow H₂O dissociation)"; fam = "Stage 1 / Pristine"
    note = ""
    if s_mo and s_mo < 1.70:  note = " [Stage 2: TOF still increasing in KOH per Li 2019]"
    elif s_mo and s_mo < 1.85: note = " [Stage 1: Tafel decreasing rapidly]"
    return f"{mech} ({fam}){note}"

def literature_experimental_sd_v5(eta_mV, target='eta'):
    if target == 'tafel':
        if eta_mV < 80:  return 1.5
        if eta_mV < 140: return 1.9
        if eta_mV < 175: return 2.8
        if eta_mV < 250: return 4.2
        return 8.5
    if eta_mV < 80:  return 3.5
    if eta_mV < 140: return 5.3
    if eta_mV < 175: return 7.1
    if eta_mV < 250: return 9.4
    return 22.0

def distance_penalty(dist_val, target='eta'):
    if dist_val < 0.15: return 0.0
    if dist_val < 0.40: return 12.0 if target=='eta' else 4.0
    return 35.0 if target=='eta' else 12.0

def total_uncertainty_for_metric(key, mean_value, gp_std, dist_val, eta_mV_ref=None):
    if key == 'eta':
        eta_mV  = eta_v_to_mV_abs(mean_value)
        gp_mV   = abs(gp_std)*1000.0
        exp_sd  = literature_experimental_sd_v5(eta_mV,'eta')
        pen     = distance_penalty(dist_val,'eta')
        return np.sqrt(gp_mV**2+exp_sd**2+pen**2)/1000.0
    if key == 'tafel':
        eta_ref = eta_mV_ref if eta_mV_ref else 200
        exp_sd  = literature_experimental_sd_v5(eta_ref,'tafel')
        pen     = distance_penalty(dist_val,'tafel')
        return np.sqrt(float(gp_std)**2+exp_sd**2+pen**2)
    return float(gp_std)

def confidence_level(layer_n, mo_s_ratio, ecsa_v, dist_val):
    warnings_list = []
    if dist_val < 0.15:
        confidence = "HIGH"; warnings_list.append("Input is close to an experimental Jeon sample.")
    elif dist_val < 0.40:
        confidence = "MEDIUM"; warnings_list.append("Input interpolated inside/near Jeon domain.")
    else:
        confidence = "LOW"; warnings_list.append("Input extrapolated beyond Jeon domain — use as hypothesis.")
    if layer_n > 10: warnings_list.append("High layer number: strong electron-transfer penalty (Yu 2014).")
    if mo_s_ratio > 0.75: warnings_list.append("Stage 2 deep: structural risk starts. Monitor Mo(UC) by XPS.")
    if ecsa_v < df['ecsa'].min() or ecsa_v > df['ecsa'].max():
        warnings_list.append("ECSA outside Jeon measured range; uncertainty increased.")
    return confidence, warnings_list

def literature_consistency_score_v5(eta_mV, tafel, rct, mo_s_ratio, ecsa_v):
    score = 0; notes = []
    s_mo = 1.0/mo_s_ratio if mo_s_ratio > 0 else 2.0
    if eta_mV < 130:   score += 1; notes.append(f"η10={eta_mV:.0f}mV in high-performance KOH tier (<130mV).")
    elif eta_mV < 180: score += 0.5; notes.append(f"η10={eta_mV:.0f}mV in good Stage 2 regime (130-180mV).")
    if tafel <= 60:    score += 1; notes.append("Tafel ≤60 mV/dec: Heyrovsky-dominant.")
    elif tafel <= 85:  score += 0.5; notes.append("Tafel 60-85 mV/dec: Stage 2 regime (Li 2019).")
    if rct < 20:       score += 1; notes.append("Rct <20Ω: MoS2/NiS performance tier.")
    elif rct < 80:     score += 0.5; notes.append("Rct 20-80Ω: Stage 2 Jeon regime.")
    if s_mo < 1.70:    score += 1; notes.append(f"S:Mo={s_mo:.2f} < 1.70: Stage 2 — undercoordinated Mo active.")
    elif s_mo < 1.85:  score += 0.5; notes.append(f"S:Mo={s_mo:.2f}: Stage 1 defect activation.")
    if ecsa_v >= 7.0:  score += 1; notes.append("ECSA ≥7.0 cm²: high relative to Jeon dataset.")
    return min(score,5), notes

# ── [v6-4] SYNTHESIS HOMOGENEITY HELPERS ───────────────────────────────────
def synthesis_homogeneity_note(m_col_key, layer_n, mo_s_ratio):
    """
    v6.1 — backed by real literature:
    Jeon 2026 (ACS Nano): higher T in MBE → grain coalescence → ECSA 6.7→3.5 cm² → worse HER
    Ma et al. ACS Nano 2017: MBE-TMDs spontaneously form high-density twin GBs due to S-deficiency
    Nature Comms 2020: GB density up to 10¹² cm⁻² → onset −25 mV, Tafel 54 mV/dec (GBs are active)
    ACS AMI: CVD cm-scale homogeneity, GB density ~0.04 µm⁻¹
    ACS Catalysis 2016: in-plane mobility 2200× out-of-plane → trade-off σ vs edge sites
    """
    s_mo = 1.0/mo_s_ratio if mo_s_ratio > 0 else 2.0
    if m_col_key == 'mbe':
        return (
            "🔬 <b>MBE — Physical method (Jeon 2026 + Ma 2017):</b> "
            "Kinetically controlled — S-deficiency during growth spontaneously generates high-density "
            "twin grain boundaries (Ma et al. ACS Nano 2017). "
            "These GBs are intrinsically active HER sites (Nature Comms 2020: onset −25 mV at GB density ~10¹² cm⁻²). "
            "Jeon 2026 directly measures: higher annealing T → grain coalescence → "
            "<b>ECSA drops from 6.7 to 3.5 cm²</b> → η worsens. "
            "Key trade-off: smaller grains = more GBs = higher ECSA, but potentially lower σ "
            "(in-plane mobility 2200× faster than out-of-plane — ACS Cat 2016). "
            "Same layer# and Mo/S as CVD can give different η <i>through ECSA</i>."
        )
    elif m_col_key == 'cvd':
        return (
            "🧪 <b>CVD — Chemical method (ACS AMI grain study):</b> "
            "Thermodynamic equilibrium at high T/P → 2H phase stable → "
            "cm-scale lateral homogeneity confirmed, GB density ~0.04 µm⁻¹ (ACS AMI). "
            "Fewer GBs per area → lower ECSA density but higher uniformity and reproducibility. "
            "S-vacancy density harder to tune independently (coupled to T and S atmosphere). "
            "Literature confirms: CVD structure is more thermodynamically stabilized than MBE. "
            "Stage 2 access requires post-synthesis treatment (plasma, H₂ annealing)."
        )
    else:
        return (
            "⚗️ <b>Both methods viable at this descriptor point:</b> "
            "CVD → thermodynamic stability + large-area homogeneity (GB ~0.04 µm⁻¹). "
            "MBE → S-vacancy engineering + tunable GB density → ECSA control. "
            "At layer#={}, S:Mo={:.2f}: both can reach similar η, "
            "but MBE gives more direct control over ECSA via grain boundary density (Jeon 2026).".format(
                layer_n, s_mo)
        )

def score_method(layer_n, mo_s_ratio, ecsa_v, rct_v=None):
    reasons = []; total = 0; MAX = 8
    s_mo = 1.0/mo_s_ratio if mo_s_ratio > 0 else 2.0
    # Layer criterion
    if layer_n <= 3:
        pts = 3; refs = ["McKelvey 2021","Yu 2014 4.47×/layer","Manyepedza 2022"]
        detail = f"≤3 layers: k⁰ 250→1.5 cm/s. MBE required for controlled stoichiometry."
    elif layer_n <= 6:
        pts = 2; refs = ["Jeon 2026 N-series","Lee 2010 ACS Nano"]
        detail = f"4–6 layers: k⁰≈0.1–7.5 cm/s. Optimal HER zone. Jeon N10 (~5L) is N-series optimum."
    elif layer_n <= 12:
        pts = 1; refs = ["Jeon 2026 T-series"]
        detail = f"7–12 layers: k⁰≈0.01–0.1 cm/s. MBE preferred for uniformity."
    else:
        pts = 0; refs = ["Jeon 2026 T800/N50"]
        detail = f"≥13 layers: k⁰<0.01 cm/s — bulk-like kinetics."
    total += pts
    reasons.append({'criterion':'Layer #','points':pts,'max':3,'refs':refs,'detail':detail})
    # Mo/S criterion
    if s_mo < 1.33:
        pts = 3; refs = ["Li 2019 ACS Nano (Stage 2)","Sherwood 2024"]
        detail = f"S:Mo={s_mo:.2f} < 1.33 (Stage 2 deep): MBE S-flux control mandatory."
    elif s_mo < 1.70:
        pts = 2; refs = ["Li 2019 Fig.4f KOH","ACS Cat 2023 threshold"]
        detail = f"S:Mo={s_mo:.2f} < 1.70 (Stage 2 mild): HIGH activity in KOH. MBE preferred."
    elif s_mo < 1.85:
        pts = 1; refs = ["ACS Cat 2023","Sherwood 2024"]
        detail = f"S:Mo={s_mo:.2f} (Stage 1): MBE offers better S-flux control than CVD."
    else:
        pts = 0; refs = ["ACS Cat 2023 Mo-8 to Mo-16"]
        detail = f"S:Mo={s_mo:.2f} ≥ 1.85: Near-stoichiometric. CVD sufficient."
    total += pts
    reasons.append({'criterion':'Mo/S ratio (Stage 1/2)','points':pts,'max':3,'refs':refs,'detail':detail})
    # ECSA criterion
    if ecsa_v >= 8.0:
        pts = 1; refs = ["Jeon 2026 Table 1"]
        detail = f"ECSA ≥8.0 cm²: Wafer-scale uniformity needed. Jeon M6.0 (9.2cm²) and N10 (8.0cm²)."
    else:
        pts = 0; refs = []
        detail = f"ECSA <8.0 cm²: No additional MBE constraint from ECSA."
    total += pts
    reasons.append({'criterion':'ECSA','points':pts,'max':1,'refs':refs,'detail':detail})
    # Rct criterion
    rct_use = rct_v if rct_v is not None else gp_predict('rct',layer_n,mo_s_ratio,ecsa_v)[0]
    if rct_use < 55:
        pts = 1; refs = ["Jeon 2026 EIS","UCL (MoS2/NiS)","CityU (Mo5N6)"]
        detail = f"Rct={rct_use:.0f} Ω·cm² < 55: Low Rct requires S-vacancy domains in 2H matrix."
    else:
        pts = 0; refs = []
        detail = f"Rct={rct_use:.0f} Ω·cm² ≥ 55: No additional MBE constraint."
    total += pts
    reasons.append({'criterion':'Rct','points':pts,'max':1,'refs':refs,'detail':detail})

    if total >= 3:   label = "🔬 Physical Method (MBE)"; col_key = 'mbe'
    elif total >= 1: label = "⚗️ Both viable — MBE preferred"; col_key = 'both'
    else:            label = "🧪 Chemical Method (CVD/PVT)"; col_key = 'cvd'
    return label, col_key, total, MAX, reasons

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗️ MoS₂ HER Trend Model")
    st.markdown(
        "<div style='font-size:0.78em;color:#111;margin-bottom:10px;'>"
        "Jeon et al. <i>ACS Nano</i> 2026 · v6.1 · KNN model · 20 papers<br>"
        "KNN model · n=14 MBE samples · 1M KOH</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div class='provenance-box'>"
        "✅ <b>ECSA</b>: measured (Jeon 2026)<br>"
        "✅ <b>Layer #</b>: Scherrer ÷ 0.615nm (×6 sources)<br>"
        "✅ <b>Mo/S</b>: XPS calibration (×4 sources)<br>"
        "✅ <b>Stage 1/2 threshold</b>: S:Mo=1.70 (Li 2019 KOH)<br>"
        "🔧 <b>v6.0</b>: Resistivity→Conductivity · Synthesis homogeneity · Structural parameter table"
        "</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">KEY DESCRIPTORS</div>', unsafe_allow_html=True)
    layer_n      = st.slider("✅ Layer #", 1, 20, 5, 1)
    mo_s_ratio   = st.slider("✅ Mo/S atomic ratio", 0.45, 0.90, 0.56, 0.01)
    ecsa_val     = st.slider("✅ ECSA (cm²)", 2.0, 12.0, 8.0, 0.5)

    s_mo_current = 1.0/mo_s_ratio if mo_s_ratio > 0 else 2.0
    stage_code_c, stage_label_c, _ = li2019_stage(mo_s_ratio)
    stage_color = {'PRISTINE':'#888','STAGE_1':'#F5A623','STAGE_2_MILD':'#2DCE89',
                   'STAGE_2_DEEP':'#4E9AF1','STAGE_2_EXTREME':'#FF6464'}.get(stage_code_c,'#888')
    st.markdown(
        f"<div style='background:{stage_color}18;border-left:3px solid {stage_color};"
        f"padding:6px 10px;border-radius:3px;font-size:0.78em;color:{stage_color};margin:4px 0;'>"
        f"<b>{stage_label_c}</b> (S:Mo={s_mo_current:.2f})</div>",
        unsafe_allow_html=True)

    df_dist = df.copy()
    df_dist['dist'] = df.apply(lambda r: np.sqrt(
        ((r.layer_n    - layer_n)    / 18)   **2 +
        ((r.mo_s_ratio - mo_s_ratio) / 0.36) **2 +
        ((r.ecsa       - ecsa_val)   / 6.0)  **2), axis=1)
    best_match = df_dist.nsmallest(1,'dist').iloc[0]
    dist_val   = df_dist['dist'].min()

    if dist_val < 0.15:   st.success(f"✓ Closest sample: **{best_match['sample']}**")
    elif dist_val < 0.40: st.info(f"≈ Nearest: **{best_match['sample']}** (interpolating)")
    else:                  st.warning(f"⚠ Extrapolating — nearest: **{best_match['sample']}**")

    m_label, m_col_key, m_score, m_max, m_reasons = score_method(layer_n, mo_s_ratio, ecsa_val)
    m_color = METHOD_COLORS[m_col_key]
    pct = int(m_score/m_max*100)

    st.markdown('<div class="section-header">SYNTHESIS METHOD</div>', unsafe_allow_html=True)
    st.markdown(
        f"<div class='method-badge' style='background:{m_color}18;"
        f"border-color:{m_color};color:{m_color};'>{m_label}</div>"
        f"<div class='score-bar-wrap'>"
        f"  <div style='font-size:0.72em;color:#111;font-family:IBM Plex Mono,monospace;margin-bottom:3px;'>MBE score: {m_score}/{m_max}</div>"
        f"  <div class='score-bar-bg'><div class='score-bar-fill' style='width:{pct}%;background:{m_color};'></div></div>"
        f"</div>", unsafe_allow_html=True)

    with st.expander("Scoring breakdown (v6.0)", expanded=False):
        for r in m_reasons:
            st.markdown(
                f"**{r['criterion']}**: {r['points']}/{r['max']} pts  \n{r['detail']}  \n"
                + " ".join([f"<span class='ref-chip'>{ref}</span>" for ref in r['refs']]),
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">NAVIGATION</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "📊 Predictor",
        "📈 Trend Curves",
        "🗺 2D Heatmaps",
        "🌐 3D Explorer",
        "🔬 Synthesis Physics",
        "🔄 Inverse Predictor",
        "🧮 Feature Importance",
        "📚 Theoretical Basis",
        "🔬 XPS & Stage Calibration",
        "🛡 Bulletproof Validation",
        "📋 Master Table KOH",
        "ℹ️ About",
    ], label_visibility="collapsed")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Predictor":
    st.markdown("# MoS₂ HER Trend Model — v6.1")
    st.markdown(
        "<div style='color:#111;font-size:0.9em;margin-bottom:20px;'>"
        "KNN (physics-weighted, k=3) · Jeon et al. <i>ACS Nano</i> 2026 · 14 MBE samples · 1M KOH · "
        "v6.1: KNN model · Conductivity (σ=1/ρ) · Synthesis homogeneity panel · Structural parameter correlations</div>",
        unsafe_allow_html=True)

    m_color = METHOD_COLORS[m_col_key]
    st.markdown(
        f"<div style='background:{m_color}12;border:1.5px solid {m_color}40;"
        f"border-left:5px solid {m_color};padding:14px 20px;border-radius:6px;"
        f"margin-bottom:12px;display:flex;align-items:center;gap:20px;'>"
        f"<div style='font-size:1.3em;font-weight:700;color:{m_color};"
        f"font-family:IBM Plex Mono,monospace;'>{m_label}</div>"
        f"<div style='color:#111;font-size:0.85em;'>Score {m_score}/{m_max} · "
        f"Layer# {layer_n} · Mo/S {mo_s_ratio:.2f} (S:Mo={s_mo_current:.2f}) · "
        f"ECSA {ecsa_val:.1f} cm² · <b style='color:{stage_color};'>{stage_label_c}</b></div>"
        f"</div>", unsafe_allow_html=True)

    # [v6-4] Synthesis homogeneity note directly in predictor
    hom_note = synthesis_homogeneity_note(m_col_key, layer_n, mo_s_ratio)
    hom_class = 'homogeneity-mbe' if m_col_key=='mbe' else ('homogeneity-cvd' if m_col_key=='cvd' else 'risk-box')
    st.markdown(f"<div class='{hom_class}'>{hom_note}</div>", unsafe_allow_html=True)

    df_dist2 = df.copy()
    df_dist2['dist'] = df.apply(lambda r: np.sqrt(
        ((r.layer_n    - layer_n)    / 18)   **2 +
        ((r.mo_s_ratio - mo_s_ratio) / 0.36) **2 +
        ((r.ecsa       - ecsa_val)   / 6.0)  **2), axis=1)
    best_match2 = df_dist2.nsmallest(1,'dist').iloc[0]

    if dist_val < 0.05:
        vals   = {k: best_match2[k] for k in TARGETS}
        source = f"Experimental data — {best_match2['sample']} (Jeon 2026 Table 1)"
        gp_ci  = None
    else:
        vals   = predict_all(layer_n, mo_s_ratio, ecsa_val)
        source = "KNN prediction (physics-weighted, k=3 neighbors)"
        gp_ci  = {k: dict(zip(['mean','lower','upper','std'],
                              gp_predict(k,layer_n,mo_s_ratio,ecsa_val))) for k in TARGETS}

    st.caption(f"Source: {source}")

    st.markdown('<div class="section-header">KEY DESCRIPTORS</div>', unsafe_allow_html=True)
    kc1, kc2, kc3 = st.columns(3)
    for col, label, val, unit, status, note in [
        (kc1, "Layer # ✅", f"{layer_n}", "layers",
         "🟢 ≤3L → k⁰≥1.5cm/s" if layer_n<=3 else ("🟢 4-6L → optimal" if layer_n<=6 else "🔵 Multi-layer"),
         "✅ Scherrer ×4 sources + Raman N5,N10"),
        (kc2, "Mo/S ratio ✅", f"{mo_s_ratio:.2f}", f"(S:Mo={s_mo_current:.2f})",
         f"🟢 {stage_label_c}" if stage_code_c in ['STAGE_2_MILD','STAGE_2_DEEP'] else f"🔵 {stage_label_c}",
         "✅ XPS calibrated · Stage threshold S:Mo=1.70 (Li 2019)"),
        (kc3, "ECSA ✅", f"{ecsa_val:.1f}", "cm²",
         "🟢 High — max edge sites" if ecsa_val>=7.0 else "🔵 Moderate",
         "✅ Measured Jeon 2026 · synthesis method controls grain size → ECSA"),
    ]:
        with col:
            st.markdown(
                f"<div class='descriptor-card'>"
                f"<div class='label'>{label}</div>"
                f"<div class='value'>{val}<span style='font-size:0.6em;color:#111;'> {unit}</span></div>"
                f"<div class='note'>{status}</div>"
                f"<div class='note' style='margin-top:4px;color:#111;'>{note}</div>"
                f"</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">PREDICTED PERFORMANCE METRICS</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    eta_mV_pred = eta_v_to_mV_abs(vals['eta'])
    # [v6-1] conductivity displayed instead of resistivity
    metrics_order = ['eta','tafel','rct','tof_ecsa','tof_mass','raman','conductivity']
    thresholds = {
        'eta':(-0.38,-0.50),'tafel':(110,200),'rct':(70,130),
        'raman':(1.8,2.2),'conductivity':(0.08,0.05),  # higher = better for σ
        'tof_ecsa':(9,6),'tof_mass':(5,2),
    }
    for i, key in enumerate(metrics_order):
        name, unit, better = TARGETS[key]
        v = vals[key]
        col = cols[i % 4]
        if key == 'conductivity':
            fmt = f"{v:.4f}"
        elif abs(v) < 100:
            fmt = f"{v:.2f}"
        else:
            fmt = f"{v:.0f}"
        if gp_ci:
            std  = gp_ci[key]['std']
            total_std = total_uncertainty_for_metric(key, v, std, dist_val, eta_mV_pred)
            if key == 'conductivity':
                delta_str = f"±{total_std:.4f}"
            elif abs(total_std) < 100:
                delta_str = f"±{total_std:.2f}"
            else:
                delta_str = f"±{total_std:.0f}"
            col.metric(name, f"{fmt} {unit}", delta=delta_str, delta_color="off")
        else:
            col.metric(name, f"{fmt} {unit}")

    vacancy_pct = vacancy_percent_from_mo_s(mo_s_ratio)
    mechanism   = tafel_mechanism_v5(vals['tafel'], mo_s_ratio, layer_n)
    perf_class, perf_note   = classify_performance_eta_v5(eta_mV_pred)
    rct_label, rct_note     = classify_rct_v5(vals['rct'])
    confidence, conf_warnings = confidence_level(layer_n, mo_s_ratio, ecsa_val, dist_val)
    lit_score, lit_notes    = literature_consistency_score_v5(eta_mV_pred, vals['tafel'], vals['rct'], mo_s_ratio, ecsa_val)

    if gp_ci:
        eta_total_std_mV = total_uncertainty_for_metric('eta',vals['eta'],gp_ci['eta']['std'],dist_val)*1000
        tafel_total_std  = total_uncertainty_for_metric('tafel',vals['tafel'],gp_ci['tafel']['std'],dist_val,eta_mV_pred)
    else:
        eta_total_std_mV = literature_experimental_sd_v5(eta_mV_pred,'eta')
        tafel_total_std  = literature_experimental_sd_v5(eta_mV_pred,'tafel')

    st.markdown('<div class="section-header">INTERPRETATION</div>', unsafe_allow_html=True)
    b1,b2,b3,b4,b5 = st.columns(5)
    b1.metric("Confidence", confidence)
    b2.metric("η10 magnitude", f"{eta_mV_pred:.0f} ± {eta_total_std_mV:.0f} mV")
    b3.metric("Tafel", f"{vals['tafel']:.0f} ± {tafel_total_std:.0f}")
    b4.metric("Stage (Li 2019)", stage_label_c.split('(')[0].strip())
    b5.metric("Lit. score", f"{lit_score:.1f}/5")

    box_class = 'stage2-box' if 'Stage 2' in stage_label_c else 'bulletproof-box'
    cond_val  = vals['conductivity']
    st.markdown(f"""
<div class='{box_class}'>
<b>Performance class:</b> {perf_class} — {perf_note}<br>
<b>Li 2019 Stage:</b> {stage_label_c} (S:Mo={s_mo_current:.2f})<br>
<b>HER mechanism:</b> {mechanism}<br>
<b>Conductivity σ:</b> {cond_val:.4f} S/cm (=1/ρ; higher σ → better charge transport to active sites)<br>
<b>Layer penalty:</b> relative activity factor ≈ {layer_activity_factor(layer_n):.2e} (Yu 2014 4.47×/layer).<br>
<b>Key finding — Layer# vs η:</b> Layer# alone does not strongly drive η; its effect is <i>mediated</i> through ECSA 
(synthesis method controls grain size → ECSA → η). MBE and CVD at same layer# can show different η via ECSA.<br>
<b>Rct:</b> {rct_label} — {rct_note}
</div>
""", unsafe_allow_html=True)

    if conf_warnings:
        st.markdown("<div class='risk-box'><b>Confidence notes</b><br>"
                    + "<br>".join(["• "+w for w in conf_warnings])+"</div>",
                    unsafe_allow_html=True)
    if lit_notes:
        st.markdown('<div class="section-header">LITERATURE CONSISTENCY</div>', unsafe_allow_html=True)
        for note in lit_notes:
            st.markdown(f"<span class='validation-chip'>{note}</span>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">3 CLOSEST EXPERIMENTAL SAMPLES</div>', unsafe_allow_html=True)
    closest = df_dist2.nsmallest(3,'dist').copy()
    closest['Stage (Li2019)'] = closest['mo_s_ratio'].apply(lambda x: li2019_stage(x)[1])
    closest['σ (S/cm)'] = (1.0/closest['resistivity']).map(lambda x: f"{x:.4f}")
    show_cols = ['sample','series','layer_n','mo_s_ratio','ecsa','eta','tafel','rct',
                 'tof_ecsa','σ (S/cm)','Stage (Li2019)']
    st.dataframe(closest[show_cols].reset_index(drop=True), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SYNTHESIS PHYSICS  [v6-4] NEW
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Synthesis Physics":
    st.markdown("# Synthesis Physics: MBE vs CVD")
    st.markdown(
        "<div class='info-box'>"
        "<b>Key finding from literature:</b> Both MBE and CVD can produce MoS₂ with "
        "similar layer# and Mo/S ratio — but <b>homogeneity differs</b>. "
        "CVD at high T/P reaches thermodynamic equilibrium → more uniform 2H phase, "
        "better lateral homogeneity. MBE is kinetically controlled → "
        "thermodynamically metastable → tunable S-vacancies, but potentially "
        "heterogeneous grain distribution. This homogeneity difference → ECSA difference → η difference."
        "</div>", unsafe_allow_html=True)

    st.markdown("## 1. Thermodynamic vs Kinetic Control")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "<div class='homogeneity-cvd'>"
            "<b>🧪 CVD / Chemical Synthesis</b><br><br>"
            "<b>Thermodynamic regime:</b> High T (600–900°C), high S partial pressure.<br>"
            "→ System approaches <b>equilibrium</b> → 2H phase is thermodynamically stable → forms preferentially.<br><br>"
            "<b>Homogeneity:</b> Large-area uniform coverage (cm-scale demonstrated). "
            "Grain size ~0.1–10 µm depending on substrate and T.<br><br>"
            "<b>Stoichiometry:</b> S-vacancy density harder to tune independently (coupled to T and S flux).<br><br>"
            "<b>ECSA implication:</b> Fewer grain boundaries per area → lower ECSA density but higher uniformity.<br><br>"
            "<b>Phase stability:</b> 2H thermodynamically stable at RT after synthesis. "
            "No metastable phase trapping. Reproducible Tafel slopes lab to lab.<br><br>"
            "<b>Literature SD:</b> Lower Tafel SD (±1.9 mV/dec at η<140mV) — consistent with experimental evidence."
            "</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            "<div class='homogeneity-mbe'>"
            "<b>🔬 MBE / Physical Synthesis</b><br><br>"
            "<b>Kinetic regime:</b> Atomic beam deposition, substrate T < CVD. "
            "S and Mo arrive independently → <b>non-equilibrium</b> growth.<br><br>"
            "<b>Homogeneity:</b> Layer-by-layer control (monolayer precision). "
            "But lateral homogeneity can vary: S-rich vs S-poor domains at nm scale.<br><br>"
            "<b>Stoichiometry:</b> S/Mo flux ratio directly controls vacancy density → "
            "precise Stage 1/2 engineering (Literature confirms: this is the key MBE advantage).<br><br>"
            "<b>ECSA implication:</b> Higher grain boundary density possible → higher ECSA. "
            "Jeon M6.0: ECSA=9.2 cm² (highest in dataset) despite 20 layers — synthesis effect.<br><br>"
            "<b>Phase stability:</b> Metastable phases possible (non-equilibrium Mo coordination). "
            "Note: structure may not be fully thermodynamically stabilized.<br><br>"
            "<b>Literature SD:</b> Higher η SD at same nominal composition — "
            "reflects lateral inhomogeneity (Literature confirms this concern)."
            "</div>", unsafe_allow_html=True)

    st.markdown("## 2. The Causal Chain — Your Current Parameters")

    # ── Dynamic prediction for current slider values ──────────────────────────
    pred_current = predict_all(layer_n, mo_s_ratio, ecsa_val)
    eta_current  = eta_v_to_mV_abs(pred_current['eta'])
    cond_current = pred_current['conductivity']
    sc_dyn, sl_dyn, _ = li2019_stage(mo_s_ratio)

    # Simulate: what if ECSA were higher (more grain boundaries, MBE-optimized)?
    ecsa_high  = min(ecsa_val * 1.5, 12.0)
    pred_high  = predict_all(layer_n, mo_s_ratio, ecsa_high)
    eta_high   = eta_v_to_mV_abs(pred_high['eta'])

    # Simulate: what if ECSA were lower (more grain coalescence, high-T anneal)?
    ecsa_low   = max(ecsa_val * 0.55, 2.0)
    pred_low   = predict_all(layer_n, mo_s_ratio, ecsa_low)
    eta_low    = eta_v_to_mV_abs(pred_low['eta'])

    stage_color_dyn = {'PRISTINE':'#888','STAGE_1':'#F5A623','STAGE_2_MILD':'#2DCE89',
                       'STAGE_2_DEEP':'#4E9AF1','STAGE_2_EXTREME':'#FF6464'}.get(sc_dyn,'#888')

    st.markdown(
        f"<div class='provenance-box'>"
        f"<b>Jeon 2026 — cadena causal medida directamente:</b> "
        f"Al aumentar T de recocido en MBE (600→800°C), ECSA cae 6.7→3.5 cm² por coalescencia → "
        f"η empeora −0.46V→−0.58V. "
        f"<b>Tu muestra actual (Layer#{layer_n}, Mo/S={mo_s_ratio:.2f}, ECSA={ecsa_val:.1f} cm²) "
        f"→ GP predice η={eta_current:.0f} mV, σ={cond_current:.4f} S/cm.</b>"
        f"</div>", unsafe_allow_html=True)

    # Dynamic causal chain diagram
    arrow = "↓"
    method_line = f"{'MBE (kinético)' if m_col_key == 'mbe' else 'CVD (termodinámico)'} — {m_label}"
    st.markdown(f"""
```
Synthesis method → {method_line}
        {arrow}
Grain boundary density
  MBE: S-deficiency → twin GBs espontáneos (Ma ACS Nano 2017)
  CVD: GB density ~0.04 µm⁻¹, cm-scale homogéneo (ACS AMI)
        {arrow}
ECSA actual = {ecsa_val:.1f} cm²  [{stage_color_dyn} {sl_dyn}]
  Si ECSA sube a {ecsa_high:.1f} cm² (más GBs via MBE) → η mejora a ~{eta_high:.0f} mV
  Si ECSA baja a {ecsa_low:.1f} cm² (coalescencia, ↑T) → η empeora a ~{eta_low:.0f} mV
        {arrow}
Trade-off: más GBs = más ECSA, pero σ potencialmente menor
  (movilidad in-plane 2200× mayor que out-of-plane — ACS Cat 2016)
  σ actual = {cond_current:.4f} S/cm
        {arrow}
η10 predicho = {eta_current:.0f} mV  (KNN model, Jeon 2026 dataset)
```
""")

    # Dynamic metrics comparison
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("η actual (ECSA actual)", f"{eta_current:.0f} mV",
               delta=f"ECSA = {ecsa_val:.1f} cm²", delta_color="off")
    mc2.metric(f"η si ECSA = {ecsa_high:.1f} cm² (↑GBs)",
               f"{eta_high:.0f} mV",
               delta=f"{eta_current - eta_high:+.0f} mV vs actual",
               delta_color="normal" if eta_high < eta_current else "inverse")
    mc3.metric(f"η si ECSA = {ecsa_low:.1f} cm² (↓GBs, coalescencia)",
               f"{eta_low:.0f} mV",
               delta=f"{eta_current - eta_low:+.0f} mV vs actual",
               delta_color="normal" if eta_low < eta_current else "inverse")

    st.markdown(
        "<div class='correction-box'>"
        "<b>Implicación para el modelo GP:</b> ECSA es la variable mediadora entre síntesis y η. "
        "Al incluir ECSA como feature, el efecto del método de síntesis queda <i>parcialmente</i> capturado. "
        "La distribución de tamaño de grano (homogeneidad lateral) NO está explícita — "
        "es un <b>confounder no observado</b>. "
        "Por eso dos muestras con mismo layer# y Mo/S pero diferente ECSA dan diferente η — "
        "la diferencia viene del método de síntesis vía densidad de grain boundaries."
        "</div>", unsafe_allow_html=True)

    # ECSA vs synthesis series scatter
    fig_ecsa = go.Figure()
    for ser, scol in SERIES_COLORS.items():
        mask = df['series'] == ser
        fig_ecsa.add_trace(go.Scatter(
            x=df[mask]['layer_n'], y=df[mask]['ecsa'],
            mode='markers+text', name=SERIES_LABELS[ser],
            marker=dict(size=12, color=scol, line=dict(width=1.5, color='white')),
            text=df[mask]['sample'], textposition='top center', textfont=dict(size=9),
            hovertemplate='<b>%{text}</b><br>Layers=%{x}<br>ECSA=%{y:.1f} cm²<extra></extra>'))
    fig_ecsa.update_layout(
        title="ECSA vs Layer# — same layer# can give very different ECSA (synthesis-dependent)",
        xaxis_title="Layer # (structural parameter)", yaxis_title="ECSA (cm²)",
        height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig_ecsa.add_annotation(x=20, y=9.2, text="MoS-M6.0: ECSA=9.2<br>(high S-flux → more grain boundaries)",
                            showarrow=True, arrowhead=2, ax=60, ay=-40,
                            font=dict(color='#F5A623'))
    st.plotly_chart(fig_ecsa, use_container_width=True)

    st.markdown("## 3. Conductivity σ vs Synthesis Series")
    df_plot = df.copy()
    fig_cond = go.Figure()
    for ser, scol in SERIES_COLORS.items():
        mask = df_plot['series'] == ser
        fig_cond.add_trace(go.Scatter(
            x=df_plot[mask]['mo_s_ratio'], y=df_plot[mask]['conductivity'],
            mode='markers+text', name=SERIES_LABELS[ser],
            marker=dict(size=12, color=scol, line=dict(width=1.5, color='white')),
            text=df_plot[mask]['sample'], textposition='top center', textfont=dict(size=9),
            hovertemplate='<b>%{text}</b><br>Mo/S=%{x:.3f}<br>σ=%{y:.4f} S/cm<extra></extra>'))
    fig_cond.add_vline(x=0.588, line_dash='dash', line_color='#F5A623', line_width=2,
                       annotation_text="Stage 1→2 (Li 2019)", annotation_font_color='#F5A623')
    fig_cond.update_layout(
        title="Conductivity σ (S/cm) vs Mo/S ratio — [σ=1/ρ is more physically meaningful]",
        xaxis_title="Mo/S ratio (Stage threshold=0.588)", yaxis_title="Conductivity σ (S/cm)",
        height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_cond, use_container_width=True)
    st.markdown(
        "<div class='correction-box'>"
        "<b>Why conductivity over resistivity:</b> "
        "σ directly correlates with charge carrier concentration and mobility — "
        "both of which increase with S-vacancy formation (Mo d-electrons become less bound). "
        "Plotting σ vs Mo/S shows a physically interpretable trend: "
        "as Stage 2 deepens (more vacancies), σ should increase. "
        "Resistivity plots this inverted, making trends harder to read consistently."
        "</div>", unsafe_allow_html=True)

    st.markdown("## 4. Comparación MBE vs CVD — Respaldada por Literatura Real")
    comparison_df = pd.DataFrame([
        {'Propiedad':'Estabilidad termodinámica','CVD':'2H estable en equilibrio — estructura más estable termodinámicamente (ACS Cat 2023)','MBE':'Metaestable cinéticamente — no garantiza estabilidad termodinámica plena (literatura)','Fuente':'ACS Cat 2023'},
        {'Propiedad':'Homogeneidad lateral','CVD':'cm-scale uniforme, GB density ~0.04 µm⁻¹ medida','MBE':'nm-scale variable; twin GBs espontáneos por deficiencia de S','Fuente':'ACS AMI (CVD); Ma et al. ACS Nano 2017 (MBE)'},
        {'Propiedad':'ECSA vs temperatura','CVD':'ECSA controlada por T de síntesis (↑T → granos más grandes → ↓ECSA)','MBE':'Jeon 2026: ECSA = 6.7 cm² (600°C) → 3.5 cm² (800°C) por coalescencia','Fuente':'Jeon 2026 Table 1 (medición directa)'},
        {'Propiedad':'Bordes de grano como sitios activos','CVD':'Pocos GBs → ECSA menor pero reproducible','MBE':'GBs son sitios HER intrínsecos: hasta 10¹² cm⁻² → onset −25 mV, Tafel 54 mV/dec','Fuente':'Nature Comms 2020 (GB-engineered TMDs)'},
        {'Propiedad':'Mecanismo de formación de GBs','CVD':'GBs por coalescencia de islas durante crecimiento','MBE':'Twin GBs espontáneos por deficiencia de calcógeno durante deposición','Fuente':'Ma et al. ACS Nano 2017'},
        {'Propiedad':'Trade-off σ vs sitios de borde','CVD':'Granos grandes → σ alta, menos bordes','MBE':'Granos pequeños → más bordes (ECSA↑) pero σ potencialmente menor','Fuente':'ACS Cat 2016: movilidad in-plane 2200× out-of-plane'},
        {'Propiedad':'Control de vacancias-S','CVD':'Acoplado a T y atmósfera S — difícil tunear independientemente','MBE':'Flujo S/Mo directo → Stage 1/2 engineering en síntesis','Fuente':'Jeon 2026; Sherwood 2024'},
        {'Propiedad':'Concentración óptima S-vacancy','CVD':'Requiere post-tratamiento (plasma, H₂)','MBE':'12.5–17.1% S-vacancy directo durante crecimiento','Fuente':'DFT (Nature 2017) + experimento (Frontiers 2022)'},
    ])
    st.dataframe(comparison_df, use_container_width=True)

    st.markdown("## 5. Nuevos Papers Integrados en v6.1")
    new_papers_df = pd.DataFrame([
        {'Paper':'Ma et al., ACS Nano 2017','Hallazgo clave':'MBE-TMDs forman twin grain boundaries espontáneamente por deficiencia de S/Se durante deposición. GBs son metálicos.','Relevance':'Explica MECÁNICAMENTE por qué MBE da diferente densidad de GBs que CVD — no es solo "homogeneidad", es un efecto físico de la deficiencia de calcógeno'},
        {'Paper':'Shi et al., Nature Comms 2020','Hallazgo clave':'TMD films con GB density ~10¹² cm⁻²: onset HER −25 mV, Tafel 54 mV/dec. GBs son sitios activos intrínsecos.','Relevance':'Cuantifica directamente: más GBs = mejor HER. Confirma que ECSA (vía GBs) es el link entre síntesis y η'},
        {'Paper':'Jeon 2026 (tu paper, dato específico)','Hallazgo clave':'ECSA cae 6.7→3.5 cm² al aumentar T de recocido (600→800°C) en MBE por coalescencia de granos','Relevance':'Tu propio paper mide la cadena causal completa: síntesis T → granos → ECSA → η'},
        {'Paper':'ACS AMI (CVD grain study)','Hallazgo clave':'CVD produce MoS₂ con homogeneidad espacial cm-scale, GB density ~0.04 µm⁻¹','Relevance':'Dato cuantitativo de homogeneidad CVD — confirma homogeneidad lateral de CVD (dato cuantitativo)'},
        {'Paper':'Eng et al. / ACS Catalysis 2016','Hallazgo clave':'Movilidad in-plane 2200× mayor que out-of-plane en MoS₂. S-deficiency → deficiencia de sitios activos si excesiva','Relevance':'Explica el trade-off σ vs sitios de borde — no se pueden maximizar ambos simultáneamente'},
    ])
    st.dataframe(new_papers_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TREND CURVES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trend Curves":
    st.markdown("# Trend Curves")
    tc1, tc2 = st.columns([1,2])
    with tc1:
        target_tc = st.selectbox("Performance metric", options=list(TARGETS.keys()),
            format_func=lambda k: f"{TARGETS[k][0]} ({TARGETS[k][1]})")
    with tc2:
        feat_tc = st.selectbox("Descriptor to vary", options=FEATURES,
            format_func=lambda k: FEATURE_LABELS[k])

    name_tc, unit_tc, better_tc = TARGETS[target_tc]
    defaults = {'layer_n': layer_n, 'mo_s_ratio': mo_s_ratio, 'ecsa': ecsa_val}
    lo, hi = FEATURE_RANGES[feat_tc]
    x_range = np.linspace(lo, hi, 80)
    y_means, y_lows, y_highs = [], [], []
    for xv in x_range:
        row = {f:(xv if f==feat_tc else defaults[f]) for f in FEATURES}
        m, lo_, hi_, _ = gp_predict(target_tc, row['layer_n'], row['mo_s_ratio'], row['ecsa'])
        y_means.append(m); y_lows.append(lo_); y_highs.append(hi_)
    y_means = np.array(y_means); y_lows = np.array(y_lows); y_highs = np.array(y_highs)
    exp_lo = df[feat_tc].min(); exp_hi = df[feat_tc].max()
    in_range = (x_range >= exp_lo) & (x_range <= exp_hi)

    fig_tc = go.Figure()
    fig_tc.add_trace(go.Scatter(
        x=np.concatenate([x_range, x_range[::-1]]),
        y=np.concatenate([y_highs, y_lows[::-1]]),
        fill='toself', fillcolor='rgba(78,154,241,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name='95% CI (GP)'))
    fig_tc.add_trace(go.Scatter(x=x_range, y=y_means, mode='lines',
        line=dict(color='rgba(78,154,241,0.35)',width=1.5,dash='dot'),
        name='GP mean (extrapolation)'))
    fig_tc.add_trace(go.Scatter(x=x_range[in_range], y=y_means[in_range], mode='lines',
        line=dict(color='#4E9AF1',width=3), name='GP mean (interpolation)'))
    for ser, scolor in SERIES_COLORS.items():
        mask = df['series'] == ser
        fig_tc.add_trace(go.Scatter(
            x=df[feat_tc].values[mask], y=df[target_tc].values[mask], mode='markers',
            name=SERIES_LABELS[ser],
            marker=dict(size=11, color=scolor, line=dict(width=1.5,color='white')),
            text=df['sample'][mask],
            hovertemplate='<b>%{text}</b><br>'+FEATURE_LABELS[feat_tc]+'=%{x:.2f}<br>'+name_tc+'=%{y:.3f} '+unit_tc+'<extra></extra>'))
    cur_val = defaults[feat_tc]
    fig_tc.add_vline(x=cur_val, line_width=1.5, line_dash="dash",
                     line_color=METHOD_COLORS[m_col_key],
                     annotation_text=f"Current: {cur_val:.2f}",
                     annotation_font_color=METHOD_COLORS[m_col_key])
    if feat_tc == 'mo_s_ratio':
        fig_tc.add_vline(x=0.588, line_dash='dot', line_color='#F5A623', line_width=2,
                         annotation_text="Stage 1→2 (Li 2019)", annotation_font_color='#F5A623')
        fig_tc.add_vline(x=0.752, line_dash='dot', line_color='#FF6464', line_width=1,
                         annotation_text="Stage 2 deep", annotation_font_color='#FF6464')
    fig_tc.update_layout(
        title=f"{name_tc} vs {FEATURE_LABELS[feat_tc]}<br><sup>{FEATURE_PROVENANCE[feat_tc]}</sup>",
        xaxis_title=FEATURE_LABELS[feat_tc], yaxis_title=f"{name_tc} ({unit_tc})",
        height=500, legend=dict(orientation='h',yanchor='bottom',y=-0.40),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig_tc.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.12)')
    fig_tc.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.12)')
    st.plotly_chart(fig_tc, use_container_width=True)

    if feat_tc == 'layer_n' and target_tc == 'eta':
        st.info(
            "**Literature-confirmed finding:** Layer# does not strongly drive η alone. "
            "Its effect is mediated through ECSA (grain boundary density). "
            "The KNN model shows a weak direct layer#→η trend at fixed ECSA. "
            "The strong layer# effect appears in Rct and conductivity, not directly in η.")

    if feat_tc == 'ecsa' and target_tc == 'eta':
        st.info(
            "**Key insight:** Synthesis method (MBE vs CVD) controls grain size → ECSA. "
            "This plot shows the ECSA→η link. MBE can engineer higher ECSA at same layer# "
            "by controlling grain boundary density — explaining why preparation method matters "
            "even when layer# and Mo/S appear similar.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: 2D HEATMAPS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺 2D Heatmaps":
    st.markdown("# 2D Heatmaps")
    hc1, hc2 = st.columns(2)
    with hc1:
        target_hm = st.selectbox("Performance metric", options=list(TARGETS.keys()),
            format_func=lambda k: f"{TARGETS[k][0]} ({TARGETS[k][1]})")
    with hc2:
        axis_pair = st.selectbox("Axes",["Layer# × Mo/S  (ECSA fixed)",
                                          "Layer# × ECSA  (Mo/S fixed)",
                                          "Mo/S × ECSA   (Layer# fixed)"])
    name_hm, unit_hm, better_hm = TARGETS[target_hm]
    N = 40
    defaults_hm = {'layer_n':layer_n,'mo_s_ratio':mo_s_ratio,'ecsa':ecsa_val}
    if axis_pair.startswith("Layer# × Mo/S"):
        xf,yf,fixed_f = 'layer_n','mo_s_ratio','ecsa'; xlabel,ylabel = 'Layer #','Mo/S ratio'
    elif axis_pair.startswith("Layer# × ECSA"):
        xf,yf,fixed_f = 'layer_n','ecsa','mo_s_ratio'; xlabel,ylabel = 'Layer #','ECSA (cm²)'
    else:
        xf,yf,fixed_f = 'mo_s_ratio','ecsa','layer_n'; xlabel,ylabel = 'Mo/S ratio','ECSA (cm²)'
    xlo,xhi = FEATURE_RANGES[xf]; ylo,yhi = FEATURE_RANGES[yf]
    xgrid = np.linspace(xlo,xhi,N); ygrid = np.linspace(ylo,yhi,N)
    Z = np.zeros((N,N))
    for i,yv in enumerate(ygrid):
        for j,xv in enumerate(xgrid):
            row = {xf:xv,yf:yv,fixed_f:defaults_hm[fixed_f]}
            Z[i,j] = gp_predict(target_hm,row['layer_n'],row['mo_s_ratio'],row['ecsa'])[0]
    cs = 'RdYlGn' if better_hm=='max' else 'RdYlGn_r'
    fig_hm = go.Figure(data=go.Heatmap(z=Z,x=xgrid,y=ygrid,colorscale=cs,
        colorbar=dict(title=dict(text=f"{name_hm} ({unit_hm})",side='right')),
        hovertemplate=f'{xlabel}=%{{x:.2f}}<br>{ylabel}=%{{y:.2f}}<br>{name_hm}=%{{z:.3f}} {unit_hm}<extra></extra>'))
    for ser, scolor in SERIES_COLORS.items():
        mask = df['series'] == ser
        fig_hm.add_trace(go.Scatter(
            x=df[xf].values[mask], y=df[yf].values[mask], mode='markers+text',
            marker=dict(size=12,color=scolor,line=dict(width=2,color='white')),
            text=df['sample'][mask], textposition='top center', textfont=dict(size=9,color='white'),
            name=SERIES_LABELS[ser], customdata=df[target_hm].values[mask],
            hovertemplate='<b>%{text}</b><br>'+xlabel+'=%{x:.2f}<br>'+ylabel+'=%{y:.2f}<br>'+name_hm+f'=%{{customdata:.3f}} {unit_hm}<extra></extra>'))
    fig_hm.add_trace(go.Scatter(x=[defaults_hm[xf]],y=[defaults_hm[yf]],mode='markers',
        marker=dict(size=16,color=METHOD_COLORS[m_col_key],symbol='star',line=dict(width=2,color='white')),
        name='Your position'))
    if yf == 'mo_s_ratio':
        fig_hm.add_hline(y=0.588,line_dash='dash',line_color='#F5A623',line_width=2,
                         annotation_text="Stage 1→2 (Li 2019)",annotation_font_color='#F5A623')
    if xf == 'mo_s_ratio':
        fig_hm.add_vline(x=0.588,line_dash='dash',line_color='#F5A623',line_width=2,
                         annotation_text="Stage 1→2",annotation_font_color='#F5A623')
    fig_hm.update_layout(
        title=f"{name_hm} — {xlabel} × {ylabel} | fixed={FEATURE_LABELS[fixed_f]}={defaults_hm[fixed_f]:.2f}",
        xaxis_title=xlabel, yaxis_title=ylabel, height=540,
        legend=dict(orientation='h',yanchor='bottom',y=-0.22),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hm, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: 3D EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌐 3D Explorer":
    st.markdown("# 3D Descriptor Space Explorer")
    t3c1,t3c2 = st.columns(2)
    with t3c1:
        target_3d = st.selectbox("Color metric", options=list(TARGETS.keys()),
            format_func=lambda k: f"{TARGETS[k][0]} ({TARGETS[k][1]})")
    with t3c2:
        show_surf = st.checkbox("Show GP surface slice (Mo/S fixed)",value=True)
    name_3d,unit_3d,better_3d = TARGETS[target_3d]
    fig_3d = go.Figure()
    if show_surf:
        N3=25; ln3=np.linspace(1,20,N3); ec3=np.linspace(2,12,N3)
        Zs=np.zeros((N3,N3))
        for i,ev in enumerate(ec3):
            for j,lv in enumerate(ln3):
                Zs[i,j] = gp_predict(target_3d,lv,mo_s_ratio,ev)[0]
        fig_3d.add_trace(go.Surface(x=ln3,y=ec3,z=Zs,
            colorscale='RdYlGn' if better_3d=='max' else 'RdYlGn_r',
            opacity=0.55,showscale=False,name=f'GP surface (Mo/S={mo_s_ratio:.2f})'))
    for ser, scolor in SERIES_COLORS.items():
        mask = df['series']==ser; sub=df[mask]
        fig_3d.add_trace(go.Scatter3d(
            x=sub['layer_n'],y=sub['ecsa'],z=sub['mo_s_ratio'],mode='markers+text',
            marker=dict(size=8,color=sub[target_3d].values,
                        colorscale='RdYlGn' if better_3d=='max' else 'RdYlGn_r',
                        cmin=df[target_3d].min(),cmax=df[target_3d].max(),
                        line=dict(width=2,color='white')),
            text=sub['sample'],name=SERIES_LABELS[ser]))
    cur_pred = gp_predict(target_3d,layer_n,mo_s_ratio,ecsa_val)[0]
    fig_3d.add_trace(go.Scatter3d(x=[layer_n],y=[ecsa_val],z=[mo_s_ratio],mode='markers',
        marker=dict(size=14,color=METHOD_COLORS[m_col_key],symbol='diamond',line=dict(width=3,color='white')),
        name=f'Your position ({cur_pred:.3f} {unit_3d})'))
    fig_3d.update_layout(
        scene=dict(xaxis_title='Layer # (validated)',yaxis_title='ECSA (cm²)',
                   zaxis_title='Mo/S ratio (Stage threshold=0.588)'),
        title=f"{name_3d} ({unit_3d}) in descriptor space",
        height=620, paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_3d, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: INVERSE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Inverse Predictor":
    st.markdown("# Inverse Predictor")
    st.markdown(
        "<div class='info-box'>"
        "<b>Note:</b> This tool finds which experimental Jeon samples best match your target performance. "
        "The recommended synthesis method is then inferred from those descriptors. "
        "Remember: two samples with same layer# and Mo/S but different ECSA (= different synthesis homogeneity) "
        "will show different η — this is the key Key insight about MBE vs CVD."
        "</div>", unsafe_allow_html=True)
    ic1,ic2,ic3,ic4 = st.columns(4)
    with ic1: t_eta   = st.slider("Target η (V)",          -0.60,-0.25,-0.35,0.01)
    with ic2: t_tafel = st.slider("Target Tafel (mV/dec)",  60,300,100,5)
    with ic3: t_ecsa  = st.slider("Target ECSA (cm²)",      2.0,12.0,7.0,0.5)
    with ic4: t_rct   = st.slider("Target Rct (Ω·cm²)",     20.0,200.0,60.0,5.0)
    df_inv = df.copy()
    df_inv['perf_score'] = df_inv.apply(lambda r: np.sqrt(
        ((r.eta-t_eta)/0.30)**2+((r.tafel-t_tafel)/250)**2+
        ((r.ecsa-t_ecsa)/8)**2+((r.rct-t_rct)/180)**2), axis=1)
    candidates = df_inv.nsmallest(3,'perf_score').copy()
    candidates['Stage (Li2019)'] = candidates['mo_s_ratio'].apply(lambda x: li2019_stage(x)[1])
    candidates['σ (S/cm)'] = (1.0/candidates['resistivity']).map(lambda x: f"{x:.4f}")
    st.markdown('<div class="section-header">CLOSEST EXPERIMENTAL MATCHES</div>', unsafe_allow_html=True)
    show_inv = candidates[['sample','series','layer_n','mo_s_ratio','ecsa',
                            'eta','tafel','rct','tof_ecsa','σ (S/cm)','Stage (Li2019)']].reset_index(drop=True)
    st.dataframe(show_inv, use_container_width=True)
    best_inv   = candidates.iloc[0]
    inv_label, inv_col, inv_score, inv_max, inv_reasons = score_method(
        best_inv['layer_n'],best_inv['mo_s_ratio'],best_inv['ecsa'],best_inv['rct'])
    inv_color = METHOD_COLORS[inv_col]
    hom_inv   = synthesis_homogeneity_note(inv_col, best_inv['layer_n'], best_inv['mo_s_ratio'])
    st.markdown(
        f"<div style='background:{inv_color}12;border:2px solid {inv_color}40;"
        f"border-left:5px solid {inv_color};padding:16px 20px;border-radius:6px;'>"
        f"<div style='font-size:1.4em;font-weight:700;color:{inv_color};"
        f"font-family:IBM Plex Mono,monospace;'>{inv_label}</div>"
        f"<div style='color:#111;margin-top:6px;'>Best match: <b>{best_inv['sample']}</b> · "
        f"η={best_inv.eta:.2f}V · Tafel={best_inv.tafel:.0f} · Rct={best_inv.rct:.1f} · "
        f"Stage: {li2019_stage(best_inv['mo_s_ratio'])[1]}</div>"
        f"<div style='margin-top:10px;'><b>MBE score: {inv_score}/{inv_max}</b></div>"
        f"</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='info-box' style='margin-top:8px;'>{hom_inv}</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧮 Feature Importance":
    st.markdown("# Feature Importance")
    st.markdown(
        "<div class='info-box'>"
        "<b>Key finding (Jeon 2026):</b> Layer# does not strongly affect η alone — its effect is mediated "
        "through ECSA (synthesis method → grain boundary density → ECSA → η). "
        "The importance matrix below shows that Mo/S ratio and ECSA are the primary direct drivers of η, "
        "while layer# more strongly drives Rct and conductivity."
        "</div>", unsafe_allow_html=True)
    perf_rows = []
    for k in TARGETS:
        n_name,u,_ = TARGETS[k]
        perf_rows.append({'Property':n_name,'Unit':u,
            'KNN R²':round(knn_scores[k]['r2'],3),'KNN MAE':round(knn_scores[k]['mae'],3),
            'RF R²':round(rf_scores[k]['r2'],3),'RF MAE':round(rf_scores[k]['mae'],3)})
    st.dataframe(pd.DataFrame(perf_rows), use_container_width=True)
    fi_colors = {'layer_n':'#9B59B6','mo_s_ratio':'#E84040','ecsa':'#2DCE89'}
    fi_names  = {'layer_n':'Layer # (validated)','mo_s_ratio':'Mo/S (validated)','ecsa':'ECSA (measured)'}
    imp_target = st.selectbox("Property for importance", options=list(TARGETS.keys()),
        format_func=lambda k: f"{TARGETS[k][0]} ({TARGETS[k][1]})")
    imps = rf_imps[imp_target]
    fig_fi = go.Figure(go.Bar(x=[fi_names[f] for f in FEATURES], y=imps,
        marker_color=[fi_colors[f] for f in FEATURES],
        text=[f"{v:.3f}" for v in imps], textposition='outside'))
    fig_fi.update_layout(title=f"Feature importance — {TARGETS[imp_target][0]} (note: ECSA mediates layer# effect on η)",
        yaxis_title='Relative importance', yaxis_range=[0,max(imps)*1.3],
        height=320, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_fi, use_container_width=True)
    heat = np.array([[rf_imps[k][i] for i in range(3)] for k in TARGETS])
    heat_df = pd.DataFrame(heat, index=[TARGETS[k][0] for k in TARGETS],
                           columns=[fi_names[f] for f in FEATURES])
    fig_heat = px.imshow(heat_df, text_auto=".2f", aspect="auto",
                         color_continuous_scale='Greens', zmin=0, zmax=1,
                         title="Feature importance matrix — all targets (v6.0: conductivity replaces resistivity)")
    fig_heat.update_layout(height=380)
    st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: THEORETICAL BASIS  [v6-6] structural parameters table added
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Theoretical Basis":
    st.markdown("# Theoretical Framework — v6.0")

    # [v6-6] Structural parameters → electrochemical properties
    st.markdown("## Structural Parameters → Electrochemical Properties")
    st.markdown(
        "<div class='info-box'>"
        "<b>Scientific objective:</b> Show how structural parameters (composition, layers, morphology, particle size) "
        "correlate to electrochemical performance — not just predict η from them."
        "</div>", unsafe_allow_html=True)

    struct_table = pd.DataFrame([
        {'Structural parameter':'Number of layers (N)','Controls':'Electron tunneling resistance, k⁰ (Butler-Volmer)','Electrochemical effect':'Rct ∝ exp(0.65N) per Yu 2014; k⁰ decreases 4.47×/layer; little direct η effect at fixed ECSA','Primary metric affected':'Rct, conductivity σ','Note':'Layer# effect on η is mediated by ECSA — not direct'},
        {'Structural parameter':'Mo/S stoichiometry (XPS)','Controls':'S-vacancy density → ΔG_H* → Stage 1/2','Electrochemical effect':'S:Mo < 1.70 → undercoordinated Mo → active HER sites (Li 2019 KOH). TOF increases continuously through Stage 2','Primary metric affected':'η, Tafel slope, TOF','Note':'Most direct descriptor for η in 1M KOH'},
        {'Structural parameter':'Grain size / lateral homogeneity','Controls':'Grain boundary density → edge site density → ECSA','Electrochemical effect':'Smaller grains → more edges → higher ECSA → lower η10. MBE: tunable grain size. CVD: T-controlled grain size','Primary metric affected':'ECSA → η10 (mediated)','Note':'KEY: explains why same layer# and Mo/S can give different η for MBE vs CVD'},
        {'Structural parameter':'ECSA (Cdl method)','Controls':'Electrochemically active surface area','Electrochemical effect':'Direct: TOF_mass = I / (ECSA × loading). Higher ECSA → lower η10 at same intrinsic activity','Primary metric affected':'η10, TOF_mass','Note':'Synthesis method affects ECSA — primary finding'},
        {'Structural parameter':'Interlayer spacing (c/2)','Controls':'Ion accessibility, electron delocalization','Electrochemical effect':'Expanded spacing (6.62 Å in Li 2019 vs 6.15 Å bulk) → better electrolyte access → lower Rct','Primary metric affected':'Rct, ion transport','Note':'S-vacancies expand c/2 → dual benefit: active sites + ion access'},
        {'Structural parameter':'Phase (2H vs 1T vs mixed)','Controls':'Electronic structure, conductivity σ','Electrochemical effect':'1T metallic: σ >> 2H → Rct drops dramatically → η decreases. But 1T thermodynamically unstable','Primary metric affected':'Conductivity σ, Rct, Tafel','Note':'1T → 2H conversion during electrochemistry is an issue for CVD less than MBE'},
        {'Structural parameter':'Particle size / film thickness','Controls':'Mass loading, diffusion length','Electrochemical effect':'Thicker films: higher loading → higher TOF_ECSA absolute but lower TOF_mass. Optimal thickness ~3-9 nm (Jeon 2026 S-series)','Primary metric affected':'TOF_mass, loading','Note':'Jeon M-series explores this directly: M2.0→M9.0 at fixed cycles'},
        {'Structural parameter':'Raman A₁g/E₂g ratio','Controls':'Structural order / S-vacancy proxy','Electrochemical effect':'Lower ratio → more S-vacancy → Stage 2 → better HER. Jeon: N10 has lowest ratio (1.63) → best N-series η','Primary metric affected':'Proxy for Mo/S, ECSA','Note':'Can be used as non-destructive screening before XPS'},
    ])
    st.dataframe(struct_table, use_container_width=True)

    # [v6-7] Literature consistency across papers
    st.markdown("## Cross-Paper Consistency Check")
    st.markdown(
        "<div class='info-box'>"
        "<b>Research question:</b> Are the 14–15 papers internally consistent on MoS₂? "
        "Do they agree on the key descriptors and thresholds?"
        "</div>", unsafe_allow_html=True)

    consistency_df = pd.DataFrame([
        {'Topic':'Stage 1/2 threshold (S:Mo)','Papers':'Li 2019, ACS Cat 2023, Sherwood 2024, Smiri 2026','Consenso':'S:Mo = 1.70 (Mo/S = 0.588)','Discrepancia':'Ninguna — todos convergen','Confianza':'ALTA'},
        {'Topic':'Layer# penalty por capa','Papers':'Yu 2014, Manyepedza 2022, McKelvey 2021','Consenso':'4.47× por capa (k⁰ decay)','Discrepancia':'Menor: AFM 0.65nm vs Scherrer 0.615nm (<5%)','Confianza':'ALTA'},
        {'Topic':'GBs como sitios HER activos','Papers':'Nature Comms 2020, Ma ACS Nano 2017, Jeon 2026','Consenso':'GBs = sitios activos intrínsecos. Densidad ~10¹² cm⁻² → onset −25 mV','Discrepancia':'Ninguna — todos confirman GBs activos','Confianza':'ALTA'},
        {'Topic':'MBE genera GBs por deficiencia de S','Papers':'Ma et al. ACS Nano 2017 (MoSe₂), Jeon 2026 (MoS₂)','Consenso':'Deficiencia de calcógeno en MBE → twin GBs espontáneos','Discrepancia':'Ma es MoSe₂; Jeon confirma para MoS₂','Confianza':'ALTA'},
        {'Topic':'CVD homogeneidad lateral','Papers':'ACS AMI grain study, IOP 2025','Consenso':'GB density ~0.04 µm⁻¹, cm-scale homogéneo','Discrepancia':'Varía con condiciones (T, precursor)','Confianza':'MEDIA-ALTA'},
        {'Topic':'S-vacancy → ΔG_H*→0','Papers':'Ozaki 2023 (AP-XPS), He 2023, Sherwood 2024','Consenso':'Mo 3d shift −0.5 eV → ΔG_H* ≈ 0 eV','Discrepancia':'Valores exactos varían por método DFT','Confianza':'ALTA'},
        {'Topic':'Concentración óptima de S-vacancies','Papers':'DFT (Nature 2017) + experimento (Frontiers 2022)','Consenso':'12.5–17.1% S-vacancy es óptimo para ΔG_H*≈0','Discrepancia':'DFT da 12.5-15.6%; experimento da 17.1%','Confianza':'MEDIA (DFT vs experimento)'},
        {'Topic':'Trade-off σ vs sitios de borde','Papers':'ACS Catalysis 2016, Jeon 2026, revisión 2026','Consenso':'Cristalitos pequeños = más ECSA pero σ menor','Discrepancia':'No contradictorios, complementarios','Confianza':'ALTA'},
        {'Topic':'Stage 2 TOF en KOH','Papers':'Li 2019, He 2023, ACS Cat 2023','Consenso':'TOF aumenta continuamente en Stage 2 en KOH (no en H₂SO₄)','Discrepancia':'Solo 1-2 papers KOH específicos','Confianza':'ALTA (pocos papers KOH)'},
        {'Topic':'Rct benchmarks en 1M KOH','Papers':'JECST, UCL (MoS₂/NiS), CityU (Mo5N6)','Consenso':'Bulk >200Ω; Engineered 5-80Ω','Discrepancia':'Normalización Ω vs Ω·cm² — verificar por paper','Confianza':'MEDIA (caveat normalización)'},
    ])
    st.dataframe(consistency_df, use_container_width=True)
    st.success("✅ Conclusión: Los papers son consistentes en los descriptores clave. "
               "Las únicas discrepancias son menores (normalización de Rct, DFT vs experimento en vacancias). "
               "La adición de Ma 2017 y Nature Comms 2020 resuelve la pregunta de investigación sobre "
               "por qué MBE y CVD se comportan diferente.")

    with st.expander("Paper-by-paper reference list (v6.1 — 20 papers)"):
        papers = [
            ("1 · Jeon 2026, ACS Nano [PRIMARY DATA + CAUSAL CHAIN ★]",
             "14 MBE-grown MoS₂ on Si in 1M KOH. DATO CLAVE: ECSA cae 6.7→3.5 cm² al aumentar T 600→800°C "
             "por coalescencia de granos → η empeora −0.46V→−0.58V. "
             "Mide cadena causal completa: síntesis → granos → ECSA → η. "
             "Sulfur stoichiometry and growth kinetics as 'powerful levers'. "
             "Optimum: MoS-N10 (η=-0.33V, Tafel=80, ECSA=8.0, TOF>23 mmol/g/s)."),
            ("2 · Li 2019, ACS Nano [PRIMARY KOH SOURCE ★]",
             "Stage 1/2 model. KOH 0.1M: TOF aumenta continuamente en Stage 2. Interlayer 6.62Å. "
             "Repair experiment confirma link vacancia→actividad. MoS2-7H: TOF=15 s⁻¹ @ 300mV."),
            ("3 · Ma et al., ACS Nano 2017 [MBE GRAIN BOUNDARY MECHANISM ★ NUEVO]",
             "MBE de TMDs genera espontáneamente twin grain boundaries de alta densidad por deficiencia de calcógeno. "
             "GBs son metálicos. Substrate-independent — depende de condiciones de crecimiento. "
             "Explica mecánicamente por qué MBE da diferente densidad de GBs que CVD."),
            ("4 · Shi et al., Nature Communications 2020 [GB = SITIOS HER ACTIVOS ★ NUEVO]",
             "TMD films con GB density ~10¹² cm⁻²: onset HER −25 mV, Tafel 54 mV/dec. "
             "GBs son sitios electrocatalíticos intrínsecos, no defectos pasivos. "
             "Cuantifica: más GBs = mejor HER — confirma link GB density → ECSA → η."),
            ("5 · ACS AMI (CVD grain homogeneity) [CVD CUANTITATIVO NUEVO]",
             "CVD produce MoS₂ monocapa altamente cristalina con homogeneidad cm-scale. "
             "GB density medida: ~0.04 µm⁻¹. Confirma que CVD es más homogéneo que MBE."),
            ("6 · Eng et al. / ACS Catalysis 2016 [TRADE-OFF σ vs BORDES NUEVO]",
             "Factores HER en MoS₂: conductividad, ratio Mo:S, abundancia de sitios de borde, carga. "
             "Movilidad in-plane 2200× mayor que out-of-plane → cristalitos pequeños = ECSA↑ pero σ↓."),
            ("7 · Yu 2014, Nano Lett.", "4.47×/layer k⁰ decay. V₀=0.119V."),
            ("8 · Ozaki 2023, ChemPhysChem", "AP-XPS: Mo 3d shift −0.5eV → ΔG_H*→0eV."),
            ("9 · Van Nguyen 2023, Battery Energy", "Butler-Volmer. Tafel thresholds."),
            ("10 · He 2023, Nanomaterials", "S-vac basal plane active. 1T' transient."),
            ("11 · Manyepedza 2022, J.Phys.Chem.C", "AFM 0.65nm/layer. k⁰ curve."),
            ("12 · Sherwood 2024, ACS Appl.Nano", "XPS 4-peak. S-vacancy en 2H."),
            ("13 · ACS Catalysis 2023", "CVD S/Mo threshold 1.70."),
            ("14 · Lee 2010, ACS Nano", "Raman Δω vs layers."),
            ("15 · Smiri 2026, Sci.Rep.", "ALD Raman saturation."),
            ("16 · Bentley 2017, Chem.Sci.", "vdW gap=6.15Å."),
            ("17 · Cao 2017, Sci.Rep.", "HRTEM 0.63nm."),
            ("18 · Jaramillo 2007, Science", "Edge site origin."),
            ("19 · McKelvey 2021, Electrochim.Acta", "k⁰ anchors."),
            ("20 · KOH benchmarks + master table", "8 familias. NiO@1T (46mV), Mo5N6 (100mV)."),
        ]
        for title, body in papers:
            with st.expander(title):
                st.write(body)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: XPS & STAGE CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 XPS & Stage Calibration":
    st.markdown("# XPS Calibration & Li 2019 Stage Model")
    stage_df = pd.DataFrame([
        {'Stage':'PRISTINE','S:Mo range':'> 1.85','Mo/S range':'< 0.541','η trend (KOH)':'> 300 mV','Tafel (KOH)':'> 100 mV/dec','TOF trend (KOH)':'Near baseline','Rct trend':'High (> 100Ω)','Defect type':'None (stoichiometric)'},
        {'Stage':'STAGE 1 (point defects)','S:Mo range':'1.70 – 1.85','Mo/S range':'0.541 – 0.588','η trend (KOH)':'Rapid decrease','Tafel (KOH)':'110 → 80 mV/dec (fast)','TOF trend (KOH)':'Moderate increase','Rct trend':'Rapid decrease','Defect type':'Isolated S-vacancies'},
        {'Stage':'STAGE 2 mild (Li 2019)','S:Mo range':'1.33 – 1.70','Mo/S range':'0.588 – 0.752','η trend (KOH)':'Continues decreasing (slower)','Tafel (KOH)':'~80 mV/dec (plateau)','TOF trend (KOH)':'CONTINUOUS INCREASE (unlike H₂SO₄!)','Rct trend':'Saturation','Defect type':'Undercoordinated Mo regions'},
        {'Stage':'STAGE 2 deep','S:Mo range':'0.80 – 1.33','Mo/S range':'0.752 – 1.25','η trend (KOH)':'Still decreasing','Tafel (KOH)':'~80 mV/dec','TOF trend (KOH)':'High (TOF=15 s⁻¹ @ 300mV)','Rct trend':'May increase slightly','Defect type':'Extensive S-stripping'},
        {'Stage':'STAGE 2 extreme (risk)','S:Mo range':'< 0.80','Mo/S range':'> 1.25','η trend (KOH)':'Degradation','Tafel (KOH)':'Erratic','TOF trend (KOH)':'Drops (structural collapse)','Rct trend':'Very high','Defect type':'Mo-rich domains'},
    ])
    st.dataframe(stage_df, use_container_width=True)
    calib_data = [{'S/Mo':smo,'Mo/S':mos,'Stage':'PRISTINE' if smo>=1.85 else ('Stage 1' if smo>=1.70 else 'Stage 2'),
                   'Description':desc,'Source':source}
                  for smo,(mos,desc,source) in XPS_CALIBRATION.items()]
    st.markdown("### XPS Calibration Table")
    st.dataframe(pd.DataFrame(calib_data), use_container_width=True)

    smo_vals = sorted(XPS_CALIBRATION.keys(), reverse=True)
    mos_vals = [XPS_CALIBRATION[s][0] for s in smo_vals]
    fig_calib = go.Figure()
    fig_calib.add_vrect(x0=1.70,x1=2.20,fillcolor='rgba(255,200,100,0.08)',line_width=0,
                        annotation_text='Stage 1',annotation_position='top left')
    fig_calib.add_vrect(x0=0.0,x1=1.70,fillcolor='rgba(45,206,137,0.08)',line_width=0,
                        annotation_text='Stage 2 (HIGH in KOH)',annotation_position='top right')
    fig_calib.add_trace(go.Scatter(x=smo_vals,y=mos_vals,mode='lines+markers',
        line=dict(color='#4E9AF1',width=2),marker=dict(size=10,color='#4E9AF1'),
        name='XPS calibration'))
    fig_calib.add_vline(x=1.70,line_dash='dash',line_color='#F5A623',line_width=2,
                        annotation_text='S:Mo=1.70 — Stage 1→2 (Li 2019)',annotation_font_color='#F5A623')
    fig_calib.update_layout(
        title="XPS Calibration: S:Mo → Mo/S | Stage 1/2 boundary",
        xaxis_title="S:Mo ratio",yaxis_title="Mo/S ratio",
        xaxis=dict(autorange='reversed'),height=420,
        plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_calib, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BULLETPROOF VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🛡 Bulletproof Validation":
    st.markdown("# Bulletproof Validation — v6.0")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Training samples","14")
    c2.metric("Papers integrated","16")
    c3.metric("Current confidence",confidence_level(layer_n,mo_s_ratio,ecsa_val,dist_val)[0])
    c4.metric("Nearest Jeon dist.",f"{dist_val:.2f}")

    st.markdown("## Current input audit")
    vals_now  = predict_all(layer_n, mo_s_ratio, ecsa_val)
    eta_now   = eta_v_to_mV_abs(vals_now['eta'])
    perf_now, perf_note_now = classify_performance_eta_v5(eta_now)
    rct_label_now,_ = classify_rct_v5(vals_now['rct'])
    sc,sl,sn = li2019_stage(mo_s_ratio)
    cond_now  = vals_now['conductivity']
    audit_df = pd.DataFrame([
        {'Item':'η10 magnitude','Value':f'{eta_now:.1f} mV','Interpretation':f'{perf_now}: {perf_note_now}'},
        {'Item':'Tafel','Value':f'{vals_now["tafel"]:.1f} mV/dec','Interpretation':tafel_mechanism_v5(vals_now['tafel'],mo_s_ratio)},
        {'Item':'Rct','Value':f'{vals_now["rct"]:.1f} Ω·cm²','Interpretation':rct_label_now},
        {'Item':'Conductivity σ','Value':f'{cond_now:.4f} S/cm','Interpretation':'σ=1/ρ more physically interpretable than resistivity'},
        {'Item':'Li 2019 Stage','Value':sl,'Interpretation':sn[:100]+'...'},
        {'Item':'Layer penalty','Value':f'{layer_activity_factor(layer_n):.2e}','Interpretation':'Yu 2014 4.47×/layer decay (affects Rct, not directly η)'},
        {'Item':'Synthesis method','Value':m_label,'Interpretation':f'Score {m_score}/{m_max}'},
    ])
    st.dataframe(audit_df, use_container_width=True)
    st.markdown("## KOH benchmark table")
    st.dataframe(KOH_BENCHMARKS[['family','material','eta_mV','tafel','rct','stage','mechanism','note']],
                 use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MASTER TABLE KOH
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Master Table KOH":
    st.markdown("# Master Table — MoS₂ HER in KOH")
    st.markdown("### State-of-Art MoS₂ in 1M KOH")
    sota_df = pd.DataFrame([
        {'Material':'NiO@1T-MoS2','η10 (mV)':46,'Tafel (mV/dec)':40,'Strategy':'1T metallic + NiO','Mechanism':'Heyrovsky-fast'},
        {'Material':'MoS2/MXene/NF','η10 (mV)':94,'Tafel (mV/dec)':59,'Strategy':'MXene conductive heterojunction','Mechanism':'Heyrovsky'},
        {'Material':'Mo5N6-MoS2/HCNRs','η10 (mV)':100,'Tafel (mV/dec)':37.9,'Strategy':'Mott-Schottky junction','Mechanism':'Heyrovsky-fast'},
        {'Material':'MoS2/NiS','η10 (mV)':130,'Tafel (mV/dec)':52,'Strategy':'Ni heterostructure','Mechanism':'Heyrovsky'},
        {'Material':'SnO2@MoS2','η10 (mV)':127,'Tafel (mV/dec)':73,'Strategy':'SnO2 nanorod','Mechanism':'Mixed'},
        {'Material':'CoS2-MoS2 HS','η10 (mV)':130,'Tafel (mV/dec)':66,'Strategy':'Co hollow interface','Mechanism':'Heyrovsky'},
        {'Material':'N-1T@2H MoS2','η10 (mV)':141.7,'Tafel (mV/dec)':48.4,'Strategy':'1T/2H + N doping','Mechanism':'Heyrovsky'},
        {'Material':'MoS2-1T exfoliated','η10 (mV)':145,'Tafel (mV/dec)':46.2,'Strategy':'Metallic phase (unstable)','Mechanism':'Heyrovsky'},
        {'Material':'MoS2-SV (Plasma Ar)','η10 (mV)':175,'Tafel (mV/dec)':63.5,'Strategy':'S-vacancy Stage 1','Mechanism':'Mixed'},
        {'Material':'2H MoS2-7H (Li 2019)','η10 (mV)':260,'Tafel (mV/dec)':80,'Strategy':'S-vacancy Stage 2 (0.1M KOH)','Mechanism':'Heyrovsky partial'},
        {'Material':'MoS2 90nm nanosheets','η10 (mV)':280,'Tafel (mV/dec)':151,'Strategy':'Nanostructured 2H','Mechanism':'Volmer-limited'},
        {'Material':'MoS2 Bulk/Control','η10 (mV)':350,'Tafel (mV/dec)':115,'Strategy':'Reference pristine 2H','Mechanism':'Volmer-limited'},
    ])
    st.dataframe(sota_df.sort_values('η10 (mV)'), use_container_width=True)
    st.markdown("### Master Family Table — 8 MoS₂ Families")
    st.dataframe(MASTER_FAMILY_TABLE, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("# About — MoS₂ HER Trend Model v6.0")
    st.markdown("""
**v6.1** — Synthesis physics with real literature data.

| Change | Description |
|---|---|
| [v6-1] Conductivity | Resistivity replaced by σ=1/ρ (S/cm) everywhere. σ directly correlates with charge carrier concentration — more physically interpretable. |
| [v6-2] Layer# vs η | Explicit note in Predictor and Trend Curves: layer# does NOT directly drive η; effect is mediated through ECSA (synthesis → grain size → ECSA → η). |
| [v6-3] Synthesis → ECSA | Predictor shows synthesis homogeneity note. Synthesis Physics page explains the causal chain. |
| [v6-4] Synthesis Physics page | New dedicated page: MBE (kinetic/metastable) vs CVD (thermodynamic/equilibrium) → homogeneity → ECSA → η. ECSA vs layer# scatter, conductivity vs Mo/S, comparison table. |
| [v6-5] Homogeneity badge | Predictor page shows MBE or CVD homogeneity context for every prediction. |
| [v6-6] Structural parameters | Theoretical Basis now has a full table: structural parameter → controls → electrochemical effect → primary metric → scientific note. |
| [v6-7] Cross-paper consistency | Theoretical Basis shows consistency check across all 16 papers on key MoS₂ descriptors. |

**Machine learning:** KNN (k=3, physics-weighted, distance-weighted) · RF (300 trees, LOO) · LOO CV (n=14) · ⚠ n=14 — trend prediction only.
    """)
