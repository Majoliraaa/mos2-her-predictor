"""
MoS₂ HER Trend Model — v5.1
=============================
BASE: v5.0 (funcionaba correctamente)
ADDED (profesor feedback only):
  [P-1] Resistivity → Conductivity σ = 1/ρ (S/cm)
  [P-2] Nueva página "🔬 Synthesis Physics" — MBE vs CVD con literatura real
        - Cadena causal DINÁMICA (reacciona a sliders)
        - Jeon 2026: ECSA 6.7→3.5 cm² al subir T → η empeora
        - Ma et al. ACS Nano 2017: MBE genera twin GBs por S-deficiency
        - Nature Comms 2020: GB density ~10¹² cm⁻² → onset −25 mV
        - ACS AMI: CVD GB density ~0.04 µm⁻¹
  [P-3] Theoretical Basis: tabla parámetros estructurales → propiedades electroquímicas
  [P-4] Cross-paper consistency check (20 papers)
  [P-5] Todo texto negro #111 (legibilidad)
  [P-6] Sin referencias a "CEO"
  [P-7] Título v5.1
SIN TOCAR: modelo GP+RF, métricas, clasificación, páginas existentes
"""

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
    padding: 10px 14px; margin: 8px 0; font-size: 0.82em; color: #111; font-weight: 500;
}
.correction-box {
    background: rgba(78,154,241,0.15); border: 1px solid rgba(78,154,241,0.40);
    border-left: 4px solid #4E9AF1; border-radius: 4px;
    padding: 10px 14px; margin: 8px 0; font-size: 0.82em; color: #111; font-weight: 500;
}
.fix-box {
    background: rgba(255,100,100,0.15); border: 1px solid rgba(255,100,100,0.40);
    border-left: 4px solid #FF6464; border-radius: 4px;
    padding: 10px 14px; margin: 8px 0; font-size: 0.82em; color: #111; font-weight: 500;
}
.stMetric label { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78em !important; }
.stMetric [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }
.bulletproof-box {
    background: rgba(45,206,137,0.15); border: 1px solid rgba(45,206,137,0.40);
    border-left: 4px solid #2DCE89; border-radius: 4px; padding: 12px 14px;
    margin: 10px 0; font-size: 0.86em; color: #111; font-weight: 500;
}
.risk-box {
    background: rgba(245,166,35,0.15); border: 1px solid rgba(245,166,35,0.40);
    border-left: 4px solid #F5A623; border-radius: 4px; padding: 12px 14px;
    margin: 10px 0; font-size: 0.86em; color: #111; font-weight: 500;
}
.stage2-box {
    background: rgba(45,206,137,0.18); border: 1px solid rgba(45,206,137,0.50);
    border-left: 4px solid #2DCE89; border-radius: 4px; padding: 12px 14px;
    margin: 10px 0; font-size: 0.86em; color: #111; font-weight: 500;
}
.synth-mbe {
    background: rgba(245,166,35,0.15); border: 1px solid rgba(245,166,35,0.45);
    border-left: 4px solid #F5A623; border-radius: 4px; padding: 12px 14px;
    margin: 6px 0; font-size: 0.84em; color: #111; font-weight: 500;
}
.synth-cvd {
    background: rgba(78,154,241,0.15); border: 1px solid rgba(78,154,241,0.45);
    border-left: 4px solid #4E9AF1; border-radius: 4px; padding: 12px 14px;
    margin: 6px 0; font-size: 0.84em; color: #111; font-weight: 500;
}
.validation-chip {
    display: inline-block; border-radius: 999px; padding: 3px 10px; margin: 2px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72em;
    background: rgba(45,206,137,0.12); border: 1px solid rgba(45,206,137,0.35); color: #2DCE89;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# XPS CALIBRATION TABLE
# ══════════════════════════════════════════════════════════════════════════════
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

K0_VS_LAYERS = {1:250.0, 2:7.5, 3:1.5, 5:0.1, 10:0.01, 20:0.001}

# ══════════════════════════════════════════════════════════════════════════════
# DATA — Jeon et al. ACS Nano 2026, Table 1
# ══════════════════════════════════════════════════════════════════════════════
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
        'mo_s_ratio':  [0.49,0.48,0.46,0.57,0.56,0.52,0.50,0.47,0.82,0.72,0.65,0.52,0.48,0.46],
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
    df = pd.DataFrame(data)
    # [P-1] Add conductivity column
    df['conductivity'] = 1.0 / df['resistivity']
    return df

df = load_data()

# [P-1] Add conductivity to TARGETS, keep resistivity for backward compat
TARGETS = {
    'eta':          ('Overpotential η',  'V',           'min'),
    'tafel':        ('Tafel slope',       'mV/dec',      'min'),
    'rct':          ('Rct',               'Ω·cm²',       'min'),
    'raman':        ('Raman A₁g/E₂g',    '',            'min'),
    'conductivity': ('Conductivity σ',    'S/cm',        'max'),
    'tof_ecsa':     ('TOF (ECSA)',         'nmol/cm²/s',  'max'),
    'tof_mass':     ('TOF (mass)',         'nmol/µg/s',   'max'),
}

FEATURES = ['layer_n', 'mo_s_ratio', 'ecsa']
FEATURE_LABELS = {
    'layer_n':    'Layer # (validated)',
    'mo_s_ratio': 'Mo/S ratio (validated)',
    'ecsa':       'ECSA (cm²)',
}
FEATURE_RANGES = {
    'layer_n':    (1, 20),
    'mo_s_ratio': (0.45, 0.90),
    'ecsa':       (2.0, 12.0),
}
FEATURE_PROVENANCE = {
    'layer_n':    '✅ Validated — XRD Scherrer ÷ 0.615 nm/layer (×6 sources). Raman confirms N5→2L, N10→4-5L (Lee 2010).',
    'mo_s_ratio': '✅ Validated — XPS calibration (Sherwood 2024 + ACS Cat 2023 + Smiri 2026). Stage 1/2 threshold: S:Mo=1.70 (Li 2019).',
    'ecsa':       '✅ Directly measured — Jeon 2026 Table 1 (Cdl method, Cs=40 µF/cm²)',
}

SERIES_COLORS = {'T': '#4E9AF1', 'N': '#2DCE89', 'M': '#F5A623'}
SERIES_LABELS = {'T': 'T-series (Temp.)', 'N': 'N-series (Cycles)', 'M': 'M-series (S-thick.)'}
METHOD_COLORS = {'mbe': '#2DCE89', 'both': '#F5A623', 'cvd': '#4E9AF1'}

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK TABLES (unchanged from v5.0)
# ══════════════════════════════════════════════════════════════════════════════
KOH_BENCHMARKS = pd.DataFrame([
    {'family':'Pristine 2H bulk','material':'MoS2 bulk (CVD/hydrothermal)','eta_mV':350,'tafel':115,'rct':200,'s_mo':2.00,'electrolyte':'1M KOH','stage':'Stoichiometric','mechanism':'Volmer-limited','note':'Basal plane almost inert'},
    {'family':'Pristine 2H bulk','material':'MoS2 90nm nanosheets','eta_mV':280,'tafel':151,'rct':18.1,'s_mo':2.00,'electrolyte':'1M KOH','stage':'Stoichiometric','mechanism':'Volmer-limited','note':'More edges, lower Rct'},
    {'family':'MoS2-SV (Stage 1)','material':'MoS2-SV (Plasma Ar)','eta_mV':175,'tafel':63.5,'rct':None,'s_mo':1.82,'electrolyte':'1M KOH','stage':'Stage 1 (point defects)','mechanism':'Mixed Volmer-Heyrovsky','note':'S:Mo=1.82 > 1.7 threshold'},
    {'family':'MoS2-SV (Stage 2)','material':'2H MoS2-7H (Li 2019 KOH)','eta_mV':260,'tafel':80,'rct':None,'s_mo':0.50,'electrolyte':'0.1M KOH','stage':'Stage 2 (undercoord. Mo)','mechanism':'Heyrovsky improved','note':'TOF=15 s⁻¹ @ 300mV in KOH'},
    {'family':'1T phase','material':'MoS2-1T exfoliated','eta_mV':145,'tafel':46.2,'rct':None,'s_mo':1.82,'electrolyte':'1M KOH','stage':'Metallic 1T','mechanism':'Heyrovsky','note':'Best conductivity, unstable'},
    {'family':'Heterostructure','material':'MoS2/Ni3S2','eta_mV':128,'tafel':52.4,'rct':10.0,'s_mo':1.88,'electrolyte':'1M KOH','stage':'Stage 1+interface','mechanism':'Heyrovsky','note':'Ni facilitates water dissociation'},
    {'family':'Heterostructure','material':'MoS2/NiS (HHs B)','eta_mV':130,'tafel':52.0,'rct':10.0,'s_mo':None,'electrolyte':'1M KOH','stage':'Heterostructure','mechanism':'Heyrovsky','note':'UCL: Rct<10Ω confirmed'},
    {'family':'MXene composite','material':'MoS2/MXene/NF','eta_mV':94,'tafel':59,'rct':None,'s_mo':None,'electrolyte':'1M KOH','stage':'Conductive heterojunction','mechanism':'Heyrovsky','note':'MXene reduces Rct drastically'},
    {'family':'Advanced (Mott-Schottky)','material':'Mo5N6-MoS2/HCNRs','eta_mV':100,'tafel':37.9,'rct':6.5,'s_mo':None,'electrolyte':'1M KOH','stage':'Mott-Schottky junction','mechanism':'Heyrovsky-fast','note':'Rct~5-8Ω (CityU); near Pt-like'},
    {'family':'State-of-art','material':'NiO@1T-MoS2','eta_mV':46,'tafel':40,'rct':None,'s_mo':None,'electrolyte':'1M KOH','stage':'Metallic 1T + NiO','mechanism':'Heyrovsky-fast','note':'Best reported: 1T + Ni synergy'},
    {'family':'State-of-art','material':'N-1T@2H MoS2','eta_mV':141.7,'tafel':48.4,'rct':None,'s_mo':None,'electrolyte':'1M KOH','stage':'1T/2H mixed + N-dope','mechanism':'Heyrovsky','note':'Phase control + doping'},
    {'family':'State-of-art','material':'SnO2@MoS2','eta_mV':127,'tafel':73,'rct':None,'s_mo':None,'electrolyte':'1M KOH','stage':'Nanorod heterostructure','mechanism':'Mixed','note':'SnO2 improves water dissociation'},
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

EXPERIMENTAL_SD_TABLE = pd.DataFrame([
    {'family':'State-of-art (η<80mV)','eta_sd_mV':3.5,'tafel_sd':1.5,'condition':'η10 < 80 mV','note':'Highest engineering → lowest SD'},
    {'family':'High performance (η80-140mV)','eta_sd_mV':5.3,'tafel_sd':1.9,'condition':'80 ≤ η10 < 140 mV','note':'Inverse correlation: better → more reproducible'},
    {'family':'Engineered MoS2 (η140-175mV)','eta_sd_mV':7.1,'tafel_sd':2.8,'condition':'140 ≤ η10 < 175 mV','note':''},
    {'family':'Vacancy/defect engineered (η175-250mV)','eta_sd_mV':9.4,'tafel_sd':4.2,'condition':'175 ≤ η10 < 250 mV','note':'Tafel more robust than η'},
    {'family':'Bulk/pristine (η>250mV)','eta_sd_mV':22.0,'tafel_sd':8.5,'condition':'η10 ≥ 250 mV','note':'High SD: low site density → sensitive to surface variation'},
])

# ══════════════════════════════════════════════════════════════════════════════
# GP + RF MODELS (unchanged from v5.0)
# ══════════════════════════════════════════════════════════════════════════════
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
        sy = StandardScaler().fit(y.reshape(-1, 1))
        kernel = (C(1.0,(1e-3,1e3))*Matern(length_scale=[1.0]*n,length_scale_bounds=[(0.01,100)]*n,nu=2.5)
                  + WhiteKernel(noise_level=0.1,noise_level_bounds=(1e-5,10)))
        loo_means, loo_stds_list = [], []
        for tr, te in loo.split(X):
            sx_l = StandardScaler().fit(X[tr])
            sy_l = StandardScaler().fit(y[tr].reshape(-1,1))
            gp_l = GaussianProcessRegressor(
                kernel=C(1.0,(1e-3,1e3))*Matern(length_scale=[1.0]*n,length_scale_bounds=[(0.01,100)]*n,nu=2.5)
                       +WhiteKernel(0.1,(1e-5,10)),
                n_restarts_optimizer=5,normalize_y=False,alpha=1e-6)
            gp_l.fit(sx_l.transform(X[tr]),sy_l.transform(y[tr].reshape(-1,1)).ravel())
            m_s,std_s = gp_l.predict(sx_l.transform(X[te]),return_std=True)
            loo_means.append(sy_l.inverse_transform(m_s.reshape(-1,1)).ravel()[0])
            loo_stds_list.append(std_s[0]*sy_l.scale_[0])
        loo_means = np.array(loo_means)
        avg_err = np.mean(np.abs(y-loo_means))
        avg_std = np.mean(loo_stds_list)
        calib   = avg_err/avg_std if avg_std>0 else 1.0
        gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,normalize_y=False,alpha=1e-6)
        gp.fit(sx.transform(X),sy.transform(y.reshape(-1,1)).ravel())
        gp_models[key]     = gp
        gp_scores[key]     = {'r2':r2_score(y,loo_means),'mae':mean_absolute_error(y,loo_means),'loo_preds':loo_means,'calib':calib}
        sx_dict[key]       = sx
        sy_dict[key]       = sy
        loo_stds_dict[key] = np.array(loo_stds_list)
        rf = RandomForestRegressor(n_estimators=300,max_depth=4,min_samples_leaf=2,random_state=42)
        preds = np.zeros(len(y))
        for tr,te in loo.split(X):
            rf.fit(X[tr],y[tr]); preds[te]=rf.predict(X[te])
        rf.fit(X,y)
        rf_models[key] = rf
        rf_scores[key] = {'r2':r2_score(y,preds),'mae':mean_absolute_error(y,preds),'loo_preds':preds}
        rf_imps[key]   = rf.feature_importances_
    return (gp_models,gp_scores,sx_dict,sy_dict,loo_stds_dict,rf_models,rf_scores,rf_imps)

with st.spinner("Training GP + RF models… (first load only)"):
    (gp_models,gp_scores,sx_dict,sy_dict,loo_stds_dict,rf_models,rf_scores,rf_imps) = train_models()

def gp_predict(key,ln,msr,ecsa_v):
    X_new = np.array([[ln,msr,ecsa_v]])
    sx=sx_dict[key]; sy=sy_dict[key]; gp=gp_models[key]
    m_s,std_s = gp.predict(sx.transform(X_new),return_std=True)
    mean = sy.inverse_transform(m_s.reshape(-1,1)).ravel()[0]
    std  = std_s[0]*sy.scale_[0]*gp_scores[key]['calib']
    return mean,mean-1.96*std,mean+1.96*std,std

def predict_all(ln,msr,ecsa_v):
    return {k:gp_predict(k,ln,msr,ecsa_v)[0] for k in TARGETS}

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS HELPERS (unchanged from v5.0)
# ══════════════════════════════════════════════════════════════════════════════
def eta_v_to_mV_abs(eta_v): return abs(float(eta_v))*1000.0
def layer_activity_factor(layer_n): return (1.0/4.47)**max(float(layer_n)-1.0,0.0)
def vacancy_percent_from_mo_s(mo_s_ratio):
    if mo_s_ratio<=0: return np.nan
    s_mo=1.0/float(mo_s_ratio)
    return float(min(max(0.0,(2.0-s_mo)/2.0*100.0),90.0))

def li2019_stage(mo_s_ratio):
    s_mo=1.0/float(mo_s_ratio) if mo_s_ratio>0 else 2.0
    if s_mo>1.85: return 'PRISTINE','Pre-Stage 1','Near-stoichiometric 2H MoS₂. Basal plane inert. η > 300 mV typical.'
    elif s_mo>1.70: return 'STAGE_1','Stage 1 (point defects)',(f'S:Mo={s_mo:.2f} > 1.70 threshold. Point defects activating. Tafel decreasing rapidly in KOH (Li 2019). η≈200-280 mV expected. Rct decreasing rapidly.')
    elif s_mo>1.33: return 'STAGE_2_MILD','Stage 2 (mild undercoord. Mo)',(f'S:Mo={s_mo:.2f} < 1.70 threshold — undercoordinated Mo regions forming. IN KOH: TOF continuously increases (unlike H₂SO₄). This is a HIGH activity regime, not a risk regime (Li 2019 Fig.4f). η≈150-260 mV. Jeon MoS-M3.0 (Mo/S=0.65, S:Mo=1.54) falls here: η=-0.35V ✓')
    elif s_mo>0.80: return 'STAGE_2_DEEP','Stage 2 (deep undercoord. Mo)',(f'S:Mo={s_mo:.2f} — extensive S stripping. Very high TOF in KOH. Structure of multilayer maintained (Li 2019: inner layers stable). Extreme case: MoS2-7H in 0.1M KOH achieves TOF=15 s⁻¹ @ 300mV. Structural risk begins at Mo/S > 0.75 (Mo-rich domains possible).')
    else: return 'STAGE_2_EXTREME','Stage 2 extreme (structural risk)',(f'S:Mo={s_mo:.2f} — extreme S-stripping. Mo-rich domains likely. Jeon MoS-M2.0 (Mo/S=0.82): η=-0.58V confirms over-vacancy collapse. Structural integrity compromised.')

def vacancy_regime(vacancy_pct,mo_s_ratio=None):
    if np.isnan(vacancy_pct): return "Unknown","UNKNOWN","Insufficient Mo/S information."
    if mo_s_ratio is not None: stage_code,stage_label,stage_note=li2019_stage(mo_s_ratio)
    else: stage_code='UNKNOWN'; stage_label='Unknown'; stage_note=''
    s_mo=1.0/mo_s_ratio if (mo_s_ratio and mo_s_ratio>0) else 2.0
    if stage_code=='PRISTINE': return ("Near-stoichiometric 2H MoS₂","LOW",f"Vacancy≈{vacancy_pct:.1f}% (S:Mo={s_mo:.2f} > 1.85): Pre-Stage 1. Volmer-dominated. Basal plane mostly inert. η > 300 mV, Tafel > 100 mV/dec typical in KOH.")
    elif stage_code=='STAGE_1': return ("Stage 1 — Point defect activation (Li 2019)","MEDIUM",f"Vacancy≈{vacancy_pct:.1f}% (S:Mo={s_mo:.2f}, 1.70–1.85): Point defects rapidly improving HER. Tafel decreases fast in KOH (Li 2019 Fig.4c). ΔG_H* improving toward 0 eV. η≈200-280 mV expected.")
    elif stage_code=='STAGE_2_MILD': return ("Stage 2 — Undercoordinated Mo, HIGH activity (Li 2019)","HIGH",f"Vacancy≈{vacancy_pct:.1f}% (S:Mo={s_mo:.2f}, 1.33–1.70): Undercoordinated Mo regions formed. IN KOH: TOF CONTINUES TO INCREASE (unlike H₂SO₄ — Li 2019 Fig.4f). η≈150-260 mV. Tafel≈80 mV/dec. MoS-M3.0 (Mo/S=0.65): η=-0.35V ✓")
    elif stage_code=='STAGE_2_DEEP': return ("Stage 2 deep — Extensive S-stripping, active but watch structural integrity","HIGH-RISK",f"Vacancy≈{vacancy_pct:.1f}% (S:Mo={s_mo:.2f}, 0.80–1.33): Extensive S-stripping. MoS2-7H (Li 2019): TOF=15 s⁻¹ @ 300mV in KOH. Monitor for Mo-rich domain formation.")
    else: return ("Stage 2 extreme — Structural collapse risk","RISK",f"Vacancy≈{vacancy_pct:.1f}% (S:Mo={s_mo:.2f} < 0.80): Extreme Mo-rich domains. Jeon MoS-M2.0 (Mo/S=0.82): η=-0.58V confirms structural collapse.")

def tafel_mechanism_v5(tafel,mo_s_ratio=None,layer_n=None):
    tafel=float(tafel); s_mo=(1.0/mo_s_ratio) if mo_s_ratio else None
    if tafel<=45:    fam="State-of-art (NiO@1T MoS2: 40, Mo5N6: 37.9 mV/dec)"; mech="Heyrovsky-fast / near-Pt kinetics"
    elif tafel<=60:  fam="High performance (1T, MoS2/Ni, MoS2/MXene range)"; mech="Heyrovsky dominant (electrochemical desorption)"
    elif tafel<=80:  fam="Stage 2 regime (Li 2019 KOH: ~80 mV/dec at high vacancy)"; mech="Mixed Volmer-Heyrovsky, Stage 2 improving"
    elif tafel<=100: fam="Stage 1-2 transition"; mech="Mixed regime, Volmer partially limiting"
    else:            fam="Stage 1 / Pristine (bulk-like)"; mech="Volmer-limited (slow H₂O dissociation in KOH)"
    note=""
    if s_mo and s_mo<1.70: note=" [Stage 2: TOF still increasing in KOH per Li 2019]"
    elif s_mo and s_mo<1.85: note=" [Stage 1: Tafel decreasing rapidly with more vacancies]"
    return f"{mech} ({fam}){note}"

def classify_performance_eta_v5(eta_mV):
    if eta_mV<80:  return "EXCELLENT","Comparable to NiO@1T-MoS2 (46mV) / Mo5N6 (100mV) tier — state-of-art."
    if eta_mV<130: return "HIGH","MoS2/NiS (130mV) / MoS2/MXene (94mV) tier — very strong alkaline HER."
    if eta_mV<180: return "GOOD","1T MoS2 / MoS2-SV / Stage 2 regime — good engineered MoS2 performance."
    if eta_mV<280: return "MODERATE","Stage 1 regime / MoS2 nanoflakes — improved over bulk, kinetics limited."
    if eta_mV<400: return "GOOD (Best MBE on Si)","Best performance range for MBE-grown MoS2 on Si substrate (Jeon 2026). Optimum sample MoS-N10 achieves 330mV. Further improvement requires heterostructure or phase engineering."
    return "MODERATE (MBE)","Within Jeon 2026 MBE dataset range (330-580mV). S-vacancy or ECSA engineering recommended."

def classify_rct_v5(rct,electrolyte='1M KOH'):
    if rct<10:  return "EXCELLENT Rct","Mo5N6/MoS2 tier (<10Ω); near-ideal charge transfer."
    if rct<20:  return "LOW Rct","MoS2/NiS tier (<20Ω); efficient interfacial charge transfer."
    if rct<80:  return "MODERATE Rct","Stage 1-2 regime; some charge-transfer limitation."
    if rct<150: return "HIGH Rct","Stage 1 / bulk-like; significant charge-transfer barrier."
    return "VERY HIGH Rct","Bulk MoS2 regime (>200Ω = >200Ω·cm²); poor coupling."

def literature_experimental_sd_v5(eta_mV,target='eta'):
    if target=='tafel':
        if eta_mV<80: return 1.5
        if eta_mV<140: return 1.9
        if eta_mV<175: return 2.8
        if eta_mV<250: return 4.2
        return 8.5
    if eta_mV<80: return 3.5
    if eta_mV<140: return 5.3
    if eta_mV<175: return 7.1
    if eta_mV<250: return 9.4
    return 22.0

def distance_penalty(dist_val,target='eta'):
    if dist_val<0.15: return 0.0
    if dist_val<0.40: return 12.0 if target=='eta' else 4.0
    return 35.0 if target=='eta' else 12.0

def total_uncertainty_for_metric(key,mean_value,gp_std,dist_val,eta_mV_ref=None):
    if key=='eta':
        eta_mV=eta_v_to_mV_abs(mean_value); gp_mV=abs(gp_std)*1000.0
        exp_sd=literature_experimental_sd_v5(eta_mV,'eta'); pen=distance_penalty(dist_val,'eta')
        return np.sqrt(gp_mV**2+exp_sd**2+pen**2)/1000.0
    if key=='tafel':
        eta_ref=eta_mV_ref if eta_mV_ref else 200
        exp_sd=literature_experimental_sd_v5(eta_ref,'tafel'); pen=distance_penalty(dist_val,'tafel')
        return np.sqrt(float(gp_std)**2+exp_sd**2+pen**2)
    return float(gp_std)

def confidence_level(layer_n,mo_s_ratio,ecsa_v,dist_val):
    warnings_list=[]
    if dist_val<0.15: confidence="HIGH"; warnings_list.append("Input is close to an experimental Jeon sample.")
    elif dist_val<0.40: confidence="MEDIUM"; warnings_list.append("Input interpolated inside/near the Jeon experimental domain.")
    else: confidence="LOW"; warnings_list.append("Input extrapolated beyond validated Jeon domain; use as hypothesis.")
    if layer_n>10: warnings_list.append("High layer number: strong electron-transfer penalty (Yu 2014: 4.47×/layer).")
    if mo_s_ratio>0.75: warnings_list.append("Stage 2 deep / extreme: structural risk starts. Monitor Mo(UC) by XPS.")
    if ecsa_v<df['ecsa'].min() or ecsa_v>df['ecsa'].max(): warnings_list.append("ECSA outside Jeon measured range; uncertainty increased.")
    return confidence,warnings_list

def literature_consistency_score_v5(eta_mV,tafel,rct,mo_s_ratio,ecsa_v):
    score=0; notes=[]; s_mo=1.0/mo_s_ratio if mo_s_ratio>0 else 2.0
    if eta_mV<130: score+=1; notes.append(f"η10={eta_mV:.0f}mV in high-performance KOH tier (<130mV).")
    elif eta_mV<180: score+=0.5; notes.append(f"η10={eta_mV:.0f}mV in good Stage 2 regime (130-180mV).")
    if tafel<=60: score+=1; notes.append("Tafel ≤60 mV/dec: Heyrovsky-dominant (high performance tier).")
    elif tafel<=85: score+=0.5; notes.append("Tafel 60-85 mV/dec: Stage 2 regime (Li 2019 KOH reference).")
    if rct<20: score+=1; notes.append("Rct <20Ω: MoS2/NiS performance tier.")
    elif rct<80: score+=0.5; notes.append("Rct 20-80Ω: Stage 2 Jeon regime.")
    if s_mo<1.70: score+=1; notes.append(f"S:Mo={s_mo:.2f} < 1.70: Stage 2 regime — undercoordinated Mo active.")
    elif s_mo<1.85: score+=0.5; notes.append(f"S:Mo={s_mo:.2f}: Stage 1 defect activation.")
    if ecsa_v>=7.0: score+=1; notes.append("ECSA ≥7.0 cm²: high relative to Jeon dataset.")
    return min(score,5),notes

def score_method(layer_n,mo_s_ratio,ecsa_v,rct_v=None):
    reasons=[]; total=0; MAX=8
    if layer_n<=3:   pts=3; detail="≤3 layers: k⁰ 250→1.5 cm/s (McKelvey). MBE required."; refs=["McKelvey 2021","Yu 2014 4.47×/layer","Manyepedza 2022"]
    elif layer_n<=6: pts=2; detail="4–6 layers: k⁰≈0.1–7.5 cm/s. Optimal HER zone. Jeon N10 (~5L) is N-series optimum."; refs=["Jeon 2026 N-series","Lee 2010 ACS Nano"]
    elif layer_n<=12:pts=1; detail="7–12 layers: k⁰≈0.01–0.1 cm/s. MBE preferred for uniformity."; refs=["Jeon 2026 T-series"]
    else:            pts=0; detail="≥13 layers: k⁰<0.01 cm/s — bulk-like kinetics."; refs=["Jeon 2026 T800/N50"]
    total+=pts; reasons.append({'criterion':'Layer #','points':pts,'max':3,'refs':refs,'detail':detail})
    s_mo=1.0/mo_s_ratio if mo_s_ratio>0 else 2.0
    if s_mo<1.33:   pts=3; detail=f"S:Mo={s_mo:.2f} < 1.33 (Stage 2 deep): MBE S-flux control mandatory."; refs=["Li 2019 ACS Nano (Stage 2)","Sherwood 2024"]
    elif s_mo<1.70: pts=2; detail=f"S:Mo={s_mo:.2f} < 1.70 (Stage 2 mild): Li 2019 confirms HIGH activity in KOH. MBE preferred."; refs=["Li 2019 Fig.4f KOH","ACS Cat 2023 threshold"]
    elif s_mo<1.85: pts=1; detail=f"S:Mo={s_mo:.2f} (Stage 1): MBE offers better S-flux control than CVD."; refs=["ACS Cat 2023","Sherwood 2024"]
    else:           pts=0; detail=f"S:Mo={s_mo:.2f} ≥ 1.85: Near-stoichiometric 2H-MoS₂. CVD sufficient."; refs=["ACS Cat 2023 Mo-8 to Mo-16"]
    total+=pts; reasons.append({'criterion':'Mo/S ratio (Stage 1/2)','points':pts,'max':3,'refs':refs,'detail':detail})
    if ecsa_v>=8.0: pts=1; detail="ECSA ≥8.0 cm²: Wafer-scale uniformity needed. Jeon M6.0 (9.2cm²) and N10 (8.0cm²)."; refs=["Jeon 2026 Table 1"]
    else:           pts=0; detail="ECSA <8.0 cm²: No additional MBE constraint from this criterion."; refs=[]
    total+=pts; reasons.append({'criterion':'ECSA','points':pts,'max':1,'refs':refs,'detail':detail})
    rct_use=rct_v if rct_v is not None else gp_predict('rct',layer_n,mo_s_ratio,ecsa_v)[0]
    if rct_use<55: pts=1; detail=f"Rct={rct_use:.0f} Ω·cm² < 55: Low Rct requires S-vacancy domains in 2H matrix."; refs=["Jeon 2026 EIS","UCL discovery (MoS2/NiS)","CityU (Mo5N6)"]
    else:          pts=0; detail=f"Rct={rct_use:.0f} Ω·cm² ≥ 55: No additional MBE constraint."; refs=[]
    total+=pts; reasons.append({'criterion':'Rct','points':pts,'max':1,'refs':refs,'detail':detail})
    if total>=3:   label="🔬 Physical Method (MBE)"; col_key='mbe'
    elif total>=1: label="⚗️ Both viable — MBE preferred"; col_key='both'
    else:          label="🧪 Chemical Method (CVD/PVT)"; col_key='cvd'
    return label,col_key,total,MAX,reasons

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚗️ MoS₂ HER Trend Model")
    st.markdown(
        "<div style='font-size:0.78em;color:#111;margin-bottom:10px;'>"
        "Jeon et al. <i>ACS Nano</i> 2026 · v5.1 · 20 papers<br>"
        "GP model · n=14 MBE samples · 1M KOH</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div class='provenance-box'>"
        "✅ <b>ECSA</b>: measured (Jeon 2026)<br>"
        "✅ <b>Layer #</b>: Scherrer ÷ 0.615nm (×6 sources)<br>"
        "✅ <b>Mo/S</b>: XPS calibration (×4 sources)<br>"
        "✅ <b>4.47×/layer</b>: Yu 2014 (V₀=0.119V)<br>"
        "✅ <b>Stage 1/2 threshold</b>: S:Mo=1.70 (Li 2019 KOH PRIMARY)<br>"
        "✅ <b>v5.1</b>: Conductivity σ · Synthesis Physics · Structural parameters"
        "</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">KEY DESCRIPTORS</div>', unsafe_allow_html=True)
    layer_n    = st.slider("✅ Layer #", 1, 20, 5, 1)
    mo_s_ratio = st.slider("✅ Mo/S atomic ratio", 0.45, 0.90, 0.56, 0.01)
    ecsa_val   = st.slider("✅ ECSA (cm²)", 2.0, 12.0, 8.0, 0.5)

    s_mo_current = 1.0/mo_s_ratio if mo_s_ratio>0 else 2.0
    stage_code_c,stage_label_c,_ = li2019_stage(mo_s_ratio)
    stage_color = {'PRISTINE':'#888','STAGE_1':'#F5A623','STAGE_2_MILD':'#2DCE89',
                   'STAGE_2_DEEP':'#4E9AF1','STAGE_2_EXTREME':'#FF6464'}.get(stage_code_c,'#888')
    st.markdown(
        f"<div style='background:{stage_color}18;border-left:3px solid {stage_color};"
        f"padding:6px 10px;border-radius:3px;font-size:0.78em;color:{stage_color};margin:4px 0;'>"
        f"<b>{stage_label_c}</b> (S:Mo={s_mo_current:.2f})</div>",
        unsafe_allow_html=True)

    df_dist = df.copy()
    df_dist['dist'] = df.apply(lambda r: np.sqrt(
        ((r.layer_n-layer_n)/18)**2+((r.mo_s_ratio-mo_s_ratio)/0.36)**2+((r.ecsa-ecsa_val)/6.0)**2),axis=1)
    best_match = df_dist.nsmallest(1,'dist').iloc[0]
    dist_val   = df_dist['dist'].min()

    if dist_val<0.15:   st.success(f"✓ Closest sample: **{best_match['sample']}**")
    elif dist_val<0.40: st.info(f"≈ Nearest: **{best_match['sample']}** (interpolating)")
    else:               st.warning(f"⚠ Extrapolating — nearest: **{best_match['sample']}**")

    m_label,m_col_key,m_score,m_max,m_reasons = score_method(layer_n,mo_s_ratio,ecsa_val)
    m_color = METHOD_COLORS[m_col_key]
    pct = int(m_score/m_max*100)

    st.markdown('<div class="section-header">SYNTHESIS METHOD</div>', unsafe_allow_html=True)
    st.markdown(
        f"<div class='method-badge' style='background:{m_color}18;border-color:{m_color};color:{m_color};'>{m_label}</div>"
        f"<div class='score-bar-wrap'>"
        f"  <div style='font-size:0.72em;color:#111;font-family:IBM Plex Mono,monospace;margin-bottom:3px;'>MBE score: {m_score}/{m_max}</div>"
        f"  <div class='score-bar-bg'><div class='score-bar-fill' style='width:{pct}%;background:{m_color};'></div></div>"
        f"</div>", unsafe_allow_html=True)

    with st.expander("Scoring breakdown (v5.1)", expanded=False):
        for r in m_reasons:
            st.markdown(
                f"**{r['criterion']}**: {r['points']}/{r['max']} pts  \n{r['detail']}  \n"
                +" ".join([f"<span class='ref-chip'>{ref}</span>" for ref in r['refs']]),
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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTOR (v5.0 base + conductivity added)
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Predictor":
    st.markdown("# MoS₂ HER Trend Model — v5.1")
    st.markdown(
        "<div style='color:#111;font-size:0.9em;margin-bottom:20px;'>"
        "Gaussian Process · Jeon et al. <i>ACS Nano</i> 2026 · 14 MBE samples · 1M KOH · "
        "v5.1: Conductivity σ=1/ρ · Synthesis Physics (MBE vs CVD) · 20 papers</div>",
        unsafe_allow_html=True)

    m_color = METHOD_COLORS[m_col_key]
    st.markdown(
        f"<div style='background:{m_color}12;border:1.5px solid {m_color}40;"
        f"border-left:5px solid {m_color};padding:14px 20px;border-radius:6px;"
        f"margin-bottom:20px;display:flex;align-items:center;gap:20px;'>"
        f"<div style='font-size:1.3em;font-weight:700;color:{m_color};font-family:IBM Plex Mono,monospace;'>{m_label}</div>"
        f"<div style='color:#111;font-size:0.85em;'>Score {m_score}/{m_max} · "
        f"Layer# {layer_n} · Mo/S {mo_s_ratio:.2f} (S:Mo={s_mo_current:.2f}) · "
        f"ECSA {ecsa_val:.1f} cm² · <b style='color:{stage_color};'>{stage_label_c}</b></div>"
        f"</div>", unsafe_allow_html=True)

    if dist_val<0.05:
        vals   = {k: best_match[k] for k in TARGETS}
        source = f"Experimental data — {best_match['sample']} (Jeon 2026 Table 1)"
        gp_ci  = None
    else:
        vals   = predict_all(layer_n,mo_s_ratio,ecsa_val)
        source = "GP prediction (calibrated 95% credible interval)"
        gp_ci  = {k: dict(zip(['mean','lower','upper','std'],gp_predict(k,layer_n,mo_s_ratio,ecsa_val))) for k in TARGETS}

    st.caption(f"Source: {source}")

    st.markdown('<div class="section-header">KEY DESCRIPTORS</div>', unsafe_allow_html=True)
    kc1,kc2,kc3 = st.columns(3)
    for col,label,val,unit,status,note in [
        (kc1,"Layer # ✅",f"{layer_n}","layers",
         "🟢 ≤3L → k⁰≥1.5cm/s" if layer_n<=3 else ("🟢 4-6L → optimal" if layer_n<=6 else "🔵 Multi-layer"),
         "✅ Scherrer ×4 sources + Raman N5,N10"),
        (kc2,"Mo/S ratio ✅",f"{mo_s_ratio:.2f}",f"(S:Mo={s_mo_current:.2f})",
         f"🟢 {stage_label_c}" if stage_code_c in ['STAGE_2_MILD','STAGE_2_DEEP'] else f"🔵 {stage_label_c}",
         "✅ XPS calibrated · Stage threshold S:Mo=1.70 (Li 2019)"),
        (kc3,"ECSA ✅",f"{ecsa_val:.1f}","cm²",
         "🟢 High — max edge sites" if ecsa_val>=7.0 else "🔵 Moderate",
         "✅ Measured — Jeon 2026 Table 1"),
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
    # [P-1] conductivity replaces resistivity in display
    metrics_order = ['eta','tafel','rct','tof_ecsa','tof_mass','raman','conductivity']
    thresholds = {
        'eta':(-0.38,-0.50),'tafel':(110,200),'rct':(70,130),
        'raman':(1.8,2.2),'conductivity':(0.08,0.05),
        'tof_ecsa':(9,6),'tof_mass':(5,2),
    }
    eta_mV_pred = eta_v_to_mV_abs(vals['eta'])
    for i,key in enumerate(metrics_order):
        name,unit,better = TARGETS[key]
        v = vals[key]
        col = cols[i%4]
        fmt = f"{v:.4f}" if key=='conductivity' else (f"{v:.2f}" if abs(v)<100 else f"{v:.0f}")
        if gp_ci:
            std = gp_ci[key]['std']
            total_std = total_uncertainty_for_metric(key,v,std,dist_val,eta_mV_pred)
            if key=='conductivity': delta_str=f"±{total_std:.4f}"
            elif abs(total_std)<100: delta_str=f"±{total_std:.2f}"
            else: delta_str=f"±{total_std:.0f}"
            col.metric(name,f"{fmt} {unit}",delta=delta_str,delta_color="off")
        else:
            col.metric(name,f"{fmt} {unit}")

    vacancy_pct = vacancy_percent_from_mo_s(mo_s_ratio)
    vacancy_label,vacancy_strength,vacancy_note = vacancy_regime(vacancy_pct,mo_s_ratio)
    layer_factor = layer_activity_factor(layer_n)
    mechanism    = tafel_mechanism_v5(vals['tafel'],mo_s_ratio,layer_n)
    perf_class,perf_note = classify_performance_eta_v5(eta_mV_pred)
    rct_label,rct_note   = classify_rct_v5(vals['rct'])
    confidence,conf_warnings = confidence_level(layer_n,mo_s_ratio,ecsa_val,dist_val)
    lit_score,lit_notes  = literature_consistency_score_v5(eta_mV_pred,vals['tafel'],vals['rct'],mo_s_ratio,ecsa_val)

    if gp_ci:
        eta_total_std_mV = total_uncertainty_for_metric('eta',vals['eta'],gp_ci['eta']['std'],dist_val)*1000
        tafel_total_std  = total_uncertainty_for_metric('tafel',vals['tafel'],gp_ci['tafel']['std'],dist_val,eta_mV_pred)
    else:
        eta_total_std_mV = literature_experimental_sd_v5(eta_mV_pred,'eta')
        tafel_total_std  = literature_experimental_sd_v5(eta_mV_pred,'tafel')

    st.markdown('<div class="section-header">BULLETPROOF INTERPRETATION (v5.1)</div>', unsafe_allow_html=True)
    b1,b2,b3,b4,b5 = st.columns(5)
    b1.metric("Confidence",confidence)
    b2.metric("η10 magnitude",f"{eta_mV_pred:.0f} ± {eta_total_std_mV:.0f} mV")
    b3.metric("Tafel",f"{vals['tafel']:.0f} ± {tafel_total_std:.0f}")
    b4.metric("Stage", 
    "Pristine" if stage_code_c=="PRISTINE" else
    "Stage 1" if stage_code_c=="STAGE_1" else
    "Stage 2" if stage_code_c=="STAGE_2_MILD" else
    "Stage 2 deep" if stage_code_c=="STAGE_2_DEEP" else
    "Stage 2 risk")
    b5.metric("Lit. score",f"{lit_score:.1f}/5")

    # Simple 3-line summary — easy to read during presentation
    # Line 1: How good is this catalyst?
    if eta_mV_pred < 180:
        quality_line = f"This is a <b>good catalyst</b> for hydrogen production. Overpotential of {eta_mV_pred:.0f} mV is within the best range for MBE-grown MoS₂ on silicon (Jeon 2026 optimum: 330 mV)."
    elif eta_mV_pred < 350:
        quality_line = f"This is a <b>moderate catalyst</b>. Overpotential of {eta_mV_pred:.0f} mV. Increasing ECSA or Mo/S ratio would improve performance."
    else:
        quality_line = f"This catalyst has <b>low activity</b>. Overpotential of {eta_mV_pred:.0f} mV. The Mo/S ratio is near-stoichiometric — more sulfur vacancies are needed to activate the surface."

    # Line 2: What is limiting performance?
    if stage_code_c == 'PRISTINE':
        limit_line = "The surface is mostly <b>inactive</b>. The Mo/S ratio is too close to perfect stoichiometry — the basal plane does not have enough active sites."
    elif stage_code_c == 'STAGE_1':
        limit_line = f"The catalyst is in <b>Stage 1</b>. Point defects are starting to activate the surface. Increasing Mo/S above 0.59 would enter Stage 2 and further improve activity."
    elif stage_code_c in ['STAGE_2_MILD', 'STAGE_2_DEEP']:
        limit_line = f"The catalyst is in <b>Stage 2</b>. Undercoordinated molybdenum sites are active. In alkaline conditions (KOH), activity continues to increase through this stage (Li 2019)."
    else:
        limit_line = "The catalyst has <b>too many vacancies</b>. The structure is at risk of collapse. Jeon MoS-M2.0 confirms this regime gives worse performance."

    # Line 3: What does synthesis method mean for this result?
    if m_col_key == 'mbe':
        synth_line = f"<b>Physical method (MBE) recommended.</b> MBE directly controls the S/Mo ratio during growth, giving precise access to Stage 1 or Stage 2 without post-treatment."
    elif m_col_key == 'cvd':
        synth_line = f"<b>Chemical method (CVD) is sufficient.</b> At this near-stoichiometric composition, CVD at high temperature produces the required 2H phase reliably."
    else:
        synth_line = f"<b>Both methods can work here.</b> MBE gives more control over vacancy density; CVD gives better lateral homogeneity."

    box_class = 'stage2-box' if stage_code_c in ['STAGE_2_MILD','STAGE_2_DEEP'] else 'bulletproof-box'
    st.markdown(f"""
<div class='{box_class}'>
{quality_line}<br><br>
{limit_line}<br><br>
{synth_line}
</div>
""", unsafe_allow_html=True)

    if conf_warnings:
        st.markdown("<div class='risk-box'><b>Confidence notes</b><br>"
                    +"<br>".join(["• "+w for w in conf_warnings])+"</div>",unsafe_allow_html=True)
    if lit_notes:
        st.markdown('<div class="section-header">LITERATURE CONSISTENCY</div>', unsafe_allow_html=True)
        for note in lit_notes:
            st.markdown(f"<span class='validation-chip'>{note}</span>",unsafe_allow_html=True)

    st.markdown('<div class="section-header">3 CLOSEST EXPERIMENTAL SAMPLES</div>', unsafe_allow_html=True)
    df_dist2 = df.copy()
    df_dist2['dist'] = df.apply(lambda r: np.sqrt(
        ((r.layer_n-layer_n)/18)**2+((r.mo_s_ratio-mo_s_ratio)/0.36)**2+((r.ecsa-ecsa_val)/6.0)**2),axis=1)
    closest = df_dist2.nsmallest(3,'dist').copy()
    closest['Stage (Li2019)'] = closest['mo_s_ratio'].apply(lambda x: li2019_stage(x)[1])
    closest['σ (S/cm)'] = (1.0/closest['resistivity']).map(lambda x: f"{x:.4f}")
    show_cols = ['sample','series','layer_n','mo_s_ratio','ecsa','eta','tafel','rct','tof_ecsa','tof_mass','σ (S/cm)','Stage (Li2019)']
    st.dataframe(closest[show_cols].reset_index(drop=True),use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# [P-2] NEW PAGE: SYNTHESIS PHYSICS — MBE vs CVD (dynamic)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Synthesis Physics":
    st.markdown("# Synthesis Physics: MBE vs CVD")
    st.markdown(
        "<div class='provenance-box'>"
        "<b>Key research question:</b> MBE and CVD can produce MoS₂ with similar layer# and Mo/S — "
        "but performance differs. Why? The answer lies in <b>grain boundary density → ECSA → η</b>. "
        "This page explains the mechanism with real literature data."
        "</div>", unsafe_allow_html=True)

    st.markdown("## 1. Thermodynamic vs Kinetic Control")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown(
            "<div class='synth-cvd'>"
            "<b>🧪 CVD — Chemical Synthesis</b><br><br>"
            "<b>How it works:</b> CVD operates at high temperature (600-900°C) with high sulfur "
            "partial pressure. These conditions drive the system toward thermodynamic equilibrium, "
            "so the most stable phase, 2H MoS2, forms preferentially.<br><br>"
            "<b>Grain structure:</b> Equilibrium conditions produce large, uniform grains across "
            "centimeter-scale areas. Grain boundary density is approximately 0.04 per micrometer "
            "(ACS AMI), which means fewer active edge sites per unit area.<br><br>"
            "<b>Stoichiometry:</b> S-vacancy density is harder to tune independently because it "
            "is coupled to temperature and sulfur atmosphere.<br><br>"
            "<b>ECSA result:</b> Fewer grain boundaries produce lower ECSA, but uniformity is "
            "higher and results are more reproducible across samples.<br><br>"
            "<b>Phase stability:</b> The 2H phase remains stable at room temperature after "
            "synthesis, giving consistent Tafel slopes across different labs.<br><br>"
            "<b>To reach Stage 2:</b> Post-synthesis treatment is required, such as "
            "plasma etching or H2 annealing."
            "</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            "<div class='synth-mbe'>"
            "<b>🔬 MBE — Physical Synthesis</b><br><br>"
            "<b>How it works:</b> In MBE, sulfur and molybdenum arrive as independent atomic "
            "beams. Growth happens far from thermodynamic equilibrium. Sulfur deficiency during "
            "deposition spontaneously creates high-density twin grain boundaries "
            "(Ma et al. ACS Nano 2017).<br><br>"
            "<b>Grain structure:</b> Layer-by-layer control is precise, but lateral homogeneity "
            "can vary at the nanometer scale. S-rich and S-poor domains can coexist "
            "within the same film.<br><br>"
            "<b>Stoichiometry:</b> The S/Mo flux ratio directly controls vacancy density during "
            "growth. This means Stage 1 or Stage 2 can be engineered without any "
            "post-synthesis treatment.<br><br>"
            "<b>ECSA result:</b> Higher grain boundary density produces higher ECSA. "
            "Jeon MoS-M6.0 achieves ECSA = 9.2 cm2, the highest in the dataset, "
            "despite having 20 layers. This is a direct synthesis effect.<br><br>"
            "<b>Phase stability:</b> Because growth happens out of equilibrium, "
            "thermodynamically metastable phases are possible. The structure may not "
            "be fully stabilized.<br><br>"
            "<b>To reach Stage 2:</b> Directly achievable by reducing the S-flux "
            "during growth, without any post-synthesis step."
            "</div>", unsafe_allow_html=True)

    st.markdown("## 2. The Causal Chain — Your Current Parameters")

    # Dynamic predictions
    pred_now  = predict_all(layer_n, mo_s_ratio, ecsa_val)
    eta_now   = eta_v_to_mV_abs(pred_now['eta'])
    cond_now  = pred_now['conductivity']
    sc_d,sl_d,_ = li2019_stage(mo_s_ratio)

    ecsa_high = min(ecsa_val*1.5, 12.0)
    pred_high = predict_all(layer_n, mo_s_ratio, ecsa_high)
    eta_high  = eta_v_to_mV_abs(pred_high['eta'])

    ecsa_low  = max(ecsa_val*0.55, 2.0)
    pred_low  = predict_all(layer_n, mo_s_ratio, ecsa_low)
    eta_low   = eta_v_to_mV_abs(pred_low['eta'])

    st.markdown(
        f"<div class='provenance-box'>"
        f"<b>Jeon 2026 (direct measurement):</b> "
        f"When MBE annealing temperature increases from 600 to 800°C, crystallinity improves but "
        f"ECSA drops from 6.7 to 3.5 cm² due to grain coalescence. "
        f"As a result, overpotential worsens from -0.46V to -0.58V.<br><br>"
        f"<b>Your current parameters: Layer#{layer_n}, Mo/S={mo_s_ratio:.2f}, ECSA={ecsa_val:.1f} cm² "
        f"— GP predicts η={eta_now:.0f} mV, σ={cond_now:.4f} S/cm [{sl_d}]</b>"
        f"</div>", unsafe_allow_html=True)

    # Fully dynamic text based on current slider values
    if m_col_key == 'mbe':
        synth_step1 = (
            f"Your parameters (Layer#{layer_n}, Mo/S={mo_s_ratio:.2f}, ECSA={ecsa_val:.1f} cm²) "
            f"indicate **Physical Method (MBE)**. "
            f"In MBE, sulfur deficiency during deposition spontaneously generates high-density "
            f"twin grain boundaries (Ma et al. ACS Nano 2017). "
            f"The structure is thermodynamically metastable — crystallinity and grain size are "
            f"controlled by annealing temperature and sulfur flux, not by thermodynamic equilibrium."
        )
    elif m_col_key == 'cvd':
        synth_step1 = (
            f"Your parameters (Layer#{layer_n}, Mo/S={mo_s_ratio:.2f}, ECSA={ecsa_val:.1f} cm²) "
            f"indicate **Chemical Method (CVD)**. "
            f"In CVD, high temperature and pressure drive the system toward thermodynamic equilibrium. "
            f"This produces large, uniform grains with low grain boundary density "
            f"of approximately 0.04 µm⁻¹ (ACS AMI). The 2H phase is thermodynamically stable."
        )
    else:
        synth_step1 = (
            f"Your parameters (Layer#{layer_n}, Mo/S={mo_s_ratio:.2f}, ECSA={ecsa_val:.1f} cm²) "
            f"are compatible with **both MBE and CVD**. "
            f"MBE would generate grain boundaries through sulfur deficiency (Ma et al. ACS Nano 2017), "
            f"while CVD would produce larger, more uniform grains (GB density ~0.04 µm⁻¹, ACS AMI)."
        )

    if ecsa_val >= 7.0:
        ecsa_step2 = (
            f"Your ECSA of **{ecsa_val:.1f} cm²** is relatively high within the Jeon 2026 dataset "
            f"(range: 3.5–9.2 cm²). High ECSA means high grain boundary density, which provides "
            f"more active edge sites for hydrogen evolution. "
            f"Grain boundary density can reach up to 10¹² cm⁻² in optimized MBE films — "
            f"such films show onset potential of -25 mV and Tafel slope of 54 mV/dec "
            f"(Nature Communications 2020)."
        )
    else:
        ecsa_step2 = (
            f"Your ECSA of **{ecsa_val:.1f} cm²** is relatively low within the Jeon 2026 dataset "
            f"(range: 3.5–9.2 cm²). Lower ECSA indicates fewer grain boundaries and edge sites. "
            f"This is consistent with higher annealing temperatures or CVD growth, "
            f"which produce larger grains with fewer boundaries. "
            f"Increasing ECSA toward 8–9 cm² (as in Jeon MoS-N10 or MoS-M6.0) "
            f"would be expected to improve HER performance."
        )

    stage_step3 = (
        f"Your Mo/S ratio of {mo_s_ratio:.2f} (S:Mo = {s_mo_current:.2f}) places this sample "
        f"in **{sl_d}**. "
    )
    if stage_code_c == 'STAGE_1':
        stage_step3 += (
            f"In this regime, point defects are activating. "
            f"Tafel slope decreases rapidly with increasing vacancy density in KOH (Li 2019). "
            f"Increasing Mo/S ratio further (above 0.588) would enter Stage 2, "
            f"where undercoordinated Mo regions form and TOF continues to increase."
        )
    elif stage_code_c in ['STAGE_2_MILD', 'STAGE_2_DEEP']:
        stage_step3 += (
            f"In this regime, undercoordinated Mo regions are active HER sites. "
            f"In KOH, TOF continuously increases through Stage 2, unlike in H₂SO₄ where it saturates (Li 2019). "
            f"This is a high-activity regime for alkaline HER."
        )
    else:
        stage_step3 += (
            f"Near-stoichiometric MoS₂. The basal plane is mostly inert. "
            f"Reducing the Mo/S ratio below 0.588 would enter Stage 1 (point defect activation)."
        )

    st.markdown(f"""
**Step 1 — Synthesis method determines grain structure**

{synth_step1}

**Step 2 — Grain boundary density determines ECSA**

{ecsa_step2}

**Step 3 — Stoichiometry and stage determine intrinsic activity**

{stage_step3}

**Step 4 — ECSA and intrinsic activity together determine overpotential**

With your current ECSA of {ecsa_val:.1f} cm², the model predicts η = **{eta_now:.0f} mV**.

If ECSA increases to {ecsa_high:.1f} cm² (more grain boundaries through MBE optimization), predicted η improves to approximately **{eta_high:.0f} mV**.

If ECSA decreases to {ecsa_low:.1f} cm² (grain coalescence due to higher annealing temperature), predicted η worsens to approximately **{eta_low:.0f} mV**.

**Step 5 — The conductivity trade-off**

Higher grain boundary density increases ECSA but may reduce electrical conductivity σ. In-plane electron mobility is 2200 times faster than out-of-plane mobility (ACS Catalysis 2016). Smaller grains provide more active sites but interrupt conductive pathways. Your current conductivity σ = **{cond_now:.4f} S/cm**.
""")

    mc1,mc2,mc3 = st.columns(3)
    mc1.metric("Current η", f"{eta_now:.0f} mV", delta=f"ECSA={ecsa_val:.1f} cm²", delta_color="off")
    mc2.metric(f"η if ECSA={ecsa_high:.1f} cm² (more GBs)", f"{eta_high:.0f} mV",
               delta=f"{eta_now-eta_high:+.0f} mV vs current",
               delta_color="normal" if eta_high<eta_now else "inverse")
    mc3.metric(f"η if ECSA={ecsa_low:.1f} cm² (grain coalescence)", f"{eta_low:.0f} mV",
               delta=f"{eta_now-eta_low:+.0f} mV vs current",
               delta_color="normal" if eta_low<eta_now else "inverse")

    st.markdown(
        "<div class='correction-box'>"
        "<b>Implication for the GP model:</b> ECSA is the mediating variable between synthesis method and η. "
        "By including ECSA as a feature, the model partially captures the synthesis method effect. "
        "However, grain size distribution (lateral homogeneity) is not explicitly in the model — "
        "it is an <b>unobserved variable</b>. "
        "This is why two samples with the same layer# and Mo/S but different ECSA give different η — "
        "the difference originates from the synthesis method through grain boundary density."
        "</div>", unsafe_allow_html=True)

    st.markdown("## 3. Conductivity σ vs Mo/S ratio")
    fig_cond = go.Figure()
    for ser,scol in SERIES_COLORS.items():
        mask = df['series']==ser
        fig_cond.add_trace(go.Scatter(
            x=df[mask]['mo_s_ratio'], y=df[mask]['conductivity'],
            mode='markers+text', name=SERIES_LABELS[ser],
            marker=dict(size=12,color=scol,line=dict(width=1.5,color='white')),
            text=df[mask]['sample'], textposition='top center', textfont=dict(size=9),
            hovertemplate='<b>%{text}</b><br>Mo/S=%{x:.3f}<br>σ=%{y:.4f} S/cm<extra></extra>'))
    fig_cond.add_vline(x=0.588,line_dash='dash',line_color='#F5A623',line_width=2,
                       annotation_text="Stage 1→2 (Li 2019)",annotation_font_color='#F5A623')
    fig_cond.update_layout(
        title="Conductivity σ (S/cm) vs Mo/S ratio [σ=1/ρ — more physically interpretable than resistivity]",
        xaxis_title="Mo/S ratio",yaxis_title="Conductivity σ (S/cm)",
        height=400,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_cond,use_container_width=True)

    st.markdown("## 4. MBE vs CVD — Literature Comparison")
    comparison_df = pd.DataFrame([
        {'Property':'Thermodynamic stability','CVD':'Equilibrium 2H — thermodynamically stable','MBE':'Kinetically metastable — structure may not be fully stabilized','Source':'ACS Cat 2023; Jeon 2026'},
        {'Property':'Lateral homogeneity','CVD':'cm-scale uniform, GB density ~0.04 µm⁻¹','MBE':'nm-scale variable; twin GBs spontaneous from S-deficiency','Source':'ACS AMI (CVD); Ma et al. ACS Nano 2017 (MBE)'},
        {'Property':'ECSA vs synthesis T','CVD':'↑T → larger grains → ↓ECSA','MBE':'Jeon 2026: ECSA = 6.7 cm² (600°C) → 3.5 cm² (800°C)','Source':'Jeon 2026 Table 1 (direct measurement)'},
        {'Property':'Grain boundaries as active sites','CVD':'Fewer GBs → lower but uniform ECSA','MBE':'GBs intrinsically active: GB density ~10¹² cm⁻² → onset −25 mV, Tafel 54 mV/dec','Source':'Nature Comms 2020'},
        {'Property':'GB formation mechanism','CVD':'GBs from island coalescence during growth','MBE':'Twin GBs spontaneous from chalcogen deficiency during deposition','Source':'Ma et al. ACS Nano 2017'},
        {'Property':'σ vs edge site trade-off','CVD':'Large grains → high σ, fewer edge sites','MBE':'Small grains → more ECSA but potentially lower σ','Source':'ACS Cat 2016: in-plane mobility 2200× out-of-plane'},
        {'Property':'S-vacancy engineering','CVD':'Coupled to T and S atmosphere','MBE':'Direct S/Mo flux ratio → Stage 1/2 engineering during growth','Source':'Jeon 2026; Sherwood 2024'},
    ])
    st.dataframe(comparison_df,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TREND CURVES (unchanged from v5.0)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trend Curves":
    st.markdown("# Trend Curves")
    tc1,tc2 = st.columns([1,2])
    with tc1:
        target_tc = st.selectbox("Performance metric",options=list(TARGETS.keys()),
            format_func=lambda k: f"{TARGETS[k][0]} ({TARGETS[k][1]})")
    with tc2:
        feat_tc = st.selectbox("Descriptor to vary",options=FEATURES,
            format_func=lambda k: FEATURE_LABELS[k])
    name_tc,unit_tc,better_tc = TARGETS[target_tc]
    defaults = {'layer_n':layer_n,'mo_s_ratio':mo_s_ratio,'ecsa':ecsa_val}
    lo,hi = FEATURE_RANGES[feat_tc]
    x_range = np.linspace(lo,hi,80)
    y_means,y_lows,y_highs = [],[],[]
    for xv in x_range:
        row = {f:(xv if f==feat_tc else defaults[f]) for f in FEATURES}
        m,lo_,hi_,_ = gp_predict(target_tc,row['layer_n'],row['mo_s_ratio'],row['ecsa'])
        y_means.append(m); y_lows.append(lo_); y_highs.append(hi_)
    y_means=np.array(y_means); y_lows=np.array(y_lows); y_highs=np.array(y_highs)
    exp_lo=df[feat_tc].min(); exp_hi=df[feat_tc].max()
    in_range=(x_range>=exp_lo)&(x_range<=exp_hi)
    fig_tc=go.Figure()
    fig_tc.add_trace(go.Scatter(x=np.concatenate([x_range,x_range[::-1]]),y=np.concatenate([y_highs,y_lows[::-1]]),
        fill='toself',fillcolor='rgba(78,154,241,0.12)',line=dict(color='rgba(0,0,0,0)'),name='95% CI (GP)'))
    fig_tc.add_trace(go.Scatter(x=x_range,y=y_means,mode='lines',
        line=dict(color='rgba(78,154,241,0.35)',width=1.5,dash='dot'),name='GP mean (extrapolation)'))
    fig_tc.add_trace(go.Scatter(x=x_range[in_range],y=y_means[in_range],mode='lines',
        line=dict(color='#4E9AF1',width=3),name='GP mean (interpolation)'))
    for ser,scolor in SERIES_COLORS.items():
        mask=df['series']==ser
        fig_tc.add_trace(go.Scatter(x=df[feat_tc].values[mask],y=df[target_tc].values[mask],mode='markers',
            name=SERIES_LABELS[ser],marker=dict(size=11,color=scolor,line=dict(width=1.5,color='white')),
            text=df['sample'][mask],
            hovertemplate='<b>%{text}</b><br>'+FEATURE_LABELS[feat_tc]+'=%{x:.2f}<br>'+name_tc+'=%{y:.3f} '+unit_tc+'<extra></extra>'))
    cur_val=defaults[feat_tc]
    fig_tc.add_vline(x=cur_val,line_width=1.5,line_dash="dash",line_color=METHOD_COLORS[m_col_key],
                     annotation_text=f"Current: {cur_val:.2f}",annotation_font_color=METHOD_COLORS[m_col_key])
    if feat_tc=='mo_s_ratio':
        fig_tc.add_vline(x=0.500,line_dash='dot',line_color='#888',line_width=1,
                         annotation_text="S:Mo=2.0 (stoich.)",annotation_font_color='#888')
        fig_tc.add_vline(x=0.588,line_dash='dot',line_color='#F5A623',line_width=2,
                         annotation_text="S:Mo=1.70 STAGE 1→2 (Li 2019)",annotation_font_color='#F5A623')
        fig_tc.add_vline(x=0.752,line_dash='dot',line_color='#FF6464',line_width=1,
                         annotation_text="S:Mo=1.33 (Stage 2 deep)",annotation_font_color='#FF6464')
    fig_tc.update_layout(
        title=f"{name_tc} vs {FEATURE_LABELS[feat_tc]}<br><sup>{FEATURE_PROVENANCE[feat_tc]}</sup>",
        xaxis_title=FEATURE_LABELS[feat_tc],yaxis_title=f"{name_tc} ({unit_tc})",
        height=500,legend=dict(orientation='h',yanchor='bottom',y=-0.40),
        plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
    fig_tc.update_xaxes(showgrid=True,gridcolor='rgba(128,128,128,0.12)')
    fig_tc.update_yaxes(showgrid=True,gridcolor='rgba(128,128,128,0.12)')
    st.plotly_chart(fig_tc,use_container_width=True)
    st.info("**Stage threshold (Li 2019 ACS Nano, KOH PRIMARY):** S:Mo=1.70 (Mo/S=0.588) separates Stage 1 (point defects) from Stage 2 (undercoordinated Mo). In KOH, TOF continues to increase through Stage 2 — unlike H₂SO₄.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 2D HEATMAPS (unchanged from v5.0)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺 2D Heatmaps":
    st.markdown("# 2D Heatmaps")
    hc1,hc2=st.columns(2)
    with hc1:
        target_hm=st.selectbox("Performance metric",options=list(TARGETS.keys()),format_func=lambda k:f"{TARGETS[k][0]} ({TARGETS[k][1]})")
    with hc2:
        axis_pair=st.selectbox("Axes",["Layer# × Mo/S  (ECSA fixed)","Layer# × ECSA  (Mo/S fixed)","Mo/S × ECSA   (Layer# fixed)"])
    name_hm,unit_hm,better_hm=TARGETS[target_hm]
    N=40; defaults_hm={'layer_n':layer_n,'mo_s_ratio':mo_s_ratio,'ecsa':ecsa_val}
    if axis_pair.startswith("Layer# × Mo/S"):   xf,yf,fixed_f='layer_n','mo_s_ratio','ecsa'; xlabel,ylabel='Layer # (validated)','Mo/S ratio (validated)'
    elif axis_pair.startswith("Layer# × ECSA"): xf,yf,fixed_f='layer_n','ecsa','mo_s_ratio'; xlabel,ylabel='Layer # (validated)','ECSA (cm²)'
    else:                                        xf,yf,fixed_f='mo_s_ratio','ecsa','layer_n'; xlabel,ylabel='Mo/S ratio (validated)','ECSA (cm²)'
    xlo,xhi=FEATURE_RANGES[xf]; ylo,yhi=FEATURE_RANGES[yf]
    xgrid=np.linspace(xlo,xhi,N); ygrid=np.linspace(ylo,yhi,N)
    Z=np.zeros((N,N))
    for i,yv in enumerate(ygrid):
        for j,xv in enumerate(xgrid):
            row={xf:xv,yf:yv,fixed_f:defaults_hm[fixed_f]}
            Z[i,j]=gp_predict(target_hm,row['layer_n'],row['mo_s_ratio'],row['ecsa'])[0]
    cs='RdYlGn' if better_hm=='max' else 'RdYlGn_r'
    fig_hm=go.Figure(data=go.Heatmap(z=Z,x=xgrid,y=ygrid,colorscale=cs,
        colorbar=dict(title=dict(text=f"{name_hm} ({unit_hm})",side='right')),
        hovertemplate=f'{xlabel}=%{{x:.2f}}<br>{ylabel}=%{{y:.2f}}<br>{name_hm}=%{{z:.3f}} {unit_hm}<extra></extra>'))
    for ser,scolor in SERIES_COLORS.items():
        mask=df['series']==ser
        fig_hm.add_trace(go.Scatter(x=df[xf].values[mask],y=df[yf].values[mask],mode='markers+text',
            marker=dict(size=12,color=scolor,line=dict(width=2,color='white')),
            text=df['sample'][mask],textposition='top center',textfont=dict(size=9,color='white'),
            name=SERIES_LABELS[ser],customdata=df[target_hm].values[mask],
            hovertemplate='<b>%{text}</b><br>'+xlabel+'=%{x:.2f}<br>'+ylabel+'=%{y:.2f}<br>'+name_hm+f'=%{{customdata:.3f}} {unit_hm}<extra></extra>'))
    fig_hm.add_trace(go.Scatter(x=[defaults_hm[xf]],y=[defaults_hm[yf]],mode='markers',
        marker=dict(size=16,color=METHOD_COLORS[m_col_key],symbol='star',line=dict(width=2,color='white')),name='Your position'))
    if yf=='mo_s_ratio': fig_hm.add_hline(y=0.588,line_dash='dash',line_color='#F5A623',line_width=2,annotation_text="Stage 1→2 (Li 2019)",annotation_font_color='#F5A623')
    if xf=='mo_s_ratio': fig_hm.add_vline(x=0.588,line_dash='dash',line_color='#F5A623',line_width=2,annotation_text="Stage 1→2",annotation_font_color='#F5A623')
    fig_hm.update_layout(title=f"{name_hm} — {xlabel} × {ylabel} | fixed={FEATURE_LABELS[fixed_f]}={defaults_hm[fixed_f]:.2f}",
        xaxis_title=xlabel,yaxis_title=ylabel,height=540,
        legend=dict(orientation='h',yanchor='bottom',y=-0.22),
        plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hm,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 3D EXPLORER (unchanged from v5.0)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌐 3D Explorer":
    st.markdown("# 3D Descriptor Space Explorer")
    t3c1,t3c2=st.columns(2)
    with t3c1:
        target_3d=st.selectbox("Color metric",options=list(TARGETS.keys()),format_func=lambda k:f"{TARGETS[k][0]} ({TARGETS[k][1]})")
    with t3c2:
        show_surf=st.checkbox("Show GP surface slice (Mo/S fixed)",value=True)
    name_3d,unit_3d,better_3d=TARGETS[target_3d]
    fig_3d=go.Figure()
    if show_surf:
        N3=25; ln3=np.linspace(1,20,N3); ec3=np.linspace(2,12,N3); Zs=np.zeros((N3,N3))
        for i,ev in enumerate(ec3):
            for j,lv in enumerate(ln3):
                Zs[i,j]=gp_predict(target_3d,lv,mo_s_ratio,ev)[0]
        fig_3d.add_trace(go.Surface(x=ln3,y=ec3,z=Zs,colorscale='RdYlGn' if better_3d=='max' else 'RdYlGn_r',
            opacity=0.55,showscale=False,name=f'GP surface (Mo/S={mo_s_ratio:.2f})'))
    for ser,scolor in SERIES_COLORS.items():
        mask=df['series']==ser; sub=df[mask]
        fig_3d.add_trace(go.Scatter3d(x=sub['layer_n'],y=sub['ecsa'],z=sub['mo_s_ratio'],mode='markers+text',
            marker=dict(size=8,color=sub[target_3d].values,colorscale='RdYlGn' if better_3d=='max' else 'RdYlGn_r',
                        cmin=df[target_3d].min(),cmax=df[target_3d].max(),line=dict(width=2,color='white')),
            text=sub['sample'],name=SERIES_LABELS[ser]))
    cur_pred=gp_predict(target_3d,layer_n,mo_s_ratio,ecsa_val)[0]
    fig_3d.add_trace(go.Scatter3d(x=[layer_n],y=[ecsa_val],z=[mo_s_ratio],mode='markers',
        marker=dict(size=14,color=METHOD_COLORS[m_col_key],symbol='diamond',line=dict(width=3,color='white')),
        name=f'Your position ({cur_pred:.3f} {unit_3d})'))
    fig_3d.update_layout(scene=dict(xaxis_title='Layer # (validated)',yaxis_title='ECSA (cm²)',
                                    zaxis_title='Mo/S ratio (Stage threshold=0.588)'),
        title=f"{name_3d} ({unit_3d}) in descriptor space | Stage 1→2 at Mo/S=0.588",
        height=620,paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_3d,use_container_width=True)
    st.info(f"**Your position:** Layer# {layer_n} · Mo/S {mo_s_ratio:.2f} (S:Mo={s_mo_current:.2f}) · ECSA {ecsa_val:.1f} cm² · **{stage_label_c}** → GP predicts **{name_3d} = {cur_pred:.3f} {unit_3d}** | Method: **{m_label}**")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INVERSE PREDICTOR (unchanged from v5.0)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Inverse Predictor":
    st.markdown("# Inverse Predictor")
    ic1,ic2,ic3,ic4=st.columns(4)
    with ic1: t_eta=st.slider("Target η (V)",-0.60,-0.25,-0.35,0.01)
    with ic2: t_tafel=st.slider("Target Tafel (mV/dec)",60,300,100,5)
    with ic3: t_ecsa=st.slider("Target ECSA (cm²)",2.0,12.0,7.0,0.5)
    with ic4: t_rct=st.slider("Target Rct (Ω·cm²)",20.0,200.0,60.0,5.0)
    df_inv=df.copy()
    df_inv['perf_score']=df_inv.apply(lambda r: np.sqrt(
        ((r.eta-t_eta)/0.30)**2+((r.tafel-t_tafel)/250)**2+((r.ecsa-t_ecsa)/8)**2+((r.rct-t_rct)/180)**2),axis=1)
    candidates=df_inv.nsmallest(3,'perf_score')
    best_inv=candidates.iloc[0]
    st.markdown('<div class="section-header">CLOSEST EXPERIMENTAL MATCHES</div>',unsafe_allow_html=True)
    candidates_show=candidates.copy()
    candidates_show['Stage (Li2019)']=candidates_show['mo_s_ratio'].apply(lambda x: li2019_stage(x)[1])
    show_inv=candidates_show[['sample','series','layer_n','mo_s_ratio','ecsa','eta','tafel','rct','tof_ecsa','Stage (Li2019)']].reset_index(drop=True)
    st.dataframe(show_inv,use_container_width=True)
    inv_label,inv_col,inv_score,inv_max,inv_reasons=score_method(best_inv['layer_n'],best_inv['mo_s_ratio'],best_inv['ecsa'],best_inv['rct'])
    inv_color=METHOD_COLORS[inv_col]
    st.markdown(
        f"<div style='background:{inv_color}12;border:2px solid {inv_color}40;"
        f"border-left:5px solid {inv_color};padding:16px 20px;border-radius:6px;'>"
        f"<div style='font-size:1.4em;font-weight:700;color:{inv_color};font-family:IBM Plex Mono,monospace;'>{inv_label}</div>"
        f"<div style='color:#111;margin-top:6px;'>Best match: <b>{best_inv['sample']}</b> · "
        f"η={best_inv.eta:.2f}V · Tafel={best_inv.tafel:.0f} · Rct={best_inv.rct:.1f} · Stage: {li2019_stage(best_inv['mo_s_ratio'])[1]}</div>"
        f"<div style='margin-top:10px;'><b>MBE score: {inv_score}/{inv_max}</b></div>"
        f"</div>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE (unchanged from v5.0)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧮 Feature Importance":
    st.markdown("# Feature Importance")
    perf_rows=[]
    for k in TARGETS:
        n_name,u,_=TARGETS[k]
        perf_rows.append({'Property':n_name,'Unit':u,
            'GP R²':round(gp_scores[k]['r2'],3),'GP MAE':round(gp_scores[k]['mae'],3),
            'RF R²':round(rf_scores[k]['r2'],3),'RF MAE':round(rf_scores[k]['mae'],3)})
    st.dataframe(pd.DataFrame(perf_rows),use_container_width=True)
    fi_colors={'layer_n':'#9B59B6','mo_s_ratio':'#E84040','ecsa':'#2DCE89'}
    fi_names={'layer_n':'Layer # (validated)','mo_s_ratio':'Mo/S (validated)','ecsa':'ECSA (measured)'}
    imp_target=st.selectbox("Property for importance",options=list(TARGETS.keys()),format_func=lambda k:f"{TARGETS[k][0]} ({TARGETS[k][1]})")
    imps=rf_imps[imp_target]
    fig_fi=go.Figure(go.Bar(x=[fi_names[f] for f in FEATURES],y=imps,marker_color=[fi_colors[f] for f in FEATURES],
        text=[f"{v:.3f}" for v in imps],textposition='outside'))
    fig_fi.update_layout(title=f"Feature importance — {TARGETS[imp_target][0]}",
        yaxis_title='Relative importance',yaxis_range=[0,max(imps)*1.3],
        height=320,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_fi,use_container_width=True)
    heat=np.array([[rf_imps[k][i] for i in range(3)] for k in TARGETS])
    heat_df=pd.DataFrame(heat,index=[TARGETS[k][0] for k in TARGETS],columns=[fi_names[f] for f in FEATURES])
    fig_heat=px.imshow(heat_df,text_auto=".2f",aspect="auto",color_continuous_scale='Greens',zmin=0,zmax=1,
                       title="Feature importance matrix — all targets")
    fig_heat.update_layout(height=380)
    st.plotly_chart(fig_heat,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# [P-3] PAGE: THEORETICAL BASIS — v5.0 base + structural parameters table + cross-paper check
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Theoretical Basis":
    st.markdown("# Theoretical Framework — v5.1")
    st.markdown(
        "<div class='fix-box'>"
        "<b>v5.0 PRIMARY CORRECTION — Li et al. ACS Nano 2019 as KOH PRIMARY SOURCE:</b><br>"
        "Stage 1 (S:Mo > 1.70): point defects → Tafel decreases RAPIDLY in KOH.<br>"
        "Stage 2 (S:Mo < 1.70): undercoordinated Mo → TOF CONTINUOUSLY INCREASES in KOH.<br>"
        "Interlayer spacing: 6.62 ± 0.01 Å (7th independent source)."
        "</div>", unsafe_allow_html=True)

    # [P-3] Structural parameters → electrochemical properties
    st.markdown("## Structural Parameters and Electrochemical Properties")
    struct_table = pd.DataFrame([
        {'Parameter':'Number of layers','Effect on performance':'More layers = higher charge transfer resistance. Layer# does not directly drive overpotential — its effect goes through ECSA (confirmed Jeon 2026)','Key metric':'Rct, σ'},
        {'Parameter':'Mo/S ratio (XPS)','Effect on performance':'Lower Mo/S = more S-vacancies = more active Mo sites. Below S:Mo=1.70 enters Stage 2 where activity is highest in KOH (Li 2019)','Key metric':'η, Tafel, TOF'},
        {'Parameter':'Grain size / homogeneity','Effect on performance':'Smaller grains = more grain boundaries = higher ECSA. This is why MBE and CVD give different η at the same layer# and Mo/S','Key metric':'ECSA, η'},
        {'Parameter':'ECSA','Effect on performance':'More electrochemically active area = more active sites = lower overpotential. Synthesis method controls grain size, which controls ECSA','Key metric':'η, TOF mass'},
        {'Parameter':'Interlayer spacing','Effect on performance':'S-vacancies expand spacing from 6.15 to 6.62 Angstrom, improving electrolyte access and reducing Rct','Key metric':'Rct'},
        {'Parameter':'Phase (2H vs 1T)','Effect on performance':'1T phase has much higher conductivity than 2H, which reduces Rct dramatically. But 1T is unstable and converts back to 2H during operation','Key metric':'σ, Rct'},
        {'Parameter':'Film thickness','Effect on performance':'Thicker films have higher loading but lower mass-normalized activity (TOF mass). Jeon M-series explores this directly','Key metric':'TOF mass'},
        {'Parameter':'Raman ratio A1g/E2g','Effect on performance':'Lower ratio means more S-vacancies. Jeon N10 has the lowest ratio (1.63) and the best performance in the N-series','Key metric':'Proxy for Mo/S'},
    ])
    st.dataframe(struct_table, use_container_width=True)

    # [P-4] Cross-paper consistency
    st.markdown("## Cross-Paper Consistency Check (20 papers)")
    st.markdown(
        "<div class='correction-box'>"
        "<b>Research question:</b> Do the 20 papers in this model agree on the key MoS₂ descriptors? "
        "The table below shows what each group of papers agrees on, any discrepancies, and confidence level."
        "</div>", unsafe_allow_html=True)
    consistency_df = pd.DataFrame([
        {'Topic':'Stage 1/2 threshold','Consensus':'S:Mo = 1.70 (Mo/S = 0.588)','Papers agree?':'YES — all converge','Confidence':'HIGH'},
        {'Topic':'Layer penalty per layer','Consensus':'Activity decreases 4.47x per layer added','Papers agree?':'YES — minor difference in spacing measurement only','Confidence':'HIGH'},
        {'Topic':'MBE generates grain boundaries from S-deficiency','Consensus':'S-deficiency in MBE creates twin grain boundaries spontaneously','Papers agree?':'YES — Ma 2017 (MoSe2) confirmed for MoS2 by Jeon 2026','Confidence':'HIGH'},
        {'Topic':'Grain boundaries are active HER sites','Consensus':'GB density up to 10e12 cm-2 gives onset potential -25 mV and Tafel 54 mV/dec','Papers agree?':'YES — no contradictions found','Confidence':'HIGH'},
        {'Topic':'CVD is more laterally homogeneous','Consensus':'CVD grain boundary density approximately 0.04 per micrometer, cm-scale uniform','Papers agree?':'YES — varies with growth conditions but direction is consistent','Confidence':'MEDIUM-HIGH'},
        {'Topic':'S-vacancies activate surface','Consensus':'Mo 3d shift -0.5 eV with S-vacancy brings free energy of adsorption close to 0 eV','Papers agree?':'YES — DFT values differ slightly but trend is consistent','Confidence':'HIGH'},
        {'Topic':'Stage 2 behavior differs in KOH vs acid','Consensus':'In KOH, activity continues to increase through Stage 2. In H2SO4, it saturates earlier','Papers agree?':'YES — but only 1-2 papers test specifically in KOH','Confidence':'HIGH (few KOH papers)'},
        {'Topic':'Rct benchmarks in 1M KOH','Consensus':'Bulk MoS2: above 200 Ohm. Engineered MoS2: 5 to 80 Ohm','Papers agree?':'YES — but must check if values are raw Ohm or normalized Ohm cm2','Confidence':'MEDIUM'},
        {'Topic':'Optimal S-vacancy concentration','Consensus':'12.5 to 17.1 percent S-vacancy is optimal','Papers agree?':'PARTIAL — DFT gives 12.5-15.6%, experiment gives 17.1%','Confidence':'MEDIUM'},
        {'Topic':'Smaller grains give more ECSA but lower conductivity','Consensus':'There is a trade-off between grain boundary density and electrical conductivity','Papers agree?':'YES — consistent across all papers, not contradictory','Confidence':'HIGH'},
    ])
    st.dataframe(consistency_df, use_container_width=True)
    st.success("Overall: All 20 papers are consistent on the key descriptors. "
               "The only discrepancies are minor — Rct normalization and a small difference "
               "between DFT and experiment for optimal vacancy concentration. "
               "Ma 2017 and Nature Communications 2020 together explain "
               "why MBE and CVD give different η at the same layer number and Mo/S ratio.")

    st.markdown("## Literature References")

    st.markdown("### Primary Papers — Foundation of the Model")
    st.markdown(
        "<div class='provenance-box'>"
        "<b>1. Jeon et al., ACS Nano 2026</b> — Training data source<br>"
        "Provides all 14 experimental data points used to train the GP model. "
        "Layer#, Mo/S ratio, ECSA, η, Tafel, Rct, TOF — all measured values come from this paper. "
        "Without this paper, the model does not exist."
        "</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='provenance-box'>"
        "<b>2. Li et al., ACS Nano 2019</b> — Stage 1/2 classification system<br>"
        "Defines the S:Mo = 1.70 threshold that separates Stage 1 (point defects) from "
        "Stage 2 (undercoordinated Mo). This threshold is used to classify every prediction "
        "the model makes. Also provides the key finding that in KOH, "
        "activity continues to increase through Stage 2."
        "</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='provenance-box'>"
        "<b>3. Yu et al., Nano Letters 2014</b> — Layer# penalty rule<br>"
        "Establishes that electron transfer rate k0 decreases 4.47x per layer added. "
        "This rule is used by the model to score synthesis method recommendations — "
        "fewer layers means MBE is more strongly recommended."
        "</div>", unsafe_allow_html=True)

    st.markdown("### Supporting Papers — What Each One Contributed")
    support_df = pd.DataFrame([
        {'Paper':'Ma et al., ACS Nano 2017','Contribution to model':'Explains why MBE gives different ECSA than CVD. MBE S-deficiency creates twin grain boundaries spontaneously. Used in Synthesis Physics page.'},
        {'Paper':'Shi et al., Nature Communications 2020','Contribution to model':'Quantifies that grain boundaries are active HER sites. GB density up to 10e12 cm-2 gives onset -25 mV. Used in Synthesis Physics causal chain.'},
        {'Paper':'Ozaki et al., ChemPhysChem 2023','Contribution to model':'Confirms S-vacancy mechanism via AP-XPS. Mo 3d shift -0.5 eV with vacancy brings free energy of adsorption to near 0 eV. Used in defect regime classification.'},
        {'Paper':'Sherwood et al., ACS Appl. Nano 2024','Contribution to model':'XPS calibration table. Confirms S-vacancies form in 2H phase, not 1T. Used to validate Mo/S ratio input.'},
        {'Paper':'ACS Catalysis 2023','Contribution to model':'CVD experiments confirm S:Mo = 1.70 threshold. Used to cross-validate the Li 2019 Stage 1/2 boundary.'},
        {'Paper':'Van Nguyen et al., Battery Energy 2023','Contribution to model':'Butler-Volmer equations and Tafel slope thresholds (Volmer: 120 mV/dec, Heyrovsky: 40 mV/dec). Used in HER mechanism classification.'},
        {'Paper':'He et al., Nanomaterials 2023','Contribution to model':'Reviews S-vacancy activation of basal plane and transient 1T phase during HER. Used in mechanism descriptions.'},
        {'Paper':'Manyepedza et al., J.Phys.Chem.C 2022','Contribution to model':'AFM measurement: 0.65 nm per layer. k0 values at 1, 2, 3 layers. Used to calibrate layer# to electron transfer rate.'},
        {'Paper':'Lee et al., ACS Nano 2010','Contribution to model':'Raman shift calibration vs number of layers. Used to validate layer# from Raman data in Jeon 2026.'},
        {'Paper':'Jaramillo et al., Science 2007','Contribution to model':'Original identification of MoS2 edge sites as active HER sites. Foundation for why grain boundaries and vacancies matter.'},
        {'Paper':'McKelvey et al., Electrochim. Acta 2021','Contribution to model':'Anchor values for k0 at specific layer numbers. Used to calibrate the 4.47x/layer rule from Yu 2014.'},
        {'Paper':'Bentley et al., Chem. Sci. 2017','Contribution to model':'vdW interlayer gap = 6.15 Angstrom for pristine MoS2. Used as reference for interlayer spacing comparison.'},
        {'Paper':'Cao et al., Sci. Rep. 2017','Contribution to model':'HRTEM measurement: 0.63 nm interlayer spacing. Cross-validates the Bentley 2017 value.'},
        {'Paper':'Smiri et al., Sci. Rep. 2026','Contribution to model':'Raman saturation data from ALD-grown MoS2. Used to cross-validate Raman vs layer# calibration.'},
        {'Paper':'ACS AMI CVD grain study','Contribution to model':'Quantifies CVD grain boundary density at approximately 0.04 per micrometer. Used in Synthesis Physics comparison.'},
        {'Paper':'ACS Catalysis 2016 (conductivity trade-off)','Contribution to model':'In-plane electron mobility is 2200x faster than out-of-plane. Used in Synthesis Physics to explain conductivity vs ECSA trade-off.'},
        {'Paper':'KOH benchmarks compilation','Contribution to model':'Reference values for η, Tafel, Rct of 8 MoS2 families in 1M KOH. Used to classify predicted performance against literature.'},
    ])
    st.dataframe(support_df, use_container_width=True)

    st.markdown('<div class="section-header">CORRECTED STAGE TABLE — v5.0 vs v4.4</div>', unsafe_allow_html=True)
    compare_df = pd.DataFrame({
        'Mo/S':[0.45,0.52,0.56,0.65,0.72,0.82],
        'S:Mo':[2.22,1.92,1.79,1.54,1.39,1.22],
        'v4.4 classification':['LOW (near-stoich)','LOW (near-stoich)','MEDIUM (point defect)','RISK (>22% vacancy!)','RISK','RISK'],
        'v5.0 classification':['PRISTINE (pre-Stage1)','Stage 1','Stage 1','Stage 2 mild (HIGH)','Stage 2 deep (HIGH)','Stage 2 extreme (RISK)'],
        'Jeon sample':['MoS-T800/N50','MoS-N20/M8','MoS-N10/T600','MoS-M3.0','MoS-M2.5','MoS-M2.0'],
        'Actual Jeon η':['-0.58V','-0.39V','-0.33V (best N)','-0.35V ✓HIGH','-0.49V','-0.58V'],
        'v4.4 correct?':['✓','✓','✓','✗ WRONG','✗ WRONG','✓'],
    })
    st.dataframe(compare_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: XPS & STAGE CALIBRATION (unchanged from v5.0)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 XPS & Stage Calibration":
    st.markdown("# XPS Calibration & Li 2019 Stage Model")
    st.markdown(
        "<div class='fix-box'>"
        "<b>v5.0 KEY CORRECTION:</b> The old 22%-vacancy 'RISK' threshold was based on H₂SO₄ data. "
        "Li et al. <i>ACS Nano</i> 2019 in KOH (0.1M) shows Stage 2 (S:Mo < 1.70) = "
        "<b>continuously increasing TOF</b> through Stage 2. "
        "Risk only at S:Mo < 0.80 (Mo/S > 0.75) where Mo-rich domains form."
        "</div>", unsafe_allow_html=True)
    st.markdown("### Li 2019 Stage 1 / Stage 2 Model (0.1M KOH — PRIMARY KOH SOURCE)")
    stage_df=pd.DataFrame([
        {'Stage':'PRISTINE (pre-Stage 1)','S:Mo range':'> 1.85','Mo/S range':'< 0.541','η trend (KOH)':'> 300 mV','Tafel (KOH)':'> 100 mV/dec','TOF trend (KOH)':'Near baseline','Rct trend':'High (> 100Ω)','Defect type':'None (stoichiometric)'},
        {'Stage':'STAGE 1 (point defects)','S:Mo range':'1.70 – 1.85','Mo/S range':'0.541 – 0.588','η trend (KOH)':'Rapid decrease','Tafel (KOH)':'110 → 80 mV/dec (fast)','TOF trend (KOH)':'Moderate increase','Rct trend':'Rapid decrease','Defect type':'Isolated S-vacancies'},
        {'Stage':'STAGE 2 mild (Li 2019)','S:Mo range':'1.33 – 1.70','Mo/S range':'0.588 – 0.752','η trend (KOH)':'Continues decreasing (slower)','Tafel (KOH)':'~80 mV/dec (plateau)','TOF trend (KOH)':'CONTINUOUS INCREASE (unlike H₂SO₄!)','Rct trend':'Saturation','Defect type':'Undercoordinated Mo regions'},
        {'Stage':'STAGE 2 deep','S:Mo range':'0.80 – 1.33','Mo/S range':'0.752 – 1.25','η trend (KOH)':'Still decreasing','Tafel (KOH)':'~80 mV/dec','TOF trend (KOH)':'High (TOF=15 s⁻¹ @ 300mV)','Rct trend':'May increase slightly','Defect type':'Extensive S-stripping'},
        {'Stage':'STAGE 2 extreme (risk)','S:Mo range':'< 0.80','Mo/S range':'> 1.25','η trend (KOH)':'Degradation','Tafel (KOH)':'Erratic','TOF trend (KOH)':'Drops (structural collapse)','Rct trend':'Very high','Defect type':'Mo-rich domains'},
    ])
    st.dataframe(stage_df,use_container_width=True)
    calib_data=[{'S/Mo (measured)':smo,'Mo/S (descriptor)':mos,
                 'Stage (Li 2019)':'PRISTINE' if smo>=1.85 else ('Stage 1' if smo>=1.70 else 'Stage 2'),
                 'Phase description':desc,'Source':source}
                for smo,(mos,desc,source) in XPS_CALIBRATION.items()]
    st.markdown("### XPS Calibration Table with Stage Assignment")
    st.dataframe(pd.DataFrame(calib_data),use_container_width=True)
    smo_vals=sorted(XPS_CALIBRATION.keys(),reverse=True)
    mos_vals=[XPS_CALIBRATION[s][0] for s in smo_vals]
    fig_calib=go.Figure()
    fig_calib.add_vrect(x0=1.70,x1=2.20,fillcolor='rgba(255,200,100,0.08)',line_width=0,annotation_text='Stage 1',annotation_position='top left')
    fig_calib.add_vrect(x0=0.0,x1=1.70,fillcolor='rgba(45,206,137,0.08)',line_width=0,annotation_text='Stage 2 (HIGH activity in KOH)',annotation_position='top right')
    fig_calib.add_trace(go.Scatter(x=smo_vals,y=mos_vals,mode='lines+markers',line=dict(color='#4E9AF1',width=2),marker=dict(size=10,color='#4E9AF1'),name='XPS calibration'))
    for _,row in df.iterrows():
        smo_jeon=1.0/row['mo_s_ratio']
        stage_c,_,_=li2019_stage(row['mo_s_ratio'])
        color={'PRISTINE':'#888','STAGE_1':'#F5A623','STAGE_2_MILD':'#2DCE89','STAGE_2_DEEP':'#4E9AF1','STAGE_2_EXTREME':'#FF6464'}.get(stage_c,'#888')
        fig_calib.add_trace(go.Scatter(x=[smo_jeon],y=[row['mo_s_ratio']],mode='markers+text',
            marker=dict(size=9,color=color,symbol='diamond',line=dict(width=1.5,color='white')),
            text=[row['sample']],textposition='top center',textfont=dict(size=8),
            name=row['sample'],showlegend=False,
            hovertemplate=f"<b>{row['sample']}</b><br>S:Mo=%{{x:.2f}}<br>Mo/S=%{{y:.3f}}<extra></extra>"))
    fig_calib.add_vline(x=1.70,line_dash='dash',line_color='#F5A623',line_width=2,annotation_text='S:Mo=1.70 — Stage 1→2 (Li 2019)',annotation_font_color='#F5A623')
    fig_calib.add_vline(x=1.33,line_dash='dot',line_color='#FF6464',line_width=1,annotation_text='S:Mo=1.33 — Stage 2 deep',annotation_font_color='#FF6464')
    fig_calib.update_layout(title="XPS Calibration: S:Mo → Mo/S | Stage 1/2 boundary (Li 2019 KOH)",
        xaxis_title="S:Mo ratio (decreasing = more Mo-rich)",yaxis_title="Mo/S ratio",
        xaxis=dict(autorange='reversed'),height=450,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_calib,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BULLETPROOF VALIDATION (unchanged from v5.0)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛡 Bulletproof Validation":
    st.markdown("# Bulletproof Validation Layer — v5.1")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Training samples","14")
    c2.metric("Papers integrated","20")
    c3.metric("Current confidence",confidence_level(layer_n,mo_s_ratio,ecsa_val,dist_val)[0])
    c4.metric("Nearest Jeon dist.",f"{dist_val:.2f}")
    st.markdown("## KOH benchmark table (complete)")
    st.dataframe(KOH_BENCHMARKS[['family','material','eta_mV','tafel','rct','stage','mechanism','note']],use_container_width=True)
    st.markdown("## Experimental uncertainty model")
    st.dataframe(EXPERIMENTAL_SD_TABLE,use_container_width=True)
    st.markdown("## Current input audit")
    vals_now=predict_all(layer_n,mo_s_ratio,ecsa_val)
    eta_now=eta_v_to_mV_abs(vals_now['eta'])
    perf_now,perf_note_now=classify_performance_eta_v5(eta_now)
    rct_label_now,rct_note_now=classify_rct_v5(vals_now['rct'])
    lit_score_now,lit_notes_now=literature_consistency_score_v5(eta_now,vals_now['tafel'],vals_now['rct'],mo_s_ratio,ecsa_val)
    sc,sl,sn=li2019_stage(mo_s_ratio)
    audit_df=pd.DataFrame([
        {'Item':'η10 magnitude','Value':f'{eta_now:.1f} mV','Interpretation':f'{perf_now}: {perf_note_now}'},
        {'Item':'Tafel','Value':f'{vals_now["tafel"]:.1f} mV/dec','Interpretation':tafel_mechanism_v5(vals_now['tafel'],mo_s_ratio)},
        {'Item':'Rct','Value':f'{vals_now["rct"]:.1f} Ω·cm²','Interpretation':f'{rct_label_now}: {rct_note_now}'},
        {'Item':'Conductivity σ','Value':f'{vals_now["conductivity"]:.4f} S/cm','Interpretation':'σ=1/ρ — higher σ means better charge transport to active sites'},
        {'Item':'Li 2019 Stage','Value':sl,'Interpretation':sn[:120]+'...'},
        {'Item':'S:Mo ratio','Value':f'{s_mo_current:.2f}','Interpretation':f'{"Stage 2 (HIGH in KOH)" if s_mo_current<1.70 else "Stage 1 or Pristine"}'},
        {'Item':'Layer penalty','Value':f'{layer_activity_factor(layer_n):.2e}','Interpretation':'Yu 2014 4.47×/layer decay'},
        {'Item':'Literature consistency','Value':f'{lit_score_now:.1f}/5','Interpretation':'Vs KOH master table benchmarks'},
    ])
    st.dataframe(audit_df,use_container_width=True)
    for note in lit_notes_now:
        st.markdown(f"<span class='validation-chip'>{note}</span>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MASTER TABLE KOH (unchanged from v5.0)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Master Table KOH":
    st.markdown("# Master Table — MoS₂ HER in KOH")
    st.markdown("<div class='correction-box'>All entries are 1M KOH unless noted. Exact Rct values (JECST, CityU, UCL) integrated in v5.0. Tafel is the most robust cross-lab descriptor.</div>",unsafe_allow_html=True)
    st.markdown("### State-of-Art MoS₂ in 1M KOH")
    sota_df=pd.DataFrame([
        {'Material':'NiO@1T-MoS2','η10 (mV)':46,'Tafel (mV/dec)':40,'Strategy':'1T metallic + NiO dopant','Mechanism':'Heyrovsky-fast'},
        {'Material':'MoS2/MXene/NF','η10 (mV)':94,'Tafel (mV/dec)':59,'Strategy':'MXene conductive heterojunction','Mechanism':'Heyrovsky'},
        {'Material':'Mo5N6-MoS2/HCNRs','η10 (mV)':100,'Tafel (mV/dec)':37.9,'Strategy':'Mott-Schottky junction','Mechanism':'Heyrovsky-fast'},
        {'Material':'MoS2/NiS','η10 (mV)':130,'Tafel (mV/dec)':52,'Strategy':'Ni heterostructure','Mechanism':'Heyrovsky'},
        {'Material':'SnO2@MoS2','η10 (mV)':127,'Tafel (mV/dec)':73,'Strategy':'SnO2 nanorod heterostructure','Mechanism':'Mixed'},
        {'Material':'CoS2-MoS2 HS','η10 (mV)':130,'Tafel (mV/dec)':66,'Strategy':'Co hollow interface','Mechanism':'Heyrovsky'},
        {'Material':'N-1T@2H MoS2','η10 (mV)':141.7,'Tafel (mV/dec)':48.4,'Strategy':'1T/2H mixed + N doping','Mechanism':'Heyrovsky'},
        {'Material':'MoS2-1T exfoliated','η10 (mV)':145,'Tafel (mV/dec)':46.2,'Strategy':'Metallic phase (unstable)','Mechanism':'Heyrovsky'},
        {'Material':'MoS2-SV (Plasma Ar)','η10 (mV)':175,'Tafel (mV/dec)':63.5,'Strategy':'S-vacancy Stage 1','Mechanism':'Mixed'},
        {'Material':'2H MoS2-7H (Li 2019)','η10 (mV)':260,'Tafel (mV/dec)':80,'Strategy':'S-vacancy Stage 2 (0.1M KOH)','Mechanism':'Heyrovsky partial'},
        {'Material':'MoS2 90nm nanosheets','η10 (mV)':280,'Tafel (mV/dec)':151,'Strategy':'Nanostructured 2H','Mechanism':'Volmer-limited'},
        {'Material':'MoS2 Bulk/Control','η10 (mV)':350,'Tafel (mV/dec)':115,'Strategy':'Reference pristine 2H','Mechanism':'Volmer-limited'},
    ])
    st.dataframe(sota_df.sort_values('η10 (mV)'),use_container_width=True)
    st.markdown("### Master Family Table — 8 MoS₂ Families in KOH")
    st.dataframe(MASTER_FAMILY_TABLE,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("# About — MoS₂ HER Trend Model v5.1")
    st.markdown("""
**v5.1** — v5.0 base + professor feedback additions only.

| Change | v5.0 | v5.1 |
|---|---|---|
| Resistivity | Shown as ρ (Ω·cm) | Replaced by conductivity σ=1/ρ (S/cm) |
| Synthesis Physics page | Not present | New page: MBE vs CVD with real literature, dynamic causal chain |
| Structural parameters | Not explicit | Full table: parameter → controls → electrochemical effect |
| Cross-paper consistency | Not shown | 10-row table checking 20 papers for consistency |
| Text color | Grey (#888, #666) | Black (#111) everywhere |
| References | "CEO" language | Neutral scientific language |
| Paper count | 16 | 20 (+ Ma 2017, Nature Comms 2020, ACS AMI, ACS Cat 2016) |

**Machine learning (unchanged from v5.0)**
- GP (Matérn ν=2.5, ARD, calibrated 95% CI) — primary predictor
- RF (300 trees, LOO) — feature importance only
- Leave-One-Out CV (n=14)
- ⚠ n=14 — use for trends and hypothesis generation, not replacement for experimental validation
    """)
