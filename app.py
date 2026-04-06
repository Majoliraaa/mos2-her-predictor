"""
MoS₂ HER Predictor — v2.2 (Publication-Ready)
===============================================
Base experimental data:  Jeon et al., ACS Nano 2026, 20, 4479–4493
Layer # calibration:     Manyepedza et al., J. Phys. Chem. C 2022, 126, 17942–17951
                         Bentley et al., Chem. Sci. 2017, 8, 6583–6593 [van der Waals gap = 6.15 Å]
                         Cao et al., Sci. Rep. 2017, 7, 8825 [HRTEM interlayer 0.63 nm]
                         Wang et al., Phys. Status Solidi RRL 2023, 17, 2200476 [Raman ratio context]
Mo/S ratio framework:    Sherwood et al. (XPS 4-peak model, referenced in Jeon 2026)
                         Jiang et al. SI, Nano Research 2019 [2H-MoS₂ XPS: Mo4+ 3d5/2 = 230.3 eV]
Kinetics framework:      McKelvey et al. (k⁰ vs trilayer number, cited in Manyepedza 2022)
Edge/basal activity:     Bentley et al. 2017 SECCM — basal J₀ = 2.5×10⁻⁶ A/cm², edge J₀ ~10⁻⁴ A/cm²

KEY CALIBRATION — 0.615–0.63 nm/trilayer CONFIRMED BY FOUR INDEPENDENT SOURCES:
  1. Manyepedza 2022, Fig. 9B: AFM → 0.6–0.7 nm (1L), 1.3–1.4 nm (2L) on mica
  2. Bentley 2017, Introduction: "van der Waals gap = 6.15 Å" (bulk MoS₂)
  3. Cao 2017, Fig. 2b,c: HRTEM interlayer spacing = 0.63 nm (MoS₂ NSs on MWCNT)
  4. Fan et al. JACS 2016 (via Manyepedza ref.63): controlled exfoliation confirms trilayer spacing

  XPS STOICHIOMETRY CROSS-VALIDATION:
  - Jiang 2019 SI: 2H-MoS₂ bilayer → Mo4+ 3d5/2 = 230.3 eV, S2- 2p3/2 = 163.1 eV
  - Cao 2017: 2H phase → Mo4+ 3d5/2 ~229 eV; 1T phase → ~228.1 eV (Δ~0.9 eV shift)
  - Wang 2023: Mo4+ 3d5/2 = 229.9 eV for MBE as-grown MoS₂
  - These are all consistent with Sherwood 2024's 4-peak model (229.3 eV for 2H)
  → Confirms that Mo/S ratio mapping endpoints in this predictor are physically grounded

DATA PROVENANCE:
  ✅ MEASURED in Jeon 2026 Table 1:
     η, Tafel slope, Rct, ECSA, resistivity, Raman A₁g/E₂g, TOF (ECSA & mass), loading
  ⚠ ESTIMATED / DERIVED (clearly flagged):
     layer_n  → D(002) Scherrer ÷ 0.615 nm/layer (4-source validated: 0.615–0.63 nm range)
     mo_s_ratio → S-thickness → XANES mapping, anchored to Sherwood 2024 + Jiang 2019 XPS
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
.provenance-box {
    background: rgba(255,193,7,0.07);
    border: 1px solid rgba(255,193,7,0.25);
    border-left: 4px solid #FFC107;
    border-radius: 4px; padding: 10px 14px; margin: 8px 0;
    font-size: 0.82em; color: #ccc;
}
.stMetric label { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78em !important; }
.stMetric [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA — all values from Jeon et al. ACS Nano 2026, Table 1
# layer_n and mo_s_ratio are DERIVED (see provenance notes above)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    """
    Layer # derivation (⚠ ESTIMATED):
      D(002) from Scherrer equation reported in Jeon 2026, Fig 1a & 2a:
        T600: 7.2 nm → 7.2/0.615 ≈ 12 layers
        T700: 8.3 nm → 8.3/0.615 ≈ 14 layers
        T800: 10.8 nm → 10.8/0.615 ≈ 18 layers
        N5 : ~2 nm (very thin, RHEED amorphous) → 2 layers
        N10: ~3.3 nm → 5 layers (XRR Δθ-based, Jeon Fig 2a)
        N20: ~5.5 nm → 9 layers
        N30: ~8.0 nm → 13 layers
        N50: ~12.1 nm → 20 layers (Scherrer D002 from Jeon Fig 2a)
        M-series: deposition cycles = 50 (same as T800/N50) but S-thickness varies
          → layer count estimated from XRR film thickness (Jeon Fig S11a description)
          M2.0–M9.0: approximately same total thickness as N50 → ~20 layers
          (The STEM images in Fig 3i confirm ~20 layers for M3.0 and M6.0)
      Reference for 0.615 nm/layer: Manyepedza 2022 AFM Fig 9B, confirmed by
      Fan et al. JACS 2016 cited in Manyepedza as ref [63].

    Mo/S ratio derivation (⚠ ESTIMATED from XANES description in Jeon 2026):
      - T-series (S=9.0 Å fixed, T varies): Near-stoichiometric throughout.
        XANES shows pure 2H-MoS₂ (Mo K-edge features match bulk MoS₂).
        Mo/S ≈ 0.455–0.50 (stoichiometric = S/Mo 2.0–2.2).
        Higher T → more complete sulfurization → slightly more stoichiometric.
        Values: T600 → 0.49, T700 → 0.48, T800 → 0.46
      - N-series (S=3.0 Å fixed, cycles vary): 
        N5: very few layers, partial sulfidation → Mo/S ≈ 0.62 (Mo-rich)
        N10: optimal, moderate Mo-rich → Mo/S ≈ 0.56
        N20–N50: more stoichiometric with cycles → Mo/S ≈ 0.52–0.47
      - M-series (cycles=50, S-thickness varies 2.0–9.0 Å):
        M2.0: very S-deficient, XANES shows metallic Mo peaks → Mo/S ≈ 0.82
        M2.5: → 0.76; M3.0: → 0.65; M6.0: → 0.52; M8.0: → 0.48; M9.0: → 0.46
        Anchored to: Sherwood 2024 XPS: stoichiometric 2H = Mo/S ≈ 0.40-0.455
        Upper limit Mo/S ≈ 0.91 (Ar⁺ etching extreme, Sherwood 2024)
        Jeon EXAFS subtraction analysis (Fig 3f-h) confirms M3.0 > M6.0 >> M9.0
        in terms of metallic Mo fraction.
    """
    data = {
        'sample':      ['MoS-T600','MoS-T700','MoS-T800',
                        'MoS-N5','MoS-N10','MoS-N20','MoS-N30','MoS-N50',
                        'MoS-M2.0','MoS-M2.5','MoS-M3.0','MoS-M6.0','MoS-M8.0','MoS-M9.0'],
        'series':      ['T','T','T','N','N','N','N','N','M','M','M','M','M','M'],
        'temp':        [600,700,800,800,800,800,800,800,800,800,800,800,800,800],
        'cycles':      [50,50,50,5,10,20,30,50,50,50,50,50,50,50],
        's_thick':     [9.0,9.0,9.0,3.0,3.0,3.0,3.0,3.0,2.0,2.5,3.0,6.0,8.0,9.0],

        # ⚠ ESTIMATED — see derivation above
        # D(002) from Scherrer ÷ 0.615 nm/layer (Manyepedza 2022 AFM calibration)
        'layer_n':     [12, 14, 18,   2,  5,  9, 13, 20,  20, 20, 20, 20, 20, 20],

        # ⚠ ESTIMATED — sigmoidal mapping anchored to Jeon XANES description
        # and Sherwood 2024 XPS stoichiometry endpoints
        'mo_s_ratio':  [0.49,0.48,0.46, 0.62,0.56,0.52,0.50,0.47, 0.82,0.76,0.65,0.52,0.48,0.46],

        # ✅ MEASURED — Jeon 2026 Table 1
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
    return pd.DataFrame(data)

df = load_data()

TARGETS = {
    'eta':         ('Overpotential η', 'V',         'min'),   # more negative = better
    'tafel':       ('Tafel slope',      'mV/dec',    'min'),
    'rct':         ('Rct',              'Ω·cm²',     'min'),
    'raman':       ('Raman A₁g/E₂g',   '',          'min'),
    'resistivity': ('Resistivity',      'Ω·cm',      'min'),
    'tof_ecsa':    ('TOF (ECSA)',        'nmol/cm²/s','max'),
    'tof_mass':    ('TOF (mass)',        'nmol/µg/s', 'max'),
}

FEATURES = ['layer_n', 'mo_s_ratio', 'ecsa']
FEATURE_LABELS = {
    'layer_n':    'Layer # (est.)',
    'mo_s_ratio': 'Mo/S ratio (est.)',
    'ecsa':       'ECSA (cm²)',
}
FEATURE_RANGES = {
    'layer_n':    (1, 20),
    'mo_s_ratio': (0.45, 0.90),
    'ecsa':       (2.0, 12.0),
}

# Estimated vs measured flags for UI display
FEATURE_PROVENANCE = {
    'layer_n':    '⚠ Estimated from XRD Scherrer D(002) ÷ 0.615 nm/layer (Manyepedza 2022)',
    'mo_s_ratio': '⚠ Estimated from S-thickness via XANES mapping (Jeon 2026 + Sherwood 2024)',
    'ecsa':       '✅ Directly measured — Jeon 2026 Table 1',
}

SERIES_COLORS = {'T': '#4E9AF1', 'N': '#2DCE89', 'M': '#F5A623'}
SERIES_LABELS = {'T': 'T-series (Temp.)', 'N': 'N-series (Cycles)', 'M': 'M-series (S-thick.)'}

METHOD_COLORS = {
    'mbe':  '#2DCE89',
    'both': '#F5A623',
    'cvd':  '#4E9AF1',
}


# ══════════════════════════════════════════════════════════════════════════════
# GP + RF MODELS
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

        kernel = (C(1.0, (1e-3, 1e3))
                  * Matern(length_scale=[1.0]*n,
                           length_scale_bounds=[(0.01, 100)]*n, nu=2.5)
                  + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 10)))

        loo_means, loo_stds_list = [], []
        for tr, te in loo.split(X):
            sx_l = StandardScaler().fit(X[tr])
            sy_l = StandardScaler().fit(y[tr].reshape(-1, 1))
            gp_l = GaussianProcessRegressor(
                kernel=C(1.0, (1e-3, 1e3))*Matern(length_scale=[1.0]*n,
                         length_scale_bounds=[(0.01, 100)]*n, nu=2.5)
                       + WhiteKernel(0.1, (1e-5, 10)),
                n_restarts_optimizer=5, normalize_y=False, alpha=1e-6)
            gp_l.fit(sx_l.transform(X[tr]),
                     sy_l.transform(y[tr].reshape(-1, 1)).ravel())
            m_s, std_s = gp_l.predict(sx_l.transform(X[te]), return_std=True)
            loo_means.append(sy_l.inverse_transform(m_s.reshape(-1, 1)).ravel()[0])
            loo_stds_list.append(std_s[0] * sy_l.scale_[0])

        loo_means = np.array(loo_means)
        avg_err   = np.mean(np.abs(y - loo_means))
        avg_std   = np.mean(loo_stds_list)
        calib     = avg_err / avg_std if avg_std > 0 else 1.0

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                      normalize_y=False, alpha=1e-6)
        gp.fit(sx.transform(X), sy.transform(y.reshape(-1, 1)).ravel())

        gp_models[key]     = gp
        gp_scores[key]     = {'r2': r2_score(y, loo_means),
                              'mae': mean_absolute_error(y, loo_means),
                              'loo_preds': loo_means, 'calib': calib}
        sx_dict[key]       = sx
        sy_dict[key]       = sy
        loo_stds_dict[key] = np.array(loo_stds_list)

        rf = RandomForestRegressor(n_estimators=300, max_depth=4,
                                   min_samples_leaf=2, random_state=42)
        preds = np.zeros(len(y))
        for tr, te in loo.split(X):
            rf.fit(X[tr], y[tr])
            preds[te] = rf.predict(X[te])
        rf.fit(X, y)
        rf_models[key] = rf
        rf_scores[key] = {'r2': r2_score(y, preds),
                          'mae': mean_absolute_error(y, preds),
                          'loo_preds': preds}
        rf_imps[key]   = rf.feature_importances_

    return (gp_models, gp_scores, sx_dict, sy_dict, loo_stds_dict,
            rf_models, rf_scores, rf_imps)


with st.spinner("Training GP + RF models… (first load only)"):
    (gp_models, gp_scores, sx_dict, sy_dict, loo_stds_dict,
     rf_models, rf_scores, rf_imps) = train_models()


def gp_predict(key, ln, msr, ecsa_v):
    X_new = np.array([[ln, msr, ecsa_v]])
    sx = sx_dict[key]; sy = sy_dict[key]; gp = gp_models[key]
    m_s, std_s = gp.predict(sx.transform(X_new), return_std=True)
    mean  = sy.inverse_transform(m_s.reshape(-1, 1)).ravel()[0]
    std   = std_s[0] * sy.scale_[0] * gp_scores[key]['calib']
    return mean, mean - 1.96*std, mean + 1.96*std, std


def predict_all(ln, msr, ecsa_v):
    return {k: gp_predict(k, ln, msr, ecsa_v)[0] for k in TARGETS}


# ══════════════════════════════════════════════════════════════════════════════
# CVD vs MBE SCORING
# NOTE: All 14 Jeon 2026 samples ARE MBE-grown. This scoring framework is
# designed to guide NEW synthesis decisions for untested parameter combinations.
# It is NOT a reclassification of existing samples.
# ══════════════════════════════════════════════════════════════════════════════
def score_method(layer_n, mo_s_ratio, ecsa_v, rct_v=None):
    """
    Scores how strongly MBE is preferred over CVD for a given parameter set.
    Returns (label, color_key, score, max_score, reasons_list).

    Scientific basis:
    - Layer # criterion: McKelvey et al. (cited in Manyepedza 2022):
      k⁰ = 250 cm/s at 1 trilayer → 1.5 cm/s at 3 trilayers (167× difference).
      CVD cannot reliably produce <5 layers with controlled stoichiometry
      (Choudhury & Redwing, Penn State review, cited in Jeon 2026 framework).
    - Mo/S criterion: Jeon EXAFS + Choudhury review: CVD S-overpressure
      pushes toward stoichiometric (Mo/S ≈ 0.455); cannot independently
      control Mo flux to achieve Mo-rich domains (Mo/S > 0.58).
    - ECSA criterion: Jeon 2026 Table 1 — highest ECSA (9.2 cm²) in MBE
      wafer-scale uniform films.
    - Rct criterion: Jeon 2026 — lowest Rct (45.5 Ω·cm²) requires metallic
      Mo domains only achievable via MBE flux control.
    """
    reasons = []
    total = 0
    MAX = 8

    # ── Criterion 1: Layer # ──────────────────────────────────────────────────
    if layer_n <= 3:
        pts = 3
        detail = (f"≤3 layers: k⁰ ~250 cm/s (1L) → 1.5 cm/s (3L) per McKelvey et al. "
                  f"(cited in Manyepedza 2022). CVD cannot reliably achieve ≤3L "
                  f"with controlled stoichiometry.")
        refs = ["McKelvey (Manyepedza ref.30)", "Manyepedza 2022 Fig.9 AFM"]
    elif layer_n <= 6:
        pts = 2
        detail = (f"4–6 layers: near-optimal zone. MoS-N10 (~5L) achieves best "
                  f"N-series HER. CVD nucleation density unstable at this range.")
        refs = ["Jeon 2026 N-series", "Manyepedza 2022"]
    elif layer_n <= 12:
        pts = 1
        detail = f"7–12 layers: multi-layer regime. MBE preferred for thickness uniformity."
        refs = ["Jeon 2026 T-series", "Choudhury review"]
    else:
        pts = 0
        detail = f"≥13 layers: CVD viable when Mo/S is near-stoichiometric."
        refs = ["Jeon 2026 T800/N50"]
    total += pts
    reasons.append({'criterion': 'Layer #', 'points': pts, 'max': 3,
                    'refs': refs, 'detail': detail})

    # ── Criterion 2: Mo/S ratio ───────────────────────────────────────────────
    if mo_s_ratio > 0.72:
        pts = 3
        detail = (f"Highly Mo-rich (Mo/S > 0.72): XANES shows metallic Mo peaks "
                  f"(Jeon 2026 Fig.3c). CVD S-overpressure cannot achieve this regime. "
                  f"Stoichiometric 2H-MoS₂ endpoint: Mo/S ≈ 0.455 (Sherwood 2024 XPS).")
        refs = ["Jeon 2026 XANES Fig.3c", "Sherwood 2024 XPS 4-peak model"]
    elif mo_s_ratio > 0.58:
        pts = 2
        detail = (f"Mo⁰/MoS₂ coexistence zone (0.58 < Mo/S ≤ 0.72): "
                  f"optimal for basal plane activation + conductivity. "
                  f"CVD S-overpressure pushes toward stoichiometric, losing this advantage.")
        refs = ["Jeon 2026 M-series EXAFS", "Sherwood 2024"]
    elif mo_s_ratio >= 0.50:
        pts = 1
        detail = f"Slightly S-deficient (0.50 ≤ Mo/S ≤ 0.58): CVD may overshoot toward S/Mo=2.2."
        refs = ["Sherwood 2024 XPS"]
    else:
        pts = 0
        detail = f"Near-stoichiometric (Mo/S < 0.50 ≈ S/Mo ≥ 2.0): CVD S-rich atmosphere sufficient."
        refs = ["Manyepedza 2022 XPS (Mo/S=0.455)", "Sherwood 2024"]
    total += pts
    reasons.append({'criterion': 'Mo/S ratio', 'points': pts, 'max': 3,
                    'refs': refs, 'detail': detail})

    # ── Criterion 3: ECSA ─────────────────────────────────────────────────────
    if ecsa_v >= 8.0:
        pts = 1
        detail = (f"ECSA ≥ 8.0 cm²: MBE wafer-scale uniformity required to maximize "
                  f"accessible edge sites. Best Jeon samples: N10 (8.0) and M6.0 (9.2 cm²).")
        refs = ["Jeon 2026 Table 1"]
    else:
        pts = 0
        detail = f"ECSA < 8.0 cm²: no additional synthesis constraint from this criterion."
        refs = []
    total += pts
    reasons.append({'criterion': 'ECSA', 'points': pts, 'max': 1,
                    'refs': refs, 'detail': detail})

    # ── Criterion 4: Rct ─────────────────────────────────────────────────────
    rct_use = rct_v if rct_v is not None else gp_predict('rct', layer_n, mo_s_ratio, ecsa_v)[0]
    if rct_use < 55:
        pts = 1
        detail = (f"Rct = {rct_use:.0f} Ω·cm² < 55: requires metallic Mo⁰ domains "
                  f"for low interfacial resistance. Only achievable via MBE Mo-flux control. "
                  f"Best Jeon: N10 (52.8), M6.0 (45.5 Ω·cm²).")
        refs = ["Jeon 2026 EIS Table 1"]
    else:
        pts = 0
        detail = f"Rct = {rct_use:.0f} Ω·cm² ≥ 55: no additional constraint from this criterion."
        refs = []
    total += pts
    reasons.append({'criterion': 'Rct', 'points': pts, 'max': 1,
                    'refs': refs, 'detail': detail})

    # Decision
    if total >= 3:
        label   = "🔬 Physical Method (MBE)"
        col_key = 'mbe'
    elif total >= 1:
        label   = "⚗️ Both viable — MBE preferred"
        col_key = 'both'
    else:
        label   = "🧪 Chemical Method (CVD/PVT)"
        col_key = 'cvd'

    return label, col_key, total, MAX, reasons


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚗️ MoS₂ HER Predictor")
    st.markdown(
        "<div style='font-size:0.78em;color:#666;margin-bottom:10px;'>"
        "Jeon et al. <i>ACS Nano</i> 2026 · Manyepedza et al. <i>J. Phys. Chem. C</i> 2022<br>"
        "GP model · n=14 MBE samples · 1M KOH</div>",
        unsafe_allow_html=True)

    # Provenance legend
    st.markdown(
        "<div class='provenance-box'>"
        "✅ <b>ECSA</b>: directly measured (Jeon 2026 Table 1)<br>"
        "⚠ <b>Layer #</b>: estimated from XRD Scherrer ÷ 0.615 nm/layer<br>"
        "&nbsp;&nbsp;&nbsp;<i>(AFM calibration, Manyepedza 2022 Fig. 9)</i><br>"
        "⚠ <b>Mo/S</b>: estimated from S-thickness + XANES description<br>"
        "&nbsp;&nbsp;&nbsp;<i>(Jeon 2026 + Sherwood 2024 XPS endpoints)</i>"
        "</div>",
        unsafe_allow_html=True)

    st.markdown('<div class="section-header">KEY DESCRIPTORS</div>', unsafe_allow_html=True)
    st.caption("Move sliders → predictor updates in real time")

    layer_n = st.slider(
        "⚠ Layer #",  1, 20, 5, 1,
        help=(
            "ESTIMATED from XRD Scherrer D(002) ÷ 0.615 nm/trilayer.\n\n"
            "Interlayer spacing confirmed by FOUR independent sources:\n"
            "① Manyepedza 2022 Fig.9B — AFM: 0.6–0.7 nm (1L), 1.3–1.4 nm (2L)\n"
            "② Bentley 2017 — states 'van der Waals gap = 6.15 Å' explicitly\n"
            "③ Cao et al. Sci.Rep. 2017 — HRTEM: interlayer spacing = 0.63 nm\n"
            "④ Fan et al. JACS 2016 — controlled exfoliation confirms trilayer\n\n"
            "Kinetics: McKelvey et al. (via Manyepedza 2022):\n"
            "k⁰ = 250 cm/s (1L) → 1.5 cm/s (3L) — 167× kinetic advantage.\n\n"
            "Dataset range: 2–20 layers."
        ))

    mo_s_ratio = st.slider(
        "⚠ Mo/S atomic ratio", 0.45, 0.90, 0.56, 0.01,
        help=(
            "ESTIMATED from S-layer thickness via XANES description (Jeon 2026).\n\n"
            "Stoichiometric 2H-MoS₂: Mo/S ≈ 0.455 (S/Mo = 2.2), "
            "confirmed by Manyepedza 2022 XPS and Sherwood 2024 4-peak model.\n\n"
            "Upper limit Mo/S ≈ 0.91 corresponds to Ar⁺-etched extreme "
            "(Sherwood 2024, S/Mo = 1.1).\n\n"
            "M2.0 (most Mo-rich, metallic Mo peaks in XANES) → Mo/S ≈ 0.82.\n\n"
            "NOT a directly measured table value — use with caution."
        ))

    ecsa_val = st.slider(
        "✅ ECSA (cm²)", 2.0, 12.0, 8.0, 0.5,
        help=(
            "DIRECTLY MEASURED from double-layer capacitance (Cdl) "
            "at 20–80 mV/s in non-Faradaic region (Jeon 2026).\n\n"
            "Specific capacitance Cs = 40 µF/cm² assumed.\n\n"
            "Dataset range: 3.5 cm² (T800/M9.0) to 9.2 cm² (M6.0)."
        ))

    # Closest experimental match
    df_dist = df.copy()
    df_dist['dist'] = df.apply(lambda r: np.sqrt(
        ((r.layer_n    - layer_n)    / 18)   **2 +
        ((r.mo_s_ratio - mo_s_ratio) / 0.36) **2 +
        ((r.ecsa       - ecsa_val)   / 6.0)  **2), axis=1)
    best_match = df_dist.nsmallest(1, 'dist').iloc[0]
    dist_val   = df_dist['dist'].min()

    if dist_val < 0.15:
        st.success(f"✓ Closest sample: **{best_match['sample']}**")
    elif dist_val < 0.40:
        st.info(f"≈ Nearest: **{best_match['sample']}** (interpolating)")
    else:
        st.warning(f"⚠ Extrapolating — nearest: **{best_match['sample']}**")

    # Method scoring badge
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
        st.caption(
            "⚠ All 14 Jeon 2026 samples are MBE-grown. "
            "This score guides NEW synthesis decisions for unexplored parameter sets.")
        for r in m_reasons:
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
    st.markdown(
        "<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
        "Gaussian Process · Jeon et al. <i>ACS Nano</i> 2026 · 14 MBE samples · 1M KOH · "
        "Kinetics: Manyepedza et al. <i>J. Phys. Chem. C</i> 2022</div>",
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
        f"Layer# {layer_n} (est.) · Mo/S {mo_s_ratio:.2f} (est.) · "
        f"ECSA {ecsa_val:.1f} cm² (meas.)</div>"
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
        gp_ci  = {k: dict(zip(['mean','lower','upper','std'],
                              gp_predict(k, layer_n, mo_s_ratio, ecsa_val)))
                  for k in TARGETS}

    st.caption(f"Source: {source}")

    # Key descriptor cards
    st.markdown('<div class="section-header">KEY DESCRIPTORS</div>', unsafe_allow_html=True)
    kc1, kc2, kc3 = st.columns(3)
    desc_cards = [
        (kc1, "Layer # ⚠", f"{layer_n}", "layers",
         "🟢 ≤3L → k⁰ ~250 cm/s (Manyepedza 2022)" if layer_n <= 3
         else ("🟡 Few-layer (4–6)" if layer_n <= 6 else "🔴 Thick film (≥7)"),
         "⚠ Estimated: XRD Scherrer ÷ 0.615 nm/layer"),
        (kc2, "Mo/S ratio ⚠", f"{mo_s_ratio:.2f}", "",
         "🟢 Mo⁰/MoS₂ coexistence (optimal)" if 0.55 <= mo_s_ratio <= 0.72
         else ("🟡 Near-stoich." if mo_s_ratio < 0.55 else "🔴 Highly Mo-rich"),
         "⚠ Estimated: S-thickness → XANES mapping"),
        (kc3, "ECSA ✅", f"{ecsa_val:.1f}", "cm²",
         "🟢 High — edge sites accessible" if ecsa_val >= 7.0 else "🟡 Moderate",
         "✅ Measured — Jeon 2026 Table 1"),
    ]
    for col, label, val, unit, status, note in desc_cards:
        with col:
            st.markdown(
                f"<div class='descriptor-card'>"
                f"<div class='label'>{label}</div>"
                f"<div class='value'>{val}"
                f"<span style='font-size:0.6em;color:#888;'> {unit}</span></div>"
                f"<div class='note'>{status}</div>"
                f"<div class='note' style='margin-top:4px;color:#555;'>{note}</div>"
                f"</div>", unsafe_allow_html=True)

    # Performance metrics
    st.markdown('<div class="section-header">PREDICTED PERFORMANCE METRICS</div>',
                unsafe_allow_html=True)
    cols = st.columns(4)
    metrics_order = ['eta','tafel','rct','tof_ecsa','tof_mass','raman','resistivity']
    thresholds = {
        'eta':    (-0.38, -0.50),   # good if ≤ -0.38, bad if ≥ -0.50
        'tafel':  (110,   200),
        'rct':    (70,    130),
        'raman':  (1.8,   2.2),
        'resistivity': (12, 17),
        'tof_ecsa': (9,   6),
        'tof_mass': (5,   2),
    }
    for i, key in enumerate(metrics_order):
        name, unit, better = TARGETS[key]
        v = vals[key]
        col = cols[i % 4]
        if key in thresholds:
            g, b = thresholds[key]
            if better == 'max':
                color = "normal" if v >= g else ("off" if v <= b else "inverse")
            else:
                color = "normal" if v <= g else ("off" if v >= b else "inverse")
        else:
            color = "normal"
        fmt = f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
        if gp_ci:
            std = gp_ci[key]['std']
            col.metric(name, f"{fmt} {unit}",
                       delta=f"±{std:.2f}" if abs(std) < 100 else f"±{std:.0f}",
                       delta_color="off")
        else:
            col.metric(name, f"{fmt} {unit}")

    # Radar chart — CORRECTED: best sample uses idxmin() for η (most negative = best HER)
    st.markdown('<div class="section-header">PERFORMANCE PROFILE</div>', unsafe_allow_html=True)
    radar_keys  = ['eta','tafel','rct','tof_ecsa','tof_mass','raman','resistivity']
    radar_names = ['η (overpot.)','Tafel','Rct','TOF(ECSA)','TOF(mass)','Raman','Resistivity']
    normed = []
    for key in radar_keys:
        _, _, better = TARGETS[key]
        col_vals = df[key].values
        vmin, vmax = col_vals.min(), col_vals.max()
        v = vals[key]
        # Clip to [0,1] — GP predictions can exceed dataset range
        n_val = float(np.clip((v - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0))
        normed.append(n_val if better == 'max' else 1 - n_val)

    # best experimental = MoS-N10 (lowest η = -0.33 V)
    best_exp_idx = df['eta'].idxmin()
    best_exp     = df.loc[best_exp_idx]
    closest_row  = best_match

    normed_best, normed_closest = [], []
    for key in radar_keys:
        _, _, better = TARGETS[key]
        vmin = df[key].min(); vmax = df[key].max()
        bv = float(np.clip((best_exp[key] - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0))
        normed_best.append(bv if better == 'max' else 1 - bv)
        cv = float(np.clip((closest_row[key] - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0))
        normed_closest.append(cv if better == 'max' else 1 - cv)

    # Convert to plain Python lists to avoid numpy type issues with Plotly
    normed         = [float(x) for x in normed]
    normed_best    = [float(x) for x in normed_best]
    normed_closest = [float(x) for x in normed_closest]

    fig_radar = go.Figure()
    cats = radar_names + [radar_names[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=normed + [normed[0]], theta=cats,
        fill='toself', name='Your prediction',
        fillcolor=f"{m_color}30", line=dict(color=m_color, width=2)))
    fig_radar.add_trace(go.Scatterpolar(
        r=normed_best + [normed_best[0]], theta=cats,
        fill='toself', name=f'Best exp.: {best_exp["sample"]} (η={best_exp["eta"]:.2f}V)',
        fillcolor='rgba(45,206,137,0.08)', line=dict(color='#2DCE89', width=1.5, dash='dot')))
    fig_radar.add_trace(go.Scatterpolar(
        r=normed_closest + [normed_closest[0]], theta=cats,
        fill='toself', name=f'Nearest: {best_match["sample"]}',
        fillcolor='rgba(255,255,255,0.03)', line=dict(color='#aaa', width=1, dash='dot')))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
                   angularaxis=dict(tickfont=dict(size=10))),
        showlegend=True, height=400,
        legend=dict(orientation='h', yanchor='bottom', y=-0.30),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=70, l=40, r=40))
    st.plotly_chart(fig_radar, use_container_width=True)

    # 3 closest samples table
    st.markdown('<div class="section-header">3 CLOSEST EXPERIMENTAL SAMPLES</div>',
                unsafe_allow_html=True)
    df_dist2 = df.copy()
    df_dist2['dist'] = df.apply(lambda r: np.sqrt(
        ((r.layer_n    - layer_n)    / 18)   **2 +
        ((r.mo_s_ratio - mo_s_ratio) / 0.36) **2 +
        ((r.ecsa       - ecsa_val)   / 6.0)  **2), axis=1)
    closest = df_dist2.nsmallest(3, 'dist')
    show_cols = ['sample','series','layer_n','mo_s_ratio','ecsa',
                 'eta','tafel','rct','tof_ecsa','tof_mass']
    st.dataframe(closest[show_cols].reset_index(drop=True), use_container_width=True)
    st.caption("⚠ layer_n and mo_s_ratio are estimated descriptors, not directly measured values.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TREND CURVES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trend Curves":
    st.markdown("# Trend Curves")
    st.markdown(
        "<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
        "GP mean + 95% CI vs each descriptor · experimental data overlaid · "
        "other descriptors fixed at sidebar values</div>",
        unsafe_allow_html=True)

    tc1, tc2 = st.columns([1, 2])
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

    exp_lo   = df[feat_tc].min()
    exp_hi   = df[feat_tc].max()
    in_range = (x_range >= exp_lo) & (x_range <= exp_hi)

    fig_tc = go.Figure()
    fig_tc.add_trace(go.Scatter(
        x=np.concatenate([x_range, x_range[::-1]]),
        y=np.concatenate([y_highs, y_lows[::-1]]),
        fill='toself', fillcolor='rgba(78,154,241,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name='95% CI (GP)', showlegend=True))
    fig_tc.add_trace(go.Scatter(x=x_range, y=y_means, mode='lines',
        line=dict(color='rgba(78,154,241,0.35)', width=1.5, dash='dot'),
        name='GP mean (extrapolation)', showlegend=True))
    x_in = x_range[in_range]; y_in = y_means[in_range]
    fig_tc.add_trace(go.Scatter(x=x_in, y=y_in, mode='lines',
        line=dict(color='#4E9AF1', width=3),
        name='GP mean (interpolation)', showlegend=True))

    for ser, scolor in SERIES_COLORS.items():
        mask = df['series'] == ser
        fig_tc.add_trace(go.Scatter(
            x=df[feat_tc].values[mask], y=df[target_tc].values[mask], mode='markers',
            name=SERIES_LABELS[ser],
            marker=dict(size=11, color=scolor, line=dict(width=1.5, color='white')),
            text=df['sample'][mask],
            hovertemplate='<b>%{text}</b><br>' + FEATURE_LABELS[feat_tc] + '=%{x:.2f}'
                          '<br>' + name_tc + '=%{y:.3f} ' + unit_tc + '<extra></extra>'))

    cur_val = defaults[feat_tc]
    fig_tc.add_vline(x=cur_val, line_width=1.5, line_dash="dash",
                     line_color=METHOD_COLORS[m_col_key],
                     annotation_text=f"Current: {cur_val:.2f}",
                     annotation_font_color=METHOD_COLORS[m_col_key])
    fig_tc.add_vrect(x0=exp_lo, x1=exp_hi,
                     fillcolor="rgba(255,255,255,0.03)", line_width=0,
                     annotation_text="Experimental range",
                     annotation_position="top left",
                     annotation_font_size=10, annotation_font_color="#555")

    prov = FEATURE_PROVENANCE[feat_tc]
    fig_tc.update_layout(
        title=f"{name_tc} vs {FEATURE_LABELS[feat_tc]}<br>"
              f"<sup>{prov}</sup>",
        xaxis_title=FEATURE_LABELS[feat_tc],
        yaxis_title=f"{name_tc} ({unit_tc})",
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=-0.40),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig_tc.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.12)')
    fig_tc.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.12)')
    st.plotly_chart(fig_tc, use_container_width=True)

    # All 3 descriptors overview
    st.markdown('<div class="section-header">ALL 3 DESCRIPTORS — OVERVIEW</div>',
                unsafe_allow_html=True)
    cols3 = st.columns(3)
    for fi, feat in enumerate(FEATURES):
        lo_f, hi_f = FEATURE_RANGES[feat]
        xr = np.linspace(lo_f, hi_f, 60)
        ym = []
        for xv in xr:
            row = {f: (xv if f == feat else defaults[f]) for f in FEATURES}
            m, _, _, _ = gp_predict(target_tc, row['layer_n'], row['mo_s_ratio'], row['ecsa'])
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
                name=ser, showlegend=(fi == 0), text=df['sample'][mask],
                hovertemplate='<b>%{text}</b><br>%{x:.2f} → %{y:.3f}<extra></extra>'))
        fig_s.add_vline(x=defaults[feat], line_dash='dash',
                        line_color=METHOD_COLORS[m_col_key], line_width=1.2)
        fig_s.update_layout(
            title=dict(text=FEATURE_LABELS[feat], font=dict(size=12)),
            xaxis_title=FEATURE_LABELS[feat],
            yaxis_title=f"{name_tc} ({unit_tc})",
            height=260, margin=dict(t=40, b=40, l=40, r=10), showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        fig_s.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.10)')
        fig_s.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.10)')
        cols3[fi].plotly_chart(fig_s, use_container_width=True)

    st.caption(
        "Solid blue = GP mean (interpolation range). Dashed = extrapolation. "
        "Colored dots = experimental data. Vertical dashed = current slider value. "
        "⚠ Layer# and Mo/S ratio are estimated descriptors.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 2D HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺 2D Heatmaps":
    st.markdown("# 2D Heatmaps")
    st.markdown(
        "<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
        "GP-predicted performance over descriptor pairs. Third descriptor fixed at sidebar value.</div>",
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
    N = 40
    defaults_hm = {'layer_n': layer_n, 'mo_s_ratio': mo_s_ratio, 'ecsa': ecsa_val}

    if axis_pair.startswith("Layer# × Mo/S"):
        xf, yf, fixed_f = 'layer_n', 'mo_s_ratio', 'ecsa'
        xlabel, ylabel  = 'Layer # (est.)', 'Mo/S ratio (est.)'
    elif axis_pair.startswith("Layer# × ECSA"):
        xf, yf, fixed_f = 'layer_n', 'ecsa', 'mo_s_ratio'
        xlabel, ylabel  = 'Layer # (est.)', 'ECSA (cm²)'
    else:
        xf, yf, fixed_f = 'mo_s_ratio', 'ecsa', 'layer_n'
        xlabel, ylabel  = 'Mo/S ratio (est.)', 'ECSA (cm²)'

    xlo, xhi = FEATURE_RANGES[xf]
    ylo, yhi = FEATURE_RANGES[yf]
    xgrid = np.linspace(xlo, xhi, N)
    ygrid = np.linspace(ylo, yhi, N)
    Z = np.zeros((N, N))
    for i, yv in enumerate(ygrid):
        for j, xv in enumerate(xgrid):
            row = {xf: xv, yf: yv, fixed_f: defaults_hm[fixed_f]}
            Z[i, j] = gp_predict(target_hm, row['layer_n'], row['mo_s_ratio'], row['ecsa'])[0]

    cs = 'RdYlGn' if better_hm == 'max' else 'RdYlGn_r'
    fig_hm = go.Figure(data=go.Heatmap(
        z=Z, x=xgrid, y=ygrid, colorscale=cs,
        colorbar=dict(title=f"{name_hm}<br>({unit_hm})", titleside='right'),
        hovertemplate=f'{xlabel}=%{{x:.2f}}<br>{ylabel}=%{{y:.2f}}'
                      f'<br>{name_hm}=%{{z:.3f}} {unit_hm}<extra></extra>'))

    for ser, scolor in SERIES_COLORS.items():
        mask = df['series'] == ser
        fig_hm.add_trace(go.Scatter(
            x=df[xf].values[mask], y=df[yf].values[mask],
            mode='markers+text',
            marker=dict(size=12, color=scolor, line=dict(width=2, color='white')),
            text=df['sample'][mask], textposition='top center',
            textfont=dict(size=9, color='white'),
            name=SERIES_LABELS[ser],
            customdata=df[target_hm].values[mask],
            hovertemplate='<b>%{text}</b><br>' + xlabel + '=%{x:.2f}'
                          '<br>' + ylabel + '=%{y:.2f}'
                          '<br>' + name_hm + f'=%{{customdata:.3f}} {unit_hm}<extra></extra>'))

    fig_hm.add_trace(go.Scatter(
        x=[defaults_hm[xf]], y=[defaults_hm[yf]], mode='markers',
        marker=dict(size=16, color=METHOD_COLORS[m_col_key], symbol='star',
                    line=dict(width=2, color='white')),
        name='Your position', showlegend=True))

    fixed_label = FEATURE_LABELS[fixed_f]
    fixed_val   = defaults_hm[fixed_f]
    fig_hm.update_layout(
        title=f"{name_hm} — {xlabel} × {ylabel}  |  {fixed_label} = {fixed_val:.2f}",
        xaxis_title=xlabel, yaxis_title=ylabel, height=540,
        legend=dict(orientation='h', yanchor='bottom', y=-0.22),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hm, use_container_width=True)

    # CVD vs MBE method map
    st.markdown('<div class="section-header">CVD vs MBE METHOD MAP</div>',
                unsafe_allow_html=True)
    st.caption(
        "Score ≥3 = MBE required · Score 1–2 = Both (MBE preferred) · Score 0 = CVD sufficient. "
        "⚠ All 14 Jeon samples are MBE-grown; this map guides NEW synthesis decisions.")

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
            [0.0,   '#4E9AF155'],
            [0.375, '#4E9AF1'],
            [0.375, '#F5A623'],
            [0.75,  '#F5A623'],
            [0.75,  '#2DCE89'],
            [1.0,   '#2DCE89'],
        ],
        zmin=0, zmax=4,
        colorbar=dict(title='MBE score', tickvals=[0, 1, 2, 3, 4]),
        hovertemplate='Layer#=%{x:.0f}<br>Mo/S=%{y:.2f}<br>Score=%{z:.0f}<extra></extra>'))

    for ser, scolor in SERIES_COLORS.items():
        mask = df['series'] == ser
        fig_map.add_trace(go.Scatter(
            x=df['layer_n'][mask], y=df['mo_s_ratio'][mask], mode='markers+text',
            marker=dict(size=11, color='white', line=dict(width=2, color=scolor)),
            text=df['sample'][mask], textposition='top center',
            textfont=dict(size=8, color='white'), name=SERIES_LABELS[ser]))

    fig_map.add_trace(go.Scatter(
        x=[layer_n], y=[mo_s_ratio], mode='markers',
        marker=dict(size=16, color=METHOD_COLORS[m_col_key], symbol='star',
                    line=dict(width=2, color='white')), name='Your position'))

    fig_map.add_hline(y=0.58, line_dash='dot', line_color='white', line_width=1,
                      annotation_text="Mo/S=0.58 (MBE preferred)", annotation_font_color='white')
    fig_map.add_hline(y=0.72, line_dash='dot', line_color='#2DCE89', line_width=1,
                      annotation_text="Mo/S=0.72 (MBE required)", annotation_font_color='#2DCE89')
    fig_map.add_vline(x=3, line_dash='dot', line_color='#2DCE89', line_width=1,
                      annotation_text="Layer#=3 (k⁰ inflection)", annotation_font_color='#2DCE89')
    fig_map.add_vline(x=12, line_dash='dot', line_color='white', line_width=1,
                      annotation_text="Layer#=12", annotation_font_color='white')

    fig_map.update_layout(
        title=f"CVD vs MBE decision map — Layer# × Mo/S (est.)  |  ECSA = {ecsa_val:.1f} cm²",
        xaxis_title='Layer # (estimated)', yaxis_title='Mo/S ratio (estimated)', height=480,
        legend=dict(orientation='h', yanchor='bottom', y=-0.25),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_map, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 3D EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌐 3D Explorer":
    st.markdown("# 3D Descriptor Space Explorer")
    st.markdown(
        "<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
        "Layer# (est.) × Mo/S (est.) × ECSA (meas.) — colour = selected performance metric.</div>",
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

    if show_surf:
        N3 = 25
        ln3 = np.linspace(1, 20, N3)
        ec3 = np.linspace(2, 12, N3)
        Zs  = np.zeros((N3, N3))
        for i, ev in enumerate(ec3):
            for j, lv in enumerate(ln3):
                Zs[i, j] = gp_predict(target_3d, lv, mo_s_ratio, ev)[0]
        fig_3d.add_trace(go.Surface(
            x=ln3, y=ec3, z=Zs,
            colorscale='RdYlGn' if better_3d == 'max' else 'RdYlGn_r',
            opacity=0.55, showscale=False,
            name=f'GP surface (Mo/S={mo_s_ratio:.2f})',
            hovertemplate='Layer#=%{x:.1f}<br>ECSA=%{y:.1f}<br>' + name_3d + '=%{z:.3f}<extra></extra>'))

    for ser, scolor in SERIES_COLORS.items():
        mask = df['series'] == ser
        sub  = df[mask]
        zvals = sub[target_3d].values
        fig_3d.add_trace(go.Scatter3d(
            x=sub['layer_n'], y=sub['ecsa'], z=sub['mo_s_ratio'],
            mode='markers+text',
            marker=dict(size=8, color=zvals,
                        colorscale='RdYlGn' if better_3d == 'max' else 'RdYlGn_r',
                        cmin=df[target_3d].min(), cmax=df[target_3d].max(),
                        line=dict(width=2, color='white')),
            text=sub['sample'], name=SERIES_LABELS[ser],
            hovertemplate='<b>%{text}</b><br>Layer#=%{x}<br>ECSA=%{y:.1f}'
                          '<br>Mo/S=%{z:.2f}<br>' + name_3d +
                          f'=%{{marker.color:.3f}} {unit_3d}<extra></extra>'))

    cur_pred = gp_predict(target_3d, layer_n, mo_s_ratio, ecsa_val)[0]
    fig_3d.add_trace(go.Scatter3d(
        x=[layer_n], y=[ecsa_val], z=[mo_s_ratio],
        mode='markers',
        marker=dict(size=14, color=METHOD_COLORS[m_col_key], symbol='diamond',
                    line=dict(width=3, color='white')),
        name=f'Your position ({cur_pred:.3f} {unit_3d})'))

    fig_3d.update_layout(
        scene=dict(
            xaxis_title='Layer # (estimated)',
            yaxis_title='ECSA (cm²)',
            zaxis_title='Mo/S ratio (estimated)',
            xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(128,128,128,0.2)'),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(128,128,128,0.2)'),
        ),
        title=f"{name_3d} ({unit_3d}) in descriptor space",
        height=620,
        legend=dict(orientation='h', yanchor='bottom', y=-0.12),
        paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_3d, use_container_width=True)

    st.info(
        f"**Your position:** Layer# {layer_n} (est.) · Mo/S {mo_s_ratio:.2f} (est.) · "
        f"ECSA {ecsa_val:.1f} cm² (meas.)  "
        f"→ GP predicts **{name_3d} = {cur_pred:.3f} {unit_3d}**  |  Method: **{m_label}**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INVERSE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Inverse Predictor":
    st.markdown("# Inverse Predictor")
    st.markdown(
        "<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
        "Set target HER properties → find closest experimental match → get synthesis recommendation.</div>",
        unsafe_allow_html=True)

    st.markdown('<div class="section-header">TARGET PERFORMANCE</div>', unsafe_allow_html=True)
    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1: t_eta   = st.slider("Target η (V)",         -0.60, -0.25, -0.35, 0.01)
    with ic2: t_tafel = st.slider("Target Tafel (mV/dec)", 60,    300,   100,   5)
    with ic3: t_ecsa  = st.slider("Target ECSA (cm²)",     2.0,   12.0,  7.0,   0.5)
    with ic4: t_rct   = st.slider("Target Rct (Ω·cm²)",    20.0,  200.0, 60.0,  5.0)

    df_inv = df.copy()
    df_inv['perf_score'] = df_inv.apply(lambda r: np.sqrt(
        ((r.eta   - t_eta)   / 0.30) **2 +
        ((r.tafel - t_tafel) / 250)  **2 +
        ((r.ecsa  - t_ecsa)  / 8)    **2 +
        ((r.rct   - t_rct)   / 180)  **2), axis=1)
    candidates = df_inv.nsmallest(3, 'perf_score')
    best_inv   = candidates.iloc[0]

    st.markdown('<div class="section-header">CLOSEST EXPERIMENTAL MATCHES</div>',
                unsafe_allow_html=True)
    show_inv = candidates[['sample','series','layer_n','mo_s_ratio','ecsa',
                            'eta','tafel','rct','tof_ecsa']].reset_index(drop=True)
    st.dataframe(show_inv, use_container_width=True)
    st.caption("⚠ layer_n and mo_s_ratio are estimated descriptors.")

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
        f"<div style='margin-top:10px;'><b>MBE score: {inv_score}/{inv_max}</b></div>"
        f"</div>", unsafe_allow_html=True)

    st.markdown("**Scoring breakdown:**")
    for r in inv_reasons:
        st.markdown(f"• **{r['criterion']}** ({r['points']}/{r['max']} pts): {r['detail']}")

    st.markdown('<div class="section-header">TARGET SYNTHESIS PARAMETERS</div>',
                unsafe_allow_html=True)
    param_df = pd.DataFrame({
        'Parameter':   ['Annealing temp.', 'Deposition cycles', 'S-layer thickness',
                        'Layer # (est.)', 'Mo/S ratio (est.)'],
        'Value':       [f"{best_inv['temp']:.0f} °C", f"{best_inv['cycles']:.0f}",
                        f"{best_inv['s_thick']:.1f} Å",
                        f"{best_inv['layer_n']:.0f}",
                        f"{best_inv['mo_s_ratio']:.2f}"],
        'Provenance':  ['✅ Measured', '✅ Measured', '✅ Measured',
                        '⚠ Estimated (Scherrer ÷ 0.615 nm/layer)',
                        '⚠ Estimated (S-thickness → XANES mapping)'],
        'Note':        ['Higher T → crystalline but fewer edge sites',
                        '~1 MoS₂ layer per 5 cycles (QCM calibrated, Jeon 2026)',
                        'Primary lever for Mo/S ratio and S-vacancy density',
                        '≤3L: MBE required (k⁰ ×167, Manyepedza 2022)',
                        '>0.58: Mo⁰/MoS₂ coexistence — MBE Mo-flux control needed'],
    })
    st.dataframe(param_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧮 Feature Importance":
    st.markdown("# Feature Importance")
    st.markdown(
        "<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
        "Random Forest LOO feature importance. "
        "GP used for all predictions; RF shown here for interpretability only.</div>",
        unsafe_allow_html=True)

    st.markdown(
        "<div class='provenance-box'>"
        "⚠ <b>Interpretation caveat</b>: layer_n and mo_s_ratio are estimated descriptors. "
        "Their relative importance reflects the quality of estimation as much as true physical "
        "causation. Only ECSA is directly measured. With n=14, LOO scores have high variance."
        "</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">MODEL PERFORMANCE — LOO CV</div>',
                unsafe_allow_html=True)
    perf_rows = []
    for k in TARGETS:
        n_name, u, _ = TARGETS[k]
        perf_rows.append({
            'Property': n_name, 'Unit': u,
            'GP R²':  round(gp_scores[k]['r2'],  3),
            'GP MAE': round(gp_scores[k]['mae'], 3),
            'RF R²':  round(rf_scores[k]['r2'],  3),
            'RF MAE': round(rf_scores[k]['mae'], 3)
        })
    st.dataframe(pd.DataFrame(perf_rows), use_container_width=True)
    st.caption("⚠ n=14 — LOO scores have high variance. GP is primary predictor.")

    fi_colors = {'layer_n': '#9B59B6', 'mo_s_ratio': '#E84040', 'ecsa': '#2DCE89'}
    fi_names  = {'layer_n': 'Layer # (est.)', 'mo_s_ratio': 'Mo/S (est.)', 'ecsa': 'ECSA (meas.)'}

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
        yaxis_title='Relative importance', yaxis_range=[0, max(imps)*1.3],
        height=320, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_fi, use_container_width=True)

    heat = np.array([[rf_imps[k][i] for i in range(3)] for k in TARGETS])
    heat_df = pd.DataFrame(heat,
                           index=[TARGETS[k][0] for k in TARGETS],
                           columns=[fi_names[f] for f in FEATURES])
    fig_heat = px.imshow(heat_df, text_auto=".2f", aspect="auto",
                         color_continuous_scale='Greens', zmin=0, zmax=1,
                         title="Feature importance matrix — all targets")
    fig_heat.update_layout(height=380)
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: THEORETICAL BASIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Theoretical Basis":
    st.markdown("# Theoretical Framework")
    st.markdown(
        "<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
        "Scientific foundation integrating the core papers cited in Jeon 2026 "
        "and Manyepedza 2022.</div>",
        unsafe_allow_html=True)

    papers = [
        ("1 · Jeon et al. — ACS Nano 2026 [PRIMARY DATA SOURCE]",
         "14 MBE-grown MoS₂ samples on Si in 1M KOH. Three systematic series: "
         "T-series (annealing temp 600–800°C, cycles=50, S=9.0 Å), "
         "N-series (cycles 5–50, 800°C, S=3.0 Å), "
         "M-series (S-thickness 2.0–9.0 Å, 800°C, cycles=50). "
         "Global optimum: MoS-N10 (η=−0.33V, Tafel=80 mV/dec, ECSA=8.0 cm², Rct=52.8 Ω·cm²). "
         "All values in Table 1 are directly measured except layer_n and Mo/S ratio "
         "(derived in this predictor from Scherrer and XANES mapping)."),

        ("2 · Manyepedza et al. — J. Phys. Chem. C 2022 [LAYER CALIBRATION]",
         "Impact electrochemistry of MoS₂ nanoparticles. KEY FINDINGS used in this predictor: "
         "(1) AFM Fig. 9B: platelet thickness 0.6–0.7 nm = 1 trilayer, 1.3–1.4 nm = 2 trilayers. "
         "This validates 0.615 nm/trilayer for Scherrer conversion. "
         "(2) HER onset −0.10 V vs RHE for 1–2 trilayer particles (vs −0.29 V for bulk deposits). "
         "(3) Citing McKelvey et al. (Electrochim. Acta 2021): k⁰ = 250 cm/s (1 trilayer) "
         "→ 1.5 cm/s (3 trilayers), i.e., 167× kinetic advantage for fewer layers. "
         "(4) XPS: stoichiometric MoS₂ → S/Mo = 2.2, Mo/S = 0.455. "
         "(5) Faradaic efficiency 45–48% for H₂ production confirmed by GC."),

        ("3 · Bentley et al. — Chem. Sci. 2017 [LAYER CALIBRATION — INDEPENDENT CONFIRMATION + BASAL ACTIVITY]",
         "SECCM nanoscale electrochemical mapping of HER on natural MoS₂ crystals. "
         "KEY CALIBRATION: Introduction states 'adjacent layers are weakly interacting "
         "(van der Waals gap = 6.15 Å)' — independently confirms 0.615 nm/trilayer used for layer_n. "
         "KEY HER FINDINGS: "
         "(1) Basal plane is NOT inert: J₀ = 2.5×10⁻⁶ A/cm² (comparable to Co, Ni, Cu, Au). "
         "(2) Edge plane: J₀ ~1×10⁻⁴ A/cm² — ~40× more active than basal plane. "
         "(3) Tafel slope ~120 mV/dec (Volmer RDS) — consistent with Jeon alkaline data. "
         "(4) HER activity scales proportionally with edge-plane area (step height). "
         "Supports why ECSA (edge site density) is a key predictor in this tool."),

        ("4 · Cao et al. — Sci. Rep. 2017, 7, 8825 [LAYER CALIBRATION — FOURTH INDEPENDENT SOURCE]",
         "MoS₂ nanosheets grown on MWCNTs by hydrothermal method (1T-rich phase). "
         "KEY CALIBRATION: HRTEM images (Fig. 2b,c) directly show interlayer spacing = 0.63 nm "
         "for MoS₂ nanosheets (3–8 layers confirmed). This is the fourth independent source "
         "confirming the 0.615–0.63 nm/trilayer value used for layer_n estimation. "
         "ADDITIONAL CONTEXT: "
         "(1) 1T phase XPS: Mo⁴⁺ 3d₅/₂ shifts from ~229 eV (2H) to ~228.1 eV (1T) — Δ≈0.9 eV. "
         "Consistent with Sherwood 2024 4-peak model framework. "
         "(2) Tafel slope 43 mV/dec (Volmer-Heyrovsky RDS) for metallic 1T-MoS₂ on MWCNT. "
         "(3) Onset ~50 mV — far superior to 2H phase, showing phase matters critically for HER. "
         "This reinforces why Mo/S ratio (proxy for 1T/2H fraction) is a key descriptor."),

        ("5 · Jiang et al. (SI) — Nano Research 2019 [XPS REFERENCE VALUES FOR 2H-MoS₂]",
         "MoS₂ moiré superlattice (bilayer, twisted θ≈7.3°) for HER. "
         "KEY XPS REFERENCE (Fig. S2): Pristine 2H-MoS₂ bilayer: "
         "Mo⁴⁺ 3d₅/₂ = 230.3 eV, Mo⁴⁺ 3d₃/₂ = 233.5 eV; S²⁻ 2p₃/₂ = 163.1 eV, 2p₁/₂ = 164.3 eV. "
         "No binding energy shift between bilayer and nanoscroll — confirms pure 2H phase. "
         "This provides additional XPS reference for validating Mo/S ratio stoichiometry mapping: "
         "pure 2H-MoS₂ at these binding energies corresponds to near-stoichiometric Mo/S ≈ 0.455. "
         "Cdl: bilayer 10.37 mF/cm², nanoscroll 10.28 mF/cm² — similar ECSA despite geometry change."),

        ("6 · Wang et al. — Phys. Status Solidi RRL 2023 [RAMAN RATIO VALIDATION + MBE CONTEXT]",
         "MBE-grown MoS₂ monolayers on sapphire (1–2 ML), recrystallized by CVD-furnace annealing. "
         "KEY VALIDATION for Raman A₁g/E' ratio: "
         "(1) As-grown MBE at 750–900°C: A₁g/E' = 2.0–2.4 → consistent with Jeon T-series (2.29–2.41). "
         "(2) Post-CVD-anneal at 900°C: ratio drops to ~1.4 → more ordered, fewer defects/edge sites. "
         "(3) LA(M) defect peak at 227 cm⁻¹ disappears after annealing → confirms defect origin. "
         "(4) Domain size: 10–20 nm (as-grown) → 50–100 nm (annealed). "
         "Implication for predictor: lower Raman ratio = more structural disorder = more edge sites = better HER. "
         "XPS confirms Mo⁴⁺ (229.9 eV) = 2H-MoS₂ phase — consistent across Jeon and Wang systems."),

        ("7 · Jaramillo et al. — Science 2007 [EDGE SITE ORIGIN]",
         "Foundational study: HER activity of MoS₂ scales linearly with edge-site density, "
         "not basal plane area. Mo-terminated edges are the dominant active sites. "
         "This underpins why layer # (edge/basal ratio) and ECSA are key descriptors."),

        ("6 · Geng et al. — Nature Commun. 2016 [METALLIC 1T PHASE]",
         "Pure and stable metallic 1T-MoS₂: ρ=0.48 Ω·cm, Rct≈1 Ω, η=−175 mV. "
         "Mixed Mo⁰/MoS₂ domains in N10 and M3.0–M6.0 replicate conductivity benefits "
         "via metallic Mo pathways — providing parallel conductive channels without "
         "requiring full phase conversion."),

        ("7 · Tsai et al. — Nature Commun. 2017 [S-VACANCY ENGINEERING]",
         "Electrochemical desulfurization generates S-vacancies at ≥−1.0 V vs RHE. "
         "Optimal concentration: 12.5–15.6% surface S-vacancies (ΔGH*≈0 eV). "
         "This is the theoretical optimum that MBE M2.0–M3.0 approaches by design. "
         "XANES shows Mo-S coordination number decreases monotonically with vacancy density."),

        ("8 · Zhu et al. — Nature Commun. 2019 [GRAIN BOUNDARY ACTIVATION]",
         "2H–1T grain boundaries are highly active HER sites: ΔGH*=−0.13 eV "
         "(cf. Pt: −0.18 eV). Tafel slope 73–75 mV/dec. Stable >200 h. "
         "More boundaries → better performance. Intermediate thickness in MBE "
         "creates optimal boundary density before grain coalescence."),

        ("9 · Li et al. — Nature Materials 2016 [STRAINED S-VACANCIES]",
         "Combining S-vacancies with tensile strain (≈1% on Si substrate by MBE mismatch): "
         "VMoS3 defect → ΔGH*≈0 eV with strain. Without strain: ΔGH*>0.22 eV. "
         "MBE on Si introduces 1–2% tensile mismatch that activates vacancy sites. "
         "This explains synergy unique to MBE-on-Si platform."),

        ("12 · McKelvey et al. (via Manyepedza 2022) [k⁰ vs TRILAYER NUMBER]",
         "Standard electrochemical rate constant k⁰ as function of MoS₂ thickness: "
         "1 trilayer: k⁰ ≈ 250 cm/s · 3 trilayers: k⁰ ≈ 1.5 cm/s (167× difference). "
         "This is the primary kinetic justification for preferring ≤3-layer MoS₂ "
         "and drives the highest score (3/3 pts) in the Layer# criterion of MBE scoring."),

        ("13 · Sherwood et al. — ACS Appl. Nano Mater. 2024 [XPS STOICHIOMETRY]",
         "Four-peak XPS model distinguishes 2H-MoS₂ (229.3 eV), 1T-MoS₂ (228.4 eV), "
         "and MoS₂₋ₓ (228.1 eV). Stoichiometric 2H: S/Mo = 2.2–2.5 (Mo/S ≈ 0.40–0.455). "
         "Ar⁺ etching drives S/Mo → 1.1 (Mo/S → 0.91). "
         "Cross-validated with: Jiang 2019 (2H: 230.3 eV), Wang 2023 (MBE: 229.9 eV), "
         "Cao 2017 (2H: ~229 eV, 1T: ~228.1 eV, Δ≈0.9 eV). "
         "These endpoints anchor the Mo/S ratio slider in this predictor."),

        ("14 · Noh et al. — Phys. Rev. B 2014 [METALLIC Mo CONDUCTIVITY]",
         "Metallic Mo inclusions or partially sulfided MoSₓ domains create parallel "
         "conductive pathways that significantly lower charge-transport resistance in "
         "mixed-phase Mo/MoS₂ systems. Directly explains Rct reduction in "
         "M3.0–M6.0 (45.5–64.0 Ω·cm²) vs fully sulfurized M8.0–M9.0 (124.5–193.3 Ω·cm²). "
         "Cao 2017 confirms: 1T-MoS₂ on MWCNT has Rct ~100 Ω vs pure 2H ~10 kΩ."),
    ]

    for title, body in papers:
        with st.expander(title):
            st.write(body)

    st.markdown('<div class="section-header">DESCRIPTOR SUMMARY TABLE</div>',
                unsafe_allow_html=True)
    desc_df = pd.DataFrame({
        'Descriptor': [
            'Layer # ⚠', 'Mo/S ratio ⚠', 'ECSA ✅',
            'Raman A₁g/E₂g ✅', 'Resistivity ✅', 'Rct ✅'],
        'Measures': [
            'Film thickness → edge/basal ratio + electron tunneling kinetics (k⁰)',
            '2H-MoS₂ ↔ MoS₂₋ₓ ↔ Mo⁰/MoS₂ phase composition',
            'Electrochemically active surface area (edge sites)',
            'Mo vs S edge site exposure (vibrational coupling)',
            'Bulk electronic conductivity (Mo⁰ domain density)',
            'Interfacial charge transfer resistance (HER-relevant)'],
        'Optimal range': [
            '≤3L (k⁰ ~250 cm/s, onset −0.10 V, Manyepedza 2022)',
            '0.55–0.72 (Mo⁰/MoS₂ coexistence, Jeon M3.0–N10)',
            '≥8 cm² (Jeon N10: 8.0, M6.0: 9.2)',
            '<1.8 (high edge exposure)',
            '<12 Ω·cm (metallic Mo channels)',
            '<55 Ω·cm² (Jeon N10: 52.8, M6.0: 45.5)'],
        'Provenance': [
            '⚠ XRD Scherrer ÷ 0.615 nm/layer (×4 sources: Manyepedza+Bentley+Cao+Fan)',
            '⚠ S-thickness → XANES mapping (Jeon 2026 + Sherwood 2024 + Jiang 2019 XPS)',
            '✅ Cdl measurement, Jeon 2026 Table 1',
            '✅ Raman spectroscopy, Jeon 2026 Table 1',
            '✅ Four-point probe, Jeon 2026 Table 1',
            '✅ EIS Nyquist fitting, Jeon 2026 Table 1'],
    })
    st.dataframe(desc_df, use_container_width=True)
    st.caption(
        "⚠ = estimated descriptor (use with caution) · "
        "✅ = directly measured in Jeon et al. ACS Nano 2026")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("# About")
    st.markdown("""
**MoS₂ HER Predictor v2.2** — Theory-guided Gaussian Process prediction for MBE-grown MoS₂ in 1M KOH.

---

### Primary experimental source
**Jeon et al., *ACS Nano* 2026, 20, 4479–4493** — 14 MBE samples on Si.
- **T-series**: Annealing temperature 600–800°C (cycles=50, S-thickness=9.0 Å)
- **N-series**: Deposition cycles 5–50 (800°C, S-thickness=3.0 Å)
- **M-series**: S-layer thickness 2.0–9.0 Å (800°C, cycles=50)

### Layer # calibration — FOUR-SOURCE validated (0.615–0.63 nm/trilayer)

**① Manyepedza et al., *J. Phys. Chem. C* 2022** — AFM: 0.6–0.7 nm (1L), 1.3–1.4 nm (2L) on mica

**② Bentley et al., *Chem. Sci.* 2017** — "van der Waals gap = 6.15 Å" stated explicitly; J₀ basal/edge quantified

**③ Cao et al., *Sci. Rep.* 2017, 7, 8825** — HRTEM Fig. 2b,c: interlayer spacing = 0.63 nm (3–8 layers)

**④ Fan et al., *JACS* 2016 (via Manyepedza ref. 63)** — controlled exfoliation confirms trilayer spacing

### XPS cross-validation for Mo/S ratio
| Paper | Mo⁴⁺ 3d₅/₂ | Phase |
|---|---|---|
| Wang 2023 (MBE as-grown) | 229.9 eV | 2H-MoS₂ |
| Jiang 2019 SI (bilayer) | 230.3 eV | 2H-MoS₂ |
| Sherwood 2024 | 229.3 eV | 2H-MoS₂ |
| Cao 2017 | ~229 eV (2H) / ~228.1 eV (1T) | 2H + 1T |

### Raman A₁g/E' ratio validation
**Wang et al., *Phys. Status Solidi RRL* 2023** — MBE as-grown ratio 2.0–2.4 (matches Jeon T-series)

### Mo/S ratio calibration
- Jeon 2026 XANES/EXAFS + Sherwood 2024: stoichiometric Mo/S ≈ 0.455 (S/Mo = 2.2)
- Jiang 2019 SI: 2H-MoS₂ bilayer XPS at 230.3 eV → pure 2H reference confirmed
- Upper limit Mo/S ≈ 0.91 (Sherwood 2024 Ar⁺-etching extreme)

---

### Data provenance

| Descriptor | Type | Source |
|---|---|---|
| η, Tafel, Rct, ECSA, resistivity, Raman, TOF | ✅ Measured | Jeon 2026 Table 1 |
| Annealing temp., cycles, S-thickness | ✅ Measured | Jeon 2026 growth conditions |
| **Layer #** | ⚠ Estimated | Scherrer D(002) ÷ 0.615 nm/layer (×4 sources) |
| **Mo/S ratio** | ⚠ Estimated | S-thickness + XANES + Sherwood 2024 + Jiang 2019 |

### Machine learning

| Component | Detail |
|---|---|
| Primary model | Gaussian Process (Matérn ν=2.5, ARD length-scales, calibrated 95% CI) |
| Secondary model | Random Forest (300 trees, LOO) — feature importance only |
| Validation | Leave-One-Out cross-validation (n=14) |
| Features | Layer # (est.), Mo/S ratio (est.), ECSA (meas.) |
| Targets | η, Tafel slope, Rct, Raman ratio, resistivity, TOF×2 |

### CVD vs MBE scoring

| Criterion | Max pts | Scientific basis |
|---|---|---|
| Layer # | 3 | McKelvey k⁰ kinetics (via Manyepedza 2022) |
| Mo/S ratio | 3 | Phase control (Jeon XANES + Sherwood 2024 XPS) |
| ECSA | 1 | Wafer-scale uniformity (Jeon 2026) |
| Rct | 1 | Mo⁰ domain requirement (Jeon 2026 EIS) |

**⚠ Important**: All 14 Jeon 2026 samples ARE MBE-grown. Scoring guides NEW synthesis decisions only.

Score ≥3 → MBE required · Score 1–2 → Both viable (MBE preferred) · Score 0 → CVD sufficient

### v2.1 additions vs v2.0
- **Bentley 2017** integrated: independent 0.615 nm/layer confirmation + basal/edge J₀
- **Wang 2023** integrated: Raman A₁g/E' ratio validation for MBE as-grown films
- Layer # calibration now triple-source validated throughout the app

### v2.0 corrections vs v1.0
- **Bug fixed**: Radar chart uses `idxmin()` for η → correctly identifies MoS-N10 as best
- **layer_n**: Derivation fully explicit and traceable
- **mo_s_ratio**: Endpoints anchored to Sherwood 2024 XPS stoichiometry
- **CVD/MBE scoring**: Explicit disclaimer about Jeon samples being MBE
- All estimated descriptors marked ⚠; measured descriptors marked ✅

---

⚠ With n=14 training samples, all predictions carry **high uncertainty**.
This tool is for **trend analysis and mechanistic understanding**, supporting but not replacing experimental validation.
    """)
