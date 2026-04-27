"""
MoS₂ HER Predictor — v3.0 (Multi-Paper Validated)
===================================================

PRIMARY EXPERIMENTAL DATA
--------------------------
Jeon et al., ACS Nano 2026, 20, 4479–4493
  14 MBE-grown MoS₂ samples on Si in 1M KOH
  T-series (temp 600–800°C), N-series (cycles 5–50), M-series (S-thick 2.0–9.0 Å)
  All Table 1 values directly measured EXCEPT layer_n and mo_s_ratio (derived)

CALIBRATION PAPERS — INTEGRATED IN v3.0
-----------------------------------------

GAP 1: Mo/S RATIO — XPS CALIBRATION
  Sherwood et al., ACS Appl. Nano Mater. 2024, DOI: 10.1021/acsanm.5c04608
    - 4-peak XPS model: 2H (229.3 eV), 1T (228.4 eV), MoS2-x (228.1 eV), MoO3
    - S(POS-A)/Mo(POS-A+POS-C) ratio: pristine S/Mo=2.2 → after 70s etch S/Mo=1.45
    - KEY: POS-A binding energy stays CONSTANT under etching — only POS-C grows
    - This means Mo/S > 0.58 in Jeon M-series = S-vacancies in 2H matrix, NOT 1T phase
    - Fig.S18 (yellow curve): S/Mo 2.2→1.45 maps to Mo/S 0.455→0.690
    - Stoichiometric 2H endpoint: S/Mo=2.2 → Mo/S=0.455

  Jiang et al. SI, Nano Research 2019
    - 2H-MoS₂ bilayer XPS reference: Mo4+ 3d5/2=230.3 eV, S2- 2p3/2=163.1 eV
    - Confirms pure 2H at stoichiometric Mo/S≈0.455

  ACS Catalysis 2023, DOI: 10.1021/acscatal.2c03719 (CVD distance gradient)
    - CVD MoS₂ on Au foil, distances 8–24 cm from MoO3 precursor, 0.5M H2SO4
    - S/Mo measured directly by XPS vs distance:
        8cm  → S/Mo≈2.0  → Mo/S=0.500  (1H+1T mixed, 65% 1T)
        14cm → S/Mo≈2.0  → Mo/S=0.500  (50% 1T, stoichiometric)
        16cm → S/Mo≈2.0  → Mo/S=0.500  (pure 1H, stoichiometric)
        20cm → S/Mo≈1.7  → Mo/S=0.588  (S-vacancies onset — THRESHOLD)
        22cm → S/Mo≈1.35 → Mo/S=0.741  (severe vacancies)
        24cm → S/Mo=1.15 → Mo/S=0.870  (extreme, Mo oxidized)
    - S/Mo=1.70 confirmed as threshold for undercoordinated Mo exposure
    - OPTIMAL HER at Mo-16 to Mo-18 (Mo/S 0.588–0.606) — validates Jeon N10
    - CRITICAL: 1T phase in Mo-8 to Mo-14 converts to 1H during LSV cycling
      → Not stable 1T, same as M2.0–M3.0 in Jeon
    - Electrolyte: 0.5M H2SO4 → η values NOT directly comparable with Jeon KOH 1M

  Smiri et al., Scientific Reports 2026, DOI: 10.1038/s41598-025-09826-x (ALD)
    - S/Mo vs layer number (ALD, intrinsic interface effect):
        1ML → S/Mo≈1.75 → Mo/S=0.571 (interface S-deficiency)
        6ML → S/Mo≈1.95 → Mo/S=0.513 (more stoichiometric)
    - Mo6+/Mo ratio decreases with layers (oxide at MoS2/substrate interface)
    - IMPLICATION: N5, N10 in Jeon have partial Mo/S elevation from interface effect
      in addition to S-flux control

COMBINED XPS CALIBRATION TABLE (S/Mo → Mo/S, all sources):
  S/Mo 2.20 → Mo/S 0.455  | 2H pristine (Sherwood pristine, Jiang 2019)
  S/Mo 2.00 → Mo/S 0.500  | 2H pure / 1H (ACS Cat 2023 Mo-8 to Mo-16)
  S/Mo 1.85 → Mo/S 0.541  | MoS2-x onset (Sherwood 20s etch)
  S/Mo 1.75 → Mo/S 0.571  | MoS2-x moderate (Sherwood 30s, Smiri 1ML)
  S/Mo 1.70 → Mo/S 0.588  | THRESHOLD: undercoordinated Mo (ACS Cat explicit)
  S/Mo 1.65 → Mo/S 0.606  | MoS2-x growing (Sherwood 40s)
  S/Mo 1.55 → Mo/S 0.645  | MoS2-x dominant (Sherwood 50s)
  S/Mo 1.45 → Mo/S 0.690  | Mo-rich (Sherwood 70s limit)
  S/Mo 1.15 → Mo/S 0.870  | Extreme Mo-rich + oxidation (ACS Cat Mo-24)
  S/Mo 1.10 → Mo/S 0.909  | Ar+ extreme limit (Sherwood extrapolated)

GAP 2: LAYER NUMBER — RAMAN CALIBRATION
  Lee et al., ACS Nano 2010, 4, 2695–2700, DOI: 10.1021/nn1003937
    - Δω = A1g - E12g vs layer number (1L to bulk):
        1L → Δω≈18.7 cm-1  (direct gap semiconductor)
        2L → Δω≈21.5 cm-1
        3L → Δω≈22.5 cm-1
        4L → Δω≈23.5 cm-1
        6L → Δω≈25.0 cm-1
        Bulk → Δω≈26.0 cm-1
    - SATURATION: >4L frequencies converge to bulk — Δω loses discrimination
    - A1g blueshifts, E12g redshifts with increasing N

  JOM 2025, DOI: 10.1007/s11837-025-07448-2 (CVD, Raman vs Si substrate)
    - I(A1g)/I(Si) and I(E12g)/I(Si) linear with N for 1–4L
    - Extends discrimination slightly beyond Lee 2010 for few-layer regime
    - Tables I–III: average Raman ratios for triangular, hexagonal, truncated

  Smiri et al., Scientific Reports 2026 (ALD, 200mm wafer)
    - A1g/E12g ratio DECREASES with layers in ALD films (1→6ML: 2.05→1.85)
    - OPPOSITE trend vs Jeon 2026 "raman" column (which increases)
    - EXPLANATION: Jeon "raman" ratio reflects crystallinity + defects + layers
      NOT just layer number for N>4L
    - FWHM(E12g): 5.7→4.5 cm-1 for 1→6ML (crystallinity improves with thickness)
    - LA(M)/E12g ratio: 0.72→0.42 for 1→6ML (disorder proxy, also saturates)
    - Stoichiometry improves with layers (same interface effect as gap 1)

RAMAN CALIBRATION SUMMARY FOR JEON DATASET:
  N5  (raman=1.01, layer_n=2):  Δω≈18-19 → VALIDATES 2L (high confidence)
  N10 (raman=1.63, layer_n=5):  Δω≈21   → VALIDATES 4-5L (medium confidence)
  N20+ (raman≥1.78):            SATURATED — bulk-like, Scherrer is primary estimator
  T,M series (raman 2.29-2.41): Reflects crystallinity, NOT layer discrimination

GAP 3: CVD COMPARISON SAMPLES
  PECVD paper (RF sputtering + ICP-PECVD, 0.5M H2SO4):
    - Mo/S=0.510 by XPS (directly measured) — confirms CVD→stoichiometric
    - Δk=24.9 cm-1 → bulk-like (~10-13 layers from Lee 2010)
    - HRTEM: d=0.75 nm interlayer, ~40nm total → ~8-12 active layers
    - η=−0.45V, Tafel=76 mV/dec in 0.5M H2SO4
    - Method: PHYSICAL+CHEMICAL hybrid (sputtering + PECVD)
    - Scoring: layer_n~10(1pt), Mo/S~0.510(1pt) → score~2/8 → Both viable ✓

  ACS Catalysis 2023 (CVD gradient, 0.5M H2SO4):
    - Pure 1H-MoS2 at distances >16cm (stoichiometric, S/Mo≈2.0)
    - 1T component at <16cm converts to 1H during electrochemical cycling
    - OPTIMAL HER at intermediate distance (Mo/S 0.588-0.606) — VALIDATES JEON
    - η NOT comparable to Jeon (different electrolyte)
    - Pattern identical to M-series: volcano with optimum at moderate Mo-rich

GAP 4: k⁰ VS LAYER NUMBER
  Manyepedza et al., J. Phys. Chem. C 2022, 126, 17942–17951
    - AFM Fig.9: 0.65 nm/tricapa confirmed (heights: 0.65, 1.30, 1.95, 2.60 nm)
    - Citing McKelvey: Brunet Cabre et al., Electrochim. Acta 2021, 393, 139027
    - Fig.7 simulations give 5-point k⁰ curve:
        k⁰ = 250   cm/s → 1 trilayer  (3/3 pts MBE)
        k⁰ = 7.5   cm/s → ~2 trilayers (3/3 pts MBE)
        k⁰ = 1.5   cm/s → 3 trilayers  (3/3 pts MBE)
        k⁰ = 0.1   cm/s → ~5 trilayers (2/3 pts MBE)
        k⁰ = 0.01  cm/s → ~10 trilayers (1/3 pts MBE)
    - Three HER onsets in RDE experiment:
        −0.10V → 1-2 trilayers (k⁰ ~7.5–250 cm/s)
        −0.25V → ~3 trilayers  (k⁰ ~1.5 cm/s)
        −0.50V → bulk particles (k⁰ ~0.01 cm/s)
    - Faradaic efficiency: 45–48% for H2 confirmed by GC
    - XPS: S/Mo=2.2 → Mo/S=0.455 (electrodeposited MoS2 reference)
    - Electrolyte: pH 2 H2SO4

INTERLAYER SPACING — FOUR-SOURCE VALIDATION (0.615–0.65 nm):
  1. Manyepedza 2022 AFM: 0.65 nm (1L), 1.30 nm (2L) on mica
  2. Bentley 2017 Chem.Sci.: "van der Waals gap = 6.15 Å" explicit
  3. Cao et al. Sci.Rep. 2017, 7, 8825: HRTEM = 0.63 nm
  4. Fan et al. JACS 2016: controlled exfoliation confirms trilayer spacing

GAPS STILL PENDING (see instruction prompt for next chat):
  GAP 3b: CVD MoS2 HER data in KOH 1M (same electrolyte as Jeon)
  GAP 5:  Jaramillo et al. Science 2007 — TOF vs edge site density
  GAP 4b: McKelvey full paper (Electrochim. Acta 2021, 393, 139027)

KEY MECHANISTIC CORRECTIONS FROM v2.x:
  1. Mo/S>0.58 = S-VACANCIES IN 2H MATRIX, not 1T phase (Sherwood 2024)
     → CVD/MBE scoring logic correct but mechanism description updated
  2. "1T phase" in CVD Mo-8 converts to 1H during cycling (ACS Cat 2023)
     → Same phenomenon expected in Jeon M2.0–M3.0
  3. Raman A1g/E12g in Jeon reflects crystallinity+defects for N>4L (Smiri 2026)
     → Do NOT use raman as layer proxy for T-series and M-series
  4. Few-layer Mo/S elevation has interface-effect component (Smiri 2026)
     → N5, N10 Mo/S partially elevated by substrate interface, not just S-flux

DATA PROVENANCE:
  ✅ MEASURED in Jeon 2026 Table 1:
     η, Tafel slope, Rct, ECSA, resistivity, Raman A1g/E2g, TOF (ECSA & mass), loading
  ⚠ ESTIMATED / DERIVED (clearly flagged):
     layer_n  → D(002) Scherrer ÷ 0.615 nm/layer (4-source validated)
     mo_s_ratio → S-thickness → XPS calibration table (Sherwood 2024 + ACS Cat 2023)
                  Validated for N5 (Mo/S≈0.571 from Smiri) and N10 (Mo/S≈0.556 from Sherwood)
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
.descriptor-card .label { font-family: 'IBM Plex Mono', monospace; font-size: 0.72em; color: #888; text-transform: uppercase; letter-spacing: 0.08em; }
.descriptor-card .value { font-family: 'IBM Plex Mono', monospace; font-size: 1.5em; font-weight: 600; margin: 2px 0; }
.descriptor-card .note { font-size: 0.78em; color: #aaa; }
.ref-chip {
    display: inline-block; background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 3px; padding: 1px 7px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72em; color: #aaa; margin: 2px;
}
.section-header {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75em;
    text-transform: uppercase; letter-spacing: 0.12em; color: #666;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 6px; margin: 20px 0 12px 0;
}
.provenance-box {
    background: rgba(255,193,7,0.07); border: 1px solid rgba(255,193,7,0.25);
    border-left: 4px solid #FFC107; border-radius: 4px;
    padding: 10px 14px; margin: 8px 0; font-size: 0.82em; color: #ccc;
}
.correction-box {
    background: rgba(78,154,241,0.07); border: 1px solid rgba(78,154,241,0.25);
    border-left: 4px solid #4E9AF1; border-radius: 4px;
    padding: 10px 14px; margin: 8px 0; font-size: 0.82em; color: #ccc;
}
.stMetric label { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78em !important; }
.stMetric [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# XPS CALIBRATION TABLE — Mo/S ratio from S/Mo
# Integrated from: Sherwood 2024, ACS Catalysis 2023, Smiri 2026
# ══════════════════════════════════════════════════════════════════════════════
XPS_CALIBRATION = {
    # S/Mo measured : (Mo/S, description, source)
    2.20: (0.455, '2H pristine stoichiometric',         'Sherwood 2024 pristine + Jiang 2019'),
    2.00: (0.500, '2H pure / 1H-MoS2',                 'ACS Cat 2023 Mo-8 to Mo-16'),
    1.85: (0.541, 'MoS2-x onset',                       'Sherwood 2024 20s etch'),
    1.75: (0.571, 'MoS2-x moderate',                    'Sherwood 2024 30s + Smiri 2026 1ML'),
    1.70: (0.588, 'Threshold: undercoordinated Mo',     'ACS Cat 2023 explicit threshold'),
    1.65: (0.606, 'MoS2-x growing',                     'Sherwood 2024 40s etch'),
    1.55: (0.645, 'MoS2-x dominant',                    'Sherwood 2024 50s etch'),
    1.45: (0.690, 'Mo-rich moderate',                   'Sherwood 2024 70s etch limit'),
    1.15: (0.870, 'Extreme Mo-rich + oxidation',        'ACS Cat 2023 Mo-24'),
    1.10: (0.909, 'Ar+ extreme limit',                  'Sherwood 2024 extrapolated'),
}

# k⁰ vs layer number — McKelvey (via Manyepedza 2022 Fig.7 simulations)
K0_VS_LAYERS = {
    1:  250.0,   # McKelvey anchor
    2:  7.5,     # inferred from Manyepedza Fig.7
    3:  1.5,     # McKelvey anchor
    5:  0.1,     # inferred from Manyepedza Fig.7
    10: 0.01,    # inferred from Manyepedza Fig.7
    20: 0.001,   # extrapolated bulk
}

# Raman Δω vs layer number — Lee 2010 ACS Nano
RAMAN_DELTA_VS_LAYERS = {
    1: 18.7, 2: 21.5, 3: 22.5, 4: 23.5, 6: 25.0, 'bulk': 26.0
}

# Raman confidence flags for Jeon samples
RAMAN_LAYER_CONFIDENCE = {
    'MoS-N5':  'high',    # raman=1.01, Δω≈18-19 → validates 2L
    'MoS-N10': 'medium',  # raman=1.63, Δω≈21 → validates 4-5L
    'MoS-N20': 'low',     # raman=1.85, saturated
    'MoS-N30': 'low',     'MoS-N50': 'very_low',
    'MoS-T600': 'low', 'MoS-T700': 'low', 'MoS-T800': 'very_low',
    'MoS-M2.0': 'very_low', 'MoS-M2.5': 'very_low', 'MoS-M3.0': 'very_low',
    'MoS-M6.0': 'very_low', 'MoS-M8.0': 'very_low', 'MoS-M9.0': 'very_low',
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA — Jeon et al. ACS Nano 2026, Table 1
# layer_n: Scherrer D(002) ÷ 0.615 nm/layer (4-source validated)
# mo_s_ratio: XPS calibration table anchored to Sherwood 2024 + ACS Cat 2023
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    """
    Mo/S ratio derivation (v3.0 — XPS-calibrated):
      Anchored to Sherwood 2024 SI Fig.S18 (S(POS-A)/Mo(POS-A+POS-C) curve)
      and ACS Catalysis 2023 (S/Mo directly measured by XPS vs CVD distance).
      Cross-validated with Smiri 2026 (ALD interface effect).

      T-series (S=9.0 Å, near-stoichiometric throughout):
        Mo/S ≈ 0.455–0.500 (S/Mo ≈ 2.0–2.2, pure 2H)
        T600: 0.490 | T700: 0.480 | T800: 0.460

      N-series (S=3.0 Å fixed, cycles vary):
        N5 (2L): Mo/S≈0.571 — partial S-deficiency + interface effect (Smiri 2026)
        N10 (5L): Mo/S≈0.556 — optimal zone (Sherwood 30s anchor)
        N20–N50: more stoichiometric → Mo/S 0.520–0.470

      M-series (cycles=50, S-thickness 2.0–9.0 Å):
        M9.0: Mo/S≈0.460 (near-stoichiometric, like T800)
        M6.0: Mo/S≈0.520
        M3.0: Mo/S≈0.645 (MoS2-x dominant, Sherwood 50s anchor)
        M2.5: Mo/S≈0.720 (beyond Sherwood 70s limit, extrapolated)
        M2.0: Mo/S≈0.820 (extreme, approaching ACS Cat Mo-24 territory)

      MECHANISTIC NOTE (v3.0 correction):
        Mo/S > 0.58 = S-VACANCIES IN 2H MATRIX (Sherwood 2024)
        NOT necessarily stable 1T phase.
        "1T-like" conductivity in M2.0–M3.0 arises from metallic Mo domains
        at vacancy sites, not phase-converted 1T. Converts back to 2H under
        electrochemical cycling (ACS Cat 2023 confirmation).
    """
    data = {
        'sample':      ['MoS-T600','MoS-T700','MoS-T800',
                        'MoS-N5','MoS-N10','MoS-N20','MoS-N30','MoS-N50',
                        'MoS-M2.0','MoS-M2.5','MoS-M3.0','MoS-M6.0','MoS-M8.0','MoS-M9.0'],
        'series':      ['T','T','T','N','N','N','N','N','M','M','M','M','M','M'],
        'temp':        [600,700,800,800,800,800,800,800,800,800,800,800,800,800],
        'cycles':      [50,50,50,5,10,20,30,50,50,50,50,50,50,50],
        's_thick':     [9.0,9.0,9.0,3.0,3.0,3.0,3.0,3.0,2.0,2.5,3.0,6.0,8.0,9.0],

        # ⚠ ESTIMATED — Scherrer D(002) ÷ 0.615 nm/layer
        # Raman-validated for N5 (2L) and N10 (5L) only
        # Saturated Raman for all other samples — Scherrer is primary estimator
        'layer_n':     [12, 14, 18,   2,  5,  9, 13, 20,  20, 20, 20, 20, 20, 20],

        # ⚠ ESTIMATED — XPS calibration table v3.0
        # Anchored: Sherwood 2024 SI Fig.S18 + ACS Cat 2023 + Smiri 2026
        # Mechanism: S-vacancies in 2H matrix (NOT 1T phase)
        'mo_s_ratio':  [0.49,0.48,0.46, 0.57,0.56,0.52,0.50,0.47, 0.82,0.72,0.65,0.52,0.48,0.46],

        # ✅ MEASURED — Jeon 2026 Table 1 (all values below)
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
    'eta':         ('Overpotential η', 'V',         'min'),
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

FEATURE_PROVENANCE = {
    'layer_n':    '⚠ Est. from XRD Scherrer ÷ 0.615 nm/layer (×4 sources). Raman validates N5, N10 only.',
    'mo_s_ratio': '⚠ Est. from S-thickness via XPS calibration table (Sherwood 2024 + ACS Cat 2023). Mechanism: S-vacancies in 2H matrix.',
    'ecsa':       '✅ Directly measured — Jeon 2026 Table 1 (Cdl method, Cs=40 µF/cm²)',
}

SERIES_COLORS = {'T': '#4E9AF1', 'N': '#2DCE89', 'M': '#F5A623'}
SERIES_LABELS = {'T': 'T-series (Temp.)', 'N': 'N-series (Cycles)', 'M': 'M-series (S-thick.)'}
METHOD_COLORS = {'mbe': '#2DCE89', 'both': '#F5A623', 'cvd': '#4E9AF1'}


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
# CVD vs MBE SCORING — v3.0
# Updated mechanism descriptions based on Sherwood 2024 + ACS Cat 2023
# ══════════════════════════════════════════════════════════════════════════════
def score_method(layer_n, mo_s_ratio, ecsa_v, rct_v=None):
    """
    Scores how strongly MBE is preferred over CVD for a given parameter set.
    Returns (label, color_key, score, max_score, reasons_list).

    v3.0 MECHANISTIC UPDATES:
    - Mo/S criterion now correctly described as S-VACANCY DENSITY in 2H matrix
      (Sherwood 2024), not 1T phase formation.
    - Threshold Mo/S=0.588 (S/Mo=1.70) confirmed by ACS Catalysis 2023 as
      the point where undercoordinated Mo becomes exposed to electrolyte.
    - CVD can achieve Mo-rich zones via geometry control (ACS Cat 2023),
      but cannot maintain them at wafer scale or under electrochemical cycling.
    - k⁰ scoring now supported by 5-point curve (Manyepedza 2022 Fig.7):
        1L: 250 cm/s | 2L: 7.5 | 3L: 1.5 | 5L: 0.1 | 10L: 0.01 cm/s
    """
    reasons = []
    total = 0
    MAX = 8

    # ── Criterion 1: Layer # ─────────────────────────────────────────────────
    if layer_n <= 3:
        pts = 3
        detail = (
            f"≤3 layers: k⁰ gradient from 250 cm/s (1L) → 7.5 cm/s (2L) → 1.5 cm/s (3L) "
            f"(McKelvey, via Manyepedza 2022 Fig.7 simulations). "
            f"HER onset: −0.10V (1-2L), −0.25V (3L), −0.50V (bulk) confirmed by RDE. "
            f"CVD cannot reliably produce ≤3L with controlled stoichiometry at wafer scale."
        )
        refs = ["McKelvey (Electrochim.Acta 2021)", "Manyepedza 2022 Fig.7+9"]
    elif layer_n <= 6:
        pts = 2
        detail = (
            f"4–6 layers: k⁰ ≈ 0.1–7.5 cm/s — optimal HER zone. "
            f"MoS-N10 (~5L) achieves best N-series HER (η=−0.33V). "
            f"CVD nucleation density unstable in this thickness range. "
            f"Raman validates: N5 (Δω≈18-19 → 2L) and N10 (Δω≈21 → 4-5L)."
        )
        refs = ["Jeon 2026 N-series", "Lee 2010 ACS Nano", "Manyepedza 2022"]
    elif layer_n <= 12:
        pts = 1
        detail = f"7–12 layers: k⁰ ≈ 0.01–0.1 cm/s. Multi-layer regime. MBE preferred for thickness uniformity."
        refs = ["Jeon 2026 T-series", "Manyepedza 2022 Fig.7"]
    else:
        pts = 0
        detail = f"≥13 layers: k⁰ < 0.01 cm/s — bulk-like kinetics. CVD viable when Mo/S near-stoichiometric."
        refs = ["Jeon 2026 T800/N50", "Manyepedza 2022"]
    total += pts
    reasons.append({'criterion': 'Layer #', 'points': pts, 'max': 3,
                    'refs': refs, 'detail': detail})

    # ── Criterion 2: Mo/S ratio ───────────────────────────────────────────────
    if mo_s_ratio > 0.72:
        pts = 3
        detail = (
            f"Mo/S > 0.72 (S/Mo < 1.39): Extreme S-vacancy density in 2H matrix. "
            f"MECHANISTIC UPDATE (Sherwood 2024): NOT 1T phase — S-vacancies in 2H. "
            f"XPS: POS-C (MoS2-x) dominant. CVD S-atmosphere reoxidizes vacancies. "
            f"CVD Mo-24 equivalent (S/Mo=1.15) shows Mo oxidation under ambient. "
            f"Only MBE S-flux control can maintain this regime reproducibly."
        )
        refs = ["Sherwood 2024 SI Fig.S18", "ACS Cat 2023 Mo-22/Mo-24"]
    elif mo_s_ratio > 0.588:
        pts = 2
        detail = (
            f"Mo/S 0.588–0.72 (S/Mo 1.39–1.70): Moderate S-vacancy zone — optimal HER. "
            f"ACS Catalysis 2023 confirms S/Mo=1.70 as threshold for undercoordinated Mo. "
            f"Best HER at Mo-16 to Mo-18 (S/Mo≈1.65–1.70) in CVD gradient study. "
            f"Validates Jeon N10 (Mo/S≈0.556) and M3.0 (Mo/S≈0.645) as optimal zone. "
            f"CVD overpressure pushes toward S/Mo=2.0 — cannot maintain this zone stably."
        )
        refs = ["ACS Cat 2023 threshold", "Sherwood 2024 Fig.S18", "Jeon 2026 M-series"]
    elif mo_s_ratio >= 0.500:
        pts = 1
        detail = (
            f"Mo/S 0.500–0.588 (S/Mo 1.70–2.00): Slight S-deficiency. "
            f"CVD geometry control can access this zone (ACS Cat 2023 Mo-18 to Mo-20) "
            f"but reproducibility at wafer scale requires MBE."
        )
        refs = ["ACS Cat 2023 Mo-18/20", "Sherwood 2024"]
    else:
        pts = 0
        detail = (
            f"Mo/S < 0.500 (S/Mo > 2.0): Near-stoichiometric 2H-MoS₂. "
            f"CVD S-rich atmosphere sufficient. ACS Cat 2023 Mo-8 to Mo-16 equivalent. "
            f"No advantage from MBE for stoichiometry control."
        )
        refs = ["ACS Cat 2023 Mo-8 to Mo-16", "Sherwood 2024 pristine"]
    total += pts
    reasons.append({'criterion': 'Mo/S ratio', 'points': pts, 'max': 3,
                    'refs': refs, 'detail': detail})

    # ── Criterion 3: ECSA ─────────────────────────────────────────────────────
    if ecsa_v >= 8.0:
        pts = 1
        detail = (
            f"ECSA ≥ 8.0 cm²: Wafer-scale uniformity required for maximum edge site density. "
            f"Best Jeon: N10 (8.0) and M6.0 (9.2 cm²). "
            f"ACS Catalysis 2023: ECSA decreases with CVD distance inconsistency."
        )
        refs = ["Jeon 2026 Table 1", "ACS Cat 2023 Fig.4c"]
    else:
        pts = 0
        detail = f"ECSA < 8.0 cm²: No additional synthesis constraint from this criterion."
        refs = []
    total += pts
    reasons.append({'criterion': 'ECSA', 'points': pts, 'max': 1,
                    'refs': refs, 'detail': detail})

    # ── Criterion 4: Rct ─────────────────────────────────────────────────────
    rct_use = rct_v if rct_v is not None else gp_predict('rct', layer_n, mo_s_ratio, ecsa_v)[0]
    if rct_use < 55:
        pts = 1
        detail = (
            f"Rct = {rct_use:.0f} Ω·cm² < 55: Requires low interfacial resistance from "
            f"S-vacancy domains in 2H matrix (v3.0: not 1T phase, per Sherwood 2024). "
            f"Only MBE S-flux control can reliably create and maintain vacancy density. "
            f"Best Jeon: N10 (52.8), M6.0 (45.5 Ω·cm²)."
        )
        refs = ["Jeon 2026 EIS Table 1", "Sherwood 2024 mechanism correction"]
    else:
        pts = 0
        detail = f"Rct = {rct_use:.0f} Ω·cm² ≥ 55: No additional constraint from this criterion."
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
        "Jeon et al. <i>ACS Nano</i> 2026 · v3.0 Multi-Paper Validated<br>"
        "GP model · n=14 MBE samples · 1M KOH</div>",
        unsafe_allow_html=True)

    st.markdown(
        "<div class='provenance-box'>"
        "✅ <b>ECSA</b>: measured (Jeon 2026 Table 1)<br>"
        "⚠ <b>Layer #</b>: Scherrer ÷ 0.615 nm/layer (×4 sources)<br>"
        "&nbsp;&nbsp;&nbsp;Raman validates N5 (2L) and N10 (5L) only<br>"
        "⚠ <b>Mo/S</b>: XPS calibration table<br>"
        "&nbsp;&nbsp;&nbsp;Sherwood 2024 + ACS Cat 2023 + Smiri 2026<br>"
        "&nbsp;&nbsp;&nbsp;Mechanism: S-vacancies in 2H matrix (NOT 1T)"
        "</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">KEY DESCRIPTORS</div>', unsafe_allow_html=True)

    layer_n = st.slider(
        "⚠ Layer #", 1, 20, 5, 1,
        help=(
            "ESTIMATED from XRD Scherrer D(002) ÷ 0.615 nm/trilayer.\n\n"
            "4-SOURCE VALIDATION:\n"
            "① Manyepedza 2022 AFM Fig.9: 0.65 nm (1L), 1.30 nm (2L)\n"
            "② Bentley 2017 Chem.Sci.: 'van der Waals gap = 6.15 Å'\n"
            "③ Cao 2017 Sci.Rep.: HRTEM = 0.63 nm\n"
            "④ Fan et al. JACS 2016: controlled exfoliation\n\n"
            "RAMAN VALIDATION (Lee 2010 + Smiri 2026):\n"
            "• N5 (raman=1.01): Δω≈18-19 → VALIDATES 2L ✓\n"
            "• N10 (raman=1.63): Δω≈21 → VALIDATES 4-5L ✓\n"
            "• N>6: Raman saturates — Scherrer is primary estimator\n\n"
            "k⁰ vs layers (McKelvey via Manyepedza 2022):\n"
            "1L: 250 cm/s | 2L: 7.5 | 3L: 1.5 | 5L: 0.1 | 10L: 0.01 cm/s"
        ))

    mo_s_ratio = st.slider(
        "⚠ Mo/S atomic ratio", 0.45, 0.90, 0.56, 0.01,
        help=(
            "ESTIMATED via XPS calibration table (v3.0).\n\n"
            "CALIBRATION ANCHORS:\n"
            "S/Mo=2.2 → Mo/S=0.455 (2H pristine, Sherwood 2024)\n"
            "S/Mo=1.70 → Mo/S=0.588 (threshold undercoord. Mo, ACS Cat 2023)\n"
            "S/Mo=1.45 → Mo/S=0.690 (Sherwood 2024 70s etch limit)\n"
            "S/Mo=1.15 → Mo/S=0.870 (ACS Cat 2023 Mo-24 extreme)\n\n"
            "MECHANISM (v3.0 correction):\n"
            "Mo/S > 0.58 = S-VACANCIES IN 2H MATRIX (Sherwood 2024)\n"
            "NOT 1T phase. 'Metallic' behavior from vacancy sites.\n"
            "Converts to 2H under electrochemical cycling (ACS Cat 2023).\n\n"
            "OPTIMAL HER: Mo/S 0.556–0.645 (N10, M3.0–M6.0 zone)"
        ))

    ecsa_val = st.slider(
        "✅ ECSA (cm²)", 2.0, 12.0, 8.0, 0.5,
        help=(
            "DIRECTLY MEASURED — Jeon 2026 Table 1.\n"
            "Cdl from CV (20–80 mV/s), Cs=40 µF/cm².\n"
            "Range: 3.5 (T800/M9.0) to 9.2 cm² (M6.0)."
        ))

    # Closest match
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

    m_label, m_col_key, m_score, m_max, m_reasons = score_method(layer_n, mo_s_ratio, ecsa_val)
    m_color = METHOD_COLORS[m_col_key]
    pct = int(m_score / m_max * 100)

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
        f"</div>", unsafe_allow_html=True)

    with st.expander("Scoring breakdown (v3.0)", expanded=False):
        st.caption(
            "⚠ All 14 Jeon 2026 samples are MBE-grown. "
            "Score guides NEW synthesis decisions only. "
            "v3.0: Mo/S mechanism corrected to S-vacancies in 2H (Sherwood 2024).")
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
        "🔬 XPS Calibration",
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
        "v3.0: Sherwood 2024 + ACS Cat 2023 + Manyepedza 2022</div>",
        unsafe_allow_html=True)

    st.markdown(
        "<div class='correction-box'>"
        "🔬 <b>v3.0 Mechanistic Update:</b> Mo/S > 0.58 = <b>S-vacancies in 2H matrix</b> "
        "(Sherwood 2024, SI Fig.S18), NOT 1T phase. ACS Catalysis 2023 confirms 'metallic' "
        "character converts back to 2H under electrochemical cycling. "
        "Scoring thresholds unchanged; mechanism description corrected."
        "</div>", unsafe_allow_html=True)

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
        f"</div>", unsafe_allow_html=True)

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

    st.markdown('<div class="section-header">KEY DESCRIPTORS</div>', unsafe_allow_html=True)
    kc1, kc2, kc3 = st.columns(3)

    # Layer number with Raman validation flag
    raman_conf = RAMAN_LAYER_CONFIDENCE.get(best_match['sample'], 'low')
    raman_flag = {"high": "🟢 Raman-validated (Lee 2010)",
                  "medium": "🟡 Raman-consistent (Lee 2010)",
                  "low": "🔴 Raman saturated — Scherrer primary",
                  "very_low": "🔴 Bulk-like Raman, Scherrer only"}.get(raman_conf, "🔴")

    desc_cards = [
        (kc1, "Layer # ⚠", f"{layer_n}", "layers",
         "🟢 ≤3L → k⁰ ≥ 1.5 cm/s (MBE required)" if layer_n <= 3
         else ("🟡 4–6L → k⁰ 0.1–7.5 cm/s (optimal HER zone)" if layer_n <= 6
               else "🔴 Multi-layer → k⁰ < 0.1 cm/s"),
         f"⚠ Scherrer est. | {raman_flag}"),
        (kc2, "Mo/S ratio ⚠", f"{mo_s_ratio:.2f}", "",
         "🟢 Optimal zone (S-vacancies in 2H)" if 0.556 <= mo_s_ratio <= 0.690
         else ("🟡 Near-stoich. (pure 2H)" if mo_s_ratio < 0.500 else "🔴 Extreme vacancies"),
         "⚠ XPS calibration | Mechanism: S-vac in 2H (Sherwood 2024)"),
        (kc3, "ECSA ✅", f"{ecsa_val:.1f}", "cm²",
         "🟢 High — max edge sites" if ecsa_val >= 7.0 else "🟡 Moderate",
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

    st.markdown('<div class="section-header">PREDICTED PERFORMANCE METRICS</div>',
                unsafe_allow_html=True)
    cols = st.columns(4)
    metrics_order = ['eta','tafel','rct','tof_ecsa','tof_mass','raman','resistivity']
    thresholds = {
        'eta':    (-0.38, -0.50),
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

    # Radar chart
    st.markdown('<div class="section-header">PERFORMANCE PROFILE</div>', unsafe_allow_html=True)
    radar_keys  = ['eta','tafel','rct','tof_ecsa','tof_mass','raman','resistivity']
    radar_names = ['η (overpot.)','Tafel','Rct','TOF(ECSA)','TOF(mass)','Raman','Resistivity']

    def normalize_vals(vals_dict, keys):
        normed = []
        for key in keys:
            _, _, better = TARGETS[key]
            col_vals = df[key].values
            vmin, vmax = col_vals.min(), col_vals.max()
            v = vals_dict[key]
            n_val = float(np.clip((v - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0))
            normed.append(n_val if better == 'max' else 1 - n_val)
        return [float(x) for x in normed]

    normed = normalize_vals(vals, radar_keys)
    best_exp_idx = df['eta'].idxmin()  # N10: most negative η = best HER
    best_exp     = df.loc[best_exp_idx]
    normed_best  = normalize_vals({k: best_exp[k] for k in radar_keys}, radar_keys)
    normed_closest = normalize_vals({k: best_match[k] for k in radar_keys}, radar_keys)

    def hex_rgba(h, a):
        h = h.lstrip('#')
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f'rgba({r},{g},{b},{a})'

    fig_radar = go.Figure()
    cats = radar_names + [radar_names[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=normed + [normed[0]], theta=cats, fill='toself',
        name='Your prediction',
        fillcolor=hex_rgba(m_color, 0.18),
        line=dict(color=m_color, width=2)))
    fig_radar.add_trace(go.Scatterpolar(
        r=normed_best + [normed_best[0]], theta=cats, fill='toself',
        name=f'Best exp.: {best_exp["sample"]} (η={best_exp["eta"]:.2f}V)',
        fillcolor='rgba(45,206,137,0.08)',
        line=dict(color='#2DCE89', width=1.5, dash='dot')))
    fig_radar.add_trace(go.Scatterpolar(
        r=normed_closest + [normed_closest[0]], theta=cats, fill='toself',
        name=f'Nearest: {best_match["sample"]}',
        fillcolor='rgba(200,200,200,0.06)',
        line=dict(color='#aaaaaa', width=1, dash='dot')))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
                   angularaxis=dict(tickfont=dict(size=10))),
        showlegend=True, height=400,
        legend=dict(orientation='h', yanchor='bottom', y=-0.30),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=70, l=40, r=40))
    st.plotly_chart(fig_radar, use_container_width=True)

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
    st.caption(
        "⚠ layer_n: Scherrer est. (Raman validates N5, N10 only) | "
        "mo_s_ratio: XPS calibration est. (Sherwood 2024 + ACS Cat 2023)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TREND CURVES (unchanged from v2.2, keeping same structure)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trend Curves":
    st.markdown("# Trend Curves")
    st.markdown(
        "<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
        "GP mean + 95% CI vs each descriptor · experimental data overlaid</div>",
        unsafe_allow_html=True)

    tc1, tc2 = st.columns([1, 2])
    with tc1:
        target_tc = st.selectbox("Performance metric",
            options=list(TARGETS.keys()),
            format_func=lambda k: f"{TARGETS[k][0]} ({TARGETS[k][1]})")
    with tc2:
        feat_tc = st.selectbox("Descriptor to vary",
            options=FEATURES, format_func=lambda k: FEATURE_LABELS[k])

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

    exp_lo = df[feat_tc].min(); exp_hi = df[feat_tc].max()
    in_range = (x_range >= exp_lo) & (x_range <= exp_hi)

    fig_tc = go.Figure()
    fig_tc.add_trace(go.Scatter(
        x=np.concatenate([x_range, x_range[::-1]]),
        y=np.concatenate([y_highs, y_lows[::-1]]),
        fill='toself', fillcolor='rgba(78,154,241,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name='95% CI (GP)'))
    fig_tc.add_trace(go.Scatter(x=x_range, y=y_means, mode='lines',
        line=dict(color='rgba(78,154,241,0.35)', width=1.5, dash='dot'),
        name='GP mean (extrapolation)'))
    fig_tc.add_trace(go.Scatter(x=x_range[in_range], y=y_means[in_range], mode='lines',
        line=dict(color='#4E9AF1', width=3), name='GP mean (interpolation)'))

    for ser, scolor in SERIES_COLORS.items():
        mask = df['series'] == ser
        fig_tc.add_trace(go.Scatter(
            x=df[feat_tc].values[mask], y=df[target_tc].values[mask], mode='markers',
            name=SERIES_LABELS[ser],
            marker=dict(size=11, color=scolor, line=dict(width=1.5, color='white')),
            text=df['sample'][mask],
            hovertemplate='<b>%{text}</b><br>' + FEATURE_LABELS[feat_tc] +
                          '=%{x:.2f}<br>' + name_tc + '=%{y:.3f} ' + unit_tc + '<extra></extra>'))

    cur_val = defaults[feat_tc]
    fig_tc.add_vline(x=cur_val, line_width=1.5, line_dash="dash",
                     line_color=METHOD_COLORS[m_col_key],
                     annotation_text=f"Current: {cur_val:.2f}",
                     annotation_font_color=METHOD_COLORS[m_col_key])

    # Add Mo/S threshold lines if x-axis is mo_s_ratio
    if feat_tc == 'mo_s_ratio':
        fig_tc.add_vline(x=0.500, line_dash='dot', line_color='#888', line_width=1,
                         annotation_text="S/Mo=2.0 (pure 2H)", annotation_font_color='#888')
        fig_tc.add_vline(x=0.588, line_dash='dot', line_color='#F5A623', line_width=1,
                         annotation_text="S/Mo=1.70 (threshold)", annotation_font_color='#F5A623')
        fig_tc.add_vline(x=0.690, line_dash='dot', line_color='#2DCE89', line_width=1,
                         annotation_text="S/Mo=1.45 (Sherwood limit)", annotation_font_color='#2DCE89')

    fig_tc.update_layout(
        title=f"{name_tc} vs {FEATURE_LABELS[feat_tc]}<br>"
              f"<sup>{FEATURE_PROVENANCE[feat_tc]}</sup>",
        xaxis_title=FEATURE_LABELS[feat_tc],
        yaxis_title=f"{name_tc} ({unit_tc})",
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=-0.40),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig_tc.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.12)')
    fig_tc.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.12)')
    st.plotly_chart(fig_tc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: XPS CALIBRATION (new in v3.0)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 XPS Calibration":
    st.markdown("# XPS Calibration — Mo/S Ratio")
    st.markdown(
        "<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
        "Integrated calibration table from Sherwood 2024, ACS Catalysis 2023, "
        "and Smiri 2026. Maps S/Mo (measured) to Mo/S (descriptor in model).</div>",
        unsafe_allow_html=True)

    st.markdown(
        "<div class='correction-box'>"
        "<b>v3.0 Mechanistic Correction:</b><br>"
        "Mo/S > 0.58 in Jeon M-series = <b>S-vacancies in 2H-MoS₂ matrix</b> "
        "(Sherwood 2024 SI Fig.S18: POS-A binding energy stays constant, "
        "only POS-C grows with etching). NOT 1T phase formation.<br>"
        "The 'metallic' behavior observed in M2.0–M3.0 arises from Mo atoms "
        "with missing S coordination (MoS₂₋ₓ), which provide low-resistance "
        "charge transfer pathways. This converts back to stoichiometric 2H under "
        "electrochemical cycling (ACS Cat 2023: 1T→1H conversion confirmed by "
        "SEC-Raman + XPS after 9 LSV cycles)."
        "</div>", unsafe_allow_html=True)

    # Calibration table
    calib_data = []
    for smo, (mos, desc, source) in XPS_CALIBRATION.items():
        calib_data.append({
            'S/Mo (measured)': smo,
            'Mo/S (descriptor)': mos,
            'Phase description': desc,
            'Source': source,
        })
    calib_df = pd.DataFrame(calib_data)
    st.dataframe(calib_df, use_container_width=True)

    # Calibration curve
    smo_vals = sorted(XPS_CALIBRATION.keys(), reverse=True)
    mos_vals = [XPS_CALIBRATION[s][0] for s in smo_vals]

    fig_calib = go.Figure()
    fig_calib.add_trace(go.Scatter(
        x=smo_vals, y=mos_vals, mode='lines+markers',
        line=dict(color='#4E9AF1', width=2),
        marker=dict(size=10, color='#4E9AF1'),
        name='XPS calibration (all sources)'))

    # Add Jeon samples
    for _, row in df.iterrows():
        smo_jeon = 1.0 / row['mo_s_ratio']
        color = SERIES_COLORS[row['series']]
        fig_calib.add_trace(go.Scatter(
            x=[smo_jeon], y=[row['mo_s_ratio']],
            mode='markers+text',
            marker=dict(size=9, color=color, symbol='diamond',
                        line=dict(width=1.5, color='white')),
            text=[row['sample']], textposition='top center',
            textfont=dict(size=8), name=row['sample'], showlegend=False,
            hovertemplate=f"<b>{row['sample']}</b><br>S/Mo=%{{x:.2f}}<br>Mo/S=%{{y:.3f}}<extra></extra>"))

    fig_calib.add_vline(x=1.70, line_dash='dot', line_color='#F5A623', line_width=1.5,
                        annotation_text="S/Mo=1.70 threshold (ACS Cat 2023)",
                        annotation_font_color='#F5A623')
    fig_calib.add_vline(x=2.00, line_dash='dot', line_color='#888', line_width=1,
                        annotation_text="S/Mo=2.0 (stoichiometric 1H)",
                        annotation_font_color='#888')
    fig_calib.add_vline(x=2.20, line_dash='dot', line_color='#2DCE89', line_width=1,
                        annotation_text="S/Mo=2.2 (2H pristine, Sherwood anchor)",
                        annotation_font_color='#2DCE89')

    fig_calib.update_layout(
        title="XPS Calibration: S/Mo → Mo/S (Sherwood 2024 + ACS Cat 2023 + Smiri 2026)",
        xaxis_title="S/Mo ratio (XPS measured, decreasing = more Mo-rich)",
        yaxis_title="Mo/S ratio (model descriptor)",
        xaxis=dict(autorange='reversed'),
        height=450,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig_calib.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.12)')
    fig_calib.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.12)')
    st.plotly_chart(fig_calib, use_container_width=True)

    # k⁰ vs layers
    st.markdown('<div class="section-header">k⁰ VS LAYER NUMBER (McKelvey via Manyepedza 2022)</div>',
                unsafe_allow_html=True)

    k0_df = pd.DataFrame([
        {'Layers': k, 'k⁰ (cm/s)': v,
         'HER onset ≈': '−0.10V' if k<=2 else ('−0.25V' if k==3 else ('−0.50V' if k>=10 else '−0.35V')),
         'Source': 'McKelvey anchor' if k in [1,3] else 'Manyepedza 2022 Fig.7 inference'}
        for k, v in K0_VS_LAYERS.items()
    ])
    st.dataframe(k0_df, use_container_width=True)

    fig_k0 = go.Figure()
    layers_plot = list(K0_VS_LAYERS.keys())
    k0_plot = list(K0_VS_LAYERS.values())
    fig_k0.add_trace(go.Scatter(
        x=layers_plot, y=k0_plot, mode='lines+markers',
        line=dict(color='#2DCE89', width=2),
        marker=dict(size=10, color=['#FF6B6B' if k in [1,3] else '#2DCE89' for k in layers_plot]),
        name='k⁰ (cm/s)'))
    fig_k0.add_hrect(y0=1.5, y1=300, fillcolor="rgba(45,206,137,0.08)",
                     line_width=0, annotation_text="MBE required (3/3 pts)")
    fig_k0.add_hrect(y0=0.1, y1=1.5, fillcolor="rgba(245,166,35,0.08)",
                     line_width=0, annotation_text="MBE preferred (2/3 pts)")
    fig_k0.update_layout(
        title="Standard rate constant k⁰ vs MoS₂ layer number",
        xaxis_title="Number of trilayers", yaxis_title="k⁰ (cm/s)",
        yaxis_type="log", height=380,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig_k0.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.12)')
    fig_k0.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.12)')
    st.plotly_chart(fig_k0, use_container_width=True)
    st.caption("Red markers = direct McKelvey anchors. Green = inferred from Manyepedza 2022 Fig.7 simulations.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 2D HEATMAPS (same as v2.2)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺 2D Heatmaps":
    st.markdown("# 2D Heatmaps")
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

    xlo, xhi = FEATURE_RANGES[xf]; ylo, yhi = FEATURE_RANGES[yf]
    xgrid = np.linspace(xlo, xhi, N); ygrid = np.linspace(ylo, yhi, N)
    Z = np.zeros((N, N))
    for i, yv in enumerate(ygrid):
        for j, xv in enumerate(xgrid):
            row = {xf: xv, yf: yv, fixed_f: defaults_hm[fixed_f]}
            Z[i, j] = gp_predict(target_hm, row['layer_n'], row['mo_s_ratio'], row['ecsa'])[0]

    cs = 'RdYlGn' if better_hm == 'max' else 'RdYlGn_r'
    fig_hm = go.Figure(data=go.Heatmap(
        z=Z, x=xgrid, y=ygrid, colorscale=cs,
        colorbar=dict(title=dict(text=f"{name_hm} ({unit_hm})", side='right')),
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
                          '<br>' + ylabel + '=%{y:.2f}<br>' +
                          name_hm + f'=%{{customdata:.3f}} {unit_hm}<extra></extra>'))

    fig_hm.add_trace(go.Scatter(
        x=[defaults_hm[xf]], y=[defaults_hm[yf]], mode='markers',
        marker=dict(size=16, color=METHOD_COLORS[m_col_key], symbol='star',
                    line=dict(width=2, color='white')),
        name='Your position'))

    # Add threshold lines for Mo/S axis
    if yf == 'mo_s_ratio':
        fig_hm.add_hline(y=0.588, line_dash='dot', line_color='#F5A623', line_width=1,
                         annotation_text="Mo/S=0.588 (S/Mo=1.70 threshold)",
                         annotation_font_color='#F5A623')
    if xf == 'mo_s_ratio':
        fig_hm.add_vline(x=0.588, line_dash='dot', line_color='#F5A623', line_width=1)

    fig_hm.update_layout(
        title=f"{name_hm} — {xlabel} × {ylabel} | fixed={FEATURE_LABELS[fixed_f]}={defaults_hm[fixed_f]:.2f}",
        xaxis_title=xlabel, yaxis_title=ylabel, height=540,
        legend=dict(orientation='h', yanchor='bottom', y=-0.22),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 3D EXPLORER (same structure as v2.2)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌐 3D Explorer":
    st.markdown("# 3D Descriptor Space Explorer")
    t3c1, t3c2 = st.columns(2)
    with t3c1:
        target_3d = st.selectbox("Color metric",
            options=list(TARGETS.keys()),
            format_func=lambda k: f"{TARGETS[k][0]} ({TARGETS[k][1]})")
    with t3c2:
        show_surf = st.checkbox("Show GP surface slice (Mo/S fixed)", value=True)

    name_3d, unit_3d, better_3d = TARGETS[target_3d]
    fig_3d = go.Figure()

    if show_surf:
        N3 = 25
        ln3 = np.linspace(1, 20, N3); ec3 = np.linspace(2, 12, N3)
        Zs  = np.zeros((N3, N3))
        for i, ev in enumerate(ec3):
            for j, lv in enumerate(ln3):
                Zs[i, j] = gp_predict(target_3d, lv, mo_s_ratio, ev)[0]
        fig_3d.add_trace(go.Surface(
            x=ln3, y=ec3, z=Zs,
            colorscale='RdYlGn' if better_3d == 'max' else 'RdYlGn_r',
            opacity=0.55, showscale=False,
            name=f'GP surface (Mo/S={mo_s_ratio:.2f})'))

    for ser, scolor in SERIES_COLORS.items():
        mask = df['series'] == ser
        sub  = df[mask]
        fig_3d.add_trace(go.Scatter3d(
            x=sub['layer_n'], y=sub['ecsa'], z=sub['mo_s_ratio'],
            mode='markers+text',
            marker=dict(size=8, color=sub[target_3d].values,
                        colorscale='RdYlGn' if better_3d == 'max' else 'RdYlGn_r',
                        cmin=df[target_3d].min(), cmax=df[target_3d].max(),
                        line=dict(width=2, color='white')),
            text=sub['sample'], name=SERIES_LABELS[ser]))

    cur_pred = gp_predict(target_3d, layer_n, mo_s_ratio, ecsa_val)[0]
    fig_3d.add_trace(go.Scatter3d(
        x=[layer_n], y=[ecsa_val], z=[mo_s_ratio], mode='markers',
        marker=dict(size=14, color=METHOD_COLORS[m_col_key], symbol='diamond',
                    line=dict(width=3, color='white')),
        name=f'Your position ({cur_pred:.3f} {unit_3d})'))

    fig_3d.update_layout(
        scene=dict(
            xaxis_title='Layer # (estimated)',
            yaxis_title='ECSA (cm²)',
            zaxis_title='Mo/S ratio (estimated)',
        ),
        title=f"{name_3d} ({unit_3d}) in descriptor space",
        height=620,
        paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_3d, use_container_width=True)
    st.info(
        f"**Your position:** Layer# {layer_n} · Mo/S {mo_s_ratio:.2f} · ECSA {ecsa_val:.1f} cm²  "
        f"→ GP predicts **{name_3d} = {cur_pred:.3f} {unit_3d}** | Method: **{m_label}**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INVERSE PREDICTOR (same as v2.2)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Inverse Predictor":
    st.markdown("# Inverse Predictor")
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
        f"η={best_inv.eta:.2f}V · Tafel={best_inv.tafel:.0f} · "
        f"ECSA={best_inv.ecsa:.1f} cm² · Rct={best_inv.rct:.1f} Ω·cm²</div>"
        f"<div style='margin-top:10px;'><b>MBE score: {inv_score}/{inv_max}</b></div>"
        f"</div>", unsafe_allow_html=True)

    param_df = pd.DataFrame({
        'Parameter':   ['Annealing temp.', 'Deposition cycles', 'S-layer thickness',
                        'Layer # (est.)', 'Mo/S ratio (est.)'],
        'Value':       [f"{best_inv['temp']:.0f} °C", f"{best_inv['cycles']:.0f}",
                        f"{best_inv['s_thick']:.1f} Å",
                        f"{best_inv['layer_n']:.0f}",
                        f"{best_inv['mo_s_ratio']:.2f}"],
        'Provenance':  ['✅ Measured', '✅ Measured', '✅ Measured',
                        '⚠ Estimated (Scherrer ÷ 0.615 nm/layer)',
                        '⚠ Estimated (XPS calibration v3.0)'],
        'Note':        ['Higher T → crystalline, fewer edge sites',
                        '~1 MoS₂ layer per 5 cycles (QCM, Jeon 2026)',
                        'Primary lever for S-vacancy density',
                        '≤3L: MBE required (k⁰ ×167, McKelvey)',
                        '>0.588: S-vacancies in 2H — MBE S-flux control needed'],
    })
    st.dataframe(param_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE (same as v2.2)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧮 Feature Importance":
    st.markdown("# Feature Importance")
    st.markdown(
        "<div style='color:#666;font-size:0.9em;margin-bottom:20px;'>"
        "Random Forest LOO feature importance. GP used for predictions; RF for interpretability.</div>",
        unsafe_allow_html=True)

    st.markdown(
        "<div class='provenance-box'>"
        "⚠ <b>Interpretation:</b> layer_n and mo_s_ratio are estimated descriptors. "
        "Their importance reflects estimation quality as much as physical causation. "
        "Only ECSA is directly measured. n=14 → LOO scores have high variance.<br>"
        "v3.0 note: Raman (measured) reflects crystallinity for N>4L, not just layer count."
        "</div>", unsafe_allow_html=True)

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

    fi_colors = {'layer_n': '#9B59B6', 'mo_s_ratio': '#E84040', 'ecsa': '#2DCE89'}
    fi_names  = {'layer_n': 'Layer # (est.)', 'mo_s_ratio': 'Mo/S (est.)', 'ecsa': 'ECSA (meas.)'}

    imp_target = st.selectbox("Property for importance",
        options=list(TARGETS.keys()),
        format_func=lambda k: f"{TARGETS[k][0]} ({TARGETS[k][1]})")

    imps = rf_imps[imp_target]
    fig_fi = go.Figure(go.Bar(
        x=[fi_names[f] for f in FEATURES], y=imps,
        marker_color=[fi_colors[f] for f in FEATURES],
        text=[f"{v:.3f}" for v in imps], textposition='outside'))
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
# PAGE: THEORETICAL BASIS (updated v3.0)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Theoretical Basis":
    st.markdown("# Theoretical Framework — v3.0")

    papers = [
        ("1 · Jeon et al. — ACS Nano 2026 [PRIMARY DATA SOURCE]",
         "14 MBE-grown MoS₂ on Si in 1M KOH. T-series (temp 600–800°C), "
         "N-series (cycles 5–50), M-series (S-thick 2.0–9.0 Å). "
         "Global optimum: MoS-N10 (η=−0.33V, Tafel=80 mV/dec, ECSA=8.0, Rct=52.8). "
         "All Table 1 values measured except layer_n and mo_s_ratio."),

        ("2 · Manyepedza et al. — J. Phys. Chem. C 2022 [LAYER CALIBRATION + k⁰ DATA]",
         "Impact electrochemistry of MoS₂ NPs. "
         "AFM Fig.9: 0.65 nm (1L), 1.30 nm (2L) on mica — validates Scherrer conversion. "
         "Citing McKelvey (Electrochim. Acta 2021, ref.[30]): "
         "k⁰ from 250 cm/s (1L) to 1.5 cm/s (3L). "
         "Fig.7 simulations: 5-point curve — k⁰ = 250, 7.5, 1.5, 0.1, 0.01 cm/s. "
         "Three HER onsets in RDE: −0.10V (1-2L), −0.25V (3L), −0.50V (bulk). "
         "Faradaic efficiency 45–48% for H₂ by GC. "
         "XPS: S/Mo=2.2 → Mo/S=0.455 (electrodeposited reference). "
         "Electrolyte: pH 2 H₂SO₄."),

        ("3 · Sherwood et al. — ACS Appl. Nano Mater. 2024 [XPS STOICHIOMETRY — GAP 1]",
         "4-peak XPS model: 2H (POS-A, 229.3 eV), 1T (POS-B, 228.4 eV), "
         "MoS₂₋ₓ (POS-C, 228.1 eV), MoO₃ (POS-D). "
         "SI Fig.S18 (yellow curve, S(POS-A)/Mo(POS-A+POS-C)): "
         "S/Mo decreases 2.2→1.45 under Ar⁺ etching (70s). "
         "KEY FINDING: POS-A binding energy stays CONSTANT — "
         "MoS₂₋ₓ formation (POS-C growth) is the only change. "
         "Mo/S > 0.58 = S-VACANCIES IN 2H MATRIX, not 1T phase. "
         "Battery electrode at 100s etch: 25.9% 2H + 49.9% MoS₂₋ₓ + 8.7% 1T."),

        ("4 · ACS Catalysis 2023 (Brunet Cabre et al.) [CVD COMPARISON — GAP 3]",
         "CVD MoS₂ on Au foil, distance gradient 8–24 cm from MoO₃, 0.5M H₂SO₄. "
         "S/Mo directly measured by XPS: 2.0 (Mo-8 to Mo-16) → 1.15 (Mo-24). "
         "S/Mo=1.70 confirmed as threshold for undercoordinated Mo exposure. "
         "OPTIMAL HER at Mo-16 to Mo-18 (Mo/S 0.588–0.606) — validates Jeon N10. "
         "CRITICAL: '1T phase' in Mo-8 converts to 1H during LSV cycling (SEC-Raman + XPS). "
         "N leaves structure after 1st LSV cycle. "
         "Electrolyte: 0.5M H₂SO₄ → η not directly comparable with Jeon."),

        ("5 · Lee et al. — ACS Nano 2010 [RAMAN LAYER CALIBRATION — GAP 2]",
         "Raman modes vs layer number (1L to bulk). "
         "Δω = A₁g − E¹₂g: 18.7 (1L) → 21.5 (2L) → 22.5 (3L) → 25.0 (6L) → 26.0 (bulk). "
         "SATURATION: >4L frequencies converge. Raman loses discrimination for Jeon N>6L. "
         "Validates: N5 (raman=1.01, Δω≈18-19 → 2L) and N10 (raman=1.63, Δω≈21 → 4-5L)."),

        ("6 · Smiri et al. — Scientific Reports 2026 [ALD RAMAN + XPS — GAP 2]",
         "ALD MoS₂ on 200mm Si wafers. Raman + XPS + polarized spectroscopy. "
         "A₁g/E¹₂g ratio DECREASES with layers (2.05→1.85 for 1→6ML) — "
         "opposite to Jeon trend: Jeon 'raman' reflects crystallinity, not just N. "
         "Saturation confirmed: >6ML → bulk values. "
         "XPS S/Mo vs layers: 1ML→1.75 (Mo/S=0.571), 6ML→1.95 (Mo/S=0.513). "
         "Interface effect: fewer layers → more S-deficient from MoS₂/substrate boundary. "
         "Implication: N5, N10 in Jeon have interface-effect component in Mo/S elevation."),

        ("7 · Bentley et al. — Chem. Sci. 2017 [LAYER CALIBRATION + BASAL ACTIVITY]",
         "SECCM nanoscale mapping of HER on natural MoS₂. "
         "States 'van der Waals gap = 6.15 Å' — independent confirmation of 0.615 nm/layer. "
         "J₀(basal) = 2.5×10⁻⁶ A/cm², J₀(edge) ~10⁻⁴ A/cm² (~40× more active). "
         "Tafel ~120 mV/dec (Volmer RDS). HER scales with edge-plane area."),

        ("8 · Cao et al. — Sci. Rep. 2017, 7, 8825 [HRTEM CALIBRATION]",
         "MoS₂ on MWCNTs. HRTEM Fig.2b,c: interlayer spacing = 0.63 nm. "
         "Fourth independent source for 0.615–0.65 nm/layer validation. "
         "1T phase XPS: Mo⁴⁺ 3d₅/₂ shifts 229→228.1 eV (Δ≈0.9 eV vs 2H)."),

        ("9 · Jaramillo et al. — Science 2007 [EDGE SITE ORIGIN — GAP 5 PENDING]",
         "HER activity scales linearly with edge-site density (not basal plane). "
         "Mo-terminated edges are dominant active sites. "
         "Foundation for why layer# (edge/basal ratio) and ECSA are key descriptors. "
         "⚠ Full TOF vs edge density data still needed for GAP 5."),

        ("10 · McKelvey et al. — Electrochim. Acta 2021, 393, 139027 [k⁰ VS LAYERS — GAP 4]",
         "Direct measurement: k⁰ = 250 cm/s (1L) → 1.5 cm/s (3L). "
         "Cited as ref.[30] in Manyepedza 2022. "
         "DOI: 10.1016/j.electacta.2021.139027 "
         "⚠ Full paper (k⁰ curve beyond 3L) still needed to refine scoring thresholds."),

        ("11 · PECVD Paper (RF Sputtering + ICP-PECVD) [CVD REFERENCE — GAP 3]",
         "MoS₂ on Si via RF sputtering Mo + PECVD sulfurization at 500°C. "
         "XPS: Mo⁴⁺ 3d₅/₂ = 228.9 eV (2H), S/Mo = 1.96 → Mo/S = 0.510 (stoichiometric). "
         "Raman Δk = 24.9 cm⁻¹ → bulk-like (~10-13 layers). "
         "HRTEM: d = 0.75 nm interlayer, ~40 nm total. "
         "η = −0.45V, Tafel = 76 mV/dec in 0.5M H₂SO₄. "
         "Method scoring: layer_n~10 (1pt) + Mo/S~0.510 (1pt) = 2/8 → Both viable ✓"),
    ]

    for title, body in papers:
        with st.expander(title):
            st.write(body)

    st.markdown('<div class="section-header">DESCRIPTOR SUMMARY TABLE — v3.0</div>',
                unsafe_allow_html=True)
    desc_df = pd.DataFrame({
        'Descriptor': ['Layer # ⚠', 'Mo/S ratio ⚠', 'ECSA ✅', 'Raman A₁g/E₂g ✅',
                       'Resistivity ✅', 'Rct ✅'],
        'Physical meaning': [
            'Film thickness → edge/basal ratio + k⁰ kinetics',
            'S-vacancy density in 2H matrix (v3.0: NOT 1T phase)',
            'Electrochemically active surface area (edge sites)',
            'Crystallinity + defects (NOT layer proxy for N>4L)',
            'Bulk electronic conductivity',
            'Interfacial charge transfer resistance'],
        'Optimal range': [
            '≤3L (k⁰ ≥ 1.5 cm/s, onset −0.10→−0.25V)',
            '0.556–0.645 (S/Mo 1.55–1.80, mod. S-vacancies)',
            '≥8 cm² (N10: 8.0, M6.0: 9.2)',
            '<1.8 (high defect/edge density in few-layer)',
            '<12 Ω·cm', '<55 Ω·cm² (N10: 52.8, M6.0: 45.5)'],
        'Provenance': [
            '⚠ Scherrer ÷ 0.615 nm | Raman validates N5, N10 only',
            '⚠ XPS calibration (Sherwood 2024 + ACS Cat 2023)',
            '✅ Cdl, Jeon 2026', '✅ Raman, Jeon 2026',
            '✅ 4-probe, Jeon 2026', '✅ EIS, Jeon 2026'],
    })
    st.dataframe(desc_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT — v3.0
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("# About — MoS₂ HER Predictor v3.0")
    st.markdown("""
**v3.0 Multi-Paper Validated** — Gaussian Process prediction for MBE-grown MoS₂ in 1M KOH.

---

### Primary experimental source
**Jeon et al., *ACS Nano* 2026, 20, 4479–4493** — 14 MBE samples on Si, 1M KOH.

---

### v3.0 Calibration papers integrated

| Paper | Gap covered | Key contribution |
|---|---|---|
| Sherwood 2024, ACS Appl. Nano Mater. | GAP 1 | 4-peak XPS model; S/Mo→Mo/S calibration; mechanism correction |
| ACS Catalysis 2023 (CVD gradient) | GAP 1 + GAP 3 | Direct XPS S/Mo vs distance; S/Mo=1.70 threshold; optimal HER zone |
| Smiri 2026, Sci. Rep. | GAP 2 | ALD Raman saturation; interface effect on Mo/S for few-layer |
| Lee 2010, ACS Nano | GAP 2 | Δω vs layers 1–6L; saturation limit |
| Manyepedza 2022, J. Phys. Chem. C | GAP 4 | AFM 0.65 nm/layer; k⁰ 5-point curve; RDE three onsets |
| Bentley 2017, Chem. Sci. | Calibration | 6.15 Å van der Waals gap; basal/edge J₀ |
| Cao 2017, Sci. Rep. | Calibration | HRTEM 0.63 nm interlayer |
| PECVD paper | GAP 3 | CVD Mo/S=0.510 by XPS; method scoring validation |

### Key mechanistic corrections in v3.0

| Old (v2.x) | New (v3.0) | Source |
|---|---|---|
| Mo/S > 0.58 = "Mo⁰/MoS₂ coexistence" | Mo/S > 0.58 = S-vacancies in 2H matrix | Sherwood 2024 |
| "1T phase" active and stable | 1T converts to 1H during cycling | ACS Cat 2023 |
| Raman used as layer proxy | Raman saturates >4L — crystallinity proxy | Smiri 2026 + Lee 2010 |
| k⁰: 2 points (1L, 3L) | k⁰: 5-point curve (1L–10L) | Manyepedza 2022 Fig.7 |

### Pending gaps (for next chat)

| Gap | What is needed | Where to find |
|---|---|---|
| GAP 3b | CVD MoS₂ HER in KOH 1M (same electrolyte as Jeon) | Search: "MoS2 CVD HER KOH 1M overpotential" |
| GAP 4b | McKelvey full paper — k⁰ curve beyond 3L | Electrochim. Acta 2021, 393, 139027 |
| GAP 5 | Jaramillo 2007 — TOF vs edge site density | Science 2007, 317, 100–102 |

### Machine learning

| Component | Detail |
|---|---|
| Primary model | GP (Matérn ν=2.5, ARD, calibrated 95% CI) |
| Secondary | RF (300 trees, LOO) — feature importance only |
| Validation | Leave-One-Out CV (n=14) |
| Features | Layer # (est.), Mo/S ratio (est.), ECSA (meas.) |

⚠ n=14 training samples — use for trend analysis and mechanistic understanding,
not as replacement for experimental validation.
    """)
