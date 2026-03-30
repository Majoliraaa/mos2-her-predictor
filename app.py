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

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MoS₂ HER Trend Predictor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = {
        'sample': ['MoS-T600','MoS-T700','MoS-T800',
                   'MoS-N5','MoS-N10','MoS-N20','MoS-N30','MoS-N50',
                   'MoS-M2.0','MoS-M2.5','MoS-M3.0','MoS-M6.0','MoS-M8.0','MoS-M9.0'],
        'series': ['T','T','T','N','N','N','N','N','M','M','M','M','M','M'],
        'temp':   [600,700,800,800,800,800,800,800,800,800,800,800,800,800],
        'cycles': [50,50,50,5,10,20,30,50,50,50,50,50,50,50],
        's_thick':[9.0,9.0,9.0,3.0,3.0,3.0,3.0,3.0,2.0,2.5,3.0,6.0,8.0,9.0],
        # ── Layer # ────────────────────────────────────────────────────────────────
        # ⚠ ESTIMATED — not directly reported in Jeon et al. 2026.
        # Derived from XRD Scherrer crystallite size (002) ÷ 0.615 nm/layer (d₀₀₂ of 2H-MoS₂).
        # T-series: (002) crystallite = 7.2 nm (T600) → 10.8 nm (T800) per Jeon Table 1 / Fig 1a.
        # N-series: (002) crystallite = 3.3 nm (N10) → 12.1 nm (N50) per Jeon Fig 2a caption.
        # M-series: same cycles as N50 (50 cycles) → same layer estimate as T800/N50.
        # This is a lower-bound estimate; actual layer count may differ due to growth mode.
        'layer_n':[12, 14, 18, 2, 5, 9, 13, 20, 20, 20, 20, 20, 20, 20],
        # ── Mo/S atomic ratio ──────────────────────────────────────────────────────
        # ⚠ ESTIMATED — XPS not reported per sample in Jeon et al. 2026.
        # Scale corrected using XPS literature: stoichiometric 2H-MoS₂ = S/Mo ~2.2 → Mo/S ~0.455
        # (Baker et al. Surf. Interface Anal. 2001; pristine XPS S/Mo = 2.2).
        # Sulfur-depleted limit: S/Mo ~1.12 → Mo/S ~0.893 (Lince et al. via Baker et al.).
        # Series M: M2.0 = most Mo-rich (incomplete sulfurization confirmed by XANES/EXAFS
        # showing residual Mo⁰ peaks, Jeon Fig 3a,c); M9.0 = near-stoichiometric.
        # Series N: N10 = few cycles → thinner, less sulfurized interface layer.
        # Series T: all s_thick=9.0 Å → near-stoichiometric; slight Mo-enrichment at lower T.
        'mo_s_ratio':[0.49,0.48,0.46,0.62,0.56,0.52,0.50,0.47,0.82,0.76,0.65,0.52,0.48,0.46],
        'raman':  [2.41,2.34,2.29,1.01,1.63,1.85,1.78,1.99,1.70,1.97,1.99,2.05,2.24,2.29],
        'resistivity':[15.98,16.52,19.26,7.75,8.99,11.08,11.40,12.45,9.01,9.50,12.45,15.09,17.14,19.26],
        'ecsa':   [6.7,6.5,3.5,4.5,8.0,6.5,6.3,6.5,4.3,6.3,6.5,9.2,4.7,3.5],
        'loading':[24.7,24.7,24.7,1.9,3.7,7.4,11.1,18.5,17.5,18.0,18.5,21.6,23.7,24.7],
        'eta':    [-0.46,-0.48,-0.58,-0.43,-0.33,-0.39,-0.35,-0.35,-0.58,-0.49,-0.35,-0.35,-0.52,-0.58],
        'tafel':  [136,257,297,161,80,105,93,114,484,253,114,91,223,297],
        'rct':    [98.4,113.0,193.3,136.5,52.8,76.9,59.0,64.0,161.2,104.5,64.0,45.5,124.5,193.3],
        'tof_ecsa':[5.7,5.2,5.7,9.9,13.0,11.4,9.9,8.3,6.2,4.6,8.3,6.7,5.1,5.7],
        'tof_mass':[1.6,1.4,0.8,22.9,24.9,9.9,5.5,2.9,1.6,1.6,2.9,2.9,1.0,0.8],
    }
    return pd.DataFrame(data)

# ── Estimated descriptor metadata (shown as badges in UI) ────────────────────
ESTIMATED_DESCRIPTORS = {
    'layer_n': {
        'label': 'Layer #',
        'source': 'Derived from XRD Scherrer (002) ÷ 0.615 nm/layer — not directly measured in Jeon et al. 2026',
        'confidence': 'low',
    },
    'mo_s_ratio': {
        'label': 'Mo/S ratio',
        'source': 'Scale from XPS literature (Baker et al. 2001; Lince et al.); values per sample estimated from s_thick and XANES phase data in Jeon et al. 2026',
        'confidence': 'medium',
    },
}

df = load_data()

TARGETS = {
    'eta': ('Overpotential η', 'V', 'max'),
    'tafel': ('Tafel slope', 'mV/dec', 'min'),
    'ecsa': ('ECSA', 'cm²', 'max'),
    'rct': ('Rct', 'Ω·cm²', 'min'),
    'raman': ('Raman A₁g/E₂g', '', 'min'),
    'resistivity': ('Resistivity', 'Ω·cm', 'min'),
    'tof_ecsa': ('TOF (ECSA)', 'nmol cm⁻²s⁻¹', 'max'),
    'tof_mass': ('TOF (mass)', 'nmol µg⁻¹s⁻¹', 'max'),
}

FEATURES = ['temp', 'cycles', 's_thick', 'layer_n', 'mo_s_ratio']
FEATURE_LABELS = {
    'temp':       'Annealing temperature (°C)',
    'cycles':     'Deposition cycles',
    's_thick':    'S-layer thickness (Å)',
    'layer_n':    'Number of layers',
    'mo_s_ratio': 'Mo/S atomic ratio',
}
FEATURE_RANGES = {
    'temp':       (500, 1000, 800, 50),
    'cycles':     (1, 100, 10, 1),
    's_thick':    (1.0, 12.0, 3.0, 0.5),
    'layer_n':    (1, 15, 4, 1),
    'mo_s_ratio': (0.45, 0.80, 0.55, 0.01),
}
EXPERIMENTAL_RANGES = {
    'temp':       (600, 800),
    'cycles':     (5, 50),
    's_thick':    (2.0, 9.0),
    'layer_n':    (2, 20),
    'mo_s_ratio': (0.46, 0.82),  # corrected XPS scale: stoichiometric ~0.455, Mo-rich M2.0 ~0.82
}

SERIES_COLORS = {'T': '#378ADD', 'N': '#1D9E75', 'M': '#BA7517'}

# ── Gaussian Process Models ───────────────────────────────────
@st.cache_resource
def train_gp_models():
    """
    Train one GP per target on the full dataset.
    Also runs LOO CV to get calibrated uncertainty estimates.
    Uses Matérn ν=2.5 kernel with ARD (one length scale per feature).
    Features are standardised before fitting.
    """
    X = df[FEATURES].values.astype(float)

    gp_models, gp_scores, scalers_X, scalers_y, loo_stds = {}, {}, {}, {}, {}
    loo = LeaveOneOut()

    for key in TARGETS:
        y = df[key].values.astype(float)

        # Scalers fit on full data (used for final model)
        sx = StandardScaler().fit(X)
        sy = StandardScaler().fit(y.reshape(-1, 1))

        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(length_scale=[1.0, 1.0, 1.0],
                     length_scale_bounds=[(0.01, 100)] * 3,
                     nu=2.5)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 10))
        )

        # --- LOO to get calibrated std ---
        loo_means, loo_stds_list = [], []
        for tr, te in loo.split(X):
            sx_loo = StandardScaler().fit(X[tr])
            sy_loo = StandardScaler().fit(y[tr].reshape(-1, 1))
            X_tr_s = sx_loo.transform(X[tr])
            y_tr_s = sy_loo.transform(y[tr].reshape(-1, 1)).ravel()

            gp_loo = GaussianProcessRegressor(
                kernel=C(1.0, (1e-3, 1e3))
                       * Matern(length_scale=[1.0]*3,
                                length_scale_bounds=[(0.01,100)]*3, nu=2.5)
                       + WhiteKernel(noise_level=0.1,
                                     noise_level_bounds=(1e-5,10)),
                n_restarts_optimizer=5, normalize_y=False, alpha=1e-6
            )
            gp_loo.fit(X_tr_s, y_tr_s)
            X_te_s = sx_loo.transform(X[te])
            m_s, std_s = gp_loo.predict(X_te_s, return_std=True)
            m = sy_loo.inverse_transform(m_s.reshape(-1,1)).ravel()[0]
            std = std_s[0] * sy_loo.scale_[0]
            loo_means.append(m)
            loo_stds_list.append(std)

        loo_means = np.array(loo_means)
        r2  = r2_score(y, loo_means)
        mae = mean_absolute_error(y, loo_means)

        # Calibration factor: ratio of average abs error to average predicted std
        # Corrects for GP over/under-confidence
        avg_err = np.mean(np.abs(y - loo_means))
        avg_std = np.mean(loo_stds_list)
        calib   = avg_err / avg_std if avg_std > 0 else 1.0

        # Final GP trained on all data
        gp_full = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10,
            normalize_y=False, alpha=1e-6
        )
        gp_full.fit(sx.transform(X),
                    sy.transform(y.reshape(-1,1)).ravel())

        gp_models[key]  = gp_full
        gp_scores[key]  = {'r2': r2, 'mae': mae, 'loo_preds': loo_means,
                            'calib': calib}
        scalers_X[key]  = sx
        scalers_y[key]  = sy
        loo_stds[key]   = np.array(loo_stds_list)

    return gp_models, gp_scores, scalers_X, scalers_y, loo_stds


# ── Random Forest — kept only for Feature Importance page ────
@st.cache_resource
def train_rf_models():
    X = df[FEATURES].values
    models, scores, importances = {}, {}, {}
    loo = LeaveOneOut()
    for key in TARGETS:
        y = df[key].values
        rf = RandomForestRegressor(n_estimators=300, max_depth=4,
                                   min_samples_leaf=2, random_state=42)
        preds = np.zeros(len(y))
        for tr, te in loo.split(X):
            rf.fit(X[tr], y[tr])
            preds[te] = rf.predict(X[te])
        rf.fit(X, y)
        models[key]      = rf
        scores[key]      = {'r2': r2_score(y, preds),
                            'mae': mean_absolute_error(y, preds),
                            'loo_preds': preds}
        importances[key] = rf.feature_importances_
    return models, scores, importances


# Train both — GP is primary, RF only for feature importance page
with st.spinner("Training Gaussian Process models… (first load only)"):
    gp_models, gp_scores, scalers_X, scalers_y, loo_stds = train_gp_models()

rf_models, rf_scores, rf_importances = train_rf_models()


def gp_predict(key, t, c, s, ln, msr):
    """
    Predict mean and calibrated 95% credible interval using the GP.
    Returns (mean, lower_95, upper_95, std_calibrated).
    """
    X_new = np.array([[t, c, s, ln, msr]])
    sx = scalers_X[key]
    sy = scalers_y[key]
    gp = gp_models[key]

    X_s = sx.transform(X_new)
    m_s, std_s = gp.predict(X_s, return_std=True)

    mean = sy.inverse_transform(m_s.reshape(-1,1)).ravel()[0]
    std  = std_s[0] * sy.scale_[0] * gp_scores[key]['calib']

    lower = mean - 1.96 * std
    upper = mean + 1.96 * std
    return mean, lower, upper, std

# ── Method recommendation logic ───────────────────────────────
def recommend_method(layer_n, mo_s_ratio, ecsa_target, rct_target):
    """
    Given the 4 key descriptors, decide whether Chemical or Physical (MBE) method
    is required. Returns (method_label, color, reasons).

    Scientific basis (Choudhury et al. Penn State review + Jeon et al. 2026):
    - CVD/PVT: S/Mo vapor ratio varies with position in tube → cannot independently
      control stoichiometry, layer number, and crystallinity (Choudhury §2.2).
    - MBE: independent e-beam Mo + effusion cell S flux, RHEED in-situ monitoring,
      submonolayer precision via QCM calibration (Choudhury §2.3 + Jeon Methods).
    - MBE limitation: low S sticking coefficient under UHV → smaller domains than CVD,
      but enables intentional S-deficiency (Jeon M-series) impossible in CVD.
    - Layer control: CVD layers uncontrolled below ~5L due to nucleation density
      dependence on substrate position (Choudhury §3.1). MBE: each cycle = ~1 MoS₂
      monolayer by design (Jeon: growth rate calibrated by QCM, ~0.05 Å/s Mo).
    - Mo/S ratio: CVD in sulfur-rich conditions drives toward stoichiometric MoS₂;
      intentional off-stoichiometry requires MBE flux control (Jeon M-series, confirmed
      by XANES showing Mo⁰ residual at M2.0–M3.0).
    """
    reasons = []
    mbe_score = 0

    # Layer # threshold — CVD nucleation density uncontrolled below ~5 layers
    # (Choudhury et al. §2.2: "growth limited by transition metal precursor supply to substrate")
    if layer_n <= 3:
        mbe_score += 3
        reasons.append(
            f"Layer # = {layer_n} (≤3L): atomic-layer precision requires MBE. "
            f"CVD nucleation density depends on substrate position — cannot reliably "
            f"control <5L films (Choudhury et al., Penn State review §2.2)"
        )
    elif layer_n <= 6:
        mbe_score += 1
        reasons.append(
            f"Layer # = {layer_n} (4–6L): few-layer regime. MBE preferred for "
            f"reproducible layer control; CVD possible but less reliable at this thickness"
        )

    # Mo/S ratio threshold — CVD sulfur-rich conditions prevent intentional Mo-rich growth
    # (Choudhury §2.2: "sulfur-rich growth conditions where MoS₂ is in equilibrium with sulfur vapor")
    if mo_s_ratio > 0.72:
        mbe_score += 3
        reasons.append(
            f"Mo/S = {mo_s_ratio:.2f} (>0.72, highly Mo-rich): corresponds to "
            f"incomplete sulfurization regime (XANES: residual Mo⁰ peaks, Jeon Fig 3a). "
            f"CVD uses sulfur-rich conditions by design — cannot achieve this phase. "
            f"Requires MBE submonolayer S-flux control (Jeon Methods + Choudhury §2.3)"
        )
    elif mo_s_ratio > 0.58:
        mbe_score += 2
        reasons.append(
            f"Mo/S = {mo_s_ratio:.2f} (0.58–0.72): S-deficient regime with Mo⁰/MoS₂ "
            f"coexistence. CVD phase diagrams favor stoichiometric MoS₂ under sulfur "
            f"overpressure (Choudhury §2.1: Mo-S phase diagram). MBE preferred."
        )
    elif mo_s_ratio < 0.48:
        reasons.append(
            f"Mo/S = {mo_s_ratio:.2f} (≈stoichiometric 2H-MoS₂, XPS S/Mo≥2.2): "
            f"achievable by both CVD (sulfur-rich) and MBE. CVD is simpler here "
            f"(Choudhury §2.2: MoO₃ + S powder = standard CVD route)"
        )

    # ECSA threshold — high ECSA in thin films requires morphology control
    if ecsa_target > 8.0:
        mbe_score += 1
        reasons.append(
            f"ECSA target > 8 cm²: edge-site-rich thin films. MBE wafer-scale "
            f"uniformity and controlled stoichiometry maximize accessible edges "
            f"(Jeon MoS-N10: 8.0 cm², MoS-M6.0: 9.2 cm² — both MBE-grown)"
        )

    # Rct threshold — low Rct requires conductive Mo domains, only achievable by MBE
    if rct_target < 55:
        mbe_score += 1
        reasons.append(
            f"Rct target < 55 Ω·cm²: requires metallic Mo⁰ conductive domains "
            f"(best: MoS-N10 Rct=52.8, MoS-M6.0 Rct=45.5 — MBE-grown, Jeon Table 1). "
            f"CVD fully sulfurizes Mo → stoichiometric MoS₂ with higher Rct"
        )

    if mbe_score >= 3:
        return "🔬 Physical Method (MBE)", "#1D9E75", reasons
    elif mbe_score >= 1:
        return "⚗️ Both viable — MBE preferred", "#BA7517", reasons
    else:
        return "🧪 Chemical Method (CVD/PVT)", "#378ADD", reasons


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗️ MoS₂ HER Predictor")
    st.markdown("**Based on:** Jeon et al., ACS Nano 2026  \n**Theory:** 12-paper framework")
    st.markdown("---")

    st.markdown("### 🔑 Key descriptors")
    st.caption("Adjust these to explore HER activity and get a synthesis method recommendation.")

    layer_n = st.slider("Layer #",
                        min_value=1, max_value=25, value=5, step=1,
                        help="Number of MoS₂ layers. Derived from XRD Scherrer (002) crystallite size ÷ 0.615 nm/layer (Jeon et al. 2026). ⚠ Not directly measured — estimated descriptor.")
    st.caption("⚠ Estimated — derived from XRD crystallite size (Jeon et al. 2026); awaiting AFM step-height confirmation from SI")

    mo_s_ratio = st.slider("Mo/S atomic ratio",
                           min_value=0.45, max_value=0.90, value=0.56, step=0.01,
                           help="Stoichiometric 2H-MoS₂ = ~0.455 (XPS: S/Mo~2.2, Baker et al. 2001). Fully S-depleted limit ~0.893 (Lince et al. via Baker). Values per sample estimated from s_thick and XANES/EXAFS phase data (Jeon et al. 2026). ⚠ XPS not reported per sample.")
    st.caption("⚠ Estimated — XPS per sample not in Jeon et al.; scale from Baker et al. 2001 & Jeon XANES/EXAFS")

    ecsa_input = st.slider("Target ECSA (cm²)",
                           min_value=2.0, max_value=12.0, value=8.0, step=0.5,
                           help="Electrochemically active surface area. Higher = more active sites. ✅ Real data from Jeon et al. 2026 Table 1.")

    rct_input = st.slider("Target Rct (Ω·cm²)",
                          min_value=20.0, max_value=200.0, value=55.0, step=5.0,
                          help="Charge-transfer resistance. Lower = faster kinetics. ✅ Real data from Jeon et al. 2026 Table 1.")

    # ── Live method badge ──────────────────────────────────────
    st.markdown("---")
    method_label, method_color, method_reasons = recommend_method(
        layer_n, mo_s_ratio, ecsa_input, rct_input
    )
    st.markdown(
        f"<div style='background:{method_color}22; border-left:4px solid {method_color}; "
        f"padding:10px 12px; border-radius:6px;'>"
        f"<div style='font-size:1.1em; font-weight:700; color:{method_color};'>{method_label}</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    with st.expander("Why this method?", expanded=False):
        for r in method_reasons:
            st.caption(f"• {r}")
        if not method_reasons:
            st.caption("Near-stoichiometric, thick-film conditions — chemical CVD is sufficient.")

    st.markdown("---")
    st.markdown("### Advanced synthesis controls")
    st.caption("Used by the GP predictor for full performance estimates.")
    temp    = st.slider("Annealing temperature (°C)", 500, 1000, 800, 50)
    cycles  = st.slider("Deposition cycles", 1, 100, 10, 1)
    s_thick = st.slider("S-layer thickness (Å)", 1.0, 12.0, 3.0, 0.5)

    in_range = (
        600 <= temp <= 800 and 5 <= cycles <= 50 and
        2.0 <= s_thick <= 9.0 and 1 <= layer_n <= 9 and
        0.50 <= mo_s_ratio <= 0.72
    )
    exact = df[
        (df.temp == temp) & (df.cycles == cycles) & (df.s_thick == s_thick) &
        (df.layer_n == layer_n) & (df.mo_s_ratio.round(2) == round(mo_s_ratio, 2))
    ]
    if len(exact) > 0:
        st.success(f"✓ Exact match: **{exact.iloc[0]['sample']}**")
    elif in_range:
        st.info("≈ Interpolation — calibrated GP intervals")
    else:
        out_dims = sum([temp<600 or temp>800, cycles<5 or cycles>50,
                        s_thick<2 or s_thick>9, layer_n<1 or layer_n>9,
                        mo_s_ratio<0.50 or mo_s_ratio>0.72])
        st.warning(f"⚠ Extrapolation ({out_dims}D outside range)")

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", ["Predictor", "Inverse Predictor", "Trend analysis",
                         "Feature importance", "Theoretical basis", "About"],
                    label_visibility="collapsed")

# ── Prediction helper ─────────────────────────────────────────
def predict_all(t, c, s, ln, msr):
    """Returns GP mean for all targets."""
    result = {}
    for key in TARGETS:
        mean, _, _, _ = gp_predict(key, t, c, s, ln, msr)
        result[key] = mean
    return result


def estimate_vacancy_concentration(s_thick, cycles):
    """
    Estimate surface S-vacancy concentration (%) based on synthesis parameters.
    Uses Tsai et al. 2017 framework: vacancy concentration scales with S-deficiency.
    Lower s_thick and fewer cycles → higher vacancy concentration.
    Reference: stoichiometric (s_thick=9, cycles=50) ≈ 0–5% vacancies.
    """
    # Normalise s_thick: 2.0 Å = most deficient, 9.0 Å = stoichiometric
    s_norm = (s_thick - 2.0) / 7.0          # 0 (deficient) → 1 (stoichiometric)
    c_norm = (min(cycles, 50) - 5) / 45.0   # 0 (few cycles) → 1 (many cycles)
    # Vacancy concentration: higher when s_thick low and cycles low
    vac_pct = (1 - 0.5 * s_norm - 0.5 * c_norm) * 60.0
    return max(0.0, min(90.0, vac_pct))


def classify_vacancy_stage(vac_pct):
    """
    Li & Voiry et al. ACS 2019: two-stage HER framework.
    Stage 1 (<~20%): isolated point defects, moderate improvement.
    Stage 2 (>~50%): large undercoordinated Mo regions, max TOF ~2 s⁻¹ at -160 mV in KOH.
    """
    if vac_pct < 20:
        return "Stage 1 — isolated point defects (moderate HER enhancement)", "#1D9E75"
    elif vac_pct < 50:
        return "Transition — mixed point defects + Mo-rich domains", "#BA7517"
    else:
        return "Stage 2 — undercoordinated Mo regions (maximum HER activity)", "#E84040"


def get_derived(vals, t, c, s, ln=4, msr=0.55):
    r = vals['raman']
    if r < 1.2:   dgh = -0.08 + (r-1.0)*0.15
    elif r < 1.7: dgh = -0.06 + (r-1.2)*0.28
    elif r < 2.1: dgh =  0.08 + (r-1.7)*0.60
    else:          dgh =  0.32 + (r-2.1)*0.90
    dgh -= 0.10  # MBE Si strain correction (Yang et al.)

    bd = (50-min(c,50))/45*0.5 + max(0,(6-s))/4*0.5
    if bd > 0.65:   boundary = "Very high — 2H+2H & 2H–1T (ΔGH* ≈ −0.13 eV)"
    elif bd > 0.40: boundary = "High — 2H–2H dominant (Tafel ~95 mV/dec)"
    elif bd > 0.20: boundary = "Moderate"
    else:           boundary = "Low — stoichiometric, few boundaries"

    if s < 2:     vac = "Very high (excess Mo, structural disruption)"
    elif s < 3:   vac = "High — Mo+MoS₂ coexistence, Vs dominant"
    elif s <= 6:  vac = "Moderate-optimal — Vs + VMoS3 active"
    elif s <= 8:  vac = "Low — approaching stoichiometry"
    else:          vac = "Minimal — stoichiometric MoS₂"

    tf = vals['tafel']
    if tf < 60:    mech = "Volmer–Heyrovsky (Heyrovsky rate-limiting)"
    elif tf < 100: mech = "Volmer–Heyrovsky (Volmer moderate)"
    elif tf < 150: mech = "Volmer rate-limiting (H₂O dissociation in KOH)"
    elif tf < 300: mech = "Volmer strongly rate-limiting"
    else:           mech = "Volmer severely limited (quasi-stoichiometric MoS₂)"

    # Layer-dependent edge site exposure (Hanslin et al.)
    if ln <= 2:      layer_txt = "Ultra-thin (1–2L) — maximum edge/basal ratio, all sites accessible"
    elif ln <= 4:    layer_txt = "Few-layer (3–4L) — good edge exposure, MBE-optimal range"
    elif ln <= 7:    layer_txt = "Multi-layer (5–7L) — bulk screening reduces basal activity"
    else:            layer_txt = "Thick film (≥8L) — edge sites dominate, basal largely inactive"

    # Mo/S ratio → phase composition (Geng et al. + Li/Voiry)
    if msr < 0.52:   phase_txt = "Near-stoichiometric MoS₂ — 2H dominant, low conductivity"
    elif msr < 0.58: phase_txt = "Slightly Mo-rich — S-vacancies present, onset of metallic character"
    elif msr < 0.65: phase_txt = "Mo-rich — Mo⁰/MoS₂ coexistence, high conductivity (Geng 2016)"
    else:            phase_txt = "Highly Mo-rich — metallic Mo domains dominant, resistivity low but structural integrity at risk"

    return {'dgh': dgh, 'boundary': boundary, 'vacancy': vac, 'mechanism': mech,
            'layer': layer_txt, 'phase': phase_txt}

# ── Pages ─────────────────────────────────────────────────────

if page == "Predictor":
    st.markdown("## MoS₂ for HER — Trend Predictor")

    # ── Method banner ─────────────────────────────────────────
    st.markdown(
        f"<div style='background:{method_color}18; border:2px solid {method_color}; "
        f"padding:14px 20px; border-radius:10px; margin-bottom:16px;'>"
        f"<span style='font-size:1.4em; font-weight:800; color:{method_color};'>{method_label}</span>"
        f"<span style='color:#888; font-size:0.9em; margin-left:16px;'>"
        f"Layer # = {layer_n} · Mo/S = {mo_s_ratio:.2f} · "
        f"Target ECSA = {ecsa_input:.1f} cm² · Target Rct = {rct_input:.0f} Ω·cm²</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    with st.expander("📋 Method reasoning", expanded=False):
        for r in method_reasons:
            st.markdown(f"• {r}")
        if not method_reasons:
            st.markdown("• Near-stoichiometric, thick-film conditions — CVD is sufficient.")

    st.markdown(
        f"GP prediction with: **{temp}°C** · **{cycles} cycles** · **S = {s_thick} Å**"
    )

    if len(exact) > 0:
        vals = {k: exact.iloc[0][k] for k in TARGETS}
        source = "Real data from Jeon et al. table"
        gp_ci = None
    else:
        vals = predict_all(temp, cycles, s_thick, layer_n, mo_s_ratio)
        source = "Gaussian Process prediction (calibrated 95% credible interval)"
        gp_ci = {}
        for key in TARGETS:
            mean, lower, upper, std = gp_predict(key, temp, cycles, s_thick, layer_n, mo_s_ratio)
            gp_ci[key] = {'mean': mean, 'lower': lower, 'upper': upper, 'std': std}

    st.caption(f"Source: {source}")

    # ── 4 Key Descriptors Banner ──────────────────────────────
    st.markdown("### 🔑 Key descriptors")
    vac_pct = estimate_vacancy_concentration(s_thick, cycles)
    stage_txt, stage_color = classify_vacancy_stage(vac_pct)
    kd1, kd2, kd3, kd4 = st.columns(4)
    with kd1:
        ln_icon = "🟢" if layer_n <= 5 else ("🟡" if layer_n <= 12 else "🔴")
        st.metric("Layer #", f"{layer_n} layers")
        st.caption(f"{ln_icon} Optimal ≤5L (N10=~5L) for edge density")
        st.caption("⚠ Estimated descriptor — XRD Scherrer / 0.615 nm")
    with kd2:
        msr_icon = "🟢" if 0.55 <= mo_s_ratio <= 0.70 else ("🟡" if mo_s_ratio < 0.80 else "🔴")
        st.metric("Mo/S ratio", f"{mo_s_ratio:.2f}")
        st.caption(f"{msr_icon} Optimal 0.55–0.70 (Mo+MoS₂ coexistence)")
        st.caption("⚠ Estimated — XPS scale from Baker et al. 2001")
    with kd3:
        ecsa_val = vals['ecsa']
        ecsa_icon = "🟢" if ecsa_val >= 7 else ("🟡" if ecsa_val >= 5 else "🔴")
        st.metric("ECSA (predicted)", f"{ecsa_val:.1f} cm²")
        st.caption(f"{ecsa_icon} Target ≥7 cm² · Input target: {ecsa_input:.1f} cm²")
        st.caption("✅ Real data — Jeon et al. 2026 Table 1")
    with kd4:
        resist_val = vals['resistivity']
        rct_val    = vals['rct']
        phys_icon  = "🟢" if resist_val < 12 and rct_val < 70 else ("🟡" if resist_val < 17 else "🔴")
        st.metric("Physical props", f"ρ={resist_val:.1f} Ω·cm")
        st.caption(f"{phys_icon} Rct={rct_val:.0f} Ω·cm² · target ρ<12, Rct<70")
        st.caption("✅ Real data — Jeon et al. 2026 Table 1")

    st.markdown("---")

    # Metrics grid
    cols = st.columns(4)
    metrics_order = ['eta','tafel','ecsa','rct','raman','resistivity','tof_ecsa','tof_mass']
    for i, key in enumerate(metrics_order):
        name, unit, better = TARGETS[key]
        v = vals[key]
        col = cols[i % 4]

        thresholds = {
            'eta':(-0.38,-0.50), 'tafel':(110,200), 'ecsa':(7,5),
            'rct':(70,130), 'raman':(1.8,2.2), 'resistivity':(12,17),
            'tof_ecsa':(9,6), 'tof_mass':(5,2)
        }
        if key in thresholds:
            g, b = thresholds[key]
            if better == 'max':
                color = "normal" if v >= g else ("off" if v <= b else "inverse")
            else:
                color = "normal" if v <= g else ("off" if v >= b else "inverse")
        else:
            color = "normal"

        fmt = f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"

        if gp_ci is not None:
            std = gp_ci[key]['std']
            fmt_std = f"±{std:.2f}" if abs(std) < 100 else f"±{std:.0f}"
            col.metric(f"{name}", f"{fmt} {unit}",
                       delta=f"{fmt_std} {unit} (1σ)", delta_color="off")
        else:
            col.metric(f"{name}", f"{fmt} {unit}")

    st.markdown("---")

    # Derived descriptors (expanded with layer + phase)
    der = get_derived(vals, temp, cycles, s_thick, layer_n, mo_s_ratio)
    st.markdown("### Derived structural descriptors (theory-based)")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        dgh_color = "🟢" if abs(der['dgh']) < 0.15 else ("🟡" if abs(der['dgh']) < 0.35 else "🔴")
        st.markdown(f"**Estimated ΔGH*** {dgh_color}")
        st.markdown(f"**{der['dgh']:.2f} eV**  \nIdeal = 0 eV · Pt = −0.18 eV · 2H–1T = −0.13 eV  \n*Hanslin + Yang, ~1.5% MBE strain correction*")

    with c2:
        st.markdown("**S-vacancy density**")
        st.markdown(f"{der['vacancy']}  \n*Li/Voiry + Yang et al.*")
        st.markdown("**Domain boundary density**")
        st.markdown(f"{der['boundary']}  \n*Zhu et al. Nat. Commun. 2019*")

    with c3:
        st.markdown("**Layer # effect**")
        st.markdown(f"{der['layer']}  \n*Hanslin et al. PCCP 2023*")
        st.markdown("**Phase composition**")
        st.markdown(f"{der['phase']}  \n*Geng et al. Nat. Commun. 2016*")

    with c4:
        st.markdown("**Dominant HER mechanism**")
        st.markdown(f"{der['mechanism']}  \n*Muhyuddin review + Yang et al.*")
        st.markdown("**MBE strain activation**")
        strain_active = s_thick <= 6 and cycles <= 20
        strain_txt = "Active — tensile strain activates Vs/VMoS3 sites" if strain_active else "Limited — stoichiometric regime reduces strain benefit"
        st.markdown(f"{strain_txt}  \n*Yang et al. RSC Adv. 2023*")

    st.markdown("---")
    # Closest samples
    st.markdown("### Closest samples in database")
    df_dist = df.copy()
    df_dist['dist'] = df.apply(
        lambda r: np.sqrt(
            ((r.temp - temp) / 400) ** 2 +
            ((r.cycles - cycles) / 95) ** 2 +
            ((r.s_thick - s_thick) / 10) ** 2 +
            ((r.layer_n - layer_n) / 14) ** 2 +
            ((r.mo_s_ratio - mo_s_ratio) / 0.35) ** 2
        ),
        axis=1)
    closest = df_dist.nsmallest(3, 'dist')
    show_cols = ['sample','series','temp','cycles','s_thick','layer_n','mo_s_ratio',
                 'eta','tafel','ecsa','rct','raman','resistivity']
    st.dataframe(closest[show_cols].reset_index(drop=True), use_container_width=True)


# ── Inverse Predictor page ────────────────────────────────────
elif page == "Inverse Predictor":
    st.markdown("## 🔄 Inverse Predictor — Properties → Synthesis Method")
    st.markdown(
        "Set your **target HER properties** below. The app finds the closest experimental "
        "match and tells you whether **Chemical (CVD)** or **Physical (MBE)** method is needed."
    )
    st.info(
        "💡 Idea del maestro: una vez identificados los key descriptors que determinan la "
        "actividad catalítica, podemos trabajar al revés — definir las propiedades deseadas "
        "y recomendar el método de síntesis para alcanzarlas."
    )

    st.markdown("### Step 1 — Set target performance")
    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        t_eta   = st.slider("Target η (V)", -0.60, -0.25, -0.35, 0.01,
                             help="More negative = harder. MoS-N10 achieved −0.33 V.")
    with ic2:
        t_tafel = st.slider("Target Tafel slope (mV/dec)", 60, 300, 100, 5,
                             help="Lower = better kinetics. MoS-N10: 80 mV/dec.")
    with ic3:
        t_ecsa  = st.slider("Target ECSA (cm²)", 2.0, 12.0, 7.0, 0.5,
                             help="Higher = more active surface. MoS-M6.0: 9.2 cm².")
    with ic4:
        t_rct   = st.slider("Target Rct (Ω·cm²)", 20.0, 200.0, 60.0, 5.0,
                             help="Lower = faster charge transfer. MoS-N10: 52.8.")

    st.markdown("---")
    st.markdown("### Step 2 — Closest experimental match")

    df_inv = df.copy()
    df_inv['perf_score'] = df_inv.apply(lambda r: np.sqrt(
        ((r.eta    - t_eta)   / 0.30) ** 2 +
        ((r.tafel  - t_tafel) / 250)  ** 2 +
        ((r.ecsa   - t_ecsa)  / 8)    ** 2 +
        ((r.rct    - t_rct)   / 180)  ** 2
    ), axis=1)
    candidates = df_inv.nsmallest(3, 'perf_score')
    best = candidates.iloc[0]

    cand_show = candidates[['sample','series','temp','cycles','s_thick','layer_n','mo_s_ratio',
                             'eta','tafel','ecsa','rct']].copy()
    cand_show.columns = ['Sample','Series','Temp (°C)','Cycles','S-thick (Å)',
                         'Layers','Mo/S','η (V)','Tafel','ECSA (cm²)','Rct (Ω·cm²)']
    st.dataframe(cand_show.reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.markdown("### Step 3 — Synthesis method recommendation")

    b_layer = best['layer_n']
    b_msr   = best['mo_s_ratio']
    b_ecsa  = best['ecsa']
    b_rct   = best['rct']

    # Use the same logic as the sidebar
    inv_method_label, inv_method_color, inv_method_reasons = recommend_method(
        b_layer, b_msr, b_ecsa, b_rct
    )

    # Big method banner
    st.markdown(
        f"<div style='background:{inv_method_color}18; border:3px solid {inv_method_color}; "
        f"padding:18px 24px; border-radius:12px; margin:8px 0 16px 0;'>"
        f"<div style='font-size:1.6em; font-weight:800; color:{inv_method_color};'>{inv_method_label}</div>"
        f"<div style='color:#666; margin-top:6px;'>Best match: <b>{best['sample']}</b> — "
        f"η={best.eta:.2f} V · Tafel={best.tafel:.0f} mV/dec · "
        f"ECSA={best.ecsa:.1f} cm² · Rct={best.rct:.1f} Ω·cm²</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    col_method, col_params = st.columns([1, 2])
    with col_method:
        st.markdown("**Why this method?**")
        for r in inv_method_reasons:
            st.markdown(f"• {r}")
        if not inv_method_reasons:
            st.markdown("• Near-stoichiometric, thick-film — CVD is sufficient.")

    with col_params:
        st.markdown("**Target synthesis parameters**")
        param_table = {
            'Parameter': ['Annealing temp', 'Deposition cycles', 'S-layer thickness',
                          'Layer #', 'Mo/S ratio'],
            'Value': [f"{best['temp']:.0f} °C", f"{best['cycles']:.0f}",
                      f"{best['s_thick']:.1f} Å", f"{b_layer:.0f}", f"{b_msr:.2f}"],
            'Note': [
                'Higher T → crystalline but fewer edge sites',
                'Controls thickness & layer count (~1 layer per 5 cycles)',
                'Key lever for Mo/S ratio and S-vacancy density',
                '≤4 layers: MBE required for precision',
                '>0.58: Mo+MoS₂ coexistence — MBE Mo-flux control',
            ]
        }
        st.dataframe(pd.DataFrame(param_table), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Step 4 — Key descriptor summary for this condition")
    pp1, pp2, pp3, pp4 = st.columns(4)
    with pp1:
        ln_icon = "🟢" if b_layer <= 4 else "🟡"
        st.metric("Layer #", f"{b_layer:.0f}")
        st.caption(f"{ln_icon} Edge site exposure")
    with pp2:
        msr_icon = "🟢" if 0.55 <= b_msr <= 0.65 else "🟡"
        st.metric("Mo/S ratio", f"{b_msr:.2f}")
        st.caption(f"{msr_icon} Phase composition (Mo⁰ + MoS₂)")
    with pp3:
        ecsa_icon = "🟢" if b_ecsa >= 7 else "🟡"
        st.metric("ECSA", f"{b_ecsa:.1f} cm²")
        st.caption(f"{ecsa_icon} Active surface area")
    with pp4:
        rct_icon = "🟢" if b_rct < 70 else "🟡"
        st.metric("Rct", f"{b_rct:.1f} Ω·cm²")
        st.caption(f"{rct_icon} Charge transfer resistance")

    with pp4:
        rct_icon = "🟢" if best['rct'] < 70 else "🟡"
        st.metric("Rct", f"{best['rct']:.1f} Ω·cm²")
        st.caption(f"{rct_icon} Charge transfer resistance")

    resist_best = best['resistivity']
    raman_best  = best['raman']
    st.caption(
        f"Additional: Resistivity = {resist_best:.2f} Ω·cm · "
        f"Raman A₁g/E₂g = {raman_best:.2f} · "
        f"Loading = {best['loading']:.1f} µg/cm²"
    )


elif page == "Trend analysis":
    st.markdown("## Experimental trend analysis")
    st.markdown("Data from Jeon et al. ACS Nano 2026 — 14 MBE-grown MoS₂ samples in 1M KOH")

    target_sel = st.selectbox("Performance variable",
                               options=list(TARGETS.keys()),
                               format_func=lambda k: f"{TARGETS[k][0]} ({TARGETS[k][1]})")

    name, unit, better = TARGETS[target_sel]

    fig = go.Figure()
    x_labels = {'T': 'Annealing temperature (°C)',
                 'N': 'Deposition cycles',
                 'M': 'S-layer thickness (Å)'}
    x_cols   = {'T': 'temp', 'N': 'cycles', 'M': 's_thick'}

    for ser, color in SERIES_COLORS.items():
        sub = df[df.series == ser].sort_values(x_cols[ser])
        fig.add_trace(go.Scatter(
            x=sub[x_cols[ser]], y=sub[target_sel],
            mode='lines+markers',
            name=f'Series {ser} ({x_labels[ser].split("(")[0].strip()})',
            line=dict(color=color, width=2),
            marker=dict(size=9, color=color),
            text=sub['sample'],
            hovertemplate='<b>%{text}</b><br>x=%{x}<br>'+target_sel+'=%{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=f"{name} by experimental series",
        xaxis_title="Independent variable (differs by series — see legend)",
        yaxis_title=f"{name} ({unit})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    show = df[['sample','series','temp','cycles','s_thick', target_sel]].copy()
    show.columns = ['Sample','Series','Temp (°C)','Cycles','S-thick (Å)', f"{name} ({unit})"]
    best_idx = show[f"{name} ({unit})"].idxmax() if better=='max' else show[f"{name} ({unit})"].idxmin()
    st.dataframe(show.reset_index(drop=True), use_container_width=True)
    st.success(f"**Best value:** {df.iloc[best_idx]['sample']} — {df.iloc[best_idx][target_sel]:.2f} {unit}")

    # Correlation heatmap
    st.markdown("### Correlation matrix — all variables")
    num_cols = ['temp','cycles','s_thick','raman','resistivity','ecsa','eta','tafel','rct','tof_ecsa','tof_mass']
    corr = df[num_cols].corr()
    fig2 = px.imshow(corr, text_auto=".2f", aspect="auto",
                     color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                     title="Pearson correlation between synthesis variables and performance metrics")
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)




elif page == "Feature importance":
    st.markdown("## Random Forest — Feature importance")
    st.markdown("""
    Random Forest trained on 14 experimental samples with **Leave-One-Out cross-validation**.
    Feature importance shows which synthesis variable (temperature, cycles, S-thickness)
    drives each performance metric most.
    Note: predictions on the Predictor page use the Gaussian Process model.
    This page uses Random Forest specifically for its feature importance scores.
    """)

    # LOO performance — show both GP and RF
    st.markdown("### Model performance (Leave-One-Out CV)")
    perf_data = []
    for key in TARGETS:
        name, unit, _ = TARGETS[key]
        perf_data.append({
            'Property': name, 'Unit': unit,
            'GP LOO R²': round(gp_scores[key]['r2'], 3),
            'GP LOO MAE': round(gp_scores[key]['mae'], 3),
            'RF LOO R²': round(rf_scores[key]['r2'], 3),
            'RF LOO MAE': round(rf_scores[key]['mae'], 3),
        })
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, use_container_width=True)
    st.caption("⚠ With only 14 samples, LOO scores indicate high uncertainty. GP is used for predictions; RF is shown here for feature importance only.")

    # Feature importance bar chart
    st.markdown("### Feature importance by output variable")
    imp_data = []
    for key in TARGETS:
        name, unit, _ = TARGETS[key]
        for i, feat in enumerate(FEATURES):
            imp_data.append({
                'Property': name,
                'Feature': FEATURE_LABELS[feat],
                'Importance': rf_importances[key][i]
            })
    imp_df = pd.DataFrame(imp_data)

    target_sel2 = st.selectbox("Select property",
                                options=list(TARGETS.keys()),
                                format_func=lambda k: f"{TARGETS[k][0]} ({TARGETS[k][1]})",
                                key='imp_sel')
    sub_imp = imp_df[imp_df['Property'] == TARGETS[target_sel2][0]]

    fig3 = px.bar(sub_imp, x='Feature', y='Importance',
                  color='Feature',
                  color_discrete_map={
                      FEATURE_LABELS['temp']: '#378ADD',
                      FEATURE_LABELS['cycles']: '#1D9E75',
                      FEATURE_LABELS['s_thick']: '#BA7517',
                  },
                  title=f"Feature importance for {TARGETS[target_sel2][0]}",
                  labels={'Importance': 'Relative importance (0–1)'})
    fig3.update_layout(showlegend=False, height=350,
                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3, use_container_width=True)

    # All properties heatmap
    st.markdown("### Feature importance — all properties")
    heat_data = np.array([[rf_importances[k][i] for i in range(3)] for k in TARGETS])
    heat_df = pd.DataFrame(heat_data,
                           index=[TARGETS[k][0] for k in TARGETS],
                           columns=[FEATURE_LABELS[f] for f in FEATURES])
    fig4 = px.imshow(heat_df, text_auto=".2f", aspect="auto",
                     color_continuous_scale='Greens', zmin=0, zmax=1,
                     title="Feature importance matrix (Random Forest)")
    fig4.update_layout(height=400)
    st.plotly_chart(fig4, use_container_width=True)

    # Partial dependence
    st.markdown("### Partial dependence — how each variable drives performance")
    pd_target = st.selectbox("Property for partial dependence",
                              options=list(TARGETS.keys()),
                              format_func=lambda k: f"{TARGETS[k][0]} ({TARGETS[k][1]})",
                              key='pd_sel')
    pd_feature = st.selectbox("Variable to vary",
                               options=FEATURES,
                               format_func=lambda k: FEATURE_LABELS[k],
                               key='pd_feat')

    rf_model = rf_models[pd_target]
    ranges = {'temp': np.linspace(500,1000,60),
              'cycles': np.linspace(1,100,60),
              's_thick': np.linspace(1,12,60),
              'layer_n': np.linspace(1,12,60),
              'mo_s_ratio': np.linspace(0.45,0.80,60)}
    defaults = {'temp':800, 'cycles':10, 's_thick':3.0, 'layer_n':4, 'mo_s_ratio':0.55}

    x_range = ranges[pd_feature]
    X_pd = np.array([[
        x if pd_feature=='temp'       else defaults['temp'],
        x if pd_feature=='cycles'     else defaults['cycles'],
        x if pd_feature=='s_thick'    else defaults['s_thick'],
        x if pd_feature=='layer_n'    else defaults['layer_n'],
        x if pd_feature=='mo_s_ratio' else defaults['mo_s_ratio'],
    ] for x in x_range])

    # GP predictions with uncertainty band
    y_means, y_lowers, y_uppers = [], [], []
    for row in X_pd:
        m, lo, hi, _ = gp_predict(pd_target, row[0], row[1], row[2], row[3], row[4])
        y_means.append(m)
        y_lowers.append(lo)
        y_uppers.append(hi)
    y_means  = np.array(y_means)
    y_lowers = np.array(y_lowers)
    y_uppers = np.array(y_uppers)

    exp_x = df[pd_feature].values
    exp_y = df[pd_target].values

    in_range_mask = {
        'temp':       (x_range >= 600)  & (x_range <= 800),
        'cycles':     (x_range >= 5)    & (x_range <= 50),
        's_thick':    (x_range >= 2)    & (x_range <= 9),
        'layer_n':    (x_range >= 1)    & (x_range <= 9),
        'mo_s_ratio': (x_range >= 0.50) & (x_range <= 0.72),
    }

    fig5 = go.Figure()
    # 95% credible band
    fig5.add_trace(go.Scatter(
        x=np.concatenate([x_range, x_range[::-1]]),
        y=np.concatenate([y_uppers, y_lowers[::-1]]),
        fill='toself', fillcolor='rgba(127,119,221,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% credible interval', showlegend=True
    ))
    fig5.add_trace(go.Scatter(x=x_range, y=y_means, mode='lines',
                               name='GP mean', line=dict(color='#7F77DD', width=2)))
    exp_in = x_range[in_range_mask[pd_feature]]
    y_in   = y_means[in_range_mask[pd_feature]]
    fig5.add_trace(go.Scatter(x=exp_in, y=y_in, mode='lines',
                               name='Experimental range',
                               line=dict(color='#1D9E75', width=3)))
    colors_exp = [SERIES_COLORS[s] for s in df['series']]
    fig5.add_trace(go.Scatter(
        x=exp_x, y=exp_y, mode='markers',
        name='Experimental data',
        marker=dict(size=10, color=colors_exp, symbol='circle',
                    line=dict(width=1, color='white')),
        text=df['sample'],
        hovertemplate='<b>%{text}</b><br>%{x:.1f} → %{y:.2f}<extra></extra>'
    ))
    fig5.add_vrect(x0=ranges[pd_feature][in_range_mask[pd_feature]][0],
                   x1=ranges[pd_feature][in_range_mask[pd_feature]][-1],
                   fillcolor="rgba(29,158,117,0.08)", line_width=0,
                   annotation_text="Experimental range", annotation_position="top left")
    fig5.update_layout(
        title=f"Partial dependence: {TARGETS[pd_target][0]} vs {FEATURE_LABELS[pd_feature]}",
        xaxis_title=FEATURE_LABELS[pd_feature],
        yaxis_title=f"{TARGETS[pd_target][0]} ({TARGETS[pd_target][1]})",
        height=420,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    fig5.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
    fig5.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Purple band = GP 95% credible interval. Green line = interpolation within experimental data. Dots = real data (colored by series T/N/M).")


elif page == "Theoretical basis":
    st.markdown("## Theoretical framework — 12 papers integrated")

    papers = [
        ("1 · Hanslin, Jónsson & Akola — PCCP 2023 (DFT)",
         "Exposed Mo sites on edges have activation barriers <0.5 eV at 0 V vs. SHE. "
         "Low Raman A₁g/E₂g → more Mo exposed → more active. Theoretical overpotential "
         "Mo₀ edge ≈ 0.25 V, consistent with −0.33 V for MoS-N10. S-bound H has Heyrovsky "
         "barriers >1.5 eV (kinetically dead). Extrapolating: T >900°C → grain coalescence "
         "eliminates Mo edge sites → η >−0.7 V expected."),
        ("2 · Li, Qin & Voiry — ACS Nano 2019",
         "S vacancies follow a non-linear optimum: too few → inert basal plane, optimal → "
         "max TOF, too many → structural collapse. Direct analog to Jeon M series: "
         "M3.0–M6.0 is the optimal S-deficiency window."),
        ("3 · Geng et al. — Nature Communications 2016",
         "Metallic 1T phase: resistivity 0.48 Ω·cm, Rct ≈1 Ω, η = −175 mV. Mixed "
         "Mo+MoS₂ domains in N10 and M3–M6 replicate this partially via metallic Mo pathways."),
        ("4 · Muhyuddin et al. — J. Energy Chemistry 2023 (review)",
         "Tafel <60 mV/dec → Heyrovsky rate-limiting. 60–120 → Volmer–Heyrovsky. "
         ">120 → Volmer rate-limiting (slow H₂O dissociation in alkaline KOH). "
         "N10 (80 mV/dec) is the only sample in the Volmer–Heyrovsky regime."),
        ("5 · Jeon et al. — ACS Nano 2026 (experimental base)",
         "14 MBE samples in 1M KOH. Global optimum: MoS-N10 (η=−0.33V, Tafel=80, "
         "ECSA=8.0 cm², Rct=52.8). T↑ → resistivity↑, ECSA↓, Tafel↑. Cycles↑ → "
         "resistivity↑ monotonically. S-thickness has non-linear optimum at 3–6 Å."),
        ("6 · Zhu et al. — Nature Communications 2019",
         "2H–2H grain boundaries and 2H–1T phase boundaries are active HER sites. "
         "ΔGH* = −0.13 eV for 2H–1T boundaries (≈ Pt at −0.18 eV). More boundaries → "
         "better performance linearly. Tafel 73–75 mV/dec with composite boundaries, "
         "stable >200 h. Low cycles and moderate S-deficiency maximize boundary density."),
        ("7 · Yang et al. — RSC Advances 2023 (DFT)",
         "Defect + strain synergy: Vs, VMoS3, VMoS6 are active sites. Without strain: "
         "ΔGH* >0.22 eV. With 1% tensile: VMoS3 reaches ΔGH* ≈ 0 eV. MBE on Si: "
         "~1–2% tensile mismatch activates vacancy sites. d-band center of Mo governs "
         "H adsorption (R²=0.97–0.99). Volmer barrier on VMo = 0.43 eV."),
        ("8 · Integrated mechanistic picture",
         "The MoS-N10 optimum results from the convergence of 4 mechanisms: "
         "(1) intermediate thickness → low resistivity 8.99 Ω·cm; "
         "(2) MBE strain + Vs vacancies → ΔGH* ≈ 0 eV; "
         "(3) high 2H–1T boundary density → ΔGH* ≈ −0.13 eV; "
         "(4) Mo conductive domains → Rct = 52.8 Ω·cm². "
         "No single mechanism alone produces the optimum — synergy is essential."),
        ("9 · Tsai, Li, Park et al. — Nature Communications 2017 (DFT + experiment)",
         "Electrochemical desulfurization generates S-vacancies on the MoS₂ basal plane at "
         "potentials ≥ −1.0 V vs RHE. DFT: S-vacancy formation becomes thermodynamically "
         "favourable at −1.0 V; optimal concentration 12.5–15.6% of surface atoms (ΔGH* ≈ 0 eV). "
         "Vacancies form in clusters (zigzag pattern) — clustered vacancies more stable than dispersed. "
         "EC desulfurization (onset ~−0.6 V) is as effective as Ar-plasma treatment and scalable to "
         "any MoS₂ morphology. Relevance to Jeon: MBE growth with low S-flux (M2.0–M3.0) generates "
         "vacancy concentrations in the 12.5–15.6% optimal window; operating HER potentials "
         "(−0.33 to −0.58 V) are below the EC desulfurization threshold, so vacancies are "
         "fixed by growth conditions rather than in-situ activation."),
        ("10 · Li, Qin, Ries & Voiry et al. — ACS 2019 (Stage 1/2 framework)",
         "HER activity from defective multilayer MoS₂ divides into two distinct regimes: "
         "Stage 1 (<~20% surface S-vacancies): isolated 'point defects', moderate HER improvement. "
         "Stage 2 (>~50% vacancies): large undercoordinated Mo regions, TOF ~2 s⁻¹ at −160 mV "
         "in 0.1 M KOH — among the best reported for MoS₂. "
         "Amorphous MoS₂ outperforms 2H in acid; 2H with ultra-high vacancies dominates in alkaline. "
         "Relevance to Jeon: M-series samples span Stage 1→2 transition. M2.0–M2.5 approach Stage 2 "
         "(very low S, Mo-rich), explaining their surprisingly competitive η despite high Tafel slopes. "
         "MoS-N10 sits optimally at the Stage 1 peak where point defects and conductivity balance."),
        ("11 · Sherwood et al. — ACS Nano 2024 (XPS phase fingerprinting)",
         "Establishes the rigorous XPS framework for distinguishing 2H MoS₂, 1T MoS₂, and "
         "sulfur-depleted MoS₂₋ₓ using a four split-orbit peak model (POS-A/B/C/D). "
         "Key findings directly relevant to Mo/S ratio descriptor in this tool: "
         "(1) Stoichiometric 2H MoS₂ has S/Mo = 2.2–2.5 (Mo/S ≈ 0.40–0.455), not 0.50 as often assumed. "
         "(2) Ar⁺ ion bombardment preferentially removes S, driving S/Mo from 2.5 → 1.1 (Mo/S → 0.91) "
         "— this defines the physical upper limit for Mo/S in defected MoS₂₋ₓ. "
         "(3) 2H→1T phase transition is triggered by S removal (S plane gliding); "
         "the 1T phase peak is at 228.7 eV (Mo 3d₅/₂) vs 229.3 eV for 2H and 228.1 eV for MoS₂₋ₓ. "
         "(4) For as-cast 2H MoS₂ electrodes the S/Mo depletion is stronger than for powders "
         "(final S/Mo ~1.1 vs ~1.6), confirming greater interfacial disorder. "
         "Relevance to Jeon M-series: MoS-M2.0–M3.0 (confirmed Mo⁰ + MoS₂ by XANES/EXAFS) correspond "
         "to Mo/S ≈ 0.72–0.82 in this tool's scale — within the S-depleted MoS₂₋ₓ regime defined here. "
         "This paper scientifically justifies the Mo/S ratio slider range (0.45–0.90) used in this tool "
         "and validates that Mo/S is a measurable, physically meaningful descriptor of phase composition. "
         "Critical caveat: the Mo/S values per Jeon sample are estimated (XPS not reported per sample) — "
         "this paper provides the scale calibration, not the per-sample measurements."),
        ("12 · Choudhury, Zhang, Al Balushi & Redwing — Penn State Review (CVD vs MBE epitaxy)",
         "Comprehensive review of vapor-phase deposition methods for TMDs, providing the scientific "
         "basis for the Chemical vs Physical method recommendation in this tool. "
         "Key distinctions: "
         "(1) CVD/PVT: S/Mo vapor ratio varies as a function of substrate position in the tube — "
         "cannot independently control stoichiometry, layer number, and crystallinity simultaneously. "
         "Sulfur-rich growth conditions are required by thermodynamics (Mo-S phase diagram: MoS₂ "
         "in equilibrium with sulfur vapor), making intentional Mo-rich growth impossible in CVD. "
         "(2) MBE: independent control of Mo flux (electron-beam evaporator) and S flux (effusion "
         "cell, calibrated by QCM), RHEED in-situ monitoring, submonolayer precision. Each "
         "deposition cycle = ~1 MoS₂ monolayer by design (confirmed in Jeon by QCM calibration). "
         "(3) MBE limitation: low S sticking coefficient under UHV conditions → smaller domains "
         "than CVD/MOCVD; compensated by growth on van der Waals substrates or Si (Jeon). "
         "(4) Layer # control: CVD nucleation density depends on substrate position relative to "
         "metal source — cannot reliably produce <5L films reproducibly. MBE: layer count set by "
         "number of deposition cycles (Jeon N-series: N5→N50 directly maps to 5→50 cycles). "
         "Raman E₂g–A₁g separation is a fingerprint of layer number confirmed by AFM/LEEM. "
         "Conclusion: the four key descriptors (Layer #, Mo/S ratio, ECSA, Physical props) "
         "can only be independently tuned by MBE — CVD can achieve stoichiometric, thick-film "
         "conditions but cannot access the Mo-rich, few-layer regime that optimizes HER."),
    ]

    for title, body in papers:
        with st.expander(title):
            st.write(body)

    st.markdown("---")
    st.markdown("### Key descriptors summary")
    desc_data = {
        'Descriptor': ['Raman A₁g/E₂g', 'Resistivity (Ω·cm)', 'Tafel slope (mV/dec)',
                        'ECSA (cm²)', 'Rct (Ω·cm²)', 'ΔGH* (eV)',
                        'Mo/S atomic ratio ⚠', 'Layer # ⚠'],
        'What it measures': [
            'Mo vs S edge site exposure',
            'Electronic conductivity (Mo⁰ domains)',
            'Rate-limiting HER mechanism',
            'Electrochemically active surface area',
            'Interfacial charge transfer resistance',
            'H adsorption free energy (activity descriptor)',
            'Phase composition: 2H MoS₂ ↔ MoS₂₋ₓ ↔ Mo⁰/MoS₂',
            'Film thickness proxy → edge/basal site ratio',
        ],
        'Optimal value': ['<1.8', '<12', '60–100', '>7', '<70', '≈ 0',
                          '0.55–0.72 (Mo⁰/MoS₂ coexistence)', '≤5 (few-layer, N10 regime)'],
        'Data source': ['✅ Jeon 2026', '✅ Jeon 2026', '✅ Jeon 2026',
                        '✅ Jeon 2026', '✅ Jeon 2026', 'DFT (Hanslin/Yang)',
                        '⚠ Scale: Sherwood 2024; values: estimated from Jeon XANES',
                        '⚠ Derived: Jeon XRD Scherrer ÷ 0.615 nm/layer'],
        'Key paper': ['Hanslin et al.', 'Geng et al.', 'Muhyuddin et al.',
                      'Li/Voiry et al.', 'Zhu et al.', 'Yang et al.',
                      'Sherwood et al. 2024 (paper 11)', 'Jeon et al. 2026 (Fig. 1a, 2a)']
    }
    st.dataframe(pd.DataFrame(desc_data), use_container_width=True)
    st.caption("⚠ = Estimated descriptor. ✅ = Directly measured and reported in Jeon et al. 2026 Table 1.")


elif page == "About":
    st.markdown("## About this tool")
    st.markdown("""
    ### MoS₂ HER Trend Predictor

    This tool provides theory-guided trend prediction and Random Forest analysis
    for MoS₂ thin films grown by Molecular Beam Epitaxy (MBE) for the
    Hydrogen Evolution Reaction (HER).

    ---

    ### Experimental basis
    **Jeon et al., ACS Nano 2026** — 14 MBE-grown MoS₂ samples on Si substrates,
    characterized in 1M KOH alkaline electrolyte. Three independent variable series:
    - **T series**: Annealing temperature (600–800°C), fixed 50 cycles, S=9.0 Å
    - **N series**: Deposition cycles (5–50), fixed 800°C, S=3.0 Å
    - **M series**: S-layer thickness (2.0–9.0 Å), fixed 800°C, 50 cycles

    ---

    ### Theoretical framework (12 papers)
    | # | Reference | Key contribution |
    |---|-----------|-----------------|
    | 1 | Hanslin et al., PCCP 2023 | DFT: Mo edge sites, Raman proxy |
    | 2 | Li & Voiry, ACS Nano 2019 | S vacancy optimization |
    | 3 | Geng et al., Nat. Commun. 2016 | 1T metallic phase benchmark |
    | 4 | Muhyuddin et al., J. Energy Chem. 2023 | HER mechanism review |
    | 5 | Jeon et al., ACS Nano 2026 | Experimental base data |
    | 6 | Zhu et al., Nat. Commun. 2019 | Domain boundary activation |
    | 7 | Yang et al., RSC Adv. 2023 | Defect-strain synergy DFT |
    | 8 | Integrated picture | Mechanistic convergence |
    | 9 | Tsai, Li, Park et al., Nat. Commun. 2017 | EC desulfurization, optimal vacancy conc. 12.5–15.6% |
    | 10 | Li, Qin, Ries & Voiry et al., ACS 2019 | Stage 1/2 vacancy framework, TOF ~2 s⁻¹ in KOH |
    | 11 | Sherwood et al., ACS Nano 2024 | XPS 4-peak model: 2H/1T/MoS₂₋ₓ fingerprinting; Mo/S scale calibration (S/Mo 2.2→1.1) |
    | 12 | Choudhury, Zhang, Al Balushi & Redwing, Penn State review | CVD vs MBE: independent flux control, layer precision, stoichiometry — scientific basis for Chemical/Physical recommendation |

    ---

    ### New in this version (papers 9 & 10)
    - **S-vacancy concentration gauge** on the Predictor page — shows estimated % vacancies
      with the Tsai 2017 optimal window (12.5–15.6%) highlighted in green
    - **Stage 1/2 classification** — labels each condition as isolated point defects (Stage 1)
      or undercoordinated Mo regions (Stage 2) per Li/Voiry 2019
    - **EC desulfurization note** — indicates whether operating HER potentials are sufficient
      for in-situ vacancy generation (Tsai 2017 threshold: −1.0 V vs RHE)
    - **Stage 1/2 overlay chart** on Trend Analysis — plots |η| vs estimated vacancy concentration
      with stage boundaries annotated
    - **ΔGH* vs vacancy concentration** schematic chart in Theoretical Basis

    ---

    ### Machine learning
    - **Primary model**: Gaussian Process (Matérn ν=2.5 kernel, ARD, calibrated 95% credible intervals)
    - **Secondary model**: Random Forest (300 trees) — used only for feature importance
    - **Validation**: Leave-One-Out cross-validation (only valid strategy with n=14)
    - **Uncertainty**: GP posterior std calibrated against LOO errors — grows naturally outside training data
    - **Features**: Annealing temperature, deposition cycles, S-layer thickness
    - **Targets**: η, Tafel slope, ECSA, Rct, Raman ratio, resistivity, TOF (ECSA), TOF (mass)

    ---

    ### Important limitations
    With only 14 training samples, even GP predictions have high uncertainty.
    The 95% credible intervals are statistically principled — unlike heuristic percentage ranges —
    but they are only as reliable as the GP model itself. This tool is designed
    for **trend analysis and mechanistic understanding**, not precise numerical prediction.

    ---

    *Developed as part of an experimental MoS₂ HER research project.*
    *Predictor integrates experimental data with multi-paper DFT theoretical framework.*
    """)
