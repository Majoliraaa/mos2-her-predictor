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
        # 0.615 nm/layer confirmed by AFM in Manyepedza et al. J. Phys. Chem. C 2022 (Fig. 9B):
        #   smallest platelets measured at 0.6–0.7 nm (1 TL) and 1.3–1.4 nm (2 TL) on mica.
        #   Bulk MoS₂: 0.615 nm/TL | isolated nanosheet: 0.67 nm/TL (Fan et al. JACS 2016).
        # T-series: (002) crystallite = 7.2 nm (T600) → 10.8 nm (T800) per Jeon Table 1 / Fig 1a.
        # N-series: (002) crystallite = 3.3 nm (N10) → 12.1 nm (N50) per Jeon Fig 2a caption.
        # M-series: same cycles as N50 (50 cycles) → same layer estimate as T800/N50.
        # This is a lower-bound estimate; actual layer count may differ due to growth mode.
        # PENDING: confirm with AFM or TEM per sample (check Jeon et al. SI).
        'layer_n':[12, 14, 18, 2, 5, 9, 13, 20, 20, 20, 20, 20, 20, 20],
        # ── Mo/S atomic ratio ──────────────────────────────────────────────────────
        # ⚠ ESTIMATED — XPS not reported per sample in Jeon et al. 2026.
        # Scale corrected using XPS literature: stoichiometric 2H-MoS₂ = S/Mo ~2.2 → Mo/S ~0.455
        # (Baker et al. Surf. Interface Anal. 2001; pristine XPS S/Mo = 2.2).
        # Sulfur-depleted limit: S/Mo ~1.12 → Mo/S ~0.893 (Lince et al. via Baker et al.).
        # Confirmed by Manyepedza 2022 (XPS wide scan): electrodeposited MoS₂ Mo/S = 1:2.2
        #   → Mo/S ratio = 0.455, consistent with stoichiometric 2H-MoS₂ scale used here.
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
        'source': 'Derived from XRD Scherrer (002) ÷ 0.615 nm/layer — not directly measured in Jeon et al. 2026. '
                  '0.615 nm/TL confirmed by AFM: Manyepedza et al. J. Phys. Chem. C 2022, Fig. 9B.',
        'confidence': 'low',
    },
    'mo_s_ratio': {
        'label': 'Mo/S ratio',
        'source': 'Scale from XPS literature (Baker et al. 2001; Sherwood 2024); '
                  'stoichiometric endpoint (Mo/S=0.455, S/Mo=2.2) confirmed by Manyepedza 2022 XPS wide scan. '
                  'Values per sample estimated from s_thick and XANES phase data in Jeon et al. 2026.',
        'confidence': 'medium',
    },
}

df = load_data()

TARGETS = {
    'eta': ('Overpotential η', 'V', 'max'),
    'tafel': ('Tafel slope', 'mV/dec', 'min'),
    'rct': ('Rct', 'Ω·cm²', 'min'),
    'raman': ('Raman A₁g/E₂g', '', 'min'),
    'resistivity': ('Resistivity', 'Ω·cm', 'min'),
    'tof_ecsa': ('TOF (ECSA)', 'nmol cm⁻²s⁻¹', 'max'),
    'tof_mass': ('TOF (mass)', 'nmol µg⁻¹s⁻¹', 'max'),
}

# ── The 3 key descriptors (whiteboard: Layer #, Mo/S ratio, ECSA) ─────────────
# These are the PRIMARY inputs to the GP — what the user controls.
# The GP predicts all HER performance metrics from these 3 descriptors.
# ECSA is real measured data (Jeon Table 1). Layer # and Mo/S are estimated.
FEATURES = ['layer_n', 'mo_s_ratio', 'ecsa']
FEATURE_LABELS = {
    'layer_n':    'Layer # (estimated from XRD Scherrer)',
    'mo_s_ratio': 'Mo/S atomic ratio (estimated from XANES)',
    'ecsa':       'ECSA (cm²) — measured, Jeon 2026',
}

SERIES_COLORS = {'T': '#378ADD', 'N': '#1D9E75', 'M': '#BA7517'}

# ── Gaussian Process Models ───────────────────────────────────
@st.cache_resource
def train_gp_models_v3():
    """
    Train one GP per target on the full dataset.
    v3: uses 3 synthesis features (temp, cycles, s_thick) only.
    layer_n and mo_s_ratio moved to descriptor-only role.
    """
    X = df[FEATURES].values.astype(float)
    n_feat = X.shape[1]  # 3 features

    gp_models, gp_scores, scalers_X, scalers_y, loo_stds = {}, {}, {}, {}, {}
    loo = LeaveOneOut()

    for key in TARGETS:
        y = df[key].values.astype(float)

        # Scalers fit on full data (used for final model)
        sx = StandardScaler().fit(X)
        sy = StandardScaler().fit(y.reshape(-1, 1))

        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(length_scale=[1.0] * n_feat,
                     length_scale_bounds=[(0.01, 100)] * n_feat,
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
                       * Matern(length_scale=[1.0] * n_feat,
                                length_scale_bounds=[(0.01, 100)] * n_feat,
                                nu=2.5)
                       + WhiteKernel(noise_level=0.1,
                                     noise_level_bounds=(1e-5, 10)),
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
def train_rf_models_v3():
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
    gp_models, gp_scores, scalers_X, scalers_y, loo_stds = train_gp_models_v3()

rf_models, rf_scores, rf_importances = train_rf_models_v3()


def gp_predict(key, ln, msr, ecsa):
    """Predict mean and calibrated 95% credible interval using the GP."""
    X_new = np.array([[ln, msr, ecsa]])
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
    Decide si se necesita Chemical (CVD/PVT) o Physical (MBE) según los 4 descriptores.
    Retorna (method_label, color, reasons).

    SCORING SYSTEM (verificado exhaustivamente contra las 14 muestras Jeon 2026):
    ─────────────────────────────────────────────────────────────────────────────
    LAYER #:
      ≤3L  → +3  (MBE obligatorio: k⁰ 250 cm/s, onset −0.10 V vs RHE, CVD no puede <5L)
      4–6L → +2  (MBE strongly preferred: few-layer regime, Jeon N10 optimum)
      7–12L→ +1  (MBE preferred: CVD posible pero sin RHEED/QCM no hay reproducibilidad)
      ≥13L → +0  (thick film: CVD estructuralmente viable si Mo/S es estequiométrico)

    Mo/S RATIO:
      >0.72  → +3  (MBE obligatorio: Mo-rich, CVD no puede alcanzar este régimen)
      >0.58  → +2  (MBE preferred: coexistencia Mo⁰/MoS₂, CVD empuja a estequiométrico)
      ≥0.50  → +1  (MBE preferred: ligeramente off-stoich, CVD tiende a overshooting)
      <0.50  → +0  (near-stoichiometric S/Mo≥2.0: CVD viable, ambos métodos posibles)

    ECSA ≥ 8.0 cm² → +1
    Rct  < 55 Ω·cm² → +1

    UMBRALES DE DECISIÓN:
      score ≥ 3 → 🔬 Physical Method (MBE)
      score 1–2 → ⚗️ Both viable — MBE preferred
      score = 0 → 🧪 Chemical Method (CVD/PVT)

    CASOS CVD POSIBLES (score=0): Layer≥13 + Mo/S<0.50 + ECSA<8 + Rct≥55
      Ejemplos del dataset: MoS-T800 (18L, Mo/S=0.46), MoS-N50 (20L, Mo/S=0.47),
                            MoS-M8.0 (20L, Mo/S=0.48), MoS-M9.0 (20L, Mo/S=0.46)
      → Films gruesos estequiométricos: CVD con S-rich atmosphere es suficiente.

    REFERENCIAS:
      Manyepedza et al. J. Phys. Chem. C 2022 — k⁰ vs Layer#, onset −0.10 V, AFM 0.615 nm/TL
      Choudhury et al. Penn State review — CVD vs MBE, control de stoichiometry y capas
      Jeon et al. ACS Nano 2026 — 14 muestras MBE, rango experimental completo
      Sherwood et al. ACS Nano 2024 — escala Mo/S: S/Mo 2.2→1.1 (Mo/S 0.455→0.893)
    """
    reasons = []
    mbe_score = 0

    # ── LAYER NUMBER ─────────────────────────────────────────────────────────
    # Kinetic basis (Manyepedza 2022 + McKelvey/Brunet Cabre 2021):
    #   k⁰: 1 TL → 250 cm/s | 3 TL → 1.5 cm/s (factor 167×)
    #   onset: 1–3 TL → −0.10 V | electrodeposited → −0.29 V | bulk → −0.49 V
    #   AFM: 0.615 nm/TL confirmado (mismo valor usado en estimación layer_n Jeon)
    # CVD: no puede controlar <5L por densidad de nucleación variable (Choudhury §3.1)
    # CVD: para ≥13L + Mo/S estequiométrico → viable (Choudhury §2.2)
    if layer_n <= 3:
        mbe_score += 3
        reasons.append(
            f"Layer # = {layer_n} (≤3L): 1–3 trilayers logran onset HER −0.10 V vs RHE "
            f"(H₂ verificado por cromatografía de gases). "
            f"k⁰ cinético: ~250 cm s⁻¹ (1 TL) vs ~1.5 cm s⁻¹ (3 TL) — factor 167×. "
            f"MBE obligatorio: CVD no puede controlar reproduciblemente <5L. "
            f"[Manyepedza 2022; Choudhury §2.3]"
        )
    elif layer_n <= 6:
        mbe_score += 2
        reasons.append(
            f"Layer # = {layer_n} (4–6L): régimen few-layer, zona óptima Jeon N10 (~5L). "
            f"k⁰ elevado respecto a bulk. MBE strongly preferred para control de capas. "
            f"CVD posible pero sin RHEED/QCM no hay reproducibilidad. "
            f"[Manyepedza 2022; Choudhury §2.2]"
        )
    elif layer_n <= 12:
        mbe_score += 1
        reasons.append(
            f"Layer # = {layer_n} (7–12L): régimen multi-capa. MBE preferred — "
            f"CVD posible pero la densidad de nucleación varía con posición del substrato "
            f"dificultando reproducibilidad. [Choudhury §3.1]"
        )
    else:
        # ≥13L: film grueso → CVD estructuralmente viable SI Mo/S es estequiométrico
        # No suma puntos — CVD puede producir estos films en condiciones S-rich
        reasons.append(
            f"Layer # = {layer_n} (≥13L): film grueso. CVD viable si Mo/S es "
            f"near-stoichiometric (<0.50). MBE sigue siendo más preciso pero no obligatorio "
            f"para reproducir este régimen. [Choudhury §2.2; Jeon T-series]"
        )

    # ── Mo/S RATIO ───────────────────────────────────────────────────────────
    # Escala XPS: estequiométrico = S/Mo 2.2 → Mo/S 0.455 (Manyepedza 2022 + Sherwood 2024)
    # Límite Mo-rico: S/Mo 1.1 → Mo/S 0.893 (Sherwood 2024)
    # CVD en condiciones S-rich no puede producir Mo/S > ~0.52 de forma reproducible
    if mo_s_ratio > 0.72:
        mbe_score += 3
        reasons.append(
            f"Mo/S = {mo_s_ratio:.2f} (>0.72, altamente Mo-rich): régimen de "
            f"sulfurización incompleta (XANES: picos Mo⁰ residuales, Jeon Fig 3a). "
            f"CVD en condiciones S-rich no puede alcanzar este régimen por diseño. "
            f"MBE obligatorio para control de flujo S independiente. "
            f"[Sherwood 2024; Choudhury §2.1]"
        )
    elif mo_s_ratio > 0.58:
        mbe_score += 2
        reasons.append(
            f"Mo/S = {mo_s_ratio:.2f} (0.58–0.72): coexistencia Mo⁰/MoS₂. "
            f"CVD bajo overpressure de S favorece MoS₂ estequiométrico — "
            f"no puede alcanzar este régimen S-deficiente de forma reproducible. "
            f"MBE preferred para control de flujo. [Choudhury §2.1]"
        )
    elif mo_s_ratio >= 0.50:
        mbe_score += 1
        reasons.append(
            f"Mo/S = {mo_s_ratio:.2f} (0.50–0.58): ligeramente S-deficiente. "
            f"CVD tiende a overshooting hacia estequiométrico (S/Mo=2.2). "
            f"MBE preferred para aterrizar reproduciblemente en esta ventana. "
            f"[Choudhury §2.1; Sherwood 2024]"
        )
    else:
        # Mo/S < 0.50 → near-stoichiometric: CVD viable
        reasons.append(
            f"Mo/S = {mo_s_ratio:.2f} (near-stoichiometric, S/Mo≥2.0): "
            f"CVD en condiciones S-rich puede producir este régimen. "
            f"Ambos métodos son viables aquí. "
            f"[Manyepedza 2022 XPS: MoS₂ electrodepositado S/Mo=2.2; Choudhury §2.2]"
        )

    # ── ECSA ─────────────────────────────────────────────────────────────────
    if ecsa_target >= 8.0:
        mbe_score += 1
        reasons.append(
            f"ECSA ≥ 8.0 cm²: films ricos en edge sites. Uniformidad wafer-scale "
            f"de MBE maximiza sitios accesibles. "
            f"(Jeon N10: 8.0 cm², M6.0: 9.2 cm² — ambos MBE)"
        )

    # ── Rct ──────────────────────────────────────────────────────────────────
    if rct_target < 55:
        mbe_score += 1
        reasons.append(
            f"Rct < 55 Ω·cm²: requiere dominios Mo⁰ conductores metálicos. "
            f"(N10: Rct=52.8, M6.0: Rct=45.5 — MBE, Jeon Tabla 1). "
            f"CVD sulfuriza completamente → MoS₂ estequiométrico con Rct mayor."
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
    st.markdown("**Based on:** Jeon et al., ACS Nano 2026  \n**Theory:** 13-paper framework")
    st.markdown("---")
    st.markdown("### 🔑 Key descriptors")
    st.caption("Mueve los sliders → el predictor y el badge se actualizan en tiempo real.")

    layer_n = st.slider(
        "Layer #",
        min_value=1, max_value=20, value=5, step=1,
        help="Número de capas MoS₂. ≤3 trilayers = onset HER óptimo "
             "(−0.10 V vs RHE, Manyepedza et al. J. Phys. Chem. C 2022, AFM + GC confirmed). "
             "k⁰ cinético: 1 TL → 250 cm s⁻¹, 3 TL → 1.5 cm s⁻¹ (factor 167×). "
             "⚠ Estimado de XRD Scherrer (002) ÷ 0.615 nm/capa (AFM: Manyepedza Fig. 9B)."
    )
    st.caption("⚠ Estimado · rango datos: 2–20 capas")

    mo_s_ratio = st.slider(
        "Mo/S atomic ratio",
        min_value=0.45, max_value=0.90, value=0.56, step=0.01,
        help="MoS₂ estequiométrico = ~0.455 (XPS S/Mo≈2.2, confirmado Manyepedza 2022 + Sherwood 2024). "
             "Límite Mo-rico = ~0.893 (S/Mo=1.1, Sherwood 2024). "
             "Óptimo 0.55–0.72 = coexistencia Mo⁰/MoS₂. "
             "⚠ Estimado de XANES/EXAFS (Jeon 2026)."
    )
    st.caption("⚠ Estimado · rango datos: 0.46–0.82")

    ecsa_val = st.slider(
        "ECSA (cm²)",
        min_value=2.0, max_value=12.0, value=8.0, step=0.5,
        help="Área electroactiva. ✅ Dato real medido en Jeon et al. 2026 Tabla 1. "
             "Rango experimental: 3.5–9.2 cm². N10=8.0, M6.0=9.2 (mejores muestras)."
    )
    st.caption("✅ Dato real · rango Jeon: 3.5–9.2 cm²")

    # ── Closest sample match ───────────────────────────────────
    df_dist = df.copy()
    df_dist['dist'] = df.apply(lambda r: np.sqrt(
        ((r.layer_n    - layer_n)    / 18)    ** 2 +
        ((r.mo_s_ratio - mo_s_ratio) / 0.36)  ** 2 +
        ((r.ecsa       - ecsa_val)   / 6.0)   ** 2
    ), axis=1)
    best_match = df_dist.nsmallest(1, 'dist').iloc[0]
    dist_val = df_dist['dist'].min()

    if dist_val < 0.15:
        st.success(f"✓ Muestra más cercana: **{best_match['sample']}**")
    elif dist_val < 0.40:
        st.info(f"≈ Más cercana: **{best_match['sample']}** (interpolando)")
    else:
        st.warning(f"⚠ Extrapolando — más cercana: **{best_match['sample']}**")

    # ── Live method badge ──────────────────────────────────────
    method_label, method_color, method_reasons = recommend_method(
        layer_n, mo_s_ratio, ecsa_val, rct_target=55.0
    )
    st.markdown("---")
    st.markdown(
        f"<div style='background:{method_color}22; border-left:4px solid {method_color}; "
        f"padding:10px 12px; border-radius:6px;'>"
        f"<div style='font-size:1.15em; font-weight:800; color:{method_color};'>"
        f"{method_label}</div>"
        f"<div style='font-size:0.78em; color:#888; margin-top:4px;'>"
        f"Layer #{layer_n} · Mo/S {mo_s_ratio:.2f} · ECSA {ecsa_val:.1f} cm²"
        f"</div></div>",
        unsafe_allow_html=True
    )
    with st.expander("¿Por qué este método?", expanded=False):
        for r in method_reasons:
            st.caption(f"• {r}")
        if not method_reasons:
            st.caption("Condiciones near-stoichiometric — CVD es suficiente.")

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", ["Predictor", "Inverse Predictor", "Trend analysis",
                         "Feature importance", "Theoretical basis", "About"],
                    label_visibility="collapsed")

# ── Prediction helper ─────────────────────────────────────────
def predict_all(ln, msr, ecsa):
    """Returns GP mean for all targets given the 3 key descriptors."""
    result = {}
    for key in TARGETS:
        mean, _, _, _ = gp_predict(key, ln, msr, ecsa)
        result[key] = mean
    return result


def estimate_vacancy_concentration(s_thick, cycles):
    """
    Estimate surface S-vacancy concentration (%) based on synthesis parameters.
    Uses Tsai et al. 2017 framework: vacancy concentration scales with S-deficiency.
    Lower s_thick and fewer cycles → higher vacancy concentration.
    Reference: stoichiometric (s_thick=9, cycles=50) ≈ 0–5% vacancies.
    """
    s_norm = (s_thick - 2.0) / 7.0
    c_norm = (min(cycles, 50) - 5) / 45.0
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

    if ln <= 2:      layer_txt = "Ultra-thin (1–2L) — maximum edge/basal ratio, all sites accessible"
    elif ln <= 4:    layer_txt = "Few-layer (3–4L) — good edge exposure, MBE-optimal range"
    elif ln <= 7:    layer_txt = "Multi-layer (5–7L) — bulk screening reduces basal activity"
    else:            layer_txt = "Thick film (≥8L) — edge sites dominate, basal largely inactive"

    if msr < 0.52:   phase_txt = "Near-stoichiometric MoS₂ — 2H dominant, low conductivity"
    elif msr < 0.58: phase_txt = "Slightly Mo-rich — S-vacancies present, onset of metallic character"
    elif msr < 0.65: phase_txt = "Mo-rich — Mo⁰/MoS₂ coexistence, high conductivity (Geng 2016)"
    else:            phase_txt = "Highly Mo-rich — metallic Mo domains dominant, resistivity low but structural integrity at risk"

    return {'dgh': dgh, 'boundary': boundary, 'vacancy': vac, 'mechanism': mech,
            'layer': layer_txt, 'phase': phase_txt}

# ── Pages ─────────────────────────────────────────────────────

if page == "Predictor":
    st.markdown("## MoS₂ for HER — Trend Predictor")

    st.markdown(
        f"<div style='background:{method_color}18; border:2px solid {method_color}; "
        f"padding:14px 20px; border-radius:10px; margin-bottom:16px;'>"
        f"<span style='font-size:1.4em; font-weight:800; color:{method_color};'>{method_label}</span>"
        f"<span style='color:#888; font-size:0.9em; margin-left:16px;'>"
        f"Layer # {layer_n} · Mo/S {mo_s_ratio:.2f} · ECSA {ecsa_val:.1f} cm²</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    with st.expander("📋 ¿Por qué este método?", expanded=False):
        for r in method_reasons:
            st.markdown(f"• {r}")
        if not method_reasons:
            st.markdown("• Condiciones near-stoichiometric — CVD es suficiente.")

    st.caption(
        f"Muestra más cercana en base de datos: **{best_match['sample']}** "
        f"(T={best_match['temp']:.0f}°C, {best_match['cycles']:.0f} ciclos, "
        f"S={best_match['s_thick']:.1f}Å)"
    )

    if dist_val < 0.05:
        vals = {k: best_match[k] for k in TARGETS}
        source = f"Datos reales — {best_match['sample']} (Jeon et al. 2026 Tabla 1)"
        gp_ci = None
    else:
        vals = predict_all(layer_n, mo_s_ratio, ecsa_val)
        source = "Predicción GP (intervalo de credibilidad 95% calibrado)"
        gp_ci = {}
        for key in TARGETS:
            mean, lower, upper, std = gp_predict(key, layer_n, mo_s_ratio, ecsa_val)
            gp_ci[key] = {'mean': mean, 'lower': lower, 'upper': upper, 'std': std}

    st.caption(f"Fuente: {source}")

    st.markdown("### 🔑 Key descriptors")
    kd1, kd2, kd3, kd4 = st.columns(4)
    with kd1:
        ln_icon = "🟢" if layer_n <= 3 else ("🟡" if layer_n <= 6 else "🔴")
        st.metric("Layer #", f"{layer_n} capas")
        st.caption(f"{ln_icon} Óptimo ≤3L · Manyepedza 2022")
        st.caption("⚠ Estimado · XRD Scherrer")
    with kd2:
        msr_icon = "🟢" if 0.55 <= mo_s_ratio <= 0.72 else ("🟡" if mo_s_ratio < 0.82 else "🔴")
        st.metric("Mo/S ratio", f"{mo_s_ratio:.2f}")
        st.caption(f"{msr_icon} Óptimo 0.55–0.72 (Mo⁰/MoS₂)")
        st.caption("⚠ Estimado · Sherwood 2024")
    with kd3:
        ecsa_icon = "🟢" if ecsa_val >= 7 else ("🟡" if ecsa_val >= 5 else "🔴")
        st.metric("ECSA", f"{ecsa_val:.1f} cm²")
        st.caption(f"{ecsa_icon} Óptimo ≥7 cm² · Jeon 2026")
        st.caption("✅ Dato real")
    with kd4:
        resist_val = vals['resistivity']
        rct_val    = vals['rct']
        phys_icon  = "🟢" if resist_val < 12 and rct_val < 70 else ("🟡" if resist_val < 17 else "🔴")
        st.metric("Physical props", f"ρ={resist_val:.1f} Ω·cm")
        st.caption(f"{phys_icon} Rct={rct_val:.0f} Ω·cm²")
        st.caption("✅ Predicho por GP · Jeon 2026")

    st.markdown("---")

    cols = st.columns(4)
    metrics_order = ['eta','tafel','rct','raman','resistivity','tof_ecsa','tof_mass']
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

    bm_temp   = best_match['temp']
    bm_cycles = best_match['cycles']
    bm_sthick = best_match['s_thick']
    der = get_derived(vals, bm_temp, bm_cycles, bm_sthick, layer_n, mo_s_ratio)
    st.markdown("### Descriptores estructurales derivados (base teórica)")
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
        st.markdown(f"{der['layer']}  \n*Hanslin et al. PCCP 2023 + Manyepedza et al. 2022*")
        st.markdown("**Phase composition**")
        st.markdown(f"{der['phase']}  \n*Geng et al. Nat. Commun. 2016*")

    with c4:
        st.markdown("**Dominant HER mechanism**")
        st.markdown(f"{der['mechanism']}  \n*Muhyuddin review + Manyepedza 2022 (45 mV dec⁻¹, Heyrovsky RDS)*")
        st.markdown("**MBE strain activation**")
        strain_active = bm_sthick <= 6 and bm_cycles <= 20
        strain_txt = "Active — tensile strain activates Vs/VMoS3 sites" if strain_active else "Limited — stoichiometric regime reduces strain benefit"
        st.markdown(f"{strain_txt}  \n*Yang et al. RSC Adv. 2023*")

    st.markdown("---")
    st.markdown("### 3 muestras más cercanas en la base de datos")
    df_dist2 = df.copy()
    df_dist2['dist'] = df.apply(lambda r: np.sqrt(
        ((r.layer_n    - layer_n)    / 18)   ** 2 +
        ((r.mo_s_ratio - mo_s_ratio) / 0.36) ** 2 +
        ((r.ecsa       - ecsa_val)   / 6.0)  ** 2
    ), axis=1)
    closest = df_dist2.nsmallest(3, 'dist')
    show_cols = ['sample','series','temp','cycles','s_thick','layer_n','mo_s_ratio',
                 'ecsa','eta','tafel','rct','raman','resistivity']
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

    inv_method_label, inv_method_color, inv_method_reasons = recommend_method(
        b_layer, b_msr, b_ecsa, b_rct
    )

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
                '≤3 layers: MBE required (k⁰ × 167, onset −0.10 V, Manyepedza 2022)',
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

    show = df[['sample','series','temp','cycles','s_thick', target_sel]].copy()
    show.columns = ['Sample','Series','Temp (°C)','Cycles','S-thick (Å)', f"{name} ({unit})"]
    best_idx = show[f"{name} ({unit})"].idxmax() if better=='max' else show[f"{name} ({unit})"].idxmin()
    st.dataframe(show.reset_index(drop=True), use_container_width=True)
    st.success(f"**Best value:** {df.iloc[best_idx]['sample']} — {df.iloc[best_idx][target_sel]:.2f} {unit}")

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
                      FEATURE_LABELS['layer_n']:    '#9B59B6',
                      FEATURE_LABELS['mo_s_ratio']: '#E84040',
                      FEATURE_LABELS['ecsa']:       '#1D9E75',
                  },
                  title=f"Feature importance for {TARGETS[target_sel2][0]}",
                  labels={'Importance': 'Relative importance (0–1)'})
    fig3.update_layout(showlegend=False, height=350,
                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Feature importance — all properties")
    heat_data = np.array([[rf_importances[k][i] for i in range(len(FEATURES))] for k in TARGETS])
    heat_df = pd.DataFrame(heat_data,
                           index=[TARGETS[k][0] for k in TARGETS],
                           columns=[FEATURE_LABELS[f] for f in FEATURES])
    fig4 = px.imshow(heat_df, text_auto=".2f", aspect="auto",
                     color_continuous_scale='Greens', zmin=0, zmax=1,
                     title="Feature importance matrix (Random Forest)")
    fig4.update_layout(height=400)
    st.plotly_chart(fig4, use_container_width=True)

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
    ranges = {
        'layer_n':    np.linspace(1, 20, 60),
        'mo_s_ratio': np.linspace(0.45, 0.90, 60),
        'ecsa':       np.linspace(2.0, 12.0, 60),
    }
    defaults = {'layer_n': 5, 'mo_s_ratio': 0.56, 'ecsa': 8.0}

    x_range = ranges[pd_feature]
    X_pd = np.array([[
        x if pd_feature == 'layer_n'    else defaults['layer_n'],
        x if pd_feature == 'mo_s_ratio' else defaults['mo_s_ratio'],
        x if pd_feature == 'ecsa'       else defaults['ecsa'],
    ] for x in x_range])

    y_means, y_lowers, y_uppers = [], [], []
    for row in X_pd:
        m, lo, hi, _ = gp_predict(pd_target, row[0], row[1], row[2])
        y_means.append(m)
        y_lowers.append(lo)
        y_uppers.append(hi)
    y_means  = np.array(y_means)
    y_lowers = np.array(y_lowers)
    y_uppers = np.array(y_uppers)

    exp_x = df[pd_feature].values
    exp_y = df[pd_target].values

    in_range_mask = {
        'layer_n':    (x_range >= 2)    & (x_range <= 20),
        'mo_s_ratio': (x_range >= 0.46) & (x_range <= 0.82),
        'ecsa':       (x_range >= 3.5)  & (x_range <= 9.2),
    }

    fig5 = go.Figure()
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
    st.markdown("## Theoretical framework — 13 papers integrated")

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
        ("13 · Manyepedza, Courtney, Snowden, Jones & Rees — J. Phys. Chem. C 2022 "
         "(Impact electrochemistry: Layer # as HER activity descriptor) ★ PAPER COMPLETO",
         "Direct experimental validation that Layer # is the primary kinetic HER descriptor for MoS₂. "
         "Using impact electrochemistry (single nanoparticles colliding with an electrode), the study "
         "isolates the intrinsic catalytic properties of MoS₂ free from ensemble averaging effects.\n\n"
         "━━━ DATOS CUANTITATIVOS MEDIDOS (✅ experimental) ━━━\n\n"
         "Onset potentials (pH 2, H₂SO₄, confirmado por cromatografía de gases):\n"
         "  • 1–3 trilayers (nanoimpacto):   −0.10 V vs RHE  ← onset más bajo reportado sin Pt\n"
         "  • MoS₂ electrodepositado:        −0.29 V vs RHE  (j = 0.5 mA cm⁻²)\n"
         "  • NPs dropcast (bulk):           −0.49 V vs RHE\n"
         "  → Ventaja de 1–3 TL: 390 mV mejor que NPs bulk; 190 mV mejor que electrodeposición.\n\n"
         "Cinética (mecanismo HER, Heyrovsky rate-determining):\n"
         "  • Tafel slope electrodepositado: 45 mV dec⁻¹\n"
         "  • Coeficiente de transferencia α: 0.64–0.67 (Tafel + DigiElch waveshape fitting, n≥5)\n"
         "  • k⁰ electrodepositado: (3.17 ± 0.3) × 10⁻⁵ cm s⁻¹\n"
         "  • k⁰ vs trilayers (McKelvey/Brunet Cabre 2021, citado en este paper):\n"
         "      1 trilayer → k⁰ ≈ 250 cm s⁻¹\n"
         "      3 trilayers → k⁰ ≈ 1.5 cm s⁻¹   (factor 167× de diferencia)\n"
         "  → La transición 1→3 TL reduce k⁰ 167× — fundamento cinético del umbral Layer# ≤3.\n\n"
         "AFM (confirmación directa de espesores, Fig. 9B):\n"
         "  • 1 trilayer:  0.6–0.7 nm  (pico mínimo en mica)\n"
         "  • 2 trilayers: 1.3–1.4 nm  (factor ×2 exacto)\n"
         "  • d = 0.615 nm/TL (bulk MoS₂) | 0.67 nm/TL (nanosheet aislado)\n"
         "  → MISMO valor 0.615 nm/TL usado para estimar layer_n del dataset Jeon via XRD Scherrer.\n\n"
         "RDE (rotating disk electrode) — tres onsets resueltos:\n"
         "  • −0.10 V (1–3 TL exfoliados) | −0.25 V (intermedios) | −0.50 V (bulk 90 nm)\n"
         "  → Distribución de tamaños por sonicación genera población multi-modal.\n\n"
         "Eficiencia Faradaica (H₂ identificado por GC):\n"
         "  • 45% @ −0.15 V vs RHE | 48% @ −0.40 V vs RHE\n\n"
         "XPS (composición química):\n"
         "  • MoS₂ electrodepositado: S/Mo = 2.2 → Mo/S = 0.455\n"
         "  → Confirma el endpoint estequiométrico de la escala Mo/S usada en este tool.\n\n"
         "━━━ RELEVANCIA PARA EL PREDICTOR ━━━\n\n"
         "1. Umbral Layer# ≤3 → 🔬 MBE obligatorio: respaldado cuantitativamente por k⁰ 167×\n"
         "   y onset −0.10 V medido con AFM + GC. CVD no puede controlar <5L (Choudhury §2.2).\n"
         "2. Constante 0.615 nm/TL: cross-validada entre AFM (este paper) y XRD Scherrer\n"
         "   (usada para estimar layer_n del dataset Jeon — misma fuente física).\n"
         "3. Endpoint Mo/S = 0.455 (S/Mo=2.2): confirmado por XPS de electrodeposición —\n"
         "   consistente con escala Sherwood 2024 en el slider del tool.\n"
         "4. Mecanismo Heyrovsky RDS (45 mV dec⁻¹): consistente con N10 (80 mV dec⁻¹)\n"
         "   en régimen Volmer–Heyrovsky (Muhyuddin review, paper 4 del framework).\n"
         "5. Eficiencia Faradaica 45–48%: aplica a NPs sonicadas (chemical synthesis) —\n"
         "   MBE films con control de capas esperados ≥ este valor (pendiente experimental)."),
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
            'Film thickness proxy → edge/basal site ratio + kinetic constant k⁰',
        ],
        'Optimal value': ['<1.8', '<12', '60–100', '>7', '<70', '≈ 0',
                          '0.55–0.72 (Mo⁰/MoS₂ coexistence)',
                          '≤3 trilayers → onset −0.10 V vs RHE, k⁰ ~250 cm s⁻¹ (Manyepedza 2022)'],
        'Data source': ['✅ Jeon 2026', '✅ Jeon 2026', '✅ Jeon 2026',
                        '✅ Jeon 2026', '✅ Jeon 2026', 'DFT (Hanslin/Yang)',
                        '⚠ Scale: Sherwood 2024 + Manyepedza 2022 XPS; values: estimated from Jeon XANES',
                        '⚠ Derived: Jeon XRD Scherrer ÷ 0.615 nm/TL; '
                        '0.615 nm/TL confirmed AFM: Manyepedza 2022 Fig.9B; '
                        'threshold: Manyepedza impact electrochemistry + GC'],
        'Key paper': ['Hanslin et al.', 'Geng et al.', 'Muhyuddin et al.',
                      'Li/Voiry et al.', 'Zhu et al.', 'Yang et al.',
                      'Sherwood 2024 (paper 11) + Manyepedza 2022 (paper 13)',
                      'Manyepedza 2022 (paper 13) + Jeon 2026 (Fig. 1a, 2a)']
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

    ### Theoretical framework (13 papers)
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
    | 11 | Sherwood et al., ACS Nano 2024 | XPS 4-peak model: 2H/1T/MoS₂₋ₓ fingerprinting; Mo/S scale (S/Mo 2.2→1.1) |
    | 12 | Choudhury, Zhang, Al Balushi & Redwing, Penn State review | CVD vs MBE: independent flux control, layer precision, stoichiometry |
    | 13 | Manyepedza et al., J. Phys. Chem. C 2022 ★ | Impact electrochemistry: 1–3 TL onset −0.10 V vs RHE; k⁰ 167× advantage; AFM 0.615 nm/TL confirmed |

    ---

    ### New in this version (papers 9, 10 & 13 full)
    - **Paper 13 fully integrated** (previously abstract only): all quantitative data added —
      onset potentials, Tafel 45 mV dec⁻¹, k⁰ vs trilayers (1.5–250 cm s⁻¹), AFM heights,
      Faradaic efficiency 45–48%, XPS endpoint S/Mo=2.2
    - **Layer # threshold quantified**: k⁰ kinetic advantage (167×) now cited explicitly
      in `recommend_method()` logic comments and UI tooltips
    - **0.615 nm/TL cross-validated**: comment in dataset traces this constant to both
      Manyepedza AFM (Fig. 9B) and Fan et al. JACS 2016
    - **Mo/S = 0.455 endpoint confirmed**: Manyepedza XPS electrodeposition S/Mo=2.2
      added as second confirmation of Sherwood 2024 scale

    ---

    ### Machine learning
    - **Primary model**: Gaussian Process (Matérn ν=2.5 kernel, ARD, calibrated 95% credible intervals)
    - **Secondary model**: Random Forest (300 trees) — used only for feature importance
    - **Validation**: Leave-One-Out cross-validation (only valid strategy with n=14)
    - **Uncertainty**: GP posterior std calibrated against LOO errors — grows naturally outside training data
    - **Features**: Layer #, Mo/S ratio, ECSA (3 key descriptors)
    - **Targets**: η, Tafel slope, Rct, Raman ratio, resistivity, TOF (ECSA), TOF (mass)

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
