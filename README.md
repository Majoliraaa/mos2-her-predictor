# MoS₂ HER Trend Predictor

Interactive web app for trend analysis and prediction of MoS₂ thin film electrocatalysts for the Hydrogen Evolution Reaction (HER).

## Features

- **Trend predictor** — input synthesis parameters (temperature, cycles, S-thickness) and get predicted electrochemical performance
- **Trend analysis** — visualize experimental trends from Jeon et al. (ACS Nano 2026) with correlation matrix
- **Feature importance** — Random Forest analysis showing which synthesis variable drives each property
- **Partial dependence plots** — how each variable influences performance with interpolation/extrapolation zones
- **Theoretical basis** — 8-paper integrated framework from DFT and experimental literature

## Data

Based on 14 MBE-grown MoS₂ samples from:
> Jeon et al., *ACS Nano* 2026, 20, 4479–4493

## Theoretical framework

1. Hanslin, Jónsson & Akola — *PCCP* 2023 (DFT edge sites)
2. Li, Qin & Voiry — *ACS Nano* 2019 (S vacancy optimization)
3. Geng et al. — *Nature Communications* 2016 (1T phase benchmark)
4. Muhyuddin et al. — *J. Energy Chemistry* 2023 (HER mechanism review)
5. Jeon et al. — *ACS Nano* 2026 (experimental base)
6. Zhu et al. — *Nature Communications* 2019 (domain boundaries)
7. Yang et al. — *RSC Advances* 2023 (defect-strain synergy)
8. Integrated mechanistic picture

## Deployment (Streamlit Cloud)

1. Fork or upload this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository and `app.py` as the main file
5. Click **Deploy** — your app will be live at a public URL

## Local installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

With n=14 training samples, Random Forest predictions have high uncertainty outside the experimental range. This tool is designed for **trend analysis and mechanistic understanding**, not precise numerical prediction.
