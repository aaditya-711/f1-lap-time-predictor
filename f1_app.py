import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Apex | F1 Lap Time Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "f1_laptime_model.pkl"
ENCODERS_PATH = BASE_DIR / "label_encoders.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.pkl"
DATA_PATH = BASE_DIR / "sample_data.csv"

# -------------------------------------------------------------------
# STYLING
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --f1-red: #e10600;
        --f1-bg: #0b0d10;
        --f1-panel: #15191f;
        --f1-muted: #9aa3ad;
        --f1-line: #2a3039;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 75% 0%, rgba(225, 6, 0, 0.10), transparent 35%),
            #0b0d10;
        color: #f4f5f7;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(11,13,16,0.92); }
    [data-testid="stSidebar"] {
        background: #111419;
        border-right: 1px solid #272c34;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1450px; }
    h1, h2, h3, .brand-title {
        font-family: 'Barlow Condensed', 'Inter', sans-serif !important;
        letter-spacing: 0.02em;
    }
    .brand-row {
        display: flex; align-items: center; gap: 12px;
        margin-bottom: 0.2rem;
    }
    .brand-mark {
        display: flex; align-items: center; justify-content: center;
        width: 48px; height: 42px; border-radius: 8px;
        background: var(--f1-red); color: white;
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 22px; font-weight: 800; font-style: italic;
        box-shadow: 0 8px 24px rgba(225,6,0,.22);
    }
    .eyebrow {
        color: #ff625c; font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.18em; text-transform: uppercase;
    }
    .hero-title {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: clamp(2.5rem, 5vw, 4.3rem);
        font-weight: 800; line-height: .98; margin: 0.35rem 0 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle { color: #a7afb9; font-size: 0.98rem; max-width: 760px; }
    .section-label {
        color: #a7afb9; font-size: 0.72rem; font-weight: 700;
        letter-spacing: .16em; text-transform: uppercase; margin-bottom: .35rem;
    }
    .panel {
        background: linear-gradient(145deg, #191e25, #12161b);
        border: 1px solid #2a3039; border-radius: 16px;
        padding: 1.15rem 1.25rem; height: 100%;
    }
    .prediction-panel {
        background: linear-gradient(125deg, rgba(225,6,0,.20), rgba(24,28,34,.98) 52%);
        border: 1px solid rgba(225,6,0,.55); border-radius: 18px;
        padding: 1.4rem 1.5rem; position: relative; overflow: hidden;
    }
    .prediction-panel:after {
        content: "F1"; position: absolute; right: 16px; bottom: -42px;
        font-family: 'Barlow Condensed', sans-serif; font-size: 10rem;
        font-weight: 800; color: rgba(255,255,255,.035); font-style: italic;
        pointer-events: none;
    }
    .prediction-value {
        font-family: 'Barlow Condensed', sans-serif; font-size: clamp(2.8rem, 5vw, 4.2rem);
        line-height: 1; font-weight: 800; letter-spacing: -.02em; margin: .35rem 0;
    }
    .prediction-unit { color: #b4bdc8; font-size: .9rem; }
    .metric-card {
        background: #15191f; border: 1px solid #2a3039; border-radius: 14px;
        padding: 1rem 1.1rem; min-height: 112px;
    }
    .metric-label {
        color: #a0a9b4; font-size: .72rem; text-transform: uppercase;
        font-weight: 700; letter-spacing: .12em;
    }
    .metric-value {
        font-family: 'Barlow Condensed', sans-serif; font-size: 2rem;
        font-weight: 700; margin-top: .25rem; line-height: 1.1;
    }
    .muted { color: #9aa3ad; font-size: .86rem; }
    .pill {
        display: inline-block; border: 1px solid #3b424d; border-radius: 999px;
        color: #dce1e7; padding: .28rem .65rem; font-size: .72rem;
        font-weight: 700; letter-spacing: .07em; text-transform: uppercase;
    }
    .red-rule { height: 4px; width: 72px; background: #e10600; border-radius: 3px; margin: .8rem 0 1.2rem; }
    div[data-testid="stButton"] > button {
        background: #e10600; color: white; border: 1px solid #e10600;
        border-radius: 9px; font-weight: 700; min-height: 3rem;
        transition: all .15s ease;
    }
    div[data-testid="stButton"] > button:hover {
        background: #b80500; border-color: #ff3b35; color: white;
    }
    div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label {
        color: #dce1e7; font-weight: 600;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #2a3039; border-radius: 12px; background: #12161b;
    }
    hr { border-color: #2a3039; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    required = [MODEL_PATH, ENCODERS_PATH, FEATURES_PATH]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required model artifact(s): " + ", ".join(missing)
        )

    # Only load pickle files from a trusted source. Pickle can execute code.
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    with ENCODERS_PATH.open("rb") as file:
        encoders = pickle.load(file)
    with FEATURES_PATH.open("rb") as file:
        features = pickle.load(file)
    return model, encoders, features


try:
    model, encoders, feature_columns = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the model files: {exc}")
    st.info(
        "Keep f1_laptime_model.pkl, label_encoders.pkl, and "
        "feature_columns.pkl in the same folder as f1_app.py."
    )
    st.stop()

# -------------------------------------------------------------------
# SIDEBAR — INPUTS
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-mark">APEX</div>
            <div>
                <div style="font-weight:800;font-size:1.05rem;">RACE LAB</div>
                <div class="eyebrow">Lap-time simulator</div>
            </div>
        </div>
        <hr>
        <div class="section-label">01 / Race conditions</div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Driver & tyres", expanded=True):
        driver = st.selectbox("Driver", list(encoders["Driver"].classes_))
        compound = st.selectbox("Tyre compound", list(encoders["Compound"].classes_))
        tyre_life = st.slider("Tyre life (laps)", 1, 40, 10)
        fresh_tyres = 1 if tyre_life <= 3 else 0

    with st.expander("Circuit & weather", expanded=True):
        weather = st.selectbox("Weather", list(encoders["Weather"].classes_))
        track_status = st.selectbox("Track status", list(encoders["TrackStatus"].classes_))
        track_temp = st.slider("Track temperature (°C)", 25, 55, 40)
        air_temp = st.slider("Air temperature (°C)", 18, 35, 25)

    with st.expander("Race strategy", expanded=True):
        lap_number = st.slider("Lap number", 1, 57, 20)
        fuel_load = st.slider("Fuel load (model units)", 50, 110, 80)
        position = st.slider("Current position", 1, 20, 5)
        drs_available = st.selectbox("DRS available", [0, 1], format_func=lambda x: "Available" if x else "Unavailable")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Predictions depend on the training dataset and its feature ranges.")
    predict_clicked = st.button("⚑  Predict lap time", use_container_width=True)

# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="brand-row">
        <div class="brand-mark">F1</div>
        <div class="eyebrow">APEX PERFORMANCE INTELLIGENCE · MACHINE LEARNING</div>
    </div>
    <div class="hero-title">LAP TIME<br> PREDICTOR<span style="color:#e10600">.</span></div>
    <div class="hero-subtitle">
        Explore how race conditions, tyre life and strategy inputs influence a
        model-estimated lap time. Adjust the setup in the left panel to run a prediction.
    </div>
    <div class="red-rule"></div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# INPUT ENCODING AND PREDICTION
# -------------------------------------------------------------------
input_data = {
    "LapNumber": lap_number,
    "TyreLife": tyre_life,
    "TrackTemp": track_temp,
    "AirTemp": air_temp,
    "FuelLoad": fuel_load,
    "Position": position,
    "DRS_Available": drs_available,
    "FreshTyres": fresh_tyres,
    "Compound_encoded": int(encoders["Compound"].transform([compound])[0]),
    "Weather_encoded": int(encoders["Weather"].transform([weather])[0]),
    "TrackStatus_encoded": int(encoders["TrackStatus"].transform([track_status])[0]),
    "Driver_encoded": int(encoders["Driver"].transform([driver])[0]),
}

try:
    input_df = pd.DataFrame([input_data]).reindex(columns=feature_columns)
    if input_df.isna().any().any():
        missing_features = input_df.columns[input_df.isna().any()].tolist()
        raise ValueError(f"Input values missing for model features: {missing_features}")
    prediction = float(model.predict(input_df)[0])
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

# Keep this explicit: the existing app computes these metrics over the full CSV.
# They are reference/in-sample metrics, not held-out test-set metrics.
reference_metrics = None
if DATA_PATH.exists():
    try:
        df = pd.read_csv(DATA_PATH)
        categorical_cols = ["Driver", "Compound", "Weather", "TrackStatus"]
        df_encoded = df.copy()
        for col in categorical_cols:
            df_encoded[f"{col}_encoded"] = encoders[col].transform(df_encoded[col].astype(str))
        X = df_encoded.reindex(columns=feature_columns)
        y = pd.to_numeric(df_encoded["LapTime"], errors="coerce")
        valid = X.notna().all(axis=1) & y.notna()
        if valid.any():
            y_pred = model.predict(X.loc[valid])
            reference_metrics = {
                "r2": r2_score(y.loc[valid], y_pred),
                "mae": mean_absolute_error(y.loc[valid], y_pred),
                "n": int(valid.sum()),
            }
    except Exception as exc:
        st.warning(f"Dataset reference metrics could not be calculated: {exc}")

# -------------------------------------------------------------------
# DASHBOARD
# -------------------------------------------------------------------
left, right = st.columns([1.35, 1], gap="large")
with left:
    st.markdown(
        f"""
        <div class="prediction-panel">
            <div class="section-label">Predicted lap time</div>
            <div class="prediction-value">{prediction:.3f}<span style="font-size:1.35rem;font-weight:600"> s</span></div>
            <div class="prediction-unit">Model estimate · not a live telemetry reading</div>
            <div style="margin-top:1rem;display:flex;gap:8px;flex-wrap:wrap">
                <span class="pill">{compound} tyre</span>
                <span class="pill">Lap {lap_number}</span>
                <span class="pill">P{position}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Current input snapshot</div>', unsafe_allow_html=True)
    st.markdown(f"**Driver:** {driver}  ")
    st.markdown(f"**Conditions:** {weather} · {track_status}  ")
    st.markdown(f"**Track / air:** {track_temp}°C / {air_temp}°C  ")
    st.markdown(f"**Tyre life / fuel:** {tyre_life} laps / {fuel_load} model units  ")
    st.markdown(f"**DRS:** {'Available' if drs_available else 'Unavailable'}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")
st.markdown('<div class="section-label">Model performance overview</div>', unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
if reference_metrics:
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">R² · full dataset</div>'
            f'<div class="metric-value">{reference_metrics["r2"]:.3f}</div>'
            f'<div class="muted">Reference fit, not held-out testing</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">MAE · full dataset</div>'
            f'<div class="metric-value">{reference_metrics["mae"]:.3f} s</div>'
            f'<div class="muted">Mean absolute error</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Rows evaluated</div>'
            f'<div class="metric-value">{reference_metrics["n"]:,}</div>'
            f'<div class="muted">Rows with valid features and target</div></div>',
            unsafe_allow_html=True,
        )
else:
    st.info("Dataset reference metrics are unavailable. Prediction is still available.")

st.markdown("")
with st.expander("Feature importance", expanded=False):
    importances = getattr(model, "feature_importances_", None)
    if importances is not None and len(importances) == len(feature_columns):
        importance_df = pd.DataFrame(
            {"Feature": feature_columns, "Importance": importances}
        ).sort_values("Importance", ascending=True)
        st.bar_chart(importance_df.set_index("Feature"), horizontal=True)
        st.caption(
            "Tree-based feature importance describes how the fitted model used "
            "features; it does not prove causation or guarantee real-world relevance."
        )
    else:
        st.info("This model does not expose compatible feature-importance values.")

with st.expander("About this model & limitations"):
    st.markdown(
        """
        - **Model:** saved XGBoost regressor loaded from the existing `.pkl` artifacts.
        - **Inputs:** the same 12 model features used by the current training script.
        - **Metrics:** the dashboard's reference metrics are calculated on the full CSV.
          They are not an independent test-set score. Use the training script's held-out
          test metrics when reporting generalization.
        - **Interpretation:** results are estimates from the available dataset, not
          official F1 timing, live telemetry, or guaranteed race strategy advice.
        - **Artifact safety:** only load pickle files from a source you trust.
        """
    )

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;color:#7f8995;font-size:.76rem">
        <span>APEX RACE LAB · F1 LAP TIME PREDICTOR</span>
        <span>Built with Streamlit · XGBoost</span>
    </div>
    """,
    unsafe_allow_html=True,
)
