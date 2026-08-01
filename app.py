from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="CarValue Egypt",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# Paths and artifact loading
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "used_cars" / "hatla2ee_cars_august_2025.csv"

MODEL_PATH = MODEL_DIR / "best_xgboost.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"


@st.cache_resource
def load_artifacts():
    required = [MODEL_PATH, PREPROCESSOR_PATH, FEATURE_COLUMNS_PATH]
    missing = [path.name for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing))

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    return model, preprocessor, feature_columns


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)

    for column in ["company", "model", "color", "location"]:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    return df


try:
    model, preprocessor, feature_columns = load_artifacts()
except Exception as error:
    st.error("The car-price model files could not be loaded.")
    st.code(str(error))
    st.stop()

raw_df = load_dataset()


# =========================================================
# Helpers
# =========================================================

def clean_options(series, fallback):
    if series is None:
        return fallback

    values = pd.Series(series).dropna().astype(str).str.strip()
    values = values[values.ne("")]

    result = sorted(values.unique().tolist())
    return result or fallback


def models_for_company(company_name):
    if raw_df is None or not {"company", "model"}.issubset(raw_df.columns):
        return ["Elantra", "Corolla", "Sunny", "Cerato", "Tiguan"]

    subset = raw_df.loc[raw_df["company"] == company_name, "model"]
    return clean_options(subset, ["Other"])


def predict_price(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns)

    transformed = preprocessor.transform(input_df)
    predicted_log_price = model.predict(transformed)[0]
    predicted_price = float(np.expm1(predicted_log_price))

    if not np.isfinite(predicted_price) or predicted_price <= 0:
        raise ValueError("The model returned an invalid prediction.")

    return predicted_price


def gauge_geometry(estimate, mae):
    """Work out the dial ceiling and needle angle for the result gauge."""
    ceiling_candidates = [3_000_000, (estimate + mae) * 1.15]
    gauge_max = max(ceiling_candidates)
    gauge_max = np.ceil(gauge_max / 250_000) * 250_000

    fraction = min(max(estimate / gauge_max, 0.0), 1.0)
    angle_deg = -90 + 180 * fraction

    return gauge_max, angle_deg


# =========================================================
# Styling — dashboard / instrument-cluster identity
# =========================================================
#
# Token system:
#   Ink        #161b22   (bezel, primary text)
#   Paper      #f4efe4   (main surface — warm, not stark white)
#   Amber      #f0a63a   (gauge sweep, primary accent, ignition glow)
#   Teal       #1f6f78   (secondary accent, low-range gauge band)
#   Redline    #d1483f   (high-range gauge band, used sparingly)
#   Steel      #8f99a6   (muted labels, hairlines)
#
# Type:
#   Space Grotesk  — display (hero, price readout)
#   IBM Plex Sans  — body / UI labels
#   IBM Plex Mono  — data, odometer-style numerals, spec tags

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
    :root {
        --ink: #161b22;
        --ink-soft: #2a3341;
        --paper: #f4efe4;
        --paper-panel: #ffffff;
        --amber: #f0a63a;
        --teal: #1f6f78;
        --redline: #d1483f;
        --steel: #8f99a6;
        --hairline: #e2dccb;
    }

    .stApp { background: var(--paper); color: var(--ink); }
    .block-container { max-width: 1400px; padding-top: 0.6rem; padding-bottom: 3rem; }

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    /* ---------- Bezel / header ---------- */

    .bezel {
        background: var(--ink);
        border-radius: 20px;
        padding: 1.7rem 2rem 1.4rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 18px 40px rgba(22, 27, 34, 0.35);
    }

    .plate {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.22em;
        color: var(--amber);
        border: 1px solid rgba(240, 166, 58, 0.45);
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        margin-bottom: 0.7rem;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        letter-spacing: -0.02em;
        color: #f6f3ea;
        margin: 0 0 0.3rem;
    }

    .hero-sub {
        color: #a9b3c1;
        font-size: 0.98rem;
        max-width: 640px;
        margin-bottom: 1.3rem;
    }

    .dial-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem; }

    .dial-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(240, 166, 58, 0.18);
        border-radius: 14px;
        padding: 0.8rem 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }

    .ring {
        width: 42px; height: 42px; border-radius: 50%;
        background: conic-gradient(var(--amber) calc(var(--pct) * 1%), rgba(255,255,255,0.12) 0);
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }

    .ring::after {
        content: ""; width: 30px; height: 30px; border-radius: 50%;
        background: var(--ink);
    }

    .dial-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        color: var(--steel);
        text-transform: uppercase;
    }

    .dial-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #f6f3ea;
        margin-top: 0.1rem;
    }

    /* ---------- Section labels (spec-plate style) ---------- */

    .spec-tag {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        color: var(--teal);
        text-transform: uppercase;
        border-bottom: 2px solid var(--teal);
        padding-bottom: 0.15rem;
        margin-bottom: 0.9rem;
    }

    .panel {
        background: var(--paper-panel);
        border: 1px solid var(--hairline);
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 8px 22px rgba(22, 27, 34, 0.05);
        height: 100%;
    }

    /* ---------- Streamlit control overrides ---------- */

    [data-testid="stWidgetLabel"] p {
        color: var(--ink-soft) !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
    }

    div[data-testid="stForm"] {
        background: var(--paper-panel);
        border: 1px solid var(--hairline);
        border-radius: 18px;
        padding: 1.3rem 1.4rem 0.6rem;
    }

    div[data-testid="stFormSubmitButton"] button {
        width: 100% !important;
        min-height: 3.2rem !important;
        border-radius: 999px !important;
        border: 1px solid var(--amber) !important;
        background: var(--amber) !important;
        color: var(--ink) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        box-shadow: 0 0 0 6px rgba(240, 166, 58, 0.14) !important;
        transition: box-shadow 0.15s ease !important;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        box-shadow: 0 0 0 9px rgba(240, 166, 58, 0.22) !important;
    }

    /* ---------- Result: gauge cluster ---------- */

    .result-shell {
        background: var(--ink);
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 18px 40px rgba(22, 27, 34, 0.3);
    }

    .gauge-wrap {
        width: 260px; height: 140px;
        position: relative;
        margin: 0.4rem auto 0.2rem;
        overflow: hidden;
    }

    .gauge-dial {
        width: 260px; height: 260px;
        border-radius: 50%;
        position: absolute; top: 0; left: 0;
        background: conic-gradient(
            from 180deg,
            var(--teal) 0deg 60deg,
            var(--amber) 60deg 130deg,
            var(--redline) 130deg 180deg,
            transparent 180deg 360deg
        );
    }

    .gauge-hole {
        width: 190px; height: 190px;
        border-radius: 50%;
        background: var(--ink);
        position: absolute; top: 35px; left: 35px;
    }

    .gauge-needle {
        position: absolute;
        bottom: 0; left: 50%;
        width: 3px; height: 118px;
        background: #f6f3ea;
        border-radius: 3px 3px 0 0;
        transform-origin: bottom center;
        margin-left: -1.5px;
    }

    .gauge-hub {
        width: 14px; height: 14px; border-radius: 50%;
        background: var(--amber);
        position: absolute; bottom: -7px; left: 50%; margin-left: -7px;
        box-shadow: 0 0 0 4px rgba(240, 166, 58, 0.2);
    }

    .odometer {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.1rem;
        font-weight: 600;
        color: #f6f3ea;
        text-align: center;
        letter-spacing: 0.02em;
        margin-top: 0.4rem;
    }

    .odometer-unit {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: var(--amber);
        text-align: center;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .range-readout {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        color: #a9b3c1;
        text-align: center;
        margin-top: 0.5rem;
    }

    /* ---------- Vehicle registration card ---------- */

    .reg-card {
        background: var(--paper-panel);
        border: 1px solid var(--hairline);
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
    }

    .reg-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.6rem;
        margin-top: 0.9rem;
    }

    .reg-item {
        background: var(--paper);
        border: 1px solid var(--hairline);
        border-radius: 12px;
        padding: 0.7rem 0.8rem;
    }

    .reg-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--steel);
    }

    .reg-value {
        font-weight: 700;
        color: var(--ink);
        font-size: 0.98rem;
        margin-top: 0.15rem;
    }

    /* ---------- Idle state ---------- */

    .idle-shell {
        background: var(--paper-panel);
        border: 1px dashed #cbbf9e;
        border-radius: 18px;
        padding: 2.2rem 1.6rem;
        text-align: center;
        color: var(--steel);
    }

    .idle-shell .spec-tag { color: var(--steel); border-bottom-color: var(--steel); }

    .idle-lamp {
        width: 10px; height: 10px; border-radius: 50%;
        background: var(--steel);
        display: inline-block;
        margin-right: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Header / bezel
# =========================================================

st.markdown(
    """
    <div class="bezel">
        <div class="plate">HATLA2EE DATASET · AUG 2025</div>
        <div class="hero-title">🚘 CarValue Egypt</div>
        <div class="hero-sub">
            A gauge cluster for used-car pricing — enter a vehicle's specs and
            read its estimated market value straight off the dial.
        </div>
        <div class="dial-row">
            <div class="dial-card">
                <div class="ring" style="--pct: 83;"></div>
                <div>
                    <div class="dial-label">Fit (R²)</div>
                    <div class="dial-value">0.833</div>
                </div>
            </div>
            <div class="dial-card">
                <div class="ring" style="--pct: 45;"></div>
                <div>
                    <div class="dial-label">Avg. Error</div>
                    <div class="dial-value">≈ 119K EGP</div>
                </div>
            </div>
            <div class="dial-card">
                <div class="ring" style="--pct: 100;"></div>
                <div>
                    <div class="dial-label">Model</div>
                    <div class="dial-value">Tuned XGBoost</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Main workspace
# =========================================================

companies = clean_options(
    raw_df["company"] if raw_df is not None and "company" in raw_df.columns else None,
    ["Hyundai", "Kia", "Toyota", "Nissan", "BMW", "Mercedes"],
)

colors = clean_options(
    raw_df["color"] if raw_df is not None and "color" in raw_df.columns else None,
    ["Black", "White", "Silver", "Gray", "Red", "Blue"],
)

locations = clean_options(
    raw_df["location"] if raw_df is not None and "location" in raw_df.columns else None,
    ["Cairo", "Alexandria", "Giza", "6 October"],
)

selection_col, options_col = st.columns([1.15, 1], gap="large")

with selection_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<span class="spec-tag">Vehicle ID</span>', unsafe_allow_html=True)

    company = st.selectbox("Company", companies, key="company_selector")
    car_model = st.selectbox("Model", models_for_company(company), key=f"model_selector_{company}")
    year = st.number_input("Manufacturing Year", min_value=1972, max_value=2026, value=2020, step=1)

    st.markdown("</div>", unsafe_allow_html=True)

with options_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<span class="spec-tag">Market Conditions</span>', unsafe_allow_html=True)

    mileage = st.number_input("Mileage (km)", min_value=0, max_value=1_000_000, value=80_000, step=5_000)
    color = st.selectbox("Color", colors)
    location = st.selectbox("Location", locations)

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

with st.form("car_features_form"):
    st.markdown('<span class="spec-tag">Transmission &amp; Equipment</span>', unsafe_allow_html=True)

    transmission_col, features_col = st.columns([1, 1.5], gap="large")

    with transmission_col:
        transmission = st.radio("Transmission", ["Automatic", "Manual", "Unknown"], horizontal=False)

    with features_col:
        f1, f2, f3 = st.columns(3)
        with f1:
            air_conditioner = st.checkbox("Air Conditioner", value=True)
        with f2:
            power_steering = st.checkbox("Power Steering", value=True)
        with f3:
            remote_control = st.checkbox("Remote Control", value=True)

    submitted = st.form_submit_button("▶  Estimate Market Value")


# =========================================================
# Result — gauge cluster
# =========================================================

st.write("")

if not submitted:
    st.markdown(
        """
        <div class="idle-shell">
            <span class="spec-tag"><span class="idle-lamp"></span>Gauge Idle</span>
            <h3 style="color:#3a4454; margin: 0.4rem 0 0.2rem;">Ready when you are</h3>
            <p style="margin:0;">Fill in the vehicle specs above, then hit <b>Estimate Market Value</b> to bring the needle to life.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    car_age = max(0, 2026 - int(year))

    input_data = {
        "company": company,
        "model": car_model,
        "mileage": int(mileage),
        "color": color,
        "transmission": transmission,
        "location": location,
        "listing_age_days": 0,
        "air_conditioner": int(air_conditioner),
        "automatic": int(transmission == "Automatic"),
        "power_steering": int(power_steering),
        "remote_control": int(remote_control),
        "car_age": int(car_age),
    }

    try:
        estimate = predict_price(input_data)
        mae = 119_246
        lower = max(0, estimate - mae)
        upper = estimate + mae
        gauge_max, angle_deg = gauge_geometry(estimate, mae)

        result_left, result_right = st.columns([1, 1.1], gap="large")

        with result_left:
            st.markdown(
                f"""
                <div class="result-shell">
                    <div class="gauge-wrap">
                        <div class="gauge-dial"></div>
                        <div class="gauge-hole"></div>
                        <div class="gauge-needle" style="transform: rotate({angle_deg:.1f}deg);"></div>
                        <div class="gauge-hub"></div>
                    </div>
                    <div class="odometer-unit">Estimated Market Value</div>
                    <div class="odometer">{estimate:,.0f} EGP</div>
                    <div class="range-readout">Typical range · {lower:,.0f} – {upper:,.0f} EGP</div>
                    <div class="range-readout" style="margin-top:0.2rem; color:#6f7f8f;">Dial ceiling · {gauge_max:,.0f} EGP</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with result_right:
            st.markdown(
                f"""
                <div class="reg-card">
                    <span class="spec-tag">Vehicle Summary</span>
                    <div class="reg-grid">
                        <div class="reg-item">
                            <div class="reg-label">Vehicle</div>
                            <div class="reg-value">{company} {car_model}</div>
                        </div>
                        <div class="reg-item">
                            <div class="reg-label">Year</div>
                            <div class="reg-value">{int(year)}</div>
                        </div>
                        <div class="reg-item">
                            <div class="reg-label">Mileage</div>
                            <div class="reg-value">{int(mileage):,} km</div>
                        </div>
                        <div class="reg-item">
                            <div class="reg-label">Transmission</div>
                            <div class="reg-value">{transmission}</div>
                        </div>
                        <div class="reg-item">
                            <div class="reg-label">Location</div>
                            <div class="reg-value">{location}</div>
                        </div>
                        <div class="reg-item">
                            <div class="reg-label">Car Age</div>
                            <div class="reg-value">{car_age} years</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.info(
            "The actual market price may differ because trim level, engine size, "
            "condition, accident history, service history, and ownership history "
            "are not available in the dataset."
        )

        with st.expander("View model input"):
            st.dataframe(pd.DataFrame([input_data]), use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# =========================================================
# Footer
# =========================================================

st.divider()

with st.expander("About this Project"):
    st.markdown(
        """
### Final Model

- Tuned XGBoost Regressor
- Test R²: **0.8333**
- MAE: **119,246 EGP**
- RMSE: **385,681 EGP**

### Inputs Used

- Company and model
- Manufacturing year
- Mileage
- Color
- Transmission
- Location
- Selected equipment
"""
    )

st.caption("© 2026 CarValue Egypt | Used Car Price Prediction")