from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="Egypt CarValue",
    page_icon="🚗",
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
    required = [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        FEATURE_COLUMNS_PATH,
    ]

    missing = [path.name for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing)
        )

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

    subset = raw_df.loc[
        raw_df["company"] == company_name,
        "model",
    ]

    return clean_options(subset, ["Other"])


def predict_price(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns)

    encoded = preprocessor.transform(input_df)
    predicted_log_price = model.predict(encoded)[0]
    predicted_price = float(np.expm1(predicted_log_price))

    if not np.isfinite(predicted_price) or predicted_price <= 0:
        raise ValueError("The model returned an invalid prediction.")

    return predicted_price


# =========================================================
# Styling
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(180deg, #f8f1e7 0%, #f4eadc 100%);
        color: #1f1f1f;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        background:
            linear-gradient(135deg, rgba(182,32,37,0.96), rgba(125,18,22,0.96)),
            url("https://images.unsplash.com/photo-1503736334956-4c8f8e92946d?auto=format&fit=crop&w=1600&q=80");
        background-size: cover;
        background-position: center;
        border-radius: 28px;
        padding: 2rem 2.2rem;
        color: white;
        box-shadow: 0 18px 45px rgba(81, 31, 20, 0.22);
        margin-bottom: 1.2rem;
    }

    .hero-kicker {
        font-size: 0.84rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #f3d99c;
        margin-bottom: 0.4rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        color: #ffe9e9;
        font-size: 1.05rem;
    }

    .egypt-strip {
        display: flex;
        gap: 0;
        width: 100%;
        height: 7px;
        border-radius: 99px;
        overflow: hidden;
        margin: 1rem 0 1.4rem;
    }

    .egypt-strip span {
        flex: 1;
    }

    .red { background: #ce1126; }
    .white { background: #ffffff; }
    .black { background: #000000; }

    .step-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid #e7d8c4;
        border-radius: 22px;
        padding: 1.2rem 1.25rem;
        box-shadow: 0 10px 26px rgba(87, 60, 33, 0.08);
        margin-bottom: 1rem;
    }

    .step-title {
        color: #8f1d22;
        font-size: 1.05rem;
        font-weight: 850;
        margin-bottom: 0.75rem;
    }

    .market-card {
        background: #1d6f8a;
        color: white;
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: 0 12px 28px rgba(29,111,138,0.18);
        height: 100%;
    }

    .market-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #d4edf5;
    }

    .market-value {
        font-size: 1.35rem;
        font-weight: 850;
        margin-top: 0.25rem;
    }

    .result-wrap {
        background: linear-gradient(135deg, #fffdf9, #f6ead7);
        border: 1px solid #d9c6a6;
        border-radius: 28px;
        padding: 1.6rem;
        box-shadow: 0 16px 40px rgba(87, 60, 33, 0.12);
        margin-top: 1rem;
    }

    .price-card {
        background: linear-gradient(135deg, #1f1f1f, #3a3a3a);
        color: white;
        border-radius: 24px;
        padding: 1.8rem;
        text-align: center;
    }

    .price-label {
        color: #e8d8b2;
        font-size: 0.86rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    .price-value {
        font-size: 3.4rem;
        font-weight: 900;
        margin: 0.4rem 0;
    }

    .price-range {
        color: #e8e8e8;
        font-size: 1rem;
    }

    .tip-card {
        background: #fff8ec;
        border: 1px solid #e8d3ad;
        border-radius: 18px;
        padding: 1rem 1.1rem;
    }

    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stButton"] button {
        width: 100% !important;
        min-height: 3.2rem !important;
        border-radius: 14px !important;
        border: 1px solid #b62025 !important;
        background: #b62025 !important;
        color: white !important;
        font-size: 1rem !important;
        font-weight: 850 !important;
    }

    div[data-testid="stFormSubmitButton"] button *,
    div[data-testid="stButton"] button * {
        color: white !important;
    }

    div[data-testid="stFormSubmitButton"] button:hover,
    div[data-testid="stButton"] button:hover {
        background: #8f1d22 !important;
        border-color: #8f1d22 !important;
    }

    [data-testid="stWidgetLabel"] {
        color: #2d2a26 !important;
        font-weight: 750 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Header
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Egyptian Used-Car Market</div>
        <div class="hero-title">🚗 Egypt CarValue</div>
        <div class="hero-subtitle">
            سعّر عربيتك بسهولة — get an estimated Egyptian market price in seconds.
        </div>
    </div>

    <div class="egypt-strip">
        <span class="red"></span>
        <span class="white"></span>
        <span class="black"></span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Market snapshot row
# =========================================================

s1, s2, s3, s4 = st.columns(4)

snapshot_items = [
    ("Model", "Tuned XGBoost"),
    ("R² Score", "0.8333"),
    ("Average Error", "≈ 119K EGP"),
    ("Market", "Egypt"),
]

for column, (label, value) in zip([s1, s2, s3, s4], snapshot_items):
    with column:
        st.markdown(
            f"""
            <div class="market-card">
                <div class="market-label">{label}</div>
                <div class="market-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")


# =========================================================
# Input sections
# =========================================================

companies = clean_options(
    raw_df["company"]
    if raw_df is not None and "company" in raw_df.columns
    else None,
    ["Hyundai", "Kia", "Toyota", "Nissan", "BMW", "Mercedes"],
)

colors = clean_options(
    raw_df["color"]
    if raw_df is not None and "color" in raw_df.columns
    else None,
    ["Black", "White", "Silver", "Gray", "Red", "Blue"],
)

locations = clean_options(
    raw_df["location"]
    if raw_df is not None and "location" in raw_df.columns
    else None,
    ["Cairo", "Alexandria", "Giza", "6 October"],
)

st.markdown(
    '<div class="step-title">1. Choose the car</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    company = st.selectbox(
        "Company",
        companies,
        key="company_selector",
    )

with c2:
    car_model = st.selectbox(
        "Model",
        models_for_company(company),
        key=f"model_selector_{company}",
    )

with c3:
    year = st.number_input(
        "Manufacturing Year",
        min_value=1972,
        max_value=2026,
        value=2020,
        step=1,
    )

with st.form("car_valuation_form"):
    st.markdown(
        '<div class="step-title">2. Add usage and market details</div>',
        unsafe_allow_html=True,
    )

    u1, u2, u3 = st.columns(3)

    with u1:
        mileage = st.number_input(
            "Mileage (km)",
            min_value=0,
            max_value=1_000_000,
            value=80_000,
            step=5_000,
        )

    with u2:
        color = st.selectbox(
            "Color",
            colors,
        )

    with u3:
        location = st.selectbox(
            "Location",
            locations,
        )

    st.markdown(
        '<div class="step-title">3. Select transmission and features</div>',
        unsafe_allow_html=True,
    )

    transmission = st.radio(
        "Transmission",
        ["Automatic", "Manual", "Unknown"],
        horizontal=True,
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        air_conditioner = st.checkbox(
            "Air Conditioner",
            value=True,
        )

    with f2:
        power_steering = st.checkbox(
            "Power Steering",
            value=True,
        )

    with f3:
        remote_control = st.checkbox(
            "Remote Control",
            value=True,
        )

    submitted = st.form_submit_button(
        "Calculate Egyptian Market Price"
    )


# =========================================================
# Prediction result
# =========================================================

if submitted:
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

        st.markdown(
            '<div class="result-wrap">',
            unsafe_allow_html=True,
        )

        left_result, right_result = st.columns([1.15, 1])

        with left_result:
            st.markdown(
                f"""
                <div class="price-card">
                    <div class="price-label">Estimated Market Value</div>
                    <div class="price-value">{estimate:,.0f} EGP</div>
                    <div class="price-range">
                        Typical range: {lower:,.0f} – {upper:,.0f} EGP
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right_result:
            st.markdown(
                f"""
                <div class="tip-card">
                    <h3>Vehicle Summary</h3>
                    <p><b>Car:</b> {company} {car_model}</p>
                    <p><b>Year:</b> {int(year)}</p>
                    <p><b>Age:</b> {car_age} years</p>
                    <p><b>Mileage:</b> {int(mileage):,} km</p>
                    <p><b>Transmission:</b> {transmission}</p>
                    <p><b>Location:</b> {location}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

        st.info(
            "The final market price may vary because trim level, engine size, "
            "condition, accident history, service history, and ownership history "
            "are not available in the dataset."
        )

        with st.expander("View model input"):
            st.dataframe(
                pd.DataFrame([input_data]),
                use_container_width=True,
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# =========================================================
# Footer
# =========================================================

st.divider()

st.markdown(
    """
### About this estimator

Egypt CarValue was trained on Egyptian used-car listings and compares several
regression models. The tuned XGBoost model achieved the strongest performance
and is used for the final market estimate.
"""
)

st.caption(
    "© 2026 Egypt CarValue | Egyptian Used-Car Price Prediction"
)