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

    transformed = preprocessor.transform(input_df)
    predicted_log_price = model.predict(transformed)[0]
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
        background: #f6f8fb;
        color: #1d2735;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.1rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #eef4f8 100%);
        border: 1px solid #dfe7ee;
        border-radius: 26px;
        padding: 1.7rem 1.9rem;
        box-shadow: 0 16px 40px rgba(41, 57, 75, 0.08);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        color: #15324b;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        color: #667789;
        font-size: 1.02rem;
    }

    .section-title {
        color: #15324b;
        font-size: 1.15rem;
        font-weight: 850;
        margin-bottom: 0.8rem;
    }

    .panel {
        background: white;
        border: 1px solid #e1e8ef;
        border-radius: 22px;
        padding: 1.2rem;
        box-shadow: 0 10px 28px rgba(41, 57, 75, 0.06);
    }

    .summary-card {
        background: #ffffff;
        border: 1px solid #e1e8ef;
        border-radius: 18px;
        padding: 1rem 1.05rem;
        box-shadow: 0 8px 20px rgba(41, 57, 75, 0.05);
        height: 100%;
    }

    .summary-label {
        color: #7a8a9a;
        font-size: 0.78rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .summary-value {
        color: #15324b;
        font-size: 1.35rem;
        font-weight: 850;
        margin-top: 0.25rem;
    }

    .result-shell {
        background: white;
        border: 1px solid #dfe7ee;
        border-radius: 24px;
        padding: 1.35rem;
        box-shadow: 0 14px 34px rgba(41, 57, 75, 0.08);
    }

    .price-card {
        background: linear-gradient(135deg, #163b5b 0%, #205d7b 100%);
        color: white;
        border-radius: 22px;
        padding: 1.7rem;
        text-align: left;
    }

    .price-label {
        color: #cfe2ec;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 750;
    }

    .price-value {
        font-size: 3rem;
        font-weight: 900;
        margin: 0.35rem 0 0.5rem;
    }

    .price-range {
        color: #e3edf3;
        font-size: 0.98rem;
    }

    .detail-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.7rem;
        margin-top: 1rem;
    }

    .detail-item {
        background: #f8fafc;
        border: 1px solid #e3e9ef;
        border-radius: 14px;
        padding: 0.85rem;
    }

    .detail-label {
        color: #7b8b9b;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 750;
    }

    .detail-value {
        color: #1d3348;
        font-size: 1rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    .empty-result {
        background: white;
        border: 1px dashed #cbd6df;
        border-radius: 22px;
        padding: 2rem 1.4rem;
        text-align: center;
        color: #6f7f8f;
    }

    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stButton"] button {
        width: 100% !important;
        min-height: 3.15rem !important;
        border-radius: 13px !important;
        border: 1px solid #1d5f7d !important;
        background: #1d5f7d !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
    }

    div[data-testid="stFormSubmitButton"] button *,
    div[data-testid="stButton"] button * {
        color: #ffffff !important;
    }

    div[data-testid="stFormSubmitButton"] button:hover,
    div[data-testid="stButton"] button:hover {
        background: #174d66 !important;
        border-color: #174d66 !important;
    }

    [data-testid="stWidgetLabel"] {
        color: #24384c !important;
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
        <div class="hero-title">🚘 CarValue Egypt</div>
        <div class="hero-subtitle">
            Estimate the market value of a used car in Egypt using a tuned XGBoost model.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Quick model summary
# =========================================================

s1, s2, s3, s4 = st.columns(4)

summary_items = [
    ("Model", "Tuned XGBoost"),
    ("R² Score", "0.8333"),
    ("Average Error", "≈ 119K EGP"),
    ("Market", "Egypt"),
]

for column, (label, value) in zip([s1, s2, s3, s4], summary_items):
    with column:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">{label}</div>
                <div class="summary-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")


# =========================================================
# Main workspace
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

selection_col, options_col = st.columns([1.15, 1], gap="large")

with selection_col:
    st.markdown(
        '<div class="section-title">Vehicle Selection</div>',
        unsafe_allow_html=True,
    )

    company = st.selectbox(
        "Company",
        companies,
        key="company_selector",
    )

    car_model = st.selectbox(
        "Model",
        models_for_company(company),
        key=f"model_selector_{company}",
    )

    year = st.number_input(
        "Manufacturing Year",
        min_value=1972,
        max_value=2026,
        value=2020,
        step=1,
    )

with options_col:
    st.markdown(
        '<div class="section-title">Market Details</div>',
        unsafe_allow_html=True,
    )

    mileage = st.number_input(
        "Mileage (km)",
        min_value=0,
        max_value=1_000_000,
        value=80_000,
        step=5_000,
    )

    color = st.selectbox(
        "Color",
        colors,
    )

    location = st.selectbox(
        "Location",
        locations,
    )

st.write("")

with st.form("car_features_form"):
    st.markdown(
        '<div class="section-title">Transmission and Features</div>',
        unsafe_allow_html=True,
    )

    transmission_col, features_col = st.columns([1, 1.5], gap="large")

    with transmission_col:
        transmission = st.radio(
            "Transmission",
            ["Automatic", "Manual", "Unknown"],
            horizontal=False,
        )

    with features_col:
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
        "Estimate Market Value"
    )


# =========================================================
# Result area
# =========================================================

st.write("")
st.markdown(
    '<div class="section-title">Valuation Result</div>',
    unsafe_allow_html=True,
)

if not submitted:
    st.markdown(
        """
        <div class="empty-result">
            <h3>Ready when you are</h3>
            <p>
                Enter the car details above, then click
                <b>Estimate Market Value</b>.
            </p>
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

        result_left, result_right = st.columns([1.2, 1], gap="large")

        with result_left:
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

        with result_right:
            st.markdown(
                f"""
                <div class="result-shell">
                    <div class="section-title">Vehicle Summary</div>

                    <div class="detail-grid">
                        <div class="detail-item">
                            <div class="detail-label">Vehicle</div>
                            <div class="detail-value">{company} {car_model}</div>
                        </div>

                        <div class="detail-item">
                            <div class="detail-label">Year</div>
                            <div class="detail-value">{int(year)}</div>
                        </div>

                        <div class="detail-item">
                            <div class="detail-label">Mileage</div>
                            <div class="detail-value">{int(mileage):,} km</div>
                        </div>

                        <div class="detail-item">
                            <div class="detail-label">Transmission</div>
                            <div class="detail-value">{transmission}</div>
                        </div>

                        <div class="detail-item">
                            <div class="detail-label">Location</div>
                            <div class="detail-value">{location}</div>
                        </div>

                        <div class="detail-item">
                            <div class="detail-label">Car Age</div>
                            <div class="detail-value">{car_age} years</div>
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

st.caption(
    "© 2026 CarValue Egypt | Used Car Price Prediction"
)