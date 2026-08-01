
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="CarValue Egypt",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "used_cars" / "hatla2ee_cars_august_2025.csv"

MODEL_PATH = MODEL_DIR / "best_xgboost.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"


@st.cache_resource
def load_artifacts():
    missing = [
        p.name
        for p in [MODEL_PATH, PREPROCESSOR_PATH, FEATURE_COLUMNS_PATH]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing)
        )

    return (
        joblib.load(MODEL_PATH),
        joblib.load(PREPROCESSOR_PATH),
        joblib.load(FEATURE_COLUMNS_PATH),
    )


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)

    for col in ["company", "model", "color", "location"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    return df


def clean_options(series, fallback):
    if series is None:
        return fallback

    values = pd.Series(series).dropna().astype(str).str.strip()
    values = values[values.ne("")]
    result = sorted(values.unique().tolist())

    return result or fallback


def predict_price(values):
    frame = pd.DataFrame([values]).reindex(columns=feature_columns)
    transformed = preprocessor.transform(frame)
    predicted_log = model.predict(transformed)[0]
    prediction = float(np.expm1(predicted_log))

    if not np.isfinite(prediction) or prediction <= 0:
        raise ValueError("The model returned an invalid prediction.")

    return prediction


try:
    model, preprocessor, feature_columns = load_artifacts()
except Exception as error:
    st.error(str(error))
    st.stop()

raw_df = load_data()


def models_for_company(company):
    if raw_df is None or not {"company", "model"}.issubset(raw_df.columns):
        return ["Elantra", "Corolla", "Sunny", "Cerato", "Tiguan"]

    subset = raw_df.loc[raw_df["company"] == company, "model"]
    return clean_options(subset, ["Other"])


st.markdown(
    """
    <style>
    .stApp {
        background: #07111c;
        color: white;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 24px;
        background:
            radial-gradient(circle at top right, rgba(31, 146, 193, 0.35), transparent 35%),
            linear-gradient(135deg, #0d2436, #081722);
        border: 1px solid rgba(255,255,255,0.09);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 850;
        letter-spacing: -0.04em;
    }

    .hero-subtitle {
        color: #a9bdcb;
        font-size: 1rem;
    }

    .input-card {
        background: #0d1d29;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 1.2rem;
    }

    .result-card {
        background: linear-gradient(145deg, #0f6c91, #10364d);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 24px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 24px 70px rgba(0,0,0,0.25);
    }

    .result-label {
        color: #cce2ee;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.13em;
    }

    .result-price {
        color: white;
        font-size: 3rem;
        font-weight: 850;
        margin: 0.5rem 0;
    }

    .result-range {
        color: #d8e8f0;
    }

    .stButton > button {
        width: 100%;
        min-height: 3.1rem;
        border-radius: 13px;
        border: none;
        background: #1684b1;
        color: white;
        font-weight: 750;
    }

    .stButton > button:hover {
        background: #1a99ca;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🚘 CarValue Egypt</div>
        <div class="hero-subtitle">
            Estimate the Egyptian market value of a used car using your tuned XGBoost model.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

form_col, result_col = st.columns([1.45, 1], gap="large")

with form_col:
    st.subheader("Vehicle details")

    c1, c2, c3 = st.columns(3)

    with c1:
        company = st.selectbox("Company", companies)

    with c2:
        car_model = st.selectbox(
            "Model",
            models_for_company(company),
            key=f"model_{company}",
        )

    with c3:
        year = st.number_input(
            "Manufacturing year",
            min_value=1972,
            max_value=2026,
            value=2020,
            step=1,
        )

    with st.form("car_form"):
        c4, c5, c6 = st.columns(3)

        with c4:
            mileage = st.number_input(
                "Mileage (km)",
                min_value=0,
                max_value=1_000_000,
                value=80_000,
                step=5_000,
            )

        with c5:
            color = st.selectbox("Color", colors)

        with c6:
            location = st.selectbox("Location", locations)

        transmission = st.radio(
            "Transmission",
            ["Automatic", "Manual", "Unknown"],
            horizontal=True,
        )

        e1, e2, e3 = st.columns(3)

        with e1:
            air_conditioner = st.checkbox("Air conditioner", value=True)

        with e2:
            power_steering = st.checkbox("Power steering", value=True)

        with e3:
            remote_control = st.checkbox("Remote control", value=True)

        submitted = st.form_submit_button("Estimate vehicle value")


with result_col:
    st.subheader("Market estimate")

    if not submitted:
        st.markdown(
            """
            <div class="input-card">
                <h3>Ready for valuation</h3>
                <p>
                    Enter the vehicle details and select its equipment,
                    then click <b>Estimate vehicle value</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Final model", "Tuned XGBoost")
        m2.metric("R² on EGP scale", "0.8333")

        m3, m4 = st.columns(2)
        m3.metric("MAE", "≈ 119K EGP")
        m4.metric("Target", "Used-car price")

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

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Estimated market value</div>
                    <div class="result-price">{estimate:,.0f} EGP</div>
                    <div class="result-range">
                        Typical range: {lower:,.0f} – {upper:,.0f} EGP
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            a, b = st.columns(2)
            a.metric("Vehicle", f"{company} {car_model}")
            b.metric("Car age", f"{car_age} years")

            c, d = st.columns(2)
            c.metric("Mileage", f"{int(mileage):,} km")
            d.metric("Transmission", transmission)

            with st.expander("View model input"):
                st.dataframe(
                    pd.DataFrame([input_data]),
                    use_container_width=True,
                )

            st.info(
                "Actual value may differ because trim level, engine size, "
                "condition, accident history, and service history are unavailable."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

st.caption("© 2026 CarValue Egypt | Used Car Price Prediction")