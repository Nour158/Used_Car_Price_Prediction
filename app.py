from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CarValue Egypt",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "used_cars" / "hatla2ee_cars_august_2025.csv"

MODEL_PATH = MODEL_DIR / "best_xgboost.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"

st.markdown(
    '''
    <style>
    .stApp {
        background: linear-gradient(180deg, #07131f 0%, #0c1e2d 100%);
        color: #f4f7fb;
    }

    [data-testid="stSidebar"] {
        background: #081621;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #b7c6d5;
        font-size: 1.05rem;
        margin-bottom: 1.4rem;
    }

    .hero-card {
        padding: 1.4rem 1.5rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(28,115,165,0.30), rgba(7,26,39,0.78));
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 1rem;
    }

    .result-card {
        padding: 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0d5d83, #12354b);
        border: 1px solid rgba(255,255,255,0.14);
        text-align: center;
    }

    .result-label {
        color: #c7dce9;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    .result-price {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0.4rem 0;
    }

    .result-range {
        color: #dceaf2;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        min-height: 3.2rem;
        font-size: 1.05rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a8dbf, #126f99);
        color: white;
        border: none;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)

@st.cache_resource
def load_artifacts():
    required = [MODEL_PATH, PREPROCESSOR_PATH, FEATURE_COLUMNS_PATH]
    missing = [path.name for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing model files: "
            + ", ".join(missing)
            + ". Place them inside the root models folder."
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

    company_models = raw_df.loc[
        raw_df["company"] == company_name,
        "model"
    ]

    return clean_options(
        company_models,
        clean_options(raw_df["model"], ["Other"])
    )


def predict_price(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns)

    encoded = preprocessor.transform(input_df)
    predicted_log_price = model.predict(encoded)[0]
    predicted_price = float(np.expm1(predicted_log_price))

    if not np.isfinite(predicted_price) or predicted_price <= 0:
        raise ValueError("The model returned an invalid prediction.")

    return predicted_price


try:
    model, preprocessor, feature_columns = load_artifacts()
except Exception as error:
    st.error(str(error))
    st.stop()

raw_df = load_dataset()

st.markdown(
    '<div class="main-title">CarValue Egypt</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">A modern Egyptian used-car valuation experience.</div>',
    unsafe_allow_html=True
)

st.markdown(
    '''
    <div class="hero-card">
        <b>Enter the vehicle details</b> to receive an estimated market value
        from the tuned XGBoost model.
    </div>
    ''',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Model snapshot")
    st.metric("Best model", "Tuned XGBoost")
    st.metric("R² on EGP scale", "0.8333")
    st.metric("MAE", "≈ 119K EGP")
    st.caption("Predictions are estimates, not guaranteed selling prices.")

companies = clean_options(
    raw_df["company"]
    if raw_df is not None and "company" in raw_df.columns
    else None,
    ["Hyundai", "Kia", "Toyota", "Nissan", "BMW", "Mercedes"]
)

colors = clean_options(
    raw_df["color"]
    if raw_df is not None and "color" in raw_df.columns
    else None,
    ["Black", "White", "Silver", "Gray", "Red", "Blue"]
)

locations = clean_options(
    raw_df["location"]
    if raw_df is not None and "location" in raw_df.columns
    else None,
    ["Cairo", "Alexandria", "Giza", "6 October", "Nasr city"]
)

# Company and model stay outside the form so the model list updates instantly.
st.subheader("Vehicle identity")

c1, c2, c3 = st.columns(3)

with c1:
    company = st.selectbox(
        "Company",
        companies,
        key="company_selector"
    )

company_models = models_for_company(company)

with c2:
    car_model = st.selectbox(
        "Model",
        company_models,
        key=f"model_selector_{company}"
    )

with c3:
    year = st.number_input(
        "Manufacturing year",
        min_value=1972,
        max_value=2026,
        value=2020,
        step=1
    )

with st.form("valuation_form"):
    st.subheader("Usage and market details")

    c4, c5, c6 = st.columns(3)

    with c4:
        mileage = st.number_input(
            "Mileage (km)",
            min_value=0,
            max_value=1_000_000,
            value=80_000,
            step=5_000
        )

    with c5:
        color = st.selectbox("Color", colors)

    with c6:
        location = st.selectbox("Location", locations)

    st.subheader("Transmission and equipment")

    transmission = st.radio(
        "Transmission",
        ["Automatic", "Manual", "Unknown"],
        horizontal=True
    )

    e1, e2, e3 = st.columns(3)

    with e1:
        air_conditioner = st.checkbox(
            "Air conditioner",
            value=True
        )

    with e2:
        power_steering = st.checkbox(
            "Power steering",
            value=True
        )

    with e3:
        remote_control = st.checkbox(
            "Remote control",
            value=True
        )

    submitted = st.form_submit_button("Estimate market value")

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
        estimated_price = predict_price(input_data)

        mae_egp = 119_246
        lower = max(0, estimated_price - mae_egp)
        upper = estimated_price + mae_egp

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f'''
            <div class="result-card">
                <div class="result-label">Estimated market value</div>
                <div class="result-price">{estimated_price:,.0f} EGP</div>
                <div class="result-range">
                    Typical error band: {lower:,.0f} – {upper:,.0f} EGP
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.write("")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Vehicle", f"{company} {car_model}")
        m2.metric("Car age", f"{car_age} years")
        m3.metric("Mileage", f"{int(mileage):,} km")
        m4.metric("Transmission", transmission)

        with st.expander("View model input"):
            st.dataframe(
                pd.DataFrame([input_data]),
                use_container_width=True
            )

        st.info(
            "Trim level, engine size, condition, accident history, service "
            "history, and other unavailable details may change the real price."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")