from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ------------------ PATHS ------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "ev_range_model.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "india_ev_range_dataset.csv"

FEATURES = [
    "make",
    "model",
    "variant",
    "battery_capacity_kWh",
    "motor_power_kW",
    "vehicle_weight_kg",
    "motor_efficiency_pct",
    "avg_driving_speed_kmh",
    "ambient_temperature_C",
    "city_road_type",
    "battery_age_years",
]

CATEGORICAL = ["make", "model", "variant", "city_road_type"]

# ------------------ PAGE CONFIG ------------------

st.set_page_config(
    page_title="EV Range Predictor",
    page_icon="🔋",
    layout="wide",
)

# ------------------ LOAD FILES ------------------


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


st.markdown(
    """
<style>
/* Main app background */
.stApp {
    background-color: #0B1220;
    color: #F1F5F9;
}

/* Main content area */
[data-testid="stAppViewContainer"] {
    background-color: #2E7D32 ;
}

/* Sidebar, if used */
[data-testid="stSidebar"] {
    background-color: #111C2E;
}

/* Header */
.hero {
    background: linear-gradient(135deg, #123B4A, #087F8C);
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 24px;
    border: 1px solid #285568;
}

.hero h1 {
    color: #FFFFFF;
}

.hero p {
    color: #D7F3F0;
}

/* Form and cards */
div[data-testid="stForm"],
div[data-testid="stMetric"] {
    background-color: #111C2E;
    border: 1px solid #26374D;
    border-radius: 14px;
}

/* Input fields */
div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input {
    background-color: #172438;
    color: #F1F5F9;
}

/* Buttons */
div[data-testid="stFormSubmitButton"] button {
    background-color: #0D9488;
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #0F766E;
    color: #FFFFFF;
    border: 1px solid #5EEAD4;
}

/* Divider */
hr {
    border-color: #26374D;
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<style>
/* Main content */
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */
.hero {
    padding: 2rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #102a43, #087f8c);
    margin-bottom: 1.5rem;
    border: 1px solid #24566b;
}
.hero h1 {
    color: #ffffff;
    font-size: 2.2rem;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #d8f3f0;
    font-size: 1rem;
}

/* Section headings */
h2, h3 {
    letter-spacing: -0.3px;
}

/* Input cards */
div[data-testid="stForm"] {
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.4rem;
    background: rgba(30, 41, 59, 0.25);
}

/* Inputs */
div[data-testid="stSelectbox"],
div[data-testid="stNumberInput"] {
    margin-bottom: 0.6rem;
}

/* Buttons */
div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    border-radius: 10px;
    min-height: 3rem;
    font-weight: 600;
    transition: 0.2s ease;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #132d3a;
    border: 1px solid #24566b;
    padding: 1.2rem;
    border-radius: 14px;
}

/* Tables */
div[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 10px;
    overflow: hidden;
}

/* Mobile spacing */
@media (max-width: 640px) {
    .block-container {
        padding: 1rem;
    }
    .hero {
        padding: 1.3rem;
    }
    .hero h1 {
        font-size: 1.7rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="hero">
    <h1>🔋 EV Range Predictor</h1>
    <p>Estimate your electric vehicle's real-world driving range.</p>
</div>
""",
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.stop()

model = load_model()
df = load_data()

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Missing dataset columns: {missing}")
    st.stop()

# Ensure this is the matching 11-feature pipeline
try:
    expected = list(model.named_steps["preprocessor"].feature_names_in_)
    if set(expected) != set(FEATURES) or len(expected) != 11:
        st.error(
            "Your saved model does not match these 11 features. "
            "Retrain and save the 11-feature pipeline in Jupyter."
        )
        st.stop()
except (AttributeError, KeyError):
    st.error(
        "The saved file is not the expected preprocessing pipeline. "
        "Save the complete preprocessing + model pipeline from Jupyter."
    )
    st.stop()


def options(column):
    values = df[column].dropna().unique().tolist()
    return sorted(values, key=lambda x: str(x).lower())


st.subheader("🚘 Vehicle Information")
st.caption("Enter the details below. Fields are based on your dataset.")

with st.form("ev_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        make = st.selectbox("Vehicle Make", options("make"))
        model_name = st.selectbox("Vehicle Model", options("model"))
        variant = st.selectbox("Variant", options("variant"))
        battery_capacity = st.number_input(
            "Battery Capacity (kWh)",
            min_value=0.1,
            value=40.0,
            step=1.0,
        )
        motor_power = st.number_input(
            "Motor Power (kW)",
            min_value=0.1,
            value=100.0,
            step=5.0,
        )
        vehicle_weight = st.number_input(
            "Vehicle Weight (kg)",
            min_value=1.0,
            value=1600.0,
            step=50.0,
        )

    with col2:
        motor_efficiency = st.number_input(
            "Motor Efficiency (%)",
            min_value=0.1,
            max_value=100.0,
            value=90.0,
            step=1.0,
        )
        avg_speed = st.number_input(
            "Average Driving Speed (km/h)",
            min_value=0.1,
            value=40.0,
            step=5.0,
        )
        temperature = st.number_input(
            "Ambient Temperature (°C)",
            min_value=-30.0,
            max_value=60.0,
            value=25.0,
            step=1.0,
        )
        city_road = st.selectbox(
            "City / Road Type",
            options("city_road_type"),
        )
        battery_age = st.number_input(
            "Battery Age (years)",
            min_value=0.0,
            max_value=30.0,
            value=1.0,
            step=0.5,
        )

    submitted = st.form_submit_button(
        "⚡ Predict EV Range",
        type="primary",
        use_container_width=True,
    )

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "make": make,
                "model": model_name,
                "variant": variant,
                "battery_capacity_kWh": battery_capacity,
                "motor_power_kW": motor_power,
                "vehicle_weight_kg": vehicle_weight,
                "motor_efficiency_pct": motor_efficiency,
                "avg_driving_speed_kmh": avg_speed,
                "ambient_temperature_C": temperature,
                "city_road_type": city_road,
                "battery_age_years": battery_age,
            }
        ],
        columns=FEATURES,
    )

    try:
        prediction = float(model.predict(input_data)[0])

        st.divider()
        st.subheader("📊 Prediction Result")

        result_col, info_col = st.columns([1, 2])

        with result_col:
            st.metric(
                "Estimated Real-World Range",
                f"{prediction:.1f} km",
            )

        with info_col:
            st.info(
                "This is an experimental ML estimate, not a "
                "guaranteed driving distance. Actual range varies "
                "with traffic, weather, driving behaviour and "
                "battery condition."
            )

        with st.expander("View submitted vehicle details"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")
