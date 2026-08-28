# ============================================================
# WATER QUALITY CLASSIFICATION AND PREDICTION SYSTEM
# ============================================================

import streamlit as st
import pandas as pd
import joblib


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Water Quality Prediction",
    page_icon="💧",
    layout="centered"
)


# ------------------------------------------------------------
# LOAD TRAINED ML MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("water_quality_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(str(e))
    st.stop()


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.title("💧 Water Quality Prediction System")

st.write(
    "Enter the water-quality parameters below to predict "
    "whether the sample is potable or not potable."
)

st.divider()


# ------------------------------------------------------------
# INPUT SECTION
# ------------------------------------------------------------

st.subheader("🔬 Water Quality Parameters")

col1, col2 = st.columns(2)

with col1:

    ph = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=7.0,
        step=0.1
    )

    hardness = st.number_input(
        "Hardness",
        min_value=0.0,
        value=196.0,
        step=1.0
    )

    solids = st.number_input(
        "Solids",
        min_value=0.0,
        value=22000.0,
        step=100.0
    )

    chloramines = st.number_input(
        "Chloramines",
        min_value=0.0,
        value=7.0,
        step=0.1
    )

    sulfate = st.number_input(
        "Sulfate",
        min_value=0.0,
        value=330.0,
        step=1.0
    )


with col2:

    conductivity = st.number_input(
        "Conductivity",
        min_value=0.0,
        value=420.0,
        step=1.0
    )

    organic_carbon = st.number_input(
        "Organic Carbon",
        min_value=0.0,
        value=14.0,
        step=0.1
    )

    trihalomethanes = st.number_input(
        "Trihalomethanes",
        min_value=0.0,
        value=66.0,
        step=1.0
    )

    turbidity = st.number_input(
        "Turbidity",
        min_value=0.0,
        value=4.0,
        step=0.1
    )


st.divider()


# ------------------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------------------

if st.button(
    "🔍 Predict Water Quality",
    use_container_width=True
):

    # Create input DataFrame.
    # Column names MUST match the training dataset exactly.

    new_water_sample = pd.DataFrame({
        "ph": [ph],
        "Hardness": [hardness],
        "Solids": [solids],
        "Chloramines": [chloramines],
        "Sulfate": [sulfate],
        "Conductivity": [conductivity],
        "Organic_carbon": [organic_carbon],
        "Trihalomethanes": [trihalomethanes],
        "Turbidity": [turbidity]
    })


    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(new_water_sample)[0]

    except Exception as e:

        st.error("Prediction failed.")
        st.code(str(e))
        st.stop()


    # --------------------------------------------------------
    # DISPLAY PREDICTION
    # --------------------------------------------------------

    st.subheader("📊 Prediction Result")

    if prediction == 1:

        st.success(
            "🟢 POTABLE\n\n"
            "The model classified this water sample as potable."
        )

    else:

        st.error(
            "🔴 NOT POTABLE\n\n"
            "The model classified this water sample as not potable."
        )


    # --------------------------------------------------------
    # PREDICTION PROBABILITY
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            new_water_sample
        )[0]

        st.subheader("Prediction Probability")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Not Potable",
                f"{probabilities[0] * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Potable",
                f"{probabilities[1] * 100:.2f}%"
            )


# ------------------------------------------------------------
# PROJECT INFORMATION
# ------------------------------------------------------------

st.divider()

with st.expander("ℹ️ About This Project"):

    st.write(
        """
        **Project Title:**  
        Machine Learning-Based Water Quality Classification
        and Prediction System

        **Machine Learning Task:**  
        Binary Classification

        **Target:**  
        Potability

        **Input Parameters:**  
        pH, Hardness, Solids, Chloramines, Sulfate,
        Conductivity, Organic Carbon, Trihalomethanes,
        and Turbidity.
        """
    )


# ------------------------------------------------------------
# DISCLAIMER
# ------------------------------------------------------------

st.divider()

st.caption(
    "⚠️ This application provides a machine-learning prediction "
    "based on the training dataset. It does not replace laboratory "
    "testing or official drinking-water quality certification."
)