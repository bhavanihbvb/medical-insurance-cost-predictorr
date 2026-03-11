import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# Load trained models
# -----------------------------
@st.cache_resource
def load_models():

    if not os.path.exists("models.pkl"):
        st.error("models.pkl not found. Run train_models.py first.")
        st.stop()

    with open("models.pkl", "rb") as f:
        payload = pickle.load(f)

    return payload


payload = load_models()

models = payload["models"]
results = payload["results"]
best_model = payload["best"]

# -----------------------------
# PAGE TITLE
# -----------------------------

st.title("🏥 Medical Insurance Cost Predictor")
st.write("Predict insurance charges using Machine Learning")

st.divider()

# -----------------------------
# USER INPUTS
# -----------------------------

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 30)

with col2:
    sex = st.selectbox("Sex", ["male", "female"])

with col3:
    children = st.selectbox("Children", [0,1,2,3,4,5])

col4, col5, col6 = st.columns(3)

with col4:
    bmi = st.slider("BMI", 15.0, 50.0, 25.0)

with col5:
    smoker = st.selectbox("Smoker", ["yes","no"])

with col6:
    region = st.selectbox(
        "Region",
        ["southwest","southeast","northwest","northeast"]
    )

st.divider()

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict Insurance Cost"):

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "age_bmi": age * bmi
    }])

    predictions = {}

    for name, model in models.items():
        pred = model.predict(input_data)[0]
        predictions[name] = pred

    best_prediction = predictions[best_model]

    st.success(f"Predicted Insurance Cost: **${best_prediction:,.2f}**")

    # -----------------------------
    # MODEL PREDICTIONS
    # -----------------------------

    st.subheader("Model Predictions")

    pred_df = pd.DataFrame({
        "Model": list(predictions.keys()),
        "Prediction": list(predictions.values())
    })

    fig = px.bar(
        pred_df,
        x="Model",
        y="Prediction",
        color="Model"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # RISK GAUGE
    # -----------------------------

    risk_score = min(int(best_prediction / 60000 * 100), 100)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Insurance Risk Level"},
        gauge={
            "axis": {"range": [0,100]},
            "steps":[
                {"range":[0,33],"color":"lightgreen"},
                {"range":[33,66],"color":"yellow"},
                {"range":[66,100],"color":"red"}
            ]
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------

st.divider()
st.subheader("Model Performance")

res_df = pd.DataFrame(results)

fig2 = px.bar(
    res_df,
    x="Model",
    y="R2",
    color="Model"
)

st.plotly_chart(fig2, use_container_width=True)