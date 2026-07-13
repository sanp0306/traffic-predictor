import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. SET UP PAGE CONFIGURATION & TITLE
st.set_page_config(page_title="Traffic Volume Predictor", page_icon="🚦", layout="centered")
st.title("🚦 Traffic Volume Prediction App")
st.write("Input the current traffic parameters below to predict the traffic density using our trained machine learning models.")

# 2. LOAD TRAINED MODELS AND SCALER
@st.cache_resource
def load_assets():
    models = {
        "Logistic Regression": joblib.load("logistic_model.pkl"),
        "Random Forest": joblib.load("random_forest_model.pkl"),
        "Gradient Boosting": joblib.load("gradient_boosting_model.pkl")
    }
    scaler = joblib.load("scaler.pkl")
    return models, scaler

try:
    models, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# 3. SIDEBAR - MODEL SELECTION
st.sidebar.header("Model Settings")
selected_model_name = st.sidebar.selectbox(
    "Choose ML Model", 
    list(models.keys())
)
model = models[selected_model_name]

# 4. USER INPUT INTERFACE (FORM)
st.subheader("Traffic Parameters")
col1, col2 = st.columns(2)

with col1:
    car_count = st.number_input("Car Count", min_value=0, max_value=500, value=25, step=1)
    bike_count = st.number_input("Bike Count", min_value=0, max_value=500, value=5, step=1)
    bus_count = st.number_input("Bus Count", min_value=0, max_value=200, value=2, step=1)

with col2:
    truck_count = st.number_input("Truck Count", min_value=0, max_value=200, value=10, step=1)
    total_volume = car_count + bike_count + bus_count + truck_count
    st.metric(label="Calculated Total Volume", value=total_volume)

st.markdown("### Temporal Parameters")
day_of_month = st.selectbox("Day of the Month", list(range(1, 32)), index=9)
day_of_week = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
time_window = st.text_input("Time Window String (e.g. 12:00:00 AM)", value="12:00:00 AM")

# 5. DATA PREPROCESSING AND FEATURE MAPPING
# Map categorical values to numeric codes matching your training data setup
day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
day_code = day_mapping[day_of_week]

# Simple mock string length hashing/encoding fallback for Time Window if your training utilized label encoding
# (Adjust this specific feature transformation if your pipeline treated time window differently!)
time_encoded = len(time_window) 

# Assemble into the precise feature shape expected by your Scikit-Learn Scaler
# Make sure the ordering matches your original X DataFrame layout:
features = np.array([[car_count, bike_count, bus_count, truck_count, total_volume, day_of_month, day_code, time_encoded]])

# 6. LIVE PREDICTION PIPELINE
if st.button("Generate Traffic Prediction", type="primary"):
    with st.spinner("Calculating predictions..."):
        try:
            # Apply standard scaler transformation
            input_scaled = scaler.transform(features)
            
            # Run inference execution on the dynamically selected model
            prediction_raw = model.predict(input_scaled)[0]
            
            # Automatically convert continuous decimals (like 1.97) to nearest category integer
            try:
                prediction = int(round(float(prediction_raw)))
            except (ValueError, TypeError):
                prediction = prediction_raw

            # Target text mappings for output display
            target_display = {0: "Low Traffic", 1: "Medium Traffic", 2: "High Traffic"}
            result_text = target_display.get(prediction, f"Custom Value ({prediction_raw:.2f})")

            # Render clean results with color indicators
            st.markdown("---")
            if prediction == 0:
                st.success(f"Prediction: **{result_text}** 🟢 (Raw score: {prediction_raw:.2f})")
            elif prediction == 1:
                st.warning(f"Prediction: **{result_text}** 🟡 (Raw score: {prediction_raw:.2f})")
            elif prediction == 2:
                st.error(f"Prediction: **{result_text}** 🔴 (Raw score: {prediction_raw:.2f})")
            else:
                st.info(f"Prediction: **{result_text}** 🔵")
                
        except Exception as error:
            st.error(f"An error occurred during pipeline execution: {error}")
