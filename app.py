import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Load the shared preprocessing scaler
scaler = joblib.load("scaler.pkl")

# Target text mappings for output display
target_display = {0: "Low Traffic", 1: "Medium Traffic", 2: "High Traffic"}

# ==========================================
# 2. STREAMLIT USER INTERFACE SETUP
# ==========================================
st.title("🚦 Multi-Model Traffic Situation Predictor")
st.write(
    "Test live traffic predictions across your three different trained classifiers."
)

# Sidebar layout for choosing the ML model
st.sidebar.header("Model Configuration")
selected_model_name = st.sidebar.selectbox(
    "Choose Active Classifier",
    ["Random Forest", "Logistic Regression", "Decision Tree"],
)

# Dynamically load the .pkl file based on the sidebar dropdown selection
model_files = {
    "Random Forest": "random_forest_model.pkl",
    "Logistic Regression": "logistic_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
}
model = joblib.load(model_files[selected_model_name])
st.sidebar.success(f"Running Inference via: **{selected_model_name}**")

# Interactive input fields for vehicle metrics split into columns
st.header("Current Vehicle Metrics")
col1, col2 = st.columns(2)
with col1:
    car = col1.number_input("Car Count", min_value=0, value=25)
    bike = col1.number_input("Bike Count", min_value=0, value=5)
with col2:
    bus = col2.number_input("Bus Count", min_value=0, value=2)
    truck = col2.number_input("Truck Count", min_value=0, value=10)

# Input fields for date, day, and time windows
st.header("Temporal Parameters")
date = st.selectbox("Day of the Month", list(range(1, 32)), index=9)  # Defaults to 10
day = st.selectbox(
    "Day of the Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
)
time_slot = st.text_input("Time Window String (e.g. 12:00:00 AM)", "12:00:00 AM")

# ==========================================
# 3. LIVE PREDICTION PIPELINE
# ==========================================
# ==========================================
# 3. LIVE PREDICTION PIPELINE
# ==========================================
if st.button("Generate Traffic Prediction"):
    total = car + bike + bus + truck
    heavy_vehicle_ratio = (bus + truck) / (total + 1e-5)

    # 1. Start with an empty DataFrame containing ONLY the expected columns from your training
    expected_columns = scaler.feature_names_in_
    input_ready = pd.DataFrame(0.0, index=[0], columns=expected_columns)

    # 2. Set the numerical features directly
    input_ready["CarCount"] = float(car)
    input_ready["BikeCount"] = float(bike)
    input_ready["BusCount"] = float(bus)
    input_ready["TruckCount"] = float(truck)
    input_ready["Heavy_Vehicle_Ratio"] = float(heavy_vehicle_ratio)

    # 3. Dynamically set the dummy columns to 1 if they match your expected training features
    date_col = f"Date_{date}"
    day_col = f"Day of the week_{day}"
    time_col = f"Time_{time_slot}"

    if date_col in input_ready.columns:
        input_ready[date_col] = 1.0
    if day_col in input_ready.columns:
        input_ready[day_col] = 1.0
    if time_col in input_ready.columns:
        input_ready[time_col] = 1.0
    else:
        # Visual alert to help you debug your exact training time format
        st.sidebar.warning(f"Note: '{time_slot}' format wasn't recognized by the model training columns.")

    # Scale the aligned structure
    input_scaled = scaler.transform(input_ready)

    # Run inference execution
    prediction = model.predict(input_scaled)[0]
    
    # Map to display string
    result_text = target_display.get(prediction, str(prediction))

    # Render results
    if prediction == 0 or "low" in str(result_text).lower():
        st.success(f"Prediction: **{result_text}** 🟢")
    elif prediction == 1 or "medium" in str(result_text).lower() or "normal" in str(result_text).lower():
        st.warning(f"Prediction: **{result_text}** 🟡")
    else:
        st.error(f"Prediction: **{result_text}** 🔴 (Raw score: {prediction})")
    else:
        st.info(f"Prediction: **{result_text}** 🔵")
