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
if st.button("Generate Traffic Prediction"):
    # Apply your engineered feature calculation on the raw user inputs
    total = car + bike + bus + truck
    heavy_vehicle_ratio = (bus + truck) / (total + 1e-5)

    # Reconstruct the single row DataFrame matching original data structure
    input_data = pd.DataFrame(
        [
            {
                "Date": date,
                "Day of the week": day,
                "Time": time_slot,
                "CarCount": car,
                "BikeCount": bike,
                "BusCount": bus,
                "TruckCount": truck,
                "Heavy_Vehicle_Ratio": heavy_vehicle_ratio,
            }
        ]
    )

    # Run get_dummies on the categoricals
    cat_cols = ["Date", "Day of the week", "Time"]
    input_encoded = pd.get_dummies(input_data, columns=cat_cols)

    # Use the scaler metadata to perfectly align dummy variables matching your training columns
    expected_columns = scaler.feature_names_in_
    input_ready = input_encoded.reindex(columns=expected_columns, fill_value=0)

    # Scale the aligned array structure
    input_scaled = scaler.transform(input_ready.astype(float))

    # Run inference execution on the dynamically selected model
    # Run inference execution on the dynamically selected model
    prediction_raw = model.predict(input_scaled)[0]
    
    # Automatically convert continuous decimals (like 1.97) to the nearest category integer (like 2)
    try:
        prediction = int(round(float(prediction_raw)))
    except (ValueError, TypeError):
        prediction = prediction_raw

    # Target text mappings for output display
    target_display = {0: "Low Traffic", 1: "Medium Traffic", 2: "High Traffic"}
    result_text = target_display.get(prediction, f"Custom Value ({prediction_raw:.2f})")

    # Render clean results with the right color indicators
    if prediction == 0:
        st.success(f"Prediction: **{result_text}** 🟢 (Raw score: {prediction_raw:.2f})")
    elif prediction == 1:
        st.warning(f"Prediction: **{result_text}** 🟡 (Raw score: {prediction_raw:.2f})")
    elif prediction == 2:
        st.error(f"Prediction: **{result_text}** 🔴 (Raw score: {prediction_raw:.2f})")
    else:
        st.info(f"Prediction: **{result_text}** 🔵")
