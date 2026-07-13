import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# ... [Keep your layout and EDA code the same] ...

elif page == "🔮 Traffic Prediction":
    st.header("Predict Traffic Situation")
    
    # User Inputs
    col1, col2 = st.columns(2)
    with col1:
        car_count = st.number_input("Car Count", min_value=0, max_value=500, value=20)
        bike_count = st.number_input("Bike Count", min_value=0, max_value=200, value=5)
        day_of_week = st.selectbox("Day of the week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    with col2:
        bus_count = st.number_input("Bus Count", min_value=0, max_value=100, value=5)
        truck_count = st.number_input("Truck Count", min_value=0, max_value=200, value=15)
        
    total_vehicles = car_count + bike_count + bus_count + truck_count
    st.info(f"💡 Calculated Total Vehicles: **{total_vehicles}**")

    # Load the corrected Pickled files
    @st.cache_resource
    def load_prediction_assets():
        with open('logistic_modell.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler

    if st.button("🚀 Predict Traffic Situation"):
        try:
            model, scaler = load_prediction_assets()
            
            # Simple manual Label encoding matching the training day order alpha-sorted
            days_mapping = {'Friday': 0, 'Monday': 1, 'Saturday': 2, 'Sunday': 3, 'Thursday': 4, 'Tuesday': 5, 'Wednesday': 6}
            day_encoded = days_mapping[day_of_week]
            
            # 1. Shape the raw input features
            raw_features = np.array([[car_count, bike_count, bus_count, truck_count, total_vehicles, day_encoded]])
            
            # 2. Scale features exactly how the logistic regression expects them
            scaled_features = scaler.transform(raw_features)
            
            # 3. Predict
            prediction = model.predict(scaled_features)[0]
            
            # Display Result
            if prediction.lower() == 'high':
                st.error(f"🚨 Predicted Situation: **{prediction.upper()} TRAFFIC**")
            elif prediction.lower() == 'medium':
                st.warning(f"⚠️ Predicted Situation: **{prediction.upper()} TRAFFIC**")
            else:
                st.success(f"✅ Predicted Situation: **{prediction.upper()} TRAFFIC**")
                
        except Exception as e:
            st.error(f"Error executing prediction: {e}")
