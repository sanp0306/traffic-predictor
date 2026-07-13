import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Set Page Configuration
st.set_page_config(
    page_title="Traffic Situation Predictor & Analyzer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚗 Traffic Situation Predictor & Dashboard")
st.write("Analyze traffic distributions and predict real-time congestion levels using our corrected Logistic Regression framework.")

# ----------------------------------------------------
# Data Loading Utility
# ----------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists('Dataset_Traffic.csv'):
        return pd.read_csv('Dataset_Traffic.csv')
    else:
        st.error("⚠️ Dataset_Traffic.csv not detected in the current directory.")
        return None

df = load_data()

if df is not None:
    # Sidebar Setup
    st.sidebar.header("Navigation Panel")
    page = st.sidebar.radio("Navigate to:", ["📊 Dataset & Analytics", "🔮 Traffic Prediction"])

    # ----------------------------------------------------
    # Page 1: Dataset & Analytics Dashboard
    # ----------------------------------------------------
    if page == "📊 Dataset & Analytics":
        st.header("Exploratory Data Analysis Dashboard")
        
        # Upper KPI Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records Analyzed", f"{df.shape[0]:,}")
        col2.metric("Mean Car Volume", int(df['CarCount'].mean()))
        col3.metric("Mean Heavy Truck Volume", int(df['TruckCount'].mean()))
        col4.metric("Monitored Weekdays", df['Day of the week'].nunique())
        
        st.subheader("📋 Dataset Preview (Top 10 Batches)")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Graphical Distribution Layout
        st.subheader("📈 Statistical Visualizations")
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            st.markdown("**Traffic Congestion Category Counts**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x='Traffic Situation', palette='viridis', ax=ax)
            ax.set_ylabel("Data Point Frequency")
            st.pyplot(fig)
            
        with fig_col2:
            st.markdown("**Cumulative Vehicle Counts Averaged Across Weekdays**")
            fig, ax = plt.subplots(figsize=(6, 4))
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sns.barplot(data=df, x='Day of the week', y='Total', order=day_order, errorbar=None, palette='magma', ax=ax)
            plt.xticks(rotation=45)
            ax.set_ylabel("Average Volume")
            st.pyplot(fig)

    # ----------------------------------------------------
    # Page 2: Prediction Pipeline
    # ----------------------------------------------------
    elif page == "🔮 Traffic Prediction":
        st.header("Logistic Regression Prediction Interface")
        st.markdown("Supply localized vehicle counts to evaluate the targeted traffic outcome.")
        
        col1, col2 = st.columns(2)
        with col1:
            car_count = st.number_input("Car Count", min_value=0, max_value=1000, value=30, step=1)
            bike_count = st.number_input("Bike/Two-Wheeler Count", min_value=0, max_value=500, value=10, step=1)
            day_of_week = st.selectbox("Day of the Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        with col2:
            bus_count = st.number_input("Bus Count", min_value=0, max_value=300, value=5, step=1)
            truck_count = st.number_input("Truck/Heavy Vehicle Count", min_value=0, max_value=500, value=15, step=1)
            
        total_vehicles = car_count + bike_count + bus_count + truck_count
        st.info(f"💡 Calculated Cumulative Feature Volume (`Total`): **{total_vehicles}**")

        @st.cache_resource
        def load_prediction_assets():
            try:
                with open('logistic_modell.pkl', 'rb') as f:
                    loaded_model = pickle.load(f)
                with open('scaler.pkl', 'rb') as f:
                    loaded_scaler = pickle.load(f)
                return loaded_model, loaded_scaler
            except Exception as e:
                st.error(f"Failed to access backend PKL assets: {e}")
                return None, None

        if st.button("🚀 Execute Classification Engine"):
            model, scaler = load_prediction_assets()
            
            if model is not None and scaler is not None:
                try:
                    days_mapping = {
                        'Friday': 0, 'Monday': 1, 'Saturday': 2, 
                        'Sunday': 3, 'Thursday': 4, 'Tuesday': 5, 'Wednesday': 6
                    }
                    day_encoded = days_mapping[day_of_week]
                    
                    raw_features = np.array([[car_count, bike_count, bus_count, truck_count, total_vehicles, day_encoded]])
                    scaled_features = scaler.transform(raw_features)
                    prediction = model.predict(scaled_features)[0]
                    
                    st.subheader("🎯 Classification Outcome")
                    if prediction.lower() == 'high':
                        st.error(f"🚨 Classification Target: **{prediction.upper()} TRAFFIC CONGESTION**")
                    elif prediction.lower() == 'medium':
                        st.warning(f"⚠️ Classification Target: **{prediction.upper()} TRAFFIC VOLUME**")
                    else:
                        st.success(f"✅ Classification Target: **{prediction.upper()} TRAFFIC CLEAR**")
                        
                except Exception as eval_err:
                    st.error(f"An error occurred during calculation processing: {eval_err}")
