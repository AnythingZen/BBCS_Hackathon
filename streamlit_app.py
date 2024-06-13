import streamlit as st
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
import pandas as pd
import subprocess

try:
    subprocess.call(['pip', 'install', '-r', 'requirements.txt'])
except Exception as e:
    st.error(f"Error installing dependencies: {e}")


# Load the saved model
svc_model_loaded = load('SVC_model.joblib')

# Create a Streamlit app title
st.title("Employee Attrition Prediction App")

# Create a form to input data
with st.form("employee_data"):
    name = st.text_input("Name")
    age = st.number_input("Age")
    daily_rate = st.number_input("Daily Rate")
    environment_satisfaction = st.selectbox("Environment Satisfaction", [0, 1, 2, 3])
    hourly_rate = st.number_input("Hourly Rate")
    job_satisfaction = st.selectbox("Job Satisfaction", [0, 1, 2, 3])
    monthly_income = st.number_input("Monthly Income")
    monthly_rate = st.number_input("Monthly Rate")
    num_companies_worked = st.number_input("Number of Companies Worked")
    performance_rating = st.selectbox("Performance Rating", ["Average", "High"])
    relationship_satisfaction = st.selectbox("Relationship Satisfaction", [0, 1, 2, 3])
    total_working_years = st.number_input("Total Working Years")
    work_life_balance = st.selectbox("Work Life Balance", [0, 1, 2, 3])
    years_at_company = st.number_input("Years at Company")
    years_in_current_role = st.number_input("Years in Current Role")
    years_since_last_promotion = st.number_input("Years since Last Promotion")
    
    # Convert performance rating to 0 or 1
    if performance_rating == "Average":
        performance_rating = 0
    else:
        performance_rating = 1
    
    # Create a list to store the input data
    input_datas = pd.DataFrame([[age, daily_rate, hourly_rate, monthly_income, monthly_rate,
                    num_companies_worked, total_working_years, years_at_company,
                    years_in_current_role, years_since_last_promotion,
                    environment_satisfaction, job_satisfaction,
                    performance_rating, relationship_satisfaction,
                    work_life_balance]])
    
    # Create a button to submit the form
    submitted = st.form_submit_button("Predict")

    # Process the input data and make predictions
    if submitted:
        scaler = StandardScaler()
        X_test = scaler.fit_transform(input_datas)

        prediction = svc_model_loaded.predict(X_test)
        
        # Display the prediction outcome
        if prediction == 0:
            prediction = 'Stay'
        else:
            prediction = 'Leave'
        st.write(f'{name} is predicted to {prediction}')


