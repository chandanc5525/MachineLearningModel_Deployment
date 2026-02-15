import streamlit as st
import pandas as pd
import pickle

# Load trained pipeline
with open("models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Telco Churn Prediction")

# Numeric features
numeric_cols = {
    "SeniorCitizen": (0, 1, 0),
    "tenure": (0, 72, 12),
    "MonthlyCharges": (20, 120, 70),
    "TotalCharges": (20, 8000, 200)
}

# Categorical features
categorical_cols = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
}

# Collect numeric inputs
numeric_input = {}
for col, (min_val, max_val, default) in numeric_cols.items():
    numeric_input[col] = st.slider(col, min_value=min_val, max_value=max_val, value=default)

# Collect categorical inputs
categorical_input = {}
for col, options in categorical_cols.items():
    categorical_input[col] = st.selectbox(col, options)

# Combine inputs into DataFrame
input_data = {**numeric_input, **categorical_input}
input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict"):
    pred = model.predict(input_df)
    st.subheader(f"Churn Prediction: {'Yes' if pred[0] == 1 else 'No'}")
