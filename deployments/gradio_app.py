import pickle
import pandas as pd
import gradio as gr

# Load trained pipeline
with open("models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define feature columns
numeric_cols = {
    "SeniorCitizen": (0, 1, 0),
    "tenure": (0, 72, 12),
    "MonthlyCharges": (20, 120, 70),
    "TotalCharges": (20, 8000, 200)
}

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

# Build Gradio inputs
inputs = []

# Numeric sliders
for col, (min_val, max_val, default) in numeric_cols.items():
    inputs.append(gr.Slider(minimum=min_val, maximum=max_val, value=default, label=col))

# Categorical dropdowns
for col, options in categorical_cols.items():
    inputs.append(gr.Dropdown(choices=options, value=options[0], label=col))

# Prediction function
def predict_churn(*args):
    # Create dataframe from inputs
    all_cols = list(numeric_cols.keys()) + list(categorical_cols.keys())
    df = pd.DataFrame([args], columns=all_cols)
    pred = model.predict(df)
    return "Yes" if pred[0] == 1 else "No"

# Gradio Interface
iface = gr.Interface(
    fn=predict_churn,
    inputs=inputs,
    outputs="text",
    title="Telco Churn Prediction",
    description="Enter numeric values using sliders and select options from dropdowns."
)

iface.launch()
