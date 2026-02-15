from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature metadata
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

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        data = {}
        for col in numeric_cols:
            data[col] = float(request.form[col])
        for col in categorical_cols:
            data[col] = request.form[col]

        df = pd.DataFrame([data])
        pred = model.predict(df)
        prediction = "Yes" if pred[0] == 1 else "No"

    return render_template("index.html",
                           numeric_cols=numeric_cols,
                           categorical_cols=categorical_cols,
                           prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
