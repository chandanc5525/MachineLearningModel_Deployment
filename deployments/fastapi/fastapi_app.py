from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pickle
import pandas as pd


# Path Configuration

BASE_DIR = Path(__file__).resolve().parent          # deployments/fastapi
PROJECT_ROOT = BASE_DIR.parent.parent              # MachineLearningModel_Deployment


# App Initialization

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")


# Load Model

with open(PROJECT_ROOT / "models" / "churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature Metadata (Same as Flask)

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

# GET Route

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "prediction": None
        }
    )

# POST Route

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request):

    form = await request.form()
    data = {}

    # Numeric fields
    for col in numeric_cols:
        data[col] = float(form[col])

    # Categorical fields
    for col in categorical_cols:
        data[col] = form[col]

    df = pd.DataFrame([data])
    pred = model.predict(df)

    prediction = "Yes" if pred[0] == 1 else "No"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "prediction": prediction
        }
    )
