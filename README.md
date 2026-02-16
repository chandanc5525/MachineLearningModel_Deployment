# 📊 Customer Churn Prediction App

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Flask%20%7C%20Streamlit%20%7C%20Gradio-green)
![ML](https://img.shields.io/badge/Model-RandomForest-orange)
![Package Manager](https://img.shields.io/badge/Package%20Manager-uv-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)


End-to-end Machine Learning project for predicting customer churn.

Includes:
- Data preprocessing

- Model training (Random Forest)

- Pickle model generation

- Deployment using FastAPI, Flask, Streamlit, Gradio

- HTML + CSS frontend

------------------------------------------------------------
🚀 SETUP INSTRUCTIONS (STEP-BY-STEP)
------------------------------------------------------------

### 1️⃣ Fork the repository (from GitHub UI)

### 2️⃣ Clone your fork

git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git

cd YOUR-REPO-NAME

------------------------------------------------------------
### 3️⃣ Install uv (if not installed)
------------------------------------------------------------

#### Mac / Linux

curl -Ls https://astral.sh/uv/install.sh | sh

#### Windows (PowerShell)

irm https://astral.sh/uv/install.ps1 | iex

#### Verify installation

uv --version

------------------------------------------------------------
### 4️⃣ Create Virtual Environment
------------------------------------------------------------

uv venv

#### Activate virtual environment

#### Windows

.venv\Scripts\activate

#### Mac / Linux

source .venv/bin/activate

------------------------------------------------------------
### 5️⃣ Install Project Dependencies
------------------------------------------------------------

uv sync

------------------------------------------------------------
### 6️⃣ Train Model (Generate Pickle File)
------------------------------------------------------------

#### Make sure dataset exists:

#### data/churn.csv

python train_model.py

#### This creates:

#### models/churn_model.pkl

------------------------------------------------------------
🌐 RUN APPLICATIONS
------------------------------------------------------------

------------------------------------------------------------
### ▶️ Run Flask App
------------------------------------------------------------

python deployments/flask/flask_app.py

#### Open in browser:

#### http://127.0.0.1:5000/

------------------------------------------------------------
### ▶️ Run FastAPI App
------------------------------------------------------------

uvicorn deployments.fastapi.fastapi_app:app --reload

####  Open:

#### http://127.0.0.1:8000/

#### Swagger Docs:

####  http://127.0.0.1:8000/docs

------------------------------------------------------------
### ▶️ Run Streamlit App
------------------------------------------------------------

streamlit run deployments/streamlit_app.py

------------------------------------------------------------
### ▶️ Run Gradio App
------------------------------------------------------------

python deployments/gradio_app.py

------------------------------------------------------------
📂 PROJECT STRUCTURE
------------------------------------------------------------
```
.
├── data/
│   └── churn.csv
├── models/
│   └── churn_model.pkl
├── deployments/
│   ├── flask/
│   ├── fastapi/
│   ├── streamlit_app.py
│   └── gradio_app.py
├── train_model.py
├── pyproject.toml
└── README.md

```
------------------------------------------------------------
📌 IMPORTANT NOTES
------------------------------------------------------------

- Train the model before running any app.

- Entire preprocessing + model is saved inside pickle.

- No manual encoding required in deployment.

- SMOTE (if used) is only for training, never for production API.

------------------------------------------------------------
🎯 DONE
------------------------------------------------------------

You now have a fully working ML project with multiple deployment options.
