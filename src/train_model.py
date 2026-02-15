import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Paths
data_path = os.path.join("data", "churn.csv")  
model_dir = os.path.join("models")            
os.makedirs(model_dir, exist_ok=True)
pickle_path = os.path.join(model_dir, "churn_model.pkl")

# Load dataset
df = pd.read_csv(data_path)

# Convert TotalCharges to numeric if exists
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Encode target
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Features and target
X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn']

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Build preprocessing + model pipeline
preprocessor = ColumnTransformer([
    ('num', 'passthrough', numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Save pickle
with open(pickle_path, "wb") as f:
    pickle.dump(pipeline, f)

print("Pickle file saved at", pickle_path)
