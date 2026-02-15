import pickle

def load_pipeline(path="../models/artifacts/churn_pipeline.pkl"):
    """Load the trained pipeline."""
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline
