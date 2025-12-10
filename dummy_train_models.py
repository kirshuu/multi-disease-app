import numpy as np
import json
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Feature names must match app.py
HEART_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

DIABETES_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

PARKINSON_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ",
    "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR"
]


def make_dummy_heart():
    n_features = len(HEART_FEATURES)
    # random training data
    X = np.random.randn(500, n_features)
    y = np.random.randint(0, 2, size=500)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)

    joblib.dump(model, MODELS_DIR / "heart_model.pkl")

    meta = {
        "feature_names": HEART_FEATURES,
        "class_names": ["No Disease", "Disease"],
        "target_name": "target",
        "metric": "accuracy",
        "metric_value": 0.0  # dummy
    }
    with open(MODELS_DIR / "heart_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Created dummy heart_model.pkl and heart_metadata.json")


def make_dummy_diabetes():
    n_features = len(DIABETES_FEATURES)
    X = np.random.randn(500, n_features)
    y = np.random.randint(0, 2, size=500)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])
    pipeline.fit(X, y)

    joblib.dump(pipeline, MODELS_DIR / "diabetes_model.pkl")

    meta = {
        "feature_names": DIABETES_FEATURES,
        "class_names": ["No Diabetes", "Diabetes"],
        "target_name": "Outcome",
        "metric": "accuracy",
        "metric_value": 0.0
    }
    with open(MODELS_DIR / "diabetes_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Created dummy diabetes_model.pkl and diabetes_metadata.json")


def make_dummy_parkinson():
    n_features = len(PARKINSON_FEATURES)
    X = np.random.randn(500, n_features)
    y = np.random.randint(0, 2, size=500)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, C=1.0, gamma="scale"))
    ])
    pipeline.fit(X, y)

    joblib.dump(pipeline, MODELS_DIR / "parkinson_model.pkl")

    meta = {
        "feature_names": PARKINSON_FEATURES,
        "class_names": ["Healthy", "Parkinson"],
        "target_name": "status",
        "metric": "accuracy",
        "metric_value": 0.0
    }
    with open(MODELS_DIR / "parkinson_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Created dummy parkinson_model.pkl and parkinson_metadata.json")


def main():
    make_dummy_heart()
    make_dummy_diabetes()
    make_dummy_parkinson()
    print("All dummy models created.")


if __name__ == "__main__":
    main()
