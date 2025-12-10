import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# ============== CONFIG ==============

MODELS_DIR = Path("models")

# feature lists – must match the features you used to train the models
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


# ============== HELPERS ==============

@st.cache_resource
def load_model(model_name: str):
    """
    Load model and metadata.
    Expects:
        models/<name>_model.pkl
        models/<name>_metadata.json
    """
    model_path = MODELS_DIR / f"{model_name}_model.pkl"
    meta_path = MODELS_DIR / f"{model_name}_metadata.json"

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()
    if not meta_path.exists():
        st.error(f"Metadata file not found: {meta_path}")
        st.stop()

    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return model, meta


def simple_rule_based_explanation(disease, inputs):
    """
    Very simple rule-based explanation, not mathematically exact like LIME,
    but enough to show clinicians which values look risk-raising.
    """
    explanations = []

    if disease == "Heart Disease":
        if inputs["age"] >= 55:
            explanations.append("Higher age increases heart disease risk.")
        if inputs["chol"] >= 240:
            explanations.append("High cholesterol is associated with heart disease.")
        if inputs["thalach"] <= 120:
            explanations.append("Low maximum heart rate may indicate poorer cardiac fitness.")
        if inputs["exang"] == 1:
            explanations.append("Exercise-induced angina is a strong risk indicator.")
        if inputs["oldpeak"] >= 2:
            explanations.append("ST depression (oldpeak) above 2 is concerning.")
    elif disease == "Diabetes":
        if inputs["Glucose"] >= 140:
            explanations.append("High glucose is a primary indicator for diabetes.")
        if inputs["BMI"] >= 30:
            explanations.append("Obesity (BMI ≥ 30) is a strong risk factor.")
        if inputs["Age"] >= 45:
            explanations.append("Higher age is associated with increased diabetes risk.")
        if inputs["Insulin"] >= 200:
            explanations.append("Elevated insulin suggests impaired glucose regulation.")
    elif disease == "Parkinson's Disease":
        if inputs["MDVP:Jitter(%)"] > 0.01:
            explanations.append("Increased jitter indicates instability in voice frequency.")
        if inputs["MDVP:Shimmer"] > 0.04:
            explanations.append("Higher shimmer indicates amplitude variations in speech.")
        if inputs["NHR"] > 0.05:
            explanations.append("High noise-to-harmonics ratio is associated with dysphonia.")
        if inputs["HNR"] < 20:
            explanations.append("Low harmonics-to-noise ratio may indicate voice impairment.")

    if not explanations:
        explanations.append(
            "No major risk thresholds crossed; model may be using subtler patterns."
        )

    return explanations


# ============== UI: INPUT FORMS ==============

def collect_heart_inputs():
    st.subheader("Heart Disease – Patient Data Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Sex (0=female, 1=male)", [0, 1])
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting BP (trestbps)", 80, 250, 120)

    with col2:
        chol = st.number_input("Cholesterol (chol)", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate (thalach)", 60, 250, 150)

    with col3:
        exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
        oldpeak = st.number_input("ST Depression (oldpeak)", -2.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of ST segment", [0, 1, 2])
        ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    inputs = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    return inputs


def collect_diabetes_inputs():
    st.subheader("Diabetes – Patient Data Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        Glucose = st.number_input("Glucose", 0, 300, 120)
        BloodPressure = st.number_input("Blood Pressure", 0, 200, 70)

    with col2:
        SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
        Insulin = st.number_input("Insulin", 0, 900, 80)
        BMI = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)

    with col3:
        DiabetesPedigreeFunction = st.number_input(
            "Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01
        )
        Age = st.number_input("Age", 1, 120, 35)

    inputs = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }
    return inputs


def collect_parkinson_inputs():
    st.subheader("Parkinson's Disease – Patient Data Input (Voice Features)")

    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", 50.0, 300.0, 150.0)
        fhi = st.number_input("MDVP:Fhi(Hz)", 50.0, 600.0, 200.0)
        flo = st.number_input("MDVP:Flo(Hz)", 50.0, 300.0, 100.0)
        jitter_percent = st.number_input("MDVP:Jitter(%)", 0.0, 1.0, 0.01, step=0.001)

    with col2:
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", 0.0, 0.01, 0.0001, step=0.0001)
        rap = st.number_input("MDVP:RAP", 0.0, 1.0, 0.01, step=0.001)
        ppq = st.number_input("MDVP:PPQ", 0.0, 1.0, 0.01, step=0.001)
        jitter_ddp = st.number_input("Jitter:DDP", 0.0, 1.0, 0.03, step=0.001)

    with col3:
        shimmer = st.number_input("MDVP:Shimmer", 0.0, 1.0, 0.03, step=0.001)
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", 0.0, 5.0, 0.3, step=0.01)
        apq3 = st.number_input("Shimmer:APQ3", 0.0, 1.0, 0.02, step=0.001)
        apq5 = st.number_input("Shimmer:APQ5", 0.0, 1.0, 0.02, step=0.001)
        apq = st.number_input("MDVP:APQ", 0.0, 1.0, 0.03, step=0.001)
        dda = st.number_input("Shimmer:DDA", 0.0, 1.0, 0.06, step=0.001)
        nhr = st.number_input("NHR", 0.0, 1.0, 0.02, step=0.001)
        hnr = st.number_input("HNR", 0.0, 50.0, 20.0, step=0.1)

    inputs = {
        "MDVP:Fo(Hz)": fo,
        "MDVP:Fhi(Hz)": fhi,
        "MDVP:Flo(Hz)": flo,
        "MDVP:Jitter(%)": jitter_percent,
        "MDVP:Jitter(Abs)": jitter_abs,
        "MDVP:RAP": rap,
        "MDVP:PPQ": ppq,
        "Jitter:DDP": jitter_ddp,
        "MDVP:Shimmer": shimmer,
        "MDVP:Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": apq3,
        "Shimmer:APQ5": apq5,
        "MDVP:APQ": apq,
        "Shimmer:DDA": dda,
        "NHR": nhr,
        "HNR": hnr
    }
    return inputs


# ============== STREAMLIT LAYOUT ==============

st.set_page_config(page_title="Multi-Disease Diagnosis App", layout="wide")

st.title("Multi-Disease Diagnosis App")
st.markdown(
    """
This app performs **multi-disease diagnosis** (Heart Disease, Diabetes, and Parkinson's)  
using ML models (RF, KNN, SVM – trained offline) and shows **simple explanations**
based on clinically relevant thresholds.
"""
)

disease = st.sidebar.selectbox(
    "Select Disease",
    ("Heart Disease", "Diabetes", "Parkinson's Disease")
)

st.sidebar.markdown("---")
st.sidebar.subheader("Future Scope")
st.sidebar.write(
    "- Integrate true LIME-based explanations once environment issues are resolved.\n"
    "- Connect with Hospital EHR systems.\n"
    "- Deploy securely on cloud with authentication for clinicians."
)

if disease == "Heart Disease":
    inputs = collect_heart_inputs()
    feature_list = HEART_FEATURES
    model_key = "heart"
elif disease == "Diabetes":
    inputs = collect_diabetes_inputs()
    feature_list = DIABETES_FEATURES
    model_key = "diabetes"
else:
    inputs = collect_parkinson_inputs()
    feature_list = PARKINSON_FEATURES
    model_key = "parkinson"

col_pred, col_exp = st.columns([1, 1])

with col_pred:
    st.subheader("Prediction")

    if st.button("Predict Diagnosis"):
        model, meta = load_model(model_key)

        # construct dataframe in correct feature order
        x_df = pd.DataFrame([[inputs[f] for f in feature_list]], columns=feature_list)

        y_pred = model.predict(x_df)[0]

        # probability if available
        try:
            proba = model.predict_proba(x_df)[0]
            prob = proba[int(y_pred)]
        except Exception:
            prob = None

        class_names = meta.get("class_names", ["Class 0", "Class 1"])
        label_text = class_names[int(y_pred)] if int(y_pred) < len(class_names) else str(y_pred)

        if "no" in label_text.lower():
            color = "green"
        else:
            color = "red"

        st.markdown(
            f"**Diagnosis Result:** "
            f"<span style='color:{color}; font-size:24px'>{label_text}</span>",
            unsafe_allow_html=True
        )

        if prob is not None:
            st.write(f"Model confidence for this class: **{prob*100:.2f}%**")

        st.session_state["last_inputs"] = (disease, inputs)

with col_exp:
    st.subheader("Explain Prediction")

    if st.button("Explain Last Prediction"):
        if "last_inputs" not in st.session_state:
            st.warning("Please run a prediction first.")
        else:
            d_name, last_inputs = st.session_state["last_inputs"]
            st.write("This explanation uses *rule-based clinical thresholds* "
                     "to highlight which entered values are clinically risky.")
            rules = simple_rule_based_explanation(d_name, last_inputs)
            for i, r in enumerate(rules, start=1):
                st.markdown(f"{i}. {r}")

            st.markdown("#### Entered Values")
            st.json(last_inputs)
