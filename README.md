Multi-Disease Diagnosis App (Heart, Diabetes, Parkinson’s)
Deployed App:
👉 https://multi-disease-app-vvxpb6d77ecdkevquceung.streamlit.app/
________________________________________
🧠 Overview
This project implements a Multi-Disease Diagnosis System capable of predicting:
•	Heart Disease
•	Diabetes
•	Parkinson’s Disease
using pre-trained Machine Learning models (RF, KNN, SVM).
The app provides an easy-to-use web interface built with Streamlit, allowing users to input clinical data and receive:
✔ Prediction (High Risk / Low Risk)
✔ Explanation of prediction (rule-based thresholds, LIME-ready architecture)
✔ A clean deployment on Streamlit Cloud
This project is ideal for clinical research prototypes, academic demonstrations, and healthcare ML projects.
________________________________________
🚀 Live App
Access the deployed app here:
👉 https://multi-disease-app-vvxpb6d77ecdkevquceung.streamlit.app/
No installation required — works in any browser.
________________________________________
🧩 Features
1. Multi-Disease Prediction
Select one of:
•	Heart Disease
•	Diabetes
•	Parkinson’s Disease
Each disease dynamically loads its own input form based on its unique clinical features.
2. Pre-trained ML Models
Models used:
Disease	Model Used	Notes
Heart Disease	Random Forest	Trained on structured tabular features
Diabetes	KNN with scaling	Pipeline: StandardScaler → KNN
Parkinson’s Disease	SVM (RBF)	Probability outputs enabled
3. Explainability (XAI)
•	Explanation panel after prediction
•	Shows feature effects based on logical clinical thresholds
•	Architecture prepared for LIME/SHAP integration in future deployment
•	Metadata JSON files included for transparency
4. Streamlit Web App
•	Fully cloud-hosted
•	Clean, responsive UI
•	No setup required on client side
________________________________________
📂 Project Structure
multi-disease-app/
│
├── app.py                      # Main Streamlit app
├── dummy_train_models.py       # Script to generate demo models
├── requirements.txt            # Dependencies for Streamlit Cloud
│
├── models/                     # Pre-trained ML models + metadata
│   ├── heart_model.pkl
│   ├── heart_metadata.json
│   ├── diabetes_model.pkl
│   ├── diabetes_metadata.json
│   ├── parkinson_model.pkl
│   ├── parkinson_metadata.json
│
└── README.md                   # Project documentation
________________________________________
🔧 Installation (Local Setup)
If you want to run the project locally:
1. Clone the repo
git clone https://github.com/kirshuu/multi-disease-app.git
cd multi-disease-app
2. Install dependencies
pip install -r requirements.txt
3. Run app
streamlit run app.py
________________________________________
📊 Machine Learning Models
The ML models expect clinically meaningful numerical inputs.
Heart Disease — Features:
•	age
•	sex
•	cp
•	trestbps
•	chol
•	fbs
•	restecg
•	thalach
•	exang
•	oldpeak
•	slope
•	ca
•	thal
Diabetes — Features:
•	Pregnancies
•	Glucose
•	BloodPressure
•	SkinThickness
•	Insulin
•	BMI
•	DiabetesPedigreeFunction
•	Age
Parkinson’s — Features:
•	MDVP:Fo(Hz)
•	MDVP:Fhi(Hz)
•	MDVP:Flo(Hz)
•	MDVP:Jitter(%)
•	… (and other key acoustic biomarkers)
________________________________________
🧪 Dummy Models for Demo
For deployment without real datasets, the file dummy_train_models.py generates synthetic demo models.
These models allow the app to:
•	run online
•	demonstrate UI + workflows
•	show XAI outputs
They are NOT medically accurate and are for academic/demo purposes only.
________________________________________
🛠 Future Enhancements
•	Integrate true LIME/SHAP explainability
•	Replace dummy models with real trained models
•	Deploy with secure authentication for clinicians
•	Integration with hospital EHR systems
•	Add support for additional diseases
________________________________________
📝 Author
Shubham Rajput
GitHub: https://github.com/kirshuu

