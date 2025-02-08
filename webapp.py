import numpy as np
import pickle
import streamlit as st

# Load the trained model and scaler
with open("trained_model2.sav", "rb") as file:
    loaded_data = pickle.load(file)

model = loaded_data["model"]
scaler = loaded_data["scaler"]

# Function for diabetes prediction
def diabetes_prediction(input_data):
    # Convert input data to numpy array and reshape it
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64).reshape(1, -1)

    # Standardize the input data
    input_data_scaled = scaler.transform(input_data_as_numpy_array)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    return "🩸 The person is diabetic" if prediction[0] == 1 else "✅ The person is not diabetic"


# Streamlit UI
st.markdown(
    """
    <style>
        .stApp {
            background:#1E1E2E;
            padding: 40px;
            border-radius: 15px;
            color:white;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: white;
            margin-bottom: 20px;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: white;
            margin-bottom: 30px;
        }
        div[data-testid="stSlider"] label, 
        div[data-testid="stSlider"] span, 
        div[data-testid="stMarkdownContainer"] {
            color: white !important;
            font-weight: bold;
            font-size: 20px;
        }
        div[data-testid="stSlider"] {
            margin-bottom: 20px;
        }
        .stAlert {
            color: white !important;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 12px;
            font-size: 16px;
            width: 100%;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>🔬 Diabetes Prediction Web App</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle'>Enter the details below to check diabetes risk:</h3>", unsafe_allow_html=True)

# UI with two columns
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.slider('Number of Pregnancies', 0, 20, 1)
    Glucose = st.slider('Glucose Level', 50, 200, 100)
    BloodPressure = st.slider('Blood Pressure value', 50, 150, 80)
    SkinThickness = st.slider('Skin Thickness value', 0, 100, 20)

with col2:
    Insulin = st.slider('Insulin Level', 0, 500, 100)
    BMI = st.slider('BMI value', 10.0, 50.0, 25.0)
    DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
    Age = st.slider('Age of the Person', 10, 100, 30)

# Prediction logic
if st.button('🩺 Get Diabetes Test Result'):
    input_values = [
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]
    diagnosis = diabetes_prediction(input_values)
    st.markdown(f"<div class='stAlert'>{diagnosis}</div>", unsafe_allow_html=True)
