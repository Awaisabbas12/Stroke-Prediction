import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load saved model
model = joblib.load(r'final_pipeline.pkl')

# Title
st.title("🧠 Stroke Prediction App")
st.write("""
This app predicts the likelihood of a stroke based on user input.
It uses a Random Forest model trained on health and demographic data, with SMOTE to handle class imbalance.
""")

# Sidebar for input
st.sidebar.header("Input Features")

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    age = st.sidebar.slider('Age', 0, 100, 30)
    hypertension = st.sidebar.selectbox('Hypertension (0 = No, 1 = Yes)', [0, 1])
    heart_disease = st.sidebar.selectbox('Heart Disease (0 = No, 1 = Yes)', [0, 1])
    ever_married = st.sidebar.selectbox('Ever Married', ['Yes', 'No'])
    work_type = st.sidebar.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    Residence_type = st.sidebar.selectbox('Residence Type', ['Urban', 'Rural'])
    avg_glucose_level = st.sidebar.slider('Average Glucose Level', 50.0, 300.0, 100.0)
    bmi = st.sidebar.slider('BMI', 10.0, 60.0, 25.0)
    smoking_status = st.sidebar.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

    data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }

    return pd.DataFrame([data])

threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
# Collect input
input_df = user_input_features()


# Show input
st.subheader("User Input:")
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Show results
st.subheader("Prediction Result")
stroke_label = ['No Stroke', 'Stroke']
predicted_label = stroke_label[prediction[0]]
predicted_proba = prediction_proba[0]

# Display predicted class and its probability
st.write(f"**Prediction:** {stroke_label[prediction[0]]}")
st.write(f"**Probability:** {prediction_proba[0][prediction[0]]:.2f}")
# Show full probability distribution
st.write("### Probability for each class:")
prob_df = pd.DataFrame({
    "Class": stroke_label,
    "Probability": predicted_proba
})
st.dataframe(prob_df)

# Optional: add a simple bar chart
st.write("### Probability Chart:")
st.bar_chart(prob_df.set_index("Class"))
