# ============================================================
# Logistic Regression - Diabetes Prediction
# Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)


# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

st.title("🩺 Diabetes Prediction using Logistic Regression")
st.write(
    "Enter the patient's information below to predict the diabetes outcome."
)


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

@st.cache_data
def load_data():
    # Load the diabetes dataset
    return pd.read_csv("diabetes.csv")


df = load_data()


# ------------------------------------------------------------
# Prepare Data
# ------------------------------------------------------------

# Separate input features and target variable
X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# Feature Scaling
# ------------------------------------------------------------

# StandardScaler puts all numerical features on a similar scale
scaler = StandardScaler()

# Fit scaler only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler to transform test data
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# Train Logistic Regression Model
# ------------------------------------------------------------

model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_scaled, y_train)


# ------------------------------------------------------------
# Model Evaluation
# ------------------------------------------------------------

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)


# ------------------------------------------------------------
# Display Model Performance
# ------------------------------------------------------------

st.subheader("📊 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Precision", f"{precision:.2%}")
col3.metric("Recall", f"{recall:.2%}")
col4.metric("F1 Score", f"{f1:.2%}")
col5.metric("ROC-AUC", f"{roc_auc:.2%}")


# ------------------------------------------------------------
# User Input Section
# ------------------------------------------------------------

st.subheader("👤 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )


with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=1000,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

if st.button("🔍 Predict Diabetes", use_container_width=True):

    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    # Scale the input using the already fitted scaler
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Get probability of diabetes
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Prediction: Diabetes")
    else:
        st.success("✅ Prediction: No Diabetes")

    st.write(
        f"Probability of diabetes: **{probability:.2%}**"
    )


# ------------------------------------------------------------
# Dataset Preview
# ------------------------------------------------------------

with st.expander("📋 View Dataset"):

    st.write("Dataset Shape:", df.shape)

    st.dataframe(df.head(10))


# ------------------------------------------------------------
# Important Note
# ------------------------------------------------------------

st.info(
    "This application is for educational and demonstration purposes "
    "and should not be used as a medical diagnosis."
)