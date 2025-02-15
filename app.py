import streamlit as st
import numpy as np
import joblib

# Load models and scalers
bank_model = joblib.load("random_forest_bank_model.pkl")
telecom_model = joblib.load("random_forest_telecom_model.pkl")
bank_scaler = joblib.load("scaler_bank.pkl")
telecom_scaler = joblib.load("scaler_telecom.pkl")

# Function to predict churn
def predict_churn(model, scaler, features):
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    return "Churned" if prediction[0] == 1 else "Not Churned"

# Streamlit UI
st.title("Churn Prediction App")

# Model Selection
model_type = st.radio("Choose the type of Churn Prediction:", ["Bank Customer", "Telecom Customer"])

def validate_input(inputs):
    """Function to validate inputs for null values, wrong data types, or blank spaces"""
    for key, value in inputs.items():
        if value == '' or value is None:
            st.error(f"{key} cannot be empty or None!")
            return False
        if isinstance(value, str) and value.isspace():
            st.error(f"{key} cannot be blank space!")
            return False
    return True

if model_type == "Bank Customer":
    st.header("Bank Customer Churn Prediction")

    # Input fields
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100)
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10)
    balance = st.number_input("Balance")
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=4)
    has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.radio("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary")
    satisfaction_score = st.slider("Satisfaction Score", 1, 5)
    card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])
    points_earned = st.number_input("Points Earned", min_value=0)

    inputs = {
        "Credit Score": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "Number of Products": num_of_products,
        "Has Credit Card": has_cr_card,
        "Is Active Member": is_active_member,
        "Estimated Salary": estimated_salary,
        "Satisfaction Score": satisfaction_score,
        "Card Type": card_type,
        "Points Earned": points_earned
    }

    if validate_input(inputs):  # Validate inputs
        # One-hot Encoding
        geography_encoded = [1 if geography == "France" else 0, 1 if geography == "Germany" else 0, 1 if geography == "Spain" else 0]
        gender_encoded = [1 if gender == "Male" else 0, 1 if gender == "Female" else 0]
        card_type_encoded = [1 if card_type == "DIAMOND" else 0, 1 if card_type == "GOLD" else 0, 1 if card_type == "SILVER" else 0, 1 if card_type == "PLATINUM" else 0]

        # Create Feature Array
        features = np.array([credit_score, age, tenure, balance, num_of_products,
                             has_cr_card, is_active_member, estimated_salary,
                             satisfaction_score, points_earned] + geography_encoded + gender_encoded + card_type_encoded)

        if st.button("Predict"):
            result = predict_churn(bank_model, bank_scaler, features)
            st.success(f"Predicted Churn Status: {result}")

elif model_type == "Telecom Customer":
    st.header("Telecom Customer Churn Prediction")

    # Input fields
    tenure = st.number_input("Tenure", min_value=0, max_value=100)
    monthly_charges = st.number_input("Monthly Charges")
    total_charges = st.number_input("Total Charges")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless_billing = st.radio("Paperless Billing?", ["Yes", "No"])
    senior_citizen = st.radio("Senior Citizen?", ["Yes", "No"])
    streaming_tv = st.radio("Streaming TV?", ["Yes", "No")
    streaming_movies = st.radio("Streaming Movies?", ["Yes", "No"])
    multiple_lines = st.radio("Multiple Lines?", ["Yes", "No"])
    phone_service = st.radio("Phone Service?", ["Yes", "No"])
    device_protection = st.radio("Device Protection?", ["Yes", "No"])
    online_backup = st.radio("Online Backup?", ["Yes", "No"])
    partner = st.radio("Partner?", ["Yes", "No"])
    dependents = st.radio("Dependents?", ["Yes", "No"])
    tech_support = st.radio("Tech Support?", ["Yes", "No"])
    online_security = st.radio("Online Security?", ["Yes", "No"])
    gender = st.selectbox("Gender", ["Male", "Female"])

    inputs = {
        "Tenure": tenure,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
        "Contract": contract,
        "Internet Service": internet_service,
        "Payment Method": payment_method,
        "Paperless Billing": paperless_billing,
        "Senior Citizen": senior_citizen,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Multiple Lines": multiple_lines,
        "Phone Service": phone_service,
        "Device Protection": device_protection,
        "Online Backup": online_backup,
        "Partner": partner,
        "Dependents": dependents,
        "Tech Support": tech_support,
        "Online Security": online_security,
        "Gender": gender
    }

    if validate_input(inputs):  # Validate inputs
        # One-hot Encoding
        contract_encoded = [1 if contract == "Month-to-month" else 0, 1 if contract == "One year" else 0, 1 if contract == "Two year" else 0]
        internet_service_encoded = [1 if internet_service == "Fiber optic" else 0, 1 if internet_service == "DSL" else 0, 1 if internet_service == "No" else 0]
        payment_method_encoded = [1 if payment_method == "Electronic check" else 0, 1 if payment_method == "Mailed check" else 0, 1 if payment_method == "Bank transfer (automatic)" else 0, 1 if payment_method == "Credit card (automatic)" else 0]
        gender_encoded = [1 if gender == "Male" else 0, 1 if gender == "Female" else 0]

        # Feature Array
        features = np.array([paperless_billing, senior_citizen, streaming_tv, streaming_movies,
                             multiple_lines, phone_service, device_protection, online_backup,
                             partner, dependents, tech_support, online_security,
                             monthly_charges, total_charges, tenure] + contract_encoded + internet_service_encoded + payment_method_encoded + gender_encoded)

        if st.button("Predict"):
            result = predict_churn(telecom_model, telecom_scaler, features)
            st.success(f"Predicted Churn Status: {result}")
