import warnings
import numpy as np
import joblib

# Suppress Specific Sklearn Warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load your scaler and model
scaler_loaded = joblib.load("scaler.pkl")
rf_model_loaded = joblib.load("rf_model.pkl")

# Defining Feature Names (This is your predefined feature list)
feature_names = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Complain',
                 'SatisfactionScore', 'PointsEarned', 'France', 'Germany', 'Spain',
                 'Male', 'Female', 'DIAMOND', 'GOLD', 'SILVER', 'PLATINUM']

# Function to Get User Input with Validation
def get_user_input():
    print("Enter Customer Details for Churn Prediction:\n")

    # Helper function for numeric input with validation
    def get_numeric_input(prompt, input_type, lower=None, upper=None):
        while True:
            try:
                value = input_type(input(f"{prompt}: "))
                if (lower is not None and value < lower) or (upper is not None and value > upper):
                    print(f"Please enter a value between {lower} and {upper}.")
                    continue
                return value
            except ValueError:
                print("Invalid input! Please enter a valid number.")

    # Getting User Inputs for Numerical Features
    tenure = get_numeric_input("Tenure", int, 0)
    monthly_charges = get_numeric_input("Monthly Charges", float, 0)
    total_charges = get_numeric_input("Total Charges", float, 0)

    # Getting User Input for Categorical Features with Validation
    def get_categorical_input(prompt, valid_options):
        while True:
            value = input(f"{prompt} ({'/'.join(valid_options)}): ")
            if value in valid_options:
                return value
            else:
                print(f"Invalid input! Please choose from: {', '.join(valid_options)}")

    contract = get_categorical_input("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = get_categorical_input("Internet Service", ["Fiber optic", "DSL", "No"])
    payment_method = get_categorical_input("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    gender = get_categorical_input("Gender", ["Male", "Female"])

    # Getting Binary Inputs (Yes/No) for Other Features
    paperless_billing = get_numeric_input("Paperless Billing (1 for Yes, 0 for No)", int, 0, 1)
    senior_citizen = get_numeric_input("Senior Citizen (1 for Yes, 0 for No)", int, 0, 1)
    streaming_tv = get_numeric_input("Streaming TV (1 for Yes, 0 for No)", int, 0, 1)
    streaming_movies = get_numeric_input("Streaming Movies (1 for Yes, 0 for No)", int, 0, 1)
    multiple_lines = get_numeric_input("Multiple Lines (1 for Yes, 0 for No)", int, 0, 1)
    phone_service = get_numeric_input("Phone Service (1 for Yes, 0 for No)", int, 0, 1)
    device_protection = get_numeric_input("Device Protection (1 for Yes, 0 for No)", int, 0, 1)
    online_backup = get_numeric_input("Online Backup (1 for Yes, 0 for No)", int, 0, 1)
    partner = get_numeric_input("Partner (1 for Yes, 0 for No)", int, 0, 1)
    dependents = get_numeric_input("Dependents (1 for Yes, 0 for No)", int, 0, 1)
    tech_support = get_numeric_input("Tech Support (1 for Yes, 0 for No)", int, 0, 1)
    online_security = get_numeric_input("Online Security (1 for Yes, 0 for No)", int, 0, 1)

    # One-hot Encoding Categorical Variables
    contract_encoded = [1 if contract == "Month-to-month" else 0,
                        1 if contract == "One year" else 0,
                        1 if contract == "Two year" else 0]

    internet_service_encoded = [1 if internet_service == "Fiber optic" else 0,
                                1 if internet_service == "DSL" else 0,
                                1 if internet_service == "No" else 0]

    payment_method_encoded = [1 if payment_method == "Electronic check" else 0,
                              1 if payment_method == "Mailed check" else 0,
                              1 if payment_method == "Bank transfer (automatic)" else 0,
                              1 if payment_method == "Credit card (automatic)" else 0]

    gender_encoded = [1 if gender == "Male" else 0, 1 if gender == "Female" else 0]

    # Creating Feature Array with Only the Relevant 27 Features
    features = np.array([paperless_billing, senior_citizen, streaming_tv, streaming_movies,
                         multiple_lines, phone_service, device_protection, online_backup,
                         partner, dependents, tech_support, online_security,
                         monthly_charges, total_charges, tenure] +
                        contract_encoded + internet_service_encoded + payment_method_encoded + gender_encoded).reshape(1, -1)

    return features

# Get the User Input
user_data = get_user_input()

# Scale User Input
user_data_scaled = scaler_loaded.transform(user_data)

# Make Prediction
prediction = rf_model_loaded.predict(user_data_scaled)

# Displaying Result
print("\nPredicted Churn Status:\n", "Churned" if prediction[0] == 1 else "Not Churned")
