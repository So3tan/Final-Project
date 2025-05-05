import streamlit as st
import pandas as pd
import joblib
from catboost import Pool

# Load CatBoost model
model = joblib.load(r"C:\Users\braid\Downloads\insurance_predict_model.pkl")
expected_features = model.feature_names_

st.set_page_config(page_title="Claim Prediction", layout="wide")
st.title("🧾 Travel Insurance Claim Prediction")
st.write("Input the details of a single insurance claim to predict approval.")

# Define all input options
agencies = ['CBH', 'CWT', 'JZI', 'KML', 'EPX', 'C2B', 'JWT', 'RAB', 'SSI', 'ART', 'CSR', 'CCR',
 'ADM', 'LWC', 'TTW', 'TST']
agency_types = ['Airlines', 'Travel Agency']
products = ['Comprehensive Plan', 'Rental Vehicle Excess Insurance', 'Value Plan',
 'Basic Plan', 'Premier Plan', '2 way Comprehensive Plan', 'Bronze Plan',
 'Silver Plan', 'Annual Silver Plan', 'Cancellation Plan',
 '1 way Comprehensive Plan', 'Ticket Protector', '24 Protect', 'Gold Plan',
 'Annual Gold Plan', 'Single Trip Travel Protect Silver',
 'Individual Comprehensive Plan', 'Spouse or Parents Comprehensive Plan',
 'Annual Travel Protect Silver', 'Single Trip Travel Protect Platinum',
 'Annual Travel Protect Gold', 'Single Trip Travel Protect Gold',
 'Annual Travel Protect Platinum', 'Child Comprehensive Plan',
 'Travel Cruise Protect', 'Travel Cruise Protect Family']
channels = ['Online', 'Offline']

# Sidebar for user input
st.sidebar.header("Input Travel Claim Details")

agency = st.sidebar.selectbox("Agency", agencies)
agency_type = st.sidebar.selectbox("Agency Type", agency_types)
product_name = st.sidebar.selectbox("Product Name", products)
channel = st.sidebar.selectbox("Distribution Channel", channels)
net_sales = st.sidebar.slider("Net Sales", 0, 1000000, 1000)
commission = st.sidebar.slider("Commission (in value)", 0, 100000, 1000)
age = st.sidebar.slider("Age", 18, 100, 30)
profit = st.sidebar.slider("Profit", 0, 1000000, 1000)
duration = st.sidebar.slider("Duration (days)", 1, 365, 10)

# Create input DataFrame
input_df = pd.DataFrame({
    'Agency': [agency],
    'Agency Type': [agency_type],
    'Product Name': [product_name],
    'Distribution Channel': [channel],
    'Duration': [duration],
    'Net Sales': [net_sales],
    'Commision (in value)': [commission],
    'Age': [age],
    'Profit': [profit]
})

# ✅ Convert to numeric for these columns
input_df['Agency Type'] = input_df['Agency Type'].map({'Travel Agency': 0, 'Airlines': 1})
input_df['Distribution Channel'] = input_df['Distribution Channel'].map({'Online': 0, 'Offline': 1})

# Add AgeGroup feature based on Age
input_df['AgeGroup'] = pd.cut(input_df['Age'],
    bins=[0, 18, 30, 45, 60, 100],
    labels=['Teen', 'Young Adult', 'Adult', 'Mid-Age', 'Senior']
)

# One-hot encode AgeGroup
input_df = pd.get_dummies(input_df, columns=['AgeGroup'], drop_first=True)

# Reindex to match model's expected features
input_df = input_df.reindex(columns=expected_features, fill_value=0)

# Display user input
st.subheader("User Input Summary")
st.write("Here’s a snapshot of your selections:")
st.dataframe(input_df)

# Predict button
if st.button("Predict"):
    # Only Agency + Product Name are categoricals
    cat_features = ['Agency', 'Product Name']

    # Create CatBoost Pool
    input_pool = Pool(input_df, cat_features=cat_features)

    # Get probability prediction
    prob = model.predict(input_pool, prediction_type="Probability")[0]
    prob_rejection, prob_approval = prob[0], prob[1]

    # Final class prediction
    prediction = 'Yes' if prob_approval >= 0.5 else 'No'

    # Display results
    st.subheader("Prediction Result:")
    if prediction == 'Yes':
        st.success(f"✅ Claim is likely to be Approved.")
    else:
        st.error(f"🚨 Claim is likely to be Rejected — Possible Fraud or Invalid Submission.")

    st.markdown(f"""
    ### 🔢 Prediction Probabilities:
    - ✅ **Approval:** {prob_approval:.2%}
    - ❌ **Rejection:** {prob_rejection:.2%}
    """)
