import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Files Load Karein
model = joblib.load('loan_model.pkl')
encoders = joblib.load('encoders.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Loan Guard AI", layout="wide")
st.title("🏦 Loan Eligibility & Fraud Detection System")

# 2. Form Banayein
with st.form("prediction_form"):
    st.subheader("Applicant Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", encoders['gender'].classes_)
        income = st.number_input("Monthly Income (₹)", value=50000)
        cibil = st.number_input("CIBIL Score", 300, 900, 750)
        
    with col2:
        loan_type = st.selectbox("Loan Type", encoders['loan_type'].classes_)
        # Dataset ke valid options yahan se aayenge
        loan_purpose = st.selectbox("Purpose of Loan", encoders['purpose_of_loan'].classes_)
        amount = st.number_input("Loan Amount Requested (₹)", value=200000)
        tenure = st.slider("Tenure (Months)", 12, 360, 36)
        
    with col3:
        emp_status = st.selectbox("Employment", encoders['employment_status'].classes_)
        prop_status = st.selectbox("Property Ownership", encoders['property_ownership_status'].classes_)
        existing_emi = st.number_input("Existing Monthly EMIs (₹)", value=0)
        monthly_spend = st.number_input("Avg Monthly Spending (₹)", value=15000)

    st.divider()
    past_fraud = st.radio("Known Fraud History or High-Risk Flags?", ["No", "Yes"], horizontal=True)
    
    submitted = st.form_submit_button("Assess Loan Risk")

# 3. Prediction Logic
if submitted:
    with st.spinner('Ai is analyzing the risk factors...'):
        try:
            # Saare columns jo model ko chahiye
            data = {
                'loan_type': loan_type,
                'loan_amount_requested': amount,
                'loan_tenure_months': tenure,
                'interest_rate_offered': 10.5,
                'purpose_of_loan': loan_purpose,
                'employment_status': emp_status,
                'monthly_income': income,
                'cibil_score': cibil,
                'existing_emis_monthly': existing_emi,
                'debt_to_income_ratio': ((existing_emi + monthly_spend) / income) if income > 0 else 0,
                'property_ownership_status': prop_status,
                'applicant_age': age,
                'gender': gender,
                'number_of_dependents': 0,
                'fraud_flag': 1 if past_fraud == "Yes" else 0,
                'avg_spend': monthly_spend,
                'total_spend': monthly_spend * 12,
                'txn_count': 50,
                'past_fraud_count': 1 if past_fraud == "Yes" else 0,
                'high_risk_flag': 1 if past_fraud == "Yes" else 0
            }
        
            input_df = pd.DataFrame([data])
            
            # Encoding (Translator)
            for col, le in encoders.items():
                if col in input_df.columns:
                    input_df[col] = le.transform(input_df[col].astype(str))
                    
            # Column Order Match Karein
            input_df = input_df[model_columns]
            
            # Result Nikalein
            prediction = model.predict(input_df)[0]
            status_label = encoders['loan_status'].inverse_transform([prediction])[0]
            
            # Result Show Karein
            st.subheader("Analysis Results")
            if status_label.lower() == 'approved' and past_fraud == "No":
                st.success(f"✅ ELIGIBILITY: {status_label.upper()}")
                st.info("RISK LEVEL: LOW")
                st.balloons()
            else:
                st.error(f"❌ ELIGIBILITY: REJECTED")
                if past_fraud == "Yes":
                    st.warning("⚠️ FRAUD ALERT: Past high-risk activity detected.")
                else:
                    st.warning("⚠️ REASON: Profile criteria not met (Low CIBIL/Income).")
                    
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.info("Tip: Make sure you ran 'train_model.py' to generate fresh .pkl files.")