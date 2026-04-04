import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. SET PAGE CONFIG (Only ONCE at the very top)
st.set_page_config(
    page_title="Loan Guard AI",
    page_icon="🏦",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# 2. HIDE GITHUB ICON & FOOTER
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# 3. LOAD FILES
model = joblib.load('loan_model.pkl')
encoders = joblib.load('encoders.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("🏦 Loan Eligibility & Fraud Detection System")

# 4. CREATE FORM
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

# 5. PREDICTION LOGIC
if submitted:
    with st.spinner('AI is analyzing the risk factors...'):
        try:
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
            
            for col, le in encoders.items():
                if col in input_df.columns:
                    input_df[col] = le.transform(input_df[col].astype(str))
                    
            input_df = input_df[model_columns]
            
            prediction = model.predict(input_df)[0]
            status_label = encoders['loan_status'].inverse_transform([prediction])[0]
            
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