import streamlit as st
import joblib
import numpy as np

model = joblib.load('rf_model.pkl')  # loads the trained Random Forest model from file into memory

st.title("🏦 Loan Default Predictor") #displays a large heading on the web page
st.write("Enter the borrower details below to predict likelihood of loan default.") #displays normal text below the title
st.subheader("Borrower Details")

age = st.number_input("Age", min_value=18, max_value=100, value=35)

monthly_income_inr = st.number_input("Monthly Income (₹)", min_value=0, max_value=500000, value=25000)
monthly_income = monthly_income_inr / 95

debt_ratio = st.number_input("Debt Ratio", min_value=0.0, max_value=10.0, value=0.5)

revolving_utilization = st.number_input("Revolving Utilization (0 to 1)", min_value=0.0, max_value=1.0, value=0.3)

num_open_credit = st.number_input("Number of Open Credit Lines", min_value=0, max_value=50, value=5)

num_real_estate = st.number_input("Number of Real Estate Loans", min_value=0, max_value=20, value=1)

num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)

num_30_59_days = st.number_input("Times 30-59 Days Late", min_value=0, max_value=20, value=0)

num_60_89_days = st.number_input("Times 60-89 Days Late", min_value=0, max_value=20, value=0)

num_90_days = st.number_input("Times 90+ Days Late", min_value=0, max_value=20, value=0)

#What st.number_input does:
#creates a number input box on the web page where the user can type or click arrows to change the value.

#What value Does
#value is the default number that appears in the input box when the app first loads — before the user changes anything.

if st.button('PREDICT'):
    input_data = np.array([[
        revolving_utilization,
        age,
        num_30_59_days,
        debt_ratio,
        monthly_income,
        num_open_credit,
        num_90_days,
        num_real_estate,
        num_60_89_days,
        num_dependents
    ]])

    prediction = model.predict(input_data)[0]    #returns 0 or 1 — default or no default
    probability = model.predict_proba(input_data)[0][1]    #returns probability of default as decimal

    st.subheader('Prediction Result')

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default — Probability: {probability * 100:.1f}%")  #converts 0.78 to 78.0%
    else:
        st.success(f"✅ Low Risk of Default — Probability: {probability * 100:.1f}%") 
#st.error -- shows a RED box with warning message
#st.success -- shows a GREEN box with safe message  
#f stands for float (decimal number)
#.1 means 1 digit after the decimal point


