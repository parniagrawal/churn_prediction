import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load everything
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Create a sample input row (you can also load it from your actual data)
sample_input = pd.DataFrame([[0.0]*len(feature_columns)], columns=feature_columns)

# Title
st.title("🛍️ Superstore Profitability Predictor")
st.write("Predict whether an order will be profitable based on key details.")

# Select common inputs
st.subheader("Enter Key Order Info")

# Example: common features — adjust based on your actual column names
sample_input['Sales'] = st.number_input("Sales Amount", value=200.0)
sample_input['Quantity'] = st.number_input("Quantity", value=2)
sample_input['Discount'] = st.slider("Discount", 0.0, 1.0, 0.1, 0.01)
sample_input['Shipping.Cost'] = st.number_input("Shipping Cost", value=15.0)

# Optional: show advanced inputs
with st.expander("🛠️ Advanced Features"):
    for col in sample_input.columns:
        if col not in ['Sales', 'Quantity', 'Discount', 'Shipping.Cost']:
            sample_input[col] = st.number_input(col, value=0.0)

# Predict button
if st.button("Predict"):
    input_scaled = scaler.transform(sample_input)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ This order is likely to be **Profitable**!")
    else:
        st.error("❌ This order is likely to be **Not Profitable**.")
