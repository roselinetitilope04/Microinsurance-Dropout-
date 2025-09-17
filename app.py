import streamlit as st
import pandas as pd
import numpy as np
from skops.io import load
from twilio.rest import Client
import requests
import smtplib
from email.mime.text import MIMEText

# -----------------------------
# Load Model
# -----------------------------
pipeline = load("xgb_pipeline.skops", trusted=[
    'numpy.dtype',
    'sklearn.compose._column_transformer._RemainderColsList',
    'xgboost.core.Booster',
    'xgboost.sklearn.XGBClassifier'
])

# -----------------------------
# User Input / CSV Upload
# -----------------------------
st.title("Microinsurance Dropout Risk Dashboard")

uploaded_file = st.file_uploader("Upload CSV with beneficiaries", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("Raw Data")
    st.dataframe(data.head())

    # Predict dropout risk
    predictions = pipeline.predict_proba(data)[:,1]  # probability of dropout
    data["Dropout_Risk"] = predictions
    st.subheader("Predicted Dropout Risk")
    st.dataframe(data[["Dropout_Risk"]].head())

    # -----------------------------
    # Regional Dashboard
    # -----------------------------
    st.subheader("Dropout Risk by Region")
    region_summary = data.groupby("Region")["Dropout_Risk"].mean().reset_index()
    st.bar_chart(region_summary.rename(columns={"Dropout_Risk": "Avg_Risk"}).set_index("Region"))

    # -----------------------------
    # Alert System
    # -----------------------------
    HIGH_RISK_THRESHOLD = 0.7
    high_risk_data = data[data["Dropout_Risk"] > HIGH_RISK_THRESHOLD].copy()  # avoid SettingWithCopyWarning

    # Create a label for alerts (since no Beneficiary_Name)
    high_risk_data["Label"] = (
        "Age: " + high_risk_data["Age"].astype(str) +
        ", Region: " + high_risk_data["Region"].astype(str) +
        ", Claims: " + high_risk_data["Total_Claims"].astype(str)
    )

    st.subheader("High Risk Beneficiaries")
    st.dataframe(high_risk_data)

    if not high_risk_data.empty:
        st.warning(f"{len(high_risk_data)} beneficiaries are at high dropout risk!")

        # --- SMS via Twilio ---
        TWILIO_SID = "your_sid"
        TWILIO_AUTH = "your_auth_token"
        TWILIO_PHONE = "+1234567890"
        TARGET_PHONE = "+2345678901"

        client = Client(TWILIO_SID, TWILIO_AUTH)
        for _, row in high_risk_data.iterrows():
            msg_body = f"ALERT: {row['Label']} has high dropout risk ({row['Dropout_Risk']:.2f})"
            client.messages.create(body=msg_body, from_=TWILIO_PHONE, to=TARGET_PHONE)

        # --- Slack Webhook ---
        SLACK_WEBHOOK = "your_slack_webhook_url"
        for _, row in high_risk_data.iterrows():
            payload = {"text": f"ALERT: {row['Label']} has high dropout risk ({row['Dropout_Risk']:.2f})"}
            requests.post(SLACK_WEBHOOK, json=payload)

        # --- Email Notification ---
        SMTP_SERVER = "smtp.gmail.com"
        SMTP_PORT = 587
        EMAIL = "your_email@gmail.com"
        PASSWORD = "your_email_password"
        for _, row in high_risk_data.iterrows():
            msg = MIMEText(f"ALERT: {row['Label']} has high dropout risk ({row['Dropout_Risk']:.2f})")
            msg['Subject'] = "High Dropout Risk Alert"
            msg['From'] = EMAIL
            msg['To'] = "recipient_email@example.com"
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL, PASSWORD)
                server.send_message(msg)

    st.success("Predictions complete and alerts sent!")
