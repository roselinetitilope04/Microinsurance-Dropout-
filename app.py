import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import shap
import xgboost as xgb

st.set_page_config(
    page_title="Microinsurance Dropout Dashboard",
    layout="wide"
)

# =======================
# Sidebar: File upload
# =======================
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# =======================
# 🏠 Overview
# =======================
st.title("🏠 Microinsurance Dropout Risk Dashboard")

st.markdown("""
### 📊 Current Dropout Rate
**63.8%**  
Potential improvement: **-5.0%**
""")

st.markdown("""
### 💰 Annual Cost
Total Cost: **₦34.5M**  
Recoverable: **₦14.8M**
""")

st.markdown("""
### 🎯 Model Accuracy
Accuracy: **95.6%**  
Can catch **96% of high-risk beneficiaries**
""")

st.markdown("""
### 🏆 3-Year ROI
ROI: **586%**  
Payback period: **4.3 months**
""")

st.markdown("---")
st.subheader("📋 Project Overview")
st.markdown("""
**Objective:** Predict microinsurance dropout risk in Sub-Saharan Africa  

**Dataset:** 10,829 beneficiaries across 4 integrated datasets:  
- Policy information  
- Mobile money transactions  
- Weather data  
- Clinic access metrics  

**Key Achievement:** AI model identifies 95.6% of dropouts 1-6 months in advance  

**Business Impact:** ₦14.8M annual savings potential through targeted interventions
""")

st.markdown("---")
st.subheader("🚨 Critical Discovery")
st.markdown("""
**\"Months_Since_Claim\"** is **3.7x more predictive of dropout** than any other factor.  
Beneficiaries who haven't claimed recently are at extreme risk.  

**Immediate Action Required:** Target 2,500+ beneficiaries with 90+ days since last claim
""")

# =======================
# Load and process uploaded CSV
# =======================
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("File loaded successfully!")

        # Ensure required columns exist
        required_columns = ["Label", "Probability", "Phone", "Region"]
        for col in required_columns:
            if col not in data.columns:
                data[col] = None

        # Slider for probability threshold
        threshold = st.sidebar.slider(
            "Set probability threshold for high-risk",
            0.0, 1.0, 0.7, 0.01
        )

        # Filter high-risk rows
        high_risk = data[data["Probability"].fillna(0) > threshold]

        st.subheader("High-Risk Beneficiaries")
        st.dataframe(high_risk)

        # =======================
        # ⚡ Quick Actions (Simulate SMS Alerts)
        # =======================
        st.subheader("⚡ Quick Actions: Simulated SMS Alerts")
        for _, row in high_risk.iterrows():
            phone = row['Phone'] if row['Phone'] else "No phone provided"
            label = row['Label'] if row['Label'] else "Unknown"
            st.write(f"SMS would be sent to {phone}: ALERT: {label} has high dropout risk ({row['Probability']*100:.2f}%)")

        # Download button
        to_download = BytesIO()
        high_risk.to_csv(to_download, index=False)
        to_download.seek(0)
        st.download_button(
            "Download High-Risk Records",
            data=to_download,
            file_name="high_risk_records.csv",
            mime="text/csv"
        )

        # =======================
        # 🎯 AI Insights (Feature Importance)
        # =======================
        st.subheader("🎯 AI Insights (Feature Importance)")

        feature_cols = [col for col in data.columns if col not in ["Label", "Probability", "Phone", "Region"]]
        if feature_cols:
            X = data[feature_cols].fillna(0)
            y = data["Label"].fillna(0)
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            model.fit(X, y)

            # SHAP values
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

            st.text("Top 10 features driving dropout risk:")
            feature_importance = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": abs(shap_values.values).mean(axis=0)
            }).sort_values(by="Importance", ascending=False).head(10)
            st.table(feature_importance)

            # SHAP summary plot
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(bbox_inches='tight')
        else:
            st.info("No features available for AI insights.")

        # =======================
        # 🌍 Regional Analysis
        # =======================
        st.subheader("🌍 Dropout Risk by Region")
        if "Region" in data.columns:
            region_summary = high_risk.groupby("Region")["Probability"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10,5))
            sns.barplot(x=region_summary.index, y=region_summary.values, palette="Reds_r", ax=ax)
            ax.set_ylabel("Average Dropout Probability")
            ax.set_xlabel("Region")
            ax.set_title("High-Risk Beneficiaries by Region")
            st.pyplot(fig)
        else:
            st.info("Region column not found in the dataset.")

        # =======================
        # 📈 Business Impact
        # =======================
        st.subheader("📈 Business Impact")
        total_cost = 34_500_000  # Example
        recoverable = 14_800_000
        potential_savings_pct = recoverable / total_cost * 100

        st.metric("Total Cost", f"₦{total_cost/1e6:.1f}M")
        st.metric("Recoverable", f"₦{recoverable/1e6:.1f}M", delta=f"{potential_savings_pct:.1f}%")

        # ROI Chart
        roi = 586
        payback_months = 4.3
        st.write(f"ROI: {roi}%  |  Payback: {payback_months} months")

    except Exception as e:
        st.error(f"Error loading file: {e}")
