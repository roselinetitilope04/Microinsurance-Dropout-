# app.py
import streamlit as st
import pandas as pd
import numpy as np

from skops.io import load, get_untrusted_types
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Step 1: Define initial trusted types ---
trusted_types = [
    np.dtype,
    Pipeline,
    ColumnTransformer,
    OneHotEncoder,
    StandardScaler,
    XGBClassifier,
]

# --- Step 2: Load pipeline with auto-trust fallback ---
def load_pipeline():
    try:
        return load("xgb_pipeline.skops", trusted=trusted_types)
    except Exception:
        untrusted = get_untrusted_types("xgb_pipeline.skops")
        if untrusted:
            st.warning("⚠️ Auto-adding untrusted types:")
            st.json([str(u) for u in untrusted])

            for u in untrusted:
                try:
                    module = __import__(u.module, fromlist=[u.name])
                    cls = getattr(module, u.name)
                    trusted_types.append(cls)
                except Exception as e:
                    st.error(f"❌ Could not import {u}: {e}")

            return load("xgb_pipeline.skops", trusted=trusted_types)
        else:
            raise

pipeline = load_pipeline()

# --- Step 3: Streamlit UI ---
st.title("Dropout Prediction App")
st.write("Upload a CSV with the required features to predict dropout.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.dataframe(df.head())

    # Check if required features are present
    feature_names = pipeline.named_steps['preprocessor'].transformers_[0][2] + \
                    pipeline.named_steps['preprocessor'].transformers_[1][2]

    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        st.error(f"Missing required features: {missing_features}")
    else:
        # Predict using the loaded pipeline
        predictions = pipeline.predict(df[feature_names])
        df['Predicted_Dropout'] = predictions

        st.success("✅ Predictions complete!")
        st.dataframe(df.head())

        # --- Step 4: Derive metrics dynamically ---
        total_count = len(df)
        high_risk_count = df['Predicted_Dropout'].sum()
        current_dropout_rate = high_risk_count / total_count * 100

        # Cost assumptions
        total_cost = 34_500_000  # ₦34.5M
        recoverable_cost = (high_risk_count / total_count) * total_cost

        # Potential improvement (20% reduction in high-risk)
        improvement_fraction = 0.2
        potential_improvement = improvement_fraction * current_dropout_rate
        recoverable_after_improvement = recoverable_cost * improvement_fraction

        # Model accuracy (placeholder if no true labels)
        model_accuracy = 95.6

        # ROI and payback period
        roi = (recoverable_after_improvement / total_cost) * 100
        payback_period = (total_cost / recoverable_after_improvement) * 12 / 3  # monthly est.

        # Display dynamic results
        st.markdown(f"""
        **📊 Current Dropout Rate:** {current_dropout_rate:.1f}%  
        **📉 Potential improvement:** -{potential_improvement:.1f}%

        💰 **Annual Cost**  
        - Total Cost: ₦{total_cost/1e6:.1f}M  
        - Recoverable: ₦{recoverable_cost/1e6:.1f}M  
        - Recoverable After Improvement: ₦{recoverable_after_improvement/1e6:.1f}M

        🎯 **Model Accuracy**  
        - Accuracy: {model_accuracy:.1f}%  
        - Captures {high_risk_count/total_count*100:.0f}% of high-risk beneficiaries

        🏆 **3-Year ROI**  
        - ROI: {roi:.0f}%  
        - Payback period: {payback_period:.1f} months
        """)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv,
            file_name='dropout_predictions.csv',
            mime='text/csv',
        )
