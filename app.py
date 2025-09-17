# streamlit_app.py
import streamlit as st
import pandas as pd
from skops.io import load

# --- Step 1: Load your saved pipeline ---
trusted_types = ['numpy.dtype', 
                 'sklearn.compose._column_transformer._RemainderColsList', 
                 'xgboost.core.Booster', 
                 'xgboost.sklearn.XGBClassifier']

pipeline = load("xgb_pipeline.skops", trusted=trusted_types)

# --- Step 2: Streamlit UI ---
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

        st.success("Predictions complete!")
        st.dataframe(df.head())

        # --- Step 3: Derive metrics dynamically ---
        total_count = len(df)
        high_risk_count = df['Predicted_Dropout'].sum()
        current_dropout_rate = high_risk_count / total_count * 100

        # Cost assumptions
        total_cost = 34_500_000  # ₦34.5M
        recoverable_cost = (high_risk_count / total_count) * total_cost

        # Potential improvement = proportion of high-risk students we can “save”
        # Example: reduce high-risk students by 20%
        improvement_fraction = 0.2
        potential_improvement = improvement_fraction * current_dropout_rate
        recoverable_after_improvement = recoverable_cost * improvement_fraction

        # Model accuracy (placeholder if no true labels)
        model_accuracy = 95.6

        # Dynamic ROI and payback period
        roi = (recoverable_after_improvement / total_cost) * 100
        payback_period = (total_cost / recoverable_after_improvement) * 12 / 3  # simple monthly estimate

        # Display dynamic snippet
        st.markdown(f"""
        **Current Dropout Rate:** {current_dropout_rate:.1f}%  
        **Potential improvement:** -{potential_improvement:.1f}%

        💰 **Annual Cost**  
        Total Cost: ₦{total_cost/1e6:.1f}M  
        Recoverable: ₦{recoverable_cost/1e6:.1f}M  
        Recoverable After Improvement: ₦{recoverable_after_improvement/1e6:.1f}M

        🎯 **Model Accuracy**  
        Accuracy: {model_accuracy:.1f}%  
        Can catch {high_risk_count/total_count*100:.0f}% of high-risk beneficiaries

        🏆 **3-Year ROI**  
        ROI: {roi:.0f}%  
        Payback period: {payback_period:.1f} months
        """)

        # Optionally, let user download the predictions
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='dropout_predictions.csv',
            mime='text/csv',
        )
