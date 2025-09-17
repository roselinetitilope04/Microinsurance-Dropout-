import streamlit as st
import pandas as pd
from io import BytesIO

st.title("High-Risk Dropout Alert App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("File loaded successfully!")

        # Check required columns exist
        required_columns = ["Label", "Probability", "Phone"]
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            # Slider for probability threshold
            threshold = st.slider(
                "Set probability threshold for high-risk:",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.01
            )

            # Filter high-risk rows
            high_risk_data = data[data["Probability"] > threshold]

            if high_risk_data.empty:
                st.info("No high-risk records found for the selected threshold.")
            else:
                st.subheader("High-Risk Records")
                st.dataframe(high_risk_data)

                # Simulate sending SMS (instead of Twilio)
                st.subheader("Simulated SMS Alerts")
                for _, row in high_risk_data.iterrows():
                    st.write(f"SMS would be sent to {row['Phone']}: "
                             f"ALERT: {row['Label']} has high dropout risk "
                             f"({row['Probability']*100:.2f}%)")

                # Add download button
                st.subheader("Download High-Risk Records")
                to_download = BytesIO()
                high_risk_data.to_csv(to_download, index=False)
                to_download.seek(0)
                st.download_button(
                    label="Download CSV",
                    data=to_download,
                    file_name="high_risk_records.csv",
                    mime="text/csv"
                )

    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty.")
    except pd.errors.ParserError:
        st.error("Error parsing CSV file. Check for formatting issues.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
