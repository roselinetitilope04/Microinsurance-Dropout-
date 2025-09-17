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

        # Ensure required columns exist
        required_columns = ["Label", "Probability", "Phone"]
        added_cols = []
        for col in required_columns:
            if col not in data.columns:
                data[col] = None  # Initialize missing columns with None
                added_cols.append(col)

        if added_cols:
            st.warning(f"Added missing columns: {', '.join(added_cols)}")

        # Slider for probability threshold
        threshold = st.slider(
            "Set probability threshold for high-risk:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01
        )

        # Filter high-risk rows
        high_risk_data = data[data["Probability"].fillna(0) > threshold]

        if high_risk_data.empty:
            st.info("No high-risk records found for the selected threshold.")
        else:
            st.subheader("High-Risk Records")
            st.dataframe(high_risk_data)

            # Simulate sending SMS
            st.subheader("Simulated SMS Alerts")
            for _, row in high_risk_data.iterrows():
                phone = row['Phone'] if row['Phone'] else "No phone provided"
                label = row['Label'] if row['Label'] else "Unknown"
                st.write(f"SMS would be sent to {phone}: "
                         f"ALERT: {label} has high dropout risk "
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
