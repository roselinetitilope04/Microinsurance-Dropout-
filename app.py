import streamlit as st
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Insurer Dashboard", layout="wide")

st.title("Insurer Dashboard: High-Risk Dropout Tracking")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("File loaded successfully!")

        # Ensure required columns exist
        required_columns = ["Label", "Probability", "Phone", "Region"]
        for col in required_columns:
            if col not in data.columns:
                data[col] = None

        # Sidebar controls
        st.sidebar.header("Filters")
        threshold = st.sidebar.slider(
            "Set probability threshold for high-risk:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01
        )

        regions = data['Region'].dropna().unique().tolist()
        selected_regions = st.sidebar.multiselect("Select Region(s):", options=regions, default=regions)

        # Filter high-risk
        high_risk_data = data[
            (data["Probability"].fillna(0) > threshold) &
            (data["Region"].isin(selected_regions))
        ]

        # KPI Cards
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Beneficiaries", len(data))
        col2.metric("High-Risk Count", len(high_risk_data))
        col3.metric("Average Risk Probability", f"{high_risk_data['Probability'].mean():.2f}" if not high_risk_data.empty else "0.00")

        # Region Summary Table
        st.subheader("High-Risk Summary by Region")
        if high_risk_data.empty:
            st.info("No high-risk records found for selected threshold/region(s).")
        else:
            region_summary = high_risk_data.groupby("Region").agg(
                High_Risk_Count=("Probability", "count"),
                Avg_Risk_Prob=("Probability", "mean")
            ).reset_index()
            st.dataframe(region_summary)

            # Visualizations
            fig1, ax1 = plt.subplots(figsize=(8,4))
            sns.barplot(data=region_summary, x="Region", y="High_Risk_Count", palette="Reds", ax=ax1)
            ax1.set_title("High-Risk Beneficiaries per Region")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots(figsize=(8,4))
            sns.histplot(high_risk_data['Probability'], bins=20, color='orange', kde=True, ax=ax2)
            ax2.set_title("Distribution of High-Risk Probabilities")
            ax2.set_xlabel("Probability")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

            # Beneficiary-level table
            st.subheader("High-Risk Beneficiaries")
            st.dataframe(high_risk_data)

            # Simulated SMS Alerts
            st.subheader("Simulated SMS Alerts")
            for _, row in high_risk_data.iterrows():
                phone = row['Phone'] if row['Phone'] else "No phone provided"
                label = row['Label'] if row['Label'] else "Unknown"
                st.write(f"SMS would be sent to {phone}: ALERT: {label} has high dropout risk ({row['Probability']*100:.2f}%)")

            # Download CSV
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
