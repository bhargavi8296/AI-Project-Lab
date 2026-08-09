import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent
# Page configuration
st.set_page_config(
    page_title="Transaction Anomaly Detector",
    page_icon="💳",
    layout="wide"
)


# Simple styling
st.markdown(
    """
    <style>
    .main-title {
        background: linear-gradient(90deg, #1e3a8a, #0f766e);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
    }

    .main-title h1 {
        margin: 0;
        color: white;
    }

    .main-title p {
        margin-top: 8px;
        color: #e0f2fe;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Header
st.markdown(
    """
    <div class="main-title">
        <h1>💳 Transaction Anomaly Detector</h1>
        <p>
            Upload transaction data and identify
            potentially suspicious transactions
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# Basic project information
st.info(
    "This application uses an Isolation Forest model "
    "to identify unusual transaction patterns."
)


# Upload CSV
uploaded_file = st.file_uploader(
    "Upload your transaction CSV file",
    type=["csv"]
)


# Run only when a file is uploaded
if uploaded_file is not None:

    # Read CSV file
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(
        data.head(),
        use_container_width=True
    )


    # Columns required by the model
    required_columns = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount"]
    )


    # Find missing columns
    missing_columns = []

    for column in required_columns:

        if column not in data.columns:

            missing_columns.append(column)


    # Stop if required columns are missing
    if len(missing_columns) > 0:

        st.error(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

        st.stop()


    # Load saved model
    try:

        artifact = joblib.load(
            BASE_DIR/"transaction_anomaly_detector.pkl"
        )

        model = artifact["model"]
        features = artifact["features"]

    except FileNotFoundError:

        st.error(
            "Model file not found. Keep "
            "transaction_anomaly_detector.pkl "
            "in the same folder as app.py."
        )

        st.stop()


    # Create TransactionHour
    data["TransactionHour"] = (
        data["Time"] // 3600
    ) % 24


    # Apply log transformation to Amount
    data["LogTransactionAmount"] = np.log1p(
        data["Amount"]
    )


    # Select V1 to V28
    model_input = data[
        [f"V{i}" for i in range(1, 29)]
        + [
            "TransactionHour",
            "LogTransactionAmount"
        ]
    ]


    # Keep features in the same training order
    model_input = model_input[features]


    # Predict transactions
    with st.spinner(
        "Checking transactions..."
    ):

        predictions = model.predict(
            model_input
        )

        anomaly_scores = model.decision_function(
            model_input
        )


    # Convert model output into labels
    data["Prediction"] = np.where(
        predictions == -1,
        "Suspicious",
        "Normal"
    )


    # Add anomaly score
    data["AnomalyScore"] = anomaly_scores


    # Calculate counts
    total_transactions = len(data)

    suspicious_transactions = (
        data["Prediction"] == "Suspicious"
    ).sum()

    normal_transactions = (
        total_transactions
        - suspicious_transactions
    )

    anomaly_rate = (
        suspicious_transactions
        / total_transactions
    ) * 100


    # Display summary
    st.subheader("Prediction Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        total_transactions
    )

    col2.metric(
        "Normal",
        normal_transactions
    )

    col3.metric(
        "Suspicious",
        suspicious_transactions
    )

    col4.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.2f}%"
    )


    # Show status
    if suspicious_transactions > 0:

        st.warning(
            f"{suspicious_transactions} suspicious "
            "transaction(s) detected."
        )

    else:

        st.success(
            "No suspicious transactions detected."
        )


    # Create two tabs
    tab1, tab2 = st.tabs(
        [
            "All Predictions",
            "Suspicious Transactions"
        ]
    )


    # All transaction results
    with tab1:

        st.dataframe(
            data,
            use_container_width=True
        )


    # Only suspicious transactions
    with tab2:

        suspicious_data = data[
            data["Prediction"] == "Suspicious"
        ].sort_values(
            by="AnomalyScore"
        )

        if suspicious_data.empty:

            st.success(
                "No suspicious transactions found."
            )

        else:

            st.dataframe(
                suspicious_data,
                use_container_width=True
            )


    # Convert results into CSV
    result_csv = data.to_csv(
        index=False
    ).encode("utf-8")


    # Download button
    st.download_button(
        label="Download Prediction Results",
        data=result_csv,
        file_name="transaction_predictions.csv",
        mime="text/csv"
    )


else:

    st.write(
        "Upload a CSV file to start anomaly detection."
    )
