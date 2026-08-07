
import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="centered"
)


@st.cache_resource
def load_model():
    bundle = joblib.load(
        "customer_segmentation_model.pkl"
    )
    return bundle


bundle = load_model()

model = bundle["model"]
scaler = bundle["scaler"]
cluster_names = bundle["cluster_names"]
features = bundle["features"]


st.title("Customer Segmentation System")

st.write(
    "Enter the customer's purchasing behaviour "
    "to identify their segment."
)


recency = st.number_input(
    "Recency (days since last purchase)",
    min_value=0,
    value=30,
    step=1
)

frequency = st.number_input(
    "Frequency (number of orders)",
    min_value=1,
    value=5,
    step=1
)

monetary = st.number_input(
    "Monetary (total customer spending)",
    min_value=0.01,
    value=1000.00,
    step=100.00
)


if st.button(
    "Predict Customer Segment",
    type="primary"
):
    new_customer = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    new_customer_log = np.log1p(
        new_customer
    )

    new_customer_scaled = scaler.transform(
        new_customer_log
    )

    new_customer_scaled = pd.DataFrame(
        new_customer_scaled,
        columns=features
    )

    cluster = model.predict(
        new_customer_scaled
    )[0]

    segment = cluster_names[cluster]

    st.success(
        f"Customer Segment: {segment}"
    )

    st.write(
        f"Cluster Number: {cluster}"
    )

    if segment == "High-Value Customers":
        st.info(
            "This customer purchases frequently, "
            "spends more and has purchased recently."
        )

    elif segment == "Regular Customers":
        st.info(
            "This customer has moderate purchasing "
            "frequency and spending."
        )

    else:
        st.warning(
            "This customer has not purchased recently "
            "and may require a re-engagement campaign."
        )
