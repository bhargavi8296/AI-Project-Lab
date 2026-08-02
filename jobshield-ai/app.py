
import joblib
import streamlit as st


st.set_page_config(
    page_title="JobShield AI",
    page_icon="🛡️",
    layout="centered"
)


@st.cache_resource
def load_files():
    model = joblib.load(
        "fake_job_detector.pkl"
    )

    vectorizer = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    return model, vectorizer


model, vectorizer = load_files()


def predict_job_posting(text, threshold=0.50):
    text_tfidf = vectorizer.transform(
        [text]
    )

    fraud_probability = model.predict_proba(
        text_tfidf
    )[0][1]

    if fraud_probability >= threshold:
        prediction = "Potentially Fraudulent"
    else:
        prediction = "Likely Legitimate"

    return prediction, fraud_probability


st.title("🛡️ JobShield AI")

st.write(
    "Analyze a job posting using NLP and machine "
    "learning to identify potential fraud risk."
)

title = st.text_input(
    "Job Title",
    placeholder="Example: Java Developer"
)

company_profile = st.text_area(
    "Company Profile",
    placeholder="Enter information about the company...",
    height=100
)

description = st.text_area(
    "Job Description",
    placeholder="Paste the complete job description...",
    height=180
)

requirements = st.text_area(
    "Requirements",
    placeholder="Enter skills, education and experience requirements...",
    height=120
)

benefits = st.text_area(
    "Benefits",
    placeholder="Enter salary, incentives and other benefits...",
    height=100
)

if st.button(
    "Analyze Job Posting",
    type="primary",
    use_container_width=True
):
    combined_text = " ".join([
        title,
        company_profile,
        description,
        requirements,
        benefits
    ]).strip()

    if not combined_text:
        st.warning(
            "Please enter job-posting information."
        )

    elif len(combined_text.split()) < 10:
        st.warning(
            "Please provide more details for reliable analysis."
        )

    else:
        prediction, probability = predict_job_posting(
            combined_text
        )

        st.subheader("Analysis Result")

        if prediction == "Potentially Fraudulent":
            st.error(
                "⚠️ Potentially Fraudulent"
            )
        else:
            st.success(
                "✅ Likely Legitimate"
            )

        st.metric(
            "Fraud Risk Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(
            int(probability * 100)
        )

        if probability >= 0.50:
            st.warning(
                "Verify the employer, company website, "
                "email domain and requests for money or "
                "sensitive information before applying."
            )

        st.caption(
            "This prediction is a risk indicator, not proof "
            "that a job posting is fraudulent."
        )
