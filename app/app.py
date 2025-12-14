
import streamlit as st
import pickle
from preprocessing import clean_text

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

vectorizer = pickle.load(open("C:/Users/avani/OneDrive/Desktop/Fake_News/models/vectorizer.pkl", "rb"))
lr_model = pickle.load(open("C:/Users/avani/OneDrive/Desktop/Fake_News/models/model_lr.pkl", "rb"))
mnb_model = pickle.load(open("C:/Users/avani/OneDrive/Desktop/Fake_News/models/model_mnb.pkl", "rb"))
xgb_model = pickle.load(open("C:/Users/avani/OneDrive/Desktop/Fake_News/models/model_xgb.pkl", "rb"))



st.markdown("""
<style>

/* App background */
.main {
    background-color: #0e1117;
    color: #e0e0e0;
}

/* Title box */
.title-box {
    background: linear-gradient(135deg, #1f6feb, #0d47a1);
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0px 6px 16px rgba(0,0,0,0.6);
}

/* Text area */
textarea {
    background-color: #161b22 !important;
    color: #e6edf3 !important;
    border-radius: 12px !important;
    border: 1px solid #30363d !important;
    font-size: 16px !important;
}

/* Select box */
.stSelectbox > div > div {
    background-color: #161b22 !important;
    color: #e6edf3 !important;
    border-radius: 10px !important;
    border: 1px solid #30363d !important;
}

/* Button */
.stButton>button {
    background-color: #238636;
    color: white;
    border-radius: 10px;
    padding: 12px 24px;
    font-size: 17px;
    font-weight: 600;
    border: none;
    transition: 0.25s;
}

.stButton>button:hover {
    background-color: #2ea043;
    transform: scale(1.05);
}

/* Result boxes */
.result-box {
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    margin-top: 20px;
}

.fake {
    background-color: #3a0f14;
    color: #ff7b72;
    border: 1px solid #8e1519;
}

.real {
    background-color: #0f2f1f;
    color: #3fb950;
    border: 1px solid #238636;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background-color: #161b22;
    color: #c9d1d9;
    border-radius: 10px;
    padding: 10px;
    margin-right: 6px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #238636 !important;
    color: white !important;
}

/* Progress bar */
.stProgress > div > div {
    background-color: #238636;
}

/* Footer remove */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)



st.markdown("""
<div class="title-box">
    <h1>📰 Fake News Detection System</h1>
    <h3>NLP + Machine Learning Classification</h3>
</div>
""", unsafe_allow_html=True)

st.write("Enter a news article and select a model to classify whether it is fake or real.")


col1, col2 = st.columns([2, 1])

with col1:
    news = st.text_area("Enter News Content", height=240)

with col2:
    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Multinomial Naive Bayes", "XGBoost"]
    )


if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        with st.spinner("Analyzing news content..."):
            cleaned_text = clean_text(news)
            vec_input = vectorizer.transform([cleaned_text])

            if model_choice == "Logistic Regression":
                pred = lr_model.predict(vec_input)[0]
                confidence = max(lr_model.predict_proba(vec_input)[0])
            elif model_choice == "Multinomial Naive Bayes":
                pred = mnb_model.predict(vec_input)[0]
                confidence = max(mnb_model.predict_proba(vec_input)[0])
            else:
                pred = xgb_model.predict(vec_input)[0]
                confidence = max(xgb_model.predict_proba(vec_input)[0])

        if pred == 0:
            st.markdown(
                "<div class='result-box fake'>🔴 Fake News Detected</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-box real'>🟢 Real News Detected</div>",
                unsafe_allow_html=True
            )

        st.progress(float(confidence))
        st.caption(f"Model confidence: {confidence:.2f}")



st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📊 Model Performance Insights")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Accuracy",
    "🧮 Confusion Matrix",
    "📈 ROC Curve",
    "☁️ Word Cloud"
])

with tab1:
    st.image("C:/Users/avani/OneDrive/Desktop/Fake_News/plots/accuracy_comparison.png")

with tab2:
    st.image("C:/Users/avani/OneDrive/Desktop/Fake_News/plots/confusion_matrices.png")

with tab3:
    st.image("C:/Users/avani/OneDrive/Desktop/Fake_News/plots/roc_curve.png")

with tab4:
    st.image("C:/Users/avani/OneDrive/Desktop/Fake_News/plots/wordcloud.png")

st.markdown(
    "<hr><center>Built with NLP, Machine Learning & Streamlit</center>",
    unsafe_allow_html=True
)

