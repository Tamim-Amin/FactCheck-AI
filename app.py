import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="FactCheck AI",
    page_icon="Oc",
    layout="centered",
    initial_sidebar_state="expanded"
)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@st.cache_resource
def load_resources():
    try:
        vector_form = pickle.load(open('vector.pkl', 'rb'))
        load_model = pickle.load(open('model.pkl', 'rb'))
        return vector_form, load_model
    except FileNotFoundError:
        return None, None

vector_form, load_model = load_resources()

port_stem = PorterStemmer()

def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)
    return con

def fake_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

st.markdown("""
    <style>
    .stTextArea textarea {
        font-size: 16px;
    }
    div.stButton > button:first-child {
        background-color: #0099ff;
        color: white;
        font-size: 20px;
        border-radius: 10px;
        padding: 10px 24px;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #007acc;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910768.png", width=100)
    st.title("About the App")
    st.info(
        "This tool uses a Machine Learning model to detect potentially fake news articles based on linguistic patterns."
    )
    st.markdown("---")
    st.write("**How to use:**")
    st.write("1. Paste the news text in the main window.")
    st.write("2. Click 'Analyze News'.")
    st.write("3. View the reliability score.")
    st.markdown("---")
    st.caption("Built with Streamlit & Scikit-Learn")

st.title("📰 FactCheck AI")
st.markdown("### Detect Fake News with Machine Learning")
st.write("Paste your news article below to analyze its credibility.")

sentence = st.text_area(
    "News Content",
    placeholder="Paste the full text of the article here...",
    height=250,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_btt = st.button("🔍 Analyze News")

if predict_btt:
    if not vector_form or not load_model:
        st.error("Error: Model files (vector.pkl or model.pkl) not found. Please upload them.")
    elif not sentence.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text patterns..."):
            try:
                prediction_class = fake_news(sentence)
                
                st.markdown("---")
                
                if prediction_class == [0]:
                    st.error("⚠️ **ALERT: Potential FAKE News Detected**")
                    st.write("The model has identified patterns consistent with unreliable or fake news sources.")
                elif prediction_class == [1]:
                    st.success("✅ **STATUS: Likely REAL News**")
                    st.write("The content appears to be consistent with reliable news reporting.")
                else:
                    st.warning("Unable to determine with high confidence.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")