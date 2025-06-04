import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("news.csv")

# Train model
@st.cache_resource
def train_model(data):
    x_train, x_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = vectorizer.fit_transform(x_train)
    tfidf_test = vectorizer.transform(x_test)

    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(tfidf_train, y_train)
    acc = accuracy_score(y_test, model.predict(tfidf_test))

    return model, vectorizer, acc

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.write("Enter a news article or paragraph to detect whether it's **Real** or **Fake**.")

user_input = st.text_area("🖊️ Paste the news content here:")

if st.button("Analyze"):
    with st.spinner("Processing..."):
        df = load_data()
        model, vectorizer, acc = train_model(df)
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]

        st.subheader("🔍 Result:")
        if prediction == "FAKE":
            st.error("🚫 This news is **FAKE**!")
        else:
            st.success("✅ This news is **REAL**!")

        st.info(f"Model Accuracy: {round(acc * 100, 2)}%")
