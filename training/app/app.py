import streamlit as st
import pickle
import re
import nltk
import pdfplumber
import spacy
import docx2txt
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

# Load trained components
model = pickle.load(open("model.pkl", "rb"))  # OneVsRestClassifier
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))  # MultiLabelBinarizer

# Clean the resume text
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Extract resume text from PDF or DOCX
def extract_text(file):
    if file.name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text()
        return text
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    else:
        return ""

# Extract all skills (using spaCy)
def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text.strip() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "LANGUAGE", "PERSON", "SKILLS", "EXPERIENCE"]]
    return sorted(set(skills))

# UI setup
st.set_page_config(page_title="Resume Role Predictor", layout="centered")
st.title("Resume Role Predictor")
st.markdown("Upload a resume file (`.pdf` or `.docx`) or paste text to predict job roles and extract skills.")

uploaded_file = st.file_uploader("📄 Upload Resume", type=["pdf", "docx"])

resume_input = ""
if uploaded_file:
    resume_input = extract_text(uploaded_file)
    st.text_area("📄 Extracted Resume Text", resume_input, height=250)
else:
    resume_input = st.text_area("Or Paste Resume Text", height=250)

if st.button("🔍 Predict Role(s)"):
    if resume_input.strip() == "":
        st.warning("⚠️ Please provide resume content.")
    else:
        cleaned = clean_text(resume_input)
        vectorized = vectorizer.transform([cleaned])
        predictions = model.predict(vectorized)

        roles = label_encoder.inverse_transform(predictions)
        if isinstance(roles[0], (list, tuple)):
            role_list = roles[0]
            st.success("✅ **Predicted Role(s):** " + ", ".join(role_list))
        else:
            st.success("✅ **Predicted Role:** " + roles[0])

        with st.expander("🔧 Extracted Skills"):
            skills = extract_skills(resume_input)
            if skills:
                st.markdown("🛠️ **Skills Found:**")
                st.write(", ".join(skills))
            else:
                st.write("No skills detected.")
