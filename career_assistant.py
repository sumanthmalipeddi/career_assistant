import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pdfplumber
import requests
import os
import re
from dotenv import load_dotenv
from fpdf import FPDF

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

MAX_TOKENS = 8000
GROQ_MODEL = "llama3-70b-8192"

# Load pickles
model = joblib.load("pickle_files/salary_model.pkl")
encoder = joblib.load("pickle_files/encoder.pkl")
mlb = joblib.load("pickle_files/mlb.pkl")
feature_order = joblib.load("pickle_files/feature_order.pkl")

def strip_non_latin(text):
    return text.encode('latin-1', errors='ignore').decode('latin-1')

def create_pdf(summary, text):
    clean_summary = strip_non_latin(summary)
    clean_text = strip_non_latin(text)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Updated Resume Summary:\n" + clean_summary + "\n\n")
    pdf.multi_cell(0, 10, "Full Resume Content:\n\n" + clean_text)
    path = "/tmp/Updated_Resume.pdf"
    pdf.output(path)
    return path

st.set_page_config(page_title="Career Assistant", layout="wide")

# Adaptive CSS for both light and dark mode
st.markdown("""
<style>
.stMarkdown, .stMarkdown div, .stMarkdown pre, .stMarkdown p, .stMarkdown li {
    color: inherit !important;
    background-color: inherit !important;
}
@media (prefers-color-scheme: dark) {
    .stMarkdown pre, .stMarkdown code {
        color: #ffffff !important;
        background-color: #22272e !important;
    }
}
@media (prefers-color-scheme: light) {
    .stMarkdown pre, .stMarkdown code {
        color: #000000 !important;
        background-color: #f9f9f9 !important;
    }
}
.stTextInput input, .stNumberInput input, div[data-baseweb="select"] > div {
    color: inherit !important;
}
div[data-baseweb="popover"] div {
    color: inherit !important;
    background-color: inherit !important;
}
.stFileUploader label, .uploadedFileName, .stCheckbox label {
    color: inherit !important;
}
section[data-testid="stSidebar"] div[data-baseweb="radio"] label {
    color: inherit !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "\U0001F4BC AI Job Salary Predictor",
    "\U0001F4C4 Resume Intelligence Analyzer",
    "\U0001F4CB Resume Summary Improver"
])

# --- Salary Predictor Page ---
if page == "\U0001F4BC AI Job Salary Predictor":
    st.title("\U0001F4BC AI Job Salary Predictor")
    st.markdown("Estimate salary based on job title, location, experience, and skills.")

    job_title = st.selectbox("\U0001F9FE Job Title", options=[
        "Data Scientist", "Machine Learning Engineer", "AI Engineer", "Business Analyst",
        "Data Analyst", "Data Engineer", "BI Developer", "MLOps Engineer",
        "AI Specialist", "Data Architect", "Big Data Engineer", "NLP Specialist",
        "Generative AI Engineer", "DataOps Engineer", "AI Ethics Engineer"])

    location = st.selectbox("\U0001F4CD Location", options=[
        "Bengaluru", "Hyderabad", "Pune", "Chennai", "Delhi / NCR",
        "Mumbai (All Areas)", "Noida", "Gurugram", "Remote", "Kolkata"])

    exp_min = st.number_input("\U0001F522 Minimum Experience", 0.0, 30.0, 2.0)
    exp_max = st.number_input("\U0001F522 Maximum Experience", 0.0, 30.0, 5.0)
    skills_input = st.text_input("\U0001F9E0 Skills (comma-separated)", "python, sql, mlops")

    def predict_salary(title, loc, min_exp, max_exp, skills):
        skills = [s.lower().strip() for s in skills]
        valid_skills = [s for s in skills if s in mlb.classes_]
        cat_df = pd.DataFrame([[title, loc]], columns=['standardizedTitle', 'standardizedLocation'])
        cat_encoded = encoder.transform(cat_df)
        cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(['standardizedTitle', 'standardizedLocation']))
        skill_arr = mlb.transform([valid_skills]) if valid_skills else np.zeros((1, len(mlb.classes_)))
        skill_df = pd.DataFrame(skill_arr, columns=mlb.classes_)
        num_df = pd.DataFrame([[min_exp, max_exp]], columns=["expMin", "expMax"])
        X_input = pd.concat([num_df, cat_df, skill_df], axis=1).reindex(columns=feature_order, fill_value=0)
        log_salary = model.predict(X_input)[0]
        return round(np.expm1(log_salary), 2)

    if st.button("\U0001F4B0 Predict Salary"):
        result = predict_salary(job_title, location, exp_min, exp_max, skills_input.split(","))
        st.success(f"\U0001F4B0 Estimated Salary: ₹{result} LPA")

# --- Resume Intelligence Analyzer Page ---
if page == "\U0001F4C4 Resume Intelligence Analyzer":
    st.title("\U0001F4C4 Resume Intelligence Analyzer")
    st.markdown("Analyze your resume to see how well it aligns with job descriptions and identify improvement areas.")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf", key="analyzer_page")
    job_role = st.selectbox("\U0001F3AF Target Job Role", [
        "Data Scientist", "Machine Learning Engineer", "AI Engineer", "Business Analyst",
        "Data Analyst", "Data Engineer", "MLOps Engineer", "BI Developer"], key="analyzer_job")

    if uploaded_file and job_role:
        with pdfplumber.open(uploaded_file) as pdf:
            resume_text = '\n'.join([page.extract_text() or "" for page in pdf.pages])[:MAX_TOKENS]

        with st.spinner("💬 Analyzing resume against job requirements..."):
            try:
                prompt = f"""
                Act as an ATS resume analyzer. Given the resume below and the target role '{job_role}', analyze the following:
                1. ATS match score (percentage)
                2. Key strengths in resume
                3. Weak areas and missing keywords
                4. Suggestions to improve alignment
                5. Top 3 certifications that would boost profile for this role

                Resume:
                {resume_text[:MAX_TOKENS]}
                """
                res = requests.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": GROQ_MODEL, "messages": [
                        {"role": "system", "content": "You are an expert resume reviewer and career advisor."},
                        {"role": "user", "content": prompt}
                    ], "temperature": 0.5})
                res.raise_for_status()
                analysis = res.json()['choices'][0]['message']['content']
            except Exception as e:
                st.error(f"❌ Failed to analyze resume: {str(e)}")
                st.stop()

        st.success("✅ Resume Analysis Completed")
        st.markdown("### 🧠 AI-Powered Resume Insights")
        st.markdown(f"<div style='background-color:var(--secondary-background-color, #f9f9f9); padding: 15px; border-radius: 10px;'><pre style='white-space: pre-wrap;'>{analysis}</pre></div>", unsafe_allow_html=True)
        with st.spinner("🔎 Fetching latest certifications for the role..."):
            try:
                search_query = f"top certifications for {job_role} site:coursera.org OR site:udemy.com"
                tavily_res = requests.post(
                    "https://api.tavily.com/search",
                    headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
                    json={"query": search_query, "num_results": 5}
                )
                tavily_res.raise_for_status()
                links = tavily_res.json().get("results", [])
                st.markdown("### 🎓 Suggested Certifications (Live Search Results)")
                for item in links:
                    st.markdown(f"- [{item.get('title', 'Certification Link')}]({item.get('url', '#')})")
            except Exception as e:
                st.warning(f"⚠️ Unable to fetch live certifications: {str(e)}")

# --- Resume Summary Improver Page ---

def extract_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])[:MAX_TOKENS]

def extract_summary_section(resume_text):
    summary_patterns = [
        r"(?:^|\n)(summary|professional summary|profile|about me|career summary)\s*[:\-]*\s*\n(.+?)(?:\n[A-Z][^\n]{0,40}\n|\Z)",
        r"(?:^|\n)(summary|professional summary|profile|about me|career summary)\s*[:\-]*\s*(.+?)(?:\n[A-Z][^\n]{0,40}\n|\Z)",
    ]
    for pattern in summary_patterns:
        match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)
        if match:
            summary = match.group(2).strip()
            return summary[:800]
    lines = resume_text.strip().split('\n')
    fallback = "\n".join(lines[:10])
    return fallback[:800]

if page == "\U0001F4CB Resume Summary Improver":
    st.title("\U0001F4CB Resume Summary Improver")
    st.markdown("Improve the summary section of your resume using LLM-based rewriting.")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf", key="summary_page")
    job_role = st.selectbox("\U0001F3AF Target Job Role", [
        "Data Scientist", "Machine Learning Engineer", "AI Engineer", "Business Analyst",
        "Data Analyst", "Data Engineer", "MLOps Engineer", "BI Developer"], key="summary_job")

    if uploaded_file and job_role:
        resume_text = extract_text(uploaded_file)
        summary_section = extract_summary_section(resume_text)

        st.markdown("#### ✂️ Extracted Summary Section")
        st.info(summary_section)

        with st.spinner("\U0001F4AC Rewriting Summary Section..."):
            try:
                prompt = f"""Rewrite the following resume summary in 4-6 lines to maximize ATS score for the role of {job_role}:\n\n{summary_section}"""
                res = requests.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": GROQ_MODEL, "messages": [
                        {"role": "system", "content": "You are a professional resume writer."},
                        {"role": "user", "content": prompt}
                    ], "temperature": 0.5})
                res.raise_for_status()
                new_summary = res.json()['choices'][0]['message']['content']
            except Exception as e:
                st.error(f"\u274c Failed to rewrite summary: {str(e)}")
                st.stop()

        st.success("\u2705 Rewritten Summary Generated")
        st.markdown("### \u270D\ufe0f Improved Summary")
        st.markdown(new_summary)

        if st.checkbox("\U0001F4C4 Generate and Download Updated Resume"):
            pdf_path = create_pdf(new_summary, resume_text)
            with open(pdf_path, "rb") as f:
                st.download_button("\u2B07\ufe0f Download Updated Resume", data=f, file_name="Updated_Resume_Summary.pdf", mime="application/pdf")
