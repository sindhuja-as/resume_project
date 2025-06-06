import streamlit as st
import fitz  # PyMuPDF
import re
from io import BytesIO
from docx import Document
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
from evaluation_utils import evaluate_resume
import logging
import openai
from dotenv import load_dotenv
import os

load_dotenv()
client = openai.OpenAI (
# Set the base URL and key for Groq
    api_key = os.getenv("OPENROUTER_API_KEY"),
    base_url = "https://openrouter.ai/api/v1"
)

st.set_page_config(page_title="AI Career Assistant", layout="wide")
page = st.sidebar.selectbox("Navigate", ["Resume Enhancer", "Interview Coach"])
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
expected_sections = {
    "Summary": ["summary", "profile", "objective"],
    "Work Experience": ["experience", "work history", "employment"],
    "Skills": ["skills", "technical skills"],
    "Education": ["education", "academics"],
    "Projects": ["projects", "personal projects"],
    "Certifications": ["certifications", "licenses"],
    "Contact": ["contact", "phone", "email"]
}

import logging

# Configure logging
logging.basicConfig(
    filename='ai_career_assist.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# ---------- Helper Functions ----------#
def mask_pii(text):
    ''' To mask Personally Identifiable information present in the input resume'''
    replacements = {}

    # Mask email addresses
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    for i, email in enumerate(emails):
        token = f"[EMAIL_{i}]"
        text = text.replace(email, token)
        replacements[token] = email

    # Mask phone numbers
    phones = re.findall(r"\+?\d[\d\s\-]{7,}\d", text)
    for i, phone in enumerate(phones):
        token = f"[PHONE_{i}]"
        text = text.replace(phone, token)
        replacements[token] = phone

    # Mask name-like patterns (very basic assumption)
    names = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b", text)
    for i, name in enumerate(set(names)):
        token = f"[NAME_{i}]"
        text = text.replace(name, token)
        replacements[token] = name

    return text, replacements

def unmask_pii(text, replacements):
    ''' To restore Personally Identifiable Information in the rewritten resume'''
    for token, original in replacements.items():
        text = text.replace(token, original)
    return text

def extract_text_from_pdf(pdf_file):
    ''' To convert data from pdf to txt format'''
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def clean_and_tokenize(text):
    '''Remove special characters and lowercase''' 
    text = re.sub(r'[^a-zA-Z ]', '', text).lower()
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return words

def extract_keywords(text, top_n=20):
    ''' To extract keywords from resume'''
    vectorizer = CountVectorizer(max_features=top_n, stop_words='english')
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

def get_resume_suggestions(resume_words, job_words):
    ''' To suggest keywords to improve resume'''
    missing_keywords = [word for word in job_words if word not in resume_words]
    suggestions = missing_keywords[:3] if missing_keywords else ["Your resume is well-aligned!"]
    return suggestions

def calculate_match_score(resume_words, job_keywords):
    ''' To calculate match score percentage between resume and job description'''
    if job_keywords.size == 0:
        return 0, []
    matched = [word for word in job_keywords if word in resume_words]
    score = (len(matched) / len(job_keywords)) * 100
    return round(score, 2), matched

def create_docx_from_text(text):
    ''' To convert txt to word format'''
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    return doc

def rewrite_resume(resume_text, job_description):
    ''' To rewrite the resume based on the job description'''
    masked_resume, replacements = mask_pii(resume_text)
    prompt = f"""
    You are an expert resume writer.

    Here is a job description:
    {job_description}

    Here is the original resume:
    {masked_resume}

    Rewrite the resume to better match the job, without lying or making up experience.
    Output a clean, professional version.
    """

    try:
        response = client.chat.completions.create(
    model="mistralai/mistral-7b-instruct",
    messages=[
        
        {"role": "user", "content": prompt}
    ]
)
        masked_output =  response.choices[0].message.content
        final_output = unmask_pii(masked_output, replacements)
        return final_output
    except Exception as e:
        return f"⚠️ Error from local model: {e}"

def check_section_presence(text: str) -> dict:
    ''' To check the presence of sections in the resume'''
    section_presence = {}
    text_lower = text.lower()
    for section, keywords in expected_sections.items():
        found = any(re.search(r"\b" + re.escape(kw) + r"\b", text_lower) for kw in keywords)
        section_presence[section] = "✅" if found else "❌"
    return section_presence

def section_score(section_presence: dict) -> int:
    ''' To calculate section completeness score'''
    total = len(section_presence)
    present = sum(1 for val in section_presence.values() if val == "✅")
    return int((present / total) * 100)

def generate_interview_questions(job_description, question_type, count=1):
    ''' To generate questions for interview coach module'''
    prompt = f"""
    You are an expert interview coach.

    Generate {5 * count} unique {question_type.lower()} for a candidate based on the job description below.

    JOB DESCRIPTION:
    {job_description}

    For each question, provide a high-quality model answer.

    Format:
    Q1: ...
    A1: ...
    Q2: ...
    A2: ...
    (and so on)
    """

    try:
        response = client.chat.completions.create(
    model="mistralai/mistral-7b-instruct",
    messages=[
        
        {"role": "user", "content": prompt}
    ]
)
        answer_text = response.choices[0].message.content

        questions, answers = [], []
        lines = answer_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith("q"):
                questions.append(line)
            elif line.lower().startswith("a"):
                answers.append(line)

        return questions, answers

    except Exception as e:
        return [f"⚠️ Error: {e}"], []



if page == "Interview Coach":
    ''' Module to generate interview questions based on job description'''
    #  Initialize session state variables
    if "question_count" not in st.session_state:
        st.session_state.question_count = 1
    if "questions" not in st.session_state:
        st.session_state.questions = []
        st.session_state.answers = []

    st.title("🤖 AI Career Assistant: Interview Coach")

    job_description = st.text_area("📝 Paste the Job Description", height=200)
    

    def load_questions():
        ''' To load interview questions'''
        questions, answers = generate_interview_questions(
            job_description,
            question_type,
            st.session_state.question_count
        )
        st.session_state.questions = questions
        st.session_state.answers = answers

    if job_description:
        question_type = st.radio("Question Type", ["Behavioral Questions", "Technical Questions", "Role-Specific Questions"])
        if st.button("Generate Questions"):
            logging.info("Interview questions generation triggered.")
            st.session_state.question_count = 1
            load_questions()

        if st.session_state.questions:
            st.subheader("🧠 Interview Questions & Answers")
            for q, a in zip(st.session_state.questions, st.session_state.answers):
                st.markdown(f"**{q}**")
                st.markdown(f"✅ {a}")
            if st.button("More Questions"):
                st.session_state.question_count += 1
                load_questions()
            
        interview_answer = st.text_area("🗣️ Generated Interview Answer")
        if st.button("Evaluate Answer"):
            logging.info("Answer evaluation initiated.")
            if job_description and interview_answer:
                scores = evaluate_resume(candidate=interview_answer, reference=job_description)
                st.subheader("📊 Evaluation Metrics")
                st.metric("ROUGE-1", f"{scores['ROUGE-1']:.2f}")
                st.metric("ROUGE-2", f"{scores['ROUGE-2']:.2f}")
                st.metric("ROUGE-L", f"{scores['ROUGE-L']:.2f}")
                # st.metric("BERTScore-F1", f"{scores['BERTScore-F1']:.2f}")
                st.metric("Cosine Similarity", f"{scores['Cosine Similarity']:.2f}")
            else:
                st.warning("Please provide both job description and answer.")

# ---------- Streamlit UI ----------

if page == "Resume Enhancer":
    ''' Module to rewrite resume based on job description'''
    st.title("🤖 AI Career Assistant: Resume Enhancer")
    st.subheader("Upload your resume & paste a job description")
    uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
    job_description = st.text_area("🧾 Paste Job Description")

    if uploaded_file and job_description:
        with st.spinner("Analyzing resume..."):
            file_type = uploaded_file.name.split(".")[-1].lower()
            if file_type == 'pdf':
                resume_text = extract_text_from_pdf(uploaded_file)
                resume_words = clean_and_tokenize(resume_text)
                job_keywords = extract_keywords(job_description)
                suggestions = get_resume_suggestions(resume_words, job_keywords)
                match_score, matched_keywords = calculate_match_score(resume_words, job_keywords)
                st.success("Analysis Complete ✅")                
                st.markdown(f"### 📊 Resume Match Score: `{match_score}%`")
                st.progress(int(match_score))
                st.markdown("### ✅ Matched Keywords:")
                st.write(", ".join(matched_keywords) if matched_keywords else "No keywords matched.")
                st.markdown("### ✨ Suggestions to Improve Resume:")
                for i, s in enumerate(suggestions, 1):
                    st.markdown(f"**{i}.** {s}")
              
            else:
                st.error("Unsupported file format.")

        st.markdown("---")
        with open("resume_template.docx", "rb") as file:
            
            st.download_button(
                label="📥 Download Resume Template",
                data=file,
                file_name="resume_template.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        
        if st.button("🛠 Rewrite Resume as .txt"):
            
            logging.info("Resume rewrite triggered.")
            rewritten = rewrite_resume(resume_text, job_description)
            logging.info("Rewritten resume generated successfully.")
            st.markdown("### ✍️ Rewritten Resume (from Mistral)")
            st.text_area("Preview", value=rewritten, height=400)
            st.download_button("📥 Download Rewritten Resume", data=rewritten, file_name="rewritten_resume.txt")
            match_score, matched_keywords = calculate_match_score(rewritten, job_keywords)
            st.success("Analysis Complete ✅")
            st.markdown(f"### 📊 Rewritten Resume Match Score: `{match_score}%`")
            st.progress(int(match_score))
            st.markdown("### ✅ Matched Keywords:")
            st.write(", ".join(matched_keywords) if matched_keywords else "No keywords matched.")
            section_presence = check_section_presence(rewritten)
            section_score_value = section_score(section_presence)
            st.subheader("📑 Section Presence Check")

            for section, status in section_presence.items():
                st.write(f"{section}: {status}")
                
            st.metric("Section Completeness", f"{section_score_value}%")

        if st.button("🔁 Rewrite Resume as .docx"):

            logging.info("Resume rewrite triggered.")         
            rewritten = rewrite_resume(resume_text, job_description)
            logging.info("Rewritten resume generated successfully.")
            new_doc = create_docx_from_text(rewritten)
            buffer = BytesIO()
            new_doc.save(buffer)
            buffer.seek(0)
            st.download_button("📥 Download Rewritten Resume (.docx)", buffer, file_name="rewritten_resume.docx")
            match_score, matched_keywords = calculate_match_score(rewritten, job_keywords)
            st.success("Analysis Complete ✅")
            st.markdown(f"### 📊 Rewritten Resume Match Score: `{match_score}%`")
            st.progress(int(match_score))
            st.markdown("### ✅ Matched Keywords:")
            st.write(", ".join(matched_keywords) if matched_keywords else "No keywords matched.")
            section_presence = check_section_presence(rewritten)
            section_score_value = section_score(section_presence)
            st.subheader("📑 Section Presence Check")

            for section, status in section_presence.items():
                st.write(f"{section}: {status}")

            st.metric("Section Completeness", f"{section_score_value}%")
            scoresip = evaluate_resume(job_description, resume_text)
            logging.info(f"Input resume metrics: {scoresip}")         
            st.subheader("📊 Evaluation Metrics of input resume")                    
            st.metric("ROUGE-1", f"{scoresip['ROUGE-1']:.2f}")
            st.metric("ROUGE-2", f"{scoresip['ROUGE-2']:.2f}")
            st.metric("ROUGE-L", f"{scoresip['ROUGE-L']:.2f}")
            # st.metric("BERTScore-F1", f"{scores['BERTScore-F1']:.2f}")
            st.metric("Cosine Similarity", f"{scoresip['Cosine Similarity']:.2f}")              
            scoresop = evaluate_resume(job_description, rewritten)
            logging.info(f"Output resume metrics: {scoresop}")
            st.subheader("📊 Evaluation Metrics of output resume")
            st.metric("ROUGE-1", f"{scoresop['ROUGE-1']:.2f}")
            st.metric("ROUGE-2", f"{scoresop['ROUGE-2']:.2f}")
            st.metric("ROUGE-L", f"{scoresop['ROUGE-L']:.2f}")
            # st.metric("BERTScore-F1", f"{scores['BERTScore-F1']:.2f}")
            st.metric("Cosine Similarity", f"{scoresop['Cosine Similarity']:.2f}")        
            
        # if st.button("Evaluate Resume"):
        #     ''' To generate performance metrics for both input and rewritten resumes'''
        #     if "rewritten" in st.session_state:
                
            
        
    

