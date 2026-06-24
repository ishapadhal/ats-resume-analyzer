import re
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import PyPDF2

# Load environment variables
load_dotenv()

# ---------- API KEY ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------- GROQ CLIENT ----------
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ---------- PDF TEXT EXTRACTION ----------
def input_pdf_setup(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text


# ---------- LLM FUNCTION ----------
def get_llm_response(prompt, resume_text, job_desc):

    final_prompt = f"""
You are an experienced ATS (Applicant Tracking System) expert.

Job Description:
{job_desc}

Resume:
{resume_text}

Instructions:
{prompt}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Groq API Error: {e}")
        return None


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="ATS Resume Expert")
st.header("📄 ATS Resume Tracking System")

input_text = st.text_area("Paste Job Description")

uploaded_file = st.file_uploader(
    "Upload Your Resume (PDF)",
    type=["pdf"]
)

submit1 = st.button("Tell Me About Resume")
submit3 = st.button("Percentage Match")


# ---------- PROMPTS ----------
input_prompt1 = """
Review the resume against the job description.

Provide:
1. Strengths
2. Weaknesses
3. Suggestions for improvement
"""

input_prompt3 = """
Analyze the resume against the job description.

Return EXACTLY in this format:

ATS Score: XX%

Missing Keywords:
- keyword1
- keyword2
- keyword3

Final Thoughts:
Short summary explaining overall fit.
"""


# ---------- TELL ME ABOUT RESUME ----------
if submit1:

    if uploaded_file is not None:

        resume_text = input_pdf_setup(uploaded_file)

        response = get_llm_response(
            input_prompt1,
            resume_text,
            input_text
        )

        if response:
            st.subheader("Response")
            st.write(response)

        else:
            st.error("No response received from model.")

    else:
        st.warning("Please upload a resume.")


# ---------- PERCENTAGE MATCH ----------
if submit3:

    if uploaded_file is not None:

        resume_text = input_pdf_setup(uploaded_file)

        response = get_llm_response(
            input_prompt3,
            resume_text,
            input_text
        )

        if response:

            st.subheader("ATS Analysis")
            st.write(response)

            # Extract score
            match = re.search(
                r'(?:ATS Score|Match Percentage)\s*:?\s*(\d+)%',
                response,
                re.IGNORECASE
            )

            if match:
                score = int(match.group(1))

                score = max(0, min(score, 100))

                st.subheader("ATS Score")
                st.progress(score)
                st.success(f"ATS Score: {score}%")

            else:
                st.warning("Could not extract ATS score.")

        else:
            st.error("No response received from model.")

    else:
        st.warning("Please upload a resume.")
