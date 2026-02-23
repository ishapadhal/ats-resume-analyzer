import re
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from openai import OpenAI
import PyPDF2

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ---------- PDF TEXT EXTRACTION ----------

def input_pdf_setup(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# ---------- HUGGING FACE FUNCTION ----------

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def get_llm_response(prompt, resume_text, job_desc):

    final_prompt = f"""
You are a professional ATS system.

Job Description:
{job_desc}

Resume:
{resume_text}

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

        print("RAW GROQ RESPONSE:", response)

        return response.choices[0].message.content

    except Exception as e:
        print("GROQ ERROR:", e)
        return None
    

# ---------- STREAMLIT UI ----------

st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")

input_text = st.text_area("Job Description:")
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

submit1 = st.button("Tell Me About Resume")
submit3 = st.button("Percentage Match")

input_prompt1 = """
Review the resume against the job description.
Highlight strengths and weaknesses.
"""

input_prompt3 = """
Give:
1. Match Percentage
2. Missing Keywords
3. Final Thoughts
"""

if submit1:
    if uploaded_file:
        resume_text = input_pdf_setup(uploaded_file)
        response = get_llm_response(input_prompt3, resume_text, input_text)
        st.subheader("Response:")

        match = re.search(r'ATS Score:\s*(\d+)%', response)
        if match:
            score = int(match.group(1))
            st.progress(score)
            st.success(f"ATS Score: {score}%")
        else:
            st.warning("No ATS Score found in response.")
    else:
        st.warning("Please upload a resume.")

if submit3:
    if uploaded_file:
        resume_text = input_pdf_setup(uploaded_file)

        try:
            response = get_llm_response(input_prompt3, resume_text, input_text)

            if response:
                st.subheader("Response:")
                st.write(response)

                match = re.search(r'ATS Score:\s*(\d+)%', response)

                if match:
                    score = int(match.group(1))
                    st.progress(score)
                    st.success(f"ATS Score: {score}%")
                else:
                    st.warning("Could not extract ATS score from response.")
            else:
                st.error("No response received from model.")

        except Exception as e:
            st.error(f"API Error: {str(e)}")

    else:
        st.warning("Please upload a resume.")