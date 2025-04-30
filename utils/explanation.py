from openai import OpenAI
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

openai_key = st.secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("OpenAI API key not found. Define it in secrets.toml or in .env")
    st.stop()

client = OpenAI(api_key=openai_key)



def generate_explanation_with_llm_job_to_cv(cv_filename, domain_score, skills_score, matching_score, matched_skills, domain_selected):
    explanation_prompt = f"""
You are an AI system helping in matching developers to jobs.

A CV ({cv_filename}) was selected based on these evaluation scores:
- Industry knowledge match ({domain_selected} domain): {domain_score * 100:.2f}%
- Technical Skills match: {skills_score * 100:.2f}%
- Semantic Job-CV Matching: {matching_score * 100:.2f}%

The candidate demonstrated the following relevant technical skills: {', '.join([skill for skill, (matched, _) in matched_skills.items() if matched])}.

Please generate a clear, concise explanation (around 100-150 words) for why this CV is the best match for the job.
Focus on:
- How industry experience helped
- How technical skills aligned
- How overall experience matches the job description
Use a professional and formal tone.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": explanation_prompt}],
        temperature=0.4,
    )

    explanation = response.choices[0].message.content
    return explanation


def generate_explanation_with_llm_cv_to_job(cv_filename, domain_score, skills_score, matching_score, matched_skills, domain_selected):
    explanation_prompt = f"""
You are an AI assistant that helps developers find the most suitable job based on their experience.

A CV ({cv_filename}) was matched to a job based on these evaluation scores:
- Domain relevance ({domain_selected} domain): {domain_score * 100:.2f}%
- Technical Skills match: {skills_score * 100:.2f}%
- Overall semantic alignment with job description: {matching_score * 100:.2f}%

The candidate demonstrated proficiency in the following skills: {', '.join(matched_skills)}.

Please generate a clear, professional explanation (around 100-150 words) that explains why this job is the best fit for the candidate.
Focus on:
- How the candidate's experience in the domain supports this job
- How the technical skills align with the job requirements
- How the overall background and expertise make the candidate suitable for this role
Use a formal and convincing tone, as if preparing a recommendation report.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": explanation_prompt}],
        temperature=0.4,
    )

    explanation = response.choices[0].message.content
    return explanation