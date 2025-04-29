import os
from src.db.session import SessionLocal
from src.db.models import JobDescription, JobIndustryScore
from src.preprocessing.parse_jd import parse_job_description

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JD_DIR = os.path.join(PROJECT_ROOT, "DataSet", "job_descriptions")
PROMPT_PATH = os.path.join(PROJECT_ROOT, "src", "prompts", "jd_prompt.py")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    jd_prompt = f.read()

def main():
    session = SessionLocal()
    job_files = [f for f in os.listdir(JD_DIR) if f.endswith(".docx")]

    for idx, job_fname in enumerate(job_files):
        job_path = os.path.join(JD_DIR, job_fname)
        print(f"Process Job Description {idx+1}/{len(job_files)}: {job_fname}")

        industry_scores, explanations = parse_job_description(job_path, jd_prompt)

        job = JobDescription(filename=job_fname, text=open(job_path, encoding="utf-8").read())
        session.add(job)
        session.commit()

        for industry, score in industry_scores.items():
            explanation = explanations.get(industry, "")
            session.add(JobIndustryScore(
                job_id=job.id,
                industry=industry,
                score=score,
                explanation=explanation
            ))
    session.commit()
    session.close()
    print("DB done (jd)")

if __name__ == "__main__":
    main()