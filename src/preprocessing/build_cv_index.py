import os
from src.db.session import SessionLocal
from src.db.models import CV, CVIndustryScore
from src.preprocessing.parse_cv import parse_cv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CV_DIR = os.path.join(PROJECT_ROOT, "DataSet", "cv")
PROMPT_PATH = os.path.join(PROJECT_ROOT, "src", "prompts", "cv_prompt.py")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    cv_prompt = f.read()

def main():
    session = SessionLocal()
    cv_files = [f for f in os.listdir(CV_DIR) if f.endswith(".docx")]

    for idx, cv_fname in enumerate(cv_files):
        cv_path = os.path.join(CV_DIR, cv_fname)
        print(f"Process CV {idx+1}/{len(cv_files)}: {cv_fname}")

        industry_scores, explanations = parse_cv(cv_path, cv_prompt)

        cv = CV(filename=cv_fname)
        session.add(cv)
        session.commit()

        for industry, score in industry_scores.items():
            explanation = explanations.get(industry, "")
            session.add(CVIndustryScore(
                cv_id=cv.id,
                industry=industry,
                score=score,
                explanation=explanation
            ))
    session.commit()
    session.close()
    print("DB done (cv)")

if __name__ == "__main__":
    main()