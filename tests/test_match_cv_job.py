import pytest
import numpy as np
from unittest.mock import patch
from pages.match_cv_to_job import (
    get_keyword_matching_scores,
    get_matching_scores_between_cv_and_job_descriptions,
    filter_jobs_by_industry
)
from utils.explanation import generate_explanation_with_llm_cv_to_job

CV_ADRIAN_TEXT = """
Adrian Dumitrescu Cojocaru
Technical Skills
- JavaScript, ReactJS, Node.js- Java, Spring Boot, SQL- AWS, Docker, Kubernetes- HTML, CSS, Bootstrap, TypeScript- Python, Django, PostgreSQL- AngularJS, Git, REST APIs- Google Cloud, TensorFlow, PyTorch
Foreign Languages
- English: C1- Spanish: B2- French: B1
Education
- University Name: University Politehnica of Bucharest- Program Duration: 4 years- Master degree Name: University Politehnica of Bucharest- Program Duration: 2 years
Certifications
- AWS Certified Solutions Architect – Professional- Google Professional Cloud Architect- Certified Kubernetes Administrator (CKA)
Project Experience
1. Real-Time Data Analytics Platform   Led the development of a real-time data analytics platform using Java, Spring Boot, and SQL to process and analyze large datasets efficiently. Implemented REST APIs to facilitate seamless data integration and retrieval, enhancing the platform's interoperability with external systems. Deployed the solution on AWS using Docker and Kubernetes to ensure scalability and high availability, resulting in a 50% reduction in data processing time. Technologies and tools used: Java, Spring Boot, SQL, REST APIs, AWS, Docker, Kubernetes.
2. AI-Powered Recommendation System   Spearheaded the creation of an AI-powered recommendation system leveraging Python, Django, and PostgreSQL to deliver personalized content to users. Utilized TensorFlow and PyTorch for building and training machine learning models, achieving a 20% increase in user engagement. Deployed the system on Google Cloud, ensuring robust performance and scalability to handle millions of requests daily. Technologies and tools used: Python, Django, PostgreSQL, TensorFlow, PyTorch, Google Cloud.
3. Responsive Web Application for Online Learning   Directed the development of a responsive web application for an online learning platform using ReactJS, Node.js, and TypeScript. Integrated interactive features and a dynamic user interface with HTML, CSS, and Bootstrap to enhance user experience and accessibility. Implemented CI/CD pipelines with Git to streamline the development process and ensure rapid deployment of new features. Technologies and tools used: ReactJS, Node.js, TypeScript, HTML, CSS, Bootstrap, Git.
"""

JOB_TEXT = """
Job Title:
Tech Lead Machine Learning Engineer
Company Overview:
InnovateTech Solutions is a leading provider of cutting-edge technology solutions, dedicated to transforming industries through advanced machine learning and artificial intelligence. Our mission is to empower businesses with intelligent systems that drive efficiency, innovation, and growth. We pride ourselves on fostering a collaborative and inclusive work environment where creativity and innovation thrive.
Key Responsibilities:
- Lead the design, development, and deployment of machine learning models and algorithms to solve complex business problems.- Collaborate with cross-functional teams to integrate machine learning solutions into existing systems and workflows.- Mentor and guide a team of machine learning engineers, providing technical direction and support.- Conduct research to stay abreast of the latest advancements in machine learning and AI, and apply these insights to improve existing solutions.- Oversee the end-to-end lifecycle of machine learning projects, from data collection and preprocessing to model evaluation and deployment.- Ensure the scalability, reliability, and performance of machine learning models in production environments.- Communicate complex technical concepts to non-technical stakeholders, ensuring alignment with business objectives.
Required Qualifications:
- Bachelor’s or Master’s degree in Computer Science, Data Science, Machine Learning, or a related field.- 5+ years of experience in machine learning engineering, with at least 2 years in a leadership role.- Strong proficiency in programming languages such as Python, R, or Java, and experience with machine learning frameworks like TensorFlow, PyTorch, or scikit-learn.- Proven track record of deploying machine learning models in production environments.- Solid understanding of data structures, algorithms, and statistical methods.- Excellent problem-solving skills and the ability to work independently and collaboratively.
Preferred Skills:
- Ph.D. in a related field is a plus.- Experience with cloud platforms such as AWS, Google Cloud, or Azure.- Familiarity with big data technologies like Hadoop, Spark, or Kafka.- Strong background in natural language processing, computer vision, or reinforcement learning.- Experience with DevOps practices and tools for continuous integration and deployment.
"""

def test_get_keyword_matching_scores_real_data():
    custom_skills = [("Python", 30), ("Java", 30), ("TensorFlow", 20), ("AWS", 20)]
    job_text = JOB_TEXT
    score = get_keyword_matching_scores(custom_skills, job_text)
    assert score == 1.0  

def test_get_keyword_matching_scores_no_matches():
    custom_skills = [("Ruby", 50), ("C#", 50)]
    job_text = JOB_TEXT
    score = get_keyword_matching_scores(custom_skills, job_text)
    assert score == 0.0  

@patch("pages.match_cv_to_job.nlp")
def test_get_matching_scores_between_cv_and_job_descriptions_real_data(mock_nlp):
    mock_doc = type('Doc', (), {'similarity': lambda x: 0.9})
    mock_nlp.return_value = mock_doc

    cv_text = CV_ADRIAN_TEXT
    job_texts = [JOB_TEXT, "Unrelated Job text"]
    progress_bar = type('Progress', (), {'progress': lambda x: None})
    # Adaugă metoda `text` la obiectul mock Status
    status_text = type('Status', (), {'empty': lambda: None, 'text': lambda x: None})

    scores = get_matching_scores_between_cv_and_job_descriptions(cv_text, job_texts, progress_bar, status_text)
    assert len(scores) == 2  
    assert all(0 <= score <= 1 for score in scores)  
    assert scores[0] == pytest.approx(0.9, 0.1) 
@patch("pages.match_cv_to_job.nlp")
def test_get_matching_scores_between_cv_and_job_descriptions_empty_jobs(mock_nlp):
    mock_doc = type('Doc', (), {'similarity': lambda x: 0.9})
    mock_nlp.return_value = mock_doc

    cv_text = CV_ADRIAN_TEXT
    job_texts = []
    progress_bar = type('Progress', (), {'progress': lambda x: None})
    status_text = type('Status', (), {'empty': lambda: None, 'text': lambda x: None})

    scores = get_matching_scores_between_cv_and_job_descriptions(cv_text, job_texts, progress_bar, status_text)
    assert len(scores) == 0 

@patch("pages.match_cv_to_job.load_docx_from_folder")
@patch("pages.match_cv_to_job.get_job_industry_scores")
def test_filter_jobs_by_industry_real_data(mock_get_job_industry_scores, mock_load_docx_from_folder):
    mock_load_docx_from_folder.return_value = (
        [JOB_TEXT, "Unrelated Job text"],
        ["job1.docx", "job2.docx"],
        None
    )
    
    mock_get_job_industry_scores.return_value = np.array([0.95, 0.0])  
    
    db_path = "dummy_path"
    jd_folder = "dummy_folder"
    selected_industry = "IT"
    
    job_texts, job_filenames, industry_scores = filter_jobs_by_industry(db_path, jd_folder, selected_industry)
    
    assert len(job_texts) == 1  
    assert job_texts == [JOB_TEXT]
    assert job_filenames == ["job1.docx"]
    assert industry_scores.tolist() == [0.95]

def test_final_score_calculation_real_data():
    industry_scores = np.array([0.95, 0.5]) 
    skill_scores_norm = np.array([1.0, 0.0]) 
    cv_job_matching_scores = np.array([0.9, 0.3])  
    
    final_scores = (
        0.1 * industry_scores +
        0.3 * skill_scores_norm +
        0.6 * cv_job_matching_scores
    )
    
    expected_scores = np.array([
        0.1 * 0.95 + 0.3 * 1.0 + 0.6 * 0.9,  # 0.095 + 0.3 + 0.54 = 0.935
        0.1 * 0.5 + 0.3 * 0.0 + 0.6 * 0.3   # 0.05 + 0.0 + 0.18 = 0.23
    ])
    
    assert final_scores.tolist() == pytest.approx(expected_scores.tolist(), 0.01)

@patch("pages.match_cv_to_job.Document")
@patch("pages.match_cv_to_job.get_cv_industries", return_value=["IT"])
@patch("pages.match_cv_to_job.filter_jobs_by_industry")
@patch("pages.match_cv_to_job.get_matching_scores_between_cv_and_job_descriptions")
@patch("pages.match_cv_to_job.get_keyword_matching_scores")
@patch("pages.match_cv_to_job.generate_explanation_with_llm_cv_to_job")
def test_full_flow_real_data(
    mock_generate_explanation,
    mock_get_keyword_matching_scores,
    mock_get_matching_scores,
    mock_filter_jobs_by_industry,
    mock_get_cv_industries,
    mock_document
):
    mock_paragraph = type('Para', (), {'text': CV_ADRIAN_TEXT})
    mock_document.return_value.paragraphs = [mock_paragraph]
    
    mock_filter_jobs_by_industry.return_value = (
        [JOB_TEXT, "Unrelated Job text"],
        ["job1.docx", "job2.docx"],
        np.array([0.95, 0.5])
    )
    
    mock_get_matching_scores.return_value = np.array([0.9, 0.3])
    
    mock_get_keyword_matching_scores.side_effect = [1.0, 0.0]
    
    mock_generate_explanation.return_value = "Adrian's CV matches due to his skills in Python, Java, TensorFlow, and AWS."
    
    custom_skills = [("Python", 30), ("Java", 30), ("TensorFlow", 20), ("AWS", 20)]
    cv_text = CV_ADRIAN_TEXT
    job_texts = [JOB_TEXT, "Unrelated Job text"]
    industry_scores = np.array([0.95, 0.5])
    
    skill_scores = np.array([
        mock_get_keyword_matching_scores(custom_skills, job_text)
        for job_text in job_texts
    ])
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    skill_scores_norm = scaler.fit_transform(skill_scores.reshape(-1, 1)).flatten()
    
    progress_bar = type('Progress', (), {'progress': lambda x: None})
    status_text = type('Status', (), {'empty': lambda: None})
    cv_job_matching_scores = mock_get_matching_scores(cv_text, job_texts, progress_bar, status_text)
    
    final_scores = (
        0.1 * industry_scores +
        0.3 * skill_scores_norm +
        0.6 * cv_job_matching_scores
    )
    
    best_job_idx = np.argmax(final_scores)
    assert best_job_idx == 0  
    
    explanation = mock_generate_explanation.return_value
    assert explanation == "Adrian's CV matches due to his skills in Python, Java, TensorFlow, and AWS."