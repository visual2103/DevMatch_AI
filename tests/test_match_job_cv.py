import pytest
import numpy as np
from unittest.mock import patch
from pages.match_job_to_cvs import (
    get_keyword_matching_scores,
    get_cv_keyword_matching_scores,
    get_skills_matched_for_cv,
    get_matching_scores_between_cvs_and_job_description,
    filter_cvs_by_industry
)
from utils.explanation import generate_explanation_with_llm_job_to_cv

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
    cv_text = CV_ADRIAN_TEXT
    score = get_keyword_matching_scores(custom_skills, cv_text)
    assert score == 1.0  

def test_get_keyword_matching_scores_no_matches():
    custom_skills = [("C#", 50), ("Ruby", 50)]
    cv_text = CV_ADRIAN_TEXT
    score = get_keyword_matching_scores(custom_skills, cv_text)
    assert score == 0.0  

def test_get_keyword_matching_scores_empty_text():
    custom_skills = [("Python", 50), ("Java", 50)]
    cv_text = ""
    score = get_keyword_matching_scores(custom_skills, cv_text)
    assert score == 0.0  

def test_get_cv_keyword_matching_scores_real_data():
    custom_skills = [("Python", 30), ("Java", 30), ("TensorFlow", 20), ("AWS", 20)]
    cv_texts = [
        CV_ADRIAN_TEXT,  
        "I have experience in Ruby programming.",  
        "I know Java and AWS."  
    ]
    scores = get_cv_keyword_matching_scores(custom_skills, cv_texts)
    expected_scores = np.array([1.0, 0.0, 0.5])  # 100% 0% 50% (Java + AWS)
    assert scores.tolist() == expected_scores.tolist()

def test_get_cv_keyword_matching_scores_empty_list():
    custom_skills = [("Python", 50), ("Java", 50)]
    cv_texts = []
    scores = get_cv_keyword_matching_scores(custom_skills, cv_texts)
    assert len(scores) == 0  

def test_get_skills_matched_for_cv_real_data():
    custom_skills = [("Python", 30), ("Java", 30), ("TensorFlow", 20), ("AWS", 20)]
    cv_text = CV_ADRIAN_TEXT
    skill_matches = get_skills_matched_for_cv(custom_skills, cv_text)
    expected_matches = {
        "Python": (True, 30),
        "Java": (True, 30),
        "TensorFlow": (True, 20),
        "AWS": (True, 20)
    }
    assert skill_matches == expected_matches

def test_get_skills_matched_for_cv_no_matches():
    custom_skills = [("Ruby", 50), ("C#", 50)]
    cv_text = CV_ADRIAN_TEXT
    skill_matches = get_skills_matched_for_cv(custom_skills, cv_text)
    expected_matches = {
        "Ruby": (False, 50),
        "C#": (False, 50)
    }
    assert skill_matches == expected_matches


@patch("pages.match_job_to_cvs.nlp")
def test_get_matching_scores_between_cvs_and_job_description_real_data(mock_nlp):
    mock_doc = type('Doc', (), {'similarity': lambda x: 0.9})
    mock_nlp.return_value = mock_doc

    cv_texts = [CV_ADRIAN_TEXT, "Unrelated CV text"]
    job_text = JOB_TEXT
    progress_bar = type('Progress', (), {'progress': lambda x: None})
    status_text = type('Status', (), {'empty': lambda: None, 'text': lambda x: None})

    scores = get_matching_scores_between_cvs_and_job_description(cv_texts, job_text, progress_bar, status_text)
    assert len(scores) == 2  
    assert all(0 <= score <= 1 for score in scores)  
    assert scores[0] == pytest.approx(0.9, 0.1)  

@patch("pages.match_job_to_cvs.nlp")
def test_get_matching_scores_between_cvs_and_job_description_empty_cvs(mock_nlp):
    mock_doc = type('Doc', (), {'similarity': lambda x: 0.9})
    mock_nlp.return_value = mock_doc

    cv_texts = []
    job_text = JOB_TEXT
    progress_bar = type('Progress', (), {'progress': lambda x: None})
    status_text = type('Status', (), {'empty': lambda: None, 'text': lambda x: None})

    scores = get_matching_scores_between_cvs_and_job_description(cv_texts, job_text, progress_bar, status_text)
    assert len(scores) == 0  

@patch("pages.match_job_to_cvs.load_docx_from_folder")
@patch("pages.match_job_to_cvs.get_cv_industry_scores")
def test_filter_cvs_by_industry_real_data(mock_get_cv_industry_scores, mock_load_docx_from_folder):
    mock_load_docx_from_folder.return_value = (
        [CV_ADRIAN_TEXT, "Unrelated CV text"],
        ["cv1.docx", "cv2.docx"],
        None
    )
    
    mock_get_cv_industry_scores.return_value = np.array([0.95, 0.0])  #0.95 for IT
    
    db_path = "dummy_path"
    cv_folder = "dummy_folder"
    selected_industry = "IT"
    
    cv_texts, cv_filenames, industry_scores = filter_cvs_by_industry(db_path, cv_folder, selected_industry)
    
    assert len(cv_texts) == 1  
    assert cv_texts == [CV_ADRIAN_TEXT]
    assert cv_filenames == ["cv1.docx"]
    assert industry_scores.tolist() == [0.95]

def test_final_score_calculation_real_data():
    industry_scores = np.array([0.95, 0.5])  
    skills_scores = np.array([1.0, 0.0])  
    semantic_scores = np.array([0.9, 0.3])  
    
    final_scores = (
        0.1 * industry_scores +
        0.3 * skills_scores +
        0.6 * semantic_scores
    )
    
    expected_scores = np.array([
        0.1 * 0.95 + 0.3 * 1.0 + 0.6 * 0.9,  # 0.095 + 0.3 + 0.54 = 0.935
        0.1 * 0.5 + 0.3 * 0.0 + 0.6 * 0.3   # 0.05 + 0.0 + 0.18 = 0.23
    ])
    
    assert final_scores.tolist() == pytest.approx(expected_scores.tolist(), 0.01)

@patch("pages.match_job_to_cvs.Document")
@patch("pages.match_job_to_cvs.get_job_industry", return_value="IT")
@patch("pages.match_job_to_cvs.get_job_id_by_filename", return_value=1)
@patch("pages.match_job_to_cvs.filter_cvs_by_industry")
@patch("pages.match_job_to_cvs.get_matching_scores_between_cvs_and_job_description")
@patch("pages.match_job_to_cvs.get_cv_keyword_matching_scores")
@patch("pages.match_job_to_cvs.generate_explanation_with_llm_job_to_cv")
def test_full_flow_real_data(
    mock_generate_explanation,
    mock_get_cv_keyword_matching_scores,
    mock_get_matching_scores,
    mock_filter_cvs_by_industry,
    mock_get_job_id_by_filename,
    mock_get_job_industry,
    mock_document
):
    mock_paragraph = type('Para', (), {'text': JOB_TEXT})
    mock_document.return_value.paragraphs = [mock_paragraph]
    
    mock_filter_cvs_by_industry.return_value = (
        [CV_ADRIAN_TEXT, "Unrelated CV text"],
        ["cv1.docx", "cv2.docx"],
        np.array([0.95, 0.5])
    )
    
    mock_get_matching_scores.return_value = np.array([0.9, 0.3])
    
    mock_get_cv_keyword_matching_scores.return_value = np.array([1.0, 0.0])
    
    mock_generate_explanation.return_value = "Adrian's CV matches due to his skills in Python, Java, TensorFlow, and AWS."
    
    custom_skills = [("Python", 30), ("Java", 30), ("TensorFlow", 20), ("AWS", 20)]
    job_text = JOB_TEXT
    cv_texts = [CV_ADRIAN_TEXT, "Unrelated CV text"]
    industry_scores = np.array([0.95, 0.5])
    
    skills_scores = mock_get_cv_keyword_matching_scores(custom_skills, cv_texts)
    
    progress_bar = type('Progress', (), {'progress': lambda x: None})
    status_text = type('Status', (), {'empty': lambda: None})
    semantic_scores = mock_get_matching_scores(cv_texts, job_text, progress_bar, status_text)
    
    final_scores = (
        0.1 * industry_scores +
        0.3 * skills_scores +
        0.6 * semantic_scores
    )
    
    best_cv_idx = np.argmax(final_scores)
    assert best_cv_idx == 0  
    
    explanation = mock_generate_explanation.return_value
    assert explanation == "Adrian's CV matches due to his skills in Python, Java, TensorFlow, and AWS."