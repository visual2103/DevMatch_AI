# DevMatch AI

**DevMatch AI** is an intelligent talent allocation system that automatically matches developers to projects based on their skills, industry experience, and project requirements. Built with Python and leveraging machine learning and NLP techniques, it ensures optimal candidate selection and increases matchmaking accuracy.

---
<img width="1240" alt="image" src="https://github.com/user-attachments/assets/be5d264d-caaf-48df-a2aa-4d5e0a026b8f" />


## Features

- **Automated Matching**: Match CVs to job descriptions and vice versa, returning top candidates with match scores.  
- **Multi-Criteria Scoring**:  
  - Industry experience (10% weight)  
  - Predefined technical skills (30% weight)  
  - Semantic CV–job description alignment (60% weight)  
- **Interactive Interfaces**:  
  - Input a job description to get top 5 matching CVs  
  - Upload a CV to find the best-matching job  
- **Explanations**: For each match, an explanation detailing why the candidate was selected.  
- **Configurable Weights**: Adjust skill weightings per job requirement.  
- **Testing Suite**: Pytest-based tests for document processing, matching logic, and database queries.  

---

## Tech Stack

- **Python 3.11**  
- **FastAPI**: Web framework for building RESTful APIs  
- **SQLite**: Storage for CV and job metadata  
- **SQLAlchemy**: ORM for database operations  
- **OpenAI API**: NLP and semantic analysis  
- **Python-dotenv**: Load environment variables from `.env`  
- **Pytest**: Testing framework  

---

## Environment Variables

Create a `.env` file in the project root .


