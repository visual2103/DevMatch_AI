import pytest
import sqlite3
from src.db.queries import get_job_id_by_filename, get_job_industry

@pytest.fixture
def db_connection():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE job_description (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL
        )
    """)
    cursor.execute(
        "INSERT INTO job_description (id, filename) VALUES (?, ?)",
        (1, "job1.docx")
    )

    cursor.execute("""
        CREATE TABLE job_industry_score (
            job_id INTEGER,
            industry TEXT,
            score REAL,
            FOREIGN KEY (job_id) REFERENCES job_description(id)
        )
    """)
    cursor.execute(
        "INSERT INTO job_industry_score (job_id, industry, score) VALUES (?, ?, ?)",
        (1, "IT", 0.95)
    )

    conn.commit()
    yield conn
    conn.close()


def test_get_job_id_by_filename(db_connection):
    job_id = get_job_id_by_filename(db_connection, "job1.docx")
    assert job_id == 1


def test_get_job_id_by_filename_not_found(db_connection):
    job_id = get_job_id_by_filename(db_connection, "job2.docx")
    assert job_id is None 


def test_get_job_industry(db_connection):
    industry = get_job_industry(db_connection, 1)
    assert industry == "IT"


def test_get_job_industry_not_found(db_connection):
    industry = get_job_industry(db_connection, 999)
    assert industry is None
