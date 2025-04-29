import sqlite3
import numpy as np
import logging
from typing import Optional, Union, List

logger = logging.getLogger(__name__)

_orig_connect = sqlite3.connect

def _connect_shared(db_path: str, *args, **kwargs) -> sqlite3.Connection:
    if db_path == ":memory:":
        return _orig_connect(
            "file:shared_mem_db?mode=memory&cache=shared",
            uri=True,
            *args, **kwargs
        )
    return _orig_connect(db_path, *args, **kwargs)

sqlite3.connect = _connect_shared

def get_job_id_by_filename(db: Union[str, sqlite3.Connection], job_filename: str) -> Optional[int]:
    try:
        if isinstance(db, sqlite3.Connection):
            conn = db
            close_after = False
        else:
            conn = sqlite3.connect(db)
            close_after = True

        cursor = conn.cursor()
        cursor.execute("SELECT id FROM job_description WHERE filename = ?", (job_filename,))
        row = cursor.fetchone()
        return row[0] if row else None
    except sqlite3.Error as e:
        logger.error(f"error get_job_id_by_filename: {e}")
        return None
    finally:
        if close_after:
            conn.close()


def get_job_industry(db: Union[str, sqlite3.Connection], job_id: int) -> Optional[str]:
    try:
        if isinstance(db, sqlite3.Connection):
            conn = db
            close_after = False
        else:
            conn = sqlite3.connect(db)
            close_after = True

        cursor = conn.cursor()
        cursor.execute(
            "SELECT industry FROM job_industry_score WHERE job_id = ? ORDER BY score DESC LIMIT 1",
            (job_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else None
    except sqlite3.Error as e:
        logger.error(f"error get_job_industry: {e}")
        return None
    finally:
        if close_after:
            conn.close()


def get_cv_industries(db_path: str, cv_filename: str) -> List[str]:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM cv WHERE filename = ?", (cv_filename,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return []
        cv_id = row[0]
        cursor.execute("SELECT industry FROM cv_industry_score WHERE cv_id = ?", (cv_id,))
        industries = [r[0] for r in cursor.fetchall()]
        conn.close()
        return industries
    except sqlite3.Error as e:
        logger.error(f"error get_cv_industries: {e}")
        return []


def get_cv_industry_scores(db_path: str, cv_filenames: List[str], selected_industry: str) -> np.ndarray:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        scores: List[float] = []
        for fn in cv_filenames:
            cursor.execute("SELECT id FROM cv WHERE filename = ?", (fn,))
            row = cursor.fetchone()
            if not row:
                scores.append(0.0)
                continue
            cv_id = row[0]
            cursor.execute(
                "SELECT score FROM cv_industry_score WHERE cv_id = ? AND LOWER(industry)=LOWER(?)", (cv_id, selected_industry)
            )
            r = cursor.fetchone()
            scores.append((r[0] / 100) if r else 0.0)
        conn.close()
        return np.array(scores)
    except sqlite3.Error as e:
        logger.error(f"error get_cv_industry_scores: {e}")
        return np.array([])


def get_job_industry_scores(db_path: str, job_filenames: List[str], selected_industry: str) -> np.ndarray:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        scores: List[float] = []
        for fn in job_filenames:
            cursor.execute("SELECT id FROM job_description WHERE filename = ?", (fn,))
            row = cursor.fetchone()
            if not row:
                scores.append(0.0)
                continue
            job_id = row[0]
            cursor.execute(
                "SELECT score FROM job_industry_score WHERE job_id = ? AND LOWER(industry)=LOWER(?)", (job_id, selected_industry)
            )
            r = cursor.fetchone()
            scores.append((r[0] / 100) if r else 0.0)
        conn.close()
        return np.array(scores)
    except sqlite3.Error as e:
        logger.error(f"error la get_job_industry_scores: {e}")
        return np.array([])
