import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH      = os.path.join(PROJECT_ROOT, "data", "cvs_metadata.sqlite")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine       = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base         = declarative_base()