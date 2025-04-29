from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, Index, CheckConstraint
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class CV(Base):
    __tablename__ = 'cv'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    industry_scores = relationship("CVIndustryScore", back_populates="cv", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_cv_filename', 'filename'),
        Index('idx_cv_created', 'created_at'),
    )

class CVIndustryScore(Base):
    __tablename__ = 'cv_industry_score'
    
    id = Column(Integer, primary_key=True)
    cv_id = Column(Integer, ForeignKey('cv.id', ondelete='CASCADE'), nullable=False)
    industry = Column(String, nullable=False)
    score = Column(Integer, nullable=False)
    explanation = Column(Text)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    cv = relationship("CV", back_populates="industry_scores")
    
    __table_args__ = (
        CheckConstraint('score >= 0 AND score <= 100', name='check_score_range'),
        Index('idx_cv_industry', 'cv_id', 'industry'),
    )

class JobDescription(Base):
    __tablename__ = 'job_description'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    industries = relationship("JobIndustryScore", back_populates="job_description", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_job_filename', 'filename'),
        Index('idx_job_created', 'created_at'),
    )

class JobIndustryScore(Base):
    __tablename__ = 'job_industry_score'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey('job_description.id', ondelete='CASCADE'), nullable=False)
    industry = Column(String, nullable=False)
    score = Column(Integer, nullable=False)
    explanation = Column(Text)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    job_description = relationship("JobDescription", back_populates="industries")
    
    __table_args__ = (
        CheckConstraint('score >= 0 AND score <= 100', name='check_score_range'),
        Index('idx_job_industry', 'job_id', 'industry'),
    )