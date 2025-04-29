from docx import Document
from src.preprocessing.openai_client import analyze_text
from src.preprocessing.document_processor import DocumentProcessor
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    full_text = [para.text for para in doc.paragraphs]
    return "\n".join(full_text)

def parse_job_description(file_path: str, prompt: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    try:
        text = DocumentProcessor.read_docx(file_path)
        if not text:
            raise ValueError(f"Could not extract text from job description: {file_path}")
            
        return analyze_text(text, prompt)
        
    except Exception as e:
        logger.error(f"Error processing job description {file_path}: {str(e)}")
        raise