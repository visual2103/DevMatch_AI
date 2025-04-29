from src.preprocessing.openai_client import analyze_text
from src.preprocessing.document_processor import DocumentProcessor
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

def parse_cv(file_path: str, prompt: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    try:
        text = DocumentProcessor.read_docx(file_path)
        if not text:
            raise ValueError(f"Could not extract text from CV: {file_path}")

        return analyze_text(text, prompt)

    except Exception as e:
        logger.error(f"Error processing CV {file_path}: {str(e)}")
        raise
from src.preprocessing.openai_client import analyze_text
from src.preprocessing.document_processor import DocumentProcessor
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

def parse_cv(file_path: str, prompt: str) -> Tuple[Dict[str, int], Dict[str, str]]:

    try:
        text = DocumentProcessor.read_docx(file_path)
        if not text:
            raise ValueError(f"Could not extract text from CV: {file_path}")

        return analyze_text(text, prompt)

    except Exception as e:
        logger.error(f"error processing CV {file_path}: {str(e)}")
        raise