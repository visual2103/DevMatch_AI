from docx import Document
import logging
from typing import Optional
from pathlib import Path
import PyPDF2
import magic

logger = logging.getLogger(__name__)

class DocumentProcessor:    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    @staticmethod
    def read_docx(file_path: str) -> Optional[str]:
        try:
            path = Path(file_path)
            
            if path.stat().st_size > DocumentProcessor.MAX_FILE_SIZE:
                raise ValueError(f"File {file_path} is too large. Maximum size is {DocumentProcessor.MAX_FILE_SIZE/1024/1024}MB")
            
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(str(path))
            if file_type != 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                raise ValueError(f"File must be a .docx file, got {file_type}")
                
            doc = Document(file_path)
            full_text = [para.text for para in doc.paragraphs if para.text.strip()]
            
            if not full_text:
                logger.warning(f"No text content found in {file_path}")
                return None
                
            return "\n".join(full_text)
            
        except Exception as e:
            logger.error(f"Error reading document {file_path}: {str(e)}")
            raise
            
    @staticmethod
    def read_pdf(file_path: str) -> Optional[str]:
        try:
            path = Path(file_path)
            
            # Check file size
            if path.stat().st_size > DocumentProcessor.MAX_FILE_SIZE:
                raise ValueError(f"File {file_path} is too large. Maximum size is {DocumentProcessor.MAX_FILE_SIZE/1024/1024}MB")
            
            # Check file type
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(str(path))
            if file_type != 'application/pdf':
                raise ValueError(f"File must be a PDF file, got {file_type}")
                
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                    
            if not text.strip():
                logger.warning(f"no text content found in {file_path}")
                return None
                
            return text.strip()
            
        except Exception as e:
            logger.error(f"error reading PDF {file_path}: {str(e)}")
            raise