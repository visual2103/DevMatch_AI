import pytest
from unittest.mock import patch
from src.preprocessing.document_processor import DocumentProcessor

@pytest.fixture
def mock_docx():
    return ["Paragraph 1", "Paragraph 2"]

@patch("src.preprocessing.document_processor.Document")
@patch("src.preprocessing.document_processor.magic.Magic.from_file", return_value="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
@patch("src.preprocessing.document_processor.Path.stat", return_value=type('Stat', (), {'st_size': 1024}))
def test_read_docx_success(mock_stat, mock_magic, mock_document, mock_docx):
    mock_document.return_value.paragraphs = [type('Para', (), {'text': para}) for para in mock_docx]
    
    result = DocumentProcessor.read_docx("fake_path.docx")
    assert result == "Paragraph 1\nParagraph 2"

@patch("src.preprocessing.document_processor.Path.stat", return_value=type('Stat', (), {'st_size': DocumentProcessor.MAX_FILE_SIZE + 1}))
def test_read_docx_file_too_large(mock_stat):
    with pytest.raises(ValueError, match="File fake_path.docx is too large"):
        DocumentProcessor.read_docx("fake_path.docx")

@patch("src.preprocessing.document_processor.magic.Magic.from_file", return_value="application/pdf")
@patch("src.preprocessing.document_processor.Path.stat", return_value=type('Stat', (), {'st_size': 1024}))
def test_read_docx_invalid_file_type(mock_stat, mock_magic):
    with pytest.raises(ValueError, match="File must be a .docx file"):
        DocumentProcessor.read_docx("fake_path.docx")