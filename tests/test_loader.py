"""Tests for document loader module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.ingestion.loader import DocumentLoader


def test_loader_text_file(tmp_path):
    """Test loading and cleaning txt files."""
    loader = DocumentLoader()
    
    # Create temp text file
    temp_txt = tmp_path / "sample.txt"
    content = "Hello,   world! \n\n This is testing clean text.  Line 3."
    temp_txt.write_text(content, encoding="utf-8")
    
    result = loader.load(str(temp_txt))
    
    assert result["metadata"]["file_type"] == "txt"
    assert "sample.txt" in result["metadata"]["source"]
    assert "Hello, world! This is testing clean text. Line 3." in result["text"]
    assert len(result["pages"]) >= 1


@patch("backend.ingestion.loader.PdfReader")
def test_loader_pdf_file(mock_pdf_reader, tmp_path):
    """Test loading PDF files."""
    loader = DocumentLoader()
    
    # Mock PDF extractor
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF Page content here"
    
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf_reader.return_value = mock_pdf
    
    # Create dummy file path
    dummy_pdf = tmp_path / "sample.pdf"
    dummy_pdf.touch()
    
    result = loader.load(str(dummy_pdf))
    
    assert result["metadata"]["file_type"] == "pdf"
    assert "PDF Page content here" in result["text"]
    assert result["pages"][0]["page_number"] == 1


@patch("backend.ingestion.loader.DocxDocument")
def test_loader_docx_file(mock_docx_document, tmp_path):
    """Test loading DOCX files."""
    loader = DocumentLoader()
    
    # Mock Paragraphs
    mock_para = MagicMock()
    mock_para.text = "Docx paragraph content"
    
    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para]
    mock_docx_document.return_value = mock_doc
    
    dummy_docx = tmp_path / "sample.docx"
    dummy_docx.touch()
    
    result = loader.load(str(dummy_docx))
    
    assert result["metadata"]["file_type"] == "docx"
    assert "Docx paragraph content" in result["text"]


@patch("backend.ingestion.loader.Presentation")
def test_loader_pptx_file(mock_presentation, tmp_path):
    """Test loading PPTX files."""
    loader = DocumentLoader()
    
    # Mock Presentation Slides
    mock_shape = MagicMock()
    mock_shape.text = "PPTX Slide Content"
    mock_shape.has_table = False
    
    mock_slide = MagicMock()
    mock_slide.shapes = [mock_shape]
    
    mock_prs = MagicMock()
    mock_prs.slides = [mock_slide]
    mock_presentation.return_value = mock_prs
    
    dummy_pptx = tmp_path / "sample.pptx"
    dummy_pptx.touch()
    
    result = loader.load(str(dummy_pptx))
    
    assert result["metadata"]["file_type"] == "pptx"
    assert "PPTX Slide Content" in result["text"]
