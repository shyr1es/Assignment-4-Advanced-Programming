import pytest
from src.constitution_parser import extract_text_from_pdf

def test_extract_text_from_pdf():
    text = extract_text_from_pdf("C:/Users/johan/OneDrive/Desktop/asik3_part1/project/pdf_kz_const.pdf")
    assert len(text) > 0, "Text extraction failed"

def test_query_chromadb():
    query_text = "What does Article 1 of the Constitution say?"
    results = query_chromadb(query_text)
    assert results is not None, "ChromaDB query failed"
