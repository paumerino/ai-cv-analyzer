"""Unit tests for the PDF parser and ATS checks."""

import pymupdf
import pytest

from src.parser import parse_cv_pdf


def _build_pdf_bytes(pages_text: list[str]) -> bytes:
    """Create a synthetic in-memory PDF for deterministic testing."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        rect = pymupdf.Rect(50, 50, 545, 790)
        rc = page.insert_textbox(rect, text, fontsize=9)
        if text.strip() and rc < 0:
            raise ValueError("Synthetic test text overflowed the PDF page rectangle.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


SAMPLE_CLEAN_CV = """
Jane Doe
Computer Engineering Student
jane.doe@example.com | +34 612 345 678 | linkedin.com/in/janedoe | github.com/janedoe

PROFILE
Final-year Computer Engineering student focused on Artificial Intelligence, Machine Learning,
Data Science, and Python software engineering. Available from March for a 6-month internship
in analytics, data engineering, or applied AI.

EDUCATION
B.Sc. in Computer Engineering (2023 - 2027)
Universitat Rovira i Virgili, Tarragona, Spain
Relevant Coursework: Artificial Intelligence, Machine Learning, Databases, Statistics, Algorithms.

TECHNICAL SKILLS
Languages: Python, SQL, Java, C, C++, JavaScript
Data Science & AI: pandas, NumPy, scikit-learn, Matplotlib, Machine Learning, Data Analysis
Tools & Systems: Git, Docker, Linux

PROJECTS
Hybrid AI CV Analyzer
Designed and built a modular NLP pipeline in Python using PyMuPDF, sentence embeddings,
Pydantic, and pytest to evaluate resume-to-job compatibility, detect ATS formatting errors,
and generate structured feedback.

Multi-Agent System in Minecraft
Built a distributed system of three autonomous agents communicating through structured JSON
messages to explore terrain, gather resources, and automate construction tasks using multiple
search strategies.

WORK EXPERIENCE
Data Intern - Tech Corp (Summer 2026)
Cleaned, validated, and analyzed large customer datasets using Python and SQL to support
business teams. Built automated reporting dashboards, documented data workflows, and
collaborated daily with international stakeholders under tight deadlines.

LANGUAGES
Spanish (Native), Catalan (Native), English (B2 - Working proficiency)
"""


def test_clean_cv_scores_high() -> None:
    pdf_bytes = _build_pdf_bytes([SAMPLE_CLEAN_CV])
    result = parse_cv_pdf(pdf_bytes)

    assert result.page_count == 1
    assert result.ats_score == 100
    assert result.contact.email == "jane.doe@example.com"
    assert result.contact.phone is not None
    assert result.contact.github is not None
    assert result.contact.linkedin is not None
    assert "education" in result.detected_sections
    assert "skills" in result.detected_sections
    assert "experience" in result.detected_sections
    assert result.issues == []


def test_empty_pdf_flags_unreadable() -> None:
    pdf_bytes = _build_pdf_bytes([""])
    result = parse_cv_pdf(pdf_bytes)

    issue_codes = {issue.code for issue in result.issues}
    assert "UNREADABLE_OR_SCANNED_PDF" in issue_codes
    assert "MISSING_EMAIL" in issue_codes
    assert result.ats_score < 60


def test_too_many_pages_adds_warning() -> None:
    pdf_bytes = _build_pdf_bytes([SAMPLE_CLEAN_CV, SAMPLE_CLEAN_CV, SAMPLE_CLEAN_CV])
    result = parse_cv_pdf(pdf_bytes)

    issue_codes = {issue.code for issue in result.issues}
    assert result.page_count == 3
    assert "TOO_MANY_PAGES" in issue_codes


def test_missing_file_raises_error() -> None:
    with pytest.raises(FileNotFoundError):
        parse_cv_pdf("non_existent_resume.pdf")