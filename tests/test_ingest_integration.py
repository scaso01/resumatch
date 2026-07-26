"""Ingest tests that use real files and real parsers.

Every test in test_ingest.py patches `resumatch.ingest.pdfplumber` and
`resumatch.ingest.Document` with hand-built mocks, so until now neither
pdfplumber nor python-docx had ever processed actual bytes here -- despite
opening a resume being the first thing any user does.

Fixtures are fabricated (see tests/fixtures/_generate.py).
"""

from __future__ import annotations

import io

import pytest

from resumatch.ingest import parse_docx, parse_file, parse_pdf

pytestmark = pytest.mark.usefixtures("fixtures_dir")


@pytest.fixture
def pdf_bytes(fixtures_dir) -> bytes:
    return (fixtures_dir / "sample_resume.pdf").read_bytes()


@pytest.fixture
def docx_bytes(fixtures_dir) -> bytes:
    return (fixtures_dir / "sample_resume.docx").read_bytes()


class TestRealPdf:
    def test_parses_from_path(self, fixtures_dir):
        parsed = parse_pdf(fixtures_dir / "sample_resume.pdf")
        assert "JORDAN AVERY" in parsed.raw_text
        assert parsed.file_type == "pdf"
        assert parsed.page_count == 1
        assert parsed.word_count > 100

    def test_parses_from_bytes(self, pdf_bytes):
        parsed = parse_pdf(io.BytesIO(pdf_bytes))
        assert "Northwind Systems" in parsed.raw_text

    def test_extracts_contact_details(self, fixtures_dir):
        parsed = parse_pdf(fixtures_dir / "sample_resume.pdf")
        assert parsed.contact.email == "jordan.avery@example.com"
        assert parsed.contact.phone is not None
        assert parsed.contact.linkedin is not None

    def test_keeps_the_numbers_that_drive_scoring(self, fixtures_dir):
        """Quantification is a scored signal, so digits must survive parsing."""
        text = parse_pdf(fixtures_dir / "sample_resume.pdf").raw_text
        for figure in ("74%", "820ms", "190ms", "40 million", "$120,000"):
            assert figure in text, f"{figure!r} lost during PDF extraction"


class TestRealDocx:
    def test_parses_from_path(self, fixtures_dir):
        parsed = parse_docx(fixtures_dir / "sample_resume.docx")
        assert "JORDAN AVERY" in parsed.raw_text
        assert parsed.file_type == "docx"
        assert parsed.word_count > 100

    def test_parses_from_bytes(self, docx_bytes):
        parsed = parse_docx(io.BytesIO(docx_bytes))
        assert "Cobalt Analytics" in parsed.raw_text

    def test_extracts_contact_details(self, fixtures_dir):
        parsed = parse_docx(fixtures_dir / "sample_resume.docx")
        assert parsed.contact.email == "jordan.avery@example.com"


class TestParseFileDispatch:
    def test_pdf_and_docx_agree_on_content(self, fixtures_dir):
        """The two parsers are separate code paths over the same resume."""
        from_pdf = parse_file(fixtures_dir / "sample_resume.pdf", "sample_resume.pdf")
        from_docx = parse_file(fixtures_dir / "sample_resume.docx", "sample_resume.docx")

        for marker in ("JORDAN AVERY", "Northwind Systems", "EDUCATION", "SKILLS"):
            assert marker in from_pdf.raw_text
            assert marker in from_docx.raw_text

    def test_detects_sections(self, fixtures_dir):
        parsed = parse_file(fixtures_dir / "sample_resume.pdf", "sample_resume.pdf")
        found = {section.name.lower() for section in parsed.sections}
        assert {"experience", "education", "skills"} <= found, f"got {found}"

    def test_rejects_unsupported_type(self, fixtures_dir):
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_file(fixtures_dir / "sample_jd.txt", "sample_jd.txt")
