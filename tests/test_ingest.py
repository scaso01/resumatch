"""Tests for resumatch.ingest module.

Tests PDF and DOCX parsing, contact extraction, file dispatching, and
edge cases. Uses mocks for pdfplumber and python-docx since no real
fixture files are present.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from resumatch.ingest import (
    _extract_contact,
    parse_docx,
    parse_file,
    parse_pdf,
)
from resumatch.models import ParsedResume


# =============================================================================
# Contact extraction
# =============================================================================


class TestExtractContact:
    """Tests for _extract_contact helper."""

    def test_extracts_email(self):
        text = "John Doe\njohn@example.com\n555-1234"
        contact = _extract_contact(text)
        assert contact.email == "john@example.com"

    def test_extracts_phone_parentheses(self):
        text = "Jane Smith\njane@test.com | (555) 123-4567"
        contact = _extract_contact(text)
        assert contact.phone == "(555) 123-4567"

    def test_extracts_phone_dashes(self):
        text = "Name\n555-123-4567"
        contact = _extract_contact(text)
        assert contact.phone == "555-123-4567"

    def test_extracts_linkedin(self):
        text = "Name\nlinkedin.com/in/johndoe"
        contact = _extract_contact(text)
        assert contact.linkedin == "linkedin.com/in/johndoe"

    def test_extracts_linkedin_with_https(self):
        text = "Name\nhttps://www.linkedin.com/in/johndoe"
        contact = _extract_contact(text)
        assert contact.linkedin == "https://www.linkedin.com/in/johndoe"

    def test_extracts_name_from_first_line(self):
        text = "John Smith\njohn@email.com"
        contact = _extract_contact(text)
        assert contact.name == "John Smith"

    def test_skips_blank_lines_for_name(self):
        text = "\n\n  \nAlice Jones\nalice@email.com"
        contact = _extract_contact(text)
        assert contact.name == "Alice Jones"

    def test_no_email_returns_none(self):
        text = "No email here\nJust text"
        contact = _extract_contact(text)
        assert contact.email is None

    def test_no_phone_returns_none(self):
        text = "No phone\njust text"
        contact = _extract_contact(text)
        assert contact.phone is None

    def test_no_linkedin_returns_none(self):
        text = "No linkedin\njust text"
        contact = _extract_contact(text)
        assert contact.linkedin is None

    def test_full_contact_line(self):
        text = "John Smith\njohn.smith@email.com | (555) 123-4567 | linkedin.com/in/johnsmith | New York, NY"
        contact = _extract_contact(text)
        assert contact.name == "John Smith"
        assert contact.email == "john.smith@email.com"
        assert contact.phone == "(555) 123-4567"
        assert contact.linkedin == "linkedin.com/in/johnsmith"


# =============================================================================
# PDF parsing (mocked)
# =============================================================================


class TestParsePdf:
    """Tests for parse_pdf using mocked pdfplumber."""

    def _mock_pdfplumber(self, pages_text: list[str]):
        """Create a mock pdfplumber context manager returning given page texts."""
        mock_pages = []
        for text in pages_text:
            page = MagicMock()
            page.extract_text.return_value = text
            mock_pages.append(page)

        mock_pdf = MagicMock()
        mock_pdf.pages = mock_pages
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        return mock_pdf

    @patch("resumatch.ingest.pdfplumber")
    def test_parse_pdf_from_path(self, mock_pdfplumber, tmp_path):
        page_text = "John Doe\njohn@example.com\n\nEXPERIENCE\nDid things"
        mock_pdfplumber.open.return_value = self._mock_pdfplumber([page_text])

        pdf_file = tmp_path / "resume.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        result = parse_pdf(str(pdf_file))
        assert isinstance(result, ParsedResume)
        assert result.page_count == 1
        assert result.word_count > 0
        assert "John Doe" in result.raw_text
        assert result.file_type == "pdf"

    @patch("resumatch.ingest.pdfplumber")
    def test_parse_pdf_from_bytes(self, mock_pdfplumber):
        page_text = "Jane Smith\njane@test.com"
        mock_pdfplumber.open.return_value = self._mock_pdfplumber([page_text])

        bytes_io = io.BytesIO(b"%PDF-fake")
        result = parse_pdf(bytes_io)
        assert isinstance(result, ParsedResume)
        assert result.contact.name == "Jane Smith"

    @patch("resumatch.ingest.pdfplumber")
    def test_parse_pdf_multiple_pages(self, mock_pdfplumber):
        pages = ["Page 1 text", "Page 2 text", "Page 3 text"]
        mock_pdfplumber.open.return_value = self._mock_pdfplumber(pages)

        bytes_io = io.BytesIO(b"%PDF-fake")
        result = parse_pdf(bytes_io)
        assert result.page_count == 3
        assert "Page 1" in result.raw_text
        assert "Page 3" in result.raw_text

    @patch("resumatch.ingest.pdfplumber")
    def test_parse_pdf_empty_page(self, mock_pdfplumber):
        mock_pdfplumber.open.return_value = self._mock_pdfplumber([""])

        bytes_io = io.BytesIO(b"%PDF-fake")
        result = parse_pdf(bytes_io)
        assert result.page_count == 1
        assert result.word_count == 0

    @patch("resumatch.ingest.pdfplumber")
    def test_parse_pdf_extracts_contact(self, mock_pdfplumber):
        page_text = "Alice\nalice@email.com | (123) 456-7890 | linkedin.com/in/alice"
        mock_pdfplumber.open.return_value = self._mock_pdfplumber([page_text])

        bytes_io = io.BytesIO(b"%PDF-fake")
        result = parse_pdf(bytes_io)
        assert result.contact.email == "alice@email.com"
        assert result.contact.phone == "(123) 456-7890"
        assert result.contact.linkedin == "linkedin.com/in/alice"


# =============================================================================
# DOCX parsing (mocked)
# =============================================================================


class TestParseDocx:
    """Tests for parse_docx using mocked python-docx."""

    @patch("resumatch.ingest.Document")
    def test_parse_docx_from_path(self, mock_document, tmp_path):
        para1 = MagicMock()
        para1.text = "Bob Builder"
        para2 = MagicMock()
        para2.text = "bob@email.com"
        mock_doc = MagicMock()
        mock_doc.paragraphs = [para1, para2]
        mock_document.return_value = mock_doc

        docx_file = tmp_path / "resume.docx"
        docx_file.write_bytes(b"fake-docx")

        result = parse_docx(str(docx_file))
        assert isinstance(result, ParsedResume)
        assert "Bob Builder" in result.raw_text
        assert result.file_type == "docx"
        assert result.page_count == 1

    @patch("resumatch.ingest.Document")
    def test_parse_docx_from_bytes(self, mock_document):
        para1 = MagicMock()
        para1.text = "Carol Danvers"
        mock_doc = MagicMock()
        mock_doc.paragraphs = [para1]
        mock_document.return_value = mock_doc

        bytes_io = io.BytesIO(b"fake-docx")
        result = parse_docx(bytes_io)
        assert result.contact.name == "Carol Danvers"

    @patch("resumatch.ingest.Document")
    def test_parse_docx_word_count(self, mock_document):
        para1 = MagicMock()
        para1.text = "one two three four five"
        mock_doc = MagicMock()
        mock_doc.paragraphs = [para1]
        mock_document.return_value = mock_doc

        bytes_io = io.BytesIO(b"fake-docx")
        result = parse_docx(bytes_io)
        assert result.word_count == 5

    @patch("resumatch.ingest.Document")
    def test_parse_docx_extracts_contact(self, mock_document):
        para1 = MagicMock()
        para1.text = "Dave"
        para2 = MagicMock()
        para2.text = "dave@example.org | (999) 888-7777"
        mock_doc = MagicMock()
        mock_doc.paragraphs = [para1, para2]
        mock_document.return_value = mock_doc

        bytes_io = io.BytesIO(b"fake-docx")
        result = parse_docx(bytes_io)
        assert result.contact.email == "dave@example.org"
        assert result.contact.phone == "(999) 888-7777"


# =============================================================================
# File dispatcher
# =============================================================================


class TestParseFile:
    """Tests for parse_file dispatcher."""

    @patch("resumatch.ingest.parse_pdf")
    def test_dispatches_pdf(self, mock_parse_pdf):
        mock_parse_pdf.return_value = ParsedResume(raw_text="pdf text", word_count=2)
        result = parse_file(io.BytesIO(b"data"), "resume.pdf")
        mock_parse_pdf.assert_called_once()
        assert result.file_name == "resume.pdf"

    @patch("resumatch.ingest.parse_docx")
    def test_dispatches_docx(self, mock_parse_docx):
        mock_parse_docx.return_value = ParsedResume(raw_text="docx text", word_count=2)
        result = parse_file(io.BytesIO(b"data"), "resume.docx")
        mock_parse_docx.assert_called_once()
        assert result.file_name == "resume.docx"

    @patch("resumatch.ingest.parse_docx")
    def test_dispatches_doc_extension(self, mock_parse_docx):
        mock_parse_docx.return_value = ParsedResume(raw_text="doc text", word_count=2)
        parse_file(io.BytesIO(b"data"), "resume.doc")
        mock_parse_docx.assert_called_once()

    def test_unsupported_extension_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_file(io.BytesIO(b"data"), "resume.txt")

    def test_unsupported_extension_odt(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_file(io.BytesIO(b"data"), "resume.odt")

    @patch("resumatch.ingest.parse_pdf")
    def test_case_insensitive_extension(self, mock_parse_pdf):
        mock_parse_pdf.return_value = ParsedResume(raw_text="text", word_count=1)
        parse_file(io.BytesIO(b"data"), "Resume.PDF")
        mock_parse_pdf.assert_called_once()


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_pdf("/nonexistent/path/resume.pdf")

    def test_file_not_found_docx_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_docx("/nonexistent/path/resume.docx")
