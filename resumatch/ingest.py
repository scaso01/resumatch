"""
ResuMatch - Resume Ingestion Module
=====================================
PDF and DOCX parsing to raw text + metadata.

Supports both file paths (str/Path) and in-memory bytes (BytesIO).
Extracts raw text, page count, word count, and contact information.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Union

import pdfplumber
from docx import Document

from resumatch.config import logger
from resumatch.models import ContactInfo, ParsedResume
from resumatch.sections import detect_sections

# ---------------------------------------------------------------------------
# Contact extraction regexes
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(?:\+?1[-.\s]?)?"
    r"(?:\(?\d{3}\)?[-.\s]?)"
    r"\d{3}[-.\s]?\d{4}"
)
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+")


def _extract_contact(text: str) -> ContactInfo:
    """Extract contact info from resume text using regex patterns.

    Looks for email, phone, LinkedIn URL, and infers name from the first
    non-empty line.
    """
    email_match = _EMAIL_RE.search(text)
    phone_match = _PHONE_RE.search(text)
    linkedin_match = _LINKEDIN_RE.search(text)

    # Name: first non-empty line that is NOT an email/phone/url-only line
    name = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip lines that are entirely an email, phone, or URL
        if _EMAIL_RE.fullmatch(stripped) or _PHONE_RE.fullmatch(stripped):
            continue
        name = stripped
        break

    return ContactInfo(
        name=name,
        email=email_match.group(0) if email_match else None,
        phone=phone_match.group(0) if phone_match else None,
        linkedin=linkedin_match.group(0) if linkedin_match else None,
    )


def _resolve_input(file_path_or_bytes: Union[str, Path, io.BytesIO]) -> tuple[io.BytesIO | None, Path | None]:
    """Resolve input to either a BytesIO stream or a file Path.

    Returns (bytes_io, None) for in-memory input, or (None, path) for file paths.
    """
    if isinstance(file_path_or_bytes, io.BytesIO):
        file_path_or_bytes.seek(0)
        return file_path_or_bytes, None
    path = Path(file_path_or_bytes)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return None, path


# ---------------------------------------------------------------------------
# PDF text normalization
# ---------------------------------------------------------------------------

def _rejoin_wrapped_lines(text: str) -> str:
    """Rejoin lines that pdfplumber broke at visual line boundaries.

    If a line starts with a lowercase letter and looks like prose continuation,
    join it to the previous line (e.g. a bullet that wrapped).
    """
    lines = text.splitlines()
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            merged.append("")
            continue
        # Don't merge emails, URLs, or very short lines
        if "@" in stripped or stripped.startswith("http") or len(stripped) < 5:
            merged.append(line)
            continue
        # Continuation: starts lowercase (mid-sentence wrap from PDF)
        if merged and merged[-1] and stripped[0].islower():
            merged[-1] = merged[-1].rstrip() + " " + stripped
        else:
            merged.append(line)
    return "\n".join(merged)


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

def parse_pdf(file_path_or_bytes: Union[str, Path, io.BytesIO]) -> ParsedResume:
    """Parse a PDF file into a ParsedResume.

    Args:
        file_path_or_bytes: Path to a PDF file, or a BytesIO containing PDF data.

    Returns:
        ParsedResume with raw_text, page_count, word_count, and contact info.
    """
    logger.debug("Parsing PDF input")
    bytes_io, path = _resolve_input(file_path_or_bytes)

    source = bytes_io if bytes_io is not None else str(path)
    pages_text: list[str] = []

    with pdfplumber.open(source) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)

    raw_text = "\n".join(pages_text)
    # Clean up pdfplumber extraction artifacts
    raw_text = re.sub(r" {2,}", " ", raw_text)  # collapse double spaces
    raw_text = _rejoin_wrapped_lines(raw_text)  # merge continuation lines
    word_count = len(raw_text.split())
    contact = _extract_contact(raw_text)

    logger.info("PDF parsed: %d pages, %d words", page_count, word_count)

    return ParsedResume(
        raw_text=raw_text,
        page_count=page_count,
        word_count=word_count,
        contact=contact,
        file_type="pdf",
    )


# ---------------------------------------------------------------------------
# DOCX parsing
# ---------------------------------------------------------------------------

def parse_docx(file_path_or_bytes: Union[str, Path, io.BytesIO]) -> ParsedResume:
    """Parse a DOCX file into a ParsedResume.

    Args:
        file_path_or_bytes: Path to a DOCX file, or a BytesIO containing DOCX data.

    Returns:
        ParsedResume with raw_text, estimated page_count, word_count,
        and contact info.
    """
    logger.debug("Parsing DOCX input")
    bytes_io, path = _resolve_input(file_path_or_bytes)

    source = bytes_io if bytes_io is not None else str(path)
    doc = Document(source)

    # Extract header/footer text (contact info often lives here)
    header_footer_lines: list[str] = []
    for section in doc.sections:
        if section.header:
            for p in section.header.paragraphs:
                if p.text.strip():
                    header_footer_lines.append(p.text)
        if section.footer:
            for p in section.footer.paragraphs:
                if p.text.strip():
                    header_footer_lines.append(p.text)

    paragraphs = []
    _WML_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    for p in doc.paragraphs:
        text = p.text
        # Detect list items: python-docx strips bullet markers from formatted lists.
        # Check for numPr (numbering properties) in the paragraph XML to identify
        # bullets/numbered items, and prepend a marker so extract_bullets can find them.
        is_list_item = False
        try:
            elem = p._element
            # Only process real lxml elements, not mocks
            if hasattr(elem, 'tag') and isinstance(elem.tag, str):
                pPr = elem.find(f"{_WML_NS}pPr")
                if pPr is not None:
                    is_list_item = pPr.find(f"{_WML_NS}numPr") is not None
        except (AttributeError, TypeError):
            pass
        if text.strip() and is_list_item:
            text = "\u2022 " + text
        paragraphs.append(text)

    # Extract text from tables (skills sections are often formatted as tables)
    for table in doc.tables:
        for row in table.rows:
            row_text = "  ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)

    raw_text = "\n".join(paragraphs)
    # Prepend header/footer text so contact extraction sees it
    if header_footer_lines:
        raw_text = "\n".join(header_footer_lines) + "\n" + raw_text

    word_count = len(raw_text.split())
    contact = _extract_contact(raw_text)

    # Estimate page count from word count (~450 words per resume page)
    page_count = max(1, round(word_count / 450))

    logger.info("DOCX parsed: ~%d pages, %d words", page_count, word_count)

    return ParsedResume(
        raw_text=raw_text,
        page_count=page_count,
        word_count=word_count,
        contact=contact,
        file_type="docx",
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def parse_file(file_path_or_bytes: Union[str, Path, io.BytesIO], filename: str) -> ParsedResume:
    """Dispatch to the correct parser based on file extension.

    Args:
        file_path_or_bytes: Path or BytesIO containing the file data.
        filename: Original filename (used to determine extension).

    Returns:
        ParsedResume from the appropriate parser.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = Path(filename).suffix.lower()
    logger.debug("parse_file dispatching for extension '%s'", ext)

    if ext == ".pdf":
        result = parse_pdf(file_path_or_bytes)
    elif ext in (".docx", ".doc"):
        result = parse_docx(file_path_or_bytes)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Supported types: .pdf, .docx")

    result.file_name = filename

    # Run section detection on the raw text if no sections were found yet
    if not result.sections:
        result.sections = detect_sections(result.raw_text)

    return result
