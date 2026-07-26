"""
ResuMatch - Section Detection Module
======================================
Section detection via regex + heuristics.

Detects common resume section headings (EXPERIENCE, EDUCATION, SKILLS, etc.)
using regex patterns for ALL CAPS lines, Title Case with colons, and lines
ending with colons. Maps heading variations to canonical section names.
"""

from __future__ import annotations

import re

from resumatch.models import ResumeSection

# ---------------------------------------------------------------------------
# Canonical section name mapping
# ---------------------------------------------------------------------------

# Maps lowercase variations to canonical section names.
_SECTION_ALIASES: dict[str, str] = {
    # Experience
    "experience": "experience",
    "work experience": "experience",
    "work history": "experience",
    "professional experience": "experience",
    "employment history": "experience",
    "employment": "experience",
    "career history": "experience",
    "relevant experience": "experience",
    # Education
    "education": "education",
    "academic background": "education",
    "academic history": "education",
    "educational background": "education",
    # Skills
    "skills": "skills",
    "technical skills": "skills",
    "core competencies": "skills",
    "competencies": "skills",
    "key skills": "skills",
    "areas of expertise": "skills",
    "proficiencies": "skills",
    "technologies": "skills",
    "technology": "skills",
    "tools": "skills",
    "methodologies": "skills",
    "tools & technologies": "skills",
    "tools and technologies": "skills",
    "tools | technology | methodologies": "skills",
    # Summary / Objective
    "summary": "summary",
    "professional summary": "summary",
    "executive summary": "summary",
    "career summary": "summary",
    "objective": "summary",
    "career objective": "summary",
    "profile": "summary",
    "about me": "summary",
    "personal statement": "summary",
    # Projects
    "projects": "projects",
    "personal projects": "projects",
    "key projects": "projects",
    "selected projects": "projects",
    # Certifications
    "certifications": "certifications",
    "certificates": "certifications",
    "licenses": "certifications",
    "licenses and certifications": "certifications",
    "certifications and licenses": "certifications",
    # Awards
    "awards": "awards",
    "honors": "awards",
    "awards and honors": "awards",
    "achievements": "awards",
    "honors and awards": "awards",
    # Volunteer
    "volunteer": "volunteer",
    "volunteer experience": "volunteer",
    "volunteering": "volunteer",
    "community involvement": "volunteer",
    # Publications
    "publications": "publications",
    "papers": "publications",
    "research": "publications",
    "research publications": "publications",
    # Interests
    "interests": "interests",
    "hobbies": "interests",
    "hobbies and interests": "interests",
    "personal interests": "interests",
    "activities": "interests",
}

# Known canonical section names (for fallback matching)
_CANONICAL_NAMES = {
    "experience", "education", "skills", "summary", "projects",
    "certifications", "awards", "volunteer", "publications", "interests",
}

# ---------------------------------------------------------------------------
# Heading detection regex patterns
# ---------------------------------------------------------------------------

# ALL CAPS line (2+ chars, optionally followed by a colon)
# Allow commas, pipes, parens, periods in addition to & /
_ALL_CAPS_RE = re.compile(r"^([A-Z][A-Z &/,.|().\-]+[A-Z.]):?\s*$")

# Title Case line ending with colon
_TITLE_COLON_RE = re.compile(r"^([A-Z][a-zA-Z &/]+):\s*$")

# Title Case line (standalone, 2-4 words, no trailing content)
_TITLE_CASE_RE = re.compile(r"^([A-Z][a-z]+(?:\s+[A-Za-z]+){0,3})\s*$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_section_name(heading: str) -> str:
    """Map a heading string to a canonical section name.

    Looks up the lowercase heading in the alias table. If not found,
    checks if any alias key is contained in the heading. Falls back to
    the lowercased heading itself.

    Args:
        heading: The raw heading text from the resume.

    Returns:
        Canonical section name (e.g. "experience", "skills").
    """
    key = heading.strip().lower().rstrip(":")
    # Direct lookup
    if key in _SECTION_ALIASES:
        return _SECTION_ALIASES[key]

    # Substring match: check if any alias key is contained in the heading
    for alias, canonical in sorted(_SECTION_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in key:
            return canonical

    return key


def extract_bullets(section_text: str) -> list[str]:
    """Extract bullet points from section text.

    Recognizes lines starting with -, *, bullet character, or numbered
    patterns (1., 1), etc.). Strips the bullet marker from the result.

    Fallback: when PDF text extraction strips bullet markers, treats
    non-empty content lines (10+ chars, starts with capital or verb)
    as implicit bullets.

    Args:
        section_text: The raw content of a resume section.

    Returns:
        List of bullet text strings (without the leading marker).
    """
    bullet_re = re.compile(
        r"^\s*"
        r"(?:[-*\u2022\u2023\u25E6\u2043\u2219\uf0b7\uf0a7\uf076\uf0d8]"  # bullet chars incl. Wingdings PUA
        r"|\d+[.)]\s*"                               # numbered: 1. or 1)
        r")"
        r"\s*(.*)"
    )
    # PDF often replaces bullet markers with leading whitespace (space-indented lines)
    _indent_bullet_re = re.compile(r"^ +([A-Z][a-z].*)")
    bullets: list[str] = []
    for line in section_text.splitlines():
        m = bullet_re.match(line)
        if m:
            text = m.group(1).strip()
            if text:
                bullets.append(text)

    # Only use space-indented heuristic as fallback when no explicit markers found.
    # This prevents role summary paragraphs from being treated as bullets.
    if not bullets:
        for line in section_text.splitlines():
            m = _indent_bullet_re.match(line)
            if m:
                text = m.group(1).strip()
                if len(text) >= 10:
                    bullets.append(text)

    # Fallback: PDF/DOCX extraction may strip bullet markers, leaving plain lines.
    # Treat lines that look like bullet-length statements as implicit bullets.
    if not bullets:
        _date_re = re.compile(
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
            r"Dec(?:ember)?)\s+\d{4}"
        )
        for line in section_text.splitlines():
            text = line.strip()
            if len(text) < 10:
                continue
            # Skip ALL-CAPS short lines (section headers, company names)
            if text.isupper() and len(text.split()) <= 5:
                continue
            # Skip lines containing date patterns (job titles, company headers)
            if _date_re.search(text):
                continue
            # Skip lines ending with "Present" (job title/date ranges)
            if text.rstrip().endswith("Present"):
                continue
            # Skip short title-case lines (likely job titles)
            if text.istitle() and len(text.split()) <= 6:
                continue
            # Skip lines with multiple pipes/commas and few words (metadata lines)
            if (text.count("|") >= 1 or text.count(",") >= 2) and len(text.split()) <= 8:
                continue
            # Accept lines starting with a capital letter or common verb forms
            if text[0].isupper() or text[0].isdigit():
                bullets.append(text)

    return bullets


def _is_heading(line: str) -> str | None:
    """Test whether a line is a section heading.

    Returns the captured heading text if it matches, or None.
    """
    # Normalize non-breaking spaces to regular spaces
    stripped = line.strip().replace("\xa0", " ")
    if not stripped or len(stripped) < 2:
        return None

    # ALL CAPS check (allow commas, pipes, parens in addition to & /)
    m = _ALL_CAPS_RE.match(stripped)
    if m:
        return m.group(1).strip()

    # Title Case with colon
    m = _TITLE_COLON_RE.match(stripped)
    if m:
        candidate = m.group(1).strip()
        if normalize_section_name(candidate) in _CANONICAL_NAMES:
            return candidate

    # Title Case standalone (must map to a known section)
    m = _TITLE_CASE_RE.match(stripped)
    if m:
        candidate = m.group(1).strip()
        if normalize_section_name(candidate) in _CANONICAL_NAMES:
            return candidate

    # Lowercase / any-case standalone: short line that maps to a known section
    # Handles DOCX where headings may be styled but exported as lowercase
    if len(stripped.split()) <= 5 and "\t" not in stripped:
        # Try the line as-is (handles "skills", "education", "professional experience")
        if normalize_section_name(stripped) in _CANONICAL_NAMES:
            return stripped
        # Try pipe-separated parts (handles "tools | technology | methodologies")
        if "|" in stripped:
            first_part = stripped.split("|")[0].strip()
            if normalize_section_name(first_part) in _CANONICAL_NAMES:
                return stripped

    return None


def detect_sections(text: str) -> list[ResumeSection]:
    """Detect sections in resume text by finding heading lines.

    Scans each line for heading patterns, then collects the content
    between consecutive headings. Bullets within each section are
    extracted automatically.

    Args:
        text: Full resume text.

    Returns:
        List of ResumeSection objects in document order.
    """
    lines = text.splitlines()
    headings: list[tuple[int, str]] = []  # (line_index, heading_text)

    for idx, line in enumerate(lines):
        heading = _is_heading(line)
        if heading is not None:
            # Only keep headings that map to a known canonical section name
            # This filters out names, titles, and other ALL-CAPS non-section lines
            canonical = normalize_section_name(heading)
            if canonical in _CANONICAL_NAMES:
                headings.append((idx, heading))

    if not headings:
        return []

    sections: list[ResumeSection] = []
    seen_names: dict[str, int] = {}  # canonical name -> index in sections list

    for i, (start_idx, heading) in enumerate(headings):
        # Content runs from the line after the heading to the line before the
        # next heading (or end of text).
        if i + 1 < len(headings):
            end_idx = headings[i + 1][0]
        else:
            end_idx = len(lines)

        content_lines = lines[start_idx + 1 : end_idx]
        content = "\n".join(content_lines).strip()
        canonical = normalize_section_name(heading)
        bullets = extract_bullets(content)

        # Merge duplicate section names (e.g. two skills sections)
        if canonical in seen_names:
            existing = sections[seen_names[canonical]]
            existing.content = existing.content + "\n" + content
            existing.bullets = existing.bullets + bullets
            existing.end_line = end_idx - 1
        else:
            seen_names[canonical] = len(sections)
            sections.append(
                ResumeSection(
                    name=canonical,
                    heading=heading,
                    content=content,
                    bullets=bullets,
                    start_line=start_idx,
                    end_line=end_idx - 1,
                )
            )

    return sections
