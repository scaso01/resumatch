"""Tests for resumatch.sections module.

Tests section detection, heading normalization, bullet extraction,
and edge cases with various heading styles.
"""

from __future__ import annotations


from resumatch.sections import (
    detect_sections,
    extract_bullets,
    normalize_section_name,
)


# =============================================================================
# normalize_section_name
# =============================================================================


class TestNormalizeSectionName:
    """Tests for normalize_section_name mapping."""

    def test_direct_match(self):
        assert normalize_section_name("experience") == "experience"

    def test_work_history_maps_to_experience(self):
        assert normalize_section_name("Work History") == "experience"

    def test_work_experience_maps_to_experience(self):
        assert normalize_section_name("Work Experience") == "experience"

    def test_professional_experience(self):
        assert normalize_section_name("Professional Experience") == "experience"

    def test_technical_skills_maps_to_skills(self):
        assert normalize_section_name("Technical Skills") == "skills"

    def test_core_competencies_maps_to_skills(self):
        assert normalize_section_name("Core Competencies") == "skills"

    def test_education_direct(self):
        assert normalize_section_name("Education") == "education"

    def test_career_objective_maps_to_summary(self):
        assert normalize_section_name("Career Objective") == "summary"

    def test_certifications_direct(self):
        assert normalize_section_name("Certifications") == "certifications"

    def test_awards_and_honors(self):
        assert normalize_section_name("Awards and Honors") == "awards"

    def test_volunteer_experience(self):
        assert normalize_section_name("Volunteer Experience") == "volunteer"

    def test_research_maps_to_publications(self):
        assert normalize_section_name("Research") == "publications"

    def test_hobbies_maps_to_interests(self):
        assert normalize_section_name("Hobbies") == "interests"

    def test_unknown_heading_returns_lowercase(self):
        assert normalize_section_name("Random Section") == "random section"

    def test_strips_trailing_colon(self):
        assert normalize_section_name("Skills:") == "skills"

    def test_strips_whitespace(self):
        assert normalize_section_name("  Education  ") == "education"


# =============================================================================
# extract_bullets
# =============================================================================


class TestExtractBullets:
    """Tests for extract_bullets with different marker styles."""

    def test_dash_bullets(self):
        text = "- First item\n- Second item\n- Third item"
        bullets = extract_bullets(text)
        assert bullets == ["First item", "Second item", "Third item"]

    def test_asterisk_bullets(self):
        text = "* Alpha\n* Beta"
        bullets = extract_bullets(text)
        assert bullets == ["Alpha", "Beta"]

    def test_unicode_bullet_char(self):
        text = "\u2022 Bullet one\n\u2022 Bullet two"
        bullets = extract_bullets(text)
        assert bullets == ["Bullet one", "Bullet two"]

    def test_numbered_bullets_dot(self):
        text = "1. First\n2. Second\n3. Third"
        bullets = extract_bullets(text)
        assert bullets == ["First", "Second", "Third"]

    def test_numbered_bullets_paren(self):
        text = "1) Alpha\n2) Beta"
        bullets = extract_bullets(text)
        assert bullets == ["Alpha", "Beta"]

    def test_mixed_bullets(self):
        text = "- Dash item\n* Star item\n1. Numbered item"
        bullets = extract_bullets(text)
        assert len(bullets) == 3

    def test_empty_text_returns_empty(self):
        assert extract_bullets("") == []

    def test_no_bullets_short_lines_returns_empty(self):
        text = "Short.\nTiny."
        assert extract_bullets(text) == []

    def test_indented_bullets(self):
        text = "  - Indented one\n  - Indented two"
        bullets = extract_bullets(text)
        assert bullets == ["Indented one", "Indented two"]

    def test_empty_bullet_lines_skipped(self):
        text = "- Real item\n-\n- Another real item"
        bullets = extract_bullets(text)
        assert bullets == ["Real item", "Another real item"]

    def test_fallback_pdf_stripped_bullets(self):
        """PDF text with no bullet markers — should fall back to content lines."""
        text = (
            "Led migration of monolithic application to microservices\n"
            "Managed team of 8 engineers delivering 3 product releases\n"
            "Implemented automated testing pipeline increasing coverage to 92%"
        )
        bullets = extract_bullets(text)
        assert len(bullets) == 3
        assert "Led migration" in bullets[0]

    def test_fallback_skips_short_lines(self):
        """Fallback should skip lines shorter than 10 characters."""
        text = "Led a major infrastructure overhaul project\nShort\nBuilt real-time data pipelines"
        bullets = extract_bullets(text)
        assert len(bullets) == 2
        assert all(len(b) >= 10 for b in bullets)

    def test_fallback_not_used_when_markers_present(self):
        """When standard bullet markers exist, fallback should NOT activate."""
        text = "- First item\n- Second item\nThis line has no bullet but is long enough to qualify"
        bullets = extract_bullets(text)
        # Should only get the two with markers, not the fallback line
        assert bullets == ["First item", "Second item"]


# =============================================================================
# detect_sections — ALL CAPS headings
# =============================================================================


class TestDetectSectionsAllCaps:
    """Tests for detect_sections with ALL CAPS headings."""

    def test_detects_experience(self):
        text = "John Doe\n\nEXPERIENCE\n\nSome work here"
        sections = detect_sections(text)
        names = [s.name for s in sections]
        assert "experience" in names

    def test_detects_education(self):
        text = "Name\n\nEDUCATION\n\nBS in CS"
        sections = detect_sections(text)
        names = [s.name for s in sections]
        assert "education" in names

    def test_detects_skills(self):
        text = "Name\n\nSKILLS\n\nPython, Java"
        sections = detect_sections(text)
        names = [s.name for s in sections]
        assert "skills" in names

    def test_detects_multiple_sections(self):
        text = "Name\n\nEXPERIENCE\n\nWork stuff\n\nEDUCATION\n\nSchool stuff\n\nSKILLS\n\nPython"
        sections = detect_sections(text)
        names = [s.name for s in sections]
        assert names == ["experience", "education", "skills"]

    def test_section_content_captured(self):
        text = "Name\n\nEXPERIENCE\n\n- Did thing one\n- Did thing two\n\nEDUCATION\n\nMIT"
        sections = detect_sections(text)
        exp = [s for s in sections if s.name == "experience"][0]
        assert "Did thing one" in exp.content
        assert "Did thing two" in exp.content

    def test_section_bullets_extracted(self):
        text = "Name\n\nEXPERIENCE\n\n- Built systems\n- Led teams\n\nSKILLS\n\nPython"
        sections = detect_sections(text)
        exp = [s for s in sections if s.name == "experience"][0]
        assert len(exp.bullets) == 2
        assert "Built systems" in exp.bullets


# =============================================================================
# detect_sections — Title Case headings
# =============================================================================


class TestDetectSectionsTitleCase:
    """Tests for detect_sections with Title Case headings."""

    def test_title_case_education(self):
        text = "Name\n\nEducation\n\nBS in CS"
        sections = detect_sections(text)
        names = [s.name for s in sections]
        assert "education" in names

    def test_title_case_skills(self):
        text = "Name\n\nSkills\n\nPython, Java"
        sections = detect_sections(text)
        names = [s.name for s in sections]
        assert "skills" in names

    def test_work_history_title_case(self):
        text = "Name\n\nWork History\n\nWorked at company"
        sections = detect_sections(text)
        names = [s.name for s in sections]
        assert "experience" in names


# =============================================================================
# detect_sections — edge cases
# =============================================================================


class TestDetectSectionsEdgeCases:
    """Edge case tests for detect_sections."""

    def test_empty_text_returns_empty(self):
        assert detect_sections("") == []

    def test_no_headings_returns_empty(self):
        text = "Just a block of text with no clear sections.\nAnother line."
        assert detect_sections(text) == []

    def test_preserves_heading_text(self):
        text = "Name\n\nEXPERIENCE\n\nSome work"
        sections = detect_sections(text)
        assert sections[0].heading == "EXPERIENCE"

    def test_start_and_end_line_set(self):
        text = "Name\n\nEXPERIENCE\n\nWork here\n\nEDUCATION\n\nSchool"
        sections = detect_sections(text)
        assert sections[0].start_line < sections[1].start_line


# =============================================================================
# detect_sections — with sample fixture
# =============================================================================


class TestDetectSectionsWithFixture:
    """Tests using the sample_resume_text fixture from conftest."""

    def test_sample_resume_has_three_sections(self, sample_resume_text):
        sections = detect_sections(sample_resume_text)
        assert len(sections) == 3

    def test_sample_resume_section_names(self, sample_resume_text):
        sections = detect_sections(sample_resume_text)
        names = [s.name for s in sections]
        assert "experience" in names
        assert "education" in names
        assert "skills" in names

    def test_sample_resume_experience_has_bullets(self, sample_resume_text):
        sections = detect_sections(sample_resume_text)
        exp = [s for s in sections if s.name == "experience"][0]
        assert len(exp.bullets) >= 5

    def test_sample_weak_resume_sections(self, sample_weak_resume_text):
        sections = detect_sections(sample_weak_resume_text)
        names = [s.name for s in sections]
        assert "experience" in names
        assert "education" in names
        assert "skills" in names
