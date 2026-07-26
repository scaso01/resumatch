"""Tests for resumatch.scoring -- impact, presentation, competencies, aggregation, zones."""

import pytest

from resumatch.models import (
    CompetenciesScore,
    ContactInfo,
    ImpactScore,
    ParsedResume,
    PresentationScore,
    ResumeScore,
    ResumeSection,
    ScoreZone,
)
from resumatch.scoring import (
    SKILL_CATEGORIES,
    classify_zone,
    score_competencies,
    score_impact,
    score_presentation,
    score_resume,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_resume(
    bullets: list[str] | None = None,
    skills_content: str = "",
    sections: list[ResumeSection] | None = None,
    contact: ContactInfo | None = None,
    page_count: int = 1,
    raw_text: str = "",
) -> ParsedResume:
    """Build a ParsedResume for testing."""
    if sections is None:
        sections = []
        if bullets is not None:
            sections.append(
                ResumeSection(
                    name="experience",
                    heading="EXPERIENCE",
                    content="...",
                    bullets=bullets,
                )
            )
        if skills_content:
            sections.append(
                ResumeSection(
                    name="skills",
                    heading="SKILLS",
                    content=skills_content,
                    bullets=[],
                )
            )
    return ParsedResume(
        raw_text=raw_text or "Sample resume text",
        sections=sections,
        contact=contact or ContactInfo(),
        page_count=page_count,
        word_count=len((raw_text or "Sample resume text").split()),
    )


@pytest.fixture
def strong_resume_config():
    """Config dict for scoring tests."""
    return {
        "scoring": {
            "weights": {"impact": 40, "presentation": 30, "competencies": 30},
            "zones": {"green": 85, "yellow": 50},
            "impact": {
                "action_verb_target": 0.90,
                "quantification_target": 0.60,
                "specificity_target": 0.70,
            },
            "presentation": {
                "ideal_pages_min": 1,
                "ideal_pages_max": 2,
                "required_sections": ["experience", "education", "skills"],
                "grammar_penalty_per_error": 2,
                "grammar_penalty_cap": 20,
            },
            "competencies": {
                "categories": ["technical", "leadership", "communication", "analytical", "domain"],
                "min_skills_for_full_score": 10,
                "min_categories_for_full_score": 4,
            },
        },
    }


# =============================================================================
# classify_zone
# =============================================================================

class TestClassifyZone:
    def test_green_zone(self):
        assert classify_zone(90.0) == ScoreZone.GREEN

    def test_green_boundary(self):
        assert classify_zone(85.0) == ScoreZone.GREEN

    def test_yellow_zone(self):
        assert classify_zone(70.0) == ScoreZone.YELLOW

    def test_yellow_boundary(self):
        assert classify_zone(50.0) == ScoreZone.YELLOW

    def test_red_zone(self):
        assert classify_zone(30.0) == ScoreZone.RED

    def test_red_boundary(self):
        assert classify_zone(49.9) == ScoreZone.RED

    def test_zero_is_red(self):
        assert classify_zone(0.0) == ScoreZone.RED

    def test_hundred_is_green(self):
        assert classify_zone(100.0) == ScoreZone.GREEN


# =============================================================================
# score_impact
# =============================================================================

class TestScoreImpact:
    def test_strong_resume_high_impact(self, strong_resume_config):
        bullets = [
            "Led migration of monolithic application to microservices, reducing deployment time by 75%",
            "Managed team of 8 engineers, delivering 3 major product releases on schedule",
            "Implemented automated testing pipeline that increased code coverage from 45% to 92%",
            "Reduced cloud infrastructure costs by $200K annually through optimization",
        ]
        resume = _make_resume(bullets=bullets)
        result = score_impact(resume, strong_resume_config)
        assert isinstance(result, ImpactScore)
        assert result.score >= 60.0
        assert result.action_verb_rate >= 0.9
        assert result.quantification_rate >= 0.9
        assert result.weak_verbs_found == []

    def test_weak_resume_low_impact(self, strong_resume_config):
        bullets = [
            "Helped maintain the systems",
            "Assisted team members with their work",
            "Worked on various projects",
            "Was responsible for some tasks",
        ]
        resume = _make_resume(bullets=bullets)
        result = score_impact(resume, strong_resume_config)
        assert result.score < 40.0
        assert result.action_verb_rate == 0.0
        assert len(result.weak_verbs_found) >= 3

    def test_no_experience_section(self, strong_resume_config):
        resume = _make_resume(sections=[
            ResumeSection(name="skills", heading="SKILLS", content="Python, Java"),
        ])
        result = score_impact(resume, strong_resume_config)
        assert result.score == 0.0

    def test_score_between_0_and_100(self, strong_resume_config):
        bullets = ["Led a team", "Helped with testing"]
        resume = _make_resume(bullets=bullets)
        result = score_impact(resume, strong_resume_config)
        assert 0.0 <= result.score <= 100.0

    def test_bullets_without_metrics_populated(self, strong_resume_config):
        bullets = [
            "Led a migration project to completion",
            "Managed a team of engineers across departments",
        ]
        resume = _make_resume(bullets=bullets)
        result = score_impact(resume, strong_resume_config)
        # These bullets have no $, %, or explicit numbers (wait, "a" is not a number,
        # but let's check — plain numbers will match digits)
        # Actually neither has digits, so both should be without metrics
        assert len(result.bullets_without_metrics) == 2

    def test_content_fallback_when_no_bullets(self, strong_resume_config):
        """When experience section has content but no bullets, fall back to content lines."""
        resume = _make_resume(
            sections=[
                ResumeSection(
                    name="experience",
                    heading="EXPERIENCE",
                    content="Led migration of systems to cloud infrastructure reducing costs by 40%\nManaged team of 12 engineers delivering critical projects on time",
                    bullets=[],  # No bullets extracted (PDF stripped markers)
                ),
            ],
        )
        result = score_impact(resume, strong_resume_config)
        # Should NOT return 0 — content lines should be used as fallback
        assert result.score > 0.0
        assert result.action_verb_rate > 0.0


# =============================================================================
# score_presentation
# =============================================================================

class TestScorePresentation:
    def test_complete_resume_high_score(self, strong_resume_config, sample_parsed_resume):
        result = score_presentation(sample_parsed_resume, strong_resume_config)
        assert isinstance(result, PresentationScore)
        assert result.score >= 50.0
        assert result.section_completeness == 1.0
        assert result.missing_sections == []

    def test_missing_sections_penalized(self, strong_resume_config):
        resume = _make_resume(
            sections=[
                ResumeSection(name="experience", heading="EXPERIENCE", content="...", bullets=["Led a project"]),
            ],
            contact=ContactInfo(name="Test", email="test@test.com"),
        )
        result = score_presentation(resume, strong_resume_config)
        assert result.section_completeness < 1.0
        assert len(result.missing_sections) >= 1

    def test_no_contact_info_penalized(self, strong_resume_config):
        resume = _make_resume(
            sections=[
                ResumeSection(name="experience", heading="EXPERIENCE", content="..."),
                ResumeSection(name="education", heading="EDUCATION", content="..."),
                ResumeSection(name="skills", heading="SKILLS", content="Python"),
            ],
        )
        result = score_presentation(resume, strong_resume_config)
        assert result.contact_completeness == 0.0

    def test_full_contact_info(self, strong_resume_config):
        resume = _make_resume(
            sections=[
                ResumeSection(name="experience", heading="EXPERIENCE", content="..."),
                ResumeSection(name="education", heading="EDUCATION", content="..."),
                ResumeSection(name="skills", heading="SKILLS", content="Python"),
            ],
            contact=ContactInfo(
                name="John", email="john@example.com", phone="555-1234",
                linkedin="linkedin.com/in/john", location="NYC",
            ),
        )
        result = score_presentation(resume, strong_resume_config)
        assert result.contact_completeness == 1.0

    def test_score_between_0_and_100(self, strong_resume_config):
        resume = _make_resume()
        result = score_presentation(resume, strong_resume_config)
        assert 0.0 <= result.score <= 100.0

    def test_grammar_errors_populated(self, strong_resume_config):
        resume = _make_resume(raw_text="Led a team  of engineers to  deliver a project")
        result = score_presentation(resume, strong_resume_config)
        assert len(result.grammar_errors) >= 1


# =============================================================================
# score_competencies
# =============================================================================

class TestScoreCompetencies:
    def test_many_skills_high_score(self, strong_resume_config):
        skills = "Python, Java, Go, TypeScript, AWS, Docker, Kubernetes, PostgreSQL, Redis, Machine Learning, Agile, Team Leadership, System Design, CI/CD"
        resume = _make_resume(skills_content=skills)
        # Config uses min_skills_for_full_score=15 but actual config.yaml lowered to 10
        result = score_competencies(resume, strong_resume_config)
        assert isinstance(result, CompetenciesScore)
        assert result.score >= 50.0
        assert result.skills_count >= 10
        assert result.category_count >= 2

    def test_few_skills_low_score(self, strong_resume_config):
        resume = _make_resume(skills_content="Computers, Office, Typing")
        result = score_competencies(resume, strong_resume_config)
        assert result.score < 30.0
        assert result.skills_count <= 5

    def test_no_skills_section(self, strong_resume_config):
        resume = _make_resume(sections=[
            ResumeSection(name="experience", heading="EXPERIENCE", content="...", bullets=["Led a project"]),
        ])
        result = score_competencies(resume, strong_resume_config)
        assert result.score == 0.0
        assert result.skills_count == 0

    def test_categories_covered(self, strong_resume_config):
        skills = "Python, Team Leadership, Presentation, Data Analysis, Agile"
        resume = _make_resume(skills_content=skills)
        result = score_competencies(resume, strong_resume_config)
        assert result.category_count >= 4
        assert "technical" in result.categories_covered
        assert "leadership" in result.categories_covered

    def test_score_between_0_and_100(self, strong_resume_config):
        resume = _make_resume(skills_content="Python, Java")
        result = score_competencies(resume, strong_resume_config)
        assert 0.0 <= result.score <= 100.0

    def test_skill_categories_dict_has_five_categories(self):
        assert len(SKILL_CATEGORIES) == 5
        expected = {"technical", "leadership", "communication", "analytical", "domain"}
        assert set(SKILL_CATEGORIES.keys()) == expected


# =============================================================================
# score_resume (aggregation)
# =============================================================================

class TestScoreResume:
    def test_full_resume_returns_resume_score(self, sample_parsed_resume):
        result = score_resume(sample_parsed_resume)
        assert isinstance(result, ResumeScore)
        assert isinstance(result.impact, ImpactScore)
        assert isinstance(result.presentation, PresentationScore)
        assert isinstance(result.competencies, CompetenciesScore)

    def test_overall_between_0_and_100(self, sample_parsed_resume):
        result = score_resume(sample_parsed_resume)
        assert 0.0 <= result.overall <= 100.0

    def test_zone_is_valid(self, sample_parsed_resume):
        result = score_resume(sample_parsed_resume)
        assert result.zone in (ScoreZone.GREEN, ScoreZone.YELLOW, ScoreZone.RED)

    def test_weights_sum_to_100(self, sample_parsed_resume):
        result = score_resume(sample_parsed_resume)
        assert sum(result.weights.values()) == 100

    def test_strong_resume_scores_above_50(self, sample_parsed_resume):
        result = score_resume(sample_parsed_resume)
        assert result.overall >= 50.0

    def test_weak_resume_scores_below_strong(self, sample_parsed_resume):
        weak = _make_resume(
            bullets=[
                "Helped maintain the systems",
                "Assisted team members",
                "Worked on various tasks",
            ],
            skills_content="Computers, Office",
            contact=ContactInfo(name="Jane"),
            raw_text="Helped maintain the systems\nAssisted team members\nWorked on various tasks",
        )
        strong_result = score_resume(sample_parsed_resume)
        weak_result = score_resume(weak)
        assert weak_result.overall < strong_result.overall

    def test_empty_resume_does_not_crash(self):
        empty = _make_resume(sections=[], raw_text="Nothing here")
        result = score_resume(empty)
        assert 0.0 <= result.overall <= 100.0
