"""Tests for resumatch.models module."""

from resumatch.models import (
    AnalysisResult,
    CompetenciesScore,
    ContactInfo,
    FeedbackCategory,
    FeedbackItem,
    FeedbackPriority,
    HealthStatus,
    ImpactScore,
    JDMatchResult,
    KeywordMatch,
    ParsedResume,
    PresentationScore,
    ResumeScore,
    ResumeSection,
    ScoreZone,
    SkillCategory,
)


class TestScoreZone:
    def test_green_zone(self):
        assert ScoreZone.GREEN == "green"

    def test_yellow_zone(self):
        assert ScoreZone.YELLOW == "yellow"

    def test_red_zone(self):
        assert ScoreZone.RED == "red"


class TestContactInfo:
    def test_empty_contact(self):
        c = ContactInfo()
        assert c.name is None
        assert c.email is None

    def test_full_contact(self):
        c = ContactInfo(
            name="John Smith",
            email="john@example.com",
            phone="555-1234",
            linkedin="linkedin.com/in/john",
            location="New York, NY",
            website="john.dev",
        )
        assert c.name == "John Smith"
        assert c.email == "john@example.com"


class TestResumeSection:
    def test_section_creation(self):
        s = ResumeSection(name="experience", heading="EXPERIENCE", content="Some content")
        assert s.name == "experience"
        assert s.bullets == []

    def test_section_with_bullets(self):
        s = ResumeSection(
            name="experience",
            heading="EXPERIENCE",
            content="Bullets below",
            bullets=["Led team of 5", "Increased revenue by 20%"],
        )
        assert len(s.bullets) == 2


class TestParsedResume:
    def test_empty_resume(self):
        r = ParsedResume(raw_text="")
        assert r.raw_text == ""
        assert r.sections == []
        assert r.page_count == 1

    def test_resume_with_sections(self, sample_parsed_resume):
        assert len(sample_parsed_resume.sections) == 3
        assert sample_parsed_resume.contact.name == "John Smith"


class TestImpactScore:
    def test_defaults(self):
        s = ImpactScore()
        assert s.score == 0.0
        assert s.action_verb_rate == 0.0

    def test_with_values(self):
        s = ImpactScore(score=85.0, action_verb_rate=0.9, quantification_rate=0.6)
        assert s.score == 85.0
        assert s.action_verb_rate == 0.9


class TestPresentationScore:
    def test_defaults(self):
        s = PresentationScore()
        assert s.grammar_errors == []
        assert s.missing_sections == []

    def test_with_errors(self):
        s = PresentationScore(
            score=70.0,
            grammar_errors=["Missing comma"],
            missing_sections=["skills"],
        )
        assert len(s.grammar_errors) == 1


class TestCompetenciesScore:
    def test_defaults(self):
        s = CompetenciesScore()
        assert s.skills_found == []
        assert s.skills_count == 0

    def test_with_skills(self):
        s = CompetenciesScore(
            score=80.0,
            skills_found=["Python", "AWS", "Leadership"],
            skills_count=3,
            categories_covered=["technical", "leadership"],
            category_count=2,
        )
        assert s.skills_count == 3
        assert s.category_count == 2


class TestResumeScore:
    def test_default_zone_is_red(self):
        s = ResumeScore()
        assert s.zone == ScoreZone.RED

    def test_weights_default_matches_config(self):
        """The model default must not drift from config.yaml's real weights."""
        from resumatch.config import CONFIG

        s = ResumeScore()
        assert s.weights == CONFIG["scoring"]["weights"]
        assert sum(s.weights.values()) == 100

    def test_custom_score(self):
        s = ResumeScore(overall=90.0, zone=ScoreZone.GREEN)
        assert s.overall == 90.0
        assert s.zone == ScoreZone.GREEN


class TestFeedbackItem:
    def test_rules_feedback(self):
        f = FeedbackItem(
            category=FeedbackCategory.IMPACT,
            priority=FeedbackPriority.HIGH,
            message="Add metrics to bullet points",
        )
        assert f.source == "rules"
        assert f.suggestion is None

    def test_llm_feedback(self):
        f = FeedbackItem(
            category=FeedbackCategory.PRESENTATION,
            priority=FeedbackPriority.MEDIUM,
            message="Consider restructuring",
            suggestion="Move skills section before experience",
            source="llm",
        )
        assert f.source == "llm"
        assert f.suggestion is not None


class TestKeywordMatch:
    def test_unmatched_keyword(self):
        k = KeywordMatch(keyword="kubernetes")
        assert k.found_in_resume is False
        assert k.similarity == 0.0

    def test_matched_keyword(self):
        k = KeywordMatch(keyword="python", found_in_resume=True, similarity=1.0, matched_term="Python")
        assert k.found_in_resume is True


class TestJDMatchResult:
    def test_empty_match(self):
        m = JDMatchResult()
        assert m.overall_match == 0.0
        assert m.matched_keywords == []

    def test_match_with_data(self):
        m = JDMatchResult(
            overall_match=0.75,
            keyword_match=0.8,
            semantic_match=0.7,
            jd_keywords=["python", "aws", "docker"],
            missing_keywords=["terraform"],
        )
        assert len(m.jd_keywords) == 3
        assert len(m.missing_keywords) == 1


class TestAnalysisResult:
    def test_analysis_result(self, sample_parsed_resume):
        result = AnalysisResult(
            resume=sample_parsed_resume,
            score=ResumeScore(overall=85.0, zone=ScoreZone.GREEN),
            feedback=[
                FeedbackItem(
                    category=FeedbackCategory.IMPACT,
                    priority=FeedbackPriority.LOW,
                    message="Good use of action verbs",
                )
            ],
        )
        assert result.score.overall == 85.0
        assert len(result.feedback) == 1
        assert result.jd_match is None


class TestHealthStatus:
    def test_default_health(self):
        h = HealthStatus()
        assert h.status == "ok"
        assert h.spacy_loaded is False
        assert h.version == "0.1.0"


class TestSkillCategory:
    def test_all_categories(self):
        assert len(SkillCategory) == 5
        assert SkillCategory.TECHNICAL == "technical"
        assert SkillCategory.LEADERSHIP == "leadership"


class TestFeedbackPriority:
    def test_priority_values(self):
        assert FeedbackPriority.HIGH == "high"
        assert FeedbackPriority.MEDIUM == "medium"
        assert FeedbackPriority.LOW == "low"
