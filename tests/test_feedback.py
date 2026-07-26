"""Tests for resumatch.feedback module."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from resumatch.feedback import (
    generate_feedback,
    generate_impact_feedback,
    generate_presentation_feedback,
    generate_competencies_feedback,
    enhance_feedback_with_llm,
)
from resumatch.models import (
    CompetenciesScore,
    FeedbackCategory,
    FeedbackItem,
    FeedbackPriority,
    ImpactScore,
    PresentationScore,
    ResumeScore,
    ScoreZone,
)


class TestImpactFeedback:
    def test_low_action_verb_rate_high_priority(self):
        impact = ImpactScore(action_verb_rate=0.3, quantification_rate=0.5, specificity_score=0.7)
        items = generate_impact_feedback(impact)
        verb_items = [i for i in items if "action verb" in i.message.lower()]
        assert len(verb_items) >= 1
        assert verb_items[0].priority == FeedbackPriority.HIGH

    def test_medium_action_verb_rate(self):
        impact = ImpactScore(action_verb_rate=0.7, quantification_rate=0.6, specificity_score=0.7)
        items = generate_impact_feedback(impact)
        verb_items = [i for i in items if "action verb" in i.message.lower()]
        assert len(verb_items) >= 1
        assert verb_items[0].priority == FeedbackPriority.MEDIUM

    def test_good_action_verb_rate_no_verb_feedback(self):
        impact = ImpactScore(action_verb_rate=0.95, quantification_rate=0.7, specificity_score=0.8)
        items = generate_impact_feedback(impact)
        verb_items = [i for i in items if "action verb" in i.message.lower()]
        assert len(verb_items) == 0

    def test_weak_verbs_detected(self):
        impact = ImpactScore(weak_verbs_found=["helped", "assisted", "worked"])
        items = generate_impact_feedback(impact)
        weak_items = [i for i in items if "weak verb" in i.message.lower()]
        assert len(weak_items) >= 1

    def test_low_quantification_high_priority(self):
        impact = ImpactScore(quantification_rate=0.1, action_verb_rate=0.9, specificity_score=0.7)
        items = generate_impact_feedback(impact)
        quant_items = [i for i in items if "quantif" in i.message.lower()]
        assert len(quant_items) >= 1
        assert quant_items[0].priority == FeedbackPriority.HIGH

    def test_bullets_without_metrics(self):
        impact = ImpactScore(
            bullets_without_metrics=["Managed team projects", "Worked on deliverables"],
            action_verb_rate=0.9,
            specificity_score=0.7,
        )
        items = generate_impact_feedback(impact)
        metric_items = [i for i in items if "lack" in i.message.lower() and "outcome" in i.message.lower()]
        assert len(metric_items) >= 1

    def test_low_specificity(self):
        impact = ImpactScore(specificity_score=0.2, action_verb_rate=0.9, quantification_rate=0.6)
        items = generate_impact_feedback(impact)
        spec_items = [i for i in items if "vague" in i.message.lower() or "specific" in i.message.lower()]
        assert len(spec_items) >= 1


class TestPresentationFeedback:
    def test_bad_page_length(self):
        pres = PresentationScore(page_length_score=0.3)
        items = generate_presentation_feedback(pres)
        page_items = [i for i in items if "page" in i.message.lower()]
        assert len(page_items) >= 1

    def test_grammar_errors(self):
        pres = PresentationScore(grammar_errors=["Missing comma", "Spelling error", "Run-on sentence"])
        items = generate_presentation_feedback(pres)
        grammar_items = [i for i in items if "grammar" in i.message.lower()]
        assert len(grammar_items) >= 1

    def test_many_grammar_errors_high_priority(self):
        pres = PresentationScore(grammar_errors=[f"Error {i}" for i in range(8)])
        items = generate_presentation_feedback(pres)
        grammar_items = [i for i in items if "grammar" in i.message.lower()]
        assert grammar_items[0].priority == FeedbackPriority.HIGH

    def test_missing_sections(self):
        pres = PresentationScore(missing_sections=["skills", "education"])
        items = generate_presentation_feedback(pres)
        section_items = [i for i in items if "missing" in i.message.lower() and "section" in i.message.lower()]
        assert len(section_items) >= 1

    def test_incomplete_contact(self):
        pres = PresentationScore(contact_completeness=0.3)
        items = generate_presentation_feedback(pres)
        contact_items = [i for i in items if "contact" in i.message.lower()]
        assert len(contact_items) >= 1

    def test_formatting_issues(self):
        pres = PresentationScore(formatting_score=0.3)
        items = generate_presentation_feedback(pres)
        format_items = [i for i in items if "format" in i.message.lower()]
        assert len(format_items) >= 1


class TestCompetenciesFeedback:
    def test_few_skills(self):
        comp = CompetenciesScore(skills_count=3, category_count=2)
        items = generate_competencies_feedback(comp)
        skill_items = [i for i in items if "skill" in i.message.lower()]
        assert len(skill_items) >= 1
        assert skill_items[0].priority == FeedbackPriority.HIGH

    def test_low_category_coverage(self):
        comp = CompetenciesScore(skills_count=10, category_count=1, categories_covered=["technical"])
        items = generate_competencies_feedback(comp)
        cat_items = [i for i in items if "categor" in i.message.lower()]
        assert len(cat_items) >= 1

    def test_low_relevance(self):
        comp = CompetenciesScore(skills_count=10, category_count=4, relevance_score=0.2)
        items = generate_competencies_feedback(comp)
        rel_items = [i for i in items if "standard" in i.message.lower() or "relevance" in i.message.lower()]
        assert len(rel_items) >= 1


class TestGenerateFeedback:
    def test_green_zone_feedback(self):
        score = ResumeScore(
            overall=90.0,
            zone=ScoreZone.GREEN,
            impact=ImpactScore(action_verb_rate=0.95, quantification_rate=0.7, specificity_score=0.8),
            presentation=PresentationScore(page_length_score=1.0, section_completeness=1.0, contact_completeness=1.0, formatting_score=0.9),
            competencies=CompetenciesScore(skills_count=15, category_count=5, relevance_score=0.9),
        )
        items = generate_feedback(score)
        general_items = [i for i in items if i.category == FeedbackCategory.GENERAL]
        assert any("green" in i.message.lower() for i in general_items)

    def test_red_zone_feedback(self):
        score = ResumeScore(
            overall=30.0,
            zone=ScoreZone.RED,
            impact=ImpactScore(action_verb_rate=0.2, quantification_rate=0.1, specificity_score=0.2),
            presentation=PresentationScore(missing_sections=["skills", "education"], contact_completeness=0.2),
            competencies=CompetenciesScore(skills_count=2, category_count=1),
        )
        items = generate_feedback(score)
        high_items = [i for i in items if i.priority == FeedbackPriority.HIGH]
        assert len(high_items) >= 3

    def test_feedback_sorted_by_priority(self):
        score = ResumeScore(
            overall=50.0,
            impact=ImpactScore(action_verb_rate=0.3, quantification_rate=0.1, specificity_score=0.3),
            presentation=PresentationScore(formatting_score=0.3, contact_completeness=0.3),
            competencies=CompetenciesScore(skills_count=3, category_count=1),
        )
        items = generate_feedback(score)
        priorities = [i.priority for i in items]
        priority_order = {FeedbackPriority.HIGH: 0, FeedbackPriority.MEDIUM: 1, FeedbackPriority.LOW: 2}
        numeric = [priority_order[p] for p in priorities]
        assert numeric == sorted(numeric)

    def test_all_feedback_has_category(self):
        score = ResumeScore(overall=50.0, impact=ImpactScore(action_verb_rate=0.3))
        items = generate_feedback(score)
        for item in items:
            assert item.category in FeedbackCategory

    def test_all_feedback_has_message(self):
        score = ResumeScore(overall=50.0)
        items = generate_feedback(score)
        for item in items:
            assert item.message


class TestEnhanceFeedbackWithLLM:
    @pytest.mark.asyncio
    async def test_llm_disabled_returns_original(self):
        feedback = [FeedbackItem(category=FeedbackCategory.IMPACT, priority=FeedbackPriority.HIGH, message="test")]
        with patch("resumatch.feedback.CONFIG", {"llm": {"enabled": False}}):
            result = await enhance_feedback_with_llm(feedback, "resume text", {})
        assert result == feedback

    @pytest.mark.asyncio
    async def test_llm_unavailable_returns_original(self):
        feedback = [FeedbackItem(category=FeedbackCategory.IMPACT, priority=FeedbackPriority.HIGH, message="test")]
        mock_module = MagicMock()
        mock_module.is_llm_available = AsyncMock(return_value=False)
        mock_module.get_client = MagicMock()
        with patch("resumatch.feedback.CONFIG", {"llm": {"enabled": True}}):
            with patch.dict("sys.modules", {"resumatch.llm": mock_module}):
                result = await enhance_feedback_with_llm(feedback, "resume text", {})
        assert result == feedback

    @pytest.mark.asyncio
    async def test_llm_import_error_returns_original(self):
        feedback = [FeedbackItem(category=FeedbackCategory.IMPACT, priority=FeedbackPriority.HIGH, message="test")]
        with patch("resumatch.feedback.CONFIG", {"llm": {"enabled": True}}):
            with patch.dict("sys.modules", {"resumatch.llm": None}):
                result = await enhance_feedback_with_llm(feedback, "resume text", {})
        assert result == feedback
