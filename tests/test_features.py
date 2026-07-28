"""Tests for resumatch.features -- action verbs, quantification, specificity, grammar, formatting."""


import pytest

from resumatch.features import (
    STRONG_VERBS,
    WEAK_VERBS,
    _get_language_tool,
    analyze_action_verbs,
    analyze_quantification,
    analyze_specificity,
    check_formatting,
    check_grammar,
)
from resumatch.models import ResumeSection


# =============================================================================
# STRONG_VERBS / WEAK_VERBS sets
# =============================================================================

class TestVerbSets:
    def test_strong_verbs_is_nonempty_set(self):
        assert isinstance(STRONG_VERBS, set)
        assert len(STRONG_VERBS) >= 100

    def test_weak_verbs_is_nonempty_set(self):
        assert isinstance(WEAK_VERBS, set)
        assert len(WEAK_VERBS) >= 20

    def test_strong_verbs_all_lowercase(self):
        for verb in STRONG_VERBS:
            assert verb == verb.lower(), f"Strong verb {verb!r} should be lowercase"

    def test_weak_verbs_all_lowercase(self):
        for verb in WEAK_VERBS:
            assert verb == verb.lower(), f"Weak verb {verb!r} should be lowercase"

    def test_no_overlap_between_strong_and_weak(self):
        overlap = STRONG_VERBS & WEAK_VERBS
        assert not overlap, f"Verbs in both sets: {overlap}"

    def test_key_strong_verbs_present(self):
        expected = {"led", "managed", "developed", "implemented", "increased", "reduced",
                    "created", "designed", "built", "launched", "negotiated", "optimized",
                    "transformed", "orchestrated", "spearheaded", "accelerated",
                    "streamlined", "pioneered"}
        missing = expected - STRONG_VERBS
        assert not missing, f"Missing strong verbs: {missing}"

    def test_key_weak_verbs_present(self):
        expected = {"helped", "assisted", "worked", "responsible", "participated",
                    "was", "did", "made", "had", "got", "used", "tried"}
        missing = expected - WEAK_VERBS
        assert not missing, f"Missing weak verbs: {missing}"


# =============================================================================
# analyze_action_verbs
# =============================================================================

class TestAnalyzeActionVerbs:
    def test_all_strong_verbs(self):
        bullets = [
            "Led migration of monolithic application to microservices",
            "Managed team of 8 engineers",
            "Implemented automated testing pipeline",
            "Reduced cloud infrastructure costs by $200K",
        ]
        rate, weak = analyze_action_verbs(bullets)
        assert rate == 1.0
        assert weak == []

    def test_all_weak_verbs(self):
        bullets = [
            "Helped maintain the systems",
            "Assisted team members with their work",
            "Worked at Company A for 3 years",
            "Was responsible for various tasks",
        ]
        rate, weak = analyze_action_verbs(bullets)
        assert rate == 0.0
        assert len(weak) == 4

    def test_mixed_verbs(self):
        bullets = [
            "Led a team of 5 developers",
            "Helped with debugging",
            "Built a new CI/CD pipeline",
            "Worked on various projects",
        ]
        rate, weak = analyze_action_verbs(bullets)
        assert rate == 0.5
        assert "helped" in weak
        assert "worked" in weak

    def test_empty_bullets(self):
        rate, weak = analyze_action_verbs([])
        assert rate == 0.0
        assert weak == []

    def test_single_strong_bullet(self):
        rate, weak = analyze_action_verbs(["Developed a REST API"])
        assert rate == 1.0
        assert weak == []

    def test_single_weak_bullet(self):
        rate, weak = analyze_action_verbs(["Helped with testing"])
        assert rate == 0.0
        assert weak == ["helped"]

    def test_bullets_with_dash_prefix(self):
        bullets = ["- Led the engineering team", "- Built the deployment pipeline"]
        rate, weak = analyze_action_verbs(bullets)
        assert rate == 1.0

    def test_bullets_with_whitespace(self):
        bullets = ["  ", "", "  Led a project  "]
        rate, weak = analyze_action_verbs(bullets)
        # Only 1 non-empty bullet
        assert rate == 1.0

    def test_unknown_verb_not_counted_as_weak(self):
        bullets = ["Collaborated with the design team"]
        rate, weak = analyze_action_verbs(bullets)
        assert rate == 1.0
        assert weak == []

    def test_case_insensitive_matching(self):
        bullets = ["LED the team", "MANAGED the project"]
        rate, weak = analyze_action_verbs(bullets)
        assert rate == 1.0


# =============================================================================
# analyze_quantification
# =============================================================================

class TestAnalyzeQuantification:
    def test_dollar_amounts(self):
        bullets = ["Reduced costs by $200K annually", "Saved $50,000 in Q3"]
        rate, without = analyze_quantification(bullets)
        assert rate == 1.0
        assert without == []

    def test_percentages(self):
        bullets = ["Increased code coverage from 45% to 92%"]
        rate, without = analyze_quantification(bullets)
        assert rate == 1.0

    def test_multiplier(self):
        bullets = ["Achieved 3x improvement in deployment speed"]
        rate, without = analyze_quantification(bullets)
        assert rate == 1.0

    def test_large_numbers(self):
        bullets = ["Serving 10M+ requests daily", "Handling 500K events per second"]
        rate, without = analyze_quantification(bullets)
        assert rate == 1.0

    def test_no_metrics(self):
        bullets = [
            "Worked on various projects",
            "Helped with system maintenance",
        ]
        rate, without = analyze_quantification(bullets)
        assert rate == 0.0
        assert len(without) == 2

    def test_mixed_bullets(self):
        bullets = [
            "Led migration reducing deployment time by 75%",
            "Collaborated with product team on features",
            "Built pipeline handling 500K events per second",
        ]
        rate, without = analyze_quantification(bullets)
        assert abs(rate - 2 / 3) < 0.01
        assert len(without) == 1

    def test_empty_bullets(self):
        rate, without = analyze_quantification([])
        assert rate == 0.0
        assert without == []

    def test_plain_numbers_detected(self):
        bullets = ["Managed team of 8 engineers"]
        rate, without = analyze_quantification(bullets)
        assert rate == 1.0


# =============================================================================
# analyze_specificity
# =============================================================================

class TestAnalyzeSpecificity:
    def test_specific_bullets_score_high(self):
        bullets = [
            "Led migration of monolithic application to microservices, reducing deployment time by 75%",
            "Managed team of 8 engineers, delivering 3 major product releases on schedule",
        ]
        score = analyze_specificity(bullets)
        assert score > 0.6

    def test_vague_bullets_score_low(self):
        bullets = [
            "Worked on various tasks",
            "Was responsible for helping with projects",
            "Assisted in various projects",
            "Participated in day-to-day work",
        ]
        score = analyze_specificity(bullets)
        assert score < 0.4

    def test_empty_bullets_return_zero(self):
        assert analyze_specificity([]) == 0.0

    def test_single_vague_bullet(self):
        score = analyze_specificity(["Worked on various tasks"])
        assert score < 0.4

    def test_single_specific_bullet(self):
        score = analyze_specificity(["Reduced AWS infrastructure costs by $200K through container optimization"])
        assert score > 0.5

    def test_score_between_zero_and_one(self):
        bullets = ["Did some work", "Led a team of 50 engineers to deliver a $2M project"]
        score = analyze_specificity(bullets)
        assert 0.0 <= score <= 1.0


# =============================================================================
# check_grammar
# =============================================================================

class TestCheckGrammar:
    def test_double_spaces_detected(self):
        text = "Led a team  of engineers to  deliver a project"
        errors = check_grammar(text)
        assert any("double spaces" in e.lower() for e in errors)

    def test_inconsistent_endings_detected(self):
        text = "- Led a team of engineers.\n- Built a new pipeline\n- Deployed the application."
        errors = check_grammar(text)
        assert any("inconsistent" in e.lower() for e in errors)

    def test_consistent_endings_no_error(self):
        text = "- Led a team of engineers.\n- Built a new pipeline.\n- Deployed the application."
        errors = check_grammar(text)
        assert not any("inconsistent" in e.lower() for e in errors)

    def test_empty_text_no_errors(self):
        assert check_grammar("") == []
        assert check_grammar("   ") == []

    def test_clean_text_no_errors(self):
        text = "Led a team of engineers\nBuilt a new pipeline\nDeployed the application"
        errors = check_grammar(text)
        assert errors == []

    def test_real_grammar_errors_detected(self):
        """Bad grammar and spelling must be caught, not silently scored clean.

        Regression guard: check_grammar was once a stub that only looked at
        whitespace and bullet punctuation, so a resume full of errors scored
        full marks. Skipped when LanguageTool cannot start (no Java runtime);
        CI installs a JRE, so this runs there.
        """
        if _get_language_tool() is None:
            pytest.skip("LanguageTool unavailable -- no Java runtime")

        text = "He are a backend engineer. I has built many system. Builded a pipeline."
        errors = check_grammar(text)

        assert errors, "check_grammar found no errors in deliberately broken text"
        assert any("Builded" in e for e in errors), f"misspelling not flagged: {errors}"

    def test_clean_prose_still_passes_language_tool(self):
        """Resume bullets are fragments by design and must not be false-flagged."""
        if _get_language_tool() is None:
            pytest.skip("LanguageTool unavailable -- no Java runtime")

        text = (
            "- Led a team of 8 engineers to deliver a payments platform\n"
            "- Reduced p99 latency by 43% via query optimization\n"
            "- Owned CI/CD for 12 microservices"
        )
        assert check_grammar(text) == []

    def test_technology_names_are_not_spelling_errors(self):
        """Naming your tools must not cost you points.

        LanguageTool's dictionary is general English and does not contain most
        technology names, so an unfiltered spell check flags every one of them.
        """
        if _get_language_tool() is None:
            pytest.skip("LanguageTool unavailable -- no Java runtime")

        text = (
            "- Built services with FastAPI, Redis, and PostgreSQL on AWS\n"
            "- Ran pytest and numpy jobs on Kubernetes via kubectl\n"
            "- Migrated Northwind Systems to GraphQL and CI/CD\n"
        )
        assert check_grammar(text) == []

    def test_misspelling_at_bullet_start_still_caught(self):
        """The name filter keys off mid-sentence capitals, so a capitalised
        word opening a bullet must still be spell-checked."""
        if _get_language_tool() is None:
            pytest.skip("LanguageTool unavailable -- no Java runtime")

        errors = check_grammar("- Builded a pipeline\n- Managed a team\n")
        assert any("Builded" in e for e in errors), f"missed bullet-start typo: {errors}"


# =============================================================================
# check_formatting
# =============================================================================

class TestCheckFormatting:
    def test_ideal_one_page(self):
        sections = [ResumeSection(name="experience", heading="EXPERIENCE", content="...")]
        result = check_formatting(sections, page_count=1)
        assert result["page_score"] == 1.0

    def test_ideal_two_pages(self):
        sections = [ResumeSection(name="experience", heading="EXPERIENCE", content="...")]
        result = check_formatting(sections, page_count=2)
        assert result["page_score"] == 1.0

    def test_three_pages_penalized(self):
        sections = [ResumeSection(name="experience", heading="EXPERIENCE", content="...")]
        result = check_formatting(sections, page_count=3)
        assert result["page_score"] < 1.0

    def test_five_plus_pages_heavily_penalized(self):
        sections = [ResumeSection(name="experience", heading="EXPERIENCE", content="...")]
        result = check_formatting(sections, page_count=5)
        assert result["page_score"] <= 0.1

    def test_consistent_uppercase_headings(self):
        sections = [
            ResumeSection(name="experience", heading="EXPERIENCE", content="..."),
            ResumeSection(name="education", heading="EDUCATION", content="..."),
            ResumeSection(name="skills", heading="SKILLS", content="..."),
        ]
        result = check_formatting(sections, page_count=1)
        assert result["heading_consistency"] is True

    def test_inconsistent_headings(self):
        sections = [
            ResumeSection(name="experience", heading="EXPERIENCE", content="..."),
            ResumeSection(name="education", heading="Education", content="..."),
            ResumeSection(name="skills", heading="skills", content="..."),
        ]
        result = check_formatting(sections, page_count=1)
        assert result["heading_consistency"] is False

    def test_mixed_bullet_prefixes(self):
        sections = [
            ResumeSection(name="experience", heading="EXPERIENCE", content="...",
                          bullets=["- Led a team", "* Built a pipeline"]),
        ]
        result = check_formatting(sections, page_count=1)
        assert result["bullet_consistency"] is False

    def test_consistent_bullet_prefixes(self):
        sections = [
            ResumeSection(name="experience", heading="EXPERIENCE", content="...",
                          bullets=["- Led a team", "- Built a pipeline"]),
        ]
        result = check_formatting(sections, page_count=1)
        assert result["bullet_consistency"] is True

    def test_empty_sections(self):
        result = check_formatting([], page_count=1)
        assert result["page_score"] == 0.0

    def test_zero_pages(self):
        sections = [ResumeSection(name="experience", heading="EXPERIENCE", content="...")]
        result = check_formatting(sections, page_count=0)
        assert result["page_score"] == 0.0

    def test_date_consistency_with_mixed_formats(self):
        sections = [
            ResumeSection(name="experience", heading="EXPERIENCE",
                          content="Jan 2020 - Present at Company A, 06/2017 at Company B"),
        ]
        result = check_formatting(sections, page_count=1)
        assert result["date_consistency"] is False

    def test_date_consistency_with_uniform_formats(self):
        sections = [
            ResumeSection(name="experience", heading="EXPERIENCE",
                          content="Jan 2020 - Dec 2022\nJun 2017 - Dec 2019"),
        ]
        result = check_formatting(sections, page_count=1)
        assert result["date_consistency"] is True
