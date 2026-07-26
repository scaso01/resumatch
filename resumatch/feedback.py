"""
ResuMatch - Feedback Generation
=================================
Rules-based feedback engine with optional LLM enhancement.
Generates prioritized, actionable suggestions based on resume scores.
"""

from __future__ import annotations

from resumatch.config import CONFIG, logger
from resumatch.models import (
    FeedbackCategory,
    FeedbackItem,
    FeedbackPriority,
    ImpactScore,
    PresentationScore,
    CompetenciesScore,
    ResumeScore,
    ParsedResume,
)


def generate_impact_feedback(impact: ImpactScore) -> list[FeedbackItem]:
    """Generate feedback for the Impact module."""
    items = []

    # Action verb feedback
    if impact.action_verb_rate < 0.5:
        items.append(FeedbackItem(
            category=FeedbackCategory.IMPACT,
            priority=FeedbackPriority.HIGH,
            message=f"Only {impact.action_verb_rate:.0%} of your bullet points start with strong action verbs.",
            suggestion="Start each bullet with a powerful verb like Led, Developed, Implemented, Increased, Reduced, Designed, Built, or Launched.",
        ))
    elif impact.action_verb_rate < 0.9:
        items.append(FeedbackItem(
            category=FeedbackCategory.IMPACT,
            priority=FeedbackPriority.MEDIUM,
            message=f"{impact.action_verb_rate:.0%} of bullets start with strong action verbs — aim for 90%+.",
            suggestion="Replace weaker verbs like 'Helped', 'Assisted', 'Worked on' with specific action verbs.",
        ))

    # Weak verbs found
    if impact.weak_verbs_found:
        unique_weak = sorted(set(impact.weak_verbs_found))[:5]
        items.append(FeedbackItem(
            category=FeedbackCategory.IMPACT,
            priority=FeedbackPriority.HIGH,
            message=f"Weak verbs detected: {', '.join(unique_weak)}.",
            suggestion="Replace each weak verb with a specific, results-oriented alternative.",
        ))

    # Quantification feedback
    if impact.quantification_rate < 0.3:
        items.append(FeedbackItem(
            category=FeedbackCategory.IMPACT,
            priority=FeedbackPriority.HIGH,
            message=f"Only {impact.quantification_rate:.0%} of bullets contain quantified results.",
            suggestion="Add numbers, percentages, dollar amounts, or timeframes to every achievement. Example: 'Reduced costs by $200K annually' instead of 'Reduced costs'.",
        ))
    elif impact.quantification_rate < 0.6:
        items.append(FeedbackItem(
            category=FeedbackCategory.IMPACT,
            priority=FeedbackPriority.MEDIUM,
            message=f"{impact.quantification_rate:.0%} of bullets are quantified — target 60%+.",
            suggestion="Think about scale (team size, users), speed (time saved), money (revenue, cost savings), and quality (error reduction, satisfaction scores).",
        ))

    # Bullets without metrics
    if impact.bullets_without_metrics:
        count = len(impact.bullets_without_metrics)
        example = impact.bullets_without_metrics[0][:80]
        items.append(FeedbackItem(
            category=FeedbackCategory.IMPACT,
            priority=FeedbackPriority.MEDIUM,
            message=f"{count} bullet(s) lack measurable outcomes.",
            suggestion=f"Add a metric to: \"{example}...\"",
        ))

    # Specificity feedback
    if impact.specificity_score < 0.4:
        items.append(FeedbackItem(
            category=FeedbackCategory.IMPACT,
            priority=FeedbackPriority.HIGH,
            message="Bullet points are too vague — they lack specific details.",
            suggestion="Replace generic phrases like 'various tasks' or 'responsible for' with specific technologies, project names, and outcomes.",
        ))
    elif impact.specificity_score < 0.7:
        items.append(FeedbackItem(
            category=FeedbackCategory.IMPACT,
            priority=FeedbackPriority.MEDIUM,
            message="Some bullet points could be more specific.",
            suggestion="Name the tools, technologies, frameworks, and methodologies you used.",
        ))

    return items


def generate_presentation_feedback(presentation: PresentationScore) -> list[FeedbackItem]:
    """Generate feedback for the Presentation module."""
    items = []

    # Page length
    if presentation.page_length_score < 0.5:
        items.append(FeedbackItem(
            category=FeedbackCategory.PRESENTATION,
            priority=FeedbackPriority.MEDIUM,
            message="Resume length is outside the ideal 1-2 page range.",
            suggestion="Aim for 1 page (early career) or 2 pages (10+ years experience). Remove outdated or irrelevant entries.",
        ))

    # Grammar
    if presentation.grammar_errors:
        error_count = len(presentation.grammar_errors)
        severity = FeedbackPriority.HIGH if error_count > 5 else FeedbackPriority.MEDIUM
        items.append(FeedbackItem(
            category=FeedbackCategory.PRESENTATION,
            priority=severity,
            message=f"Found {error_count} grammar/spelling issue(s).",
            suggestion=f"Fix: {presentation.grammar_errors[0]}" if presentation.grammar_errors else None,
        ))

    # Missing sections
    if presentation.missing_sections:
        items.append(FeedbackItem(
            category=FeedbackCategory.PRESENTATION,
            priority=FeedbackPriority.HIGH,
            message=f"Missing required section(s): {', '.join(presentation.missing_sections)}.",
            suggestion="Add the missing sections. Most resumes need Experience, Education, and Skills at minimum.",
        ))

    # Section completeness
    if presentation.section_completeness < 0.6:
        items.append(FeedbackItem(
            category=FeedbackCategory.PRESENTATION,
            priority=FeedbackPriority.HIGH,
            message="Resume is missing several standard sections.",
            suggestion="A complete resume typically includes: Summary, Experience, Education, Skills, and optionally Certifications/Projects.",
        ))

    # Contact completeness
    if presentation.contact_completeness < 0.6:
        items.append(FeedbackItem(
            category=FeedbackCategory.PRESENTATION,
            priority=FeedbackPriority.MEDIUM,
            message="Contact information is incomplete.",
            suggestion="Include: full name, email, phone number, LinkedIn URL, and city/state.",
        ))

    # Formatting
    if presentation.formatting_score < 0.5:
        items.append(FeedbackItem(
            category=FeedbackCategory.PRESENTATION,
            priority=FeedbackPriority.LOW,
            message="Formatting inconsistencies detected.",
            suggestion="Use consistent date formats, bullet styles, and heading capitalization throughout.",
        ))

    return items


def generate_competencies_feedback(competencies: CompetenciesScore) -> list[FeedbackItem]:
    """Generate feedback for the Competencies module."""
    items = []

    # Skills breadth
    if competencies.skills_count < 5:
        items.append(FeedbackItem(
            category=FeedbackCategory.COMPETENCIES,
            priority=FeedbackPriority.HIGH,
            message=f"Only {competencies.skills_count} distinct skills identified.",
            suggestion="List 10-15 relevant skills. Include technical skills, tools, frameworks, and methodologies.",
        ))
    elif competencies.skills_count < 10:
        items.append(FeedbackItem(
            category=FeedbackCategory.COMPETENCIES,
            priority=FeedbackPriority.MEDIUM,
            message=f"{competencies.skills_count} skills found — consider adding more.",
            suggestion="Expand your skills section with industry-standard keywords that match your experience.",
        ))

    # Category coverage
    if competencies.category_count < 3:
        missing_categories = {"technical", "leadership", "communication", "analytical", "domain"} - set(competencies.categories_covered)
        items.append(FeedbackItem(
            category=FeedbackCategory.COMPETENCIES,
            priority=FeedbackPriority.MEDIUM,
            message=f"Skills only cover {competencies.category_count} categories.",
            suggestion=f"Add skills in: {', '.join(sorted(missing_categories)[:3])}.",
        ))

    # Relevance
    if competencies.relevance_score < 0.4:
        items.append(FeedbackItem(
            category=FeedbackCategory.COMPETENCIES,
            priority=FeedbackPriority.MEDIUM,
            message="Skills listed may not be industry-standard keywords.",
            suggestion="Use recognized skill names (e.g., 'Python' not 'coding', 'Agile' not 'flexible work style').",
        ))

    return items


def generate_feedback(score: ResumeScore, resume: ParsedResume | None = None) -> list[FeedbackItem]:
    """Generate all rules-based feedback from a ResumeScore.

    Args:
        score: The scored resume.
        resume: Optional parsed resume for additional context.

    Returns:
        List of FeedbackItem sorted by priority (HIGH first).
    """
    items = []

    items.extend(generate_impact_feedback(score.impact))
    items.extend(generate_presentation_feedback(score.presentation))
    items.extend(generate_competencies_feedback(score.competencies))

    # Overall feedback
    if score.overall >= 85:
        items.append(FeedbackItem(
            category=FeedbackCategory.GENERAL,
            priority=FeedbackPriority.LOW,
            message=f"Strong resume! Overall score: {score.overall:.0f}/100 (Green zone).",
            suggestion="Fine-tune with the specific suggestions above to push even higher.",
        ))
    elif score.overall >= 50:
        items.append(FeedbackItem(
            category=FeedbackCategory.GENERAL,
            priority=FeedbackPriority.MEDIUM,
            message=f"Good foundation. Overall score: {score.overall:.0f}/100 (Yellow zone).",
            suggestion="Focus on the HIGH priority items above to reach the Green zone (85+).",
        ))
    else:
        items.append(FeedbackItem(
            category=FeedbackCategory.GENERAL,
            priority=FeedbackPriority.HIGH,
            message=f"Resume needs significant improvement. Overall score: {score.overall:.0f}/100 (Red zone).",
            suggestion="Start with the HIGH priority items — especially adding metrics to bullets and completing missing sections.",
        ))

    # Sort by priority: HIGH > MEDIUM > LOW
    priority_order = {FeedbackPriority.HIGH: 0, FeedbackPriority.MEDIUM: 1, FeedbackPriority.LOW: 2}
    items.sort(key=lambda x: priority_order.get(x.priority, 99))

    return items


async def enhance_feedback_with_llm(
    feedback: list[FeedbackItem],
    resume_text: str,
    scores: dict,
) -> list[FeedbackItem]:
    """Optionally enhance feedback using LLM for personalized suggestions.

    Falls back gracefully to rules-only feedback if LLM unavailable.
    """
    if not CONFIG.get("llm", {}).get("enabled", False):
        return feedback

    try:
        from resumatch.llm import get_client, is_llm_available

        if not await is_llm_available():
            logger.warning("LLM unavailable — using rules-only feedback")
            return feedback

        client = get_client()
        llm_feedback_raw = await client.generate_feedback(resume_text, scores)

        for item_dict in llm_feedback_raw:
            try:
                feedback.append(FeedbackItem(
                    category=FeedbackCategory(item_dict.get("category", "general")),
                    priority=FeedbackPriority(item_dict.get("priority", "medium")),
                    message=item_dict.get("message", ""),
                    suggestion=item_dict.get("suggestion"),
                    source="llm",
                ))
            except (ValueError, KeyError) as e:
                logger.warning("Skipping malformed LLM feedback item: %s", e)

    except Exception as e:
        logger.warning("LLM feedback enhancement failed: %s", e)

    return feedback
