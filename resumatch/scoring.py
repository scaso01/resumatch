"""
ResuMatch - Scoring Engine
============================
Three-module scoring: Impact, Presentation, Competencies.
Aggregates into overall score with zone classification.
"""

from __future__ import annotations

import re

from resumatch.config import CONFIG, logger
from resumatch.features import (
    analyze_action_verbs,
    analyze_quantification,
    analyze_specificity,
    check_formatting,
    check_grammar,
)
from resumatch.models import (
    CompetenciesScore,
    ImpactScore,
    ParsedResume,
    PresentationScore,
    ResumeScore,
    ScoreZone,
)

# ---------------------------------------------------------------------------
# Skills Categorization
# ---------------------------------------------------------------------------

SKILL_CATEGORIES: dict[str, set[str]] = {
    "technical": {
        "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#",
        "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql", "html",
        "css", "react", "angular", "vue", "node", "django", "flask", "fastapi",
        "spring", "express", "docker", "kubernetes", "aws", "azure", "gcp",
        "terraform", "ci/cd", "git", "linux", "postgresql", "mongodb", "redis",
        "mysql", "elasticsearch", "graphql", "rest", "api", "machine learning",
        "deep learning", "nlp", "data science", "tensorflow", "pytorch",
        "pandas", "numpy", "spark", "hadoop", "kafka", "rabbitmq", "jenkins",
        "github actions", "ansible", "nginx", "microservices", "system design",
        "ec2", "s3", "lambda", "rds", "ecs", "dynamodb", "cloudformation",
        "next.js", "rails", "ruby on rails", ".net", "asp.net", "svelte",
        "tailwind", "sass", "webpack", "vite",
        "jira", "confluence", "figma", "notion", "slack",
        "airflow", "dbt", "snowflake", "databricks", "bigquery",
        "prometheus", "grafana", "datadog",
    },
    "leadership": {
        "team leadership", "mentoring", "coaching", "strategic planning",
        "stakeholder management", "project management", "program management",
        "people management", "cross-functional", "executive communication",
        "change management", "decision making", "conflict resolution",
        "performance management", "talent development", "succession planning",
        "organizational development", "budget management", "vendor management",
    },
    "communication": {
        "presentation", "public speaking", "technical writing", "documentation",
        "client relations", "negotiation", "facilitation", "storytelling",
        "copywriting", "editing", "proposal writing", "report writing",
        "interpersonal skills", "active listening", "cross-cultural communication",
    },
    "analytical": {
        "data analysis", "problem solving", "critical thinking", "research",
        "statistical analysis", "financial analysis", "business analysis",
        "process improvement", "root cause analysis", "forecasting",
        "quantitative analysis", "market research", "competitive analysis",
        "requirements analysis", "gap analysis", "trend analysis",
    },
    "domain": {
        "agile", "scrum", "devops", "cloud computing", "cybersecurity",
        "compliance", "risk management", "product management", "ux design",
        "mobile development", "web development", "e-commerce", "fintech",
        "healthcare it", "saas", "b2b", "b2c", "digital transformation",
        "quality assurance", "testing", "automation", "data engineering",
        "site reliability", "information security", "blockchain",
        "machine learning ops", "mlops", "data governance", "api design",
        "event-driven", "serverless",
    },
}

# Flattened lookup: skill_name -> category_name
_SKILL_TO_CATEGORY: dict[str, str] = {}
for _cat, _skills in SKILL_CATEGORIES.items():
    for _skill in _skills:
        _SKILL_TO_CATEGORY[_skill] = _cat


# ---------------------------------------------------------------------------
# classify_zone
# ---------------------------------------------------------------------------

def classify_zone(score: float) -> ScoreZone:
    """Classify a numeric score into green/yellow/red zone using CONFIG thresholds."""
    zones = CONFIG.get("scoring", {}).get("zones", {})
    green_threshold = zones.get("green", 85)
    yellow_threshold = zones.get("yellow", 50)

    if score >= green_threshold:
        return ScoreZone.GREEN
    elif score >= yellow_threshold:
        return ScoreZone.YELLOW
    else:
        return ScoreZone.RED


# ---------------------------------------------------------------------------
# score_impact
# ---------------------------------------------------------------------------

def score_impact(parsed: ParsedResume, config: dict) -> ImpactScore:
    """Score resume impact from experience bullets.

    Weights: action_verb_rate (40%), quantification_rate (35%), specificity (25%).
    """
    # Gather all experience bullets
    experience_bullets: list[str] = []
    for section in parsed.sections:
        if section.name.lower() in ("experience", "work experience", "work history", "employment"):
            experience_bullets.extend(section.bullets)

    if not experience_bullets:
        # Fallback: analyze experience section content directly (PDF may strip bullets)
        for section in parsed.sections:
            if section.name.lower() in ("experience", "work experience", "work history", "employment"):
                if section.content.strip():
                    experience_bullets = [
                        line.strip() for line in section.content.splitlines()
                        if line.strip() and len(line.strip()) >= 10
                    ]
                    break
        if not experience_bullets:
            logger.debug("No experience content found for impact scoring")
            return ImpactScore()
        logger.debug("Using %d content lines as fallback bullets", len(experience_bullets))

    action_verb_rate, weak_verbs_found = analyze_action_verbs(experience_bullets)

    # For quantification, also check role summary paragraphs — they are accomplishment
    # statements that should contain metrics, even though they aren't formatted as bullets.
    all_accomplishment_lines = list(experience_bullets)
    for section in parsed.sections:
        if section.name.lower() in ("experience", "work experience", "work history", "employment"):
            bullet_set = set(section.bullets)
            for line in section.content.splitlines():
                text = line.lstrip("\u2022 ").strip()
                if text and len(text) >= 40 and text not in bullet_set and text[0].isupper():
                    all_accomplishment_lines.append(text)

    quantification_rate, bullets_without_metrics = analyze_quantification(all_accomplishment_lines)
    specificity_score = analyze_specificity(experience_bullets)

    # Weighted composite with power curve to soften harsh linear scoring.
    # rate ** 0.6 maps: 0.8 -> ~0.88, 0.6 -> ~0.74, 0.4 -> ~0.57
    raw = (
        (action_verb_rate ** 0.6) * 0.40
        + (quantification_rate ** 0.6) * 0.35
        + (specificity_score ** 0.6) * 0.25
    ) * 100.0

    score = max(0.0, min(100.0, raw))

    logger.debug(
        "Impact score: %.1f (verbs=%.2f, quant=%.2f, spec=%.2f)",
        score, action_verb_rate, quantification_rate, specificity_score,
    )

    return ImpactScore(
        score=round(score, 1),
        action_verb_rate=round(action_verb_rate, 3),
        quantification_rate=round(quantification_rate, 3),
        specificity_score=round(specificity_score, 3),
        weak_verbs_found=weak_verbs_found,
        bullets_without_metrics=bullets_without_metrics,
    )


# ---------------------------------------------------------------------------
# score_presentation
# ---------------------------------------------------------------------------

def score_presentation(parsed: ParsedResume, config: dict) -> PresentationScore:
    """Score resume presentation quality.

    Weights: page_length (20%), grammar (30%), section_completeness (25%),
             formatting (15%), contact_completeness (10%).
    """
    pres_config = config.get("scoring", {}).get("presentation", {})

    # -- Page length score (0-1) --
    ideal_min = pres_config.get("ideal_pages_min", 1)
    ideal_max = pres_config.get("ideal_pages_max", 2)
    if ideal_min <= parsed.page_count <= ideal_max:
        page_score = 1.0
    elif parsed.page_count == 0:
        page_score = 0.0
    elif parsed.page_count == ideal_max + 1:
        page_score = 0.6
    elif parsed.page_count == ideal_max + 2:
        page_score = 0.3
    else:
        page_score = 0.1

    # -- Grammar score (0-1) --
    grammar_errors = check_grammar(parsed.raw_text)
    penalty_per = pres_config.get("grammar_penalty_per_error", 2)
    penalty_cap = pres_config.get("grammar_penalty_cap", 20)
    grammar_penalty = min(len(grammar_errors) * penalty_per, penalty_cap)
    grammar_score = max(0.0, (100.0 - grammar_penalty)) / 100.0

    # -- Section completeness (0-1) --
    required_sections = pres_config.get("required_sections", ["experience", "education", "skills"])
    found_sections = {s.name.lower() for s in parsed.sections}
    missing_sections = [rs for rs in required_sections if rs.lower() not in found_sections]
    if required_sections:
        section_completeness = (len(required_sections) - len(missing_sections)) / len(required_sections)
    else:
        section_completeness = 1.0

    # -- Formatting score (0-1) --
    fmt = check_formatting(parsed.sections, parsed.page_count)
    fmt_checks = [
        fmt.get("date_consistency", True),
        fmt.get("bullet_consistency", True),
        fmt.get("heading_consistency", True),
    ]
    fmt_pass_rate = sum(1 for c in fmt_checks if c) / len(fmt_checks) if fmt_checks else 1.0
    formatting_score = (fmt_pass_rate * 0.6) + (fmt.get("page_score", 1.0) * 0.4)

    # -- Contact completeness (0-1) --
    contact = parsed.contact
    contact_fields = [contact.name, contact.email, contact.phone, contact.linkedin, contact.location]
    filled = sum(1 for f in contact_fields if f)
    contact_completeness = filled / len(contact_fields)

    # -- Weighted composite --
    raw = (
        page_score * 0.20
        + grammar_score * 0.30
        + section_completeness * 0.25
        + formatting_score * 0.15
        + contact_completeness * 0.10
    ) * 100.0

    score = max(0.0, min(100.0, raw))

    logger.debug(
        "Presentation score: %.1f (page=%.2f, grammar=%.2f, sections=%.2f, fmt=%.2f, contact=%.2f)",
        score, page_score, grammar_score, section_completeness, formatting_score, contact_completeness,
    )

    return PresentationScore(
        score=round(score, 1),
        page_length_score=round(page_score, 3),
        grammar_score=round(grammar_score, 3),
        section_completeness=round(section_completeness, 3),
        formatting_score=round(formatting_score, 3),
        contact_completeness=round(contact_completeness, 3),
        grammar_errors=grammar_errors,
        missing_sections=missing_sections,
    )


# ---------------------------------------------------------------------------
# score_competencies
# ---------------------------------------------------------------------------

def _extract_skills(parsed: ParsedResume) -> list[str]:
    """Extract skill strings from skills section + scan entire resume for skill evidence."""
    skills: list[str] = []

    # 1. Explicit skills from skills section(s) — split on delimiters
    for section in parsed.sections:
        if section.name.lower() in ("skills", "technical skills", "core competencies", "competencies"):
            content = section.content
            raw_skills: list[str] = []
            for delimiter in [",", ";", "|", "\n"]:
                if delimiter in content:
                    raw_skills = [s.strip() for s in content.split(delimiter) if s.strip()]
                    break
            if not raw_skills:
                raw_skills = [s.strip() for s in content.split() if s.strip()]
            skills.extend(raw_skills)

    # 2. Scan all sections for known skill keywords (experience bullets, projects, etc.)
    #    This catches competency evidence mentioned in context, not just listed explicitly.
    all_text = " ".join(
        section.content for section in parsed.sections
    ).lower()
    for skill_name in _SKILL_TO_CATEGORY:
        # Use word-boundary matching to avoid false positives (e.g. "r" in "for")
        if re.search(r"\b" + re.escape(skill_name) + r"\b", all_text):
            skills.append(skill_name)

    return skills


def _categorize_skill(skill_lower: str) -> str | None:
    """Match a skill string to a category. Returns category name or None."""
    # Direct match
    if skill_lower in _SKILL_TO_CATEGORY:
        return _SKILL_TO_CATEGORY[skill_lower]

    # Substring match (e.g., "team leadership skills" matches "team leadership")
    for cat_skill, cat_name in _SKILL_TO_CATEGORY.items():
        if cat_skill in skill_lower or skill_lower in cat_skill:
            return cat_name

    return None


def score_competencies(parsed: ParsedResume, config: dict) -> CompetenciesScore:
    """Score skills breadth (40%), category coverage (35%), relevance (25%).

    Relevance is measured by how many skills match known categories.
    """
    comp_config = config.get("scoring", {}).get("competencies", {})
    min_skills = comp_config.get("min_skills_for_full_score", 15)
    min_categories = comp_config.get("min_categories_for_full_score", 4)

    raw_skills = _extract_skills(parsed)
    skills_lower = [s.lower() for s in raw_skills]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_skills: list[str] = []
    for s in raw_skills:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            unique_skills.append(s)

    # Categorize skills
    categories_found: set[str] = set()
    categorized_count = 0
    for skill in skills_lower:
        cat = _categorize_skill(skill)
        if cat:
            categories_found.add(cat)
            categorized_count += 1

    skills_count = len(unique_skills)
    category_count = len(categories_found)

    # Sub-scores (0-1)
    breadth = min(skills_count / min_skills, 1.0) if min_skills > 0 else 1.0
    coverage = min(category_count / min_categories, 1.0) if min_categories > 0 else 1.0
    relevance = categorized_count / len(skills_lower) if skills_lower else 0.0

    # Weighted composite
    raw_score = (
        breadth * 0.40
        + coverage * 0.35
        + relevance * 0.25
    ) * 100.0

    score = max(0.0, min(100.0, raw_score))

    logger.debug(
        "Competencies score: %.1f (breadth=%.2f, coverage=%.2f, relevance=%.2f, skills=%d, cats=%d)",
        score, breadth, coverage, relevance, skills_count, category_count,
    )

    return CompetenciesScore(
        score=round(score, 1),
        skills_found=unique_skills,
        skills_count=skills_count,
        categories_covered=sorted(categories_found),
        category_count=category_count,
        relevance_score=round(relevance, 3),
    )


# ---------------------------------------------------------------------------
# score_resume (aggregator)
# ---------------------------------------------------------------------------

def score_resume(parsed: ParsedResume) -> ResumeScore:
    """Aggregate all three module scores into overall score with zone classification."""
    weights = CONFIG.get("scoring", {}).get("weights", {})
    impact_weight = weights.get("impact", 40)
    presentation_weight = weights.get("presentation", 20)
    competencies_weight = weights.get("competencies", 40)

    impact = score_impact(parsed, CONFIG)
    presentation = score_presentation(parsed, CONFIG)
    competencies = score_competencies(parsed, CONFIG)

    # Weighted average
    total_weight = impact_weight + presentation_weight + competencies_weight
    if total_weight == 0:
        total_weight = 100

    overall = (
        impact.score * impact_weight
        + presentation.score * presentation_weight
        + competencies.score * competencies_weight
    ) / total_weight

    overall = max(0.0, min(100.0, round(overall, 1)))
    zone = classify_zone(overall)

    logger.info(
        "Resume score: %.1f (%s) — impact=%.1f, presentation=%.1f, competencies=%.1f",
        overall, zone.value, impact.score, presentation.score, competencies.score,
    )

    return ResumeScore(
        overall=overall,
        zone=zone,
        impact=impact,
        presentation=presentation,
        competencies=competencies,
        weights={
            "impact": impact_weight,
            "presentation": presentation_weight,
            "competencies": competencies_weight,
        },
    )
