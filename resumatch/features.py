"""
ResuMatch - Feature Extraction
================================
Extract quality signals from parsed resume content:
action verbs, quantification, specificity, grammar, formatting.
"""

from __future__ import annotations

import re

from resumatch.config import logger
from resumatch.models import ResumeSection

# ---------------------------------------------------------------------------
# Action Verb Sets
# ---------------------------------------------------------------------------

STRONG_VERBS: set[str] = {
    "accelerated", "achieved", "acquired", "adapted", "administered", "advanced",
    "advocated", "allocated", "analyzed", "architected", "assembled", "assessed",
    "automated", "balanced", "boosted", "built", "captured", "centralized",
    "championed", "coached", "collaborated", "communicated", "compiled",
    "completed", "conceived", "consolidated", "constructed", "converted",
    "coordinated", "crafted", "created", "cultivated", "customized",
    "decreased", "defined", "delegated", "delivered", "deployed", "designed",
    "developed", "devised", "directed", "discovered", "doubled", "drove",
    "earned", "eliminated", "enabled", "engineered", "enhanced", "established",
    "evaluated", "exceeded", "executed", "expanded", "expedited", "facilitated",
    "forecasted", "formulated", "founded", "generated", "grew", "guided",
    "headed", "identified", "implemented", "improved", "increased",
    "influenced", "initiated", "innovated", "inspected", "instituted",
    "integrated", "introduced", "invented", "launched", "led", "leveraged",
    "managed", "mapped", "maximized", "mentored", "merged", "migrated",
    "minimized", "modernized", "monitored", "negotiated", "operated",
    "optimized", "orchestrated", "organized", "outperformed", "overhauled",
    "oversaw", "piloted", "pioneered", "planned", "presented", "prioritized",
    "produced", "programmed", "projected", "promoted", "proposed",
    "published", "raised", "rebuilt", "recommended", "reconciled", "recruited",
    "redesigned", "reduced", "reengineered", "reformed", "regulated",
    "rehabilitated", "reinforced", "reorganized", "resolved", "restructured",
    "revamped", "revitalized", "revolutionized", "scaled", "secured",
    "simplified", "sold", "solved", "spearheaded", "standardized",
    "steered", "strategized", "streamlined", "strengthened", "supervised",
    "surpassed", "sustained", "systematized", "trained", "transformed",
    "tripled", "troubleshot", "unified", "upgraded", "validated",
}

WEAK_VERBS: set[str] = {
    "assisted", "attempted", "contributed", "did", "experienced", "got",
    "had", "helped", "handled", "involved", "knew", "made", "managed to",
    "oversaw some", "participated", "responsible", "ran", "saw", "served",
    "supported", "tried", "used", "utilized", "was", "went", "worked",
}


def analyze_action_verbs(bullets: list[str]) -> tuple[float, list[str]]:
    """Analyze first word of each bullet for strong/weak action verbs.

    Returns:
        (strong_verb_rate, list_of_weak_verbs_found)
    """
    if not bullets:
        return 0.0, []

    strong_count = 0
    weak_verbs_found: list[str] = []

    for bullet in bullets:
        stripped = bullet.strip().lstrip("-*> ")
        if not stripped:
            continue
        first_word = stripped.split()[0].lower().rstrip(".,;:")
        if first_word in STRONG_VERBS:
            strong_count += 1
        elif first_word in WEAK_VERBS:
            weak_verbs_found.append(first_word)

    total = len([b for b in bullets if b.strip()])
    if total == 0:
        return 0.0, []

    rate = strong_count / total
    logger.debug("Action verb analysis: %d/%d strong, weak=%s", strong_count, total, weak_verbs_found)
    return rate, weak_verbs_found


_QUANT_RE = re.compile(
    r"""
    \b\d[\d,]*\.?\d*\s*[%$KMBkmb]   # 75%, 200K, 10M, 3B, $50
    | \$[\d,]+                        # $200,000
    | \b\d+[xX]\b                     # 3x, 10X
    | \b\d[\d,]*\.?\d*\b              # plain numbers like 500000
    """,
    re.VERBOSE,
)


def analyze_quantification(bullets: list[str]) -> tuple[float, list[str]]:
    """Detect metrics/numbers in bullets.

    Returns:
        (quantification_rate, bullets_without_metrics)
    """
    if not bullets:
        return 0.0, []

    with_metrics = 0
    without_metrics: list[str] = []

    for bullet in bullets:
        stripped = bullet.strip()
        if not stripped:
            continue
        if _QUANT_RE.search(stripped):
            with_metrics += 1
        else:
            without_metrics.append(stripped)

    total = len([b for b in bullets if b.strip()])
    if total == 0:
        return 0.0, []

    rate = with_metrics / total
    logger.debug("Quantification analysis: %d/%d with metrics", with_metrics, total)
    return rate, without_metrics


_VAGUE_PHRASES = [
    "various tasks",
    "responsible for",
    "helped with",
    "worked on",
    "assisted in",
    "participated in",
    "was involved",
    "involved in",
    "duties included",
    "tasked with",
    "in charge of",
    "various projects",
    "many things",
    "different tasks",
    "some projects",
    "day-to-day",
    "doing various",
]


def analyze_specificity(bullets: list[str]) -> float:
    """Score 0-1 measuring how specific bullet points are.

    Penalizes vague phrases. Rewards numbers, technology names, and metrics.
    """
    if not bullets:
        return 0.0

    scores: list[float] = []

    for bullet in bullets:
        stripped = bullet.strip()
        if not stripped:
            continue

        score = 0.65  # baseline (raised from 0.5)

        # Penalize vague phrases
        lower = stripped.lower()
        vague_count = sum(1 for phrase in _VAGUE_PHRASES if phrase in lower)
        score -= vague_count * 0.15

        # Reward quantification
        if _QUANT_RE.search(stripped):
            score += 0.3

        # Reward longer, more detailed bullets (>10 words is good)
        word_count = len(stripped.split())
        if word_count >= 10:
            score += 0.1
        elif word_count < 5:
            score -= 0.1

        # Reward specific technology mentions (simple heuristic: capitalized words
        # that aren't sentence starters and aren't common English)
        words = stripped.split()
        if len(words) > 1:
            tech_signals = sum(
                1 for w in words[1:]
                if (w[0].isupper() or any(c in w for c in "/.+#")) and len(w) > 1
            )
            score += min(tech_signals * 0.05, 0.2)

        # Reward parenthetical details (e.g., "(Python, React)")
        if "(" in stripped and ")" in stripped:
            score += 0.1

        # Reward action result patterns (e.g., "resulting in", "leading to")
        result_patterns = ["resulting in", "leading to", "which led to", "enabling", "achieving"]
        if any(rp in lower for rp in result_patterns):
            score += 0.1

        scores.append(max(0.0, min(1.0, score)))

    if not scores:
        return 0.0

    result = sum(scores) / len(scores)
    logger.debug("Specificity analysis: %.2f (from %d bullets)", result, len(scores))
    return result


def check_grammar(text: str) -> list[str]:
    """Basic grammar checks (stub -- real implementation needs language-tool-python).

    Current checks:
    - Double spaces
    - Missing period at end of bullet
    - Inconsistent bullet endings (mix of period/no-period)
    """
    if not text or not text.strip():
        return []

    errors: list[str] = []

    # Check for double spaces
    if "  " in text:
        count = text.count("  ")
        errors.append(f"Found {count} instance(s) of double spaces")

    # Split into lines and analyze only explicitly-marked bullet lines.
    # Role summary paragraphs (unmarked lines) are excluded — they use different
    # punctuation conventions (full sentences with periods) than bullets.
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    bullet_lines = [line for line in lines if line.startswith(("-", "*", ">", "\u2022"))]

    if len(bullet_lines) >= 2:
        ends_with_period = [line.endswith(".") for line in bullet_lines]
        period_count = sum(ends_with_period)

        if 0 < period_count < len(bullet_lines):
            errors.append(
                f"Inconsistent bullet endings: {period_count}/{len(bullet_lines)} "
                "end with a period (should be all or none)"
            )

    logger.debug("Grammar check: %d error(s) found", len(errors))
    return errors


def check_formatting(sections: list[ResumeSection], page_count: int) -> dict:
    """Evaluate formatting consistency and page count.

    Returns dict with:
        date_consistency: bool -- all date formats are consistent
        bullet_consistency: bool -- bullets use consistent style
        heading_consistency: bool -- headings use consistent casing
        page_score: float 0-1 -- based on ideal 1-2 pages
    """
    result: dict = {
        "date_consistency": True,
        "bullet_consistency": True,
        "heading_consistency": True,
        "page_score": 1.0,
    }

    if not sections:
        result["page_score"] = 0.0
        return result

    # -- Page score (ideal 1-2 pages) --
    if page_count <= 0:
        result["page_score"] = 0.0
    elif page_count <= 2:
        result["page_score"] = 1.0
    elif page_count == 3:
        result["page_score"] = 0.6
    elif page_count == 4:
        result["page_score"] = 0.3
    else:
        result["page_score"] = 0.1

    # -- Heading consistency (all upper, all title, or mixed?) --
    headings = [s.heading for s in sections if s.heading]
    if len(headings) >= 2:
        all_upper = all(h.isupper() for h in headings)
        all_title = all(h.istitle() for h in headings)
        all_lower = all(h.islower() for h in headings)
        if not (all_upper or all_title or all_lower):
            result["heading_consistency"] = False

    # -- Bullet consistency (check if bullet prefixes are consistent) --
    all_bullets: list[str] = []
    for section in sections:
        all_bullets.extend(section.bullets)

    if len(all_bullets) >= 2:
        prefixes: set[str] = set()
        for bullet in all_bullets:
            stripped = bullet.strip()
            if stripped.startswith("- "):
                prefixes.add("dash")
            elif stripped.startswith("* "):
                prefixes.add("asterisk")
            elif stripped.startswith("> "):
                prefixes.add("arrow")
            else:
                prefixes.add("none")
        if len(prefixes) > 1:
            result["bullet_consistency"] = False

    # -- Date consistency (check for date patterns in content) --
    date_patterns = {
        "month_year": re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b"),
        "mm_yyyy": re.compile(r"\b\d{1,2}/\d{4}\b"),
        "yyyy": re.compile(r"\b20\d{2}\b"),
    }
    found_formats: set[str] = set()
    for section in sections:
        text = section.content
        for fmt_name, pattern in date_patterns.items():
            if pattern.search(text):
                found_formats.add(fmt_name)

    if len(found_formats) > 1:
        # Multiple date formats detected (e.g., "Jan 2020" and "01/2020")
        # Having both "month_year" and "yyyy" is okay (e.g., "Jan 2020 - 2021")
        # but "mm_yyyy" mixed with "month_year" is inconsistent
        if "mm_yyyy" in found_formats and "month_year" in found_formats:
            result["date_consistency"] = False

    logger.debug("Formatting check: %s", result)
    return result
