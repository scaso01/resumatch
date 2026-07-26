"""
ResuMatch Dashboard - UI Components
=====================================
Score cards, radar chart, feedback list, and JD match display.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


# Zone colors
ZONE_COLORS = {
    "green": "#22c55e",
    "yellow": "#eab308",
    "red": "#ef4444",
}

PRIORITY_ICONS = {
    "high": "🔴",
    "medium": "🟡",
    "low": "🟢",
}

CATEGORY_LABELS = {
    "impact": "Impact",
    "presentation": "Presentation",
    "competencies": "Competencies",
    "general": "General",
}


def render_score_card(score: dict) -> None:
    """Render the overall score card with zone indicator."""
    overall = score.get("overall", 0)
    zone = score.get("zone", "red").lower()
    color = ZONE_COLORS.get(zone, "#6b7280")

    st.markdown(
        f"""
        <div class="score-card" style="border-left: 6px solid {color};">
            <div class="score-value" style="color: {color};">{overall:.0f}</div>
            <div class="score-label">out of 100</div>
            <div class="zone-badge" style="background: {color};">{zone.upper()} ZONE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sub-scores
    impact = score.get("impact", {})
    presentation = score.get("presentation", {})
    competencies = score.get("competencies", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Impact (40%)", f"{impact.get('score', 0):.0f}")
    with col2:
        st.metric("Presentation (30%)", f"{presentation.get('score', 0):.0f}")
    with col3:
        st.metric("Competencies (30%)", f"{competencies.get('score', 0):.0f}")


def render_radar_chart(score: dict) -> None:
    """Render a radar chart of sub-scores."""
    impact = score.get("impact", {})
    presentation = score.get("presentation", {})
    competencies = score.get("competencies", {})

    categories = ["Impact", "Presentation", "Competencies"]
    values = [
        impact.get("score", 0),
        presentation.get("score", 0),
        competencies.get("score", 0),
    ]
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    zone = score.get("zone", "red").lower()
    color = ZONE_COLORS.get(zone, "#6b7280")

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor=f"rgba({_hex_to_rgb(color)}, 0.2)",
        line=dict(color=color, width=2),
        name="Score",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickvals=[25, 50, 75, 100]),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_feedback_list(feedback: list[dict]) -> None:
    """Render prioritized feedback items."""
    if not feedback:
        st.success("No feedback items — your resume looks great!")
        return

    # Group by priority
    high = [f for f in feedback if f.get("priority") == "high"]
    medium = [f for f in feedback if f.get("priority") == "medium"]
    low = [f for f in feedback if f.get("priority") == "low"]

    for label, items, icon in [
        ("High Priority", high, "🔴"),
        ("Medium Priority", medium, "🟡"),
        ("Low Priority", low, "🟢"),
    ]:
        if not items:
            continue
        st.subheader(f"{icon} {label}")
        for item in items:
            category = CATEGORY_LABELS.get(item.get("category", ""), item.get("category", ""))
            with st.expander(f"**[{category}]** {item.get('message', '')}"):
                if item.get("suggestion"):
                    st.markdown(f"**Suggestion:** {item['suggestion']}")
                if item.get("source") == "llm":
                    st.caption("Enhanced by AI")


def render_jd_match(jd_match: dict) -> None:
    """Render JD matching results."""
    # These arrive already expressed as percentages -- matching.py returns
    # round(value * 100, 1). Scaling them again here rendered a 32.6% match
    # as "3260%".
    overall = jd_match.get("overall_match", 0.0)
    keyword = jd_match.get("keyword_match", 0.0)
    semantic = jd_match.get("semantic_match", 0.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Match", f"{overall:.0f}%")
    with col2:
        st.metric("Keyword Match", f"{keyword:.0f}%")
    with col3:
        st.metric("Semantic Match", f"{semantic:.0f}%")

    # Progress bars
    st.progress(min(overall / 100, 1.0), text=f"Overall: {overall:.0f}%")

    # Matched keywords
    matched = jd_match.get("matched_keywords", [])
    missing = jd_match.get("missing_keywords", [])

    if matched:
        st.subheader("Matched Keywords")
        # Each entry is a KeywordMatch record, not a bare string; formatting
        # the whole record printed its dict repr on the page.
        st.markdown(" ".join(f"`{kw.get('keyword', '')}`" for kw in matched))

    if missing:
        st.subheader("Missing Keywords")
        st.markdown(" ".join(f"~~{kw}~~" for kw in missing))


def render_improvements(improve_result: dict) -> None:
    """Render before/after bullet improvements."""
    if not improve_result.get("llm_available"):
        st.warning("LLM server unavailable — cannot generate improvements.")
        return

    improvements = improve_result.get("improvements", [])
    analyzed = improve_result.get("bullets_analyzed", 0)
    improved = improve_result.get("bullets_improved", 0)

    st.metric("Bullets Analyzed", analyzed)
    st.metric("Bullets Improved", improved)

    if not improvements:
        st.success("All bullets look strong — no improvements needed!")
        return

    for i, imp in enumerate(improvements):
        with st.expander(f"Bullet {i + 1} — {imp.get('section', '')} ({imp.get('reason', '')})"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original:**")
                st.markdown(
                    f'<div style="padding:8px;background:#fef2f2;border-left:3px solid #ef4444;border-radius:4px;">{imp["original"]}</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown("**Improved:**")
                st.markdown(
                    f'<div style="padding:8px;background:#f0fdf4;border-left:3px solid #22c55e;border-radius:4px;">{imp["rewritten"]}</div>',
                    unsafe_allow_html=True,
                )
            if st.button("Copy improved bullet", key=f"copy_{i}"):
                st.code(imp["rewritten"], language=None)


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to comma-separated RGB string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r}, {g}, {b}"
