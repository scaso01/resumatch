"""Dashboard tests that actually render the Streamlit app.

The old tests/test_dashboard.py never touched Streamlit: no AppTest, no widget
interaction, no render assertion -- and all 185 of its tests were disabled by a
hardcoded `skipif(True, ...)`. These run the real app.

Scope note: `st.file_uploader` is not one of the widgets AppTest can drive, so
the upload click itself cannot be simulated. Everything downstream of it is
covered here using genuine output from the real API, which is where all of the
rendering code -- score card, radar chart, feedback list, JD match, parsed
resume tabs -- lives.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from resumatch.api import app as api_app

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit is required for dashboard tests"
).AppTest

DASHBOARD = "resumatch/dashboard/app.py"


def _analyze(fixtures_dir, jd_text: str | None = None) -> dict:
    """Run the real analysis pipeline and return the API's response."""
    client = TestClient(api_app)
    files = {
        "file": (
            "sample_resume.pdf",
            (fixtures_dir / "sample_resume.pdf").read_bytes(),
            "application/pdf",
        )
    }
    if jd_text is None:
        response = client.post("/api/v1/analyze", files=files)
    else:
        response = client.post(
            "/api/v1/analyze-with-jd", files=files, data={"jd_text": jd_text}
        )
    response.raise_for_status()
    return response.json()


@pytest.fixture
def analysis(fixtures_dir) -> dict:
    return _analyze(fixtures_dir)


def _dashboard_showing(result: dict):
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.session_state["result"] = result
    at.session_state["filename"] = "sample_resume.pdf"
    at.run()
    return at


def test_dashboard_loads_empty(fixtures_dir):
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Upload a resume" in i.value for i in at.info)


def test_renders_a_real_analysis(analysis):
    at = _dashboard_showing(analysis)

    assert not at.exception, at.exception
    assert any("sample_resume.pdf" in s.value for s in at.subheader)


def test_shows_the_score_and_its_zone(analysis):
    at = _dashboard_showing(analysis)
    rendered = " ".join(str(m.value) for m in at.markdown)

    overall = analysis["score"]["overall"]
    assert str(round(overall, 1)) in rendered or str(int(overall)) in rendered
    assert analysis["score"]["zone"].upper() in rendered.upper()


def test_feedback_items_are_rendered(analysis):
    """Feedback is grouped into priority sections, one expander per item."""
    feedback = analysis["feedback"]
    if not feedback:
        pytest.skip("fixture resume generated no feedback items")

    at = _dashboard_showing(analysis)

    headings = " ".join(s.value for s in at.subheader)
    priorities = {item["priority"] for item in feedback}
    for priority in priorities:
        assert priority.capitalize() in headings, (
            f"no {priority} priority section rendered: {headings!r}"
        )

    labels = " ".join(str(e.label) for e in at.expander)
    for item in feedback:
        assert item["message"][:40] in labels, (
            f"feedback message never reached the page: {item['message']!r}"
        )


def test_parsed_resume_tab_shows_contact_details(analysis):
    at = _dashboard_showing(analysis)
    assert not at.exception
    contact_blobs = " ".join(str(j.value) for j in at.json)
    assert "jordan.avery@example.com" in contact_blobs


@pytest.mark.slow
def test_jd_match_percentages_are_not_scaled_twice(fixtures_dir):
    """The API returns percentages already; the view must not multiply again.

    It used to, which rendered a 32.6% match as "3260%".
    """
    jd_text = (fixtures_dir / "sample_jd.txt").read_text(encoding="utf-8")
    result = _analyze(fixtures_dir, jd_text=jd_text)
    at = _dashboard_showing(result)

    assert not at.exception, at.exception
    shown = {m.label: m.value for m in at.metric}
    for label in ("Overall Match", "Keyword Match", "Semantic Match"):
        assert label in shown, f"{label} never rendered; saw {sorted(shown)}"
        percent = float(str(shown[label]).rstrip("%"))
        assert 0 <= percent <= 100, f"{label} rendered as {shown[label]}"


@pytest.mark.slow
def test_matched_keywords_render_as_words_not_dicts(fixtures_dir):
    """Each match is a record; printing the record dumped its dict repr."""
    jd_text = (fixtures_dir / "sample_jd.txt").read_text(encoding="utf-8")
    result = _analyze(fixtures_dir, jd_text=jd_text)
    matched = result["jd_match"]["matched_keywords"]
    if not matched:
        pytest.skip("this JD matched no keywords in the fixture resume")

    at = _dashboard_showing(result)
    rendered = " ".join(str(m.value) for m in at.markdown)

    assert "found_in_resume" not in rendered, "raw record leaked onto the page"
    assert f"`{matched[0]['keyword']}`" in rendered


@pytest.mark.slow
def test_jd_match_tab_appears_only_with_a_jd(fixtures_dir):
    jd_text = (fixtures_dir / "sample_jd.txt").read_text(encoding="utf-8")

    without = _dashboard_showing(_analyze(fixtures_dir))
    with_jd = _dashboard_showing(_analyze(fixtures_dir, jd_text=jd_text))

    assert not with_jd.exception
    # st.tabs labels surface as the tab container's contents; compare counts.
    assert len(with_jd.tabs) > len(without.tabs), (
        f"expected an extra tab with a JD: {len(with_jd.tabs)} vs {len(without.tabs)}"
    )
