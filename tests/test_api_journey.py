"""The journey a real user takes: upload a resume, get a score back.

tests/test_api.py patches parse_file, score_resume and generate_feedback, so it
proves only that the route wires JSON together. Nothing here is mocked -- these
run the full chain (_read_upload -> parse_file -> detect_sections ->
score_resume -> generate_feedback) over a real PDF and a real DOCX.

conftest.py's _block_real_network allowlists http://testserver, so TestClient
works while genuine outbound calls stay blocked.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from resumatch.api import app
from resumatch.config import CONFIG


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def resume_pdf(fixtures_dir) -> bytes:
    return (fixtures_dir / "sample_resume.pdf").read_bytes()


@pytest.fixture
def resume_docx(fixtures_dir) -> bytes:
    return (fixtures_dir / "sample_resume.docx").read_bytes()


def _zone_for(overall: float) -> str:
    zones = CONFIG["scoring"]["zones"]
    if overall >= zones["green"]:
        return "green"
    if overall >= zones["yellow"]:
        return "yellow"
    return "red"


class TestAnalyzeJourney:
    def test_pdf_upload_returns_a_real_score(self, client, resume_pdf):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("sample_resume.pdf", resume_pdf, "application/pdf")},
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["resume"]["file_type"] == "pdf"
        assert "JORDAN AVERY" in body["resume"]["raw_text"]

        score = body["score"]
        assert 0.0 <= score["overall"] <= 100.0
        assert score["zone"] == _zone_for(score["overall"])
        # Weights come from config.yaml, not the model default.
        assert score["weights"] == CONFIG["scoring"]["weights"]

    def test_score_is_a_weighted_sum_of_its_modules(self, client, resume_pdf):
        """The headline number must actually follow from the module scores."""
        body = client.post(
            "/api/v1/analyze",
            files={"file": ("sample_resume.pdf", resume_pdf, "application/pdf")},
        ).json()
        score = body["score"]
        weights = CONFIG["scoring"]["weights"]

        expected = (
            score["impact"]["score"] * weights["impact"]
            + score["presentation"]["score"] * weights["presentation"]
            + score["competencies"]["score"] * weights["competencies"]
        ) / 100.0
        assert score["overall"] == pytest.approx(expected, abs=0.5)

    def test_a_strong_resume_scores_its_strengths(self, client, resume_pdf):
        """The fixture is deliberately verb-heavy and quantified."""
        score = client.post(
            "/api/v1/analyze",
            files={"file": ("sample_resume.pdf", resume_pdf, "application/pdf")},
        ).json()["score"]

        assert score["impact"]["action_verb_rate"] > 0.5
        assert score["impact"]["quantification_rate"] > 0.5
        assert score["competencies"]["skills_count"] >= 8
        assert not score["presentation"]["missing_sections"]

    def test_feedback_is_returned_and_well_formed(self, client, resume_pdf):
        body = client.post(
            "/api/v1/analyze",
            files={"file": ("sample_resume.pdf", resume_pdf, "application/pdf")},
        ).json()
        assert isinstance(body["feedback"], list)
        for item in body["feedback"]:
            assert item["category"] in {
                "impact",
                "presentation",
                "competencies",
                "general",
            }
            assert item["priority"] in {"high", "medium", "low"}
            assert item["message"].strip()

    def test_docx_upload_scores_close_to_the_pdf(self, client, resume_pdf, resume_docx):
        """Same resume, two formats, two parsers -- scores should not diverge."""
        pdf_score = client.post(
            "/api/v1/analyze",
            files={"file": ("r.pdf", resume_pdf, "application/pdf")},
        ).json()["score"]["overall"]
        docx_score = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "r.docx",
                    resume_docx,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        ).json()["score"]["overall"]

        assert docx_score == pytest.approx(pdf_score, abs=10.0)

    def test_rejects_unsupported_file_type(self, client):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert response.status_code >= 400


class TestJdMatchJourney:
    """JD matching over the keyword path.

    NOTE: conftest.py blocks real network, so KeyBERT and sentence-transformers
    cannot fetch all-MiniLM-L6-v2 and both fall back. The keyword half of
    matching is exercised for real here; the semantic half is not covered by
    any test and needs the models present to be verified.
    """

    @pytest.mark.slow
    def test_matching_against_a_job_description(
        self, client, resume_pdf, fixtures_dir
    ):
        jd_text = (fixtures_dir / "sample_jd.txt").read_text(encoding="utf-8")
        response = client.post(
            "/api/v1/analyze-with-jd",
            files={"file": ("sample_resume.pdf", resume_pdf, "application/pdf")},
            data={"jd_text": jd_text},
        )
        assert response.status_code == 200, response.text

        match = response.json()["jd_match"]
        assert match is not None
        # matching.py returns percentages, not 0-1 ratios.
        assert 0.0 <= match["overall_match"] <= 100.0
        assert 0.0 <= match["keyword_match"] <= 100.0
        assert match["jd_keywords"], "no keywords extracted from the JD"

        # The fixture resume and JD deliberately share Python, Kafka,
        # Kubernetes, PostgreSQL and AWS, so real overlap must be found.
        assert match["keyword_match"] > 20.0
        assert match["matched_keywords"], "no keyword overlap detected"

    @pytest.mark.slow
    def test_an_unrelated_jd_matches_worse(self, client, resume_pdf, fixtures_dir):
        """A matcher that scores everything the same would be useless."""
        jd_text = (fixtures_dir / "sample_jd.txt").read_text(encoding="utf-8")
        unrelated = (
            "Pastry Chef. Prepare laminated doughs, manage a bakery line, "
            "decorate wedding cakes, order flour and butter, and train "
            "junior bakers on croissant lamination and tempering chocolate."
        )
        scores = {}
        for label, text in (("relevant", jd_text), ("unrelated", unrelated)):
            body = client.post(
                "/api/v1/analyze-with-jd",
                files={"file": ("r.pdf", resume_pdf, "application/pdf")},
                data={"jd_text": text},
            ).json()
            scores[label] = body["jd_match"]["keyword_match"]

        assert scores["relevant"] > scores["unrelated"], scores


class TestHealthJourney:
    def test_health_reports_component_state(self, client):
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ok"
        assert isinstance(body["spacy_loaded"], bool)
        assert isinstance(body["llm_available"], bool)

    def test_config_endpoint_echoes_live_weights(self, client):
        body = client.get("/api/v1/config").json()
        assert body["scoring"]["weights"] == CONFIG["scoring"]["weights"]
