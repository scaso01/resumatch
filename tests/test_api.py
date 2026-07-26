"""Tests for resumatch.api module."""

import io
import pytest
from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient

from resumatch.api import app
from resumatch.models import (
    ContactInfo,
    FeedbackCategory,
    FeedbackItem,
    FeedbackPriority,
    ImpactScore,
    JDMatchResult,
    ParsedResume,
    ResumeScore,
    ResumeSection,
    ScoreZone,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_parsed_resume():
    return ParsedResume(
        raw_text="John Smith\njohn@email.com\nEXPERIENCE\n- Led team",
        sections=[ResumeSection(name="experience", heading="EXPERIENCE", content="Led team", bullets=["Led team"])],
        contact=ContactInfo(name="John Smith", email="john@email.com"),
        page_count=1,
        word_count=10,
        file_name="test.pdf",
    )


@pytest.fixture
def mock_score():
    return ResumeScore(overall=75.0, zone=ScoreZone.YELLOW, impact=ImpactScore(score=70.0))


@pytest.fixture
def mock_feedback():
    return [FeedbackItem(category=FeedbackCategory.IMPACT, priority=FeedbackPriority.MEDIUM, message="Add metrics")]


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"


class TestConfigEndpoint:
    def test_config_returns_scoring(self, client):
        resp = client.get("/api/v1/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "scoring" in data
        assert "matching" in data


class TestAnalyzeEndpoint:
    def test_analyze_pdf(self, client, mock_parsed_resume, mock_score, mock_feedback):
        with patch("resumatch.ingest.parse_file", return_value=mock_parsed_resume), \
             patch("resumatch.scoring.score_resume", return_value=mock_score), \
             patch("resumatch.feedback.generate_feedback", return_value=mock_feedback):
            resp = client.post(
                "/api/v1/analyze",
                files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert "feedback" in data

    def test_analyze_rejects_unsupported_type(self, client):
        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("resume.txt", io.BytesIO(b"text"), "text/plain")},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_analyze_rejects_large_file(self, client):
        big_content = b"x" * (11 * 1024 * 1024)
        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("resume.pdf", io.BytesIO(big_content), "application/pdf")},
        )
        assert resp.status_code == 413

    def test_analyze_handles_parse_error(self, client):
        with patch("resumatch.ingest.parse_file", side_effect=ValueError("Bad PDF")):
            resp = client.post(
                "/api/v1/analyze",
                files={"file": ("resume.pdf", io.BytesIO(b"bad"), "application/pdf")},
            )
        assert resp.status_code == 500


class TestParseEndpoint:
    def test_parse_returns_resume(self, client, mock_parsed_resume):
        with patch("resumatch.ingest.parse_file", return_value=mock_parsed_resume):
            resp = client.post(
                "/api/v1/parse",
                files={"file": ("resume.pdf", io.BytesIO(b"%PDF"), "application/pdf")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "raw_text" in data
        assert "sections" in data


class TestScoreEndpoint:
    def test_score_json_resume(self, client, mock_score):
        with patch("resumatch.scoring.score_resume", return_value=mock_score):
            resp = client.post(
                "/api/v1/score",
                json={
                    "raw_text": "John Smith\nEXPERIENCE\n- Led team",
                    "sections": [],
                    "contact": {},
                    "page_count": 1,
                    "word_count": 5,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "overall" in data


class TestAnalyzeWithJDEndpoint:
    def test_analyze_with_jd(self, client, mock_parsed_resume, mock_score, mock_feedback):
        mock_match = JDMatchResult(overall_match=0.75, keyword_match=0.8, semantic_match=0.7)
        with patch("resumatch.ingest.parse_file", return_value=mock_parsed_resume), \
             patch("resumatch.scoring.score_resume", return_value=mock_score), \
             patch("resumatch.feedback.generate_feedback", return_value=mock_feedback), \
             patch("resumatch.matching.match_jd", return_value=mock_match):
            resp = client.post(
                "/api/v1/analyze-with-jd",
                files={"file": ("resume.pdf", io.BytesIO(b"%PDF"), "application/pdf")},
                data={"jd_text": "Looking for a Python developer"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["jd_match"]["overall_match"] == 0.75


class TestImproveEndpoint:
    def test_improve_with_weak_bullets(self, client):
        """Improve endpoint identifies weak bullets and returns rewrites."""
        mock_rewritten = ["Spearheaded system maintenance achieving 99.9% uptime"]

        with patch("resumatch.llm.LLMClient.health_check", new_callable=AsyncMock, return_value=True), \
             patch("resumatch.llm.LLMClient.batch_rewrite", new_callable=AsyncMock, return_value=mock_rewritten):
            resp = client.post(
                "/api/v1/improve",
                json={
                    "raw_text": "EXPERIENCE\nHelped maintain the systems",
                    "sections": [{
                        "name": "experience",
                        "heading": "EXPERIENCE",
                        "content": "Helped maintain the systems",
                        "bullets": ["Helped maintain the systems"],
                    }],
                    "contact": {},
                    "page_count": 1,
                    "word_count": 5,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "improvements" in data
        assert data["bullets_analyzed"] == 1

    def test_improve_no_experience(self, client):
        """Improve endpoint returns empty when no experience section."""
        with patch("resumatch.llm.LLMClient.health_check", new_callable=AsyncMock, return_value=True):
            resp = client.post(
                "/api/v1/improve",
                json={
                    "raw_text": "SKILLS\nPython",
                    "sections": [{
                        "name": "skills",
                        "heading": "SKILLS",
                        "content": "Python",
                        "bullets": [],
                    }],
                    "contact": {},
                    "page_count": 1,
                    "word_count": 2,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["bullets_analyzed"] == 0
        assert data["improvements"] == []

    def test_improve_llm_unavailable(self, client):
        """Improve endpoint gracefully handles LLM being unavailable."""
        with patch("resumatch.llm.LLMClient.health_check", new_callable=AsyncMock, return_value=False):
            resp = client.post(
                "/api/v1/improve",
                json={
                    "raw_text": "EXPERIENCE\nHelped maintain the systems",
                    "sections": [{
                        "name": "experience",
                        "heading": "EXPERIENCE",
                        "content": "Helped maintain the systems",
                        "bullets": ["Helped maintain the systems"],
                    }],
                    "contact": {},
                    "page_count": 1,
                    "word_count": 5,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["llm_available"] is False
        assert data["bullets_improved"] == 0


class TestUploadValidation:
    def test_no_filename(self, client):
        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("", io.BytesIO(b""), "application/octet-stream")},
        )
        assert resp.status_code in (400, 422)  # FastAPI may return 422 for validation

    def test_docx_accepted(self, client, mock_parsed_resume, mock_score, mock_feedback):
        with patch("resumatch.ingest.parse_file", return_value=mock_parsed_resume), \
             patch("resumatch.scoring.score_resume", return_value=mock_score), \
             patch("resumatch.feedback.generate_feedback", return_value=mock_feedback):
            resp = client.post(
                "/api/v1/analyze",
                files={"file": ("resume.docx", io.BytesIO(b"PK\x03\x04"), "application/vnd.openxmlformats")},
            )
        assert resp.status_code == 200
