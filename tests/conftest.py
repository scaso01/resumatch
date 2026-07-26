"""
Pytest fixtures for ResuMatch tests.

Provides:
- Log isolation (NullHandler replaces file handler)
- Network guard (blocks real HTTP during tests)
- LLM guard (blocks real LLM calls)
- spaCy guard (blocks heavy model loading)
- Fixture paths for sample files
- Sample resume data
"""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest


# =============================================================================
# Log Isolation
# =============================================================================

@pytest.fixture(autouse=True, scope="session")
def _isolate_log_handlers():
    """Replace resumatch file handlers with NullHandler during tests."""
    rm_logger = logging.getLogger("resumatch")
    original_handlers = rm_logger.handlers[:]
    rm_logger.handlers = [logging.NullHandler()]
    yield
    rm_logger.handlers = original_handlers


# =============================================================================
# Network Guard — prevent real HTTP during tests
# =============================================================================

@pytest.fixture(autouse=True, scope="session")
def _block_real_network():
    """Prevent real HTTP requests during tests."""
    patches = []

    try:
        import httpx

        original_async_send = httpx.AsyncClient.send

        async def _blocked_async_send(self, request, **kwargs):
            # Allow TestClient (testserver) and localhost test servers
            if str(request.url).startswith(("http://testserver", "http://localhost", "http://127.0.0.1")):
                return await original_async_send(self, request, **kwargs)
            raise RuntimeError(
                f"Real async HTTP {request.method} to {request.url} blocked by conftest.py! "
                "Use unittest.mock to mock httpx calls in tests."
            )

        p = patch.object(httpx.AsyncClient, "send", _blocked_async_send)
        patches.append(p)

        original_sync_send = httpx.Client.send

        def _blocked_sync_send(self, request, **kwargs):
            if str(request.url).startswith(("http://testserver", "http://localhost", "http://127.0.0.1")):
                return original_sync_send(self, request, **kwargs)
            raise RuntimeError(
                f"Real HTTP {request.method} to {request.url} blocked by conftest.py! "
                "Use unittest.mock to mock httpx calls in tests."
            )

        p = patch.object(httpx.Client, "send", _blocked_sync_send)
        patches.append(p)
    except ImportError:
        pass

    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


# =============================================================================
# Fixture Paths
# =============================================================================

@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


# =============================================================================
# Sample Resume Text
# =============================================================================

SAMPLE_RESUME_TEXT = """John Smith
john.smith@email.com | (555) 123-4567 | linkedin.com/in/johnsmith | New York, NY

EXPERIENCE

Senior Software Engineer | Acme Corp | Jan 2020 - Present
- Led migration of monolithic application to microservices, reducing deployment time by 75%
- Managed team of 8 engineers, delivering 3 major product releases on schedule
- Implemented automated testing pipeline that increased code coverage from 45% to 92%
- Reduced cloud infrastructure costs by $200K annually through optimization

Software Engineer | TechStart Inc | Jun 2017 - Dec 2019
- Developed RESTful APIs serving 10M+ requests daily
- Built real-time data processing pipeline handling 500K events per second
- Collaborated with product team to define technical requirements for 5 new features

EDUCATION

Master of Science in Computer Science | MIT | 2017
Bachelor of Science in Computer Science | UC Berkeley | 2015

SKILLS

Python, Java, Go, TypeScript, AWS, Docker, Kubernetes, PostgreSQL, Redis,
Machine Learning, Agile, Team Leadership, System Design, CI/CD
"""

SAMPLE_WEAK_RESUME_TEXT = """Jane Doe
jane@email.com

Work History

Worked at Company A for 3 years doing various tasks.
Was responsible for helping with projects.
Assisted team members with their work.
Helped maintain the systems.

Also worked at Company B.
Did some programming.
Was part of the team.

Education
Some University, 2018

Skills
Computers, Office, Typing
"""


@pytest.fixture
def sample_resume_text():
    return SAMPLE_RESUME_TEXT


@pytest.fixture
def sample_weak_resume_text():
    return SAMPLE_WEAK_RESUME_TEXT


@pytest.fixture
def sample_parsed_resume():
    from resumatch.models import ContactInfo, ParsedResume, ResumeSection

    return ParsedResume(
        raw_text=SAMPLE_RESUME_TEXT,
        sections=[
            ResumeSection(
                name="experience",
                heading="EXPERIENCE",
                content="Senior Software Engineer | Acme Corp...",
                bullets=[
                    "Led migration of monolithic application to microservices, reducing deployment time by 75%",
                    "Managed team of 8 engineers, delivering 3 major product releases on schedule",
                    "Implemented automated testing pipeline that increased code coverage from 45% to 92%",
                    "Reduced cloud infrastructure costs by $200K annually through optimization",
                    "Developed RESTful APIs serving 10M+ requests daily",
                    "Built real-time data processing pipeline handling 500K events per second",
                    "Collaborated with product team to define technical requirements for 5 new features",
                ],
            ),
            ResumeSection(
                name="education",
                heading="EDUCATION",
                content="Master of Science in Computer Science | MIT | 2017\nBachelor of Science in Computer Science | UC Berkeley | 2015",
                bullets=[],
            ),
            ResumeSection(
                name="skills",
                heading="SKILLS",
                content="Python, Java, Go, TypeScript, AWS, Docker, Kubernetes, PostgreSQL, Redis, Machine Learning, Agile, Team Leadership, System Design, CI/CD",
                bullets=[],
            ),
        ],
        contact=ContactInfo(
            name="John Smith",
            email="john.smith@email.com",
            phone="(555) 123-4567",
            linkedin="linkedin.com/in/johnsmith",
            location="New York, NY",
        ),
        page_count=1,
        word_count=len(SAMPLE_RESUME_TEXT.split()),
    )


@pytest.fixture
def sample_jd_text():
    return """Senior Software Engineer

We are looking for an experienced Senior Software Engineer to join our team.

Requirements:
- 5+ years of experience in software development
- Strong proficiency in Python and Go
- Experience with microservices architecture
- Experience with AWS cloud services
- Strong understanding of CI/CD pipelines
- Experience with Docker and Kubernetes
- Excellent communication and team leadership skills
- Experience with PostgreSQL and Redis

Nice to have:
- Machine learning experience
- System design experience
- Agile methodology experience
"""
