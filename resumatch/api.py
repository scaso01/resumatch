"""
ResuMatch - FastAPI REST API
==============================
Upload PDF/DOCX resumes for scoring, feedback, and JD matching.
"""

from __future__ import annotations

import io
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from resumatch.config import CONFIG, logger
from resumatch.models import (
    AnalysisResult,
    BulletImprovement,
    HealthStatus,
    ImproveResult,
    JDMatchResult,
    ParsedResume,
    ResumeScore,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: preload spaCy and embedding models."""
    logger.info("ResuMatch API starting up...")
    try:
        from resumatch.matching import models_loaded
        if models_loaded():
            logger.info("Embedding models loaded.")
    except Exception as e:
        logger.warning("Could not preload models: %s", e)
    yield
    logger.info("ResuMatch API shutting down.")


app = FastAPI(
    title="ResuMatch",
    description="Open-source resume scoring engine",
    version="0.1.0",
    lifespan=lifespan,
)

cors_origins = CONFIG.get("api", {}).get("cors_origins", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_MB = CONFIG.get("api", {}).get("max_upload_mb", 10)


async def _read_upload(file: UploadFile) -> tuple[bytes, str]:
    """Read uploaded file and validate size/type."""
    if not file.filename:
        raise HTTPException(400, "No filename provided.")
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ("pdf", "docx"):
        raise HTTPException(400, f"Unsupported file type: .{ext}. Use PDF or DOCX.")
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB}MB limit.")
    return content, file.filename


@app.post("/api/v1/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    """Upload PDF/DOCX and get full score + feedback."""
    content, filename = await _read_upload(file)
    try:
        from resumatch.ingest import parse_file
        from resumatch.scoring import score_resume
        from resumatch.feedback import generate_feedback

        parsed = parse_file(io.BytesIO(content), filename)
        logger.info("Sections detected: %s", [(s.name, len(s.bullets)) for s in parsed.sections])
        if not parsed.sections:
            # Dump first 500 chars of raw text for debugging
            # Dump all short lines (potential headings)
            short_lines = [
                f"  [{i}] {repr(line.strip())}"
                for i, line in enumerate(parsed.raw_text.splitlines())
                if line.strip() and len(line.strip()) < 60
            ]
            logger.warning("NO SECTIONS DETECTED. Short lines:\n%s", "\n".join(short_lines[:30]))
        score = score_resume(parsed)
        feedback = generate_feedback(score, parsed)
        return AnalysisResult(resume=parsed, score=score, feedback=feedback)
    except Exception as e:
        logger.error("Analysis failed: %s", e)
        raise HTTPException(500, f"Analysis failed: {e}")


@app.post("/api/v1/analyze-with-jd", response_model=AnalysisResult)
async def analyze_with_jd(
    file: UploadFile = File(...),
    jd_text: str = Form(...),
):
    """Upload PDF/DOCX + JD text and get score + feedback + JD match."""
    content, filename = await _read_upload(file)
    try:
        from resumatch.ingest import parse_file
        from resumatch.scoring import score_resume
        from resumatch.feedback import generate_feedback
        from resumatch.matching import match_jd

        parsed = parse_file(io.BytesIO(content), filename)
        score = score_resume(parsed)
        feedback = generate_feedback(score, parsed)
        jd_match = match_jd(parsed, jd_text)
        return AnalysisResult(resume=parsed, score=score, feedback=feedback, jd_match=jd_match)
    except Exception as e:
        logger.error("Analysis with JD failed: %s", e)
        raise HTTPException(500, f"Analysis failed: {e}")


@app.post("/api/v1/parse", response_model=ParsedResume)
async def parse(file: UploadFile = File(...)):
    """Parse resume file without scoring."""
    content, filename = await _read_upload(file)
    try:
        from resumatch.ingest import parse_file
        return parse_file(io.BytesIO(content), filename)
    except Exception as e:
        logger.error("Parse failed: %s", e)
        raise HTTPException(500, f"Parse failed: {e}")


@app.post("/api/v1/score", response_model=ResumeScore)
async def score(resume: ParsedResume):
    """Score a pre-parsed resume JSON."""
    try:
        from resumatch.scoring import score_resume
        return score_resume(resume)
    except Exception as e:
        logger.error("Scoring failed: %s", e)
        raise HTTPException(500, f"Scoring failed: {e}")


@app.post("/api/v1/match", response_model=JDMatchResult)
async def match(resume: ParsedResume, jd_text: str = Form(...)):
    """JD match only on a pre-parsed resume."""
    try:
        from resumatch.matching import match_jd
        return match_jd(resume, jd_text)
    except Exception as e:
        logger.error("Matching failed: %s", e)
        raise HTTPException(500, f"Matching failed: {e}")


@app.post("/api/v1/improve", response_model=ImproveResult)
async def improve(resume: ParsedResume):
    """Identify weak bullets and rewrite them via LLM."""
    try:
        from resumatch.features import STRONG_VERBS, WEAK_VERBS
        from resumatch.llm import get_client, is_llm_available

        client = get_client()
        available = await is_llm_available()

        # Gather experience bullets with their section name
        bullet_sections: list[tuple[str, str]] = []
        for section in resume.sections:
            if section.name.lower() in ("experience", "work experience", "work history", "employment"):
                for bullet in section.bullets:
                    bullet_sections.append((bullet, section.heading))
                # Fallback: use content lines if no bullets
                if not section.bullets and section.content.strip():
                    for line in section.content.splitlines():
                        text = line.strip()
                        if len(text) >= 10:
                            bullet_sections.append((text, section.heading))

        if not bullet_sections:
            return ImproveResult(llm_available=available, bullets_analyzed=0)

        # Identify weak bullets (no strong verb, or no metrics)
        weak_bullets: list[tuple[str, str]] = []
        for bullet, section_name in bullet_sections:
            first_word = bullet.strip().split()[0].lower().rstrip(".,;:") if bullet.strip() else ""
            has_strong_verb = first_word in STRONG_VERBS
            has_metrics = bool(re.search(r"\d", bullet))
            if not has_strong_verb or not has_metrics:
                weak_bullets.append((bullet, section_name))

        if not weak_bullets or not available:
            return ImproveResult(
                llm_available=available,
                bullets_analyzed=len(bullet_sections),
                bullets_improved=0,
            )

        # Rewrite weak bullets via LLM
        originals = [b for b, _ in weak_bullets]
        rewritten = await client.batch_rewrite(originals)

        improvements: list[BulletImprovement] = []
        for (original, section_name), new_text in zip(weak_bullets, rewritten):
            if new_text != original:
                first_word = original.strip().split()[0].lower().rstrip(".,;:") if original.strip() else ""
                reason = "weak verb" if first_word in WEAK_VERBS else "missing metrics"
                improvements.append(BulletImprovement(
                    original=original,
                    rewritten=new_text,
                    section=section_name,
                    reason=reason,
                ))

        return ImproveResult(
            improvements=improvements,
            llm_available=available,
            bullets_analyzed=len(bullet_sections),
            bullets_improved=len(improvements),
        )
    except Exception as e:
        logger.error("Improve failed: %s", e)
        raise HTTPException(500, f"Improve failed: {e}")


@app.get("/api/v1/health", response_model=HealthStatus)
async def health():
    """Health check — API + spaCy + LLM status."""
    status = HealthStatus(status="ok", version="0.1.0")
    try:
        import spacy
        spacy.load("en_core_web_sm")
        status.spacy_loaded = True
    except Exception:
        status.spacy_loaded = False

    try:
        from resumatch.llm import is_llm_available
        status.llm_available = await is_llm_available()
    except Exception:
        status.llm_available = False

    try:
        from resumatch.matching import models_loaded
        status.models_loaded = models_loaded()
    except Exception:
        status.models_loaded = False

    return status


@app.get("/api/v1/config")
async def get_config():
    """Return current scoring weights and thresholds (read-only)."""
    return {
        "scoring": CONFIG.get("scoring", {}),
        "matching": CONFIG.get("matching", {}),
    }
