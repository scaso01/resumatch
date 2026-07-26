"""
ResuMatch - Pydantic Models
=============================
Data models for resume parsing, scoring, matching, and feedback.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ScoreZone(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class FeedbackPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FeedbackCategory(str, Enum):
    IMPACT = "impact"
    PRESENTATION = "presentation"
    COMPETENCIES = "competencies"
    GENERAL = "general"


class SkillCategory(str, Enum):
    TECHNICAL = "technical"
    LEADERSHIP = "leadership"
    COMMUNICATION = "communication"
    ANALYTICAL = "analytical"
    DOMAIN = "domain"


class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None


class ResumeSection(BaseModel):
    name: str
    heading: str
    content: str
    bullets: list[str] = Field(default_factory=list)
    start_line: int = 0
    end_line: int = 0


class ParsedResume(BaseModel):
    raw_text: str
    sections: list[ResumeSection] = Field(default_factory=list)
    contact: ContactInfo = Field(default_factory=ContactInfo)
    page_count: int = 1
    word_count: int = 0
    file_name: Optional[str] = None
    file_type: Optional[str] = None


class ImpactScore(BaseModel):
    score: float = 0.0
    action_verb_rate: float = 0.0
    quantification_rate: float = 0.0
    specificity_score: float = 0.0
    weak_verbs_found: list[str] = Field(default_factory=list)
    bullets_without_metrics: list[str] = Field(default_factory=list)


class PresentationScore(BaseModel):
    score: float = 0.0
    page_length_score: float = 0.0
    grammar_score: float = 0.0
    section_completeness: float = 0.0
    formatting_score: float = 0.0
    contact_completeness: float = 0.0
    grammar_errors: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)


class CompetenciesScore(BaseModel):
    score: float = 0.0
    skills_found: list[str] = Field(default_factory=list)
    skills_count: int = 0
    categories_covered: list[str] = Field(default_factory=list)
    category_count: int = 0
    relevance_score: float = 0.0


class ResumeScore(BaseModel):
    overall: float = 0.0
    zone: ScoreZone = ScoreZone.RED
    impact: ImpactScore = Field(default_factory=ImpactScore)
    presentation: PresentationScore = Field(default_factory=PresentationScore)
    competencies: CompetenciesScore = Field(default_factory=CompetenciesScore)
    weights: dict[str, int] = Field(default_factory=lambda: {"impact": 40, "presentation": 20, "competencies": 40})


class FeedbackItem(BaseModel):
    category: FeedbackCategory
    priority: FeedbackPriority
    message: str
    suggestion: Optional[str] = None
    source: str = "rules"


class KeywordMatch(BaseModel):
    keyword: str
    found_in_resume: bool = False
    similarity: float = 0.0
    matched_term: Optional[str] = None


class JDMatchResult(BaseModel):
    overall_match: float = 0.0
    keyword_match: float = 0.0
    semantic_match: float = 0.0
    matched_keywords: list[KeywordMatch] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    jd_keywords: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    resume: ParsedResume
    score: ResumeScore
    feedback: list[FeedbackItem] = Field(default_factory=list)
    jd_match: Optional[JDMatchResult] = None


class BulletImprovement(BaseModel):
    original: str
    rewritten: str
    section: str
    reason: str


class ImproveResult(BaseModel):
    improvements: list[BulletImprovement] = Field(default_factory=list)
    llm_available: bool = True
    bullets_analyzed: int = 0
    bullets_improved: int = 0


class HealthStatus(BaseModel):
    status: str = "ok"
    spacy_loaded: bool = False
    llm_available: bool = False
    models_loaded: bool = False
    version: str = "0.1.0"
