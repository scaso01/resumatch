"""
ResuMatch - JD Matching Module
================================
Keyword extraction, keyword matching, semantic similarity,
and combined JD-resume matching.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from resumatch.config import CONFIG, logger
from resumatch.models import JDMatchResult, KeywordMatch, ParsedResume

# =============================================================================
# Module-level model cache
# =============================================================================
_keybert_model = None
_sentence_model = None

# =============================================================================
# Matching config from config.yaml
# =============================================================================
_MATCHING_CFG = CONFIG.get("matching", {})
KEYWORD_TOP_N: int = _MATCHING_CFG.get("keyword_top_n", 20)
SEMANTIC_THRESHOLD: float = _MATCHING_CFG.get("semantic_threshold", 0.45)
KEYWORD_WEIGHT: float = _MATCHING_CFG.get("keyword_weight", 0.5)
SEMANTIC_WEIGHT: float = _MATCHING_CFG.get("semantic_weight", 0.5)

# =============================================================================
# Stopwords (~150 common English words)
# =============================================================================
STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "get", "got", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't",
    "it", "its", "itself", "just", "let", "like", "ll", "me", "might",
    "more", "most", "must", "mustn't", "my", "myself", "need", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shall",
    "shan't", "she", "should", "shouldn't", "so", "some", "such", "t",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "us", "ve", "very", "was", "wasn't", "we",
    "were", "weren't", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "won't", "would", "wouldn't", "years",
    "you", "your", "yours", "yourself", "yourselves",
    # Common resume/JD filler words
    "experience", "looking", "join", "team", "work", "working", "also",
    "well", "good", "great", "new", "use", "using", "used", "able",
    "including", "etc", "e.g", "i.e", "please", "required", "preferred",
    "strong", "excellent", "nice", "have", "must",
})

# =============================================================================
# Model Management
# =============================================================================

def _load_keybert_model():
    """Lazy-load KeyBERT model. Returns model or None if unavailable."""
    global _keybert_model
    if _keybert_model is not None:
        return _keybert_model
    try:
        from keybert import KeyBERT
        _keybert_model = KeyBERT(model="all-MiniLM-L6-v2")
        logger.info("KeyBERT model loaded successfully")
        return _keybert_model
    except Exception as e:
        logger.warning("KeyBERT model not available, using fallback extraction: %s", e)
        return None


def _load_sentence_model():
    """Lazy-load sentence-transformers model. Returns model or None if unavailable."""
    global _sentence_model
    if _sentence_model is not None:
        return _sentence_model
    try:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Sentence transformer model loaded successfully")
        return _sentence_model
    except Exception as e:
        logger.warning("Sentence transformer model not available: %s", e)
        return None


def models_loaded() -> bool:
    """Check if embedding models are currently loaded (cached)."""
    return _keybert_model is not None or _sentence_model is not None

# =============================================================================
# Keyword Extraction
# =============================================================================

def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, stripping punctuation."""
    # Split on non-alphanumeric (keep hyphens for compound terms like CI/CD)
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#/.-]*", text.lower())
    return tokens


def _simple_extract_keywords(text: str, top_n: int = 20) -> list[str]:
    """Fallback keyword extraction using frequency counts.

    Tokenizes text, removes stopwords and very short tokens,
    counts frequencies, and returns the top N.
    """
    tokens = _tokenize(text)
    # Filter: remove stopwords and tokens shorter than 2 chars
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) >= 2]
    counts = Counter(filtered)
    # Return top N by frequency, preserving order
    return [word for word, _ in counts.most_common(top_n)]


def extract_keywords(text: str, top_n: int = 20) -> list[str]:
    """Extract top keywords from JD text.

    Tries KeyBERT first for quality keyword extraction.
    Falls back to simple frequency-based extraction if KeyBERT is unavailable.

    Args:
        text: Job description text.
        top_n: Maximum number of keywords to return.

    Returns:
        List of keyword strings, ordered by relevance.
    """
    if not text or not text.strip():
        return []

    # Try KeyBERT first
    kb_model = _load_keybert_model()
    if kb_model is not None:
        try:
            keywords_with_scores = kb_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=top_n,
                use_mmr=True,
                diversity=0.5,
            )
            keywords = [kw for kw, _score in keywords_with_scores]
            if keywords:
                logger.debug("Extracted %d keywords via KeyBERT", len(keywords))
                return keywords
        except Exception as e:
            logger.warning("KeyBERT extraction failed, using fallback: %s", e)

    # Fallback: simple frequency-based
    keywords = _simple_extract_keywords(text, top_n)
    logger.debug("Extracted %d keywords via simple fallback", len(keywords))
    return keywords

# =============================================================================
# Keyword Matching
# =============================================================================

def match_keywords(
    resume_text: str, jd_keywords: list[str]
) -> tuple[float, list[KeywordMatch]]:
    """Check which JD keywords appear in the resume text (case-insensitive).

    Args:
        resume_text: Full resume text to search.
        jd_keywords: List of keywords extracted from the JD.

    Returns:
        Tuple of (match_rate as 0.0-1.0, list of KeywordMatch objects).
    """
    if not jd_keywords:
        return 0.0, []

    resume_lower = resume_text.lower()
    matches: list[KeywordMatch] = []
    found_count = 0

    for keyword in jd_keywords:
        kw_lower = keyword.lower()
        found = kw_lower in resume_lower
        matched_term: Optional[str] = None

        if found:
            found_count += 1
            # Find the actual matched term in context for reporting
            matched_term = keyword

        matches.append(
            KeywordMatch(
                keyword=keyword,
                found_in_resume=found,
                similarity=1.0 if found else 0.0,
                matched_term=matched_term,
            )
        )

    match_rate = found_count / len(jd_keywords)
    return match_rate, matches

# =============================================================================
# Semantic Matching
# =============================================================================

def _split_into_chunks(text: str, max_chunk_len: int = 512) -> list[str]:
    """Split text into chunks by paragraphs, respecting max length.

    Splits on double newlines (paragraphs) first. If a paragraph exceeds
    max_chunk_len, it is further split by sentences.
    """
    if not text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chunk_len:
            chunks.append(para)
        else:
            # Split long paragraphs by sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) + 1 <= max_chunk_len:
                    current_chunk = f"{current_chunk} {sent}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sent
            if current_chunk:
                chunks.append(current_chunk)

    return chunks if chunks else [text.strip()[:max_chunk_len]]


def compute_semantic_similarity(
    resume_text: str, jd_text: str, top_k: int = 5
) -> float:
    """Compute semantic similarity between resume and JD using embeddings.

    Breaks both texts into chunks, computes cosine similarity between all
    resume-JD chunk pairs, and returns the average of the top-K similarities.

    Args:
        resume_text: Full resume text.
        jd_text: Full job description text.
        top_k: Number of top chunk similarities to average.

    Returns:
        Similarity score 0.0-1.0, or 0.0 if model unavailable.
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    model = _load_sentence_model()
    if model is None:
        return 0.0

    try:
        resume_chunks = _split_into_chunks(resume_text)
        jd_chunks = _split_into_chunks(jd_text)

        if not resume_chunks or not jd_chunks:
            return 0.0

        # Encode all chunks
        resume_embeddings = model.encode(resume_chunks, convert_to_tensor=False)
        jd_embeddings = model.encode(jd_chunks, convert_to_tensor=False)

        # Compute cosine similarities between all pairs
        # Using numpy for efficiency
        import numpy as np

        # Normalize embeddings
        resume_norms = resume_embeddings / (
            np.linalg.norm(resume_embeddings, axis=1, keepdims=True) + 1e-10
        )
        jd_norms = jd_embeddings / (
            np.linalg.norm(jd_embeddings, axis=1, keepdims=True) + 1e-10
        )

        # Cosine similarity matrix: (num_resume_chunks x num_jd_chunks)
        similarity_matrix = np.dot(resume_norms, jd_norms.T)

        # Flatten and take top-K similarities
        all_sims = similarity_matrix.flatten()
        top_k_actual = min(top_k, len(all_sims))
        top_sims = np.sort(all_sims)[-top_k_actual:]

        avg_similarity = float(np.mean(top_sims))
        # Clamp to [0, 1]
        return max(0.0, min(1.0, avg_similarity))

    except Exception as e:
        logger.warning("Semantic similarity computation failed: %s", e)
        return 0.0

# =============================================================================
# Combined Matching
# =============================================================================

def match_jd(resume: ParsedResume, jd_text: str) -> JDMatchResult:
    """Combine keyword and semantic matching for a resume against a JD.

    Uses config weights (default 50/50) to produce an overall match score.

    Args:
        resume: Parsed resume object with raw_text.
        jd_text: Job description text.

    Returns:
        JDMatchResult with overall_match, keyword_match, semantic_match,
        matched_keywords, missing_keywords, and jd_keywords.
    """
    if not jd_text or not jd_text.strip():
        return JDMatchResult()

    resume_text = resume.raw_text

    # Step 1: Extract keywords from JD
    top_n = _MATCHING_CFG.get("keyword_top_n", KEYWORD_TOP_N)
    jd_keywords = extract_keywords(jd_text, top_n=top_n)

    # Step 2: Keyword matching
    keyword_rate, keyword_matches = match_keywords(resume_text, jd_keywords)

    # Step 3: Semantic similarity
    semantic_score = compute_semantic_similarity(resume_text, jd_text)

    # Step 4: Combine scores using config weights
    kw_weight = _MATCHING_CFG.get("keyword_weight", KEYWORD_WEIGHT)
    sem_weight = _MATCHING_CFG.get("semantic_weight", SEMANTIC_WEIGHT)
    overall = (keyword_rate * kw_weight + semantic_score * sem_weight)

    # Step 5: Identify missing keywords
    missing = [km.keyword for km in keyword_matches if not km.found_in_resume]

    logger.debug(
        "JD match: overall=%.2f keyword=%.2f semantic=%.2f (%d/%d keywords matched)",
        overall, keyword_rate, semantic_score,
        len(jd_keywords) - len(missing), len(jd_keywords),
    )

    return JDMatchResult(
        overall_match=round(overall * 100, 1),
        keyword_match=round(keyword_rate * 100, 1),
        semantic_match=round(semantic_score * 100, 1),
        matched_keywords=keyword_matches,
        missing_keywords=missing,
        jd_keywords=jd_keywords,
    )
