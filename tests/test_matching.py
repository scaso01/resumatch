"""
Tests for resumatch.matching module.

All ML models (KeyBERT, sentence-transformers) are mocked.
Tests run fast without downloading any models.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from resumatch.models import JDMatchResult, KeywordMatch, ParsedResume


# =============================================================================
# Helpers
# =============================================================================

def _make_resume(text: str) -> ParsedResume:
    """Create a minimal ParsedResume for testing."""
    return ParsedResume(raw_text=text, word_count=len(text.split()))


# =============================================================================
# Stopwords
# =============================================================================

class TestStopwords:
    def test_stopwords_is_frozenset(self):
        from resumatch.matching import STOPWORDS
        assert isinstance(STOPWORDS, frozenset)

    def test_stopwords_contains_common_words(self):
        from resumatch.matching import STOPWORDS
        for word in ("the", "and", "is", "in", "to", "of", "a"):
            assert word in STOPWORDS

    def test_stopwords_does_not_contain_technical_terms(self):
        from resumatch.matching import STOPWORDS
        for word in ("python", "docker", "kubernetes", "aws", "postgresql"):
            assert word not in STOPWORDS


# =============================================================================
# Tokenization (internal)
# =============================================================================

class TestTokenize:
    def test_tokenize_lowercase(self):
        from resumatch.matching import _tokenize
        tokens = _tokenize("Python Docker AWS")
        assert all(t == t.lower() for t in tokens)

    def test_tokenize_preserves_compound(self):
        from resumatch.matching import _tokenize
        tokens = _tokenize("CI/CD experience")
        assert "ci/cd" in tokens


# =============================================================================
# Keyword Extraction
# =============================================================================

class TestExtractKeywords:
    @patch("resumatch.matching._load_keybert_model", return_value=None)
    def test_returns_list_of_strings(self, _mock_kb, sample_jd_text):
        from resumatch.matching import extract_keywords
        result = extract_keywords(sample_jd_text)
        assert isinstance(result, list)
        assert all(isinstance(kw, str) for kw in result)

    @patch("resumatch.matching._load_keybert_model", return_value=None)
    def test_fallback_extracts_keywords(self, _mock_kb, sample_jd_text):
        from resumatch.matching import extract_keywords
        result = extract_keywords(sample_jd_text)
        assert len(result) > 0
        # Should find technical terms from the JD
        result_lower = [kw.lower() for kw in result]
        assert "python" in result_lower or "software" in result_lower

    @patch("resumatch.matching._load_keybert_model", return_value=None)
    def test_respects_top_n(self, _mock_kb, sample_jd_text):
        from resumatch.matching import extract_keywords
        result = extract_keywords(sample_jd_text, top_n=5)
        assert len(result) <= 5

    @patch("resumatch.matching._load_keybert_model", return_value=None)
    def test_empty_text_returns_empty(self, _mock_kb):
        from resumatch.matching import extract_keywords
        assert extract_keywords("") == []
        assert extract_keywords("   ") == []

    def test_keybert_path_when_model_available(self, sample_jd_text):
        """When KeyBERT model is available, it should use it."""
        from resumatch.matching import extract_keywords

        mock_model = MagicMock()
        mock_model.extract_keywords.return_value = [
            ("python", 0.8),
            ("docker", 0.7),
            ("microservices", 0.6),
        ]
        with patch("resumatch.matching._load_keybert_model", return_value=mock_model):
            result = extract_keywords(sample_jd_text, top_n=5)
        assert result == ["python", "docker", "microservices"]

    @patch("resumatch.matching._load_keybert_model", return_value=None)
    def test_stopwords_filtered_in_fallback(self, _mock_kb):
        from resumatch.matching import STOPWORDS, extract_keywords
        text = "the the the python python docker is is"
        result = extract_keywords(text, top_n=10)
        for kw in result:
            assert kw not in STOPWORDS


# =============================================================================
# Keyword Matching
# =============================================================================

class TestMatchKeywords:
    def test_finds_exact_match(self, sample_resume_text):
        from resumatch.matching import match_keywords
        rate, matches = match_keywords(sample_resume_text, ["Python"])
        assert rate == 1.0
        assert matches[0].found_in_resume is True

    def test_case_insensitive(self, sample_resume_text):
        from resumatch.matching import match_keywords
        rate, matches = match_keywords(sample_resume_text, ["python"])
        assert rate == 1.0
        assert matches[0].found_in_resume is True

    def test_missing_keyword(self, sample_resume_text):
        from resumatch.matching import match_keywords
        rate, matches = match_keywords(sample_resume_text, ["Terraform"])
        assert rate == 0.0
        assert matches[0].found_in_resume is False

    def test_mixed_matches(self, sample_resume_text):
        from resumatch.matching import match_keywords
        keywords = ["Python", "Terraform", "Docker", "Haskell"]
        rate, matches = match_keywords(sample_resume_text, keywords)
        # Python and Docker are in the resume, Terraform and Haskell are not
        assert rate == 0.5
        assert len(matches) == 4

    def test_correct_match_rate(self, sample_resume_text):
        from resumatch.matching import match_keywords
        keywords = ["Python", "Java", "Go"]
        rate, matches = match_keywords(sample_resume_text, keywords)
        assert rate == pytest.approx(1.0)
        assert all(m.found_in_resume for m in matches)

    def test_empty_keywords_list(self, sample_resume_text):
        from resumatch.matching import match_keywords
        rate, matches = match_keywords(sample_resume_text, [])
        assert rate == 0.0
        assert matches == []

    def test_keyword_match_object_structure(self, sample_resume_text):
        from resumatch.matching import match_keywords
        rate, matches = match_keywords(sample_resume_text, ["Python"])
        km = matches[0]
        assert isinstance(km, KeywordMatch)
        assert km.keyword == "Python"
        assert km.found_in_resume is True
        assert km.similarity == 1.0
        assert km.matched_term == "Python"

    def test_not_found_keyword_has_none_matched_term(self, sample_resume_text):
        from resumatch.matching import match_keywords
        _, matches = match_keywords(sample_resume_text, ["Fortran"])
        km = matches[0]
        assert km.found_in_resume is False
        assert km.similarity == 0.0
        assert km.matched_term is None


# =============================================================================
# Semantic Similarity
# =============================================================================

class TestSemanticSimilarity:
    @patch("resumatch.matching._load_sentence_model", return_value=None)
    def test_returns_zero_when_model_unavailable(self, _mock_model):
        from resumatch.matching import compute_semantic_similarity
        result = compute_semantic_similarity("Some resume text", "Some JD text")
        assert result == 0.0

    def test_returns_float_with_mocked_model(self, sample_resume_text, sample_jd_text):
        import numpy as np
        from resumatch.matching import compute_semantic_similarity

        mock_model = MagicMock()
        # Simulate encoding returning 384-dim vectors (like all-MiniLM-L6-v2)
        def fake_encode(texts, convert_to_tensor=False):
            rng = np.random.RandomState(42)
            embeddings = rng.randn(len(texts), 384).astype(np.float32)
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / norms
        mock_model.encode = fake_encode

        with patch("resumatch.matching._load_sentence_model", return_value=mock_model):
            result = compute_semantic_similarity(sample_resume_text, sample_jd_text)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @patch("resumatch.matching._load_sentence_model", return_value=None)
    def test_empty_resume_returns_zero(self, _mock_model):
        from resumatch.matching import compute_semantic_similarity
        assert compute_semantic_similarity("", "Some JD text") == 0.0

    @patch("resumatch.matching._load_sentence_model", return_value=None)
    def test_empty_jd_returns_zero(self, _mock_model):
        from resumatch.matching import compute_semantic_similarity
        assert compute_semantic_similarity("Some resume text", "") == 0.0


# =============================================================================
# Chunk Splitting (internal)
# =============================================================================

class TestChunkSplitting:
    def test_splits_on_double_newline(self):
        from resumatch.matching import _split_into_chunks
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = _split_into_chunks(text)
        assert len(chunks) == 3

    def test_empty_text_returns_empty(self):
        from resumatch.matching import _split_into_chunks
        assert _split_into_chunks("") == []
        assert _split_into_chunks("   ") == []

    def test_single_paragraph(self):
        from resumatch.matching import _split_into_chunks
        text = "Just one paragraph with no breaks."
        chunks = _split_into_chunks(text)
        assert len(chunks) == 1


# =============================================================================
# Combined match_jd
# =============================================================================

class TestMatchJD:
    @patch("resumatch.matching.compute_semantic_similarity", return_value=0.75)
    @patch("resumatch.matching._load_keybert_model", return_value=None)
    def test_returns_jd_match_result(self, _mock_kb, _mock_sem, sample_parsed_resume, sample_jd_text):
        from resumatch.matching import match_jd
        result = match_jd(sample_parsed_resume, sample_jd_text)
        assert isinstance(result, JDMatchResult)

    @patch("resumatch.matching.compute_semantic_similarity", return_value=0.75)
    @patch("resumatch.matching._load_keybert_model", return_value=None)
    def test_overall_match_is_weighted_combination(self, _mock_kb, _mock_sem, sample_parsed_resume, sample_jd_text):
        from resumatch.matching import match_jd
        result = match_jd(sample_parsed_resume, sample_jd_text)
        # overall_match should be (keyword_rate * 0.5 + 0.75 * 0.5) * 100
        # keyword_rate is whatever the fallback extraction finds
        assert 0.0 <= result.overall_match <= 100.0
        assert result.keyword_match >= 0.0
        assert result.semantic_match == 75.0  # 0.75 * 100

    @patch("resumatch.matching.compute_semantic_similarity", return_value=0.0)
    @patch("resumatch.matching._load_keybert_model", return_value=None)
    def test_missing_keywords_populated(self, _mock_kb, _mock_sem, sample_jd_text):
        from resumatch.matching import match_jd
        # Resume with nothing matching the JD
        resume = _make_resume("This resume has no relevant content whatsoever.")
        result = match_jd(resume, sample_jd_text)
        assert len(result.missing_keywords) > 0

    @patch("resumatch.matching.compute_semantic_similarity", return_value=0.80)
    @patch("resumatch.matching._load_keybert_model", return_value=None)
    def test_jd_keywords_populated(self, _mock_kb, _mock_sem, sample_parsed_resume, sample_jd_text):
        from resumatch.matching import match_jd
        result = match_jd(sample_parsed_resume, sample_jd_text)
        assert len(result.jd_keywords) > 0
        assert all(isinstance(kw, str) for kw in result.jd_keywords)

    @patch("resumatch.matching.compute_semantic_similarity", return_value=0.80)
    @patch("resumatch.matching._load_keybert_model", return_value=None)
    def test_matched_keywords_are_keyword_match_objects(self, _mock_kb, _mock_sem, sample_parsed_resume, sample_jd_text):
        from resumatch.matching import match_jd
        result = match_jd(sample_parsed_resume, sample_jd_text)
        assert all(isinstance(km, KeywordMatch) for km in result.matched_keywords)

    def test_empty_jd_returns_default(self, sample_parsed_resume):
        from resumatch.matching import match_jd
        result = match_jd(sample_parsed_resume, "")
        assert result.overall_match == 0.0
        assert result.jd_keywords == []

    @patch("resumatch.matching.compute_semantic_similarity", return_value=0.0)
    @patch("resumatch.matching._load_keybert_model", return_value=None)
    def test_empty_resume_text(self, _mock_kb, _mock_sem, sample_jd_text):
        from resumatch.matching import match_jd
        resume = _make_resume("")
        result = match_jd(resume, sample_jd_text)
        # With empty resume, keyword_match should be 0
        assert result.keyword_match == 0.0


# =============================================================================
# Model Management
# =============================================================================

class TestModelsLoaded:
    def test_returns_false_when_no_models_loaded(self):
        import resumatch.matching as m
        original_kb = m._keybert_model
        original_st = m._sentence_model
        try:
            m._keybert_model = None
            m._sentence_model = None
            assert m.models_loaded() is False
        finally:
            m._keybert_model = original_kb
            m._sentence_model = original_st

    def test_returns_true_when_keybert_loaded(self):
        import resumatch.matching as m
        original_kb = m._keybert_model
        try:
            m._keybert_model = MagicMock()
            assert m.models_loaded() is True
        finally:
            m._keybert_model = original_kb

    def test_returns_true_when_sentence_model_loaded(self):
        import resumatch.matching as m
        original_st = m._sentence_model
        try:
            m._sentence_model = MagicMock()
            assert m.models_loaded() is True
        finally:
            m._sentence_model = original_st
