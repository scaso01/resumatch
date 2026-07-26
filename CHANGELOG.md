# Changelog

## [Unreleased]

### Added
- End-to-end journey tests over real files: `test_ingest_integration.py` parses
  actual PDF/DOCX fixtures with nothing mocked, and `test_api_journey.py` drives
  `/analyze` and `/analyze-with-jd` through the unmocked pipeline
- `test_dashboard_apptest.py` renders the Streamlit dashboard via `AppTest`
- Fabricated resume/JD fixtures in `tests/fixtures/` (see `_generate.py`)
- Environment overrides for every config value:
  `RESUMATCH_<SECTION>__<KEY>`, with unknown keys failing at startup
- GitHub Actions CI on Python 3.12 and 3.13

### Changed
- **License is now MIT.** It was Business Source License 1.1 while the README
  advertised MIT; MIT is the intent.
- Default `llm.base_url` is `http://localhost:8080`, not a LAN address
- Documented scoring weights corrected to Impact 40 / Presentation 20 /
  Competencies 40, matching `config.yaml` and `docs/scoring_methodology.md`

### Removed
- `tests/test_dashboard.py`: 185 tests disabled by a hardcoded `skipif(True)`
  that duplicated the engine unit tests and had never executed
- Unused `pydantic-settings` dependency, and the dead `RESUMATCH_CONFIG` and
  `dashboard.api_url` settings that no code read

### Fixed
- `.env.example` documented six variables that nothing read; it now lists only
  real ones

## [0.1.0] - 2026-03-16

### Added
- Resume parsing for PDF (pdfplumber) and DOCX (python-docx)
- Section detection with 40+ heading variations mapped to 10 canonical names
- Feature extraction: action verbs (147 strong, 26 weak), quantification, specificity, grammar, formatting
- Three-module scoring engine: Impact (40%), Presentation (20%), Competencies (40%)
- Score zones: Green (85-100), Yellow (50-84), Red (0-49)
- Rules-based feedback generation with priority sorting (HIGH > MEDIUM > LOW)
- Optional LLM feedback enhancement via any OpenAI-compatible endpoint
- JD matching: KeyBERT keyword extraction + sentence-transformer semantic similarity
- FastAPI REST API with 8 endpoints
- Streamlit dashboard with score cards, radar chart, feedback tabs, JD match view
- Docker Compose deployment (API :8510, Dashboard :8511)
- 294 unit tests with network/LLM/log guards
- YAML-based configuration with validation
