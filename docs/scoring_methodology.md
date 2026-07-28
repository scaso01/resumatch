# Scoring Methodology

## Overview

ResuMatch scores resumes across three modules, inspired by VMock's patent (US10346803B2) and university career center methodologies. The final score is a weighted sum classified into zones.

## Modules

### Impact (40% weight)

Measures whether bullet points demonstrate concrete achievements.

| Metric | Target | What it measures |
|--------|--------|-----------------|
| Action verb rate | 90%+ | Bullets starting with strong verbs (Led, Developed, Implemented...) |
| Quantification rate | 60%+ | Bullets containing numbers, percentages, dollar amounts |
| Specificity score | 70%+ | Concrete vs vague language (technologies, project names, outcomes) |

**Scoring formula:**
- Action verb component: `verb_rate / target` (capped at 1.0) × 40
- Quantification component: `quant_rate / target` (capped at 1.0) × 35
- Specificity component: `specificity / target` (capped at 1.0) × 25
- Weak verb penalty: -5 per unique weak verb (helped, assisted, worked on...)

### Presentation (20% weight)

Measures resume structure and polish.

| Metric | Target | What it measures |
|--------|--------|-----------------|
| Page length | 1-2 pages | Ideal resume length |
| Grammar | 0 errors | Spelling, grammar, punctuation |
| Section completeness | All required | Experience, Education, Skills present |
| Formatting | Consistent | Dates, bullets, headings uniform |
| Contact info | Complete | Name, email, phone, LinkedIn, location |

**Scoring formula:**
- Page length: 1.0 if 1-2 pages, 0.5 if 3, 0.3 if 4+
- Grammar: 100 - (errors × penalty), capped at penalty_cap

**About the grammar check.** Spelling and grammar come from
[LanguageTool](https://languagetool.org/) running locally through
`language-tool-python`, which needs a Java runtime. Without Java those checks
are skipped and only formatting consistency (double spaces, bullet punctuation)
is measured.

LanguageTool's dictionary is general English, so it does not recognise most
technology names. Rather than penalise a resume for naming its tools, flagged
words that read as technology or proper nouns are ignored: anything containing
a digit or `. / + # _`, anything in all capitals, anything with an internal
capital, and any capitalised word that is not opening a sentence. So `FastAPI`,
`AWS`, `CI/CD`, `Redis` and `Northwind` pass, while `recieved` and a
bullet-opening `Builded` are still caught. The trade-off is deliberate: an
unrecognised lowercase tool name can still be flagged, which is the safer
direction to fail in. Grammar and agreement errors are never filtered.
- Sections: present_required / total_required
- Contact: filled_fields / total_fields (5)
- Formatting: heuristic score from consistency checks

### Competencies (40% weight)

Measures skills breadth and relevance.

| Metric | Target | What it measures |
|--------|--------|-----------------|
| Skills count | 15+ | Distinct identifiable skills |
| Category coverage | 4+ categories | Technical, leadership, communication, analytical, domain |
| Relevance | Industry-standard | Recognized skill keywords vs informal descriptions |

**Scoring formula:**
- Breadth: `min(skills_count / min_skills, 1.0)` × 40
- Category: `min(category_count / min_categories, 1.0)` × 30
- Relevance: `relevance_score` × 30

## Overall Score

```
overall = (impact_score × 0.40) + (presentation_score × 0.20) + (competencies_score × 0.40)
```

## Zones

| Zone | Range | Meaning |
|------|-------|---------|
| Green | 85-100 | Strong resume, minor tweaks only |
| Yellow | 50-84 | Good foundation, needs improvement |
| Red | 0-49 | Significant revision needed |

## JD Matching (Separate)

When a job description is provided:

1. **Keyword extraction**: KeyBERT extracts top-N keywords from both resume and JD
2. **Keyword match**: Jaccard-like overlap of extracted keywords
3. **Semantic match**: Cosine similarity of sentence-transformer embeddings (all-MiniLM-L6-v2)
4. **Overall match**: `(keyword_weight × keyword_match) + (semantic_weight × semantic_match)`

Default weights: 50% keyword, 50% semantic.

## References

- VMock Patent: US10346803B2 (SoftScore, keyword matching, feedback loops)
- University career center scoring rubrics (Impact/Presentation/Competencies framework)
- Resume NLP research (action verb detection, quantification extraction)
