"""Lexical (TF-IDF), skill-based, and semantic (embeddings) CV-to-Job matcher."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Canonical skill -> list of aliases (case-insensitive)
SKILL_TAXONOMY: dict[str, list[str]] = {
    # Programming languages
    "Python": ["python"],
    "SQL": ["sql"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp"],
    "Scala": ["scala"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript", "ts"],
    "HTML/CSS": ["html", "css"],
    "Bash": ["bash", "shell scripting"],
    # Data Science, ML & AI
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "neural networks"],
    "Artificial Intelligence": ["artificial intelligence", "ai", "agentic ai", "generative ai", "genai"],
    "NLP": ["nlp", "natural language processing", "text mining"],
    "Computer Vision": ["computer vision", "opencv", "image processing"],
    "Data Analysis": ["data analysis", "data analytics", "exploratory data analysis", "eda"],
    "Statistics": ["statistics", "statistical modeling", "probability"],
    "pandas": ["pandas"],
    "NumPy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "Matplotlib": ["matplotlib", "seaborn"],
    "PyTorch": ["pytorch"],
    "TensorFlow": ["tensorflow", "keras"],
    "LLMs": ["llm", "llms", "large language models", "rag", "langchain"],
    # Big Data, Databases & BI
    "Apache Spark": ["spark", "pyspark", "apache spark"],
    "Big Data": ["big data", "data pipelines", "etl", "data ingestion"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["excel"],
    # Cloud, DevOps & Engineering
    "Git": ["git", "github", "gitlab"],
    "Docker": ["docker", "containers"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Linux": ["linux", "unix"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud"],
    "REST APIs": ["rest api", "rest apis", "fastapi", "flask"],
    "Unit Testing": ["pytest", "unit testing", "unit tests", "integration testing", "integration tests", "tdd"],
    "Jira": ["jira"],
    "Agile": ["agile", "scrum"],
    "Operational Research": ["operational research", "operations research", "optimization"],
}

# Single-letter languages require case-sensitive matching to avoid false positives
CASE_SENSITIVE_SKILLS: dict[str, list[str]] = {
    "C": ["C"],
    "R": ["R"],
}


class MatchResult(BaseModel):
    jd_skills: list[str] = Field(default_factory=list)
    cv_skills: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    extra_cv_skills: list[str] = Field(default_factory=list)
    skill_coverage_score: float = 0.0
    lexical_similarity: float = 0.0
    semantic_similarity: float = 0.0
    model_name: str = DEFAULT_EMBEDDING_MODEL


def _compile_pattern(alias: str, case_sensitive: bool = False) -> re.Pattern[str]:
    """Build boundary-aware regex that handles symbols like C++ and C# cleanly."""
    escaped = re.escape(alias)
    pattern = rf"(?<![A-Za-z0-9_+#]){escaped}(?![A-Za-z0-9_+#]|\.[A-Za-z])"
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(pattern, flags)


@lru_cache(maxsize=1)
def _compiled_taxonomy() -> list[tuple[str, list[re.Pattern[str]]]]:
    compiled: list[tuple[str, list[re.Pattern[str]]]] = []
    for canonical, aliases in SKILL_TAXONOMY.items():
        compiled.append((canonical, [_compile_pattern(a, case_sensitive=False) for a in aliases]))
    for canonical, aliases in CASE_SENSITIVE_SKILLS.items():
        compiled.append((canonical, [_compile_pattern(a, case_sensitive=True) for a in aliases]))
    return compiled


def extract_skills(text: str) -> list[str]:
    """Extract canonical technical skills present in a given text."""
    if not text or not text.strip():
        return []

    found: list[str] = []
    for canonical, patterns in _compiled_taxonomy():
        if any(pattern.search(text) for pattern in patterns):
            found.append(canonical)
    return sorted(found)


def compute_lexical_similarity(cv_text: str, job_description: str) -> float:
    """Compute TF-IDF cosine similarity (0-100) as a lexical keyword baseline."""
    if not cv_text.strip() or not job_description.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([cv_text, job_description])
    except ValueError:
        return 0.0

    sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0])
    return round(max(0.0, min(1.0, sim)) * 100.0, 2)


def _chunk_text(text: str, chunk_size: int = 180, overlap: int = 40) -> list[str]:
    """Split long documents into overlapping word chunks to avoid transformer truncation."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str):
    """Lazily load and cache the sentence-transformer model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _encode_document(text: str, model_name: str) -> np.ndarray | None:
    """Compute an L2-normalized mean-pooled embedding across document chunks."""
    chunks = _chunk_text(text)
    if not chunks:
        return None

    model = _load_embedding_model(model_name)
    chunk_embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    mean_vec = np.mean(np.asarray(chunk_embeddings, dtype=np.float32), axis=0)
    norm = float(np.linalg.norm(mean_vec))
    return mean_vec / norm if norm > 0 else mean_vec


def compute_semantic_similarity(
    cv_text: str,
    job_description: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> float:
    """Compute semantic cosine similarity (0-100) using sentence embeddings."""
    vec_cv = _encode_document(cv_text, model_name)
    vec_jd = _encode_document(job_description, model_name)
    if vec_cv is None or vec_jd is None:
        return 0.0

    sim = float(np.dot(vec_cv, vec_jd))
    return round(max(0.0, min(1.0, sim)) * 100.0, 2)


def match_cv_to_job(
    cv_text: str,
    job_description: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    use_embeddings: bool = True,
) -> MatchResult:
    """Compare CV text against a job description using skills, TF-IDF, and embeddings."""
    cv_skills = extract_skills(cv_text)
    jd_skills = extract_skills(job_description)

    cv_set = set(cv_skills)
    jd_set = set(jd_skills)

    matched = sorted(cv_set & jd_set)
    missing = sorted(jd_set - cv_set)
    extra = sorted(cv_set - jd_set)

    skill_coverage = round((len(matched) / len(jd_set)) * 100.0, 2) if jd_set else 0.0
    lexical = compute_lexical_similarity(cv_text, job_description)
    semantic = (
        compute_semantic_similarity(cv_text, job_description, model_name=model_name)
        if use_embeddings
        else 0.0
    )

    return MatchResult(
        jd_skills=jd_skills,
        cv_skills=cv_skills,
        matched_skills=matched,
        missing_skills=missing,
        extra_cv_skills=extra,
        skill_coverage_score=skill_coverage,
        lexical_similarity=lexical,
        semantic_similarity=semantic,
        model_name=model_name,
    )


if __name__ == "__main__":
    import sys

    from src.parser import parse_cv_pdf

    if len(sys.argv) < 3:
        print("Usage: python -m src.matcher <path_to_cv.pdf> <path_to_job_description.txt>")
        sys.exit(1)

    parsed_cv = parse_cv_pdf(sys.argv[1])
    jd_text = Path(sys.argv[2]).read_text(encoding="utf-8")
    result = match_cv_to_job(parsed_cv.raw_text, jd_text)

    print("\n=== CV vs JOB MATCH REPORT ===")
    print(f"Skill coverage:      {result.skill_coverage_score:.1f}% ({len(result.matched_skills)}/{len(result.jd_skills)})")
    print(f"Lexical similarity:  {result.lexical_similarity:.1f}% (TF-IDF)")
    print(f"Semantic similarity: {result.semantic_similarity:.1f}% ({result.model_name})")
    print(f"\nMatched skills: {result.matched_skills}")
    print(f"Missing skills: {result.missing_skills}")
    print(f"Extra CV skills: {result.extra_cv_skills}")