"""Unit and integration tests for skill extraction and semantic job matching."""

from src.matcher import extract_skills, match_cv_to_job

BIG_DATA_JD = """
We are looking for a Big Data Developer Intern in Barcelona.
You will develop data pipelines using Apache Spark with Python or Scala,
write SQL queries, create unit tests, and work with Docker on Linux.
Knowledge of Azure or AWS and Git is highly valued.
"""

DATA_TECH_CV = """
Computer Engineering student specializing in Artificial Intelligence and Data Science.
Skills: Python, SQL, C/C++, pandas, NumPy, scikit-learn, Git, Docker, Linux.
Built distributed Python applications, automated testing workflows, and analyzed structured datasets.
"""

UNRELATED_CV = """
Experienced hospitality manager and pastry chef with 6 years leading restaurant kitchens.
Skilled in menu planning, supplier negotiations, customer service, food safety, and team scheduling.
"""


def test_skill_extraction_handles_aliases_and_symbols() -> None:
    sample = "Experienced in C/C++, sklearn, PowerBI, PySpark, R, and GitHub. Not C#."
    skills = extract_skills(sample)

    assert "C" in skills
    assert "C++" in skills
    assert "R" in skills
    assert "scikit-learn" in skills
    assert "Power BI" in skills
    assert "Apache Spark" in skills
    assert "Git" in skills


def test_skill_coverage_identifies_matched_and_missing() -> None:
    result = match_cv_to_job(DATA_TECH_CV, BIG_DATA_JD, use_embeddings=False)

    assert "Python" in result.matched_skills
    assert "SQL" in result.matched_skills
    assert "Docker" in result.matched_skills
    assert "Apache Spark" in result.missing_skills
    assert "Scala" in result.missing_skills
    assert 0.0 < result.skill_coverage_score < 100.0
    assert result.lexical_similarity > 0.0


def test_empty_job_description_does_not_crash() -> None:
    result = match_cv_to_job(DATA_TECH_CV, "", use_embeddings=False)

    assert result.jd_skills == []
    assert result.skill_coverage_score == 0.0
    assert result.lexical_similarity == 0.0


def test_semantic_matcher_ranks_relevant_cv_above_unrelated() -> None:
    tech_match = match_cv_to_job(DATA_TECH_CV, BIG_DATA_JD, use_embeddings=True)
    unrelated_match = match_cv_to_job(UNRELATED_CV, BIG_DATA_JD, use_embeddings=True)

    assert tech_match.semantic_similarity > unrelated_match.semantic_similarity + 15.0
    assert tech_match.skill_coverage_score > unrelated_match.skill_coverage_score