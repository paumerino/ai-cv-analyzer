"""Unit tests for the structured LLM advisor and deterministic fallback."""

from src.advisor import (
    AdvisorFeedback,
    LLMFeedbackPayload,
    _build_prompt,
    generate_feedback,
)
from src.matcher import MatchResult
from src.parser import ContactInfo, ParsedCV
from src.scorer import compute_score_report


def _sample_inputs():
    parsed_cv = ParsedCV(
        raw_text=(
            "Computer Engineering student skilled in Python, SQL, pandas, and Docker. "
            "Built a distributed multi-agent system with 3 autonomous agents."
        ),
        page_count=1,
        word_count=20,
        char_count=140,
        contact=ContactInfo(email="test@example.com"),
        detected_sections=["education", "skills", "experience"],
        missing_sections=[],
        passed_checks=["Selectable text extracted cleanly from PDF."],
        issues=[],
        ats_score=95,
    )
    match_result = MatchResult(
        jd_skills=["Python", "SQL", "Apache Spark", "Scala"],
        cv_skills=["Python", "SQL", "pandas", "Docker"],
        matched_skills=["Python", "SQL"],
        missing_skills=["Apache Spark", "Scala"],
        skill_coverage_score=50.0,
        lexical_similarity=20.0,
        semantic_similarity=60.0,
    )
    score_report = compute_score_report(parsed_cv, match_result)
    jd_text = "Big Data Intern role using Apache Spark, Scala, Python, and SQL."
    return parsed_cv, match_result, score_report, jd_text


def test_fallback_triggers_when_no_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    parsed_cv, match_result, score_report, jd_text = _sample_inputs()

    feedback = generate_feedback(parsed_cv, match_result, score_report, jd_text, api_key="")

    assert isinstance(feedback, AdvisorFeedback)
    assert feedback.source == "rule_based_fallback"
    assert len(feedback.priority_gaps) >= 1
    assert len(feedback.bullet_rewrites) >= 1
    assert len(feedback.interview_questions) == 3
    assert any("Apache Spark" in gap for gap in feedback.priority_gaps)


def test_prompt_grounds_llm_with_precomputed_metrics() -> None:
    parsed_cv, match_result, score_report, jd_text = _sample_inputs()
    prompt = _build_prompt(parsed_cv, match_result, score_report, jd_text)

    assert "Overall Score:" in prompt
    assert "Apache Spark" in prompt
    assert "Do NOT invent skills" in prompt


def test_pydantic_schema_parses_structured_json() -> None:
    sample_json = """
    {
      "executive_summary": "Solid Python and SQL base with clear room to grow in Big Data tools.",
      "priority_gaps": ["No Apache Spark experience listed."],
      "bullet_rewrites": [
        {
          "target_area": "Projects",
          "suggested_bullet": "Engineered a 3-agent Python system with JSON messaging and automated tests.",
          "rationale": "Highlights concrete engineering rigor."
        }
      ],
      "interview_questions": ["How does Spark distribute workloads?", "Explain SQL joins.", "How do you test pipelines?"],
      "action_plan": ["Build a small PySpark notebook.", "Add pytest tests.", "Quantify project metrics."]
    }
    """
    parsed = LLMFeedbackPayload.model_validate_json(sample_json)
    assert len(parsed.bullet_rewrites) == 1
    assert len(parsed.interview_questions) == 3