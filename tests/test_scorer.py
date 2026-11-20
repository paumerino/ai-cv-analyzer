"""Unit tests for explainable scoring and bullet-impact analysis."""

import pytest

from src.matcher import MatchResult
from src.parser import ContactInfo, ParsedCV
from src.scorer import ScoringWeights, analyze_impact, compute_score_report


def _make_parsed_cv(text: str, ats_score: int = 100) -> ParsedCV:
    return ParsedCV(
        raw_text=text,
        page_count=1,
        word_count=len(text.split()),
        char_count=len(text),
        contact=ContactInfo(email="test@example.com", phone="+34 600 000 000"),
        detected_sections=["education", "skills", "experience"],
        missing_sections=[],
        passed_checks=["Selectable text extracted cleanly from PDF."],
        issues=[],
        ats_score=ats_score,
    )


def test_weights_validation_rejects_invalid_sum() -> None:
    with pytest.raises(ValueError):
        ScoringWeights(skill_coverage=0.5, semantic_similarity=0.5, ats_health=0.2, impact_quality=0.1)


def test_impact_analysis_rewards_verbs_and_metrics_and_ignores_years() -> None:
    strong_text = (
        "Designed and implemented an ML pipeline in 2025 - 2026 that analyzed 50k records, "
        "improved F1 score by 18%, automated 3 data pipelines, and reduced latency by 2x. "
        "Built and deployed tested services while collaborating across 4 teams."
    )
    weak_text = (
        "Hardworking team player and self-starter from 2022 to 2026 looking for opportunities."
    )

    strong = analyze_impact(strong_text)
    weak = analyze_impact(weak_text)

    assert "2025" not in strong.quantified_metrics_found
    assert "2026" not in weak.quantified_metrics_found
    assert len(strong.quantified_metrics_found) >= 3
    assert len(weak.cliches_found) >= 2
    assert strong.impact_score > weak.impact_score + 50.0


def test_score_report_breakdown_sums_to_overall_score() -> None:
    cv = _make_parsed_cv(
        "Designed, built, implemented, tested, deployed, analyzed, automated, and optimized "
        "Python models achieving 92% accuracy across 5 projects and 10k rows.",
        ats_score=95,
    )
    match = MatchResult(
        jd_skills=["Python", "SQL", "Docker", "Apache Spark"],
        cv_skills=["Python", "SQL", "Docker"],
        matched_skills=["Python", "SQL", "Docker"],
        missing_skills=["Apache Spark"],
        skill_coverage_score=75.0,
        lexical_similarity=35.0,
        semantic_similarity=72.0,
    )

    report = compute_score_report(cv, match)
    expected_sum = round(sum(c.weighted_points for c in report.breakdown), 2)

    assert report.overall_score == pytest.approx(expected_sum, abs=0.05)
    assert report.fit_label in {"Good Fit", "Strong Fit"}
    assert any("Apache Spark" in tip for tip in report.priority_improvements)