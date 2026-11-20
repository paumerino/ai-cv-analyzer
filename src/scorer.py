"""Explainable weighted CV scoring and bullet-impact analysis."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from src.matcher import MatchResult, match_cv_to_job
from src.parser import ParsedCV, parse_cv_pdf

ACTION_VERBS: set[str] = {
    "achieved", "analyzed", "analysed", "architected", "automated", "built",
    "cleaned", "collaborated", "constructed", "coordinated", "created",
    "delivered", "deployed", "designed", "developed", "engineered", "enhanced",
    "evaluated", "executed", "handled", "implemented", "improved", "integrated",
    "launched", "led", "maintained", "managed", "modeled", "monitored",
    "optimized", "optimised", "organized", "organised", "planned", "processed",
    "produced", "reduced", "refactored", "researched", "resolved", "scaled",
    "streamlined", "supervised", "tested", "trained", "validated", "wrote",
}

WEAK_CLICHES: list[str] = [
    "hardworking",
    "hard-working",
    "team player",
    "self-starter",
    "go-getter",
    "detail-oriented",
    "results-driven",
    "think outside the box",
    "synergy",
    "passionate about everything",
]

# Matches meaningful metrics (%, $, €, +, x, k/M, or numeric counts) while ignoring years like 2023
QUANTIFIED_METRIC_REGEX = re.compile(
    r"(?:"
    r"\b\d+(?:\.\d+)?\s?%"
    r"|\b\d+(?:\.\d+)?[kKmM]\+?"
    r"|\b\d+\s?x\b"
    r"|[€$£]\s?\d+(?:[.,]\d+)?"
    r"|\b\d+\+\s+(?:months?|years?|projects?|users?|clients?|hours?|agents?|models?|tests?)"
    r"|\b(?:three|four|five|six|seven|eight|nine|ten|\d{1,3})\s+"
    r"(?:autonomous\s+agents?|agents?|models?|algorithms?|pipelines?|dashboards?|services?|projects?|teams?|members?|countries|languages|datasets?|users?|clients?|months?|years?)"
    r")",
    re.IGNORECASE,
)


class ScoringWeights(BaseModel):
    skill_coverage: float = 0.35
    semantic_similarity: float = 0.30
    ats_health: float = 0.20
    impact_quality: float = 0.15

    @model_validator(mode="after")
    def validate_sum(self) -> "ScoringWeights":
        total = (
            self.skill_coverage
            + self.semantic_similarity
            + self.ats_health
            + self.impact_quality
        )
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"Scoring weights must sum to 1.0 (got {total:.4f}).")
        return self


class ImpactAnalysis(BaseModel):
    action_verbs_found: list[str] = Field(default_factory=list)
    quantified_metrics_found: list[str] = Field(default_factory=list)
    cliches_found: list[str] = Field(default_factory=list)
    impact_score: float = 0.0


class ScoreComponent(BaseModel):
    name: str
    raw_score: float
    weight: float
    weighted_points: float
    explanation: str


class ScoreReport(BaseModel):
    overall_score: float
    fit_label: str
    breakdown: list[ScoreComponent]
    impact: ImpactAnalysis
    top_strengths: list[str] = Field(default_factory=list)
    priority_improvements: list[str] = Field(default_factory=list)


def analyze_impact(cv_text: str) -> ImpactAnalysis:
    """Evaluate action verbs, quantified evidence, and weak clichés deterministically."""
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", cv_text.lower()))
    verbs_found = sorted(tokens & ACTION_VERBS)

    metrics_found = sorted(
        {m.group(0).strip() for m in QUANTIFIED_METRIC_REGEX.finditer(cv_text)}
    )

    lower_text = cv_text.lower()
    cliches_found = [c for c in WEAK_CLICHES if re.search(rf"\b{re.escape(c)}\b", lower_text)]

    # Action verbs: up to 55 pts (saturates around 8 distinct strong verbs)
    verb_points = min(55.0, (len(verbs_found) / 8.0) * 55.0)
    # Quantified evidence: up to 45 pts (saturates around 4 concrete metrics)
    metric_points = min(45.0, (len(metrics_found) / 4.0) * 45.0)
    # Cliché penalty: -8 pts each (max -24)
    cliche_penalty = min(24.0, len(cliches_found) * 8.0)

    impact_score = round(max(0.0, min(100.0, verb_points + metric_points - cliche_penalty)), 2)

    return ImpactAnalysis(
        action_verbs_found=verbs_found,
        quantified_metrics_found=metrics_found,
        cliches_found=cliches_found,
        impact_score=impact_score,
    )


def _classify_fit(overall_score: float) -> str:
    if overall_score >= 75.0:
        return "Strong Fit"
    if overall_score >= 60.0:
        return "Good Fit"
    if overall_score >= 45.0:
        return "Partial Fit"
    return "Low Fit"


def compute_score_report(
    parsed_cv: ParsedCV,
    match_result: MatchResult,
    weights: ScoringWeights | None = None,
) -> ScoreReport:
    """Combine ATS health, skill coverage, semantic match, and impact into an explainable score."""
    w = weights or ScoringWeights()
    impact = analyze_impact(parsed_cv.raw_text)

    breakdown = [
        ScoreComponent(
            name="Skill Coverage",
            raw_score=round(match_result.skill_coverage_score, 2),
            weight=w.skill_coverage,
            weighted_points=round(match_result.skill_coverage_score * w.skill_coverage, 2),
            explanation=(
                f"Matched {len(match_result.matched_skills)}/{len(match_result.jd_skills)} "
                "required technical skills from the job description."
            ),
        ),
        ScoreComponent(
            name="Semantic Alignment",
            raw_score=round(match_result.semantic_similarity, 2),
            weight=w.semantic_similarity,
            weighted_points=round(match_result.semantic_similarity * w.semantic_similarity, 2),
            explanation=(
                f"Document embedding similarity ({match_result.model_name}); "
                f"TF-IDF lexical baseline was {match_result.lexical_similarity:.1f}%."
            ),
        ),
        ScoreComponent(
            name="ATS Health",
            raw_score=float(parsed_cv.ats_score),
            weight=w.ats_health,
            weighted_points=round(parsed_cv.ats_score * w.ats_health, 2),
            explanation=(
                f"{len(parsed_cv.passed_checks)} checks passed, "
                f"{len(parsed_cv.issues)} issue(s) detected."
            ),
        ),
        ScoreComponent(
            name="Impact & Writing",
            raw_score=impact.impact_score,
            weight=w.impact_quality,
            weighted_points=round(impact.impact_score * w.impact_quality, 2),
            explanation=(
                f"{len(impact.action_verbs_found)} action verbs, "
                f"{len(impact.quantified_metrics_found)} quantified metric(s), "
                f"{len(impact.cliches_found)} cliché(s)."
            ),
        ),
    ]

    overall = round(sum(comp.weighted_points for comp in breakdown), 2)
    fit_label = _classify_fit(overall)

    strengths: list[str] = []
    improvements: list[str] = []

    if parsed_cv.ats_score >= 90:
        strengths.append(f"Clean ATS structure ({parsed_cv.ats_score}/100).")
    elif parsed_cv.issues:
        improvements.append(f"Fix {len(parsed_cv.issues)} ATS issue(s): {parsed_cv.issues[0].message}")

    if match_result.matched_skills:
        strengths.append(f"Matches key skills: {', '.join(match_result.matched_skills[:6])}.")

    if match_result.missing_skills:
        improvements.append(
            f"Address missing job skills (projects/coursework): {', '.join(match_result.missing_skills[:5])}."
        )

    if len(impact.action_verbs_found) >= 6:
        strengths.append(f"Strong action-oriented phrasing ({len(impact.action_verbs_found)} action verbs).")
    else:
        improvements.append("Start experience and project bullets with stronger action verbs (e.g., Built, Engineered, Analyzed).")

    if len(impact.quantified_metrics_found) < 2:
        improvements.append(
            "Add concrete numbers to your bullets (e.g., dataset size, accuracy/F1 score, speedup, or team/user count)."
        )
    else:
        strengths.append(
            f"Includes quantified evidence ({', '.join(impact.quantified_metrics_found[:3])})."
        )

    if impact.cliches_found:
        improvements.append(f"Remove generic buzzwords: {', '.join(impact.cliches_found)}.")

    return ScoreReport(
        overall_score=overall,
        fit_label=fit_label,
        breakdown=breakdown,
        impact=impact,
        top_strengths=strengths,
        priority_improvements=improvements,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.scorer <path_to_cv.pdf> <path_to_job_description.txt>")
        sys.exit(1)

    parsed = parse_cv_pdf(sys.argv[1])
    jd_text = Path(sys.argv[2]).read_text(encoding="utf-8")
    matched = match_cv_to_job(parsed.raw_text, jd_text, use_embeddings=True)
    report = compute_score_report(parsed, matched)

    print(f"\n=== OVERALL CV SCORE: {report.overall_score:.1f}/100 ({report.fit_label}) ===")
    for comp in report.breakdown:
        print(
            f"  - {comp.name:<18} | Raw: {comp.raw_score:5.1f} | "
            f"Weight: {int(comp.weight * 100):2d}% | Points: {comp.weighted_points:5.2f} "
            f"({comp.explanation})"
        )

    print("\nStrengths:")
    for item in report.top_strengths:
        print(f"  [+] {item}")

    print("\nPriority improvements:")
    for item in report.priority_improvements:
        print(f"  [!] {item}")