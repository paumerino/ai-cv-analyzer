"""Structured LLM career advisor grounded in deterministic CV analysis."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.matcher import MatchResult, match_cv_to_job
from src.parser import ParsedCV, parse_cv_pdf
from src.scorer import ScoreReport, compute_score_report

DEFAULT_LLM_MODEL = "gemini-3.8-flash"


class BulletRewrite(BaseModel):
    target_area: str = Field(description="CV section or experience being improved.")
    suggested_bullet: str = Field(description="Concise, action-oriented, truthful CV bullet.")
    rationale: str = Field(description="Why this phrasing strengthens fit for the target role.")


class LLMFeedbackPayload(BaseModel):
    """Schema enforced on the LLM JSON output via Pydantic."""

    executive_summary: str = Field(description="2-3 sentence honest assessment of candidate fit.")
    priority_gaps: list[str] = Field(description="Top 3-4 concrete gaps relative to the job description.")
    bullet_rewrites: list[BulletRewrite] = Field(
        description="2-3 concrete CV bullet rewrites without fabricating experience."
    )
    interview_questions: list[str] = Field(
        description="3 likely technical or behavioral interview questions based on the gaps."
    )
    action_plan: list[str] = Field(
        description="3 practical steps the candidate can complete in 1-2 weeks before applying."
    )


class AdvisorFeedback(LLMFeedbackPayload):
    source: str = "rule_based_fallback"  # "llm" | "rule_based_fallback"
    model_used: str = "deterministic-fallback"
    fallback_reason: str | None = None


def _build_prompt(
    parsed_cv: ParsedCV,
    match_result: MatchResult,
    score_report: ScoreReport,
    job_description: str,
) -> str:
    """Ground the LLM prompt in deterministic metrics to avoid hallucinated scores."""
    breakdown_lines = "\n".join(
        f"- {c.name}: {c.raw_score:.1f}/100 (weight {int(c.weight * 100)}%) -> {c.explanation}"
        for c in score_report.breakdown
    )

    return f"""
You are a senior technical recruiter and AI/Data engineering hiring manager.
Review the candidate's CV against the target job description using the precomputed metrics below.

RULES:
- Do NOT invent skills, projects, or experience the candidate does not have.
- Ground every recommendation in the provided CV text, job description, and metrics.
- Keep language direct, concrete, and concise (English).

PRECOMPUTED ANALYSIS:
- Overall Score: {score_report.overall_score:.1f}/100 ({score_report.fit_label})
- Matched Skills: {match_result.matched_skills}
- Missing Skills: {match_result.missing_skills}
- Breakdown:
{breakdown_lines}

JOB DESCRIPTION:
\"\"\"{job_description[:4000]}\"\"\"

CANDIDATE CV TEXT:
\"\"\"{parsed_cv.raw_text[:5000]}\"\"\"
""".strip()


def generate_rule_based_feedback(
    parsed_cv: ParsedCV,
    match_result: MatchResult,
    score_report: ScoreReport,
    reason: str | None = None,
) -> AdvisorFeedback:
    """Deterministic fallback so the pipeline remains usable without an LLM API key."""
    missing = match_result.missing_skills[:4]
    matched = match_result.matched_skills[:4]

    summary = (
        f"Your CV scores {score_report.overall_score:.1f}/100 ({score_report.fit_label}) for this role. "
        f"You already align on {', '.join(matched) if matched else 'core engineering fundamentals'}, "
        f"while addressing {', '.join(missing) if missing else 'more quantified project results'} "
        "will have the highest impact."
    )

    gaps = [f"Missing or unmentioned job skill: {skill}." for skill in missing]
    if parsed_cv.issues:
        gaps.append(f"ATS issue ({parsed_cv.issues[0].code}): {parsed_cv.issues[0].message}")
    if not gaps:
        gaps.append("Add more quantified metrics (dataset size, latency, accuracy/F1) to project bullets.")

    rewrites = [
        BulletRewrite(
            target_area="Projects",
            suggested_bullet=(
                "Built and tested an end-to-end Python data application with modular code, "
                "automated pytest suites, and reproducible documentation on GitHub."
            ),
            rationale="Highlights software engineering rigor, testing, and reproducibility.",
        ),
        BulletRewrite(
            target_area="Technical Skills & Coursework",
            suggested_bullet=(
                f"Highlighted hands-on coursework and projects using "
                f"{', '.join(matched) if matched else 'Python and SQL'}."
            ),
            rationale="Makes existing relevant keywords easy for ATS and recruiters to verify.",
        ),
    ]

    questions = [
        f"You do not list {missing[0]} yet — how would you ramp up on it during your first month?"
        if missing
        else "Walk me through the architecture and main trade-offs of your strongest technical project.",
        "How do you validate data quality and test your code before deploying or sharing results?",
        "Tell me about a time you had to make an accurate decision under time pressure.",
    ]

    action_plan = [
        f"Build a small hands-on mini-project or tutorial covering {missing[0]} and push it to GitHub."
        if missing
        else "Polish your top GitHub repository README with architecture diagrams and evaluation metrics.",
        "Add 1-2 concrete numbers (runtime, dataset size, test count, or model metrics) to your projects.",
        "Prepare a 2-minute English pitch connecting your university coursework and projects to this role.",
    ]

    return AdvisorFeedback(
        executive_summary=summary,
        priority_gaps=gaps,
        bullet_rewrites=rewrites,
        interview_questions=questions,
        action_plan=action_plan,
        source="rule_based_fallback",
        model_used="deterministic-fallback",
        fallback_reason=reason,
    )


def generate_feedback(
    parsed_cv: ParsedCV,
    match_result: MatchResult,
    score_report: ScoreReport,
    job_description: str,
    api_key: str | None = None,
    model_name: str = DEFAULT_LLM_MODEL,
    use_llm: bool = True,
) -> AdvisorFeedback:
    """Generate structured career recommendations via Gemini API with safe deterministic fallback."""
    load_dotenv()
    resolved_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()

    if not use_llm:
        return generate_rule_based_feedback(
            parsed_cv, match_result, score_report, reason="LLM disabled by configuration."
        )

    if not resolved_key or resolved_key == "your_api_key_here":
        return generate_rule_based_feedback(
            parsed_cv,
            match_result,
            score_report,
            reason="GEMINI_API_KEY not configured; using deterministic fallback.",
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=resolved_key)
        prompt = _build_prompt(parsed_cv, match_result, score_report, job_description)

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LLMFeedbackPayload,
                temperature=0.3,
            ),
        )

        payload = LLMFeedbackPayload.model_validate_json(response.text)
        return AdvisorFeedback(
            **payload.model_dump(),
            source="llm",
            model_used=model_name,
            fallback_reason=None,
        )
    except Exception as exc:
        return generate_rule_based_feedback(
            parsed_cv,
            match_result,
            score_report,
            reason=f"LLM request failed ({type(exc).__name__}); fell back to deterministic rules.",
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.advisor <path_to_cv.pdf> <path_to_job_description.txt>")
        sys.exit(1)

    parsed = parse_cv_pdf(sys.argv[1])
    jd_text = Path(sys.argv[2]).read_text(encoding="utf-8")
    matched = match_cv_to_job(parsed.raw_text, jd_text, use_embeddings=True)
    report = compute_score_report(parsed, matched)
    feedback = generate_feedback(parsed, matched, report, jd_text)

    print(f"\n=== CAREER ADVISOR ({feedback.source} | {feedback.model_used}) ===")
    if feedback.fallback_reason:
        print(f"Note: {feedback.fallback_reason}")
    print(f"\nExecutive Summary:\n{feedback.executive_summary}")

    print("\nPriority Gaps:")
    for gap in feedback.priority_gaps:
        print(f"  - {gap}")

    print("\nSuggested Bullet Rewrites:")
    for rw in feedback.bullet_rewrites:
        print(f"  [{rw.target_area}] {rw.suggested_bullet}")
        print(f"     Why: {rw.rationale}")

    print("\nLikely Interview Questions:")
    for idx, q in enumerate(feedback.interview_questions, start=1):
        print(f"  {idx}. {q}")

    print("\nAction Plan:")
    for idx, step in enumerate(feedback.action_plan, start=1):
        print(f"  {idx}. {step}")