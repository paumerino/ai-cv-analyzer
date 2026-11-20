"""Interactive Streamlit interface for the Hybrid AI CV Analyzer."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.advisor import DEFAULT_LLM_MODEL, generate_feedback
from src.matcher import match_cv_to_job
from src.parser import parse_cv_pdf
from src.scorer import ScoringWeights, compute_score_report

load_dotenv()

SAMPLE_JD_PATH = Path("data/samples/zurich_bigdata_jd.txt")

st.set_page_config(
    page_title="Hybrid AI CV Analyzer",
    page_icon="📄",
    layout="wide",
)


def _get_default_api_key() -> str:
    """Resolve Gemini API key from environment or Streamlit Cloud secrets."""
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def _render_skill_badges(skills: list[str], bg_color: str, text_color: str) -> None:
    """Render clean pill badges safely using HTML escaping."""
    if not skills:
        st.caption("None detected.")
        return

    pills = "".join(
        f"<span style='display:inline-block; background:{bg_color}; color:{text_color}; "
        f"padding:4px 10px; margin:3px 4px 3px 0; border-radius:999px; font-size:0.85rem; "
        f"font-weight:600;'>{html.escape(skill)}</span>"
        for skill in skills
    )
    st.markdown(f"<div style='line-height:2.1;'>{pills}</div>", unsafe_allow_html=True)


def _build_breakdown_chart(breakdown) -> go.Figure:
    """Horizontal bar chart comparing raw scores (0-100) per scoring dimension."""
    names = [comp.name for comp in breakdown][::-1]
    scores = [comp.raw_score for comp in breakdown][::-1]
    weights = [int(comp.weight * 100) for comp in breakdown][::-1]

    colors = [
        "#1f9d55" if s >= 75 else "#d97706" if s >= 50 else "#dc2626"
        for s in scores
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=[f"{n} ({w}%)" for n, w in zip(names, weights)],
            orientation="h",
            marker_color=colors,
            text=[f"{s:.1f} / 100" for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Explainable Score Breakdown (Raw Scores & Weights)",
        xaxis=dict(range=[0, 112], title="Score (0–100)"),
        yaxis=dict(title=""),
        height=300,
        margin=dict(l=10, r=30, t=45, b=30),
    )
    return fig


def _build_normalized_weights(w_skill: int, w_sem: int, w_ats: int, w_imp: int) -> ScoringWeights:
    """Normalize user slider values so weights always sum strictly to 1.0."""
    total = w_skill + w_sem + w_ats + w_imp
    if total <= 0:
        return ScoringWeights()

    s = round(w_skill / total, 4)
    sem = round(w_sem / total, 4)
    ats = round(w_ats / total, 4)
    imp = round(1.0 - (s + sem + ats), 4)
    return ScoringWeights(
        skill_coverage=s,
        semantic_similarity=sem,
        ats_health=ats,
        impact_quality=imp,
    )


# ---------- SIDEBAR CONFIGURATION ----------
with st.sidebar:
    st.header("Configuration")

    use_llm = st.toggle(
        "Enable AI Career Advisor (Gemini)",
        value=True,
        help="Uses Gemini with Pydantic JSON schema. Falls back to deterministic rules if no API key is set.",
    )

    api_key_input = st.text_input(
        "Gemini API Key (optional override)",
        value="",
        type="password",
        help="Leave blank to use GEMINI_API_KEY from your local .env or Streamlit secrets.",
    )

    with st.expander("Customize Scoring Weights", expanded=False):
        st.caption("Sliders are automatically normalized to 100%.")
        w_skill = st.slider("Skill Coverage", 0, 100, 35)
        w_sem = st.slider("Semantic Alignment", 0, 100, 30)
        w_ats = st.slider("ATS Health", 0, 100, 20)
        w_imp = st.slider("Impact & Writing", 0, 100, 15)

    weights = _build_normalized_weights(w_skill, w_sem, w_ats, w_imp)

    st.divider()
    st.markdown(
        "**Architecture**\n"
        "- **Parser:** PyMuPDF deterministic ATS audit\n"
        "- **Matcher:** Taxonomy + TF-IDF + `all-MiniLM-L6-v2`\n"
        "- **Scorer:** Weighted & explainable formula\n"
        f"- **Advisor:** `{DEFAULT_LLM_MODEL}` + Pydantic JSON"
    )


# ---------- MAIN HEADER ----------
st.title("Hybrid AI CV Analyzer")
st.caption(
    "Deterministic ATS auditing, semantic job matching with sentence embeddings, "
    "explainable scoring, and structured LLM recommendations."
)

if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""

col_cv, col_jd = st.columns([1, 1.3], gap="large")

with col_cv:
    st.subheader("1. Upload Resume (PDF)")
    uploaded_pdf = st.file_uploader(
        "Upload your CV in PDF format",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded_pdf:
        st.success(f"Loaded: **{uploaded_pdf.name}** ({len(uploaded_pdf.getvalue()) / 1024:.1f} KB)")

with col_jd:
    header_col, btn_col = st.columns([2.2, 1])
    with header_col:
        st.subheader("2. Target Job Description")
    with btn_col:
        if SAMPLE_JD_PATH.exists() and st.button("Load Zurich sample", use_container_width=True):
            st.session_state.jd_text = SAMPLE_JD_PATH.read_text(encoding="utf-8")

    jd_text = st.text_area(
        "Paste the target job description here",
        key="jd_text",
        height=210,
        placeholder="Paste the job responsibilities and requirements here...",
        label_visibility="collapsed",
    )

analyze_clicked = st.button(
    "Analyze CV Compatibility",
    type="primary",
    use_container_width=True,
    disabled=not (uploaded_pdf and jd_text.strip()),
)

if not (uploaded_pdf and jd_text.strip()):
    st.info("Upload a PDF resume and paste a job description (or click **Load Zurich sample**) to run the analysis.")

if analyze_clicked and uploaded_pdf and jd_text.strip():
    resolved_key = api_key_input.strip() or _get_default_api_key()

    with st.spinner("Running PDF parser, semantic embeddings, and explainable scorer..."):
        parsed_cv = parse_cv_pdf(uploaded_pdf.getvalue())
        match_result = match_cv_to_job(parsed_cv.raw_text, jd_text, use_embeddings=True)
        score_report = compute_score_report(parsed_cv, match_result, weights=weights)

    with st.spinner("Generating structured recommendations..."):
        feedback = generate_feedback(
            parsed_cv=parsed_cv,
            match_result=match_result,
            score_report=score_report,
            job_description=jd_text,
            api_key=resolved_key,
            use_llm=use_llm,
        )

    st.session_state["last_analysis"] = {
        "parsed_cv": parsed_cv,
        "match_result": match_result,
        "score_report": score_report,
        "feedback": feedback,
    }


# ---------- RESULTS DASHBOARD ----------
if "last_analysis" in st.session_state:
    data = st.session_state["last_analysis"]
    parsed_cv = data["parsed_cv"]
    match_result = data["match_result"]
    score_report = data["score_report"]
    feedback = data["feedback"]

    st.divider()

    # Top KPI cards
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Overall Score", f"{score_report.overall_score:.1f}/100", score_report.fit_label)
    k2.metric(
        "Skill Coverage",
        f"{match_result.skill_coverage_score:.1f}%",
        f"{len(match_result.matched_skills)}/{len(match_result.jd_skills)} skills",
    )
    k3.metric(
        "Semantic Match",
        f"{match_result.semantic_similarity:.1f}%",
        f"TF-IDF: {match_result.lexical_similarity:.1f}%",
    )
    k4.metric("ATS Health", f"{parsed_cv.ats_score}/100", f"{len(parsed_cv.issues)} issue(s)")
    k5.metric(
        "Impact & Writing",
        f"{score_report.impact.impact_score:.1f}/100",
        f"{len(score_report.impact.action_verbs_found)} verbs",
    )

    tab_overview, tab_ats, tab_advisor, tab_export = st.tabs(
        ["Overview & Skills", "ATS & Impact Audit", "AI Career Advisor", "Export JSON"]
    )

    with tab_overview:
        st.plotly_chart(_build_breakdown_chart(score_report.breakdown), use_container_width=True)

        c_match, c_miss, c_extra = st.columns(3)
        with c_match:
            st.markdown("#### Matched Skills")
            _render_skill_badges(match_result.matched_skills, "#dcfce7", "#166534")
        with c_miss:
            st.markdown("#### Missing Job Skills")
            _render_skill_badges(match_result.missing_skills, "#fee2e2", "#991b1b")
        with c_extra:
            st.markdown("#### Additional CV Skills")
            _render_skill_badges(match_result.extra_cv_skills, "#e0f2fe", "#075985")

        st.divider()
        s_col, i_col = st.columns(2)
        with s_col:
            st.markdown("#### Key Strengths")
            for item in score_report.top_strengths:
                st.success(item)
        with i_col:
            st.markdown("#### Priority Improvements")
            for item in score_report.priority_improvements:
                st.warning(item)

    with tab_ats:
        ats_left, ats_right = st.columns(2)
        with ats_left:
            st.markdown("#### ATS Health Checks")
            st.caption(f"Pages: {parsed_cv.page_count} | Words: {parsed_cv.word_count}")
            for ok in parsed_cv.passed_checks:
                st.markdown(f"- :green[**PASS**] {ok}")
            for issue in parsed_cv.issues:
                badge = ":red[**ERROR**]" if issue.severity == "error" else ":orange[**WARNING**]"
                st.markdown(f"- {badge} `{issue.code}`: {issue.message}")

            st.markdown("#### Extracted Contact & Sections")
            st.json(
                {
                    "contact": parsed_cv.contact.model_dump(),
                    "detected_sections": parsed_cv.detected_sections,
                    "missing_sections": parsed_cv.missing_sections,
                }
            )

        with ats_right:
            st.markdown("#### Bullet Impact Analysis")
            st.markdown("**Action Verbs Found:**")
            _render_skill_badges(score_report.impact.action_verbs_found, "#ede9fe", "#5b21b6")

            st.markdown("**Quantified Evidence Found:**")
            _render_skill_badges(score_report.impact.quantified_metrics_found, "#fef3c7", "#92400e")

            if score_report.impact.cliches_found:
                st.markdown("**Weak Clichés to Remove:**")
                _render_skill_badges(score_report.impact.cliches_found, "#fee2e2", "#991b1b")
            else:
                st.success