"""Reproducible end-to-end ranking benchmark for the Hybrid AI CV Analyzer."""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf

from src.matcher import match_cv_to_job
from src.parser import parse_cv_pdf
from src.scorer import compute_score_report

JOB_DESCRIPTIONS: dict[str, str] = {
    "Big Data Intern": (
        "Big Data Developer Intern. Build data pipelines using Apache Spark with Python or Scala, "
        "write SQL queries, implement unit tests with pytest, and deploy containerized jobs with "
        "Docker on Linux. Familiarity with AWS or Azure and Git is desirable."
    ),
    "Data Science & ML Intern": (
        "Data Science & Pricing Analytics Intern. Develop predictive Machine Learning models using "
        "Python, pandas, NumPy, scikit-learn, and SQL. Perform exploratory data analysis, visualize "
        "metrics with Matplotlib or Power BI, and validate statistical models."
    ),
    "Applied AI / LLM Intern": (
        "Applied AI Engineering Intern. Build Generative AI and LLM applications in Python using "
        "NLP, RAG, REST APIs, and automated pytest evaluation suites. Work with Docker, Git, "
        "and cloud platforms (AWS or Azure) in an agile environment."
    ),
}

SYNTHETIC_CVS: dict[str, str] = {
    "Strong Tech": """
Alex Rivera
alex.rivera@example.com | +34 611 222 333 | linkedin.com/in/alexrivera | github.com/alexrivera

PROFILE
Computer Engineering student specializing in Data Engineering, Machine Learning, and Applied AI.

EDUCATION
B.Sc. in Computer Engineering (2023 - 2027) - Technical University

TECHNICAL SKILLS
Python, SQL, Scala, pandas, NumPy, scikit-learn, Matplotlib, Machine Learning, NLP, LLMs, RAG,
Apache Spark, Big Data, Power BI, REST APIs, Unit Testing, pytest, Docker, Git, Linux, AWS, Azure, Agile

PROJECTS
End-to-End Analytics & LLM Platform
Built and deployed 3 data pipelines in Apache Spark and Python, analyzed 50k records with pandas and
scikit-learn achieving 91% accuracy, exposed FastAPI REST APIs in Docker, and automated 25+ pytest suites.

WORK EXPERIENCE
Data Engineering Intern - CloudCorp (Summer 2026)
Engineered SQL transformations, validated statistical ML models, and reduced batch latency by 35% on AWS.
""",
    "Partial Junior": """
Sam Morales
sam.morales@example.com | +34 622 333 444 | linkedin.com/in/sammorales | github.com/sammorales

PROFILE
Computer Engineering student interested in AI, Machine Learning, and software development.

EDUCATION
B.Sc. in Computer Engineering (2023 - 2027) - Public University

TECHNICAL SKILLS
Python, SQL, Java, C++, pandas, NumPy, scikit-learn, Matplotlib, Git, Docker, Linux

PROJECTS
Multi-Agent Simulation
Designed and implemented 3 autonomous Python agents communicating over JSON to explore maps and
collect resources. Documented setup instructions and search strategies.

WORK EXPERIENCE
Operations Support Operator (Summer 2026)
Managed incoming requests, verified critical data accurately, and coordinated responses under pressure.
""",
    "Unrelated Profile": """
Jordan Blake
jordan.blake@example.com | +34 633 444 555

PROFILE
Hospitality supervisor and pastry chef with 6 years managing restaurant kitchens and bakery operations.

EDUCATION
Diploma in Culinary Arts and Hospitality Management (2020)

SKILLS
Menu planning, food safety, inventory control, supplier negotiations, customer service, team scheduling

WORK EXPERIENCE
Head Pastry Chef - Bistro Central (2022 - 2026)
Led daily kitchen preparation, managed supplier orders, trained junior staff, and maintained hygiene standards.
Supervised seasonal events and coordinated dining room operations for 120 guests daily.
""",
}


@dataclass
class BenchmarkRow:
    role: str
    profile: str
    overall_score: float
    skill_coverage: float
    semantic_similarity: float
    lexical_similarity: float
    ats_score: int


def _text_to_pdf_bytes(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    rc = page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), text.strip(), fontsize=9)
    if rc < 0:
        raise ValueError("Synthetic benchmark CV overflowed page bounds.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def run_benchmark() -> tuple[list[BenchmarkRow], int, int]:
    """Run the 3x3 CV-to-Job benchmark and verify monotonic ranking per role."""
    parsed_cache = {
        name: parse_cv_pdf(_text_to_pdf_bytes(text)) for name, text in SYNTHETIC_CVS.items()
    }

    rows: list[BenchmarkRow] = []
    roles_correctly_ranked = 0

    for role, jd_text in JOB_DESCRIPTIONS.items():
        role_scores: dict[str, float] = {}
        for profile_name, parsed_cv in parsed_cache.items():
            match = match_cv_to_job(parsed_cv.raw_text, jd_text, use_embeddings=True)
            report = compute_score_report(parsed_cv, match)
            role_scores[profile_name] = report.overall_score
            rows.append(
                BenchmarkRow(
                    role=role,
                    profile=profile_name,
                    overall_score=report.overall_score,
                    skill_coverage=match.skill_coverage_score,
                    semantic_similarity=match.semantic_similarity,
                    lexical_similarity=match.lexical_similarity,
                    ats_score=parsed_cv.ats_score,
                )
            )

        if (
            role_scores["Strong Tech"]
            > role_scores["Partial Junior"]
            > role_scores["Unrelated Profile"]
        ):
            roles_correctly_ranked += 1

    return rows, roles_correctly_ranked, len(JOB_DESCRIPTIONS)


def format_markdown_table(rows: list[BenchmarkRow]) -> str:
    header = (
        "| Target Role | Candidate Profile | Overall | Skill Cov. | Semantic | TF-IDF | ATS |\n"
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
    )
    body = "\n".join(
        f"| {r.role} | {r.profile} | **{r.overall_score:.1f}** | "
        f"{r.skill_coverage:.1f}% | {r.semantic_similarity:.1f}% | "
        f"{r.lexical_similarity:.1f}% | {r.ats_score} |"
        for r in rows
    )
    return f"{header}\n{body}"


if __name__ == "__main__":
    rows, passed, total = run_benchmark()
    print(f"\n=== REPRODUCIBLE RANKING BENCHMARK ({passed}/{total} roles ordered correctly) ===\n")
    print(format_markdown_table(rows))