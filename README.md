# Hybrid AI CV Analyzer

An end-to-end **AI & NLP resume analysis system** that combines deterministic ATS auditing, canonical skill matching, semantic document embeddings, explainable weighted scoring, and structured LLM career recommendations.

Built to avoid the limitations of pure "LLM wrapper" analyzers: **all scores and ATS checks are reproducible and computed deterministically in Python**, while the LLM is strictly grounded in those precomputed metrics via a Pydantic JSON schema.

## Architecture

```text
PDF Resume ──► [1. Parser (PyMuPDF)] ──► Text + Deterministic ATS Audit (0–100)
                                                    │
Job Description ──► [2. Matcher] ◄──────────────────┘
                     ├─ Canonical Skill Taxonomy (aliases + boundary-aware regex)
                     ├─ TF-IDF Cosine Baseline (scikit-learn)
                     └─ Chunked Semantic Embeddings (all-MiniLM-L6-v2)
                                                    │
                                                    ▼
                    [3. Explainable Scorer] ──► Weighted Score (35/30/20/15) + Impact Analysis
                                                    │
                                                    ▼
                    [4. AI Advisor] ──────────► Gemini + Pydantic JSON (with deterministic fallback)
                                                    │
                                                    ▼
                    [5. Streamlit Dashboard] ─► Interactive KPIs, Plotly charts & JSON export
```

## Key Features

- **Deterministic ATS Auditor (`src/parser.py`):** Detects scanned/unreadable PDFs, page length, missing contact details or sections, font encoding artifacts, and suspected multi-column layouts.
- **Hybrid Job Matcher (`src/matcher.py`):** Separates lexical keyword overlap (**TF-IDF**) from semantic alignment (**chunked mean-pooled sentence embeddings** to prevent 256-token truncation) and extracts canonical technical skills.
- **Explainable Scoring (`src/scorer.py`):** Computes a transparent weighted score (`35%` Skill Coverage, `30%` Semantic Alignment, `20%` ATS Health, `15%` Action Verbs & Quantified Impact) with customizable weights in the UI.
- **Grounded LLM Feedback (`src/advisor.py`):** Uses `google-genai` with strict **Pydantic JSON validation** to generate executive summaries, non-fabricated bullet rewrites, likely interview questions, and a 1–2 week action plan—falling back gracefully to deterministic rules when no API key is configured.
- **Reproducible Evaluation (`evaluation/benchmark.py`):** Automated 3×3 ranking benchmark verifying monotonic candidate ordering (`Strong Tech > Partial Junior > Unrelated Profile`) across Big Data, Data Science, and Applied AI roles.

## Screenshots

![Score Breakdown & Skills](docs/skills_breakdown.png)
![Key Strengths & Improvements](docs/strengths_improvements.png)

## Evaluation Benchmark

Run `python -m evaluation.benchmark` to reproduce the 3×3 end-to-end evaluation (**3/3 roles ordered monotonically**, validated automatically in `pytest`):

| Target Role | Candidate Profile | Overall | Skill Cov. | Semantic | TF-IDF | ATS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Big Data Intern | Strong Tech | **87.9** | 100.0% | 72.2% | 15.6% | 95 |
| Big Data Intern | Partial Junior | **51.4** | 45.5% | 41.3% | 4.5% | 95 |
| Big Data Intern | Unrelated Profile | **29.9** | 0.0% | 13.5% | 0.0% | 90 |
| Data Science & ML Intern | Strong Tech | **78.6** | 88.9% | 53.9% | 16.4% | 95 |
| Data Science & ML Intern | Partial Junior | **61.2** | 77.8% | 36.3% | 8.3% | 95 |
| Data Science & ML Intern | Unrelated Profile | **29.9** | 0.0% | 13.5% | 0.0% | 90 |
| Applied AI / LLM Intern | Strong Tech | **84.7** | 100.0% | 61.4% | 16.8% | 95 |
| Applied AI / LLM Intern | Partial Junior | **53.4** | 36.4% | 58.4% | 4.6% | 95 |
| Applied AI / LLM Intern | Unrelated Profile | **30.0** | 0.0% | 13.8% | 0.5% | 90 |

## Quickstart

### Local Setup

```bash
git clone [https://github.com/paumerino/ai-cv-analyzer.git](https://github.com/paumerino/ai-cv-analyzer.git)
cd ai-cv-analyzer
python -m venv .venv

# Windows (CMD)
.\.venv\Scripts\activate.bat
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

*(Optional)* Enable Gemini-powered recommendations by creating a `.env` file based on `.env.example`:

```env
GEMINI_API_KEY=your_api_key_here
```

Launch the web interface or run the test suite (`15` automated tests):

```bash
streamlit run app.py
python -m pytest -v
```

### Run with Docker

```bash
docker build -t ai-cv-analyzer .
docker run -p 8501:8501 --env-file .env ai-cv-analyzer
```

## Tech Stack

`Python` · `PyMuPDF` · `scikit-learn` · `sentence-transformers` · `NumPy` · `Pydantic` · `Google GenAI` · `Streamlit` · `Plotly` · `pytest` · `Docker`