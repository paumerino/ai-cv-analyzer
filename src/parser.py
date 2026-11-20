"""PDF text extraction and deterministic ATS health checks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import pymupdf
from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    email: str | None = None
    phone: str | None = None
    linkedin: str | None = None
    github: str | None = None


class ATSIssue(BaseModel):
    severity: str  # "error" | "warning"
    code: str
    message: str


class ParsedCV(BaseModel):
    raw_text: str
    page_count: int
    word_count: int
    char_count: int
    contact: ContactInfo
    detected_sections: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    encoding_artifacts: list[str] = Field(default_factory=list)
    multi_column_suspected: bool = False
    passed_checks: list[str] = Field(default_factory=list)
    issues: list[ATSIssue] = Field(default_factory=list)
    ats_score: int = 0


# Standard CV sections (English & Spanish)
SECTION_PATTERNS: dict[str, list[str]] = {
    "profile": [r"\bprofile\b", r"\bsummary\b", r"\babout me\b", r"\bperfil\b", r"\bresumen\b"],
    "education": [r"\beducation\b", r"\bacademic\b", r"\bformaci[oó]n\b", r"\beducaci[oó]n\b"],
    "experience": [
        r"\bwork experience\b",
        r"\bexperience\b",
        r"\bemployment\b",
        r"\bexperiencia\b",
    ],
    "skills": [
        r"\btechnical skills\b",
        r"\bskills\b",
        r"\btechnologies\b",
        r"\bhabilidades\b",
        r"\bcompetencias\b",
    ],
    "projects": [r"\bprojects\b", r"\bportfolio\b", r"\bproyectos\b"],
    "languages": [r"\blanguages\b", r"\bidiomas\b"],
}

ESSENTIAL_SECTIONS = {"education", "skills", "experience"}

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
LINKEDIN_REGEX = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+/?", re.IGNORECASE
)
GITHUB_REGEX = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9_-]+/?", re.IGNORECASE
)
PHONE_CANDIDATE_REGEX = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")

# Characters that often indicate broken PDF font/encoding extraction
SUSPICIOUS_CHARS_REGEX = re.compile(r"[\ufffd\u25a1\uf0b7\uf0a7\u016f]")


def _extract_phone(text: str) -> str | None:
    """Extract a phone number while ignoring date ranges like '2022 - 2026'."""
    for match in PHONE_CANDIDATE_REGEX.finditer(text):
        candidate = match.group(0).strip()
        digits = re.sub(r"\D", "", candidate)
        if 9 <= len(digits) <= 15:
            return candidate
    return None


def _extract_contact_info(text: str) -> ContactInfo:
    email_match = EMAIL_REGEX.search(text)
    linkedin_match = LINKEDIN_REGEX.search(text)
    github_match = GITHUB_REGEX.search(text)

    return ContactInfo(
        email=email_match.group(0) if email_match else None,
        phone=_extract_phone(text),
        linkedin=linkedin_match.group(0) if linkedin_match else None,
        github=github_match.group(0) if github_match else None,
    )


def _detect_sections(text: str) -> tuple[list[str], list[str]]:
    """Detect standard CV headings by scanning short lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header_candidates = [line for line in lines if len(line) <= 45]
    joined_headers = "\n".join(header_candidates).lower()

    detected: list[str] = []
    for section_name, patterns in SECTION_PATTERNS.items():
        if any(re.search(pattern, joined_headers) for pattern in patterns):
            detected.append(section_name)

    missing = sorted(ESSENTIAL_SECTIONS - set(detected))
    return detected, missing


def _detect_multi_column(doc: pymupdf.Document) -> bool:
    """Heuristic for multi-column layouts that ignores short right-aligned dates."""
    for page in doc:
        page_width = page.rect.width
        right_long_blocks = 0
        left_long_blocks = 0

        for block in page.get_text("blocks"):
            x0, _, x1, _, block_text, _, block_type = block
            if block_type != 0:  # 0 = text block
                continue

            cleaned = block_text.strip()
            if len(cleaned) < 60:
                continue

            if x0 > page_width * 0.45:
                right_long_blocks += 1
            elif x1 < page_width * 0.55:
                left_long_blocks += 1

        if left_long_blocks >= 2 and right_long_blocks >= 2:
            return True
    return False


def _evaluate_ats_health(
    raw_text: str,
    page_count: int,
    contact: ContactInfo,
    detected_sections: list[str],
    missing_sections: list[str],
    encoding_artifacts: list[str],
    multi_column_suspected: bool,
) -> tuple[int, list[str], list[ATSIssue]]:
    """Compute a transparent 0-100 ATS score from deterministic rules."""
    score = 100
    passed: list[str] = []
    issues: list[ATSIssue] = []

    words = raw_text.split()
    word_count = len(words)

    # 1. Text extractability (35 pts)
    if word_count < 50:
        score -= 35
        issues.append(
            ATSIssue(
                severity="error",
                code="UNREADABLE_OR_SCANNED_PDF",
                message=(
                    "Very little selectable text detected. The PDF may be scanned "
                    "or image-based, which most ATS parsers cannot read."
                ),
            )
        )
    else:
        passed.append("Selectable text extracted cleanly from PDF.")

    # 2. Page count & length (15 pts)
    if page_count > 2:
        score -= 10
        issues.append(
            ATSIssue(
                severity="warning",
                code="TOO_MANY_PAGES",
                message=f"CV has {page_count} pages. For junior/internship roles, 1 page is ideal (max 2).",
            )
        )
    else:
        passed.append(f"Appropriate page length ({page_count} page{'s' if page_count > 1 else ''}).")

    if 50 <= word_count < 180:
        score -= 5
        issues.append(
            ATSIssue(
                severity="warning",
                code="LOW_WORD_COUNT",
                message=f"Only {word_count} words detected. Consider adding more technical details or projects.",
            )
        )
    elif word_count > 1000:
        score -= 5
        issues.append(
            ATSIssue(
                severity="warning",
                code="HIGH_WORD_COUNT",
                message=f"{word_count} words detected. Consider tightening bullet points for readability.",
            )
        )

    # 3. Contact info (20 pts)
    if contact.email:
        passed.append(f"Email detected ({contact.email}).")
    else:
        score -= 10
        issues.append(
            ATSIssue(
                severity="error",
                code="MISSING_EMAIL",
                message="No email address detected in the extracted text.",
            )
        )

    if contact.phone:
        passed.append("Phone number detected.")
    else:
        score -= 5
        issues.append(
            ATSIssue(
                severity="warning",
                code="MISSING_PHONE",
                message="No phone number detected.",
            )
        )

    if contact.linkedin or contact.github:
        passed.append("Professional profile link detected (LinkedIn/GitHub).")
    else:
        score -= 5
        issues.append(
            ATSIssue(
                severity="warning",
                code="MISSING_TECH_LINKS",
                message="Neither LinkedIn nor GitHub profile URL was detected.",
            )
        )

    # 4. Essential sections (15 pts)
    if not missing_sections:
        passed.append(f"Core sections detected ({', '.join(detected_sections)}).")
    else:
        penalty = min(15, 5 * len(missing_sections))
        score -= penalty
        issues.append(
            ATSIssue(
                severity="warning",
                code="MISSING_SECTIONS",
                message=f"Standard section headings not detected: {', '.join(missing_sections)}.",
            )
        )

    # 5. Encoding & layout health (15 pts)
    if encoding_artifacts:
        score -= 10
        issues.append(
            ATSIssue(
                severity="warning",
                code="ENCODING_ARTIFACTS",
                message=(
                    f"Suspicious extracted characters found ({', '.join(encoding_artifacts)}). "
                    "Check PDF font encoding or replace decorative symbols."
                ),
            )
        )
    else:
        passed.append("No text encoding or glyph extraction artifacts found.")

    if multi_column_suspected:
        score -= 5
        issues.append(
            ATSIssue(
                severity="warning",
                code="MULTI_COLUMN_LAYOUT",
                message=(
                    "Multi-column text layout suspected. Some legacy ATS systems "
                    "scramble reading order across columns."
                ),
            )
        )
    else:
        passed.append("Single-column ATS-friendly layout detected.")

    return max(0, score), passed, issues


def parse_cv_pdf(source: Union[str, Path, bytes]) -> ParsedCV:
    """Parse a CV from a PDF file path or raw bytes and run ATS checks."""
    if isinstance(source, bytes):
        doc = pymupdf.open(stream=source, filetype="pdf")
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        doc = pymupdf.open(path)

    with doc:
        page_count = len(doc)
        pages_text = [page.get_text("text") for page in doc]
        raw_text = "\n".join(pages_text).strip()
        multi_column = _detect_multi_column(doc)

    contact = _extract_contact_info(raw_text)
    detected_sections, missing_sections = _detect_sections(raw_text)
    artifacts = sorted(set(SUSPICIOUS_CHARS_REGEX.findall(raw_text)))

    ats_score, passed_checks, issues = _evaluate_ats_health(
        raw_text=raw_text,
        page_count=page_count,
        contact=contact,
        detected_sections=detected_sections,
        missing_sections=missing_sections,
        encoding_artifacts=artifacts,
        multi_column_suspected=multi_column,
    )

    return ParsedCV(
        raw_text=raw_text,
        page_count=page_count,
        word_count=len(raw_text.split()),
        char_count=len(raw_text),
        contact=contact,
        detected_sections=detected_sections,
        missing_sections=missing_sections,
        encoding_artifacts=artifacts,
        multi_column_suspected=multi_column,
        passed_checks=passed_checks,
        issues=issues,
        ats_score=ats_score,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.parser <path_to_cv.pdf>")
        sys.exit(1)

    result = parse_cv_pdf(sys.argv[1])
    print(f"\n=== ATS REPORT (Score: {result.ats_score}/100) ===")
    print(f"Pages: {result.page_count} | Words: {result.word_count}")
    print(f"Contact: {result.contact.model_dump()}")
    print(f"Detected sections: {result.detected_sections}")

    print("\nPassed checks:")
    for item in result.passed_checks:
        print(f"  [OK] {item}")

    if result.issues:
        print("\nIssues found:")
        for issue in result.issues:
            print(f"  [{issue.severity.upper()}] {issue.code}: {issue.message}")