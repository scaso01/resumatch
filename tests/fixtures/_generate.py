"""Regenerate the test fixtures in this directory.

Run from the repo root:  python tests/fixtures/_generate.py

The resume below is entirely fabricated -- invented person, employers, and
figures. Nothing here belongs to a real candidate. Keep it that way: these
files ship in a public repository.

The PDF is written by hand rather than with a PDF library, so generating
fixtures adds no dependency to the project.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).parent

RESUME_LINES = [
    "JORDAN AVERY",
    "jordan.avery@example.com | (555) 010-0142 | Austin, TX",
    "linkedin.com/in/jordanavery | github.com/jordanavery",
    "",
    "SUMMARY",
    "Backend engineer with 6 years building payment and data platforms.",
    "",
    "EXPERIENCE",
    "",
    "Senior Backend Engineer, Northwind Systems (2022-2026)",
    "- Led migration of a monolith into 12 services, cutting deploy time 74%",
    "  from 45 minutes to 12 minutes across 8 teams.",
    "- Reduced p99 checkout latency from 820ms to 190ms by rewriting the",
    "  settlement path and adding a write-through cache.",
    "- Designed an idempotency layer that eliminated 99.8% of duplicate charges,",
    "  saving an estimated $310,000 in annual chargeback losses.",
    "- Mentored 4 engineers; 2 were promoted within 18 months.",
    "",
    "Backend Engineer, Cobalt Analytics (2019-2022)",
    "- Built an ETL pipeline processing 40 million events per day on 3 nodes.",
    "- Automated the regression suite, raising coverage from 31% to 88%.",
    "- Negotiated a vendor contract that saved $120,000 annually.",
    "- Migrated 14 legacy cron jobs to an orchestrated DAG, removing 22 hours",
    "  of manual operations work per month.",
    "",
    "EDUCATION",
    "B.S. Computer Science, State University, 2019",
    "",
    "SKILLS",
    "Python, Go, PostgreSQL, Redis, Kafka, Docker, Kubernetes, Terraform,",
    "AWS, FastAPI, pytest, SQL, Git, Linux, CI/CD, distributed systems",
]

JD_TEXT = """Senior Backend Engineer

We are looking for a backend engineer to own our payments platform.

Responsibilities:
- Design and operate distributed services in Python or Go
- Improve reliability and latency of high-throughput transaction paths
- Build data pipelines processing millions of events daily
- Mentor engineers and raise the engineering bar

Requirements:
- 5+ years of backend engineering experience
- Strong Python, PostgreSQL, and Kafka experience
- Experience with Docker, Kubernetes, and AWS
- Track record of measurable reliability or performance improvements
"""


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(lines: list[str]) -> bytes:
    """A minimal single-page PDF with one text line per entry."""
    content = ["BT", "/F1 10 Tf", "72 720 Td", "13 TL"]
    for line in lines:
        content.append(f"({_escape(line)}) Tj" if line else "() Tj")
        content.append("T*")
    content.append("ET")
    stream = "\n".join(content).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)


def build_docx(lines: list[str], path: Path) -> None:
    from docx import Document

    document = Document()
    for line in lines:
        document.add_paragraph(line)
    document.save(str(path))


def main() -> None:
    (FIXTURES / "sample_resume.pdf").write_bytes(build_pdf(RESUME_LINES))
    build_docx(RESUME_LINES, FIXTURES / "sample_resume.docx")
    (FIXTURES / "sample_jd.txt").write_text(JD_TEXT, encoding="utf-8")
    (FIXTURES / "sample_resume.txt").write_text(
        "\n".join(RESUME_LINES) + "\n", encoding="utf-8"
    )
    print(f"wrote fixtures to {FIXTURES}")


if __name__ == "__main__":
    main()
