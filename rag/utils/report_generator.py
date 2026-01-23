"""
Compose a formatted report string from RAG result and metadata.
"""
from typing import List
from langchain.schema import Document
from datetime import datetime

def compose_report_text(result_text: str, user_profile: dict, docs: List[Document]) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append(f"SwiftVisa AI — Visa Eligibility Report")
    lines.append(f"Generated: {ts}")
    lines.append("")
    lines.append("===== Applicant Profile =====")
    for k, v in user_profile.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("===== Result =====")
    lines.append(result_text.strip())
    lines.append("")
    lines.append("===== Cited Sources =====")
    for i, d in enumerate(docs):
        md = d.metadata or {}
        src = md.get("source", str(i+1))
        lines.append(f"[{i+1}] {src} | {md.get('country','')} / {md.get('visa_type','')}")
    return "\n".join(lines)
