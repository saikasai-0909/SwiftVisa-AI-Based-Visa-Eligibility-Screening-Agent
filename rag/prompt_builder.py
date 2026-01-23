"""
Builds a grounded prompt for answering general and eligibility-based visa questions
using retrieved policy snippets.
MIT-compliant modification.
"""

from typing import Dict, List
from langchain.schema import Document


def build_prompt(user_profile: Dict, retrieved: List[Document]) -> str:
    # -------- User Profile (Optional Context) --------
    if user_profile:
        profile_lines = [f"- {k}: {v}" for k, v in user_profile.items()]
        profile_section = "User Context (if relevant):\n" + "\n".join(profile_lines) + "\n\n"
    else:
        profile_section = ""

    # -------- Retrieved Snippets --------
    snippet_texts = []
    for i, d in enumerate(retrieved):
        md = d.metadata or {}
        src = md.get("source", "unknown")
        country = md.get("country", "unknown")
        vtype = md.get("visa_type", "unknown")

        header = f"[{i+1}] Source: {src} | {country} / {vtype}"
        snippet_texts.append(header + "\n" + d.page_content)

    snippets_section = "\n\n".join(snippet_texts)

    # -------- Instructions (Adaptive) --------
    instructions = (
        "\n\nTask:\n"
        "You are answering a visa-related question using ONLY the provided policy snippets.\n\n"
        "Step 1: Identify the intent of the question:\n"
        "- If the question is INFORMATIONAL (e.g., definition, process, timeline, documents), "
        "provide a clear, concise explanation.\n"
        "- If the question is ELIGIBILITY-BASED, evaluate eligibility as Eligible / Not Eligible / Partially Eligible.\n\n"
        "Step 2: Ground your answer strictly in the provided snippets.\n"
        "- Do NOT use external knowledge.\n"
        "- If the snippets do not contain sufficient information, explicitly say so.\n\n"
        "Step 3: Format the response appropriately:\n\n"
        "For INFORMATIONAL questions:\n"
        "Answer:\n"
        "- <Clear explanation>\n\n"
        "Sources:\n"
        "- [1] ...\n\n"
        "For ELIGIBILITY questions:\n"
        "Eligibility: <Eligible | Not Eligible | Partially Eligible>\n\n"
        "Reasoning:\n"
        "- ...\n\n"
        "Missing Requirements (if any):\n"
        "- ...\n\n"
        "Sources:\n"
        "- [1] ...\n\n"
        "Keep the answer factual, concise, and snippet-grounded."
    )

    # -------- Final Prompt --------
    return (
        "You are a visa policy assistant. Your role is to answer user questions "
        "accurately by grounding responses strictly in retrieved policy snippets.\n\n"
        + profile_section
        + "Policy Snippets:\n"
        + snippets_section
        + instructions
    )
