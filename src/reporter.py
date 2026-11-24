"""Report generation module producing explainable JSON and Markdown audit reports."""

import json
from pathlib import Path
from typing import Any

from src.models import DocumentResult


def format_risk_badge(risk: str) -> str:
    """Returns a clean plain-text label representing the integrity risk level."""
    if risk == "HIGH":
        return "HIGH"
    elif risk == "MEDIUM":
        return "MEDIUM"
    elif risk == "LOW":
        return "LOW"
    else:
        return "UNPROCESSED"


def generate_document_json(report_data: dict[str, Any], output_path: Path) -> None:
    """Writes the standardized JSON report for a single document."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)


def generate_document_markdown(report_data: dict[str, Any], output_path: Path) -> None:
    """Generates an audit-ready Markdown report designed for rapid (<30s) verification."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc_id = report_data["document_id"]
    status = report_data["status"]
    risk = report_data["total_integrity_risk"]
    scores = report_data.get("scores", {})
    peer_pct = scores.get("peer_similarity_percentage", 0.0)
    ext_pct = scores.get("external_similarity_percentage", 0.0)
    ai_prob = scores.get("ai_generated_probability", 0.0)
    discounted = report_data.get("boilerplate_chunks_discounted", 0)
    explanations = report_data.get("explanations", [])

    lines = [
        f"# Integrity Audit Report: `{doc_id}`",
        "",
        f"**Overall Status**: {status.upper()} | **Risk Assessment**: **{format_risk_badge(risk)}**",
        "",
        "## 1. Quantitative Score Breakdown",
        "",
        "| Evaluation Metric | Score | Assessment Tier | Benchmark Thresholds |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Peer Similarity** | **{peer_pct:.1f}%** | {'High' if peer_pct>=40 else 'Medium' if peer_pct>=20 else 'Low'} | Low < 20.0% ≤ Med < 40.0% ≤ High |",
        f"| **External Reference Match** | **{ext_pct:.1f}%** | {'High' if ext_pct>=40 else 'Medium' if ext_pct>=20 else 'Low'} | Low < 20.0% ≤ Med < 40.0% ≤ High |",
        f"| **AI Generation Probability** | **{ai_prob:.1f}%** | {'High' if ai_prob>=75 else 'Medium' if ai_prob>=50 else 'Low'} | Low < 50.0% ≤ Med < 75.0% ≤ High |",
        f"| **Discounted Boilerplate Chunks** | **{discounted}** | Neutralized | Stripped shared tournament header |",
        "",
        "## 2. Key Diagnostic Findings",
        "",
    ]

    for exp in explanations:
        lines.append(f"- {exp}")
    lines.append("")

    # Section 3: Peer Matches
    peer_matches = report_data.get("peer_matches", [])
    if peer_matches:
        lines.append("## 3. Peer-to-Peer Plagiarism Evidence")
        lines.append("")
        for idx, m in enumerate(peer_matches, start=1):
            loc = m.get("location", {})
            q_loc = f"Page {loc.get('query_page')}" if loc.get("query_page") else "Body"
            m_loc = f"Page {loc.get('matched_page')}" if loc.get("matched_page") else "Body"
            lines.extend([
                f"### Match #{idx}: against `{m['matched_file']}` (Cosine Similarity: {m['similarity_score']:.3f})",
                f"- **Query Location**: {q_loc}",
                f"- **Matched Location**: {m_loc}",
                "",
                "**Query Passage:**",
                f"> \"{m['query_passage']}\"",
                "",
                "**Matched Passage:**",
                f"> \"{m['matched_passage']}\"",
                "",
            ])

    # Section 4: Multimodal Matches
    mm_matches = report_data.get("multimodal_matches", [])
    if mm_matches:
        lines.append("## 4. Cross-Modal Audio / Video Matches")
        lines.append("")
        for idx, m in enumerate(mm_matches, start=1):
            loc = m.get("location", {})
            q_time = loc.get("query_timestamp")
            m_time = loc.get("matched_timestamp")
            q_loc = f"Timestamp `{q_time}`" if q_time else f"Page {loc.get('query_page') or 'N/A'}"
            m_loc = f"Timestamp `{m_time}`" if m_time else f"Page {loc.get('matched_page') or 'N/A'}"
            lines.extend([
                f"### Cross-Modal Match #{idx}: with `{m['matched_file']}` (Alignment: {m['similarity_score']:.3f})",
                f"- **Submission Location**: {q_loc}",
                f"- **Matched File Location**: {m_loc}",
                "",
                "**Transcribed / Document Passage A:**",
                f"> \"{m['query_passage']}\"",
                "",
                "**Transcribed / Document Passage B:**",
                f"> \"{m['matched_passage']}\"",
                "",
            ])

    # Section 5: External Matches
    ext_matches = report_data.get("external_matches", [])
    if ext_matches:
        lines.append("## 5. External Reference Corpus Overlaps")
        lines.append("")
        for idx, m in enumerate(ext_matches, start=1):
            loc = m.get("location", {})
            q_loc = f"Page {loc.get('query_page')}" if loc.get("query_page") else "Passage"
            lines.extend([
                f"### Reference Match #{idx}: `{m['matched_file']}` (Cosine Similarity: {m['similarity_score']:.3f})",
                f"- **Location in Document**: {q_loc}",
                "",
                "**Student Submission Passage:**",
                f"> \"{m['query_passage']}\"",
                "",
                "**External Reference Passage:**",
                f"> \"{m['matched_passage']}\"",
                "",
            ])

    if not peer_matches and not mm_matches and not ext_matches:
        lines.extend([
            "## 3. Plagiarism Evidence",
            "",
            "_No significant peer, multimodal, or external reference matches detected above threshold (0.75)._",
            "",
        ])

    lines.extend([
        "---",
        "*Report automatically compiled by Multimodal Integrity & Plagiarism Detection Engine.*",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_summary_matrix(reports: list[dict[str, Any]], output_dir: Path) -> None:
    """Generates summary_matrix.json and summary_matrix.md covering all processed files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. summary_matrix.json
    matrix_json_path = output_dir / "summary_matrix.json"
    with open(matrix_json_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    # 2. summary_matrix.md
    matrix_md_path = output_dir / "summary_matrix.md"
    lines = [
        "# Multimodal Submission Integrity Assessment — Summary Matrix",
        "",
        f"**Total Documents Evaluated**: {len(reports)}",
        "",
        "| Document | Status | Risk Level | Peer Sim (%) | Ext Sim (%) | AI Prob (%) | Boilerplate Filtered | Flagged Sources |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for r in reports:
        doc_id = r["document_id"]
        status = r["status"]
        risk = r["total_integrity_risk"]
        scores = r.get("scores", {})
        peer_pct = scores.get("peer_similarity_percentage", 0.0)
        ext_pct = scores.get("external_similarity_percentage", 0.0)
        ai_prob = scores.get("ai_generated_probability", 0.0)
        discounted = r.get("boilerplate_chunks_discounted", 0)

        flagged = set()
        for m in r.get("peer_matches", []):
            flagged.add(m["matched_file"])
        for m in r.get("multimodal_matches", []):
            flagged.add(m["matched_file"])
        for m in r.get("external_matches", []):
            flagged.add(m["matched_file"])

        flagged_str = ", ".join(sorted(list(flagged))) if flagged else "None"
        if ai_prob >= 75.0:
            flagged_str = f"Synthetic LLM ({flagged_str})" if flagged_str != "None" else "Synthetic LLM"

        lines.append(
            f"| `{doc_id}` | {status.upper()} | {format_risk_badge(risk)} | "
            f"{peer_pct:.1f}% | {ext_pct:.1f}% | {ai_prob:.1f}% | {discounted} | {flagged_str} |"
        )

    lines.extend([
        "",
        "### Risk Categorization Policy",
        "- **HIGH RISK**: Peer Similarity ≥ 40% OR External Similarity ≥ 40% OR AI Generation Likelihood ≥ 75%",
        "- **MEDIUM RISK**: Peer Similarity ≥ 20% OR External Similarity ≥ 20% OR AI Generation Likelihood ≥ 50%",
        "- **LOW RISK**: Below medium threshold guidelines",
        "- **UNPROCESSED**: Corrupted or empty submission files",
        "",
    ])

    matrix_md_path.write_text("\n".join(lines), encoding="utf-8")
