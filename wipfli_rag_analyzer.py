#!/usr/bin/env python3
"""
Wipfli DEI RAG Analyzer
=======================
Retrieval-Augmented Generation pipeline that cross-references Wipfli's DEI
Handbook policies against actual India-office staff data to surface gaps
and produce a terminal executive report.

Stack: langchain + FAISS + sentence-transformers (RAG), rich (terminal UI),
Anthropic Claude API (action plan generation).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import anthropic
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
HANDBOOK_PATH = BASE_DIR / "data" / "Wipfli_DEI_Handbook.txt"
STAFF_PATH = BASE_DIR / "data" / "wipfli_staff_data.csv"
REPORT_PATH = BASE_DIR / "reports" / "Wipfli_DEI_Audit_Feb2026.md"

console = Console(width=100)

# ── Handbook target constants (from the handbook text) ───────────────────────
TARGET_FEMALE_PCT = 40          # India-office target for IT & Consulting
TARGET_DIRECTOR_FEMALE_PCT = 40 # Director-level firm-wide target by 2027
TARGET_MANAGER_FEMALE_PCT = 50  # Manager-level firm-wide target by 2027
TENURE_GAP_THRESHOLD = 3.0     # years – flags retention concern


# =============================================================================
# 1. INGESTION — load handbook → chunk → FAISS vector store
# =============================================================================
def build_vector_store() -> FAISS:
    console.log("[bold cyan]Loading DEI Handbook …[/]")
    text = HANDBOOK_PATH.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents([text])
    console.log(f"  Split into [bold]{len(chunks)}[/] chunks (500-char windows)")

    console.log("[bold cyan]Embedding chunks into FAISS (first run downloads ~80 MB model) …[/]")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    store = FAISS.from_documents(chunks, embeddings)
    console.log(f"  FAISS index ready — [bold]{store.index.ntotal}[/] vectors")
    return store


# =============================================================================
# 2. DATA ANALYSIS — gender ratios, tenure gaps, level distribution
# =============================================================================
def analyze_staff() -> dict:
    df = pd.read_csv(STAFF_PATH)
    total = len(df)

    gender_counts = df["gender"].value_counts()
    female_pct = round(gender_counts.get("Female", 0) / total * 100, 1)
    male_pct = round(gender_counts.get("Male", 0) / total * 100, 1)

    # Per-department breakdown
    dept_stats = []
    for dept, grp in df.groupby("department"):
        f_count = (grp["gender"] == "Female").sum()
        f_pct = round(f_count / len(grp) * 100, 1)
        tenure_f = grp.loc[grp["gender"] == "Female", "years_at_firm"].mean()
        tenure_m = grp.loc[grp["gender"] == "Male", "years_at_firm"].mean()
        gap = round(tenure_m - tenure_f, 1) if pd.notna(tenure_f) else None
        dept_stats.append({
            "department": dept,
            "female": f_count,
            "male": len(grp) - f_count,
            "total": len(grp),
            "female_pct": f_pct,
            "tenure_f": round(tenure_f, 1) if pd.notna(tenure_f) else None,
            "tenure_m": round(tenure_m, 1),
            "tenure_gap": gap,
        })

    # Per-level breakdown
    level_order = ["Associate", "Senior Associate", "Manager",
                   "Senior Manager", "Director"]
    level_stats = []
    for lvl in level_order:
        sub = df[df["job_level"] == lvl]
        if sub.empty:
            continue
        f = (sub["gender"] == "Female").sum()
        level_stats.append({
            "level": lvl,
            "female": f,
            "male": len(sub) - f,
            "total": len(sub),
            "female_pct": round(f / len(sub) * 100, 1),
        })

    # Leadership (Senior Manager + Director)
    senior = df[df["job_level"].isin(["Senior Manager", "Director"])]
    f_senior = (senior["gender"] == "Female").sum()
    leadership_f_pct = round(f_senior / len(senior) * 100, 1) if len(senior) else 0.0

    return {
        "total": total,
        "female_pct": female_pct,
        "male_pct": male_pct,
        "dept_stats": dept_stats,
        "level_stats": level_stats,
        "leadership_f_pct": leadership_f_pct,
        "leadership_f_count": f_senior,
        "leadership_total": len(senior),
    }


# =============================================================================
# 3. RAG LOGIC — detect imbalances → query handbook → return aligned policies
# =============================================================================
def detect_imbalances(metrics: dict) -> list[dict]:
    """Scan metrics for gaps and build RAG search queries for each."""
    findings = []

    # Department-level gender gaps
    for ds in metrics["dept_stats"]:
        if ds["female_pct"] < 30:
            findings.append({
                "issue": (f"{ds['department']} department has only "
                          f"{ds['female_pct']}% female representation"),
                "query": (f"women representation {ds['department']} department "
                          f"India office gender equity target"),
                "dept": ds["department"],
                "severity": "HIGH",
            })
        elif ds["female_pct"] < TARGET_FEMALE_PCT:
            findings.append({
                "issue": (f"{ds['department']} department is at "
                          f"{ds['female_pct']}% female — below the 40% target"),
                "query": (f"gender equity target women representation "
                          f"{ds['department']} India"),
                "dept": ds["department"],
                "severity": "MEDIUM",
            })

        if ds["tenure_gap"] is not None and ds["tenure_gap"] > TENURE_GAP_THRESHOLD:
            sev = "HIGH" if ds["tenure_gap"] > 4 else "MEDIUM"
            findings.append({
                "issue": (f"{ds['department']} department shows a "
                          f"{ds['tenure_gap']}-year tenure gap (Male avg > Female avg)"),
                "query": ("women attrition retention tenure years India office "
                          "reduce attrition women associates"),
                "dept": ds["department"],
                "severity": sev,
            })

    # Leadership gap
    if metrics["leadership_f_pct"] < 30:
        findings.append({
            "issue": (f"Senior leadership (Director + Sr. Manager) is only "
                      f"{metrics['leadership_f_pct']}% female "
                      f"({metrics['leadership_f_count']}/{metrics['leadership_total']})"),
            "query": ("women leadership pipeline director senior manager "
                      "promotion gender representation target"),
            "severity": "HIGH",
        })

    # Pipeline narrowing
    levels = metrics["level_stats"]
    if len(levels) >= 2:
        entry_pct = levels[0]["female_pct"]
        top_pct = levels[-1]["female_pct"]
        if entry_pct - top_pct > 30:
            findings.append({
                "issue": (f"Severe pipeline narrowing: {entry_pct}% female at "
                          f"{levels[0]['level']} level drops to {top_pct}% at "
                          f"{levels[-1]['level']} level"),
                "query": ("promotion practices succession planning bias "
                          "inclusive pipeline advancement women leadership"),
                "severity": "HIGH",
            })

    return findings


def retrieve_policies(store: FAISS, findings: list[dict], k: int = 3) -> list[dict]:
    """For each finding, retrieve top-k handbook chunks via similarity search."""
    for f in findings:
        docs = store.similarity_search(f["query"], k=k)
        f["policy_chunks"] = [doc.page_content.strip() for doc in docs]
    return findings


def generate_action_plan(store: FAISS, metrics: dict, findings: list[dict]) -> list[str]:
    """Retrieve 'Inclusive Promotion Practices' policy chunks from the handbook,
    then ask Claude to propose three concrete interview/promotion policy changes
    grounded in both the data gaps and the handbook language."""

    # RAG retrieval — pull the most relevant promotion-policy chunks
    docs = store.similarity_search(
        "inclusive promotion practices succession planning bias interview "
        "hiring women leadership advancement equity",
        k=5,
    )
    policy_context = "\n\n---\n\n".join(doc.page_content.strip() for doc in docs)

    # Build a data summary for the prompt
    high_findings = [f["issue"] for f in findings if f["severity"] == "HIGH"]
    data_summary = (
        f"Office: Wipfli India (Pune / Bengaluru) — {metrics['total']} employees\n"
        f"Overall female representation: {metrics['female_pct']}%\n"
        f"Women in senior leadership (Director + Sr. Manager): "
        f"{metrics['leadership_f_pct']}% ({metrics['leadership_f_count']}"
        f"/{metrics['leadership_total']})\n\n"
        f"Key HIGH-severity findings:\n"
    )
    for item in high_findings:
        data_summary += f"  - {item}\n"

    prompt = (
        "You are an HR policy advisor for Wipfli LLP. Below are excerpts from "
        "Wipfli's DEI Handbook (retrieved via RAG) followed by current workforce "
        "data from the India office.\n\n"
        "=== HANDBOOK EXCERPTS (Promotion & Advancement Policies) ===\n"
        f"{policy_context}\n\n"
        "=== CURRENT WORKFORCE DATA ===\n"
        f"{data_summary}\n"
        "Based on the handbook's 'Inclusive Promotion Practices' commitments and "
        "the data gaps above, propose exactly THREE specific, actionable changes "
        "to Wipfli India's interview or promotion policies. Each change must:\n"
        "  1. Reference a specific handbook commitment it enforces.\n"
        "  2. Name the data gap it addresses.\n"
        "  3. Be concrete enough to implement in the next quarter.\n\n"
        "Return ONLY the three numbered items (1., 2., 3.), each as a single "
        "paragraph of 2-4 sentences. No preamble, no summary."
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        console.print(
            "[bold yellow]WARNING:[/] ANTHROPIC_API_KEY not set — "
            "skipping Claude-generated Action Plan."
        )
        return []

    console.log("[bold cyan]Generating Action Plan via Claude …[/]")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
    except anthropic.APIError as exc:
        console.print(
            f"[bold yellow]WARNING:[/] Claude API call failed — {exc.message}. "
            "Skipping Action Plan."
        )
        return []

    # Parse the numbered items
    actions = []
    for line in raw.split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and "." in line[:3]:
            actions.append(line.split(".", 1)[1].strip())
        elif actions:
            # continuation of the previous item
            actions[-1] += " " + line

    console.log(f"  Received [bold]{len(actions)}[/] action items from Claude")
    return actions[:3]


def synthesize_verdict(finding: dict) -> str:
    """Produce a plain-English verdict by matching retrieved policy text
    against the specific imbalance.  No LLM needed — pattern-based."""

    policies_text = " ".join(finding.get("policy_chunks", [])).lower()
    issue = finding["issue"].lower()
    dept = finding.get("dept", "")

    # Identify which handbook sections the retriever surfaced
    refs = []
    if "aparajita" in policies_text:
        refs.append("Aparajita BRG (Section 3.5)")
    if "women of wipfli" in policies_text or "wow" in policies_text:
        refs.append("WoW BRG (Section 3.4)")
    if "ceo" in policies_text or "pledge" in policies_text:
        refs.append("CEO Pledge (Section 2)")
    if "governance" in policies_text or "accountability" in policies_text:
        refs.append("DEI Governance (Section 4)")
    if "promotion" in policies_text and "succession" in policies_text:
        refs.append("Inclusive Promotion Practices (CEO Pledge #3)")
    if "embrace" in policies_text:
        refs.append("Embrace BRG (Section 3.1)")
    ref_label = ", ".join(refs) if refs else "General DEI Mission (Section 1)"

    # Build verdict by imbalance type
    if "tenure gap" in issue:
        return (
            f"The tenure disparity in {dept} signals higher female attrition "
            f"or later hiring. This conflicts with [{ref_label}] which commits "
            f"to 'reduce attrition among women associates in years 3-5 of "
            f"tenure' and ensure equitable retention across the India office."
        )
    if "leadership" in issue or "senior" in issue:
        return (
            f"At {finding.get('issue').split('only ')[-1].split('%')[0]}% female, "
            f"senior leadership falls far below the handbook's Director-level "
            f"target of 40% women by 2027. Relevant sections: [{ref_label}]. "
            f"The India Leadership Pipeline initiative under Aparajita "
            f"specifically aims to develop India-based talent for senior roles."
        )
    if "pipeline" in issue:
        return (
            f"The steep drop-off in female representation across job levels "
            f"indicates systemic advancement barriers. This directly "
            f"contradicts [{ref_label}] — the CEO pledges annual bias reviews "
            f"of promotion processes and the WoW BRG runs a Leadership "
            f"Accelerator program to address exactly this pattern."
        )
    if "representation" in issue or "female" in issue:
        return (
            f"{dept}'s low female ratio violates the targets set in "
            f"[{ref_label}]. The handbook's India Office DEI Focus Areas "
            f"explicitly set a target of 40% women in IT and Consulting at "
            f"mid-senior levels by 2027. The Aparajita BRG's Gender Equity "
            f"initiative includes women-in-tech mentoring for this purpose."
        )
    return f"Relevant handbook sections: [{ref_label}]."


# =============================================================================
# 4. OUTPUT — rich terminal executive report
# =============================================================================
def render_report(metrics: dict, findings: list[dict],
                   action_plan: list[str] | None = None) -> None:
    SEV_COLOR = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}

    console.print()
    console.rule(
        "[bold white on blue]  WIPFLI DEI EXECUTIVE REPORT  \u2014  India Office (Pune / Bengaluru)  [/]",
        style="blue",
    )
    console.print()

    # ── Snapshot panel ───────────────────────────────────────────────────
    console.print(Panel(
        f"[bold]Headcount:[/] {metrics['total']}    "
        f"[bold]Female:[/] {metrics['female_pct']}%    "
        f"[bold]Male:[/] {metrics['male_pct']}%    "
        f"[bold]Women in Senior Leadership:[/] {metrics['leadership_f_pct']}%  "
        f"({metrics['leadership_f_count']}/{metrics['leadership_total']})",
        title="[bold]Workforce Snapshot[/]",
        border_style="cyan",
    ))
    console.print()

    # ── Department table ─────────────────────────────────────────────────
    t1 = Table(
        title="Department Breakdown — Gender & Tenure",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
        title_style="bold",
    )
    t1.add_column("Department", style="bold")
    t1.add_column("Female", justify="center")
    t1.add_column("Male", justify="center")
    t1.add_column("Female %", justify="center")
    t1.add_column("Avg Tenure (F)", justify="center")
    t1.add_column("Avg Tenure (M)", justify="center")
    t1.add_column("Tenure Gap", justify="center")

    for ds in metrics["dept_stats"]:
        pc = "red" if ds["female_pct"] < 30 else ("yellow" if ds["female_pct"] < 40 else "green")
        g = ds["tenure_gap"]
        gc = "red" if g and g > 4 else ("yellow" if g and g > 2 else "green")
        t1.add_row(
            ds["department"],
            str(ds["female"]),
            str(ds["male"]),
            f"[{pc}]{ds['female_pct']}%[/{pc}]",
            f"{ds['tenure_f']} yrs" if ds["tenure_f"] is not None else "\u2014",
            f"{ds['tenure_m']} yrs",
            f"[{gc}]{g} yrs[/{gc}]" if g is not None else "\u2014",
        )
    console.print(t1)
    console.print()

    # ── Job-level table ──────────────────────────────────────────────────
    t2 = Table(
        title="Gender Distribution by Job Level (Pipeline View)",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
        title_style="bold",
    )
    t2.add_column("Job Level", style="bold")
    t2.add_column("Female", justify="center")
    t2.add_column("Male", justify="center")
    t2.add_column("Total", justify="center")
    t2.add_column("Female %", justify="center")
    t2.add_column("Pipeline Bar", min_width=22)

    for ls in metrics["level_stats"]:
        pct = ls["female_pct"]
        c = "red" if pct < 20 else ("yellow" if pct < 35 else "green")
        filled = int(pct / 5)              # each block ≈ 5%
        bar = f"[{c}]\u2588[/{c}]" * filled + "[dim]\u2591[/dim]" * (20 - filled)
        t2.add_row(
            ls["level"],
            str(ls["female"]),
            str(ls["male"]),
            str(ls["total"]),
            f"[{c}]{pct}%[/{c}]",
            bar,
        )
    console.print(t2)
    console.print()

    # ── Policy Alignment ─────────────────────────────────────────────────
    console.rule(
        "[bold white on magenta]  Policy Alignment \u2014 RAG Analysis  [/]",
        style="magenta",
    )
    console.print()

    for i, f in enumerate(findings, 1):
        sev = f.get("severity", "MEDIUM")
        sc = SEV_COLOR.get(sev, "white")

        # Finding header
        console.print(
            f"[bold {sc}]FINDING {i} [{sev}][/bold {sc}]  {f['issue']}"
        )
        console.print()

        # Retrieved handbook excerpts
        for j, chunk in enumerate(f.get("policy_chunks", []), 1):
            cleaned = " ".join(chunk.split())
            if len(cleaned) > 400:
                cleaned = cleaned[:397] + " …"
            console.print(Panel(
                cleaned,
                title=f"[dim]Retrieved Handbook Chunk {j}[/dim]",
                border_style="dim",
                width=96,
                padding=(0, 1),
            ))

        # Verdict
        verdict = synthesize_verdict(f)
        console.print(f"  [bold {sc}]\u279c Verdict:[/bold {sc}] {verdict}")
        console.print()
        console.print("[dim]" + "\u2500" * 96 + "[/dim]")
        console.print()

    # ── Action Plan ──────────────────────────────────────────────────────
    if action_plan:
        console.rule(
            "[bold white on green]  Action Plan \u2014 Claude-Generated Policy Recommendations  [/]",
            style="green",
        )
        console.print()
        for idx, action in enumerate(action_plan, 1):
            console.print(Panel(
                action,
                title=f"[bold]Recommendation {idx}[/bold]",
                border_style="green",
                width=96,
                padding=(0, 1),
            ))
        console.print()

    # ── Footer ───────────────────────────────────────────────────────────
    console.rule("[bold green]  End of Report  [/]", style="green")
    console.print(
        "[dim]Generated by Wipfli DEI RAG Analyzer  |  Data is synthetic  |  "
        "Powered by FAISS + sentence-transformers + LangChain + Anthropic Claude[/dim]",
        justify="center",
    )
    console.print()


# =============================================================================
# 5. MARKDOWN EXPORT
# =============================================================================
def export_markdown(metrics: dict, findings: list[dict],
                    action_plan: list[str] | None = None) -> Path:
    """Write the full executive report as a Markdown file."""
    lines: list[str] = []

    def w(text: str = "") -> None:
        lines.append(text)

    w("# Wipfli DEI Executive Report \u2014 India Office (Pune / Bengaluru)")
    w()
    w("**Report Date:** February 2026  ")
    w("**Data Classification:** Synthetic  ")
    w("**Generated by:** Wipfli DEI RAG Analyzer (FAISS + sentence-transformers + Claude)")
    w()
    w("---")
    w()

    # Snapshot
    w("## Workforce Snapshot")
    w()
    w(f"| Metric | Value |")
    w(f"|--------|-------|")
    w(f"| Total Headcount | {metrics['total']} |")
    w(f"| Female | {metrics['female_pct']}% |")
    w(f"| Male | {metrics['male_pct']}% |")
    w(f"| Women in Senior Leadership | {metrics['leadership_f_pct']}% "
      f"({metrics['leadership_f_count']}/{metrics['leadership_total']}) |")
    w()

    # Department table
    w("## Department Breakdown \u2014 Gender & Tenure")
    w()
    w("| Department | Female | Male | Female % | Avg Tenure (F) | Avg Tenure (M) | Tenure Gap |")
    w("|------------|:------:|:----:|:--------:|:--------------:|:--------------:|:----------:|")
    for ds in metrics["dept_stats"]:
        flag = " \u26a0\ufe0f" if ds["female_pct"] < 30 else ""
        tf = f"{ds['tenure_f']} yrs" if ds["tenure_f"] is not None else "\u2014"
        tg = f"{ds['tenure_gap']} yrs" if ds["tenure_gap"] is not None else "\u2014"
        w(f"| {ds['department']}{flag} | {ds['female']} | {ds['male']} | "
          f"{ds['female_pct']}% | {tf} | {ds['tenure_m']} yrs | {tg} |")
    w()

    # Level table
    w("## Gender Distribution by Job Level (Pipeline View)")
    w()
    w("| Job Level | Female | Male | Total | Female % |")
    w("|-----------|:------:|:----:|:-----:|:--------:|")
    for ls in metrics["level_stats"]:
        pct = ls["female_pct"]
        bar = "\u2588" * int(pct / 5) + "\u2591" * (20 - int(pct / 5))
        w(f"| {ls['level']} | {ls['female']} | {ls['male']} | "
          f"{ls['total']} | {pct}% `{bar}` |")
    w()

    # Policy Alignment
    w("---")
    w()
    w("## Policy Alignment \u2014 RAG Analysis")
    w()
    for i, f in enumerate(findings, 1):
        sev = f.get("severity", "MEDIUM")
        sev_icon = "\U0001f534" if sev == "HIGH" else "\U0001f7e1"
        w(f"### Finding {i} {sev_icon} [{sev}]")
        w()
        w(f"**{f['issue']}**")
        w()
        for j, chunk in enumerate(f.get("policy_chunks", []), 1):
            cleaned = " ".join(chunk.split())
            if len(cleaned) > 400:
                cleaned = cleaned[:397] + " \u2026"
            w(f"> **Handbook Reference {j}:** {cleaned}")
            w()
        verdict = synthesize_verdict(f)
        w(f"\u279c **Verdict:** {verdict}")
        w()
        w("---")
        w()

    # Action Plan
    if action_plan:
        w("## Action Plan \u2014 Claude-Generated Policy Recommendations")
        w()
        for idx, action in enumerate(action_plan, 1):
            w(f"**Recommendation {idx}:** {action}")
            w()
        w("---")
        w()

    w("*Generated by Wipfli DEI RAG Analyzer | Data is synthetic | "
      "Powered by FAISS + sentence-transformers + LangChain + Anthropic Claude*")
    w()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


# =============================================================================
# MAIN
# =============================================================================
def main():
    console.print()
    console.print(Panel(
        "[bold white]Wipfli DEI RAG Analyzer[/]\n"
        "[dim]Retrieval-Augmented Generation pipeline for policy-vs-data alignment[/]",
        border_style="blue",
        padding=(1, 2),
    ), justify="center")
    console.print()

    for path, label in [(HANDBOOK_PATH, "Handbook"), (STAFF_PATH, "Staff CSV")]:
        if not path.exists():
            console.print(f"[bold red]ERROR:[/] {label} not found at {path}")
            sys.exit(1)

    # Step 1 — Ingest handbook into FAISS
    store = build_vector_store()
    console.print()

    # Step 2 — Analyze staff data
    console.log("[bold cyan]Analyzing staff data …[/]")
    metrics = analyze_staff()
    console.log(
        f"  {metrics['total']} employees  |  "
        f"{len(metrics['dept_stats'])} departments  |  "
        f"{len(metrics['level_stats'])} job levels"
    )
    console.print()

    # Step 3 — Detect imbalances → retrieve relevant policies
    console.log("[bold cyan]Scanning for DEI imbalances …[/]")
    findings = detect_imbalances(metrics)
    console.log(f"  [bold]{len(findings)}[/] imbalance(s) detected")
    console.print()

    console.log("[bold cyan]Querying FAISS for relevant handbook policies …[/]")
    findings = retrieve_policies(store, findings, k=3)
    console.log("  Policy retrieval complete")
    console.print()

    # Step 4 — Generate Action Plan via Claude
    action_plan = generate_action_plan(store, metrics, findings)
    console.print()

    # Step 5 — Render executive report (terminal)
    render_report(metrics, findings, action_plan)

    # Step 6 — Export as Markdown
    console.log("[bold cyan]Exporting Markdown report …[/]")
    md_path = export_markdown(metrics, findings, action_plan)
    console.log(f"  Saved to [bold]{md_path}[/]")
    console.print()


if __name__ == "__main__":
    main()
