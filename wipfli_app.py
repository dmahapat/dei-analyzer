#!/usr/bin/env python3
"""
Wipfli DEI RAG Analyzer — Streamlit Web UI
===========================================
Interactive dashboard that cross-references Wipfli's DEI Handbook policies
against India-office staff data using a FAISS-backed RAG pipeline.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import anthropic
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── Page config (must be first st call) ──────────────────────────────────────
st.set_page_config(
    page_title="Wipfli DEI Analyzer",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths & constants ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
HANDBOOK_PATH = BASE_DIR / "data" / "Wipfli_DEI_Handbook.txt"
DEFAULT_STAFF_PATH = BASE_DIR / "data" / "wipfli_staff_data.csv"

TARGET_FEMALE_PCT = 40
TENURE_GAP_THRESHOLD = 3.0

SEV_COLORS = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}
SEV_ICONS = {"HIGH": "\U0001f534", "MEDIUM": "\U0001f7e1", "LOW": "\U0001f7e2"}


# =============================================================================
# INGESTION — cached so the model is only downloaded / indexed once
# =============================================================================
@st.cache_resource(show_spinner="Building FAISS vector store (first run downloads ~80 MB model)…")
def build_vector_store() -> FAISS:
    text = HANDBOOK_PATH.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return FAISS.from_documents(chunks, embeddings)


# =============================================================================
# DATA ANALYSIS
# =============================================================================
def analyze_staff(df: pd.DataFrame) -> dict:
    total = len(df)
    gender_counts = df["gender"].value_counts()
    female_pct = round(gender_counts.get("Female", 0) / total * 100, 1)
    male_pct = round(gender_counts.get("Male", 0) / total * 100, 1)

    dept_stats = []
    for dept, grp in df.groupby("department"):
        f_count = (grp["gender"] == "Female").sum()
        f_pct = round(f_count / len(grp) * 100, 1)
        tenure_f = grp.loc[grp["gender"] == "Female", "years_at_firm"].mean()
        tenure_m = grp.loc[grp["gender"] == "Male", "years_at_firm"].mean()
        gap = round(tenure_m - tenure_f, 1) if pd.notna(tenure_f) else None
        dept_stats.append({
            "department": dept,
            "female": int(f_count), "male": int(len(grp) - f_count),
            "total": len(grp), "female_pct": f_pct,
            "tenure_f": round(tenure_f, 1) if pd.notna(tenure_f) else None,
            "tenure_m": round(tenure_m, 1),
            "tenure_gap": gap,
        })

    level_order = ["Associate", "Senior Associate", "Manager",
                   "Senior Manager", "Director"]
    level_stats = []
    for lvl in level_order:
        sub = df[df["job_level"] == lvl]
        if sub.empty:
            continue
        f = (sub["gender"] == "Female").sum()
        level_stats.append({
            "level": lvl, "female": int(f), "male": int(len(sub) - f),
            "total": len(sub), "female_pct": round(f / len(sub) * 100, 1),
        })

    senior = df[df["job_level"].isin(["Senior Manager", "Director"])]
    f_senior = (senior["gender"] == "Female").sum()
    leadership_f_pct = round(f_senior / len(senior) * 100, 1) if len(senior) else 0.0

    return {
        "total": total, "female_pct": female_pct, "male_pct": male_pct,
        "dept_stats": dept_stats, "level_stats": level_stats,
        "leadership_f_pct": leadership_f_pct,
        "leadership_f_count": int(f_senior),
        "leadership_total": int(len(senior)),
    }


# =============================================================================
# RAG LOGIC
# =============================================================================
def detect_imbalances(metrics: dict) -> list[dict]:
    findings = []
    for ds in metrics["dept_stats"]:
        if ds["female_pct"] < 30:
            findings.append({
                "issue": f"{ds['department']} department has only {ds['female_pct']}% female representation",
                "query": f"women representation {ds['department']} department India office gender equity target",
                "dept": ds["department"], "severity": "HIGH",
            })
        elif ds["female_pct"] < TARGET_FEMALE_PCT:
            findings.append({
                "issue": f"{ds['department']} department is at {ds['female_pct']}% female — below the 40% target",
                "query": f"gender equity target women representation {ds['department']} India",
                "dept": ds["department"], "severity": "MEDIUM",
            })
        if ds["tenure_gap"] is not None and ds["tenure_gap"] > TENURE_GAP_THRESHOLD:
            findings.append({
                "issue": f"{ds['department']} department shows a {ds['tenure_gap']}-year tenure gap (Male avg > Female avg)",
                "query": "women attrition retention tenure years India office reduce attrition women associates",
                "dept": ds["department"],
                "severity": "HIGH" if ds["tenure_gap"] > 4 else "MEDIUM",
            })

    if metrics["leadership_f_pct"] < 30:
        findings.append({
            "issue": (f"Senior leadership (Director + Sr. Manager) is only "
                      f"{metrics['leadership_f_pct']}% female "
                      f"({metrics['leadership_f_count']}/{metrics['leadership_total']})"),
            "query": "women leadership pipeline director senior manager promotion gender representation target",
            "severity": "HIGH",
        })

    levels = metrics["level_stats"]
    if len(levels) >= 2:
        entry_pct, top_pct = levels[0]["female_pct"], levels[-1]["female_pct"]
        if entry_pct - top_pct > 30:
            findings.append({
                "issue": (f"Severe pipeline narrowing: {entry_pct}% female at "
                          f"{levels[0]['level']} level drops to {top_pct}% at "
                          f"{levels[-1]['level']} level"),
                "query": "promotion practices succession planning bias inclusive pipeline advancement women leadership",
                "severity": "HIGH",
            })
    return findings


def retrieve_policies(store: FAISS, findings: list[dict], k: int = 3) -> list[dict]:
    for f in findings:
        docs = store.similarity_search(f["query"], k=k)
        f["policy_chunks"] = [doc.page_content.strip() for doc in docs]
    return findings


def synthesize_verdict(finding: dict) -> str:
    policies_text = " ".join(finding.get("policy_chunks", [])).lower()
    issue = finding["issue"].lower()
    dept = finding.get("dept", "")

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

    if "tenure gap" in issue:
        return (f"The tenure disparity in {dept} signals higher female attrition "
                f"or later hiring. This conflicts with **{ref_label}** which commits "
                f"to 'reduce attrition among women associates in years 3-5 of tenure' "
                f"and ensure equitable retention across the India office.")
    if "leadership" in issue or "senior" in issue:
        return (f"At {finding['issue'].split('only ')[-1].split('%')[0]}% female, "
                f"senior leadership falls far below the handbook's Director-level "
                f"target of 40% women by 2027. Relevant sections: **{ref_label}**. "
                f"The India Leadership Pipeline initiative under Aparajita specifically "
                f"aims to develop India-based talent for senior roles.")
    if "pipeline" in issue:
        return (f"The steep drop-off in female representation across job levels "
                f"indicates systemic advancement barriers. This directly contradicts "
                f"**{ref_label}** — the CEO pledges annual bias reviews of promotion "
                f"processes and the WoW BRG runs a Leadership Accelerator program to "
                f"address exactly this pattern.")
    if "representation" in issue or "female" in issue:
        return (f"{dept}'s low female ratio violates the targets set in "
                f"**{ref_label}**. The handbook's India Office DEI Focus Areas "
                f"explicitly set a target of 40% women in IT and Consulting at "
                f"mid-senior levels by 2027. The Aparajita BRG's Gender Equity "
                f"initiative includes women-in-tech mentoring for this purpose.")
    return f"Relevant handbook sections: **{ref_label}**."


def generate_action_plan(store: FAISS, metrics: dict, findings: list[dict]) -> list[str]:
    docs = store.similarity_search(
        "inclusive promotion practices succession planning bias interview "
        "hiring women leadership advancement equity", k=5,
    )
    policy_context = "\n\n---\n\n".join(doc.page_content.strip() for doc in docs)

    high_findings = [f["issue"] for f in findings if f["severity"] == "HIGH"]
    data_summary = (
        f"Office: Wipfli India (Pune / Bengaluru) — {metrics['total']} employees\n"
        f"Overall female representation: {metrics['female_pct']}%\n"
        f"Women in senior leadership (Director + Sr. Manager): "
        f"{metrics['leadership_f_pct']}% ({metrics['leadership_f_count']}"
        f"/{metrics['leadership_total']})\n\nKey HIGH-severity findings:\n"
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
        return []

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
    except anthropic.APIError:
        return []

    actions = []
    for line in raw.split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and "." in line[:3]:
            actions.append(line.split(".", 1)[1].strip())
        elif actions:
            actions[-1] += " " + line
    return actions[:3]


# =============================================================================
# CHARTS
# =============================================================================
def chart_workforce_snapshot(metrics: dict) -> go.Figure:
    dept_df = pd.DataFrame(metrics["dept_stats"])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dept_df["department"], y=dept_df["female"],
        name="Female", marker_color="#e056a0",
    ))
    fig.add_trace(go.Bar(
        x=dept_df["department"], y=dept_df["male"],
        name="Male", marker_color="#4a90d9",
    ))
    fig.add_hline(y=0, line_width=0)
    fig.update_layout(
        barmode="group",
        title="Gender Distribution by Department",
        xaxis_title="Department", yaxis_title="Headcount",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
        height=380,
    )
    return fig


def chart_tenure_gap(metrics: dict) -> go.Figure:
    dept_df = pd.DataFrame(metrics["dept_stats"])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dept_df["department"], y=dept_df["tenure_f"],
        name="Female Avg Tenure", marker_color="#e056a0",
    ))
    fig.add_trace(go.Bar(
        x=dept_df["department"], y=dept_df["tenure_m"],
        name="Male Avg Tenure", marker_color="#4a90d9",
    ))
    fig.add_hline(
        y=TENURE_GAP_THRESHOLD, line_dash="dash", line_color="red",
        annotation_text=f"Alert Threshold ({TENURE_GAP_THRESHOLD}yr gap)",
        annotation_position="top left",
    )
    fig.update_layout(
        barmode="group",
        title="Average Tenure by Gender & Department",
        xaxis_title="Department", yaxis_title="Years",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
        height=380,
    )
    return fig


def chart_pipeline(metrics: dict) -> go.Figure:
    lvl_df = pd.DataFrame(metrics["level_stats"])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lvl_df["level"], y=lvl_df["female"],
        name="Female", marker_color="#e056a0",
    ))
    fig.add_trace(go.Bar(
        x=lvl_df["level"], y=lvl_df["male"],
        name="Male", marker_color="#4a90d9",
    ))
    fig.update_layout(
        barmode="stack",
        title="Job Level Pipeline — Gender Composition",
        xaxis_title="Job Level", yaxis_title="Headcount",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
        height=400,
    )
    return fig


def chart_pipeline_pct(metrics: dict) -> go.Figure:
    lvl_df = pd.DataFrame(metrics["level_stats"])
    colors = ["#e74c3c" if p < 20 else "#f39c12" if p < 35 else "#2ecc71"
              for p in lvl_df["female_pct"]]
    fig = go.Figure(go.Bar(
        x=lvl_df["level"], y=lvl_df["female_pct"],
        marker_color=colors,
        text=[f"{p}%" for p in lvl_df["female_pct"]],
        textposition="outside",
    ))
    fig.add_hline(y=40, line_dash="dash", line_color="green",
                  annotation_text="40% Target", annotation_position="top left")
    fig.update_layout(
        title="Female % by Job Level (Pipeline Leakage)",
        xaxis_title="Job Level", yaxis_title="Female %",
        template="plotly_white",
        margin=dict(t=60, b=40),
        height=400,
        yaxis_range=[0, 80],
    )
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar() -> pd.DataFrame:
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/diversity.png", width=64)
        st.title("Wipfli DEI Analyzer")
        st.caption("RAG-powered policy alignment")
        st.divider()

        st.subheader("Data Source")
        uploaded = st.file_uploader(
            "Upload a staff CSV", type=["csv"],
            help="CSV must contain columns: employee_id, gender, department, job_level, years_at_firm",
        )

        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded **{len(df)}** rows from upload")
        else:
            df = pd.read_csv(DEFAULT_STAFF_PATH)
            st.info(f"Using default dataset ({len(df)} rows)")

        required = {"gender", "department", "job_level", "years_at_firm"}
        missing = required - set(df.columns)
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            st.stop()

        st.divider()
        st.subheader("Settings")
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            st.success("Anthropic API key loaded from .env")
        else:
            st.warning("No ANTHROPIC_API_KEY — Action Plan disabled")

        st.divider()
        st.markdown(
            "<div style='text-align:center;opacity:0.5;font-size:0.75rem'>"
            "Powered by FAISS + sentence-transformers<br>+ LangChain + Anthropic Claude"
            "</div>",
            unsafe_allow_html=True,
        )

    return df


# =============================================================================
# MAIN PAGE
# =============================================================================
def main():
    df = render_sidebar()

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center'>Wipfli DEI Executive Report</h1>"
        "<p style='text-align:center;opacity:0.6'>India Office (Pune / Bengaluru) "
        "&mdash; RAG-Powered Policy Alignment Analysis</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Build vector store & run analysis ────────────────────────────────
    store = build_vector_store()
    metrics = analyze_staff(df)
    findings = detect_imbalances(metrics)
    findings = retrieve_policies(store, findings, k=3)

    # ── KPI cards ────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Headcount", metrics["total"])
    k2.metric("Female %", f"{metrics['female_pct']}%",
              delta=f"{metrics['female_pct'] - 50}pp vs parity", delta_color="inverse")
    k3.metric("Women in Sr. Leadership", f"{metrics['leadership_f_pct']}%",
              delta=f"{metrics['leadership_f_pct'] - 40}pp vs 40% target", delta_color="inverse")
    k4.metric("Findings Detected", len(findings),
              delta=f"{sum(1 for f in findings if f['severity'] == 'HIGH')} HIGH severity",
              delta_color="inverse")
    st.divider()

    # ── Charts ───────────────────────────────────────────────────────────
    st.subheader("Workforce Snapshot")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_workforce_snapshot(metrics), use_container_width=True)
    with col_b:
        st.plotly_chart(chart_tenure_gap(metrics), use_container_width=True)

    st.subheader("Job Level Pipeline")
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(chart_pipeline(metrics), use_container_width=True)
    with col_d:
        st.plotly_chart(chart_pipeline_pct(metrics), use_container_width=True)

    st.divider()

    # ── Department data table ────────────────────────────────────────────
    st.subheader("Department Breakdown")
    dept_df = pd.DataFrame(metrics["dept_stats"])
    dept_display = dept_df.rename(columns={
        "department": "Department", "female": "Female", "male": "Male",
        "female_pct": "Female %", "tenure_f": "Avg Tenure (F)",
        "tenure_m": "Avg Tenure (M)", "tenure_gap": "Tenure Gap (yrs)",
    })
    st.dataframe(
        dept_display[["Department", "Female", "Male", "Female %",
                       "Avg Tenure (F)", "Avg Tenure (M)", "Tenure Gap (yrs)"]],
        use_container_width=True, hide_index=True,
    )
    st.divider()

    # ── RAG Findings (expander cards) ────────────────────────────────────
    st.subheader("Policy Alignment — RAG Analysis")
    st.caption(f"{len(findings)} findings from handbook cross-reference")

    for i, f in enumerate(findings, 1):
        sev = f.get("severity", "MEDIUM")
        icon = SEV_ICONS.get(sev, "")
        color = SEV_COLORS.get(sev, "#999")

        with st.expander(f"{icon} **Finding {i} [{sev}]** — {f['issue']}", expanded=(sev == "HIGH")):
            left, right = st.columns([1, 1])

            with left:
                st.markdown("##### Data Finding")
                st.markdown(f"**{f['issue']}**")
                st.markdown(f"**Severity:** {icon} {sev}")
                if "dept" in f:
                    ds = next((d for d in metrics["dept_stats"]
                               if d["department"] == f["dept"]), None)
                    if ds:
                        mini = pd.DataFrame([{
                            "Metric": "Female %", "Value": f"{ds['female_pct']}%"},
                            {"Metric": "Tenure Gap", "Value": f"{ds['tenure_gap']} yrs"
                             if ds["tenure_gap"] else "—"},
                            {"Metric": "Female Count", "Value": str(ds["female"])},
                            {"Metric": "Total", "Value": str(ds["total"])},
                        ])
                        st.dataframe(mini, use_container_width=True, hide_index=True)

            with right:
                st.markdown("##### Handbook Policy Alignment")
                for j, chunk in enumerate(f.get("policy_chunks", []), 1):
                    cleaned = " ".join(chunk.split())
                    if len(cleaned) > 400:
                        cleaned = cleaned[:397] + " …"
                    st.info(f"**Reference {j}:** {cleaned}")

            st.divider()
            verdict = synthesize_verdict(f)
            st.markdown(f"**Verdict:** {verdict}")

    st.divider()

    # ── Action Plan ──────────────────────────────────────────────────────
    st.subheader("Action Plan — Claude-Generated Policy Recommendations")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning(
            "Set `ANTHROPIC_API_KEY` in your `.env` file to enable "
            "Claude-generated policy recommendations."
        )
    else:
        if st.button("Generate Action Plan", type="primary", use_container_width=True):
            with st.spinner("Querying Claude for policy recommendations…"):
                actions = generate_action_plan(store, metrics, findings)
            if actions:
                for idx, action in enumerate(actions, 1):
                    st.success(f"**Recommendation {idx}:** {action}")
            else:
                st.error(
                    "Claude API call failed — check your account balance or API key."
                )

    # ── Footer ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='text-align:center;opacity:0.4;font-size:0.8rem;padding:1rem'>"
        "Generated by Wipfli DEI RAG Analyzer &bull; Data is synthetic &bull; "
        "Powered by FAISS + sentence-transformers + LangChain + Anthropic Claude"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
