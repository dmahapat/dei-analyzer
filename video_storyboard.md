# DEI Analyzer — 3-Minute Video Storyboard & Script

## PART 1: THE PRESENTATION (0:00 – 1:30)
**Tone: McKinsey-style — authoritative, data-forward, executive-ready**

---

### SCENE 1 — The Hook (0:00 – 0:20)
**Visual:** Black screen. White text fades in: *"63% to 12.5%"*. Pause. Then: *"Where did the women go?"*

**SCRIPT:**
> "Sixty-three percent. That's the share of women at the Associate level in Wipfli India's offices in Pune and Bengaluru. By the time you reach Director, that number is twelve point five percent. One woman out of eight. This is what workforce analytics calls a Leaky Pipeline — and spreadsheets alone will never tell you *why* it's leaking, or *what your own handbook says you should do about it*."

---

### SCENE 2 — The Problem with Spreadsheets (0:20 – 0:45)
**Visual:** Side-by-side. Left: a static Excel pivot table. Right: the Wipfli DEI Analyzer dashboard (screenshot of KPI cards + pipeline chart from `wipfli_app.py`).

**SCRIPT:**
> "A traditional approach gives you a table — counts by department, maybe a bar chart. It answers *what*. But it can't cross-reference your 100-person staff dataset against a 20-page DEI Handbook and tell you which specific policy commitments are being violated. That requires a system that *reads* your policy documents, *understands* the data, and *connects the two*. That's what Retrieval-Augmented Generation does."

---

### SCENE 3 — The RAG Architecture (0:45 – 1:15)
**Visual:** Clean architecture diagram showing the pipeline flow:

```
Wipfli DEI Handbook (.txt)
        |
  RecursiveCharacterTextSplitter (500-char chunks, 80 overlap)
        |
  all-MiniLM-L6-v2 Embeddings (HuggingFace, 384-dim)
        |
  FAISS Vector Store (similarity_search, k=3)
        |                                    Staff CSV (100 rows)
        |                                         |
        |                                  analyze_staff() → metrics
        |                                         |
        +---- detect_imbalances() ← ──────────────+
        |           |
        |    retrieve_policies() → matched handbook chunks
        |           |
        |    synthesize_verdict() → pattern-based alignment
        |           |
        +────→ Claude Sonnet 4 → 3 actionable policy recommendations
```

**SCRIPT:**
> "Here's how it works. The Wipfli DEI Handbook is chunked into 500-character segments using LangChain's recursive text splitter. Each chunk is embedded into a 384-dimensional vector using the MiniLM model and stored in a FAISS index. Separately, the staff CSV is analyzed — gender ratios by department, tenure gaps, the full Associate-to-Director pipeline. When an imbalance is detected — say, Consulting at only 28 percent female — the system generates a semantic query, retrieves the three most relevant handbook passages via FAISS similarity search, and produces a verdict. For the action plan, those findings are passed with retrieved policy context to Claude, which generates three concrete, handbook-grounded recommendations."

---

### SCENE 4 — The Punchline (1:15 – 1:30)
**Visual:** The four KPI metric cards from the dashboard. Zoom into: *"Findings Detected: 9 | 5 HIGH severity"*

**SCRIPT:**
> "The result: nine policy-alignment findings generated automatically, five rated HIGH severity, each traceable to a specific section of the handbook — the CEO Pledge, the Aparajita BRG, the Women of Wipfli initiative. Not opinions. Evidence. Let me show you."

---

## PART 2: THE WALKTHROUGH (1:30 – 3:00)
**Tone: Developer-to-Developer — practical, specific, code-aware**

---

### SCENE 5 — Sidebar & Data Load (1:30 – 1:45)
**Screen action:** Open browser to `localhost:8501`. Show the sidebar: Wipfli DEI Analyzer title, "Using default dataset (100 rows)" info badge, API key status.

**SCRIPT:**
> "Alright, let's walk through the app. It's a single-file Streamlit app — `wipfli_app.py`, about 560 lines. On first load, it builds the FAISS index — you'll see the spinner say 'Building FAISS vector store' the first time since it downloads the 80-meg MiniLM model. After that, Streamlit's `@st.cache_resource` keeps it in memory. The sidebar lets you upload any CSV with `gender`, `department`, `job_level`, and `years_at_firm` columns, or it falls back to the default synthetic dataset of 100 employees."

---

### SCENE 6 — The Charts (1:45 – 2:15)
**Screen action:** Scroll through the four Plotly charts in sequence. Hover over data points for tooltips.

**SCRIPT (Chart 1 — Gender Distribution by Department):**
> "First chart: a grouped Plotly bar — gender headcount across Audit, Consulting, IT, and Tax. Each department has 25 people but the split varies. Pink for female, blue for male — all rendered with `plotly_white` template for a clean, print-ready look."

**Screen action:** Pan to the tenure gap chart. Point to the red dashed threshold line.

**SCRIPT (Chart 2 — Tenure Gap):**
> "Next to it, the tenure gap chart. That red dashed line is the 3-year alert threshold defined in the code. Any department where male average tenure exceeds female by more than 3 years gets flagged as a retention concern."

**Screen action:** Scroll to the pipeline charts. Hover over the stacked bar, then the percentage bar.

**SCRIPT (Charts 3 & 4 — Pipeline):**
> "Now the pipeline view — this is where the story lives. The stacked bar on the left shows raw headcount: Associate through Director. The bar on the right is the one that matters — female percentage by level. Watch the colors: green at 63 percent for Associate, then yellow, then red at 12.5 percent for Director. That green dashed line at 40 percent is the handbook's own target. Only the entry level meets it. The color coding is generated dynamically — red below 20, yellow below 35, green above."

---

### SCENE 7 — RAG Findings (2:15 – 2:45)
**Screen action:** Scroll to "Policy Alignment — RAG Analysis." Expand a HIGH-severity finding. Show the two-column layout: data on the left, handbook references on the right.

**SCRIPT:**
> "Below the charts, the RAG findings. Each one is an expander — HIGH severity ones open by default. Left column: the data finding with a mini stats table — female percentage, tenure gap, headcount. Right column: the three handbook chunks retrieved by FAISS. These are real passages from the DEI Handbook — references to the Aparajita BRG, the CEO Pledge, the WoW Leadership Accelerator. Below both columns, the verdict — synthesized by pattern-matching the retrieved text against the issue type. No hallucination here; the verdict function is deterministic, not LLM-generated."

---

### SCENE 8 — Action Plan & Close (2:45 – 3:00)
**Screen action:** Click "Generate Action Plan" button. Show spinner, then the three green recommendation boxes appearing.

**SCRIPT:**
> "Finally, hit 'Generate Action Plan.' This fires a single call to Claude Sonnet — the prompt includes the five retrieved promotion-policy chunks and the HIGH-severity findings as structured context. Claude returns three numbered recommendations, each referencing a specific handbook section and data gap. The whole pipeline — FAISS retrieval to Claude response — is one click. That's the DEI Analyzer: synthetic data, real architecture, actionable output."

---

## PRODUCTION NOTES

| Element | Detail |
|---|---|
| **Total runtime** | 3:00 (Part 1: 1:30 / Part 2: 1:30) |
| **Part 1 visuals** | Slides or motion graphics (Canva/Figma). Architecture diagram, data callouts, KPI screenshot. |
| **Part 2 visuals** | Screen recording of `localhost:8501` (OBS/Loom). Cursor movements scripted above. |
| **Transition** | At 1:30, cut from the last slide to the browser. Use a brief fade or a line like *"Let me show you."* |
| **Key data points to emphasize** | 63% -> 12.5% pipeline drop, 18.2% women in leadership (vs 40% target), 9 findings / 5 HIGH |
| **Tech stack callouts** | FAISS, all-MiniLM-L6-v2, LangChain RecursiveCharacterTextSplitter, Plotly, Streamlit `@st.cache_resource`, Anthropic Claude Sonnet |
