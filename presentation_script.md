# DEI Analyzer — Presentation Script & Slide Content

---

## SECTION 1: EXECUTIVE SUMMARY

### Slide Title: *What Is the DEI Analyzer?*

**Elevator Pitch (2 sentences):**

The DEI Analyzer is a Retrieval-Augmented Generation pipeline that automatically cross-references workforce demographic data against an organization's internal DEI policy handbook, surfacing compliance gaps that manual auditing routinely misses. Applied to Wipfli India's 100-person staff dataset, it identified a severe "Leaky Pipeline" — female representation collapsing from 63% at Associate level to 12.5% at Director — and traced each gap to specific, violated handbook commitments, producing an auditable, evidence-based executive report in seconds.

### Speaker Notes:

> This is not a dashboard that shows you charts and leaves interpretation to the reader. It is an analytical system that does three things no spreadsheet can: it reads your policy documents, it quantifies where the data violates those policies, and it tells you exactly which commitments are broken — with the handbook passages to prove it.
>
> The output is a structured executive report: KPI metrics, interactive visualizations, nine policy-alignment findings rated by severity, and three Claude-generated action items — each grounded in retrieved handbook language, not generalized advice.

---

## SECTION 2: THE PROBLEM STATEMENT

### Slide Title: *The Leaky Pipeline — And Why You Can't Find It in a Spreadsheet*

### Sub-slide 2A: The Data Story

**Slide Content:**

| Job Level        | Female | Male | Total | Female % | Status        |
|------------------|--------|------|-------|----------|---------------|
| Associate        | 17     | 10   | 27    | 63.0%    | Above target  |
| Senior Associate | 11     | 18   | 29    | 37.9%    | Below target  |
| Manager          | 6      | 16   | 22    | 27.3%    | Critical      |
| Senior Manager   | 3      | 11   | 14    | 21.4%    | Critical      |
| Director         | 1      | 7    | 8     | 12.5%    | Severe        |

**Key callout:** *"A 50.5 percentage-point collapse across five levels. The pipeline doesn't just leak — it hemorrhages."*

### Speaker Notes:

> At the entry level, Wipfli India exceeds its own 40% female representation target. The organization appears healthy on the surface. But promotion-level data reveals a different reality: a monotonic decline in female representation at every successive job level, culminating in a single woman among eight Directors.
>
> This pattern — the Leaky Pipeline — is one of the most well-documented phenomena in organizational DEI research. Women enter in strong numbers, but exit or stall disproportionately before reaching leadership. The critical question is not whether the pipeline leaks. It is whether the organization's own policies address the leak, and whether those policies are being enforced. That is the question this tool answers.

### Sub-slide 2B: Why Manual Auditing Fails

**Slide Content (3 bullet points):**

1. **Scale blindness.** A 270-line DEI handbook contains commitments scattered across six sections — CEO Pledge, BRG charters (Aparajita, Women of Wipfli, Embrace), governance frameworks, and India-specific focus areas. No analyst cross-references every finding against every relevant passage. They spot-check. The DEI Analyzer performs exhaustive semantic retrieval across every chunk of the document for every detected imbalance.

2. **Confirmation bias in framing.** When an HR team builds its own audit, it tends to frame findings around metrics that look favorable. An overall 38% female workforce sounds close to parity. The pipeline breakdown — which reveals the structural failure — requires deliberate, level-by-level disaggregation that manual processes often skip. The analyzer disaggregates automatically across departments, job levels, and tenure bands.

3. **No policy traceability.** A typical audit says "female representation in IT is low." It does not say: "IT's 24% female ratio violates the India Office DEI Focus Areas (Section 5), which set a target of 40% women in IT and Consulting at mid-senior levels by 2027, enforceable under the Aparajita BRG's Gender Equity initiative (Section 3.5)." The RAG pipeline produces that citation automatically.

### Speaker Notes:

> The fundamental limitation of manual DEI auditing is not effort — it is cognition. A 270-line policy document and a 100-row staff dataset produce a combinatorial space of possible cross-references that no human reviewer will exhaust. The analyzer does. It generates a semantic query for every detected imbalance, retrieves the top-k most relevant policy passages, and synthesizes a verdict that maps the data gap to the specific handbook commitment it violates. This is not automation for efficiency. It is automation for rigor.

---

## SECTION 3: THE TECH STACK

### Slide Title: *Architecture — What Powers the Analyzer*

**Slide Content (layered diagram or four-quadrant layout):**

### Layer 1: Data Manipulation — Python & Pandas

- **Role:** Ingests, cleans, and analyzes the staff CSV (100 rows, 5 columns: `employee_id`, `gender`, `department`, `job_level`, `years_at_firm`).
- **Operations:** GroupBy aggregations across department and job level. Computes female representation percentages, average tenure by gender, tenure gap (male mean minus female mean), and leadership composition (Senior Manager + Director).
- **Output:** A structured metrics dictionary consumed by the detection engine and charting layer.
- **Design decision:** Analysis runs on every page load rather than from pre-computed caches, ensuring the dashboard always reflects the uploaded dataset — critical for a tool designed to accept arbitrary staff CSVs via the sidebar uploader.

### Layer 2: Vector Similarity Search — FAISS + HuggingFace Embeddings

- **Role:** Converts the DEI Handbook into a searchable vector index for semantic retrieval.
- **Embedding model:** `all-MiniLM-L6-v2` (HuggingFace sentence-transformers). 384-dimensional embeddings, ~80 MB model, runs on CPU — no GPU required.
- **Chunking strategy:** LangChain's `RecursiveCharacterTextSplitter` with 500-character windows and 80-character overlap. Separators: `\n\n`, `\n`, `. `, ` `, `""`. This preserves paragraph boundaries while ensuring no policy clause is split mid-sentence.
- **Retrieval:** `similarity_search(query, k=3)` for per-finding policy alignment. `similarity_search(query, k=5)` for the action-plan prompt, which requires broader policy context.
- **Caching:** The FAISS index is built once per session using Streamlit's `@st.cache_resource` decorator. Subsequent interactions skip re-indexing entirely.

### Layer 3: Reasoning & Policy Alignment — Anthropic Claude (Sonnet)

- **Role:** Generates the three-item Action Plan — concrete, handbook-grounded policy recommendations.
- **How it's used:** This is not a chatbot. Claude receives a single structured prompt containing: (a) five retrieved handbook chunks focused on promotion and advancement policies, and (b) a data summary of all HIGH-severity findings. It returns exactly three numbered recommendations, each required to reference a specific handbook commitment, name the data gap it addresses, and be implementable within one quarter.
- **What Claude does NOT do:** The per-finding verdicts ("This violates Section 3.5...") are generated deterministically by the `synthesize_verdict()` function using pattern matching against retrieved policy text. This is a deliberate design choice — verdicts must be reproducible and auditable, not probabilistic. Claude is reserved for the generative task (action planning) where reasoning over multiple policy excerpts adds genuine value.

### Layer 4: Interactive Frontend — Streamlit + Plotly

- **Role:** Renders the executive report as an interactive, wide-layout web dashboard.
- **Layout:** Sidebar (data source selector, API status) → KPI metric cards (4 columns) → Plotly charts (2x2 grid) → Department breakdown table → RAG findings (expandable cards, two-column layout) → Action Plan (button-triggered).
- **Charts (Plotly `graph_objects`):**
  - Gender Distribution by Department — grouped bar (`go.Bar`, barmode="group")
  - Average Tenure by Gender — grouped bar with red dashed alert threshold at 3.0 years
  - Job Level Pipeline — stacked bar showing raw headcount composition
  - Female % by Job Level — single bar with dynamic color coding (red < 20%, yellow < 35%, green >= 35%) and a green dashed line at the 40% handbook target
- **Interactivity:** Plotly tooltips on hover, expandable finding cards (`st.expander`), file upload for custom datasets, one-click action plan generation with loading spinner.

### Speaker Notes:

> A note on the architectural separation. There are two distinct reasoning engines in this system. The verdict engine is rule-based: it pattern-matches retrieved policy text to produce deterministic, reproducible citations. The action-plan engine uses Claude for genuine generation — synthesizing recommendations that require reasoning across multiple policy excerpts and data points simultaneously. This separation is intentional. Audit findings must be reproducible; action plans benefit from generative reasoning. Mixing the two would compromise the auditability of the findings.

---

## SECTION 4: HOW IT WORKS — THE RAG PIPELINE

### Slide Title: *From Raw Data to Auditable Findings in Five Steps*

---

### Step 1: Ingestion

**Slide Content:**

```
Wipfli_DEI_Handbook.txt (270 lines)
            │
            ▼
  RecursiveCharacterTextSplitter
  ┌──────────────────────────────┐
  │ chunk_size    = 500 chars    │
  │ chunk_overlap = 80 chars     │
  │ separators    = ¶, \n, ., ␣ │
  └──────────────────────────────┘
            │
            ▼
      ~30-40 text chunks
```

**Speaker Notes:**

> The handbook is loaded as raw text and split into overlapping 500-character segments. The 80-character overlap ensures that policy clauses spanning a chunk boundary are captured in at least one complete chunk. The separator hierarchy — paragraph break first, then line break, then sentence boundary, then word boundary — preserves semantic coherence. A clause like "reduce attrition among women associates in years 3–5 of tenure" will not be split across chunks.

---

### Step 2: Vector Indexing

**Slide Content:**

```
      ~30-40 text chunks
            │
            ▼
  all-MiniLM-L6-v2 Encoder
  (384-dimensional embeddings)
            │
            ▼
  ┌───────────────────────┐
  │     FAISS Index        │
  │  (L2 distance search)  │
  │  ~30-40 vectors        │
  └───────────────────────┘
```

**Speaker Notes:**

> Each chunk is encoded into a 384-dimensional dense vector using the MiniLM model. These vectors are stored in a FAISS flat index optimized for L2 (Euclidean) distance search. At this scale — roughly 30 to 40 vectors — FAISS operates in sub-millisecond lookup time. The index is built once and cached for the duration of the Streamlit session. On first run, the model download is approximately 80 MB; subsequent runs load from the local HuggingFace cache.

---

### Step 3: Data Anomaly Detection

**Slide Content:**

```
  wipfli_staff_data.csv (100 rows)
            │
            ▼
      analyze_staff()
  ┌─────────────────────────────────┐
  │ • Gender ratio per department   │
  │ • Tenure gap (M avg − F avg)    │
  │ • Female % per job level        │
  │ • Leadership composition        │
  └─────────────────────────────────┘
            │
            ▼
    detect_imbalances()
  ┌─────────────────────────────────┐
  │ Thresholds:                     │
  │   < 30% female → HIGH           │
  │   < 40% female → MEDIUM         │
  │   tenure gap > 3yr → MEDIUM     │
  │   tenure gap > 4yr → HIGH       │
  │   pipeline drop > 30pp → HIGH   │
  │   leadership < 30% → HIGH       │
  └─────────────────────────────────┘
            │
            ▼
    9 findings (5 HIGH, 4 MEDIUM)
```

**Speaker Notes:**

> The detection engine applies threshold-based rules derived from the handbook's own targets. The 40% female target comes from the India Office DEI Focus Areas. The 3-year tenure gap threshold flags retention concerns — departments where women leave or are hired later than men. The pipeline narrowing threshold triggers when the female percentage drops by more than 30 points from the entry level to the top level. In the Wipfli India dataset, the drop is 50.5 points — from 63% at Associate to 12.5% at Director — well above the threshold.
>
> Each finding carries a severity rating and a pre-built semantic query. The query is not the finding itself — it is a purpose-built search string designed to retrieve the most relevant policy passages. For example, an IT department gender gap generates the query: "women representation IT department India office gender equity target." This query formulation step is critical to retrieval quality.

---

### Step 4: Policy Retrieval (RAG)

**Slide Content:**

```
    9 findings (each with a semantic query)
            │
            ▼
    FAISS.similarity_search(query, k=3)
            │
            ▼
  ┌──────────────────────────────────────────┐
  │ Finding: "IT has only 24% female"        │
  │                                          │
  │ Retrieved Chunk 1: "...Aparajita BRG     │
  │   Gender Equity initiative includes      │
  │   women-in-tech mentoring..."            │
  │                                          │
  │ Retrieved Chunk 2: "...India Office DEI  │
  │   Focus Areas set a target of 40%        │
  │   women in IT and Consulting..."         │
  │                                          │
  │ Retrieved Chunk 3: "...CEO pledges       │
  │   annual bias reviews of promotion       │
  │   processes..."                          │
  └──────────────────────────────────────────┘
```

**Speaker Notes:**

> For each of the nine findings, the system performs a FAISS similarity search with k=3, returning the three handbook passages whose embeddings are closest to the finding's semantic query. This is the core of the RAG pattern: the retrieval step ensures that every verdict and recommendation is grounded in actual policy language rather than parametric knowledge.
>
> The retrieved chunks typically span multiple handbook sections — a finding about IT representation might pull from the Aparajita BRG charter, the India Office Focus Areas, and the CEO Pledge simultaneously. This cross-section retrieval is precisely what manual auditing fails to do consistently.

---

### Step 5: Verdict Synthesis & Action Plan

**Slide Content:**

```
  Retrieved policy chunks + finding metadata
            │
            ├──→ synthesize_verdict()  [DETERMINISTIC]
            │     • Pattern-matches policy text
            │     • Identifies handbook sections (CEO Pledge,
            │       Aparajita BRG, WoW, Embrace, Governance)
            │     • Produces traceable, reproducible citation
            │
            └──→ generate_action_plan()  [GENERATIVE — Claude]
                  • Retrieves k=5 promotion-policy chunks
                  • Sends structured prompt with data + policy context
                  • Returns 3 recommendations, each citing a
                    handbook commitment and a data gap
```

**Verdict example (deterministic):**
> *"IT's low female ratio violates the targets set in Aparajita BRG (Section 3.5), CEO Pledge (Section 2). The handbook's India Office DEI Focus Areas explicitly set a target of 40% women in IT and Consulting at mid-senior levels by 2027. The Aparajita BRG's Gender Equity initiative includes women-in-tech mentoring for this purpose."*

**Action Plan example (Claude-generated):**
> *"Mandate that all Senior Manager-to-Director promotion panels include at least one Aparajita BRG representative as a voting member, enforcing the CEO Pledge commitment to 'annual bias reviews of promotion processes' (Section 2, Item 3). This directly addresses the 12.5% female Director representation, against the handbook's 40%-by-2027 target."*

### Speaker Notes:

> The verdict function scans the retrieved policy text for named entities — "Aparajita," "Women of Wipfli," "CEO," "governance" — and maps them to known handbook sections. It then selects a verdict template based on the finding type: tenure gap, leadership gap, pipeline narrowing, or department-level representation. The output is fully deterministic: given the same data and the same retrieved chunks, it will produce the same verdict every time. This is essential for audit credibility.
>
> The action plan is the one component that uses Claude generatively. The prompt is tightly constrained: it includes the five most relevant promotion-policy chunks (retrieved via a separate FAISS query), the full set of HIGH-severity findings, and explicit instructions to produce exactly three recommendations, each grounded in a specific handbook commitment. This is not open-ended generation — it is structured synthesis under retrieval constraints.

---

## APPENDIX: KEY METRICS REFERENCE

| Metric | Value | Source |
|---|---|---|
| Total headcount | 100 | `wipfli_staff_data.csv` |
| Overall female % | 38.0% | Across all levels |
| Female % at Associate | 63.0% | 17 of 27 |
| Female % at Director | 12.5% | 1 of 8 |
| Pipeline drop | 50.5 pp | Associate → Director |
| Women in senior leadership | 18.2% | 4 of 22 (Sr. Manager + Director) |
| Handbook target | 40% | India Office, IT & Consulting, by 2027 |
| IT department female % | 24.0% | 6 of 25 — lowest department |
| Largest tenure gap | 5.5 years | Audit department |
| Findings generated | 9 | 5 HIGH, 4 MEDIUM |
| Embedding model | all-MiniLM-L6-v2 | 384 dimensions, ~80 MB |
| Chunk configuration | 500 chars / 80 overlap | RecursiveCharacterTextSplitter |
| LLM | Claude Sonnet | Action plan generation only |
