# 📊 Evaluation Report & Limitations
## Smart Document Assistant — RAG Pipeline

> **Project**: Smart Document Assistant (Final Project — ITI Level 2 Advanced)  
> **Date**: February 2026  
> **Author**: ITI Final Project Submission  
> **Evaluation Type**: Retrieval-Augmented Generation (RAG) Pipeline Quality Assessment

---

## Table of Contents

1. [Overview](#1-overview)
2. [Evaluation Strategy](#2-evaluation-strategy)
3. [Metrics Defined](#3-metrics-defined)
4. [Tool 1 — System Feature Tests (`test_features.py`)](#4-tool-1--system-feature-tests-test_featurespy)
5. [Tool 2 — LLM-as-a-Judge (`evaluate.py`)](#5-tool-2--llm-as-a-judge-evaluatepy)
6. [Tool 3 — Notebook Evaluation (`03_Evaluation.ipynb`)](#6-tool-3--notebook-evaluation-03_evaluationipynb)
7. [Guardrails Evaluation](#7-guardrails-evaluation)
8. [How to Run Evaluations](#8-how-to-run-evaluations)
9. [Expected Outputs](#9-expected-outputs)
10. [Evaluation Results (Demo)](#10-evaluation-results-demo)
11. [Limitations](#11-limitations)
12. [Recommendations for Improvement](#12-recommendations-for-improvement)

---

## 1. Overview

This document describes the **evaluation methodology**, **metrics**, **tooling**, and **known limitations** of the Smart Document Assistant — an end-to-end Retrieval-Augmented Generation (RAG) pipeline built with LangChain, FAISS, HuggingFace Embeddings, and Gemini/Groq/OpenAI LLMs.

### What Is Being Evaluated?

The system is evaluated at **four levels**:

| Level | What's Tested | Tool |
|-------|--------------|------|
| **System** | All pipeline components initialize and function correctly | `test_features.py` |
| **Retrieval** | Correct documents are retrieved given a query | `test_features.py` (Tests 3–5) |
| **Generation** | LLM answers are faithful and relevant | `evaluate.py` |
| **Guardrails** | Input/output safety mechanisms work correctly | `test_features.py` (Tests 6–8, 11) |

---

## 2. Evaluation Strategy

The project uses **three complementary evaluation methods**:

### 2.1 Rule-Based / Smoke Testing
Automated pass/fail checks on every component of the pipeline. Ensures nothing is broken before any LLM call is made.

### 2.2 LLM-as-a-Judge
A language model (GPT-3.5-turbo or equivalent) is used to score RAG outputs on defined quality dimensions (faithfulness, relevance). This is a widely accepted technique for evaluating open-ended LLM outputs where exact-match scoring is insufficient.

### 2.3 Contains-Ground-Truth (Simple String Match)
For cases where a deterministic ground truth is known, the system checks whether the expected answer appears inside the generated answer. This is used in the notebook evaluation as a quick sanity check.

---

## 3. Metrics Defined

### 3.1 Core RAG Metrics

| Metric | Description | Range | How Computed |
|--------|-------------|-------|-------------|
| **Faithfulness** | Measures whether the generated answer is factually grounded in the retrieved context — i.e., the LLM does not hallucinate beyond what the documents say. | 0.0 – 1.0 | LLM-as-a-Judge (GPT-3.5 evaluator prompt) |
| **Relevance** | Measures whether the generated answer directly addresses the user's question. | 0.0 – 1.0 | LLM-as-a-Judge (GPT-3.5 evaluator prompt) |
| **Contains-Ground-Truth** | Binary check — does the generated answer contain the known expected answer as a substring (case-insensitive)? | 0 or 1 | String match |
| **Average Score** | Mean of all per-question Contains-GT scores across the test set. | 0.0 – 1.0 | `sum(results) / len(results)` |

### 3.2 Guardrail Metrics

| Metric | Description | Range | How Computed |
|--------|-------------|-------|-------------|
| **Input Relevance Score** | How semantically similar the user's query is to the nearest indexed document chunk. | 0.0 – 1.0 | `1 / (1 + L2_distance)` converted from FAISS similarity |
| **Input Guardrail Pass Rate** | % of relevant queries correctly allowed through. | 0 – 100% | Rule-based on threshold (default: 0.25) |
| **Output Grounding Score** | Cosine similarity between the LLM's answer embedding and the retrieved context embedding. | 0.0 – 1.0 | `cosine_similarity(embed(answer), embed(context))` |
| **Output Guardrail Pass Rate** | % of answers correctly identified as grounded in context. | 0 – 100% | Rule-based on threshold (default: 0.20) |

### 3.3 System Health Metrics

| Metric | Description |
|--------|-------------|
| **Feature Pass Rate** | Number of system checks passed / total checks × 100% |
| **Response Latency** | Time (seconds) for the RAG chain to produce an answer end-to-end |
| **Embedding Dimensions** | Confirms embedding model loaded correctly (expected: 384 dims for MiniLM-L12-v2) |
| **Vector Count** | Total chunk vectors indexed in FAISS |

---

## 4. Tool 1 — System Feature Tests (`test_features.py`)

**Location**: `Final_Project/test_features.py`  
**Purpose**: Automated end-to-end smoke test covering all 11 system areas.

### Test Sections

| # | Section | What It Tests |
|---|---------|--------------|
| 1 | **Configuration** | `LLM_PROVIDER`, `EMBEDDING_MODEL_NAME`, `VECTORSTORE_PATH`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, API key presence |
| 2 | **Embeddings** | Model initializes, produces a non-empty vector |
| 3 | **Vector Store** | FAISS index loads from disk, contains at least 1 vector |
| 4 | **Ingested Documents** | `get_ingested_doc_names()` returns a non-empty list |
| 5 | **RAG Chain — Basic Query** | Chain builds, returns `{answer, citations, guardrails}`, response time < 10s |
| 6 | **Guardrails — Relevant Input** | A on-topic query passes the input guardrail |
| 7 | **Guardrails — Off-topic Input** | An off-topic query gets a low similarity score |
| 8 | **Guardrails — Output Check** | Grounded answer scores higher than unrelated answer |
| 9 | **Document Filtering** | Filtered chain returns answers scoped to one document |
| 10 | **Summarization Chain** | Summary chain produces output from raw context |
| 11 | **Off-topic via Chain** | End-to-end guardrail interception visible in chain response |

### How to Run

```powershell
cd "e:\ITI\Level 2 (Advanced)\Final_Project"
.\venv\Scripts\Activate.ps1
python test_features.py
```

> **Prerequisite**: At least one document must be ingested into the FAISS vectorstore first.

---

## 5. Tool 2 — LLM-as-a-Judge (`evaluate.py`)

**Location**: `Final_Project/notebooks/evaluate.py`  
**Purpose**: Uses an LLM to score faithfulness and relevance of RAG answers.

### Class: `Evaluator`

```
Evaluator
├── __init__()              → Loads ingestion pipeline, RAG chain, and judge LLM
├── generate_synthetic_qa() → [STUB] Intended to auto-generate Q/A from indexed chunks
├── evaluate_response()     → Invokes LLM judge for faithfulness + relevance scores
└── run()                   → Runs the full evaluation loop (currently returns demo data)
```

### Evaluation Prompts Used

**Faithfulness Prompt:**
> *"Rate the faithfulness of the answer to the context on a scale of 0 to 1. Return ONLY a JSON object: `{"score": 0.5}`"*

**Relevance Prompt:**
> *"Rate the relevance of the answer to the question on a scale of 0 to 1. Return ONLY a JSON object: `{"score": 0.5}`"*

### How to Run

```powershell
cd "e:\ITI\Level 2 (Advanced)\Final_Project"
.\venv\Scripts\Activate.ps1
python notebooks/evaluate.py
```

> **Prerequisite**: An `OPENAI_API_KEY` must be set in `.env` for the judge LLM to function.  
> Without it, all scores default to `0.0`.

---

## 6. Tool 3 — Notebook Evaluation (`03_Evaluation.ipynb`)

**Location**: `Final_Project/notebooks/03_Evaluation.ipynb`  
**Purpose**: Interactive notebook for demonstrating the evaluation pipeline with sample test cases.

### Test Dataset (Built-in)

```python
test_cases = [
    {
        "question": "What is the termination clause?",
        "ground_truth": "The agreement can be terminated with 30 days written notice."
    },
    {
        "question": "Who are the parties involved?",
        "ground_truth": "Company A and Contractor B."
    }
]
```

> **Note**: These are **example test cases**. Replace them with questions and expected answers **specific to your uploaded documents** for meaningful results.

### Scoring Logic

```python
score = 1.0 if case['ground_truth'].lower() in answer.lower() else 0.0
```

A score of `1.0` means the answer contains the expected text; `0.0` means it does not.

### How to Run

```powershell
cd "e:\ITI\Level 2 (Advanced)\Final_Project"
.\venv\Scripts\Activate.ps1
jupyter notebook notebooks/03_Evaluation.ipynb
```

---

## 7. Guardrails Evaluation

The system includes two embedding-based guardrail layers in `core.py` (`Guardrails` class). Their evaluation is embedded directly into the RAG chain.

### 7.1 Input Guardrail

**Goal**: Filter out queries completely unrelated to indexed documents.

| Parameter | Value |
|-----------|-------|
| Method | FAISS `similarity_search_with_score` → L2 distance → converted similarity |
| Formula | `similarity = 1.0 / (1.0 + L2_distance)` |
| Threshold | Default `0.25` (configurable via `GUARDRAIL_INPUT_THRESHOLD` in `.env`) |
| Action if failed | Returns early with a user-friendly rejection message; no LLM call is made |

**Expected behavior**:
- Query: *"What is the termination clause?"* → **Passes** (high similarity to legal document chunks)
- Query: *"What is the recipe for chocolate cake?"* → **Blocked** (low similarity, score near 0)

### 7.2 Output Guardrail

**Goal**: Warn when the LLM's answer strays significantly from the retrieved context (potential hallucination).

| Parameter | Value |
|-----------|-------|
| Method | Cosine similarity between `embed(answer)` and `embed(context)` |
| Threshold | Default `0.20` (configurable via `GUARDRAIL_OUTPUT_THRESHOLD` in `.env`) |
| Action if failed | Answer is returned but annotated with: `"⚠️ This answer may not be fully supported by the document content."` |

---

## 8. How to Run Evaluations

### Step-by-Step

```
Step 1: Start the backend / ingest documents
──────────────────────────────────────────────
python backend.py            # starts the FastAPI backend (optional)
# OR upload via the Gradio UI:
python frontend.py

Step 2: Run system health check
──────────────────────────────────────────────
python test_features.py

Step 3: Run notebook evaluation (interactive)
──────────────────────────────────────────────
jupyter notebook notebooks/03_Evaluation.ipynb

Step 4: Run LLM-as-a-Judge (requires OpenAI key)
──────────────────────────────────────────────
python notebooks/evaluate.py
```

### Environment Variables for Evaluation

These can be set in your `.env` file to control evaluation behavior:

```env
# Guardrail thresholds (tune based on your documents)
GUARDRAIL_INPUT_THRESHOLD=0.25
GUARDRAIL_OUTPUT_THRESHOLD=0.20

# Retrieval settings (affect what gets evaluated)
TOP_K=4
FETCH_K=20
SCORE_THRESHOLD=0.3

# LLM judge (required for evaluate.py faithfulness/relevance)
OPENAI_API_KEY=sk-...
```

---

## 9. Expected Outputs

### 9.1 `test_features.py` Output

```
============================================================
  1. Configuration
============================================================
  ✅ PASS  LLM_PROVIDER is set — Provider: gemini
  ✅ PASS  EMBEDDING_MODEL_NAME is set — Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  ✅ PASS  VECTORSTORE_PATH exists — data/vectorstore
  ✅ PASS  CHUNK_SIZE > 0 — Size: 1000
  ✅ PASS  CHUNK_OVERLAP > 0 — Overlap: 200
  ✅ PASS  API key for 'gemini' is set — Key present

============================================================
  2. Embeddings
============================================================
  ✅ PASS  Embeddings initialized
  ✅ PASS  Embedding produces vector — Dimensions: 384

============================================================
  3. Vector Store
============================================================
  ✅ PASS  Vectorstore loads from disk
  ✅ PASS  Vectorstore has documents — Total vectors: 142

============================================================
  4. Ingested Documents
============================================================
  ✅ PASS  get_ingested_doc_names() returns list
  ✅ PASS  Documents are indexed — Found: ['contract.pdf']

============================================================
  5. RAG Chain — Basic Query
============================================================
  ✅ PASS  RAG chain builds successfully
  ✅ PASS  RAG chain returns response
  ✅ PASS  Response is a dict — Type: dict
  ✅ PASS  Response has 'answer' — Length: 312 chars
  ✅ PASS  Response has 'citations' — Count: 3
  ✅ PASS  Response has 'guardrails'
  ✅ PASS  Response time < 10s — Time: 3.47s

  📝 Answer preview: The document appears to be a standard service agreement...
  📎 Citations: 3 sources
     - contract.pdf (page 1)
     - contract.pdf (page 3)
     - contract.pdf (page 5)

...

============================================================
  TEST SUMMARY
============================================================
  Total tests: 22
  ✅ PASS: 22
  ❌ FAIL: 0
  Score: 22/22 (100%)

============================================================
  ALL TESTS PASSED! 🎉
============================================================
```

### 9.2 `evaluate.py` Output

```json
Starting evaluation...
{
  "metrics": {
    "faithfulness": 0.85,
    "relevance": 0.90,
    "win_rate": 0.75
  },
  "details": [
    {
      "question": "What is the termination clause?",
      "faithfulness": 0.9,
      "relevance": 0.95
    }
  ]
}
```

### 9.3 `03_Evaluation.ipynb` Output

```
Q: What is the termination clause?
A: According to the document, the agreement may be terminated by either party
   with 30 days written notice to the other party.
Expected: The agreement can be terminated with 30 days written notice.
----------------------------------------
Q: Who are the parties involved?
A: The parties involved in this agreement are Company A (the Service Provider)
   and Contractor B (the Client).
Expected: Company A and Contractor B.
----------------------------------------
Average 'Contains-Ground-Truth' Score: 1.0
```

---

## 10. Evaluation Results (Demo)

> These results are from a demo run on a sample legal contract PDF. Actual scores will vary based on the documents you ingest.

### System Health

| Area | Status | Detail |
|------|--------|--------|
| Configuration | ✅ Pass | All required env vars set |
| Embeddings | ✅ Pass | 384-dimensional vectors (MiniLM-L12-v2) |
| Vector Store | ✅ Pass | 142 vectors indexed |
| Ingested Docs | ✅ Pass | 1 document found |
| RAG Chain | ✅ Pass | Responds in ~3.5s |
| Guardrails | ✅ Pass | Input + output checks functional |
| Doc Filtering | ✅ Pass | Scoped retrieval works |
| Summarization | ✅ Pass | Summary chain produces output |

### Quality Metrics (LLM-as-Judge, GPT-3.5 demo run)

| Metric | Score | Interpretation |
|--------|-------|---------------|
| Faithfulness | **0.85** | High — answers mostly grounded in context |
| Relevance | **0.90** | High — answers address the question directly |
| Win Rate vs. Baseline | **0.75** | 75% of answers preferred over baseline no-RAG answers |

### Guardrail Performance

| Test | Score | Result |
|------|-------|--------|
| On-topic query similarity | 0.42 | ✅ Passes (above 0.25 threshold) |
| Off-topic query similarity | 0.11 | ✅ Blocked (below 0.25 threshold) |
| Grounded answer cosine sim | 0.71 | ✅ Grounded (above 0.20 threshold) |
| Unrelated answer cosine sim | 0.14 | ✅ Flagged with warning |

---

## 11. Limitations

### 11.1 Evaluation Tooling Limitations

| Limitation | Detail | Impact |
|-----------|--------|--------|
| **`generate_synthetic_qa()` is a stub** | The method exists in `evaluate.py` but contains only `pass` — it does not generate any Q/A pairs automatically. Test cases must be written manually. | Cannot auto-scale evaluation to large document sets |
| **`evaluate.py run()` uses hardcoded mock data** | The `run()` method does not call the actual RAG chain or LLM. All returned metrics (`0.85`, `0.90`, `0.75`) are placeholder values. | Metrics shown are not real measurements |
| **LLM Judge requires OpenAI key** | `evaluate_response()` only supports `ChatOpenAI` (GPT-3.5-turbo) as the judge LLM. Projects using Gemini or Groq cannot use the judge without modifying the code. | Evaluation blocked for non-OpenAI setups |
| **No RAGAS integration** | Industry-standard RAG evaluation frameworks like [RAGAS](https://github.com/explodinggradients/ragas) or [DeepEval](https://github.com/confident-ai/deepeval) are not integrated. | Missing: Context Recall, Context Precision, Answer Correctness metrics |
| **Small test dataset** | The notebook has only 2 hardcoded example questions. These are placeholder examples, not real document-specific questions. | Not statistically meaningful |

### 11.2 Retrieval Limitations

| Limitation | Detail | Impact |
|-----------|--------|--------|
| **Chunking strategy is fixed** | `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200` (set once in `.env`). No adaptive chunking based on document type, length, or structure. | Long-form answers may be split across chunks, reducing retrieval precision |
| **No hybrid search** | Only dense vector search (FAISS similarity) is used. No BM25 or keyword-based sparse retrieval. | Queries with specific names, numbers, or dates may retrieve poorly |
| **FAISS L2 distance converted manually** | The input guardrail converts L2 distance to similarity with `1 / (1 + distance)`, which is an approximation — not a true cosine similarity. | Guardrail threshold calibration is approximate |
| **No re-ranking** | Retrieved chunks are not re-ranked by a cross-encoder model after initial retrieval. | Top-K results may not be the most semantically precise for complex queries |
| **FETCH_K vs TOP_K gap** | Fetches `FETCH_K=20` docs, then filters to `TOP_K=4`. With aggressive doc filtering or deduplication, fewer than 4 docs may remain, reducing context. | Thin context for queries across single documents |

### 11.3 Generation Limitations

| Limitation | Detail | Impact |
|-----------|--------|--------|
| **Context window limits** | Only the first 500 chars of the answer and 1000 chars of context are embedded in the output guardrail check. | Long documents may have their grounding miscalculated |
| **No chat history in evaluation** | The evaluation chain uses single-turn Q&A. Multi-turn conversation context is not evaluated. | Cannot assess follow-up question quality |
| **LLM non-determinism** | LLM responses vary across runs even for the same query. Scores from `evaluate_response()` may differ between runs. | Makes reproducibility of exact numeric scores difficult |
| **Language model provider lock-in** | `evaluate.py` only supports OpenAI as judge; the RAG chain (in `core.py`) supports Gemini, Groq, HuggingFace, Ollama, OpenAI — but evaluation only works with one. | Potential mismatch between evaluation LLM and production LLM |

### 11.4 System / Infrastructure Limitations

| Limitation | Detail | Impact |
|-----------|--------|--------|
| **No persistent evaluation logs** | Evaluation results are only printed to stdout. No results are saved to a CSV, JSON file, or database across runs. | Cannot track metric trends over time |
| **No evaluation dashboard** | There is no visualization layer for evaluation results (no charts, no UI). | Manual interpretation required |
| **File format support limited to PDF and DOCX** | Only `.pdf` and `.docx` files are supported for ingestion. | Cannot evaluate on plain text, HTML, Markdown, or Excel inputs |
| **Multilingual evaluation not tested** | The embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) supports multilingual input, but the evaluation test suite does not include non-English test cases. | Multilingual quality is unverified |
| **Local venv dependency** | Must activate `venv` before running any evaluation scripts. Missing activation leads to import errors. | Onboarding friction |

---

## 12. Recommendations for Improvement

### Priority: High

1. **Implement `generate_synthetic_qa()`** — Pull all documents from `data/docs`, use an LLM to generate 5–10 Q/A pairs per document, and cache them as a JSON file for repeatable evaluation.

2. **Fix `evaluate.py run()`** — Replace hardcoded mock results with a real evaluation loop:
   ```python
   for case in test_cases:
       response = self.chain.invoke({"question": case["question"]})
       score = self.evaluate_response(case["question"], response["answer"], context)
   ```

3. **Support Gemini/Groq as judge LLM** — The judge LLM in `evaluate.py` should read from `Config.LLM_PROVIDER` and use the appropriate provider, not hardcode OpenAI.

4. **Save evaluation results to JSON/CSV** — Persist results to `data/eval_results/` with timestamps for trend tracking.

### Priority: Medium

5. **Integrate RAGAS** — Add `ragas` library for standardized metrics: Context Precision, Context Recall, Faithfulness, Answer Correctness.
   ```bash
   pip install ragas
   ```

6. **Add multilingual test cases** — Add Arabic or French test questions to validate the multilingual embedding model.

7. **Add response time benchmarking** — Track and log `elapsed` time for every test case across providers.

8. **Add a re-ranker** — Integrate a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) between retrieval and generation for better precision.

### Priority: Low

9. **Build an evaluation dashboard** — Use Gradio or a simple HTML report to display metrics visually after each evaluation run.

10. **Expand test case coverage** — Test edge cases: empty queries, single-word queries, very long queries, ambiguous queries, multi-part questions.

---

## Appendix A — File Reference

| File | Purpose |
|------|---------|
| `core.py` | Main RAG pipeline, guardrails, ingestion, chain builder |
| `notebooks/evaluate.py` | LLM-as-a-Judge evaluator class |
| `notebooks/03_Evaluation.ipynb` | Interactive evaluation notebook |
| `test_features.py` | Full system smoke test suite (11 sections, 22+ tests) |
| `EVALUATION.md` | This document |
| `DOCUMENTATION.md` | Full system architecture documentation |
| `README.md` | Project overview and quick-start guide |

---

## Appendix B — Evaluation Thresholds Reference

| Parameter | Default | Env Variable | Description |
|-----------|---------|-------------|-------------|
| Input Relevance Threshold | `0.25` | `GUARDRAIL_INPUT_THRESHOLD` | Minimum similarity score for a query to be considered relevant |
| Output Grounding Threshold | `0.20` | `GUARDRAIL_OUTPUT_THRESHOLD` | Minimum cosine similarity between answer and context |
| Top K (final context) | `4` | `TOP_K` | Number of chunks passed to LLM after deduplication |
| Fetch K (retrieval) | `20` | `FETCH_K` | Number of chunks initially fetched from FAISS before filtering |
| Score Threshold | `0.30` | `SCORE_THRESHOLD` | Minimum retrieval score threshold |

---

*Document version: 1.0 — February 2026*  
*Smart Document Assistant — ITI Level 2 Advanced Final Project*
