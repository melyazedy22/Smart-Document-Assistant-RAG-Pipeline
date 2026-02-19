# 📘 Smart Document Assistant — Full Technical Documentation

> This document describes every component, class, function, and design decision in the project.
> Use it for future discussions, presentations, or onboarding.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives & How They're Met](#2-objectives--how-theyre-met)
3. [Architecture Deep Dive](#3-architecture-deep-dive)
4. [core.py — The Engine](#4-corepy--the-engine)
5. [backend.py — The API](#5-backendpy--the-api)
6. [frontend.py — The UI](#6-frontendpy--the-ui)
7. [Guardrails System](#7-guardrails-system)
8. [RAG Pipeline Flow](#8-rag-pipeline-flow)
9. [Evaluation Pipeline](#9-evaluation-pipeline)
10. [Configuration Reference](#10-configuration-reference)
11. [Known Limitations](#11-known-limitations)
12. [Key Design Decisions](#12-key-design-decisions)

---

## 1. Project Overview

**What it does:** Users upload long documents (PDF/DOCX) → the system extracts text, chunks it, creates embeddings, stores them in a FAISS vector store → users ask questions via a chat interface → the system retrieves relevant chunks, sends them to an LLM, and returns an answer with source citations and guardrail checks.

**Who it's for:** Anyone who needs to quickly query and understand large documents — contracts, insurance policies, research papers, reports.

---

## 2. Objectives & How They're Met

| # | Objective | Implementation |
|---|---|---|
| 1 | Demonstrate LLM inference interfaces & microservices | FastAPI + LangServe backend exposes RAG chain as REST API. Multi-provider LLM support (Groq, Gemini, OpenAI, HuggingFace, Ollama). |
| 2 | Build end-to-end RAG pipeline | Full pipeline: Load → Split → Embed → Store → Retrieve → Generate → Cite. |
| 3 | Showcase long-form document processing | Configurable chunking (1000 chars, 200 overlap), metadata tracking (doc_name, page, chunk_id), merge-based ingestion. |
| 4 | Apply embeddings for guardrailing | `Guardrails` class uses FAISS similarity (input check) and cosine similarity (output check). |
| 5 | Evaluate retrieval/answer quality | `evaluate.py` with LLM-as-Judge scoring (faithfulness + relevance). `03_Evaluation.ipynb` for interactive testing. |

---

## 3. Architecture Deep Dive

### System Flow
```
User → Gradio UI → core.py → FAISS → LLM (Groq) → Response + Citations → User
         │
         └→ FastAPI + LangServe (alternative API access)
```

### Component Interaction
1. **frontend.py** imports directly from `core.py` (no HTTP calls to backend needed)
2. **backend.py** also imports from `core.py` and exposes it via REST API
3. Both can run independently or together

### Files & Their Roles

| File | Lines | Role |
|---|---|---|
| `core.py` | ~430 | All business logic |
| `backend.py` | 28 | API layer only |
| `frontend.py` | ~420 | UI layer only |
| `notebooks/evaluate.py` | ~100 | Evaluation logic |

---

## 4. core.py — The Engine

### 4.1 Config Class (Lines ~35-67)

Loads all settings from `.env` file. Key settings:

| Setting | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | Which LLM to use |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Embedding model |
| `VECTORSTORE_TYPE` | `faiss` | Vector store backend |
| `VECTORSTORE_PATH` | `data/vectorstore` | Where to save index |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `4` | Number of documents to retrieve |
| `FETCH_K` | `20` | Candidates before filtering |

### 4.2 Guardrails Class

See [Section 7](#7-guardrails-system) for detailed explanation.

### 4.3 IngestionPipeline Class

**Purpose:** Handles the entire document ingestion flow.

#### Key Methods:

| Method | Purpose |
|---|---|
| `_initialize_embeddings()` | Creates HuggingFace or OpenAI embedding model |
| `load_documents(file_paths)` | Loads PDF (PyMuPDF) and DOCX (Docx2txt) files into LangChain Documents |
| `split_documents(documents)` | Splits into chunks with RecursiveCharacterTextSplitter, adds metadata (chunk_id, chunk_index, doc_name) |
| `create_vectorstore(chunks)` | Creates FAISS or Chroma vectorstore from chunks |
| `load_vectorstore()` | Loads existing vectorstore from disk |
| `ingest(file_paths)` | Full pipeline: load → split → create/merge vectorstore |

#### Ingestion Flow:
```
PDF/DOCX files
    ↓
load_documents()    → List[Document] with metadata (source, page)
    ↓
split_documents()   → List[Document] chunks with metadata (chunk_id, chunk_index, doc_name)
    ↓
create_vectorstore() → FAISS index (saved to disk)
```

#### Merge Logic:
When new documents are uploaded, the system **merges** them into the existing vectorstore (doesn't replace). This allows incremental document ingestion.

### 4.4 ChainBuilder Class

**Purpose:** Builds the RAG chain and manages LLM + retriever lifecycle.

#### Key Methods:

| Method | Purpose |
|---|---|
| `_initialize_llm()` | Creates LLM based on provider (Groq, Gemini, OpenAI, HuggingFace, Ollama) |
| `reload()` | Reloads vectorstore and creates fresh retriever |
| `_format_docs(docs)` | Joins document chunks into a single context string |
| `_extract_citations(docs)` | Extracts source + page metadata as Citation objects |
| `build_rag_chain(doc_filter)` | Builds the full RAG pipeline with guardrails |
| `build_summarization_chain()` | Builds a simple summarization prompt + LLM chain |
| `get_ingested_doc_names()` | Returns list of all document names in vectorstore |

#### Citation Model:
```python
class Citation(BaseModel):
    source: str   # File path
    page: int     # Page number
    snippet: str  # First 100 chars of content
```

### 4.5 Global Functions (Public API)

These are what `frontend.py` and `backend.py` import:

| Function | Purpose |
|---|---|
| `get_rag_chain(doc_filter=None)` | Returns a ready-to-use RAG chain (reloads vectorstore first) |
| `get_summary_chain()` | Returns a summarization chain |
| `get_ingested_doc_names()` | Returns list of indexed document names |
| `ingest_files(file_paths)` | Ingests files and reloads the chain |

---

## 5. backend.py — The API

A minimal FastAPI app that uses LangServe to expose chains as REST endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/rag/invoke` | POST | Send a question, get answer + citations |
| `/rag/playground` | GET | Interactive web playground |
| `/summarize/invoke` | POST | Send text, get summary |
| `/summarize/playground` | GET | Interactive web playground |
| `/docs` | GET | Auto-generated API documentation |

**How it works:** LangServe's `add_routes()` automatically creates REST endpoints from any LangChain Runnable.

---

## 6. frontend.py — The UI

### Design
- **Theme:** Dark (ChatGPT-style), black/gray palette with green (#10a37f) accents
- **Font:** Inter (Google Fonts)
- **Layout:** Two-column — sidebar (left) + chat area (right)

### UI Components

| Component | Location | Purpose |
|---|---|---|
| File Upload | Sidebar | Upload PDF/DOCX files |
| Process Button | Sidebar | Trigger ingestion |
| Indexed Files List | Sidebar | Show all ingested documents |
| Document Dropdown | Sidebar | Filter chat to specific document |
| Chatbot | Main area | Display conversation history |
| Text Input + Send | Bottom | User message input |

### Key Functions

| Function | Purpose |
|---|---|
| `get_vectorstore_stats()` | Check if vectorstore exists and has data |
| `build_doc_list_html()` | Generate HTML for indexed files list |
| `refresh_doc_dropdown()` | Update dropdown after new upload |
| `process_upload(files)` | Handle file upload → ingest → refresh UI |
| `chat_fn(message, history, active_doc)` | Main chat handler: send query → get response → update history |

### Chat Flow:
```
User types message
    ↓
chat_fn() called with (message, history, active_doc)
    ↓
get_rag_chain(doc_filter) → builds chain with guardrails
    ↓
chain.invoke({"question": message})
    ↓
Response: {"answer": "...", "citations": [...], "guardrails": {...}}
    ↓
Format answer + citations + guardrail warnings
    ↓
Append to chat history → display in chatbot
```

---

## 7. Guardrails System

### Why Guardrails?
LLMs can hallucinate or answer questions that have nothing to do with the uploaded documents. Guardrails prevent this.

### 7.1 Input Guardrail (Safety)

**Purpose:** Block queries that are unrelated to the indexed documents.

**How it works:**
1. User sends a query
2. System runs `similarity_search_with_score(query, k=1)` on FAISS
3. FAISS returns L2 distance to the nearest document chunk
4. Convert to similarity: `similarity = 1.0 / (1.0 + distance)`
5. If similarity < threshold (0.25), **block the query** with a message

**Example:**
- User asks "What is the termination clause?" → High similarity to contract → ✅ Passes
- User asks "What is the weather today?" → Low similarity to any document → ❌ Blocked

### 7.2 Output Guardrail (Factuality)

**Purpose:** Verify the LLM's answer is grounded in the retrieved context.

**How it works:**
1. After the LLM generates an answer, embed both the answer and the context
2. Compute cosine similarity between answer embedding and context embedding
3. If similarity < threshold (0.20), **add a warning** to the answer

**Example:**
- LLM answers with facts from the document → High similarity → ✅ No warning
- LLM hallucinates information → Low similarity → ⚠️ Warning appended

### 7.3 Design Decisions
- **Fail-open:** If guardrail code errors, the query passes through (doesn't break the app)
- **Configurable thresholds:** Set via environment variables (`GUARDRAIL_INPUT_THRESHOLD`, `GUARDRAIL_OUTPUT_THRESHOLD`)
- **No external dependencies:** Uses numpy for cosine similarity + existing FAISS/embeddings

---

## 8. RAG Pipeline Flow

### Complete Flow (with Guardrails)

```
1. User sends question
        ↓
2. INPUT GUARDRAIL: Check query relevance via FAISS similarity
        ↓ (if blocked → return "off-topic" message)
3. RETRIEVAL: Fetch top-4 relevant chunks from FAISS
        ↓
4. FILTERING: If doc_filter set, keep only chunks from that document
        ↓
5. CONTEXT BUILDING: Join chunks into context string
        ↓
6. LLM GENERATION: Send (system prompt + context + question) to LLM
        ↓
7. OUTPUT GUARDRAIL: Check answer grounding via cosine similarity
        ↓ (if ungrounded → append warning)
8. CITATION EXTRACTION: Extract source file + page from chunk metadata
        ↓
9. RESPONSE: Return {answer, citations, guardrails} to UI
```

### Retrieval Settings
- **k=4**: Return top 4 most relevant chunks
- **fetch_k=20**: Consider top 20 candidates before filtering
- **search_type**: similarity (L2 distance in FAISS)

---

## 9. Evaluation Pipeline

### Files
- `notebooks/03_Evaluation.ipynb` — Interactive evaluation notebook
- `notebooks/evaluate.py` — Standalone evaluator script

### Approach: LLM-as-Judge

Instead of manual evaluation, a secondary LLM scores the RAG pipeline's responses:

| Metric | What it Measures | Scale |
|---|---|---|
| **Faithfulness** | Is the answer supported by the retrieved context? | 0.0 – 1.0 |
| **Relevance** | Does the answer address the user's question? | 0.0 – 1.0 |
| **Contains-Ground-Truth** | Does the answer include expected information? | 0 or 1 |

### How It Works
1. Define test cases with questions + expected ground truth answers
2. Run each question through the RAG chain
3. Compare RAG answer against ground truth (simple substring match)
4. For deeper evaluation, use LLM-as-Judge to score faithfulness and relevance

---

## 10. Configuration Reference

### Environment Variables (.env)

| Variable | Default | Options | Purpose |
|---|---|---|---|
| `LLM_PROVIDER` | `gemini` | `openai`, `gemini`, `groq`, `huggingface`, `ollama` | Which LLM to use |
| `OPENAI_API_KEY` | — | — | OpenAI API key |
| `GOOGLE_API_KEY` | — | — | Google Gemini API key |
| `GROQ_API_KEY` | — | — | Groq API key |
| `HUGGINGFACEHUB_API_TOKEN` | — | — | HuggingFace API token |
| `HF_REPO_ID` | `zephyr-7b-beta` | Any HF model | HuggingFace model ID |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Any sentence-transformer | Embedding model |
| `VECTORSTORE_TYPE` | `faiss` | `faiss`, `chroma` | Vector store backend |
| `VECTORSTORE_PATH` | `data/vectorstore` | Any path | Where to save vectors |
| `CHUNK_SIZE` | `1000` | Integer | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Integer | Overlap between chunks |
| `TOP_K` | `4` | Integer | Docs to retrieve |
| `FETCH_K` | `20` | Integer | Candidates before filtering |
| `GUARDRAIL_INPUT_THRESHOLD` | `0.25` | 0.0–1.0 | Input relevance threshold |
| `GUARDRAIL_OUTPUT_THRESHOLD` | `0.20` | 0.0–1.0 | Output grounding threshold |

### Current Active Configuration
- **LLM:** Groq (fast cloud inference)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (local CPU)
- **Vector Store:** FAISS (local storage)

---

## 11. Known Limitations

1. **CPU-only embeddings** — Embedding creation runs on CPU, slower for large document sets
2. **English only** — No multi-language support currently
3. **No authentication** — Anyone with network access can use the app
4. **Session-only history** — Chat history is lost on browser refresh
5. **No streaming** — Full response is returned at once, not token-by-token
6. **FAISS in-memory** — Large vectorstores consume significant RAM
7. **Guardrail thresholds** — May need tuning per domain
8. **No deduplication** — Re-uploading the same file adds duplicate chunks

---

## 12. Key Design Decisions

| Decision | Why |
|---|---|
| **Single `core.py` file** | Keeps all logic in one place — easy to understand, debug, and present |
| **FAISS over Chroma** | Faster for small-medium datasets, no external server needed |
| **HuggingFace embeddings (local)** | Free, no API calls, works offline, good quality for general text |
| **Groq as LLM** | Free tier, very fast inference (~1-2s), runs LLaMA/Mixtral models |
| **Merge-based ingestion** | Users can upload documents incrementally without replacing existing ones |
| **Fail-open guardrails** | If guardrail code crashes, the app still works (safety net) |
| **Gradio (not Streamlit)** | Better for chat interfaces, built-in Chatbot component, easy theming |
| **Frontend imports core directly** | No HTTP overhead — faster responses than calling backend API |

---

*Last updated: February 18, 2026*
