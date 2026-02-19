# 📑 Smart Document Assistant

A local-first, privacy-focused Q&A assistant for long documents (contracts, insurance policies, reports). Built with **LangChain**, **RAG**, **Gradio**, and **FastAPI**.

Users upload PDF/DOCX files, the system extracts, chunks, and embeds content, stores it in a FAISS vector store, and enables chat-based question answering with **guardrails** and **source citations**.

---

## 🏗️ Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐
│  Gradio UI   │    │  FastAPI +   │    │      core.py         │
│ (frontend.py)│───▶│  LangServe   │───▶│  ┌──────────────┐   │
│              │    │ (backend.py) │    │  │   Config      │   │
│ • Upload     │    │              │    │  ├──────────────┤   │
│ • Chat       │    │ Endpoints:   │    │  │  Guardrails   │   │
│ • Doc Select │    │  /rag        │    │  ├──────────────┤   │
│              │    │  /summarize  │    │  │  Ingestion    │   │
└──────────────┘    └──────────────┘    │  │  Pipeline     │   │
                                        │  ├──────────────┤   │
                                        │  │ ChainBuilder  │   │
                                        │  │ (RAG + LLM)  │   │
                                        │  └──────────────┘   │
                                        │         │            │
                                        │    ┌────┴────┐       │
                                        │    │  FAISS  │       │
                                        │    │ Vector  │       │
                                        │    │  Store  │       │
                                        │    └─────────┘       │
                                        └──────────────────────┘
```

### Main Components

| Component | File | Purpose |
|---|---|---|
| **Engine** | `core.py` | All logic: config, ingestion, embeddings, RAG chain, guardrails, summarization |
| **API** | `backend.py` | FastAPI + LangServe REST API (`/rag`, `/summarize`) |
| **UI** | `frontend.py` | Gradio dark-themed ChatGPT-style interface |
| **Notebooks** | `notebooks/` | Pipeline walkthrough + evaluation |

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| **Framework** | LangChain, LangServe |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Gradio |
| **Vector Store** | FAISS (default), Chroma (switchable) |
| **Embeddings** | HuggingFace SentenceTransformers (`all-MiniLM-L6-v2`) |
| **LLM Providers** | Groq (active), Gemini, OpenAI, HuggingFace, Ollama |
| **File Parsing** | PyMuPDF (PDF), Docx2txt (DOCX) |
| **Guardrails** | Custom embedding-based (input relevance + output grounding) |

---

## 📁 Project Structure

```
Final_Project/
├── core.py                  # Engine: Config, Ingestion, RAG, Guardrails
├── backend.py               # FastAPI + LangServe API
├── frontend.py              # Gradio UI
├── .env                     # API keys & configuration
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── DOCUMENTATION.md         # Detailed project documentation
├── data/
│   └── vectorstore/         # FAISS index storage
├── notebooks/
│   ├── 01_System_Overview.ipynb   # Project spec & architecture
│   ├── 02_RAG_Pipeline.ipynb      # Pipeline walkthrough
│   ├── 03_Evaluation.ipynb        # Evaluation with metrics
│   └── evaluate.py                # LLM-as-Judge evaluator
└── Project Discribtion/
    ├── Smart_Contract_Assistant_Spec.docx.pdf
    └── LLM_Orchestration_Recap.pptx
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file:
```ini
LLM_PROVIDER=groq              # Options: openai, gemini, groq, huggingface, ollama
GROQ_API_KEY=your_key_here     # Required if provider is groq
GOOGLE_API_KEY=your_key_here   # Required if provider is gemini
OPENAI_API_KEY=your_key_here   # Required if provider is openai

VECTORSTORE_TYPE=faiss
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4
FETCH_K=20
```

---

## 🖥️ Usage

### Run the UI (Recommended)
```bash
python frontend.py
```
Open: `http://127.0.0.1:7860`

### Run the Backend API
```bash
python backend.py
```
- API Docs: `http://localhost:8000/docs`
- Playground: `http://localhost:8000/rag/playground`

---

## 🔒 Guardrails

The system implements **embedding-based guardrails** for safety and factuality:

| Guardrail | Type | How It Works |
|---|---|---|
| **Input Check** | Safety | Uses FAISS similarity score to verify the query is relevant to indexed documents. Off-topic or harmful queries are blocked. |
| **Output Check** | Factuality | Uses cosine similarity between answer embedding and context embedding to verify the LLM answer is grounded in the documents. |

Both guardrails use configurable thresholds and follow a **fail-open** design (errors don't break the app).

---

## 📊 Evaluation

The project includes an evaluation pipeline (`notebooks/03_Evaluation.ipynb` + `notebooks/evaluate.py`):

| Metric | Description |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Relevance** | Does the answer address the user's question? |
| **Contains-Ground-Truth** | Does the answer contain the expected information? |

Evaluation uses an **LLM-as-Judge** approach where a secondary LLM scores the RAG pipeline's responses.

---

## ⚠️ Known Limitations

1. **Embedding model runs on CPU** — Slower on large document sets. GPU would improve performance.
2. **No multi-language support** — Currently optimized for English documents only.
3. **No authentication** — Local deployment only, no user access control.
4. **Guardrail thresholds** — May need tuning per domain; current defaults work for general documents.
5. **Single-session history** — Chat history is not persisted across browser refreshes.
6. **FAISS in-memory** — Large vectorstores may consume significant RAM.
7. **No streaming** — Responses are returned as complete text, not streamed token-by-token.

---

## 🔮 Future Enhancements

- Multi-document cross-search
- Domain-specific fine-tuned models
- Role-based access control
- Cloud deployment (Docker/Kubernetes)
- Response streaming
- Persistent chat history
=======
# Smart-Document-Assistant-RAG-Pipeline
