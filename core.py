# Disable TensorFlow to prevent import conflicts with transformers/sentence-transformers
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import hashlib
import shutil
import uuid
from typing import List, Optional, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

# Load environment variables
load_dotenv()

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    # LLM Settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # 'local', 'openai', or 'gemini'
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    HF_REPO_ID = os.getenv("HF_REPO_ID", "HuggingFaceH4/zephyr-7b-beta")
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "gpt2") 

    # Embedding Settings
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Vector Store Settings
    VECTORSTORE_TYPE = os.getenv("VECTORSTORE_TYPE", "faiss")
    VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "data/vectorstore")

    # Chunking Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

    # Retrieval Settings
    TOP_K = int(os.getenv("TOP_K", 4))
    FETCH_K = int(os.getenv("FETCH_K", 20))
    SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.3))

    # Data Paths
    DATA_DIR = os.getenv("DATA_DIR", "data/docs")
    
    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.VECTORSTORE_PATH, exist_ok=True)

# ==========================================
# 2. Guardrails
# ==========================================
import numpy as np

class Guardrails:
    """Embedding-based guardrails for input relevance and output grounding."""
    
    # Similarity thresholds
    INPUT_RELEVANCE_THRESHOLD = float(os.getenv("GUARDRAIL_INPUT_THRESHOLD", 0.25))
    OUTPUT_GROUNDING_THRESHOLD = float(os.getenv("GUARDRAIL_OUTPUT_THRESHOLD", 0.20))
    
    @staticmethod
    def _cosine_similarity(vec_a, vec_b) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    @classmethod
    def input_check(cls, query: str, vectorstore, embeddings) -> dict:
        """
        Check if the user query is relevant to the indexed documents.
        Uses semantic similarity between query embedding and nearest doc.
        Returns: {"passed": bool, "score": float, "message": str}
        """
        try:
            if not vectorstore:
                return {"passed": True, "score": 0.0, "message": "No vectorstore to check against."}
            
            # Get top-1 similar document with score
            results = vectorstore.similarity_search_with_score(query, k=1)
            if not results:
                return {"passed": True, "score": 0.0, "message": "No documents found."}
            
            doc, distance = results[0]
            # FAISS returns L2 distance — convert to similarity (lower distance = more similar)
            similarity = 1.0 / (1.0 + distance)
            passed = similarity >= cls.INPUT_RELEVANCE_THRESHOLD
            
            logger.info(f"Guardrail input check: similarity={similarity:.3f}, threshold={cls.INPUT_RELEVANCE_THRESHOLD}, passed={passed}")
            
            if not passed:
                return {
                    "passed": False,
                    "score": similarity,
                    "message": "Your question doesn't appear related to the uploaded documents. Please ask something about the document content."
                }
            return {"passed": True, "score": similarity, "message": ""}
        
        except Exception as e:
            logger.error(f"Guardrail input check error: {e}")
            return {"passed": True, "score": 0.0, "message": ""}  # Fail-open
    
    @classmethod
    def output_check(cls, answer: str, context: str, embeddings) -> dict:
        """
        Verify the LLM answer is grounded in retrieved context.
        Uses cosine similarity between answer and context embeddings.
        Returns: {"grounded": bool, "score": float, "warning": str}
        """
        try:
            if not answer or not context:
                return {"grounded": True, "score": 0.0, "warning": ""}
            
            answer_emb = embeddings.embed_query(answer[:500])  # Limit length
            context_emb = embeddings.embed_query(context[:1000])
            
            similarity = cls._cosine_similarity(answer_emb, context_emb)
            grounded = similarity >= cls.OUTPUT_GROUNDING_THRESHOLD
            
            logger.info(f"Guardrail output check: similarity={similarity:.3f}, threshold={cls.OUTPUT_GROUNDING_THRESHOLD}, grounded={grounded}")
            
            if not grounded:
                return {
                    "grounded": False,
                    "score": similarity,
                    "warning": "⚠️ This answer may not be fully supported by the document content."
                }
            return {"grounded": True, "score": similarity, "warning": ""}
        
        except Exception as e:
            logger.error(f"Guardrail output check error: {e}")
            return {"grounded": True, "score": 0.0, "warning": ""}  # Fail-open

Config.ensure_dirs()

# ==========================================
# 2. Ingestion Pipeline
# ==========================================
class IngestionPipeline:
    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
        self.embeddings = self._get_embeddings()
        self.vectorstore_type = Config.VECTORSTORE_TYPE
        self.vectorstore_path = Config.VECTORSTORE_PATH

    def _get_embeddings(self):
        if Config.LLM_PROVIDER == "openai" and Config.OPENAI_API_KEY:
             logger.info("Using OpenAI embeddings")
             return OpenAIEmbeddings(api_key=Config.OPENAI_API_KEY)
        elif Config.EMBEDDING_MODEL_NAME.startswith("models/") and Config.GOOGLE_API_KEY:
             logger.info(f"Using Google GenAI embeddings: {Config.EMBEDDING_MODEL_NAME}")
             return GoogleGenerativeAIEmbeddings(model=Config.EMBEDDING_MODEL_NAME, google_api_key=Config.GOOGLE_API_KEY)
        else:
            logger.info(f"Using HuggingFace embeddings: {Config.EMBEDDING_MODEL_NAME}")
            return HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)

    def load_document(self, file_path: str) -> List[Document]:
        if file_path.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        return loader.load()

    def split_documents(self, documents: List[Document], doc_name: str, upload_id: str) -> List[Document]:
        """Split documents into chunks with consistent, robust metadata."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        
        for i, chunk in enumerate(chunks):
            # Consistent metadata for every chunk
            chunk.metadata["doc_name"] = doc_name  # filename only, no path
            chunk.metadata["source"] = doc_name     # stable source = filename
            chunk.metadata["page"] = chunk.metadata.get("page", chunk.metadata.get("page_number", 0))
            chunk.metadata["chunk_id"] = f"{doc_name}_chunk_{i}"
            chunk.metadata["chunk_index"] = i
            chunk.metadata["upload_id"] = upload_id
            chunk.metadata["content_hash"] = hashlib.md5(chunk.page_content.encode()).hexdigest()
            
        logger.info(f"Split '{doc_name}' into {len(chunks)} chunks (upload_id={upload_id})")
        return chunks

    def create_vectorstore(self, chunks: List[Document], save_path: Optional[str] = None, replace: bool = False):
        """Create or update vectorstore. If replace=True, overwrite entirely."""
        path = save_path or self.vectorstore_path
        logger.info(f"Creating/updating vectorstore at {path} with {len(chunks)} chunks (replace={replace})")
        
        if replace:
            # Delete existing index and start fresh
            if os.path.exists(path):
                shutil.rmtree(path)
                logger.info(f"Deleted existing vectorstore at {path}")
            os.makedirs(path, exist_ok=True)
            existing_store = None
        else:
            existing_store = self.load_vectorstore(path)
        
        if self.vectorstore_type == "chroma":
            if existing_store:
                existing_store.add_documents(chunks)
                vectorstore = existing_store
            else:
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=path
                )
        else:
            if existing_store:
                # Merge new chunks into existing FAISS index
                new_store = FAISS.from_documents(chunks, self.embeddings)
                existing_store.merge_from(new_store)
                vectorstore = existing_store
            else:
                vectorstore = FAISS.from_documents(chunks, self.embeddings)
            vectorstore.save_local(path)
        
        total = vectorstore.index.ntotal if hasattr(vectorstore, 'index') else 'N/A'
        logger.info(f"Vectorstore saved. Total vectors: {total}")
        return vectorstore

    def load_vectorstore(self, path: Optional[str] = None):
        path = path or self.vectorstore_path
        if not os.path.exists(path):
             logger.warning(f"Vectorstore path does not exist: {path}")
             return None

        try:
            if self.vectorstore_type == "chroma":
                vs = Chroma(persist_directory=path, embedding_function=self.embeddings)
                logger.info(f"Loaded Chroma vectorstore from {path}")
                return vs
            else:
                if not os.path.exists(os.path.join(path, "index.faiss")):
                    logger.warning(f"index.faiss not found in {path}")
                    return None
                vs = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
                logger.info(f"Loaded FAISS vectorstore from {path}, total vectors: {vs.index.ntotal}")
                return vs
        except Exception as e:
            logger.error(f"Failed to load vectorstore from {path}: {e}")
            return None

    def ingest(self, file_paths: List[str], replace: bool = False):
        """Ingest files. Each upload gets a unique upload_id for tracking."""
        upload_id = uuid.uuid4().hex[:8]
        logger.info(f"Ingestion started — upload_id={upload_id}, replace={replace}, files={file_paths}")
        
        all_chunks = []
        for file_path in file_paths:
            doc_name = os.path.basename(file_path)
            docs = self.load_document(file_path)
            chunks = self.split_documents(docs, doc_name=doc_name, upload_id=upload_id)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return None
            
        return self.create_vectorstore(all_chunks, replace=replace)

    def reset_vectorstore(self):
        """Delete the entire vectorstore folder and recreate an empty directory."""
        path = self.vectorstore_path
        if os.path.exists(path):
            shutil.rmtree(path)
            logger.info(f"Deleted vectorstore at {path}")
        os.makedirs(path, exist_ok=True)
        logger.info(f"Recreated empty vectorstore directory at {path}")

# ==========================================
# 3. Chain Builder (RAG Logic)
# ==========================================
class Citation(BaseModel):
    source: str = Field(description="The source filename")
    page: int = Field(description="The page number")
    snippet: str = Field(description="A brief snippet")

class ChainBuilder:
    def __init__(self):
        self.ingestion = IngestionPipeline()
        self.reload()

    def reload(self):
        """Reload vectorstore and retriever from disk — removes stale state."""
        logger.info("Reloading ChainBuilder...")
        self.vectorstore = self.ingestion.load_vectorstore()
        if self.vectorstore:
            # Use a generous fetch_k; scope filtering is done post-retrieval
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": Config.FETCH_K,  # Fetch more, filter later
                }
            )
            total = self.vectorstore.index.ntotal if hasattr(self.vectorstore, 'index') else '?'
            logger.info(f"Retriever created with k={Config.FETCH_K} (fetch_k for post-filter). Total vectors: {total}")
        else:
            self.retriever = None
            logger.warning("No vectorstore found — retriever is None")
        self.llm = self._get_llm()

    def _get_llm(self):
        if Config.LLM_PROVIDER == "openai" and Config.OPENAI_API_KEY:
            return ChatOpenAI(model="gpt-3.5-turbo", api_key=Config.OPENAI_API_KEY)
        elif Config.LLM_PROVIDER == "gemini" and Config.GOOGLE_API_KEY:
            return ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=Config.GOOGLE_API_KEY, convert_system_message_to_human=True)
        elif Config.LLM_PROVIDER == "groq" and Config.GROQ_API_KEY:
            return ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=Config.GROQ_API_KEY)
        elif Config.LLM_PROVIDER == "huggingface" and Config.HUGGINGFACEHUB_API_TOKEN:
            llm = HuggingFaceEndpoint(
                repo_id=Config.HF_REPO_ID, 
                huggingfacehub_api_token=Config.HUGGINGFACEHUB_API_TOKEN,
                temperature=0.5,
                max_new_tokens=512
            )
            return ChatHuggingFace(llm=llm)
        elif Config.LLM_PROVIDER == "ollama":
            return ChatOllama(model="llama3", base_url="http://localhost:11434")
        else:
            return RunnableLambda(lambda x: "Mock LLM Response: No provider configured.")

    def _format_docs(self, docs):
        return "\n\n".join(
            f"Source: {doc.metadata.get('doc_name', 'unknown')} (Page {doc.metadata.get('page', 0)})\nContent: {doc.page_content}"
            for doc in docs
        )
    
    def _extract_citations(self, docs, max_citations: int = 5):
        """Extract unique citations, deduplicated by (doc_name, page, chunk_id)."""
        seen = set()
        citations = []
        for doc in docs:
            key = (
                doc.metadata.get('doc_name', 'unknown'),
                doc.metadata.get('page', 0),
                doc.metadata.get('chunk_id', ''),
            )
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    source=doc.metadata.get('doc_name', 'unknown'),
                    page=doc.metadata.get('page', 0),
                    snippet=doc.page_content[:100] + "..."
                )
            )
            if len(citations) >= max_citations:
                break
        return citations

    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate chunks by (doc_name, page, chunk_id) or content_hash."""
        seen_keys = set()
        seen_hashes = set()
        unique = []
        for doc in docs:
            key = (
                doc.metadata.get('doc_name', ''),
                doc.metadata.get('page', 0),
                doc.metadata.get('chunk_id', ''),
            )
            content_hash = doc.metadata.get('content_hash', hashlib.md5(doc.page_content.encode()).hexdigest())
            if key in seen_keys or content_hash in seen_hashes:
                continue
            seen_keys.add(key)
            seen_hashes.add(content_hash)
            unique.append(doc)
        return unique

    def build_rag_chain(self, doc_filter: Optional[str] = None):
        if not self.retriever:
            logger.warning("build_rag_chain called but retriever is None")
            return RunnableLambda(lambda x: {"answer": "No documents indexed. Please upload documents first.", "citations": [], "guardrails": {}})

        if doc_filter:
            system_msg = (
                f"You are a helpful assistant. Answer the question using ONLY the context below, "
                f"which comes from the document '{doc_filter}'. "
                f"If the context does not contain enough information to answer, respond with: "
                f"'Not found in the selected document.'\n\nContext:\n{{context}}"
            )
        else:
            system_msg = (
                "You are a helpful assistant. Use the following context from uploaded documents "
                "to answer the question. If the context does not contain enough information, say so."
                "\n\nContext:\n{context}"
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "{question}"),
        ])

        # Capture fresh references for use in lambdas — avoids stale closures
        retriever = self.retriever
        format_docs = self._format_docs
        extract_citations = self._extract_citations
        deduplicate_docs = self._deduplicate_docs
        active_filter = doc_filter
        vectorstore = self.vectorstore
        embeddings = self.ingestion.embeddings
        top_k = Config.TOP_K

        def retrieve_docs(inputs):
            question = inputs["question"]
            
            # === INPUT GUARDRAIL ===
            input_guard = Guardrails.input_check(question, vectorstore, embeddings)
            if not input_guard["passed"]:
                logger.warning(f"Input guardrail blocked query: '{question[:80]}'")
                return {"docs": [], "input_guardrail": input_guard}
            
            # Retrieve fetch_k docs (retriever.k is set to FETCH_K)
            raw_docs = retriever.invoke(question)
            logger.info(f"Retrieved {len(raw_docs)} raw documents for query: '{question[:80]}'")
            
            # Log retrieved doc names for diagnostics
            retrieved_names = [d.metadata.get('doc_name', '?') for d in raw_docs]
            logger.info(f"  Retrieved doc_names: {retrieved_names}")
            
            # === SCOPE FILTERING ===
            if active_filter:
                raw_docs = [d for d in raw_docs if d.metadata.get('doc_name') == active_filter]
                logger.info(f"  After scope filter (doc_name='{active_filter}'): {len(raw_docs)} docs remain")
            
            # === DEDUPLICATION ===
            docs = deduplicate_docs(raw_docs)
            logger.info(f"  After deduplication: {len(docs)} unique docs")
            
            # === TAKE TOP_K ===
            docs = docs[:top_k]
            logger.info(f"  Final top_k={top_k} docs selected")
            
            for i, doc in enumerate(docs):
                logger.info(f"    Doc {i}: doc_name={doc.metadata.get('doc_name', '?')}, page={doc.metadata.get('page', '?')}, chunk_id={doc.metadata.get('chunk_id', '?')}")
            
            return {"docs": docs, "input_guardrail": input_guard}

        def full_pipeline(inputs):
            question = inputs["question"]
            retrieval_result = retrieve_docs(inputs)
            docs = retrieval_result["docs"]
            input_guardrail = retrieval_result["input_guardrail"]
            
            # If input was blocked, return early
            if not input_guardrail["passed"]:
                return {
                    "answer": input_guardrail["message"],
                    "citations": [],
                    "guardrails": {"input": input_guardrail, "output": {}}
                }
            
            # Format context and run LLM
            context = format_docs(docs)
            chain = prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})
            citations = extract_citations(docs, max_citations=5)
            
            # === OUTPUT GUARDRAIL ===
            output_guard = Guardrails.output_check(answer, context, embeddings)
            
            return {
                "answer": answer,
                "citations": citations,
                "guardrails": {"input": input_guardrail, "output": output_guard}
            }

        return RunnableLambda(full_pipeline)

    def get_ingested_doc_names(self) -> List[str]:
        """Return sorted list of unique doc_name values from the vectorstore."""
        if not self.vectorstore:
            return []
        try:
            doc_names = set()
            if hasattr(self.vectorstore, 'docstore') and hasattr(self.vectorstore.docstore, '_dict'):
                for doc_id, doc in self.vectorstore.docstore._dict.items():
                    name = doc.metadata.get('doc_name', '')
                    if name:
                        doc_names.add(name)
            return sorted(doc_names)
        except Exception as e:
            logger.error(f"Error listing doc names: {e}")
            return []
    
    def build_summarization_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize this document within 200 words:\n\n{context}"),
        ])
        return (prompt | self.llm | StrOutputParser())

# Global Instance
chain_builder = ChainBuilder()

def get_rag_chain(doc_filter: Optional[str] = None):
    """Always reload before building chain — guarantees fresh retriever."""
    chain_builder.reload()
    return chain_builder.build_rag_chain(doc_filter=doc_filter)

def get_summary_chain():
    return chain_builder.build_summarization_chain()

def get_ingested_doc_names() -> List[str]:
    chain_builder.reload()
    return chain_builder.get_ingested_doc_names()

def ingest_files(file_paths: List[str], replace: bool = False):
    """Ingest files into the vectorstore. replace=True wipes existing index first."""
    logger.info(f"Ingesting {len(file_paths)} files (replace={replace}): {file_paths}")
    result = chain_builder.ingestion.ingest(file_paths, replace=replace)
    chain_builder.reload()
    logger.info("Ingestion complete, chain reloaded.")
    return result

def reset_knowledge_base():
    """Delete the FAISS index and all docstore metadata, recreate empty vectorstore dir."""
    logger.info("=== RESETTING KNOWLEDGE BASE ===")
    chain_builder.ingestion.reset_vectorstore()
    chain_builder.reload()
    logger.info("Knowledge base reset complete. Vectorstore is now empty.")
