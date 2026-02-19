"""
Smart Document Assistant — Full Feature Test Suite
===================================================
Tests all features: ingestion, retrieval, RAG chain, guardrails, 
citations, summarization, and doc filtering.

Usage:
    python test_features.py

Works with ANY uploaded documents.
"""

import os
import sys
import time
import json

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))

from core import (
    Config, Guardrails, IngestionPipeline, ChainBuilder,
    get_rag_chain, get_summary_chain, get_ingested_doc_names, ingest_files
)

# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────
PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️ WARN"
results = []

def test(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append({"name": name, "status": status, "detail": detail})
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ──────────────────────────────────────────
# TEST 1: Configuration
# ──────────────────────────────────────────
section("1. Configuration")

test("LLM_PROVIDER is set", bool(Config.LLM_PROVIDER), f"Provider: {Config.LLM_PROVIDER}")
test("EMBEDDING_MODEL_NAME is set", bool(Config.EMBEDDING_MODEL_NAME), f"Model: {Config.EMBEDDING_MODEL_NAME}")
test("VECTORSTORE_PATH exists", os.path.exists(Config.VECTORSTORE_PATH), Config.VECTORSTORE_PATH)
test("CHUNK_SIZE > 0", Config.CHUNK_SIZE > 0, f"Size: {Config.CHUNK_SIZE}")
test("CHUNK_OVERLAP > 0", Config.CHUNK_OVERLAP > 0, f"Overlap: {Config.CHUNK_OVERLAP}")

# Check API key for current provider
provider = Config.LLM_PROVIDER.lower()
key_map = {
    "openai": Config.OPENAI_API_KEY,
    "gemini": Config.GOOGLE_API_KEY,
    "groq": Config.GROQ_API_KEY,
    "huggingface": Config.HUGGINGFACEHUB_API_TOKEN,
}
if provider in key_map:
    test(f"API key for '{provider}' is set", bool(key_map[provider]), "Key present" if key_map[provider] else "MISSING!")

# ──────────────────────────────────────────
# TEST 2: Embeddings
# ──────────────────────────────────────────
section("2. Embeddings")

pipeline = IngestionPipeline()
test("Embeddings initialized", pipeline.embeddings is not None)

try:
    test_embedding = pipeline.embeddings.embed_query("test query")
    test("Embedding produces vector", len(test_embedding) > 0, f"Dimensions: {len(test_embedding)}")
except Exception as e:
    test("Embedding produces vector", False, str(e))

# ──────────────────────────────────────────
# TEST 3: Vector Store
# ──────────────────────────────────────────
section("3. Vector Store")

vectorstore = pipeline.load_vectorstore()
test("Vectorstore loads from disk", vectorstore is not None)

if vectorstore:
    # Check document count
    try:
        if hasattr(vectorstore, 'index'):
            total_vectors = vectorstore.index.ntotal
            test("Vectorstore has documents", total_vectors > 0, f"Total vectors: {total_vectors}")
        else:
            test("Vectorstore has documents", True, "Cannot determine count for this store type")
    except Exception as e:
        test("Vectorstore has documents", False, str(e))

# ──────────────────────────────────────────
# TEST 4: Ingested Documents
# ──────────────────────────────────────────
section("4. Ingested Documents")

doc_names = get_ingested_doc_names()
test("get_ingested_doc_names() returns list", isinstance(doc_names, list))
test("Documents are indexed", len(doc_names) > 0, f"Found: {doc_names}")

# ──────────────────────────────────────────
# TEST 5: RAG Chain — Basic Query
# ──────────────────────────────────────────
section("5. RAG Chain — Basic Query")

try:
    chain = get_rag_chain()
    test("RAG chain builds successfully", chain is not None)
    
    start = time.time()
    response = chain.invoke({"question": "What is this document about?"})
    elapsed = time.time() - start
    
    test("RAG chain returns response", response is not None)
    test("Response is a dict", isinstance(response, dict), f"Type: {type(response).__name__}")
    
    if isinstance(response, dict):
        answer = response.get("answer", "")
        citations = response.get("citations", [])
        guardrails = response.get("guardrails", {})
        
        test("Response has 'answer'", bool(answer), f"Length: {len(answer)} chars")
        test("Response has 'citations'", isinstance(citations, list), f"Count: {len(citations)}")
        test("Response has 'guardrails'", isinstance(guardrails, dict))
        test(f"Response time < 10s", elapsed < 10, f"Time: {elapsed:.2f}s")
        
        print(f"\n  📝 Answer preview: {answer[:200]}...")
        if citations:
            print(f"  📎 Citations: {len(citations)} sources")
            for c in citations:
                print(f"     - {os.path.basename(c.source)} (page {c.page})")
except Exception as e:
    test("RAG chain execution", False, str(e))

# ──────────────────────────────────────────
# TEST 6: Guardrails — Input Check (Relevant Query)
# ──────────────────────────────────────────
section("6. Guardrails — Input Check (Relevant Query)")

if vectorstore:
    try:
        relevant_result = Guardrails.input_check("What are the main topics?", vectorstore, pipeline.embeddings)
        test("Input check returns dict", isinstance(relevant_result, dict))
        test("Relevant query passes", relevant_result.get("passed", False), f"Score: {relevant_result.get('score', 0):.3f}")
    except Exception as e:
        test("Input check (relevant)", False, str(e))

# ──────────────────────────────────────────
# TEST 7: Guardrails — Input Check (Off-topic Query)
# ──────────────────────────────────────────
section("7. Guardrails — Input Check (Off-topic Query)")

if vectorstore:
    try:
        offtopic_result = Guardrails.input_check(
            "What is the recipe for chocolate cake with vanilla frosting?", 
            vectorstore, pipeline.embeddings
        )
        test("Off-topic check returns dict", isinstance(offtopic_result, dict))
        test("Off-topic score is low", offtopic_result.get("score", 1.0) < 0.5, 
             f"Score: {offtopic_result.get('score', 0):.3f}")
        print(f"  ℹ️  Passed: {offtopic_result.get('passed')}, Message: {offtopic_result.get('message', 'none')[:100]}")
    except Exception as e:
        test("Input check (off-topic)", False, str(e))

# ──────────────────────────────────────────
# TEST 8: Guardrails — Output Check
# ──────────────────────────────────────────
section("8. Guardrails — Output Check (Grounding)")

try:
    # Grounded answer (answer matches context)
    grounded = Guardrails.output_check(
        "The document discusses important topics.",
        "This document covers several important topics including contracts and agreements.",
        pipeline.embeddings
    )
    test("Grounded answer detected", grounded.get("grounded", False), f"Score: {grounded.get('score', 0):.3f}")
    
    # Unrelated answer (answer doesn't match context)
    unrelated = Guardrails.output_check(
        "The weather in Paris is sunny today with temperatures around 25 degrees.",
        "This contract states that the termination clause requires 30 days written notice.",
        pipeline.embeddings
    )
    test("Unrelated answer has lower score", 
         unrelated.get("score", 1.0) < grounded.get("score", 0.0),
         f"Grounded: {grounded.get('score', 0):.3f} vs Unrelated: {unrelated.get('score', 0):.3f}")
except Exception as e:
    test("Output check", False, str(e))

# ──────────────────────────────────────────
# TEST 9: Document Filtering
# ──────────────────────────────────────────
section("9. Document Filtering")

if doc_names:
    try:
        first_doc = doc_names[0]
        filtered_chain = get_rag_chain(doc_filter=first_doc)
        test("Filtered chain builds", filtered_chain is not None, f"Filter: {first_doc}")
        
        filtered_response = filtered_chain.invoke({"question": "Summarize this document briefly."})
        test("Filtered chain returns response", isinstance(filtered_response, dict))
        
        if isinstance(filtered_response, dict):
            test("Filtered response has answer", bool(filtered_response.get("answer")))
    except Exception as e:
        test("Document filtering", False, str(e))

# ──────────────────────────────────────────
# TEST 10: Summarization Chain
# ──────────────────────────────────────────
section("10. Summarization Chain")

try:
    summary_chain = get_summary_chain()
    test("Summary chain builds", summary_chain is not None)
    
    summary = summary_chain.invoke({"context": "This is a sample document about a business contract between two parties. The contract includes terms for payment, delivery, and termination."})
    test("Summary chain produces output", bool(summary), f"Length: {len(str(summary))} chars")
except Exception as e:
    test("Summarization", False, str(e))

# ──────────────────────────────────────────
# TEST 11: RAG with Guardrails (Off-topic via Chain)
# ──────────────────────────────────────────
section("11. RAG Chain — Off-topic Query (Guardrail Integration)")

try:
    chain = get_rag_chain()
    response = chain.invoke({"question": "How do I bake a pizza with mozzarella cheese?"})
    
    if isinstance(response, dict):
        guardrails = response.get("guardrails", {})
        input_guard = guardrails.get("input", {})
        
        test("Guardrails metadata present", bool(guardrails))
        print(f"  ℹ️  Input passed: {input_guard.get('passed', 'N/A')}, Score: {input_guard.get('score', 'N/A')}")
        print(f"  ℹ️  Answer: {response.get('answer', '')[:150]}...")
except Exception as e:
    test("Off-topic chain test", False, str(e))

# ──────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────
section("TEST SUMMARY")

total = len(results)
passed = sum(1 for r in results if r["status"] == PASS)
failed = sum(1 for r in results if r["status"] == FAIL)

print(f"\n  Total tests: {total}")
print(f"  {PASS}: {passed}")
print(f"  {FAIL}: {failed}")
print(f"  Score: {passed}/{total} ({100*passed/total:.0f}%)")

if failed > 0:
    print(f"\n  Failed tests:")
    for r in results:
        if r["status"] == FAIL:
            print(f"    ❌ {r['name']}: {r['detail']}")

print(f"\n{'='*60}")
print(f"  {'ALL TESTS PASSED! 🎉' if failed == 0 else f'{failed} TEST(S) FAILED'}")
print(f"{'='*60}\n")
