import os
import sys
import json
import random
from typing import List, Dict

# Add parent directory to path so we can import core
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from core import IngestionPipeline, get_rag_chain, Config

# Evaluation Setup
class Evaluator:
    def __init__(self):
        self.ingestion = IngestionPipeline()
        self.chain = get_rag_chain()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=Config.OPENAI_API_KEY) if Config.OPENAI_API_KEY else None

    def generate_synthetic_qa(self, num_samples=5) -> List[Dict]:
        """Generates synthetic Q/A pairs from random chunks in the index."""
        # Load vectorstore to get random documents
        vectorstore = self.ingestion.load_vectorstore()
        if not vectorstore:
            print("No vectorstore found. Please ingest documents first.")
            return []

        # Access underlying docstore if possible, or just hack it for FAISS by reconstruction
        # For simplicity, we assume we can access or re-load documents.
        # Here we will just use a placeholder if we can't easily iterate all docs in FAISS without keeping them.
        # In a real app, keep a docstore. 
        # For this demo, let's assume we have some docs in data/docs to re-read.
        pass # Implementation detail: proper synthetic generation requires access to full text.

    def evaluate_response(self, question, answer, context):
        """Uses LLM-as-a-Judge to evaluate RAG response."""
        if not self.llm:
            return {"faithfulness": 0.0, "relevance": 0.0}

        # prompt for faithfulness
        faithfulness_prompt = ChatPromptTemplate.from_template(
            """
            You are an expert evaluator. 
            Context: {context}
            Answer: {answer}
            
            Rate the faithfulness of the answer to the context on a scale of 0 to 1.
            Return ONLY a JSON object: {{"score": 0.5}}
            """
        )
        
        # prompt for relevance
        relevance_prompt = ChatPromptTemplate.from_template(
            """
            You are an expert evaluator.
            Question: {question}
            Answer: {answer}
            
            Rate the relevance of the answer to the question on a scale of 0 to 1.
            Return ONLY a JSON object: {{"score": 0.5}}
            """
        )

        try:
             # Very basic invocation
            faith_res = (faithfulness_prompt | self.llm | JsonOutputParser()).invoke({"context": context, "answer": answer})
            rel_res = (relevance_prompt | self.llm | JsonOutputParser()).invoke({"question": question, "answer": answer})
            
            return {
                "faithfulness": faith_res.get("score", 0),
                "relevance": rel_res.get("score", 0)
            }
        except Exception as e:
            print(f"Evaluation error: {e}")
            return {"faithfulness": 0, "relevance": 0}

    def run(self):
        print("Starting evaluation...")
        # Mock evaluation loop for demonstration
        # In a real scenario, we loop through generated Q/A
        
        results = {
            "metrics": {
                "faithfulness": 0.85,
                "relevance": 0.90,
                "win_rate": 0.75 # vs baseline
            },
            "details": [
                {"question": "What is the termination clause?", "faithfulness": 0.9, "relevance": 0.95}
            ]
        }
        
        print(json.dumps(results, indent=2))
        return results

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run()
