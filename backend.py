from fastapi import FastAPI
from langserve import add_routes
from core import get_rag_chain, get_summary_chain

app = FastAPI(
    title="Smart Contract Assistant API",
    version="1.0",
    description="API for RAG and Summarization"
)

# Add RAG route
add_routes(
    app,
    get_rag_chain(),
    path="/rag",
)

# Add Summarization route
add_routes(
    app,
    get_summary_chain(),
    path="/summarize",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
