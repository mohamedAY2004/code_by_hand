from fastapi import FastAPI, UploadFile, File, HTTPException,Request,APIRouter
from schemas.ChatRequestSchema import ChatRequest
chat_router = APIRouter()

@chat_router.post("/chat")
async def chat(request: Request, chat_request: ChatRequest):
    """Receive a message, retrieve relevant chunks, and return them as context."""
    query_embedding = request.app.embedder.encode([chat_request.message]).tolist()

    results = request.app.collection.query(
        query_embeddings=query_embedding,
        n_results=chat_request.top_k,
    )

    if not results["documents"] or not results["documents"][0]:
        raise HTTPException(status_code=404, detail="No documents in the database yet.")

    context_chunks = results["documents"][0]
    sources = [m.get("source") for m in results["metadatas"][0]]

    # --- Swap this block for any LLM call (OpenAI, Ollama, etc.) ---
    context = "\n\n---\n\n".join(context_chunks)
    answer = f"[LLM PLACEHOLDER]\n\nUse this context only to answer: '{chat_request.message}'"
    # ---------------------------------------------------------------

    return {
        "question": chat_request.message,
        "answer": answer,
        "context": context,
        "sources": sources,
    }