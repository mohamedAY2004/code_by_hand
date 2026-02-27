from helpers import chunk_text
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, APIRouter
import pymupdf
import uuid
import io

upload_router = APIRouter()

@upload_router.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Upload a text or PDF document, chunk it, embed it, and store in ChromaDB."""
    content = await file.read()

    if file.filename.endswith(".pdf"):
        try:
            pdf = pymupdf.open(stream=content, filetype="pdf")
            text = "\n".join(page.get_text() for page in pdf)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")
    else:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be a UTF-8 encoded text file or a PDF.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Document appears to be empty or unreadable.")

    chunks = chunk_text(text)
    embeddings = request.app.embedder.encode(chunks).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": file.filename, "chunk_index": i} for i in range(len(chunks))]
    request.app.collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)

    return {"filename": file.filename, "chunks_stored": len(chunks)}