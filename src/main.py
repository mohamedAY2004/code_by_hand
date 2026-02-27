from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from routes.base import base_router
from routes.chat import chat_router
from routes.upload import upload_router
app = FastAPI()
app.include_router(base_router)
app.include_router(chat_router)
app.include_router(upload_router)
@app.on_event("startup")
async def startup_event():
    app.embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
    app.chroma = chromadb.PersistentClient(path="./db")
    app.collection = app.chroma.get_or_create_collection("documents")
@app.on_event("shutdown")
async def shutdown_event():
    app.collection = None
    app.chroma = None
    del app.embedder