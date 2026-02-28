# Document Q&A API

A FastAPI-based REST API that lets you upload text/PDF documents, chunks and embeds them using a sentence-transformer model, stores them in ChromaDB, and retrieves relevant context for any question via semantic search.

## Project Structure

```
src/
├── main.py                     # App entry point, startup/shutdown hooks
├── requirements.txt            # Python dependencies
├── routes/
│   ├── base.py                 # GET /  — health check
│   ├── upload.py               # POST /upload  — document ingestion
│   └── chat.py                 # POST /chat  — semantic search Q&A
├── schemas/
│   └── ChatRequestSchema.py    # Pydantic model for chat requests
└── helpers/
    └── chunking.py             # Text splitting utility
```

## Prerequisites

- **Python 3.11.14**

## Setup

1. **Clone the repo**

   ```bash
   git clone <repo-url>
   cd code_by_hand
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   ```

   Activate it:

   - **Windows:** `.venv\Scripts\activate`
   - **Linux/macOS:** `source .venv/bin/activate`

3. **Install dependencies**

   ```bash
   pip install -r src/requirements.txt
   ```

## Running the API

```bash
cd src
uvicorn main:app --reload
```

The server starts at **http://127.0.0.1:8000**. Interactive docs are available at **http://127.0.0.1:8000/docs**.

> On first startup the `BAAI/bge-large-en-v1.5` embedding model (~1.3 GB) is downloaded and cached automatically.

## API Endpoints

### `GET /` — Health Check

```bash
curl http://127.0.0.1:8000/
```

```json
{ "message": "Hello, World!" }
```

### `POST /upload` — Upload a Document

Upload a `.txt` or `.pdf` file. The API extracts text, splits it into chunks, embeds each chunk, and stores them in ChromaDB.

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@path/to/document.pdf"
```

```json
{ "filename": "document.pdf", "chunks_stored": 42 }
```

### `POST /chat` — Ask a Question

Send a question and get back the most relevant document chunks.

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is X?", "top_k": 3}'
```

| Field     | Type   | Default | Description                          |
| --------- | ------ | ------- | ------------------------------------ |
| `message` | string | —       | Your question                        |
| `top_k`   | int    | 3       | Number of context chunks to retrieve |

```json
{
  "question": "What is X?",
  "answer": "[LLM PLACEHOLDER] ...",
  "context": "...relevant chunks...",
  "sources": ["document.pdf", "document.pdf"]
}
```

> The `answer` field currently returns a placeholder. Swap the marked block in `src/routes/chat.py` with a call to any LLM (OpenAI, Ollama, etc.) to get real answers.

## Notes

- Document embeddings are persisted in a local ChromaDB database at `src/db/`.
- Chunking uses LangChain's `RecursiveCharacterTextSplitter` with a chunk size of 512 and overlap of 50.
