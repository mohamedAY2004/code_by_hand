from langchain_text_splitters import RecursiveCharacterTextSplitter
def chunk_text(text: str) -> list[str]:
    """
    Split text into chunks using LangChain's RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return splitter.split_text(text)

import numpy as np
import re
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt_tab", quiet=True)
def clean_pdf_text(text: str) -> str:
    """Clean common PDF extraction artifacts before chunking."""
    # remove page numbers (standalone numbers on a line)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # normalize multiple newlines to double (paragraph boundary)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # join hyphenated line breaks (common in PDFs: "algo-\nrithm" → "algorithm")
    text = re.sub(r'-\n(\w)', r'\1', text)
    # replace single newlines with space (mid-paragraph line wrapping)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def split_into_units(text: str) -> list[str]:
    """
    Split text into small meaningful units using paragraph and 
    punctuation boundaries — more reliable than sent_tokenize on PDF text.
    """
    units = []

    # first split on paragraph boundaries (double newline)
    paragraphs = re.split(r'\n\n+', text)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # if paragraph is short enough, keep as one unit
        if len(para) < 200:
            units.append(para)
            continue

        # otherwise split on sentence-ending punctuation
        # handles "Dr.", "Fig.", "et al." better than sent_tokenize
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para)
        for s in sentences:
            s = s.strip()
            if s:
                units.append(s)

    return [u for u in units if len(u) > 20]  # drop very short noise units
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def semantic_chunk(text: str, embedder, threshold: float = 0.6, max_chunk_size: int = 1024) -> list[str]:
    """
    Clean text → split into small units → embed → merge adjacent 
    similar units into coherent chunks.
    
    This merge-upward approach is more robust than splitting sentences
    on raw PDF text which is often malformed.
    """
    cleaned = clean_pdf_text(text)
    units = split_into_units(cleaned)

    if not units:
        return []
    if len(units) == 1:
        return units

    # embed all units in one batch
    embeddings = embedder.encode(units)  # shape: (n_units, embedding_dim)

    # merge-upward: start with first unit, keep merging into current chunk
    # as long as similarity is high and size allows
    chunks = []
    current_units = [units[0]]
    current_embedding = embeddings[0]
    current_size = len(units[0])
    for i in range(1, len(units)):
        sim = cosine_sim(current_embedding, embeddings[i])
        would_exceed = current_size + len(units[i]) > max_chunk_size

        if sim >= threshold and not would_exceed:
            # similar enough and not too large → merge into current chunk
            current_units.append(units[i])
            current_size += len(units[i])
            # update running embedding as mean of merged units
            current_embedding = np.mean(embeddings[max(0, i-3):i+1], axis=0)
        else:
            # topic shifted or chunk full → flush and start new chunk
            chunks.append(" ".join(current_units).strip())
            current_units = [units[i]]
            current_embedding = embeddings[i]
            current_size = len(units[i])
    # flush last chunk
    last = " ".join(current_units).strip()
    if last:
        chunks.append(last)
    return chunks