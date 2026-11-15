# builds the BM25 and vector indexes from the data

import hashlib
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from search_aliases import SEARCH_ALIASES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
INFO_FILE = INDEX_DIR / "info.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def tokenize(text):
    text = text.replace("\\", "").replace("'", "").replace("’", "")

    words = []
    current = ""
    for character in text.lower():
        if character.isalnum() or character in "-_.":
            current += character
        else:
            if current:
                words.append(current)
            current = ""
    if current:
        words.append(current)

    cleaned = []
    for word in words:
        word = word.strip("-_.")
        if word:
            cleaned.append(word)
    return cleaned


def search_text(chunk):
    alias = SEARCH_ALIASES.get(chunk["id"].rsplit("-", 1)[0])
    return chunk["text"] + "\n\n" + alias if alias else chunk["text"]


def build_bm25(chunks):
    tokenized_chunks = []
    for chunk in chunks:
        tokenized_chunks.append(tokenize(search_text(chunk)))
    return BM25Okapi(tokenized_chunks)


def load_model():
    return SentenceTransformer(MODEL_NAME)

def normalize(vectors):
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / lengths

def fingerprint():
    data = CHUNKS_FILE.read_bytes() + MODEL_NAME.encode("utf-8") + json.dumps(SEARCH_ALIASES).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def load_or_build_embeddings(chunks, model):
    current = fingerprint()

    if EMBEDDINGS_FILE.exists() and INFO_FILE.exists():
        info = json.loads(INFO_FILE.read_text(encoding="utf-8"))
        if info["fingerprint"] == current:
            return np.load(EMBEDDINGS_FILE)

    print("Building embeddings for", len(chunks), "chunks with", MODEL_NAME, "...")
    texts = []
    for chunk in chunks:
        texts.append(search_text(chunk))
    vectors = normalize(model.encode(texts, show_progress_bar=True))

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, vectors)
    info = {"fingerprint": current, "model": MODEL_NAME, "chunk_count": len(chunks)}
    INFO_FILE.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return vectors

def main():
    chunks = load_chunks()
    build_bm25(chunks)
    vectors = load_or_build_embeddings(chunks, load_model())
    print("BM25 index:   ", len(chunks), "chunks")
    print("Vector index: ", vectors.shape[0], "chunks x", vectors.shape[1], "numbers each")
    print("Saved in", INDEX_DIR)


if __name__ == "__main__":
    main()
