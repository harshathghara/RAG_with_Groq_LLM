import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

# Load pre-trained model for embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

model = SentenceTransformer(EMBEDDING_MODEL)

def chunk_fixed_size(text, chunk_size=500, overlap=100):
    """Split text into fixed-size character chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]  # drop empty chunks


def generate_embeddings(text_file, index_folder):
    """Generate embeddings for a given text file and store in FAISS."""
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_fixed_size(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    # Generate embeddings
    embeddings = model.encode(chunks, convert_to_numpy=True)

    # Normalize for cosine similarity (dot product on unit vectors = cosine similarity)
    faiss.normalize_L2(embeddings)

    # Create FAISS index using Inner Product (cosine similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    stem = os.path.splitext(os.path.basename(text_file))[0]

    os.makedirs(index_folder, exist_ok=True)
    index_path    = os.path.join(index_folder, f"faiss_index_{stem}")
    metadata_path = os.path.join(index_folder, f"faiss_index_metadata_{stem}.npy")

    faiss.write_index(index, index_path)
    np.save(metadata_path, np.array(chunks))

    # Build BM25 index on lowercased token lists (one list per chunk)
    tokenized = [chunk.lower().split() for chunk in chunks]
    bm25      = BM25Okapi(tokenized)
    bm25_path = os.path.join(index_folder, f"bm25_index_{stem}.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    print(f"FAISS index saved:  {index_path}")
    print(f"Metadata saved:     {metadata_path}")
    print(f"BM25 index saved:   {bm25_path}")
    return index_path, metadata_path, bm25_path

if __name__ == "__main__":
    text_file = "processed_text/Test.txt"
    index_folder = "embeddings/"

    index_path, metadata_path, bm25_path = generate_embeddings(text_file, index_folder)
    print(f"Index:    {index_path}")
    print(f"Metadata: {metadata_path}")
    print(f"BM25:     {bm25_path}")
