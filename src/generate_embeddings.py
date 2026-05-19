import os
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

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

    # Save FAISS index
    os.makedirs(index_folder, exist_ok=True)
    #faiss.write_index(index, os.path.join(index_folder, "faiss_index"))
    faiss.write_index(index, os.path.join(index_folder, "faiss_index_for_Test"))



    # Save metadata (to map embeddings to original text chunks)
    #np.save(os.path.join(index_folder, "faiss_index_metadata.npy"), np.array(chunks))
    np.save(os.path.join(index_folder, "faiss_index_metadata_for_Test.npy"), np.array(chunks))


    print("Embeddings generated and FAISS index stored successfully!")

if __name__ == "__main__":
    text_file = "processed_text/Test.txt"
    index_folder = "embeddings/"

    generate_embeddings(text_file, index_folder)
