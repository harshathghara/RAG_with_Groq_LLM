import os
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Load pre-trained model for embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
model = SentenceTransformer(EMBEDDING_MODEL)

def generate_embeddings(text_file, index_folder):
    """Generate embeddings for a given text file and store in FAISS."""
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Split text into chunks (every 2 sentences)
    sentences = text.split("\n")
    chunks = [" ".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]

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
