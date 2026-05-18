import faiss
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Load the embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
model = SentenceTransformer(EMBEDDING_MODEL)

# Load FAISS index & metadata
#index_path = "embeddings/faiss_index"
#metadata_path = "embeddings/faiss_index_metadata.npy"

index_path = "embeddings/faiss_index_for_Test"
metadata_path = "embeddings/faiss_index_metadata_for_Test.npy"

index = faiss.read_index(index_path)
metadata = np.load(metadata_path, allow_pickle=True)

def search_query(query, top_k=2):
    """Search FAISS for the most relevant text chunks."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = [{"chunk": metadata[i], "score": distances[0][j]} for j, i in enumerate(indices[0])]
    return results

if __name__ == "__main__":
    print("\n🔍 PDF Search Engine (Type 'exit' to quit)\n")
    
    while True:
        query = input("Enter your search query: ")
        if query.lower() == "exit":
            print("Exiting search engine. Goodbye! 👋")
            break

        results = search_query(query)

        print("\nTop Results:")
        for res in results:
            print(f"- {res['chunk']} (Score: {res['score']:.4f})")
        print("\n" + "-"*50)

