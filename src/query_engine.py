import faiss
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from reranker import rerank

load_dotenv()

# Load the embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 10))

model = SentenceTransformer(EMBEDDING_MODEL)

# Load FAISS index & metadata
#index_path = "embeddings/faiss_index"
#metadata_path = "embeddings/faiss_index_metadata.npy"

index_path = "embeddings/faiss_index_for_Test"
metadata_path = "embeddings/faiss_index_metadata_for_Test.npy"

index = faiss.read_index(index_path)
metadata = np.load(metadata_path, allow_pickle=True)


def search_query(query, top_k=RETRIEVAL_TOP_K):
    """Search FAISS for the top_k most relevant chunks, then rerank and return the best ones."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)

    candidates = [metadata[i] for i in indices[0]]
    reranked = rerank(query, candidates)
    return [{"chunk": chunk} for chunk in reranked]


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
            print(f"- {res['chunk']}")
        print("\n" + "-"*50)

