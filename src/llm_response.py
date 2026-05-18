import faiss
import numpy as np
import os
import groq
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

# Load Groq credentials
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "gemma2-9b-it")
client = groq.Groq(api_key=GROQ_API_KEY)


def search_query(query, top_k=2):
    """Search FAISS for the most relevant text chunks."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = [metadata[i] for i in indices[0]]
    return results

def ask_groq(query, context):
    """Generate a response from Groq using retrieved context."""
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )

    return response.choices[0].message.content  # Correct output format



if __name__ == "__main__":
    print("\n🔍 PDF Search Engine with LLM (Type 'exit' to quit)\n")
    
    while True:
        query = input("Enter your search query: ")
        if query.lower() == "exit":
            print("Exiting search engine. Goodbye! 👋")
            break

        retrieved_text = search_query(query)
        context = "\n".join(retrieved_text)
        print("\n📄 Retrieved Context:\n", context)

        print("\n🤖 Groq AI Response:")
        print(ask_groq(query, context))
        print("\n" + "-"*50)

