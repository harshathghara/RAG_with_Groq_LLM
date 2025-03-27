import faiss
import numpy as np
import os
import groq
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index & metadata
#index_path = "embeddings/faiss_index"
#metadata_path = "embeddings/faiss_index_metadata.npy"

index_path = "embeddings/faiss_index_for_Test"
metadata_path = "embeddings/faiss_index_metadata_for_Test.npy"

index = faiss.read_index(index_path)
metadata = np.load(metadata_path, allow_pickle=True)

# Load Groq API key
GROQ_API_KEY = "Your_API_KEY"
client = groq.Groq(api_key=GROQ_API_KEY)  # No need for base_url


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
        model="gemma2-9b-it",
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

