import os
from dotenv import load_dotenv
from retriever import Retriever

load_dotenv()

INDEX_PATH    = os.getenv("FAISS_INDEX_PATH",    "embeddings/faiss_index_Test")
METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "embeddings/faiss_index_metadata_Test.npy")


if __name__ == "__main__":
    retriever = Retriever(INDEX_PATH, METADATA_PATH)

    print("\nPDF Search Engine  (type 'exit' to quit)\n")

    while True:
        query = input("Search: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            print("Goodbye!")
            break

        chunks = retriever.search(query)
        print("\nTop Results:")
        for i, chunk in enumerate(chunks, 1):
            page  = f"p.{chunk['page']} | " if chunk.get("page") else ""
            score = f"{100 / (1 + __import__('math').exp(-chunk['score'])):.1f}% | " if chunk.get("score") is not None else ""
            print(f"  [{i}] {page}{score}{chunk['text'][:120].strip()}")
        print("\n" + "-" * 50)
