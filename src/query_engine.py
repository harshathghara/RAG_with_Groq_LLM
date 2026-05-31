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
        for chunk in chunks:
            print(f"  - {chunk}")
        print("\n" + "-" * 50)
