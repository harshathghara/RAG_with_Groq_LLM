import os
import sys

# Make sure the src package is importable when running from the project root
sys.path.insert(0, os.path.dirname(__file__))

from src import extract_text_from_pdf, generate_embeddings, Retriever, ask_groq

# ── Configuration ─────────────────────────────────────────────────────────────
# Change this path to point at any PDF you want to query.
PDF_PATH         = "data/Test.pdf"

TEXT_OUTPUT_DIR  = "processed_text/"
INDEX_OUTPUT_DIR = "embeddings/"
# ──────────────────────────────────────────────────────────────────────────────


def run_pipeline(pdf_path: str) -> None:
    """
    Full RAG pipeline:
      1. Extract text from the PDF
      2. Chunk, embed, and build a FAISS index
      3. Start an interactive Q&A loop powered by Groq
    """

    # ── Step 1: Extract ───────────────────────────────────────────────────────
    print("\n[1/3] Extracting text from PDF...")
    os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)
    text_file = extract_text_from_pdf(pdf_path, TEXT_OUTPUT_DIR)

    # ── Step 2: Embed ─────────────────────────────────────────────────────────
    print("\n[2/3] Generating embeddings and building FAISS index...")
    index_path, metadata_path = generate_embeddings(text_file, INDEX_OUTPUT_DIR)

    # ── Step 3: Query loop ────────────────────────────────────────────────────
    print("\n[3/3] Pipeline ready — starting interactive Q&A.\n")
    retriever = Retriever(index_path, metadata_path)

    print("Type your question and press Enter.  Type 'exit' to quit.\n")
    print("-" * 60)

    while True:
        query = input("\nYour question: ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Retrieve relevant chunks (FAISS + reranker)
        chunks = retriever.search(query)
        context = "\n".join(chunks)

        print("\nRetrieved Context:")
        print(context)

        # Generate answer with Groq
        print("\nGroq AI Response:")
        print(ask_groq(query, context))
        print("\n" + "-" * 60)


if __name__ == "__main__":
    run_pipeline(PDF_PATH)
