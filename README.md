# RAG-Based PDF Search Engine with Groq LLM

A **Retrieval-Augmented Generation (RAG)** pipeline that lets you ask natural language questions against any PDF and get intelligent, well-structured answers powered by Groq's LLM.

---

## How It Works

1. **Text Extraction** — Extracts and cleans text from PDFs using `PyMuPDF`.
2. **Chunking** — Splits text into fixed-size overlapping chunks (configurable via `.env`) to preserve context at boundaries.
3. **Embedding & Indexing** — Encodes chunks into dense vectors using `SentenceTransformers` and stores them in a `FAISS` index for fast similarity search.
4. **Retrieval** — At query time, FAISS retrieves the top-N candidate chunks using cosine similarity.
5. **Reranking** — A `CrossEncoder` reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each candidate against the query and selects the most relevant ones.
6. **LLM Response** — The reranked chunks are passed as context to Groq's LLM, which generates a clear, structured answer.

---

## Features

- Extracts and cleans text from PDFs (removes bullets, symbols, empty lines)
- Fixed-size overlapping chunking with configurable `CHUNK_SIZE` and `CHUNK_OVERLAP`
- FAISS vector index with cosine similarity (L2-normalized inner product)
- Two-stage retrieval: FAISS candidate fetch → CrossEncoder reranking
- All key parameters controlled via `.env` — no code changes needed
- Groq LLM integration with a structured, user-friendly system prompt

---

## Project Structure

```
RAG_with_Groq_LLM/
│
├── data/                         # Input PDF files
│   └── Test.pdf
│
├── processed_text/               # Extracted plain text from PDFs
│   └── Test.txt
│
├── embeddings/                   # FAISS index and chunk metadata
│   ├── faiss_index_for_Test
│   └── faiss_index_metadata_for_Test.npy
│
├── src/
│   ├── extract_text.py           # PDF text extraction and cleaning
│   ├── generate_embeddings.py    # Chunking, embedding, and FAISS indexing
│   ├── reranker.py               # CrossEncoder reranking
│   ├── query_engine.py           # Interactive search with reranking
│   └── llm_response.py           # End-to-end RAG with Groq LLM
│
├── .env                          # Environment variables (not committed)
├── .env.example                  # Template for .env
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=gemma2-9b-it
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
MAX_TOKENS=1024
CHUNK_SIZE=500
CHUNK_OVERLAP=100
RETRIEVAL_TOP_K=10
RERANK_TOP_K=3
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### 3. Extract text from your PDF

```bash
python src/extract_text.py
```

### 4. Generate embeddings and build FAISS index

```bash
python src/generate_embeddings.py
```

> Re-run this step whenever you change `CHUNK_SIZE` or `CHUNK_OVERLAP`.

### 5. Run the search engine (no LLM)

```bash
python src/query_engine.py
```

### 6. Run the full RAG pipeline with LLM responses

```bash
python src/llm_response.py
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Your Groq API key |
| `GROQ_MODEL` | `gemma2-9b-it` | Groq model to use for responses |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | SentenceTransformer model for encoding |
| `MAX_TOKENS` | `1024` | Max tokens in the LLM response |
| `CHUNK_SIZE` | `500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `100` | Overlapping characters between chunks |
| `RETRIEVAL_TOP_K` | `10` | Candidate chunks fetched from FAISS |
| `RERANK_TOP_K` | `3` | Chunks passed to LLM after reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder model for reranking |

---

## Contributors

**Harsh Kumar** — AI/ML Engineer  
harshathghara19@gmail.com
