# RAG-Based PDF Search Engine with Groq LLM

A **Retrieval-Augmented Generation (RAG)** pipeline that lets you ask natural language questions against any PDF and get intelligent, well-structured answers powered by Groq's LLM.

---

## How It Works

1. **Text Extraction** — Extracts and cleans text from PDFs using `PyMuPDF` (removes bullets, symbols, and empty lines).
2. **Chunking** — Splits text into fixed-size overlapping character chunks (configurable via `.env`) to preserve context at chunk boundaries.
3. **Embedding & Indexing** — Encodes chunks into dense vectors using `SentenceTransformers` and stores them in a `FAISS` index for fast cosine-similarity search. A `BM25Okapi` keyword index is built from the same chunks and pickled alongside the FAISS files.
4. **Hybrid Retrieval** — At query time, two ranked lists are produced independently:
   - **Dense** — FAISS cosine-similarity search over the vector index.
   - **Sparse** — BM25 keyword scoring over the tokenised chunks.
   The two lists are merged into one via **Reciprocal Rank Fusion (RRF)**, giving every chunk a combined score that rewards appearing highly in either list.
5. **Reranking** — A `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the fused top-N candidates against the query and keeps the most relevant ones.
6. **LLM Response** — The reranked chunks are passed as context to Groq's LLM, which generates a clear, structured answer.

---

## Evaluation

The two-stage retrieval pipeline was benchmarked on a 20-question set derived from a sample PDF:

| Stage | Hit Rate @ 3 |
|---|---|
| FAISS-only (cosine similarity) | 80% |
| FAISS + CrossEncoder reranker | **95%** |

> **Resume-ready metric:** Improved retrieval hit rate from 80% (FAISS-only) to 95% (with cross-encoder reranking) on a 20-query benchmark (Precision@3).

Run `python src/evaluate.py` to reproduce these numbers against your own PDF and test cases.

---

## Project Structure

```
RAG_with_Groq_LLM/
│
├── data/                         # Input PDF files
│   └── Test.pdf
│
├── processed_text/               # Extracted plain text (auto-created)
│   └── Test.txt
│
├── embeddings/                   # FAISS index, BM25 index, and chunk metadata (auto-created)
│   ├── faiss_index_<name>
│   ├── faiss_index_metadata_<name>.npy
│   └── bm25_index_<name>.pkl
│
├── src/
│   ├── __init__.py               # Exposes public API for pipeline imports
│   ├── extract_text.py           # PDF text extraction and cleaning
│   ├── generate_embeddings.py    # Chunking, embedding, and FAISS indexing
│   ├── retriever.py              # Retriever class (lazy-loading, hybrid FAISS+BM25 with RRF + reranker)
│   ├── reranker.py               # CrossEncoder reranking
│   ├── query_engine.py           # Interactive search CLI (no LLM)
│   ├── llm_response.py           # ask_groq() — Groq LLM call
│   └── evaluate.py               # Hit-rate benchmark (FAISS-only vs reranked)
│
├── templates/
│   └── index.html                # Web UI (sidebar + chat, embedded CSS & JS)
│
├── uploads/                      # Uploaded PDFs (auto-created, gitignored)
│
├── app.py                        # FastAPI web server (upload, query, delete routes)
├── pipeline.py                   # CLI entry point for the full pipeline
├── ENHANCEMENTS.md               # Tracked list of upcoming features
├── .env                          # Environment variables (not committed)
├── .env.example                  # Template for .env
├── requirements.txt
└── README.md
```

---

## Web UI

The project ships with a full browser-based interface powered by **FastAPI**.

### Start the server

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

### Features

- **Drag-and-drop PDF upload** — drop any PDF onto the sidebar or click to browse
- **Live indexing status** — each document shows a live badge (Queued / Indexing / Ready / Error) that updates automatically while the PDF is being processed
- **Multi-PDF support** — upload as many PDFs as you want and switch between them instantly; each keeps its own index and chat context
- **Streaming answers** — Groq responses stream token-by-token as they arrive, just like ChatGPT
- **Retrieved context** — each answer has a collapsible "Retrieved context" section showing the exact chunks the LLM used
- **Delete documents** — trash icon on each document removes it from the sidebar and deletes all associated files from disk (uploaded PDF, extracted text, FAISS index, BM25 index, metadata); if a document is still indexing when deleted, the background process is cancelled and any partial files are cleaned up automatically

### Routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the UI |
| `POST` | `/upload` | Upload a PDF; starts background indexing |
| `GET` | `/status/{doc_id}` | Returns current processing status |
| `GET` | `/documents` | Lists all known documents |
| `POST` | `/query` | Streams a RAG answer via Server-Sent Events |
| `DELETE` | `/documents/{doc_id}` | Removes document and all its files |

### Production deployment

Replace `python app.py` with:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
```

---

## Quickstart — CLI Pipeline

If you prefer the terminal over the web UI, use `pipeline.py`. It chains every step automatically.

**1. Set your PDF path** — open `pipeline.py` and change the one line at the top:

```python
PDF_PATH = "data/Your_Document.pdf"
```

**2. Run the pipeline:**

```bash
python pipeline.py
```

This will:
- Extract and clean text from the PDF
- Chunk, embed, and build a FAISS index
- Start an interactive Q&A loop powered by Groq

The index files are named after your PDF (`faiss_index_YourDocument`, etc.), so you can index multiple PDFs without overwriting each other.

---

## Manual Step-by-Step Usage

If you prefer to run each step individually:

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Fill in `.env`:

```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=gemma2-9b-it
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_TOKENS=1024
CHUNK_SIZE=250
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=10
RERANK_TOP_K=3
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RRF_K=60
```

### 3. Extract text from your PDF

```bash
python src/extract_text.py
```

### 4. Generate embeddings and build FAISS index

```bash
python src/generate_embeddings.py
```

> Re-run this step whenever you change `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `EMBEDDING_MODEL`.

### 5. Search without LLM

```bash
python src/query_engine.py
```

### 6. Full RAG with LLM responses

```bash
python src/llm_response.py
```

### 7. Evaluate retrieval quality

```bash
python src/evaluate.py
```

> Edit the `TEST_CASES` list in `evaluate.py` to add question/answer pairs specific to your PDF.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Your Groq API key (required) |
| `GROQ_MODEL` | `gemma2-9b-it` | Groq model for answer generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model for encoding |
| `MAX_TOKENS` | `1024` | Maximum tokens in the LLM response |
| `CHUNK_SIZE` | `250` | Characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlapping characters between consecutive chunks |
| `RETRIEVAL_TOP_K` | `10` | Candidate chunks fetched from FAISS |
| `RERANK_TOP_K` | `3` | Final chunks passed to LLM after reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder model for reranking |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant (higher = smoother score merging between BM25 and vector lists) |
| `FAISS_INDEX_PATH` | `embeddings/faiss_index_Test` | Override path to FAISS index |
| `FAISS_METADATA_PATH` | `embeddings/faiss_index_metadata_Test.npy` | Override path to metadata file |

---

## Contributors

**Harsh Kumar** — AI/ML Engineer  
harshathghara19@gmail.com
