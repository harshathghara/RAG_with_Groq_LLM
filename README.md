# RAG-Based PDF Search Engine with Groq LLM

A **Retrieval-Augmented Generation (RAG)** pipeline that lets you ask natural language questions against any PDF and get intelligent, well-structured answers powered by Groq's LLM.

---

## How It Works

1. **Text Extraction** — Extracts and cleans text from PDFs using `PyMuPDF` (removes bullets, symbols, and empty lines). Page-break markers (`<<<PAGE:N>>>`) are embedded in the output so that page numbers are traceable downstream.
2. **Chunking** — Splits text into fixed-size overlapping character chunks (configurable via `.env`) to preserve context at chunk boundaries. Each chunk is stored as a metadata dict `{"text", "page", "chunk_idx"}` so source location travels with every chunk.
3. **Embedding & Indexing** — Encodes chunks into dense vectors using `SentenceTransformers` and stores them in a `FAISS` index for fast cosine-similarity search. A `BM25Okapi` keyword index is built from the same chunks and pickled alongside the FAISS files.
4. **Hybrid Retrieval** — At query time, two ranked lists are produced independently:
   - **Dense** — FAISS cosine-similarity search over the vector index.
   - **Sparse** — BM25 keyword scoring over the tokenised chunks.
   The two lists are merged into one via **Reciprocal Rank Fusion (RRF)**, giving every chunk a combined score that rewards appearing highly in either list.
5. **Reranking** — A `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the fused top-N candidates against the query. Each returned chunk carries a raw logit **confidence score** (converted to a percentage via sigmoid for display). The final result is a `list[dict]` with `text`, `page`, `chunk_idx`, and `score`.
6. **LLM Response** — The reranked chunks are passed as context to Groq's LLM, which generates a clear, structured answer. A second Groq call (`suggest_followups`) generates 4 follow-up question suggestions after the answer completes.

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
│   ├── extract_text.py           # PDF text extraction, cleaning, and page-marker injection
│   ├── generate_embeddings.py    # Page-aware chunking, embedding, FAISS + BM25 indexing; stores dict metadata
│   ├── retriever.py              # Retriever class — hybrid FAISS+BM25+RRF → CrossEncoder; returns list[dict] with page, chunk_idx, score
│   ├── reranker.py               # CrossEncoder reranking; returns (chunk_text, score) tuples
│   ├── query_engine.py           # Interactive search CLI (no LLM)
│   ├── llm_response.py           # ask_groq() — streaming Groq LLM call; suggest_followups() — 4 follow-up question chips
│   └── evaluate.py               # Hit-rate benchmark (FAISS-only vs FAISS+reranker vs BM25+FAISS+RRF+reranker)
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

- **Drag-and-drop PDF upload** — drop any PDF onto the sidebar or click to browse; document metadata (page count, file size) shown on the card
- **Live indexing status** — each document shows a live badge (Queued / Indexing / Ready / Error) that updates automatically while the PDF is being processed
- **Multi-PDF support** — upload as many PDFs as you want and switch between them instantly; each keeps its own index and chat context
- **Active document restored on refresh** — the last document you were working on is saved in `localStorage` and auto-selected when the page is reloaded; falls back to the most recently uploaded ready document if the saved one is gone
- **Streaming answers** — Groq responses stream token-by-token as they arrive; a three-dot typing indicator is shown while waiting for the first token, then transitions seamlessly into streaming text
- **Markdown rendering** — AI answers render formatted markdown (bold, lists, code blocks, headings, blockquotes) via `marked.js`
- **Retrieved context** — each answer has a collapsible section showing the exact chunks used, with a **page number** badge, **chunk position**, and a color-coded **confidence score** (CrossEncoder logit → sigmoid %) badge on each card
- **Suggested follow-up questions** — after each answer, 4 clickable follow-up chips are generated by Groq; a shimmer skeleton is shown immediately while they load so there is no jarring pop-in
- **Response action bar** — appears below each completed answer with three icon-only buttons (with hover tooltips): copy response, like, and dislike; like/dislike are mutually exclusive and highlighted when active
- **Dark / light mode toggle** — switch themes at any time; preference is saved in `localStorage`
- **Clear chat** — clears the conversation for the active document without removing the document itself
- **Download chat as PDF** — saves the full conversation as a clean PDF via `html2pdf.js`
- **Delete documents** — kebab menu on each document removes it from the sidebar and deletes all associated files from disk (uploaded PDF, extracted text, FAISS index, BM25 index, metadata); if a document is still indexing when deleted, the background process is cancelled and any partial files are cleaned up automatically

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
- Extract and clean text from the PDF (with page markers)
- Chunk, embed, and build a FAISS + BM25 index (dict metadata with page numbers)
- Start an interactive Q&A loop powered by Groq, printing `[N] p.X | YY.Y% | chunk text...` for each retrieved chunk

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

> This now embeds `<<<PAGE:N>>>` markers in the output text file so that each chunk can be traced back to its source page.

### 4. Generate embeddings and build FAISS + BM25 index

```bash
python src/generate_embeddings.py
```

> Re-run this step whenever you change `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `EMBEDDING_MODEL`, or after re-extracting text with the new page-marker format. The metadata file now stores dicts (`text`, `page`, `chunk_idx`) instead of plain strings — old `.npy` files built before this change are supported via a backward-compat fallback (page will show as `None`).

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
