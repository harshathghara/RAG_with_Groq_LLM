import os
import sys
import json
import uuid
import asyncio
import threading
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from src import extract_text_from_pdf, generate_embeddings, Retriever

load_dotenv()

app       = FastAPI(title="RAG PDF Chat")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
TEXT_DIR   = "processed_text"
INDEX_DIR  = "embeddings"

for d in (UPLOAD_DIR, TEXT_DIR, INDEX_DIR):
    os.makedirs(d, exist_ok=True)

# In-memory document registry
# docs[doc_id] = {name, status, index_path, metadata_path, retriever, error}
docs: dict     = {}
docs_lock      = threading.Lock()


# ── Background PDF processing ─────────────────────────────────────────────────

def _remove_if_exists(*paths: str) -> None:
    for p in paths:
        if p and os.path.exists(p):
            os.remove(p)


def _process_pdf(doc_id: str, pdf_path: str, cancel: threading.Event) -> None:
    """
    Runs in a daemon thread: extract → embed → mark ready.

    Checks `cancel` between each expensive step. If set, any files already
    written to disk are deleted and the thread exits without updating docs.
    """
    with docs_lock:
        if cancel.is_set():
            return
        docs[doc_id]["status"] = "processing"

    text_file  = None
    index_path = None
    meta_path  = None
    bm25_path  = None

    try:
        text_file = extract_text_from_pdf(pdf_path, TEXT_DIR)

        if cancel.is_set():
            _remove_if_exists(text_file)
            return

        index_path, meta_path, bm25_path = generate_embeddings(text_file, INDEX_DIR)

        if cancel.is_set():
            _remove_if_exists(text_file, index_path, meta_path, bm25_path)
            return

        with docs_lock:
            # doc may have been popped if delete arrived during generate_embeddings
            if doc_id in docs and not cancel.is_set():
                docs[doc_id].update(
                    status="ready",
                    text_file=text_file,
                    index_path=index_path,
                    metadata_path=meta_path,
                    bm25_path=bm25_path,
                )
            else:
                # Cancelled right at the finish line — clean up
                _remove_if_exists(text_file, index_path, meta_path, bm25_path)

    except Exception as exc:
        with docs_lock:
            if doc_id in docs:
                docs[doc_id].update(status="error", error=str(exc))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    doc_id   = uuid.uuid4().hex[:10]
    pdf_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{pdf.filename}")

    contents = await pdf.read()
    with open(pdf_path, "wb") as f:
        f.write(contents)

    cancel = threading.Event()
    with docs_lock:
        docs[doc_id] = {
            "name":     pdf.filename,
            "status":   "pending",
            "pdf_path": pdf_path,
            "cancel":   cancel,
        }

    # CPU-heavy work runs in a daemon thread so the event loop stays free
    t = threading.Thread(target=_process_pdf, args=(doc_id, pdf_path, cancel), daemon=True)
    t.start()

    return {"doc_id": doc_id, "name": pdf.filename}


@app.get("/status/{doc_id}")
async def status(doc_id: str):
    with docs_lock:
        doc = docs.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    payload = {"status": doc["status"], "name": doc["name"]}
    if doc.get("error"):
        payload["error"] = doc["error"]
    return payload


@app.get("/documents")
async def documents():
    with docs_lock:
        snapshot = list(docs.items())
    return [
        {"id": k, "name": v["name"], "status": v["status"]}
        for k, v in snapshot
    ]


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    with docs_lock:
        doc = docs.pop(doc_id, None)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Signal the background thread to abort and self-clean if still running
    cancel: threading.Event | None = doc.get("cancel")
    if cancel:
        cancel.set()

    # Delete any files that already exist on disk right now.
    # Files written by the thread *after* this point are cleaned up by the
    # thread itself when it detects the cancel flag.
    _remove_if_exists(
        doc.get("pdf_path"),
        doc.get("text_file"),
        doc.get("index_path"),
        doc.get("metadata_path"),
        doc.get("bm25_path"),
    )

    return {"deleted": doc_id}


class QueryRequest(BaseModel):
    doc_id:   str
    question: str


@app.post("/query")
async def query(body: QueryRequest):
    question = body.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    with docs_lock:
        doc = docs.get(body.doc_id)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if doc["status"] != "ready":
        raise HTTPException(status_code=400, detail="Document is still being processed. Please wait.")

    # Lazy-init the Retriever (one per document, reused across queries)
    if "retriever" not in doc:
        with docs_lock:
            if "retriever" not in doc:
                doc["retriever"] = Retriever(doc["index_path"], doc["metadata_path"], doc.get("bm25_path"))

    retriever    = doc["retriever"]
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model   = os.getenv("GROQ_MODEL", "gemma2-9b-it")
    max_tokens   = int(os.getenv("MAX_TOKENS", 1024))

    async def event_generator() -> AsyncGenerator[str, None]:
        import groq as groq_module

        # Retrieval is sync + CPU-bound — run in thread pool
        chunks  = await asyncio.to_thread(retriever.search, question)
        context = "\n\n".join(chunks)

        yield f"data: {json.dumps({'type': 'context', 'text': context})}\n\n"

        # Groq SDK is synchronous; pipe tokens to the async generator via a Queue.
        # The groq_worker thread puts tokens into the queue; we await them here.
        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def groq_worker() -> None:
            try:
                client = groq_module.Groq(api_key=groq_api_key)
                stream = client.chat.completions.create(
                    model=groq_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer using only the provided context. Be concise and clear."},
                        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
                    ],
                    max_tokens=max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        asyncio.run_coroutine_threadsafe(q.put(delta), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(q.put(exc), loop)
            finally:
                asyncio.run_coroutine_threadsafe(q.put(None), loop)  # sentinel

        threading.Thread(target=groq_worker, daemon=True).start()

        while True:
            token = await q.get()
            if token is None:
                break
            if isinstance(token, Exception):
                yield f"data: {json.dumps({'type': 'error', 'text': str(token)})}\n\n"
                break
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="localhost", port=5000, reload=True)
