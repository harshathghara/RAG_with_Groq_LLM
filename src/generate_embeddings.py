import os
import re
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 100))

# Lazily initialised on first call to generate_embeddings() so that simply
# importing this module does not trigger a heavyweight model load at startup.
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

# Marker format written by extract_text.py
_PAGE_MARKER = re.compile(r"<<<PAGE:(\d+)>>>")


def _parse_pages(raw_text: str) -> tuple[str, list[tuple[int, int]]]:
    """
    Strip <<<PAGE:N>>> markers from *raw_text* and return:
      - clean_text  : the text with all markers removed
      - boundaries  : list of (char_offset_in_clean_text, page_number) sorted by offset

    char_offset is the position in *clean_text* where that page's content begins.
    """
    boundaries: list[tuple[int, int]] = []
    parts: list[str] = []
    cursor = 0          # position in clean text built so far
    last_end = 0        # position in raw_text after the last match

    for m in _PAGE_MARKER.finditer(raw_text):
        # Text before this marker (may be empty / newlines)
        prefix = raw_text[last_end:m.start()]
        parts.append(prefix)
        cursor += len(prefix)
        # Content for this page starts right after the marker
        boundaries.append((cursor, int(m.group(1))))
        last_end = m.end()

    parts.append(raw_text[last_end:])   # tail after last marker
    return "".join(parts), boundaries


def _page_for(pos: int, boundaries: list[tuple[int, int]]) -> int | None:
    """Return the page number that contains character offset *pos* in clean text."""
    if not boundaries:
        return None
    page = boundaries[0][1]
    for offset, page_num in boundaries:
        if pos >= offset:
            page = page_num
        else:
            break
    return page


def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 100) -> list[tuple[str, int]]:
    """
    Split *text* into fixed-size character chunks with overlap.
    Returns a list of (chunk_text, start_pos) tuples.
    """
    results: list[tuple[str, int]] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            results.append((chunk, start))
        start += chunk_size - overlap
    return results


def generate_embeddings(text_file: str, index_folder: str):
    """Generate embeddings for a given text file and store in FAISS."""
    with open(text_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    clean_text, boundaries = _parse_pages(raw_text)
    chunk_tuples = chunk_fixed_size(clean_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    # Build rich metadata dicts so every chunk carries its source location
    chunks      = [text for text, _ in chunk_tuples]
    chunk_metas = [
        {"text": text, "page": _page_for(pos, boundaries), "chunk_idx": i}
        for i, (text, pos) in enumerate(chunk_tuples)
    ]

    embeddings = _get_model().encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    stem = os.path.splitext(os.path.basename(text_file))[0]
    os.makedirs(index_folder, exist_ok=True)

    index_path    = os.path.join(index_folder, f"faiss_index_{stem}")
    metadata_path = os.path.join(index_folder, f"faiss_index_metadata_{stem}.npy")
    bm25_path     = os.path.join(index_folder, f"bm25_index_{stem}.pkl")

    faiss.write_index(index, index_path)
    # Store dicts (allow_pickle required for object arrays)
    np.save(metadata_path, np.array(chunk_metas, dtype=object))

    tokenized = [c.lower().split() for c in chunks]
    bm25      = BM25Okapi(tokenized)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    print(f"FAISS index saved:  {index_path}")
    print(f"Metadata saved:     {metadata_path}  ({len(chunk_metas)} chunks)")
    print(f"BM25 index saved:   {bm25_path}")
    return index_path, metadata_path, bm25_path

if __name__ == "__main__":
    text_file = "processed_text/Test.txt"
    index_folder = "embeddings/"
    generate_embeddings(text_file, index_folder)
