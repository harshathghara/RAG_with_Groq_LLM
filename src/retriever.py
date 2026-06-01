import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from .reranker import rerank

load_dotenv()

EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RETRIEVAL_TOP_K  = int(os.getenv("RETRIEVAL_TOP_K", 10))
RRF_K            = int(os.getenv("RRF_K", 60))          # RRF constant (higher = smoother merging)


def _rrf_merge(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    """
    Reciprocal Rank Fusion over multiple ranked lists of chunk indices.

    Each list is a sequence of chunk indices ordered from most to least relevant.
    Returns a deduplicated list of indices sorted by descending RRF score.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


class Retriever:
    """
    Hybrid retriever: combines a FAISS dense index (vector search) with
    BM25 (keyword search) and merges the two ranked lists via Reciprocal
    Rank Fusion (RRF) before reranking the final candidates.

    All resources are loaded lazily on the first call to .search() to
    avoid I/O cost at import time.
    """

    def __init__(self, index_path: str, metadata_path: str, bm25_path: str | None = None):
        self._index_path    = index_path
        self._metadata_path = metadata_path
        self._bm25_path     = bm25_path
        self._model         = None
        self._index         = None
        self._metadata      = None
        self._bm25          = None

    def _load(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(EMBEDDING_MODEL)

        if self._index is None:
            if not os.path.exists(self._index_path):
                raise FileNotFoundError(
                    f"FAISS index not found at '{self._index_path}'. "
                    "Run generate_embeddings.py (or the pipeline) first."
                )
            self._index = faiss.read_index(self._index_path)

        if self._metadata is None:
            if not os.path.exists(self._metadata_path):
                raise FileNotFoundError(
                    f"Metadata file not found at '{self._metadata_path}'. "
                    "Run generate_embeddings.py (or the pipeline) first."
                )
            self._metadata = np.load(self._metadata_path, allow_pickle=True)

        if self._bm25 is None and self._bm25_path and os.path.exists(self._bm25_path):
            with open(self._bm25_path, "rb") as f:
                self._bm25 = pickle.load(f)

    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
        """
        Hybrid search: FAISS vector search + BM25 keyword search, merged
        via Reciprocal Rank Fusion, then reranked with CrossEncoder.

        Each returned dict has:
          text       – the chunk text
          page       – 1-based page number (None if index predates page tracking)
          chunk_idx  – position of the chunk within the document
          score      – raw CrossEncoder logit (sigmoid → confidence %)
        """
        self._load()

        # ── Dense (vector) retrieval ──────────────────────────────────────────
        query_embedding = self._model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        _, faiss_indices = self._index.search(query_embedding, top_k)
        dense_ranked = [int(i) for i in faiss_indices[0] if i >= 0]

        # ── Sparse (BM25) retrieval ───────────────────────────────────────────
        if self._bm25 is not None:
            tokenized_query = query.lower().split()
            bm25_scores     = self._bm25.get_scores(tokenized_query)
            sparse_ranked   = list(np.argsort(bm25_scores)[::-1][:top_k])
        else:
            sparse_ranked = []

        # ── Merge via RRF ─────────────────────────────────────────────────────
        merged_indices = (
            _rrf_merge([dense_ranked, sparse_ranked], k=RRF_K)
            if sparse_ranked else dense_ranked
        )

        # ── Build candidate list (handle old plain-string and new dict metadata) ──
        raw_candidates: list[dict] = []
        for i in merged_indices[:top_k]:
            item = self._metadata[i]
            if isinstance(item, dict):
                raw_candidates.append(item)
            else:
                # Backward compat: index built before page tracking was added
                raw_candidates.append({"text": str(item), "page": None, "chunk_idx": int(i)})

        candidate_texts = [c["text"] for c in raw_candidates]
        text_to_meta    = {c["text"]: c for c in raw_candidates}

        # ── CrossEncoder reranking ────────────────────────────────────────────
        reranked = rerank(query, candidate_texts)   # [(text, score), ...]

        results: list[dict] = []
        for text, score in reranked:
            meta = text_to_meta.get(text, {"text": text, "page": None, "chunk_idx": None})
            results.append({**meta, "score": score})
        return results
