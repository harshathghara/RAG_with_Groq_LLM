import os
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from .reranker import rerank

load_dotenv()

EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RETRIEVAL_TOP_K  = int(os.getenv("RETRIEVAL_TOP_K", 10))


class Retriever:
    """
    Wraps a FAISS index and handles the full retrieve-then-rerank flow.

    The embedding model, index, and metadata are loaded lazily on the
    first call to .search() so that importing the class has zero I/O cost.
    """

    def __init__(self, index_path: str, metadata_path: str):
        self._index_path    = index_path
        self._metadata_path = metadata_path
        self._model         = None
        self._index         = None
        self._metadata      = None

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

    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[str]:
        """
        Retrieve top_k candidate chunks from FAISS, then rerank them.

        Returns a list of text chunks ordered by relevance (best first).
        """
        self._load()

        query_embedding = self._model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        _, indices = self._index.search(query_embedding, top_k)

        candidates = [self._metadata[i] for i in indices[0]]
        return rerank(query, candidates)
