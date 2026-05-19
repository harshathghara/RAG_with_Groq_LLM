import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 3))

reranker = CrossEncoder(RERANKER_MODEL)


def rerank(query, chunks, top_k=RERANK_TOP_K):
    """Score each chunk against the query with a cross-encoder and return top_k."""
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]
