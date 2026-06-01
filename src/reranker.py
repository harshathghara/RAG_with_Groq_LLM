import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 3))

reranker = CrossEncoder(RERANKER_MODEL)


def rerank(query: str, chunks: list[str], top_k: int = RERANK_TOP_K) -> list[tuple[str, float]]:
    """
    Score each chunk against the query with a cross-encoder.
    Returns a list of (chunk_text, raw_logit_score) tuples, best first.
    The raw logit can be converted to a confidence percentage via sigmoid.
    """
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [(chunk, float(score)) for score, chunk in ranked[:top_k]]
