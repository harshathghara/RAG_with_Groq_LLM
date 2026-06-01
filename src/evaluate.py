"""
Retrieval Evaluation Script
============================
Compares three retrieval modes side-by-side:
  - Stage 1 (FAISS-only):                  top-K chunks by cosine similarity
  - Stage 2 (FAISS + CrossEncoder):         same candidates, reranked
  - Stage 3 (BM25 + FAISS + RRF + Rerank): hybrid retrieval then reranked

Metric: Hit Rate @ K  (did any returned chunk contain the expected answer?)

Usage
-----
1. Fill in the TEST_CASES list below with question/answer pairs from your PDF.
2. Run from the project root:
       python src/evaluate.py
3. Use the printed comparison table to quote a real metric on your resume.
"""

import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(__file__))

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from reranker import rerank, RERANK_TOP_K

load_dotenv()

# ---------------------------------------------------------------------------
# TEST CASES  ← fill these in with questions and a short phrase / keyword
#               that must appear in a correct chunk (case-insensitive)
# ---------------------------------------------------------------------------
TEST_CASES = [
    # Personal details
    ("What is the candidate's name?",                          "harsh kumar"),
    ("What is the candidate's email address?",                 "haku21aiml@cmrit.ac.in"),
    ("What is the candidate's mobile number?",                 "9608592615"),
    ("Where is the candidate currently located?",              "bengaluru"),

    # Education
    ("Which college did Harsh Kumar attend?",                  "cmr institute of technology"),
    ("What is the candidate's CGPA?",                         "8.94"),
    ("What branch of engineering did the candidate study?",    "artificial intelligence"),
    ("What percentage did the candidate score in 10th grade?", "95%"),

    # Skills
    ("What programming languages does the candidate know?",    "python"),
    ("What databases has the candidate worked with?",          "mysql"),

    # Projects
    ("What is the Women Safety Analytics project about?",      "surveillance"),
    ("What tools were used in the diabetes prediction project?","scikit-learn"),
    ("Which computer vision tools were used in the projects?", "opencv"),

    # Internship
    ("Where did the candidate intern?",                        "aura solutions"),
    ("What was the candidate's role during the internship?",   "data analyst"),

    # Achievements
    ("What hackathon did the candidate win first place in?",   "hackathon"),
    ("What sports has the candidate played?",                  "cricket"),

    # Certifications
    ("What online certifications does the candidate have?",    "nptel"),
    ("Which coding platform did the candidate learn DSA from?","coding ninjas"),

    # References
    ("Who is the candidate's reference from CMRIT?",           "shyam p joy"),
]
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 10))
RRF_K           = int(os.getenv("RRF_K", 60))

index_path    = os.getenv("FAISS_INDEX_PATH",    "embeddings/faiss_index_Test")
metadata_path = os.getenv("FAISS_METADATA_PATH", "embeddings/faiss_index_metadata_Test.npy")

# Derive BM25 path from the FAISS index path automatically
_stem     = os.path.basename(index_path).replace("faiss_index_", "")
bm25_path = os.path.join(os.path.dirname(index_path), f"bm25_index_{_stem}.pkl")

model    = SentenceTransformer(EMBEDDING_MODEL)
index    = faiss.read_index(index_path)
metadata = np.load(metadata_path, allow_pickle=True)

bm25: BM25Okapi | None = None
if os.path.exists(bm25_path):
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
else:
    print(f"[warn] BM25 index not found at '{bm25_path}'. "
          "Stage 3 will be skipped. Re-run generate_embeddings.py to create it.")


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _rrf_merge(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def faiss_only(query: str, top_k: int = RERANK_TOP_K) -> list[str]:
    """Return top_k chunks by FAISS cosine similarity, no reranking."""
    embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(embedding)
    _, indices = index.search(embedding, top_k)
    return [metadata[i] for i in indices[0]]


def faiss_plus_rerank(query: str, candidate_k: int = RETRIEVAL_TOP_K) -> list[str]:
    """Fetch candidate_k chunks from FAISS, then rerank and return top RERANK_TOP_K."""
    embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(embedding)
    _, indices = index.search(embedding, candidate_k)
    candidates = [metadata[i] for i in indices[0]]
    return rerank(query, candidates)


def hybrid_plus_rerank(query: str, candidate_k: int = RETRIEVAL_TOP_K) -> list[str] | None:
    """BM25 + FAISS ranked lists merged with RRF, then reranked."""
    if bm25 is None:
        return None
    # Dense
    embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(embedding)
    _, faiss_indices = index.search(embedding, candidate_k)
    dense_ranked = [int(i) for i in faiss_indices[0] if i >= 0]
    # Sparse
    bm25_scores   = bm25.get_scores(query.lower().split())
    sparse_ranked = list(np.argsort(bm25_scores)[::-1][:candidate_k])
    # Merge
    merged = _rrf_merge([dense_ranked, sparse_ranked], k=RRF_K)
    candidates = [str(metadata[i]) for i in merged[:candidate_k]]
    return rerank(query, candidates)


def hit(chunks: list[str], keyword: str) -> bool:
    """Return True if any chunk contains the keyword (case-insensitive)."""
    return any(keyword.lower() in chunk.lower() for chunk in chunks)


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate():
    if not TEST_CASES:
        print("\n  No test cases defined.")
        print("  Open src/evaluate.py and fill in the TEST_CASES list, then re-run.\n")
        return

    has_bm25 = bm25 is not None
    col_w    = 52

    stage1_hits = stage2_hits = stage3_hits = 0

    if has_bm25:
        header = (f"{'Query':<{col_w}} {'FAISS-only':^11} "
                  f"{'+ Reranker':^11} {'Hybrid+RR':^11}")
    else:
        header = f"{'Query':<{col_w}} {'FAISS-only':^11} {'+ Reranker':^11}"

    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for query, keyword in TEST_CASES:
        s1 = hit(faiss_only(query),        keyword)
        s2 = hit(faiss_plus_rerank(query), keyword)
        stage1_hits += int(s1)
        stage2_hits += int(s2)

        short_q = (query[:col_w - 3] + "...") if len(query) > col_w else query
        row = (f"{short_q:<{col_w}} "
               f"{'HIT' if s1 else 'MISS':^11} "
               f"{'HIT' if s2 else 'MISS':^11}")

        if has_bm25:
            s3_chunks = hybrid_plus_rerank(query)
            s3 = hit(s3_chunks, keyword) if s3_chunks else False
            stage3_hits += int(s3)
            row += f" {'HIT' if s3 else 'MISS':^11}"

        print(row)

    total = len(TEST_CASES)
    p1    = stage1_hits / total * 100
    p2    = stage2_hits / total * 100
    d12   = p2 - p1

    print("-" * len(header))
    summary = (f"{'Hit Rate @ ' + str(RERANK_TOP_K):<{col_w}} "
               f"{p1:^10.1f}% {p2:^10.1f}%")

    if has_bm25:
        p3  = stage3_hits / total * 100
        d23 = p3 - p2
        d13 = p3 - p1
        summary += f" {p3:^10.1f}%"

    print(summary)
    print("=" * len(header))

    print(f"\n  Stage 1 -> Stage 2  (reranking alone):        {d12:+.1f} pp")
    if has_bm25:
        print(f"  Stage 2 -> Stage 3  (adding hybrid BM25+RRF): {d23:+.1f} pp")
        print(f"  Stage 1 -> Stage 3  (total improvement):      {d13:+.1f} pp")

    best_p   = p3 if has_bm25 else p2
    best_lbl = "hybrid BM25+RRF + cross-encoder reranking" if has_bm25 else "cross-encoder reranking"

    if best_p > p1:
        print(f"\n  Resume-ready metric:")
        print(f"  \"Improved retrieval hit rate from {p1:.0f}% (FAISS-only) to "
              f"{best_p:.0f}% (with {best_lbl}) on a "
              f"{total}-query benchmark (Precision@{RERANK_TOP_K}).\"\n")
    elif best_p == p1:
        print("  All pipelines scored the same — try more diverse test cases.\n")
    else:
        print("  FAISS-only scored highest — consider tuning RETRIEVAL_TOP_K or RRF_K.\n")


if __name__ == "__main__":
    evaluate()
