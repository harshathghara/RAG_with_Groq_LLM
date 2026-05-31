"""
Retrieval Evaluation Script
============================
Compares two retrieval modes side-by-side:
  - Stage 1 (FAISS-only):           top-K chunks by cosine similarity
  - Stage 2 (FAISS + CrossEncoder): same candidates, reranked

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
sys.path.insert(0, os.path.dirname(__file__))

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
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

EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RETRIEVAL_TOP_K  = int(os.getenv("RETRIEVAL_TOP_K", 10))

index_path    = os.getenv("FAISS_INDEX_PATH",    "embeddings/faiss_index_Test")
metadata_path = os.getenv("FAISS_METADATA_PATH", "embeddings/faiss_index_metadata_Test.npy")

model    = SentenceTransformer(EMBEDDING_MODEL)
index    = faiss.read_index(index_path)
metadata = np.load(metadata_path, allow_pickle=True)


def faiss_only(query: str, top_k: int = RERANK_TOP_K) -> list[str]:
    """Return top_k chunks by FAISS cosine similarity, no reranking."""
    embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(embedding)
    distances, indices = index.search(embedding, top_k)
    return [metadata[i] for i in indices[0]]


def faiss_plus_rerank(query: str, candidate_k: int = RETRIEVAL_TOP_K) -> list[str]:
    """Fetch candidate_k chunks from FAISS, then rerank and return top RERANK_TOP_K."""
    embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(embedding)
    distances, indices = index.search(embedding, candidate_k)
    candidates = [metadata[i] for i in indices[0]]
    return rerank(query, candidates)


def hit(chunks: list[str], keyword: str) -> bool:
    """Return True if any chunk contains the keyword (case-insensitive)."""
    return any(keyword.lower() in chunk.lower() for chunk in chunks)


def evaluate():
    if not TEST_CASES:
        print("\n  No test cases defined.")
        print("  Open src/evaluate.py and fill in the TEST_CASES list, then re-run.\n")
        return

    stage1_hits = 0
    stage2_hits = 0
    col_w = 52

    header = f"{'Query':<{col_w}} {'FAISS-only':^10} {'+ Reranker':^10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for query, keyword in TEST_CASES:
        s1 = hit(faiss_only(query),        keyword)
        s2 = hit(faiss_plus_rerank(query), keyword)
        stage1_hits += int(s1)
        stage2_hits += int(s2)

        short_q = (query[:col_w - 3] + "...") if len(query) > col_w else query
        print(f"{short_q:<{col_w}} {'HIT' if s1 else 'MISS':^10} {'HIT' if s2 else 'MISS':^10}")

    total = len(TEST_CASES)
    p1 = stage1_hits / total * 100
    p2 = stage2_hits / total * 100
    delta = p2 - p1

    print("-" * len(header))
    print(f"{'Hit Rate @ ' + str(RERANK_TOP_K):<{col_w}} {p1:^9.1f}% {p2:^9.1f}%")
    print("=" * len(header))

    print(f"\n  Improvement from reranking: {delta:+.1f} percentage points")
    if delta > 0:
        print(f"\n  Resume-ready metric:")
        print(f"  \"Improved retrieval hit rate from {p1:.0f}% (FAISS-only) to "
              f"{p2:.0f}% (with cross-encoder reranking) on a "
              f"{total}-query benchmark (Precision@{RERANK_TOP_K}).\"\n")
    elif delta == 0:
        print("  Both pipelines scored the same — try more diverse test cases.\n")
    else:
        print("  FAISS-only scored higher — consider tuning RETRIEVAL_TOP_K or RERANK_TOP_K.\n")


if __name__ == "__main__":
    evaluate()
