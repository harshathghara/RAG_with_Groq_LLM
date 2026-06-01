import os
import json
import groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "gemma2-9b-it")
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", 1024))

client = groq.Groq(api_key=GROQ_API_KEY)


def ask_groq(query: str, context: str) -> str:
    """Generate an answer from Groq using the retrieved context chunks."""
    prompt = (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
    )

    return response.choices[0].message.content


def suggest_followups(query: str, answer: str, context: str) -> list[str]:
    """
    Ask Groq to generate 2-3 short follow-up questions based on the Q&A exchange.
    Returns a list of question strings, or [] on any failure.
    """
    prompt = (
        "Given the question, answer, and context below, suggest exactly 4 short "
        "follow-up questions the user might want to ask next. "
        "Return ONLY a valid JSON array of exactly 4 strings, e.g. [\"Q1?\", \"Q2?\", \"Q3?\", \"Q4?\"]. "
        "No explanation, no markdown, no extra text — just the JSON array.\n\n"
        f"Question: {query}\n\n"
        f"Answer: {answer}\n\n"
        f"Context excerpt: {context[:600]}"
    )
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You generate follow-up questions as a JSON array."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=200,
        )
        raw   = response.choices[0].message.content.strip()
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        pass
    return []
