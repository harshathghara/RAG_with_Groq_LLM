import os
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
