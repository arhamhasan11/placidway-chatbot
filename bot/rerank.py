"""Re-orders retrieved chunks by how directly each one answers the question.

Called by bot/chat.py between retrieval and drafting. Search returns ~20 candidates; OpenAI's
Decisions API answers one yes/no question per chunk ("does it help answer this?") and the 8
most likely go to the answer model. Measured: "what costs extra?" moved the optional-procedures
section from 16th to 4th. If the call fails, the search order is kept, so it can only help.
"""

import logging

import config
from bot import llm
from bot.schemas import Hit

log = logging.getLogger(__name__)

CHUNK_CHARS = 4000  # per passage: whole chunks (a review group is ~2,300); 20 of them stay well inside one call


def rerank(question: str, hits: list[Hit], k: int) -> list[Hit]:
    """The k most relevant hits, renamed C1..Ck. Ties (e.g. all "no") keep the search order."""
    if config.RERANK and len(hits) > k:
        try:
            probability = llm.decide(_input(question, hits), _questions(len(hits)))
            order = sorted(range(len(hits)), key=lambda i: -probability.get(f"p{i + 1}", 0.0))
            hits = [hits[i] for i in order]
        except llm.LLMError as exc:
            log.warning("rerank skipped, keeping search order: %s", exc)
    return [hit.model_copy(update={"ref": f"C{n}"}) for n, hit in enumerate(hits[:k], start=1)]


def _input(question: str, hits: list[Hit]) -> str:
    passages = "\n\n".join(f"P{n}: {hit.chunk.text[:CHUNK_CHARS]}" for n, hit in enumerate(hits, start=1))
    return f"Question: {question}\n\n{passages}"


def _questions(count: int) -> list[dict]:
    # One concern per question, stated as an observable condition (the Decisions guide's advice).
    return [
        {"type": "predicate", "name": f"p{n}",
         "instructions": f"Does passage P{n} state facts that directly help answer the question, or any part of it?"}
        for n in range(1, count + 1)
    ]
