"""The only module that talks to the model provider: structured chat calls and embeddings.

Called by bot/understand.py, bot/answer.py, bot/retrieve.py, bot/rerank.py and ingest/build.py.
"""

from typing import TypeVar

import httpx
import numpy as np
import openai
import pydantic

import config

T = TypeVar("T", bound=pydantic.BaseModel)

EMBED_BATCH = 100  # inputs per embeddings request; far below the API's per-request limit

_client: openai.OpenAI | None = None  # created on first use, so importing this module never needs a key


class LLMError(Exception):
    """The provider failed or returned nothing usable. chat.py shows the user the "busy" reply."""


def _get_client() -> openai.OpenAI:
    global _client
    if not config.OPENAI_API_KEY:
        raise LLMError("OPENAI_API_KEY is not set: put it in .env locally, or in Vercel / GitHub secrets.")
    if _client is None:
        # max_retries=0: a turn makes up to 4 calls at LLM_TIMEOUT_S each; retries would break vercel.json maxDuration (120 s).
        _client = openai.OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.LLM_BASE_URL,
            timeout=config.LLM_TIMEOUT_S,
            max_retries=0,
        )
    return _client


def chat_json(system: str, user: str, schema: type[T], max_tokens: int) -> T:
    """Ask the chat model for a reply that must validate as `schema` (structured outputs)."""
    client = _get_client()
    try:
        completion = client.chat.completions.parse(
            model=config.CHAT_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=schema,
            max_completion_tokens=max_tokens,
            # omit = leave the field out entirely; models without reasoning reject it even as null.
            reasoning_effort=openai.omit if config.REASONING_EFFORT is None else config.REASONING_EFFORT,
        )
    # ValidationError: OpenAI-compatible providers that don't enforce the schema can return bad JSON.
    except (openai.OpenAIError, pydantic.ValidationError) as exc:
        raise LLMError(f"chat call failed: {exc}") from exc
    message = completion.choices[0].message
    if message.refusal or message.parsed is None:
        raise LLMError(f"chat call returned no usable answer (refusal: {message.refusal!r})")
    return message.parsed


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts as float32 rows of length 1, so a dot product is the cosine similarity."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)  # an incremental build with nothing new to embed
    client = _get_client()
    rows: list[list[float]] = []
    try:
        for start in range(0, len(texts), EMBED_BATCH):
            response = client.embeddings.create(model=config.EMBED_MODEL, input=texts[start : start + EMBED_BATCH])
            rows.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
    except openai.OpenAIError as exc:
        raise LLMError(f"embedding call failed: {exc}") from exc
    vectors = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)  # the floor only matters for an all-zero vector


def decide(text: str, questions: list[dict]) -> dict[str, float]:
    """OpenAI Decisions API: yes/no questions about `text`, answered as probabilities by name.

    Called by bot/rerank.py. The pinned SDK has no helper for this endpoint yet, so it is a
    plain HTTPS call. A refused question simply gets no probability.
    """
    if not config.OPENAI_API_KEY:
        raise LLMError("OPENAI_API_KEY is not set")
    body = {"model": config.DECISIONS_MODEL, "input": text, "questions": questions}
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
    try:
        response = httpx.post(config.DECISIONS_URL, json=body, headers=headers, timeout=config.RERANK_TIMEOUT_S)
        response.raise_for_status()
        answers = response.json()["answers"]
        # Any malformed answer (missing or null probability, wrong shape) is a failed call, which the
        # reranker turns into "keep the search order" instead of a crashed request.
        return {a["name"]: float(a["probability"]) for a in answers if a.get("type") == "predicate"}
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise LLMError(f"decisions call failed: {exc}") from exc
