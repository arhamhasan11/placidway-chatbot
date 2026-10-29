"""LLM call 1: reads the latest message in context and returns its language, a standalone
English question and its intents. Called by bot/chat.py before retrieval."""

import re

import config
from bot import llm
from bot.schemas import HistoryTurn, Understanding

TURN_MAX_CHARS = 600  # the start of an earlier turn names its topic; the rest only costs tokens

# Spanish words common in questions to this bot. Only used when the LLM is unavailable.
SPANISH_WORDS = frozenset(
    {"cuánto", "cuanto", "cuesta", "precio", "precios", "tratamiento", "hola", "gracias", "quiero"}
)


def understand(message: str, history: list[HistoryTurn]) -> Understanding:
    """Resolve follow-ups ("what about Turkey?") into a question that can be searched on its own.
    `history` is already trimmed to config.HISTORY_TURNS by chat.py."""
    system = (config.PROMPTS_DIR / "understand.md").read_text(encoding="utf-8")
    result = llm.chat_json(system, render(message, history), Understanding, config.UNDERSTAND_MAX_TOKENS)
    # The model sometimes writes "es-MX" or "Spanish"; messages.yaml is keyed by two-letter codes.
    language = result.language.strip().lower().split("-")[0]
    if not re.fullmatch(r"[a-z]{2}", language):
        language = guess_language(message)
    return result.model_copy(update={"language": language})


def render(message: str, history: list[HistoryTurn]) -> str:
    """The user content: the turns chat.py chose to send, then the current message, each inside tags
    marking it as data."""
    lines = [f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.content[:TURN_MAX_CHARS]}" for turn in history]
    past = "\n".join(lines) or "(no earlier messages)"
    return f"<history>\n{past}\n</history>\n\n<message>\n{message}\n</message>"


def guess_language(text: str) -> str:
    """Cheap language guess for fixed replies when the LLM is down (busy, rate limited)."""
    # Arabic, Arabic Supplement, Arabic Presentation Forms-A and -B.
    if re.search(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufefc]", text):
        return "ar"
    lowered = text.lower()
    if any(mark in lowered for mark in "¿¡ñáéíóú") or SPANISH_WORDS & set(re.findall(r"\w+", lowered)):
        return "es"
    return "en"
