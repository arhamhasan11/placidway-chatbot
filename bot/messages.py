"""Fixed replies from messages.yaml, in the user's language."""

from functools import cache

import yaml

import config


@cache
def _load() -> dict:
    return yaml.safe_load(config.MESSAGES_FILE.read_text(encoding="utf-8"))


def t(key: str, lang: str) -> str:
    """Return message `key` in `lang` (falls back to English)."""
    texts = _load()[key]
    return texts.get(lang, texts["en"]).format(quote_url=config.QUOTE_URL)
