"""Settings files agree with each other: every fixed reply exists in every language,
sources.yaml only points at PlacidWay, and the time limits fit. Run: python -m pytest tests/test_config.py"""

import json
from string import Formatter
from urllib.parse import urlparse

import yaml

import config

# Keys the code asks bot.messages.t() for.
MESSAGE_KEYS = {"not_found", "out_of_scope", "greeting", "booking", "guard_fallback",
                "busy", "rate_limited", "disclaimer", "lead_offer", "lead_thanks"}
PAGE_FIELDS = {"slug", "url", "label", "expect", "fixture"}


def load(path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def placeholders(text: str) -> set[str]:
    return {field for _, field, _, _ in Formatter().parse(text) if field}


def test_every_message_exists_in_every_language_with_the_same_placeholders():
    messages = load(config.MESSAGES_FILE)
    assert MESSAGE_KEYS <= set(messages)
    for key, texts in messages.items():
        assert set(config.SUPPORTED_LANGUAGES) <= set(texts), key
        for lang in config.SUPPORTED_LANGUAGES:
            assert texts[lang].strip(), (key, lang)
            # A translation that drops or renames {quote_url} would break .format() or lose the link.
            assert placeholders(texts[lang]) == placeholders(texts["en"]) <= {"quote_url"}, (key, lang)
            texts[lang].format(quote_url=config.QUOTE_URL)


def test_sources_are_unique_https_placidway_pages_with_saved_fixtures():
    sources = load(config.SOURCES_FILE)
    pages = sources["pages"]
    # Not "== 7": adding a page should need only sources.yaml and its saved fixture.
    assert len(pages) >= 1
    for page in pages:
        assert PAGE_FIELDS <= set(page), page
        url = urlparse(page["url"])
        assert url.scheme == "https" and url.hostname == "www.placidway.com", page["url"]
        assert (config.ROOT / "tests" / "fixtures" / page["fixture"]).is_file(), page["fixture"]
    for field in ("slug", "url", "fixture"):
        values = [page[field] for page in pages]
        assert len(values) == len(set(values)), f"duplicate {field}"
    assert sources["content_selector"].strip()
    assert isinstance(sources["drop_selectors"], list)


def test_tunables_are_consistent():
    assert config.RETRIEVAL_MODE in {"hybrid", "dense", "bm25"}
    assert 0 < config.MIN_COSINE < 1
    assert 1 <= config.MAX_SOURCES <= config.TOP_K  # can't link more chunks than were retrieved
    assert config.QUOTE_URL.startswith("https://www.placidway.com/")
    assert 0 < config.MAX_SHRINK < 1
    assert config.HISTORY_TURNS >= 0 and config.DISCLAIMER_EVERY_N_TURNS >= 0  # 0 has a meaning for both
    assert config.MAX_HISTORY_CHARS >= config.MAX_MESSAGE_CHARS  # a past question must fit in history
    assert config.RATE_LIMIT_LEADS_PER_DAY >= 1


def test_vercel_time_limit_fits_a_whole_turn():
    limit = json.loads((config.ROOT / "vercel.json").read_text(encoding="utf-8"))["functions"]["app.py"]["maxDuration"]
    # Worst case: understand + embed + answer + guard retry, each up to LLM_TIMEOUT_S, plus database
    # wake-ups. 300 s is the most Vercel's Hobby plan allows.
    assert 4 * config.LLM_TIMEOUT_S < limit <= 300
