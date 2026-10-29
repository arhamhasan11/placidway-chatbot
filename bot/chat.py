"""One chat turn, top to bottom: 0 rate limit, 1 understand, 2 retrieve, 3 answer, 4 guard, 5 reply, 6 log.

Called by app.py for POST /api/chat. The numbered steps match the README walkthrough.
"""

import re
import time
from datetime import UTC, datetime

import config
from bot import answer, guard, llm, rerank, retrieve, store, understand
from bot.messages import t
from bot.schemas import ChatRequest, ChatResponse, Draft, Hit, HistoryTurn, Source

# Replies that come with the quote form: the bot could not help, so a human should.
LEAD_FORM_OUTCOMES = {"booking", "not_found", "guard_fallback", "busy"}

EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
PHONE = re.compile(r"\+?\d(?:[\s().-]{0,2}\d){8,}")  # 9+ digits: phone numbers, not prices or dates
DOLLAR = re.compile(r"(?<![A-Za-z])\$\s?(\d[\d,.]*\d|\d)(?![\d,.]*\s?USD)")  # "$110", not "US$110" / "$110 USD"


def handle_turn(req: ChatRequest, ip: str) -> ChatResponse:
    """Answer one message. LLM and database failures become fixed replies, never errors."""
    started = time.monotonic()
    ip_hash = store.hash_ip(ip)  # pseudonymous, rotates daily; no raw IP is stored
    log = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "ip_hash": ip_hash,
        "outcome": None,
        "language": understand.guess_language(req.message),  # replaced by the model's answer in step 1
        "intents": [],
        "message": _redact(req.message)[: config.MAX_MESSAGE_CHARS],
        "standalone_question": "",
        "best_cosine": None,
        "citations": [],
        "latency_ms": None,
        "model": config.CHAT_MODEL,
    }
    response = _respond(req, ip_hash, log)  # steps 0-5

    # 6. Log the turn. store.log_event never raises. not_found and guard_fallback form the content-gap list.
    log.update(outcome=response.outcome, language=response.language)
    log["latency_ms"] = int((time.monotonic() - started) * 1000)
    store.log_event(log)
    return response


def _respond(req: ChatRequest, ip_hash: str, log: dict) -> ChatResponse:
    """Steps 0-5. Fills `log` with what each step learned, for step 6."""
    lang = log["language"]
    # 0. Rate limit (counts come from the event log, so they hold across serverless instances).
    if (
        store.recent_count(ip_hash, 60) >= config.RATE_LIMIT_PER_MINUTE
        or store.recent_count(ip_hash, 86_400) >= config.RATE_LIMIT_PER_DAY
    ):
        return _fixed("rate_limited", lang)

    # 1. Understand: language, a standalone English question, intents. Greetings, off-topic messages and
    #    pure booking requests get a fixed reply, no retrieval. A greeting next to a request is just politeness.
    try:
        u = understand.understand(req.message, _last(req.history, config.HISTORY_TURNS))
    except llm.LLMError:
        return _fixed("busy", lang)
    lang, intents = u.language, set(u.intents)
    log.update(language=lang, intents=u.intents, standalone_question=_redact(u.standalone_question))
    core = intents - {"greeting"}
    if not core:
        return _fixed("greeting", lang)
    if core == {"out_of_scope"}:
        return _fixed("out_of_scope", lang)
    if core == {"quote_or_booking"}:
        return _fixed("booking", lang)
    medical = "medical_advice" in intents
    # Decided here, shown in step 5: steps 2-4 can also end the turn, and a medical asker still needs it then.
    disclaimer = t("disclaimer", lang) if medical and _disclaimer_due(req.history) else None

    # 2. Retrieve. A weak best cosine means the pages don't cover it: say so instead of guessing.
    try:
        # Multi-part questions get one search per part. A single-topic question searches its full
        # standalone question: a short keyword phrase retrieved worse (measured). Raw text: only the
        # log copy is redacted.
        queries = u.searches if len(u.searches) >= 2 else [u.standalone_question]
        log["searches"] = [_redact(q) for q in queries]
        hits = retrieve.search_all(queries, config.RERANK_CANDIDATES if config.RERANK else config.TOP_K)
    except llm.LLMError:
        return _fixed("busy", lang, disclaimer)
    log["best_cosine"] = max((hit.cosine for hit in hits), default=0.0)
    if not hits or (retrieve.uses_embeddings() and log["best_cosine"] < config.MIN_COSINE):
        return _fixed("not_found", lang, disclaimer)
    # Keep the TOP_K candidates that most directly answer the question (Decisions API; falls back
    # to search order on any failure).
    hits = rerank.rerank(u.standalone_question, hits, config.TOP_K)

    # 3. Answer: a draft written from these chunks only (LLM call 2).
    try:
        draft = answer.draft_answer(u.standalone_question, lang, hits, medical)
    except llm.LLMError:
        return _fixed("busy", lang, disclaimer)

    # 4. Guard: prices, percentages and citations checked in code. One retry with the problems as
    #    feedback, because a model told exactly what failed usually fixes it; then a safe fallback.
    problems = guard.check(draft, hits)
    if problems:
        try:
            draft = answer.draft_answer(u.standalone_question, lang, hits, medical, feedback=" ".join(problems))
        except llm.LLMError:
            return _fixed("busy", lang, disclaimer)
        problems = guard.check(draft, hits)
    log["citations"] = draft.citations
    if problems:
        return _fixed("guard_fallback", lang, disclaimer)

    # 5. Reply: source links built from the cited chunks (the model never writes URLs), the disclaimer
    #    from step 1, and the quote form when the visitor wants a quote or the answer states a price.
    sources = _sources(draft, hits)
    if not draft.answerable:
        # The model's sentence says what is missing, in the user's language; the offer turns it into a
        # lead. Never on a medical question: there the next step is their doctor, not a sales offer.
        text = draft.answer.strip() or t("not_found", lang)
        if not medical:
            text += f"\n\n{t('lead_offer', lang)}"
        return ChatResponse(
            answer=text,
            sources=sources,
            outcome="not_found",
            language=lang,
            disclaimer=disclaimer,
            show_lead_form=not medical,
        )
    booking = "quote_or_booking" in intents
    reply = _usd(draft.answer)
    text = f"{reply}\n\n{t('booking', lang)}" if booking else reply
    priced = bool(guard.money_amounts(draft.answer))  # every priced answer offers a free, exact quote
    return ChatResponse(
        answer=text,
        sources=sources,
        outcome="answered",
        language=lang,
        disclaimer=disclaimer,
        show_lead_form=booking or priced,
    )


def _fixed(outcome: str, lang: str, disclaimer: str | None = None) -> ChatResponse:
    """A reply from messages.yaml; each such outcome has a message with the same key."""
    return ChatResponse(
        answer=t(outcome, lang),
        sources=[],
        outcome=outcome,
        language=lang,
        disclaimer=disclaimer,
        show_lead_form=outcome in LEAD_FORM_OUTCOMES,
    )


def _sources(draft: Draft, hits: list[Hit]) -> list[Source]:
    """Pages of the cited chunks, in citation order, one link per page."""
    chunks = guard.by_ref(hits)
    sources: list[Source] = []
    for ref in draft.citations:  # guard.check has made sure every ref is a hit
        chunk = chunks[ref]
        if all(source.url != chunk.url for source in sources):
            sources.append(Source(title=chunk.label, url=chunk.url))
    return sources[: config.MAX_SOURCES]


def _disclaimer_due(history: list[HistoryTurn]) -> bool:
    """The brief: a disclaimer "when relevant, without repeating it on every answer". So not again while
    one of the last DISCLAIMER_EVERY_N_TURNS assistant replies carried it (0 = on every medical answer)."""
    replies = [turn for turn in history if turn.role == "assistant"]
    return not any(turn.disclaimer for turn in _last(replies, config.DISCLAIMER_EVERY_N_TURNS))


def _last(items: list, n: int) -> list:
    """The last n items. A knob set to 0 means none: a plain items[-0:] would return them all."""
    return items[-n:] if n > 0 else []


def _usd(text: str) -> str:
    """'$110' -> '110 USD', so the currency is explicit in every answer and language.

    Runs after guard.check, which reads both spellings as the same amount. "US$" and amounts
    already followed by "USD" are left alone.
    """
    return DOLLAR.sub(r"\1 USD", text)


def _redact(message: str) -> str:
    """Logs keep the question but not contact details people type into the chat."""
    return PHONE.sub("[phone]", EMAIL.sub("[email]", message))
