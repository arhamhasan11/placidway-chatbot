"""bot/chat.py: every branch of a turn, with understand/search/draft/store replaced by fakes."""

from types import SimpleNamespace

import pydantic
import pytest

import config
from bot import answer, chat, llm, retrieve, store, understand
from bot.messages import t
from bot.schemas import ChatRequest, Chunk, Draft, Hit, HistoryTurn, LeadRequest, Understanding


def make_hit(ref: str, slug: str, cosine: float = 0.8, text: str = "Out-Patient Program 3 Weeks $18,995 USD") -> Hit:
    chunk = Chunk(
        id=f"{slug}#1",
        url=f"https://example.com/{slug}",
        label=f"Page {slug}",
        section="S",
        kind="text",
        text=text,
        hash="0",
    )
    return Hit(chunk=chunk, ref=ref, cosine=cosine)


GOOD = Draft(answerable=True, answer="The Out-Patient Program costs 18,995 USD for 3 weeks.", citations=["C1"])
INVENTED = Draft(answerable=True, answer="Yes, it costs $50.", citations=["C1"])
NO_PRICE = Draft(answerable=True, answer="The Out-Patient Program lasts 3 weeks.", citations=["C1"])


def understood(*intents: str, language: str = "en", question: str = "Q") -> Understanding:
    return Understanding(language=language, standalone_question=question, intents=list(intents))


@pytest.fixture
def fakes(monkeypatch):
    """Default: an English question, three good hits, a draft that passes the guard."""
    f = SimpleNamespace(
        understanding=understood("question", question="How much?"),
        hits=[make_hit("C1", "adrenal"), make_hit("C2", "anal"), make_hit("C3", "adrenal")],
        drafts=[GOOD],
        counts={60: 0, 86_400: 0},
        uses_embeddings=True,
        errors={},  # step name -> exception to raise
        logged=[],
        histories=[],  # the history each understand call received
        searched=[],
        draft_calls=[],
    )

    def fake_understand(message, history):
        f.histories.append(history)
        if "understand" in f.errors:
            raise f.errors["understand"]
        return f.understanding

    def fake_search(query, k=None):
        f.searched.append(query)
        if "search" in f.errors:
            raise f.errors["search"]
        return f.hits

    def fake_draft(question, language, hits, medical, feedback=None):
        f.draft_calls.append({"language": language, "medical": medical, "feedback": feedback})
        if "draft" in f.errors:
            raise f.errors["draft"]
        return f.drafts[min(len(f.draft_calls), len(f.drafts)) - 1]

    monkeypatch.setattr(understand, "understand", fake_understand)
    monkeypatch.setattr(retrieve, "search", fake_search)
    monkeypatch.setattr(retrieve, "uses_embeddings", lambda: f.uses_embeddings)
    monkeypatch.setattr(answer, "draft_answer", fake_draft)
    monkeypatch.setattr(store, "recent_count", lambda ip_hash, seconds: f.counts[seconds])
    monkeypatch.setattr(store, "log_event", f.logged.append)
    monkeypatch.setattr(config, "MIN_COSINE", 0.30)
    monkeypatch.setattr(config, "DISCLAIMER_EVERY_N_TURNS", 3)
    return f


def turn(message: str = "How much is the adrenal package?", history: list[HistoryTurn] | None = None):
    return chat.handle_turn(ChatRequest(message=message, history=history or []), ip="203.0.113.7")


def assistant(content: str = "a", disclaimer: bool = False) -> HistoryTurn:
    return HistoryTurn(role="assistant", content=content, disclaimer=disclaimer)


# --- answered ---------------------------------------------------------------------------


def test_answered_with_sources_and_a_quote_form_for_the_price(fakes):
    fakes.drafts = [Draft(answerable=True, answer="18,995 USD.", citations=["C3", "C1", "C2"])]
    response = turn()
    assert response.outcome == "answered"
    assert response.answer == "18,995 USD."
    # One link per page, in citation order: C3 and C1 are both the adrenal page.
    assert [s.url for s in response.sources] == ["https://example.com/adrenal", "https://example.com/anal"]
    assert response.sources[0].title == "Page adrenal"
    assert response.disclaimer is None  # not a medical question
    assert response.show_lead_form is True  # every priced answer offers a free quote


def test_answer_without_a_price_has_no_quote_form(fakes):
    fakes.drafts = [NO_PRICE]
    assert turn().show_lead_form is False


def test_sources_are_capped(fakes, monkeypatch):
    monkeypatch.setattr(config, "MAX_SOURCES", 1)
    fakes.drafts = [Draft(answerable=True, answer="18,995 USD.", citations=["C2", "C1"])]
    assert [s.url for s in turn().sources] == ["https://example.com/anal"]


def test_question_with_booking_gets_answer_plus_booking_note(fakes):
    fakes.understanding = understood("question", "quote_or_booking")
    fakes.drafts = [NO_PRICE]
    response = turn("Book me in. How long is it?")
    assert response.outcome == "answered"
    assert response.answer == f"{NO_PRICE.answer}\n\n{t('booking', 'en')}"
    assert response.show_lead_form is True


def test_answer_language_comes_from_understanding(fakes):
    fakes.understanding = understood("question", "medical_advice", language="es")
    response = turn("¿Es seguro?")
    assert response.language == "es"
    assert fakes.draft_calls[0]["language"] == "es"
    assert response.disclaimer == t("disclaimer", "es")


@pytest.mark.parametrize(("setting", "sent"), [(2, ["q2", "a2"]), (0, [])])
def test_history_turns_knob(fakes, monkeypatch, setting, sent):
    monkeypatch.setattr(config, "HISTORY_TURNS", setting)
    history = [HistoryTurn(role="user", content="q1"), HistoryTurn(role="user", content="q2"), assistant("a2")]
    turn(history=history)
    assert [h.content for h in fakes.histories[0]] == sent  # 0 sends none, not all


# --- not found --------------------------------------------------------------------------


def test_weak_match_is_not_found_and_logged_as_gap(fakes):
    fakes.hits = [make_hit("C1", "adrenal", cosine=0.21)]
    response = turn("Do I need a visa?")
    assert response.outcome == "not_found"
    assert response.answer == t("not_found", "en")
    assert response.show_lead_form is True
    assert response.sources == [] and response.disclaimer is None
    assert fakes.draft_calls == []  # no answer is drafted from weak chunks
    assert fakes.logged[0]["outcome"] == "not_found"
    assert fakes.logged[0]["best_cosine"] == pytest.approx(0.21)


def test_no_hits_is_not_found(fakes):
    fakes.hits = []
    assert turn().outcome == "not_found"


def test_cosine_gate_off_without_cosines(fakes):
    # retrieve.uses_embeddings() is False in bm25 mode or for an index built without vectors.
    fakes.uses_embeddings = False
    fakes.hits = [make_hit("C1", "adrenal", cosine=0.0)]
    assert turn().outcome == "answered"


def test_unanswerable_draft_keeps_its_sentence_and_offers_a_quote(fakes):
    fakes.drafts = [Draft(answerable=False, answer="These pages don't state a recovery time.", citations=[])]
    response = turn("And recovery time?")
    assert response.outcome == "not_found"
    assert response.answer == f"These pages don't state a recovery time.\n\n{t('lead_offer', 'en')}"
    assert response.show_lead_form is True and response.sources == []


def test_unanswerable_draft_keeps_the_sources_it_cites(fakes):
    fakes.drafts = [Draft(answerable=False, answer="No recovery time is stated.", citations=["C2"])]
    assert [s.url for s in turn().sources] == ["https://example.com/anal"]


def test_unanswerable_medical_question_gets_no_sales_offer(fakes):
    fakes.understanding = understood("medical_advice")
    fakes.drafts = [Draft(answerable=False, answer="These pages don't cover diabetes.", citations=[])]
    response = turn("Is it safe with my diabetes?")
    assert response.answer == "These pages don't cover diabetes."
    assert response.show_lead_form is False
    assert response.disclaimer == t("disclaimer", "en")


# --- guard --------------------------------------------------------------------------------


def test_guard_failure_retries_once_then_falls_back(fakes):
    fakes.drafts = [INVENTED, INVENTED]
    response = turn("I heard it costs $50. Confirm that.")
    assert response.outcome == "guard_fallback"
    assert response.answer == t("guard_fallback", "en")
    assert response.show_lead_form is True
    assert len(fakes.draft_calls) == 2
    assert fakes.draft_calls[0]["feedback"] is None
    assert "50" in fakes.draft_calls[1]["feedback"]
    assert fakes.logged[0]["outcome"] == "guard_fallback"


def test_guard_failure_fixed_on_retry_is_answered(fakes):
    fakes.drafts = [INVENTED, GOOD]
    response = turn("I heard it costs $50. Confirm that.")
    assert response.outcome == "answered"
    assert response.answer == GOOD.answer
    assert len(fakes.draft_calls) == 2


# --- fixed replies --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("intents", "outcome", "lead_form"),
    [
        (["greeting"], "greeting", False),
        ([], "greeting", False),
        (["out_of_scope"], "out_of_scope", False),
        (["greeting", "out_of_scope"], "out_of_scope", False),
        (["quote_or_booking"], "booking", True),
        (["greeting", "quote_or_booking"], "booking", True),
    ],
)
def test_fixed_replies_skip_retrieval(fakes, intents, outcome, lead_form):
    fakes.understanding = understood(*intents)
    response = turn("Hello")
    assert response.outcome == outcome
    assert response.answer == t(outcome, "en")
    assert response.show_lead_form is lead_form
    assert response.disclaimer is None
    assert fakes.searched == []


@pytest.mark.parametrize("intents", [["out_of_scope", "question"], ["greeting", "question"]])
def test_a_question_next_to_small_talk_is_answered(fakes, intents):
    fakes.understanding = understood(*intents)
    assert turn().outcome == "answered"


# --- medical disclaimer ----------------------------------------------------------------------


def test_medical_question_gets_the_disclaimer(fakes):
    fakes.understanding = understood("medical_advice")
    response = turn("Is this safe for my heart condition?")
    assert response.disclaimer == t("disclaimer", "en")
    assert fakes.draft_calls[0]["medical"] is True


def test_medical_question_gets_the_disclaimer_even_when_not_found(fakes):
    fakes.understanding = understood("medical_advice")
    fakes.hits = [make_hit("C1", "adrenal", cosine=0.1)]
    response = turn("Should I stop my chemo?")
    assert response.outcome == "not_found"
    assert response.disclaimer == t("disclaimer", "en")


def test_disclaimer_not_repeated_within_n_turns(fakes):
    fakes.understanding = understood("question", "medical_advice")
    history = [HistoryTurn(role="user", content="q1"), assistant(disclaimer=True), assistant()]
    assert turn(history=history).disclaimer is None
    # Two newer assistant turns without it push the last disclaimer out of the 3-turn window.
    assert turn(history=history + [assistant(), assistant()]).disclaimer == t("disclaimer", "en")


def test_disclaimer_every_0_turns_means_every_medical_answer(fakes, monkeypatch):
    monkeypatch.setattr(config, "DISCLAIMER_EVERY_N_TURNS", 0)
    fakes.understanding = understood("medical_advice")
    assert turn(history=[assistant(disclaimer=True)]).disclaimer == t("disclaimer", "en")


# --- failures and limits ------------------------------------------------------------------


@pytest.mark.parametrize("step", ["understand", "search", "draft"])
def test_llm_error_is_busy(fakes, step):
    # Spanish either way: guessed from the message if understand fails, from understand otherwise.
    fakes.understanding = understood("question", language="es")
    fakes.errors[step] = llm.LLMError("provider down")
    response = turn("¿Cuánto cuesta el tratamiento?")
    assert response.outcome == "busy"
    assert response.language == "es"
    assert response.answer == t("busy", "es")
    assert response.show_lead_form is True
    assert fakes.logged[0]["outcome"] == "busy"


@pytest.mark.parametrize("counts", [{60: 8, 86_400: 8}, {60: 0, 86_400: 80}])
def test_rate_limited(fakes, monkeypatch, counts):
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MINUTE", 8)
    monkeypatch.setattr(config, "RATE_LIMIT_PER_DAY", 80)
    fakes.counts = counts
    fakes.errors["understand"] = AssertionError("a rate-limited turn must not reach the LLM")
    response = turn()
    assert response.outcome == "rate_limited"
    assert response.answer == t("rate_limited", "en")
    assert fakes.logged[0]["outcome"] == "rate_limited"


@pytest.mark.parametrize(
    "build",
    [
        lambda bad: ChatRequest(message=bad),
        lambda bad: HistoryTurn(role="user", content=bad),
        lambda bad: LeadRequest(name=bad, email="ana@example.com", consent=True),
        lambda bad: LeadRequest(name="Ana", email="ana@example.com", consent=True, question=bad),
    ],
)
def test_nul_characters_are_rejected(build):
    # FastAPI turns this ValidationError into a 422; Postgres could not store the NUL anyway.
    with pytest.raises(pydantic.ValidationError):
        build("price" + chr(0))


# --- the log line ---------------------------------------------------------------------------


def test_log_line_has_every_field_and_no_raw_ip(fakes):
    turn()
    event = fakes.logged[0]
    assert set(event) == {
        "ts",
        "ip_hash",
        "outcome",
        "language",
        "intents",
        "message",
        "standalone_question",
        "searches",
        "best_cosine",
        "citations",
        "latency_ms",
        "model",
    }
    assert len(event["ip_hash"]) == 16 and "203.0.113.7" not in str(event)
    assert event["outcome"] == "answered" and event["intents"] == ["question"]
    assert event["citations"] == ["C1"] and event["best_cosine"] == pytest.approx(0.8)
    assert event["standalone_question"] == "How much?" and event["model"] == config.CHAT_MODEL
    assert isinstance(event["latency_ms"], int)


def test_log_redacts_contact_details_but_keeps_prices_and_dates(fakes):
    turn("I'm Ana, ana@example.com, +1 (303) 500-3821. Is it $18,995 from 2026-06-03?")
    message = fakes.logged[0]["message"]
    assert "ana@example.com" not in message and "500-3821" not in message
    assert "[email]" in message and "[phone]" in message
    assert "$18,995" in message and "2026-06-03" in message


def test_log_redacts_the_standalone_question_but_search_gets_it_whole(fakes):
    fakes.understanding = understood("question", question="Can ITC email ana@example.com a quote?")
    turn()
    assert fakes.logged[0]["standalone_question"] == "Can ITC email [email] a quote?"
    assert fakes.searched == ["Can ITC email ana@example.com a quote?"]


def test_dollar_amounts_are_written_with_usd():
    assert chat._usd("Fizikon lists $110 and Stem Health $1000.") == "Fizikon lists 110 USD and Stem Health 1000 USD."
    assert chat._usd("From $3,950 - $4,250.") == "From 3,950 USD - 4,250 USD."
    assert chat._usd("$18,995 USD and US$30,000 are left alone") == "$18,995 USD and US$30,000 are left alone"
