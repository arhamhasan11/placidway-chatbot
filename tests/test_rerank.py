"""Reranking: most relevant first, search order on ties and on any failure. No network."""

import pytest

import config
from bot import llm, rerank
from bot.schemas import Chunk, Hit


def hits(*ids: str) -> list[Hit]:
    return [Hit(chunk=Chunk(id=i, url="u", label="P", section="S", kind="text", text=f"P | S\n{i}", hash=i),
                ref="?", cosine=0.5) for i in ids]


@pytest.fixture(autouse=True)
def rerank_on(monkeypatch):
    monkeypatch.setattr(config, "RERANK", True)


def test_most_likely_passages_come_first_and_are_renamed(monkeypatch):
    monkeypatch.setattr(llm, "decide", lambda text, questions: {"p1": 0.0, "p2": 0.2, "p3": 1.0, "p4": 0.0})
    out = rerank.rerank("what costs extra?", hits("a", "b", "c", "d"), k=3)
    assert [h.chunk.id for h in out] == ["c", "b", "a"]  # a and d tie at 0: search order decides
    assert [h.ref for h in out] == ["C1", "C2", "C3"]


def test_one_question_per_passage_with_the_passages_in_the_input(monkeypatch):
    seen = {}

    def decide(text, questions):
        seen.update(text=text, names=[q["name"] for q in questions])
        return {}

    monkeypatch.setattr(llm, "decide", decide)
    rerank.rerank("Where is ITC?", hits("addr", "doc", "faq"), k=2)
    assert seen["names"] == ["p1", "p2", "p3"]
    assert seen["text"].startswith("Question: Where is ITC?") and "P1: P | S\naddr" in seen["text"]


def test_a_failed_call_keeps_the_search_order(monkeypatch):
    def broken(text, questions):
        raise llm.LLMError("timeout")

    monkeypatch.setattr(llm, "decide", broken)
    assert [h.chunk.id for h in rerank.rerank("q", hits("a", "b", "c"), k=2)] == ["a", "b"]


def test_switched_off_or_too_few_hits_makes_no_call(monkeypatch):
    def must_not_run(text, questions):
        raise AssertionError("decide was called")

    monkeypatch.setattr(llm, "decide", must_not_run)
    assert len(rerank.rerank("q", hits("a", "b"), k=8)) == 2  # nothing to choose between
    monkeypatch.setattr(config, "RERANK", False)
    assert [h.chunk.id for h in rerank.rerank("q", hits("a", "b", "c"), k=2)] == ["a", "b"]


def test_decide_without_a_key_fails_fast():
    with pytest.raises(llm.LLMError):
        llm.decide("text", [])


@pytest.mark.parametrize("answers", [
    [{"type": "predicate", "name": "p1", "probability": None}],   # null score
    [{"type": "predicate", "name": "p1"}],                        # missing score
    "not a list",                                                 # wrong shape
])
def test_malformed_scores_become_llm_error(monkeypatch, answers):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"answers": answers}

    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: Response())
    with pytest.raises(llm.LLMError):
        llm.decide("text", [{"type": "predicate", "name": "p1", "instructions": "?"}])
