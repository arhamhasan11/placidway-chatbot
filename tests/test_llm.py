"""bot/llm.py with a fake client (no network), and how the two LLM calls build their prompts."""

from types import SimpleNamespace

import numpy as np
import openai
import pytest

import config
from bot import answer, llm, understand
from bot.schemas import Chunk, Draft, Hit, HistoryTurn, Understanding


class FakeClient:
    """Stands in for openai.OpenAI: records calls, returns canned responses or raises."""

    def __init__(self, message=None, error=None, dims=3):
        self.message, self.error, self.dims = message, error, dims
        self.parse_calls, self.embed_calls = [], []
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self.parse))
        self.embeddings = SimpleNamespace(create=self.create)

    def parse(self, **request):
        self.parse_calls.append(request)
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=self.message)])

    def create(self, model, input):
        self.embed_calls.append(input)
        if self.error:
            raise self.error
        # Returned out of order on purpose: embed() must sort by index. Row i is (i+1) * [3, 4, 0...].
        data = [
            SimpleNamespace(index=i, embedding=[3.0 * (i + 1), 4.0 * (i + 1)] + [0.0] * (self.dims - 2))
            for i in range(len(input))
        ]
        return SimpleNamespace(data=list(reversed(data)))


@pytest.fixture
def fake(monkeypatch):
    def install(**kwargs):
        client = FakeClient(**kwargs)
        monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(llm, "_client", client)
        return client

    return install


PARSED = Understanding(language="en", standalone_question="Q?", intents=["question"])


def test_chat_json_returns_parsed_object(fake, monkeypatch):
    monkeypatch.setattr(config, "REASONING_EFFORT", "none")
    client = fake(message=SimpleNamespace(refusal=None, parsed=PARSED))
    assert llm.chat_json("sys", "user", Understanding, 300) is PARSED
    request = client.parse_calls[0]
    assert request["model"] == config.CHAT_MODEL
    assert request["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}]
    assert request["response_format"] is Understanding
    assert request["max_completion_tokens"] == 300
    assert request["reasoning_effort"] == "none"


def test_reasoning_effort_left_out_when_none(fake, monkeypatch):
    monkeypatch.setattr(config, "REASONING_EFFORT", None)
    client = fake(message=SimpleNamespace(refusal=None, parsed=PARSED))
    llm.chat_json("sys", "user", Understanding, 300)
    assert client.parse_calls[0]["reasoning_effort"] is openai.omit


@pytest.mark.parametrize(
    "message",
    [SimpleNamespace(refusal="I can't help with that", parsed=None), SimpleNamespace(refusal=None, parsed=None)],
)
def test_refusal_or_empty_answer_is_llm_error(fake, message):
    fake(message=message)
    with pytest.raises(llm.LLMError):
        llm.chat_json("sys", "user", Understanding, 300)


def test_provider_error_becomes_llm_error(fake):
    fake(error=openai.OpenAIError("provider down"))
    with pytest.raises(llm.LLMError, match="provider down"):
        llm.chat_json("sys", "user", Understanding, 300)


def test_missing_key_is_a_clear_llm_error(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(llm, "_client", None)
    with pytest.raises(llm.LLMError, match="OPENAI_API_KEY"):
        llm.chat_json("sys", "user", Understanding, 300)
    with pytest.raises(llm.LLMError, match="OPENAI_API_KEY"):
        llm.embed(["text"])


def test_client_is_built_from_config(monkeypatch):
    created = []
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(openai, "OpenAI", lambda **settings: created.append(settings) or "client")
    assert llm._get_client() == "client"
    assert llm._get_client() == "client"
    assert created == [
        {"api_key": "sk-test", "base_url": config.LLM_BASE_URL, "timeout": config.LLM_TIMEOUT_S, "max_retries": 0}
    ]


def test_embed_batches_sorts_and_normalises(fake):
    client = fake()
    vectors = llm.embed([f"text {i}" for i in range(250)])
    assert [len(batch) for batch in client.embed_calls] == [100, 100, 50]
    assert vectors.dtype == np.float32 and vectors.shape == (250, 3)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(vectors[1], [0.6, 0.8, 0.0], rtol=1e-6)  # [6, 8] / 10, in input order


def test_embed_nothing_makes_no_call(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")  # would raise if a client were needed
    assert llm.embed([]).shape[0] == 0


def test_embed_error_becomes_llm_error(fake):
    fake(error=openai.OpenAIError("rate limited"))
    with pytest.raises(llm.LLMError):
        llm.embed(["text"])


# --- the two LLM calls ----------------------------------------------------------------


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake_chat_json(system, user, schema, max_tokens):
        calls.append({"system": system, "user": user, "schema": schema, "max_tokens": max_tokens})
        if schema is Understanding:
            return Understanding(language="es-MX", standalone_question="Q?", intents=["question"])
        return Draft(answerable=True, answer="A", citations=["C1"])

    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    return calls


def test_understand_sends_the_history_it_is_given_truncated(captured):
    # chat.py picks the last HISTORY_TURNS turns (tests/test_chat.py); understand sends exactly those.
    history = [
        HistoryTurn(role="user", content="What is included in the adenocarcinoma package?"),
        HistoryTurn(role="assistant", content="x" * 1000),
    ]
    result = understand.understand("And what about recovery time?", history)
    call = captured[0]
    assert call["schema"] is Understanding and call["max_tokens"] == config.UNDERSTAND_MAX_TOKENS
    assert "standalone_question" in call["system"]  # the prompt file was loaded
    assert "User: What is included in the adenocarcinoma package?" in call["user"]
    assert f"Assistant: {'x' * understand.TURN_MAX_CHARS}\n" in call["user"]
    assert "<message>\nAnd what about recovery time?\n</message>" in call["user"]
    assert result.language == "es"  # "es-MX" trimmed to the code messages.yaml uses


def test_understand_falls_back_to_guess_for_a_bad_language_code(monkeypatch):
    reply = Understanding(language="Spanish", standalone_question="Q?", intents=["question"])
    monkeypatch.setattr(llm, "chat_json", lambda system, user, schema, max_tokens: reply)
    assert understand.understand("¿Cuánto cuesta?", []).language == "es"


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("How much does it cost?", "en"),
        ("¿Cuánto cuesta el tratamiento?", "es"),
        ("precio del tratamiento", "es"),
        ("كم تكلفة العلاج؟", "ar"),
    ],
)
def test_guess_language(text, language):
    assert understand.guess_language(text) == language


def test_draft_answer_renders_chunks_and_flags(captured):
    chunk = Chunk(
        id="x#1",
        url="https://example.com",
        label="ITC Price List",
        section="Prices",
        kind="table_row",
        text="ITC Price List | Prices | Breast Cancer 18995 USD",
        hash="0",
    )
    hits = [Hit(chunk=chunk, ref="C1", cosine=0.8)]
    draft = answer.draft_answer("How much?", "ar", hits, medical=True, feedback="Cite C1.")
    call = captured[0]
    assert draft.citations == ["C1"]
    assert call["schema"] is Draft and call["max_tokens"] == config.ANSWER_MAX_TOKENS
    assert "Target language: ar" in call["user"]
    assert "Question: <question>How much?</question>" in call["user"]
    assert "Medical: yes" in call["user"]
    assert "Feedback: your previous draft failed these checks. Fix them: Cite C1." in call["user"]
    assert (
        '<chunk id="C1" source="ITC Price List">\nITC Price List | Prices | Breast Cancer 18995 USD\n</chunk>'
        in call["user"]
    )


def test_draft_answer_leaves_out_unused_lines(captured):
    answer.draft_answer("How much?", "en", [], medical=False)
    assert "Medical:" not in captured[0]["user"] and "Feedback:" not in captured[0]["user"]


def test_understand_with_no_history(captured):
    understand.understand("Hello", [])
    assert "(no earlier messages)" in captured[0]["user"]


def test_guess_language_on_arabic_presentation_forms():
    assert understand.guess_language(chr(0xFEFB)) == "ar"  # lam-alef ligature, Presentation Forms-B
