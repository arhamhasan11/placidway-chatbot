"""HTTP API: routes, validation and error handling. chat.handle_turn and
retrieve.index_info are replaced, so no index or LLM is needed. Run: python -m pytest tests/test_api.py"""

import pytest
from fastapi.testclient import TestClient

import app as app_module
import config
from bot import chat, retrieve, store
from bot.messages import t
from bot.schemas import ChatResponse, Source

INDEX = {"built_at": "2026-10-28T12:00:00+00:00", "total_chunks": 42, "loaded_chunks": 42, "has_embeddings": True}
SECRETS = ("OPENAI_API_KEY", "DATABASE_URL", "IP_HASH_SALT")
GOOD_LEAD = {"name": "Ana", "email": "ana@example.com", "consent": True, "question": "Price?",
             "language": "es"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    for name in SECRETS:
        monkeypatch.setattr(config, name, "")
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "hybrid")
    monkeypatch.setattr(config, "LOCAL_LOG_FILE", tmp_path / "log.jsonl")
    monkeypatch.setattr(retrieve, "index_info", lambda: INDEX)
    return TestClient(app_module.app)


# --- chat ----------------------------------------------------------------------------


def test_chat_passes_the_request_and_first_forwarded_ip(client, monkeypatch):
    calls = []

    def fake_turn(req, ip):
        calls.append((req, ip))
        return ChatResponse(answer="ITC is in Tijuana.", outcome="answered", language="en",
                            sources=[Source(title="ITC profile", url="https://www.placidway.com/profile/5/x")])

    monkeypatch.setattr(chat, "handle_turn", fake_turn)
    res = client.post("/api/chat", json={"message": "Where is ITC?", "history": []},
                      headers={"x-forwarded-for": "203.0.113.5, 10.0.0.1"})
    assert res.status_code == 200
    assert res.json()["answer"] == "ITC is in Tijuana."
    assert res.json()["sources"][0]["title"] == "ITC profile"
    [(req, ip)] = calls
    assert req.message == "Where is ITC?" and ip == "203.0.113.5"


def test_chat_rejects_empty_and_too_long_messages(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    too_long = "x" * (config.MAX_MESSAGE_CHARS + 1)
    assert client.post("/api/chat", json={"message": too_long}).status_code == 422


def test_unexpected_error_returns_busy_reply_without_a_traceback(client, monkeypatch):
    def broken(req, ip):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(chat, "handle_turn", broken)
    quiet_client = TestClient(app_module.app, raise_server_exceptions=False)  # return the 500, don't re-raise
    res = quiet_client.post("/api/chat", json={"message": "hi"})
    assert res.status_code == 500
    assert res.json()["answer"] == t("busy", "en")
    assert res.json()["outcome"] == "busy"
    assert "secret internal detail" not in res.text and "Traceback" not in res.text


# --- leads -----------------------------------------------------------------------------


def test_lead_is_saved_and_thanked_in_the_visitor_language(client):
    res = client.post("/api/lead", json=GOOD_LEAD)
    assert res.status_code == 200
    assert res.json() == {"ok": True, "message": t("lead_thanks", "es")}
    assert "ana@example.com" in config.LOCAL_LOG_FILE.read_text(encoding="utf-8")


BAD_LEAD_FIELDS = [{"consent": False}, {"consent": None}, {"email": "not-an-email"}, {"name": ""}]


@pytest.mark.parametrize("change", BAD_LEAD_FIELDS)
def test_lead_without_consent_or_with_bad_fields_is_rejected(client, change):
    assert client.post("/api/lead", json={**GOOD_LEAD, **change}).status_code == 422


def test_lead_limit_per_visitor_returns_429_and_saves_nothing(client, monkeypatch):
    seen = []

    def at_the_limit(ip_hash):
        seen.append(ip_hash)
        return config.RATE_LIMIT_LEADS_PER_DAY

    monkeypatch.setattr(store, "leads_today", at_the_limit)
    res = client.post("/api/lead", json=GOOD_LEAD, headers={"x-forwarded-for": "203.0.113.5"})
    assert res.status_code == 429
    assert res.json() == {"ok": False, "message": t("rate_limited", "es")}
    assert seen == [store.hash_ip("203.0.113.5")]  # counted for this visitor, by hash
    assert not config.LOCAL_LOG_FILE.exists()


def test_lead_under_the_limit_is_saved(client, monkeypatch):
    monkeypatch.setattr(store, "leads_today", lambda ip_hash: config.RATE_LIMIT_LEADS_PER_DAY - 1)
    assert client.post("/api/lead", json=GOOD_LEAD).status_code == 200


def test_lead_storage_failure_tells_the_visitor(client, monkeypatch):
    def failing(lead, ip_hash):
        raise store.StoreError("database down")

    monkeypatch.setattr(store, "save_lead", failing)
    res = client.post("/api/lead", json=GOOD_LEAD)
    assert res.status_code == 503
    assert res.json() == {"ok": False, "message": t("busy", "es")}


# --- health ------------------------------------------------------------------------------


def test_health_reports_settings_without_secrets(client, monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["git_sha"] == "local"
    assert body["index"] == INDEX
    assert body["models"] == {"chat": config.CHAT_MODEL, "embed": config.EMBED_MODEL}
    assert body["settings"] == {
        "retrieval_mode": "hybrid", "top_k": config.TOP_K, "min_cosine": config.MIN_COSINE,
        "cosine_gate": True, "history_turns": config.HISTORY_TURNS, "max_message_chars": config.MAX_MESSAGE_CHARS,
    }
    assert body["storage"] == "local"
    assert "sk-test" not in str(body)


def test_health_is_degraded_without_an_api_key(client):
    body = client.get("/api/health").json()
    assert body["status"] == "degraded"


@pytest.mark.parametrize("mode,status", [("hybrid", "degraded"), ("dense", "degraded"), ("bm25", "ok")])
def test_health_without_embeddings_is_degraded_unless_bm25(client, monkeypatch, mode, status):
    # Without embeddings there is no cosine, so hybrid/dense lose the "I don't know" gate.
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "RETRIEVAL_MODE", mode)
    monkeypatch.setattr(retrieve, "index_info", lambda: {**INDEX, "has_embeddings": False})
    body = client.get("/api/health").json()
    assert body["status"] == status
    assert body["settings"]["cosine_gate"] is False


def test_health_cosine_gate_is_off_in_bm25_mode_even_with_embeddings(client, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "bm25")
    assert client.get("/api/health").json()["settings"]["cosine_gate"] is False


def test_health_is_degraded_when_the_index_is_broken(client, monkeypatch):
    def missing_index():
        raise FileNotFoundError("data/chunks.json")

    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(retrieve, "index_info", missing_index)
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "degraded"
    assert "error" in res.json()["index"]


def test_health_shows_the_short_git_sha_on_vercel(client, monkeypatch):
    monkeypatch.setenv("VERCEL_GIT_COMMIT_SHA", "0123456789abcdef")
    assert client.get("/api/health").json()["git_sha"] == "0123456"


# --- static UI ----------------------------------------------------------------------------


def test_root_serves_the_chat_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "PlacidWay Assistant" in res.text
