"""Storage: local JSON-lines mode end to end, and the Postgres paths with a fake connection
(no database needed). Run: python -m pytest tests/test_store.py"""

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

import config
from bot import store
from bot.schemas import LeadRequest

LEAD = LeadRequest(name="Ana", email="ana@example.com", consent=True, question="Price?", language="es")


@pytest.fixture
def local(tmp_path, monkeypatch):
    """Local mode, writing to a throwaway file."""
    path = tmp_path / "log.jsonl"
    monkeypatch.setattr(config, "DATABASE_URL", "")
    monkeypatch.setattr(config, "LOCAL_LOG_FILE", path)
    return path


def records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_log_event_splits_known_columns_from_meta(local):
    store.log_event({"ip_hash": "abc", "outcome": "answered", "message": "hola", "best_cosine": 0.61,
                     "sources": ["https://www.placidway.com/x"]})
    [record] = records(local)
    assert record["type"] == "event"
    assert record["outcome"] == "answered" and record["message"] == "hola"
    assert record["latency_ms"] is None  # every known column is present, even when not given
    assert record["meta"] == {"sources": ["https://www.placidway.com/x"]}


def test_log_event_never_raises_when_the_file_cannot_be_written(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", "")
    monkeypatch.setattr(config, "LOCAL_LOG_FILE", tmp_path / "no-such-dir" / "log.jsonl")
    store.log_event({"outcome": "answered"})  # read-only disk on Vercel: must not raise


def test_limits_are_zero_without_a_database(local):
    store.log_event({"ip_hash": "abc", "outcome": "answered"})
    store.save_lead(LEAD, "abc")
    assert store.recent_count("abc", 60) == 0
    assert store.leads_today("abc") == 0


def test_save_lead_writes_a_lead_record(local):
    store.save_lead(LEAD, "abc")
    [record] = records(local)
    assert record["type"] == "lead"
    assert record["email"] == "ana@example.com" and record["consent"] is True and record["ip_hash"] == "abc"


def test_save_lead_raises_when_the_file_cannot_be_written(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", "")
    monkeypatch.setattr(config, "LOCAL_LOG_FILE", tmp_path / "no-such-dir" / "log.jsonl")
    with pytest.raises(store.StoreError):
        store.save_lead(LEAD, "abc")


def test_list_gaps_returns_only_unanswered_newest_first(local):
    turns = [("not_found", "q1"), ("answered", "q2"), ("guard_fallback", "q3"), ("busy", "q4")]
    for outcome, message in turns:
        store.log_event({"outcome": outcome, "message": message, "standalone_question": message.upper()})
    gaps = store.list_gaps()
    assert [g["message"] for g in gaps] == ["q3", "q1"]
    assert set(gaps[0]) == {"ts", "message", "standalone_question", "outcome", "best_cosine"}
    assert [g["message"] for g in store.list_gaps(limit=1)] == ["q3"]


def test_list_gaps_is_empty_before_anything_is_logged(local):
    assert store.list_gaps() == []


def on_day(monkeypatch, day: int) -> None:
    """Make store believe it is noon UTC on October `day`, 2026."""

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, day, 12, tzinfo=UTC)

    monkeypatch.setattr(store, "datetime", FixedDatetime)


def test_hash_ip_is_stable_within_a_day_and_hides_the_address(monkeypatch):
    monkeypatch.setattr(config, "IP_HASH_SALT", "test-salt")
    assert store.hash_ip("203.0.113.5") == store.hash_ip("203.0.113.5")
    assert store.hash_ip("203.0.113.5") != store.hash_ip("203.0.113.6")
    assert "203" not in store.hash_ip("203.0.113.5")
    assert len(store.hash_ip("203.0.113.5")) == 16


def test_hash_ip_changes_with_the_utc_day(monkeypatch):
    monkeypatch.setattr(config, "IP_HASH_SALT", "test-salt")
    on_day(monkeypatch, 28)
    today = store.hash_ip("203.0.113.5")
    on_day(monkeypatch, 29)
    assert store.hash_ip("203.0.113.5") != today


def test_hash_ip_is_keyed_by_the_secret_salt(monkeypatch):
    # Without the key, hashing every IPv4 address with the date can't rebuild the table.
    on_day(monkeypatch, 28)
    monkeypatch.setattr(config, "IP_HASH_SALT", "secret-one")
    one = store.hash_ip("203.0.113.5")
    expected = hmac.new(b"secret-one", b"2026-10-28|203.0.113.5", hashlib.sha256).hexdigest()[:16]
    assert one == expected
    monkeypatch.setattr(config, "IP_HASH_SALT", "secret-two")
    assert store.hash_ip("203.0.113.5") != one


def test_hash_ip_uses_the_dev_salt_when_none_is_set(monkeypatch):
    monkeypatch.setattr(config, "IP_HASH_SALT", "")
    dev = store.hash_ip("203.0.113.5")
    monkeypatch.setattr(config, "IP_HASH_SALT", store.DEV_IP_HASH_SALT)
    assert store.hash_ip("203.0.113.5") == dev


# --- Postgres paths, with psycopg.connect replaced -------------------------------------


class FakeConnection:
    """Records every statement; checks placeholders match parameters like Postgres would."""

    def __init__(self, executed: list):
        self.executed = executed

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def close(self):
        pass

    def execute(self, sql, params=()):
        assert sql.count("%s") == len(params), sql
        self.executed.append((sql, params))
        return self

    def fetchone(self):
        return {"n": 3}

    def fetchall(self):
        return [{"name": "Ana"}]


@pytest.fixture
def database(monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", "host=db.invalid dbname=test")
    monkeypatch.setattr(store, "_tables_ready", False)
    return monkeypatch


def test_log_event_inserts_columns_and_json_meta(database):
    executed = []
    database.setattr(psycopg, "connect", lambda *args, **kwargs: FakeConnection(executed))
    store.log_event({"ip_hash": "abc", "outcome": "answered", "latency_ms": 900, "chunk_ids": ["a#1"]})
    creates = [sql for sql, _ in executed if sql.startswith("CREATE")]
    assert len(creates) == 3
    sql, params = executed[-1]
    assert sql.startswith("INSERT INTO events")
    assert params[0] == "abc" and params[6] == 900
    assert json.loads(params[7]) == {"chunk_ids": ["a#1"]}


def test_recent_count_reads_the_count_and_skips_blocked_turns(database):
    executed = []
    database.setattr(psycopg, "connect", lambda *args, **kwargs: FakeConnection(executed))
    assert store.recent_count("abc", 60) == 3
    sql, params = executed[-1]
    assert params == ("abc", timedelta(seconds=60))
    # A blocked visitor's retries must not keep extending the block.
    assert "outcome IS DISTINCT FROM 'rate_limited'" in sql


def test_leads_today_counts_this_visitors_leads(database):
    executed = []
    database.setattr(psycopg, "connect", lambda *args, **kwargs: FakeConnection(executed))
    assert store.leads_today("abc") == 3
    sql, params = executed[-1]
    assert "FROM leads WHERE ip_hash = %s" in sql and params == ("abc",)


def unreachable(*args, **kwargs):
    raise psycopg.OperationalError("connection to server at db.invalid failed")


def test_database_down_fails_open_for_logging_and_rate_limits(database):
    database.setattr(psycopg, "connect", unreachable)
    store.log_event({"outcome": "answered"})  # must not raise
    assert store.recent_count("abc", 60) == 0
    assert store.leads_today("abc") == 0


def test_database_down_raises_for_leads_and_gaps(database):
    database.setattr(psycopg, "connect", unreachable)
    with pytest.raises(store.StoreError):
        store.save_lead(LEAD, "abc")
    with pytest.raises(store.StoreError):
        store.list_gaps()


def test_failed_lead_never_logs_the_email(database, caplog, capsys):
    class Rejecting(FakeConnection):
        def execute(self, sql, params=()):
            if sql.startswith("INSERT"):  # Postgres error details can quote the row
                raise psycopg.IntegrityError("Key (email)=(ana@example.com) already exists")
            return self

    database.setattr(psycopg, "connect", lambda *args, **kwargs: Rejecting([]))
    with pytest.raises(store.StoreError):
        store.save_lead(LEAD, "abc")
    captured = capsys.readouterr()
    assert "ana@example.com" not in caplog.text + captured.out + captured.err
