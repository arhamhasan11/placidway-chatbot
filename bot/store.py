"""Saves chat events and quote requests: Postgres when DATABASE_URL is set, else a local
JSON-lines file. Called by bot/chat.py (logging, rate limits) and app.py (leads).
List the questions the bot could not answer: python -m bot.store"""

import hashlib
import hmac
import json
import logging
from datetime import UTC, datetime, timedelta

import psycopg
from psycopg.rows import dict_row

import config
from bot.schemas import LeadRequest

log = logging.getLogger(__name__)

# Columns of the events table. Every other key of an event dict goes into its meta jsonb.
EVENT_COLUMNS = (
    "ip_hash", "outcome", "language", "message", "standalone_question", "best_cosine", "latency_ms",
)
GAP_OUTCOMES = ("not_found", "guard_fallback")  # questions the pages could not answer
GAP_FIELDS = ("ts", "message", "standalone_question", "outcome", "best_cosine")
# Keys hash_ip when IP_HASH_SALT is empty. Fine for local development only: the value is public,
# so production must set IP_HASH_SALT or a leaked log could be reversed by hashing every IPv4 address.
DEV_IP_HASH_SALT = "placidway-bot-local-dev"

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS events (
        id bigserial PRIMARY KEY, ts timestamptz NOT NULL DEFAULT now(),
        ip_hash text, outcome text, language text, message text,
        standalone_question text, best_cosine real, latency_ms int, meta jsonb)""",
    "CREATE INDEX IF NOT EXISTS events_ip_hash_ts ON events (ip_hash, ts)",  # rate-limit lookups
    """CREATE TABLE IF NOT EXISTS leads (
        id bigserial PRIMARY KEY, ts timestamptz NOT NULL DEFAULT now(),
        name text, email text, consent bool, question text, language text, ip_hash text)""",
)

_tables_ready = False  # the only module state: CREATE TABLE runs once per process


class StoreError(Exception):
    """Storage failed and the caller must say so (lead not saved, gaps unreadable)."""


def hash_ip(ip: str) -> str:
    """Pseudonymous visitor id: no raw IP is stored. Keyed with a secret (HMAC), so it can't be
    reversed by hashing all 4 billion IPv4 addresses; salted with the UTC date, so one visitor
    can't be followed across days. Used for both events and leads, so they match."""
    key = (config.IP_HASH_SALT or DEV_IP_HASH_SALT).encode()
    day = datetime.now(UTC).date().isoformat()
    digest = hmac.new(key, f"{day}|{ip}".encode(), hashlib.sha256).hexdigest()
    return digest[:16]  # 64 bits: no collisions at this traffic


def _connect() -> psycopg.Connection:
    """A new connection per call: a serverless function cannot keep a pool between requests."""
    global _tables_ready
    # 5 s: long enough for a sleeping Neon database to wake, short enough to fail a turn fast.
    # autocommit: every write is a single INSERT, so there is nothing to roll back.
    conn = psycopg.connect(config.DATABASE_URL, connect_timeout=5, autocommit=True, row_factory=dict_row)
    if not _tables_ready:
        try:
            for statement in SCHEMA:
                conn.execute(statement)
        except psycopg.Error:
            conn.close()
            raise
        _tables_ready = True
    return conn


def _now() -> str:
    # One fixed ISO format, so timestamps in the local file also sort as plain strings.
    return datetime.now(UTC).isoformat(timespec="seconds")


def _append_local(record: dict) -> None:
    """Append one record to the dev log file. Raises OSError (e.g. read-only disk on Vercel)."""
    with config.LOCAL_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _read_local() -> list[dict]:
    try:
        lines = config.LOCAL_LOG_FILE.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []  # nothing logged yet
    except OSError as exc:
        raise StoreError("local log unreadable") from exc
    try:
        return [json.loads(line) for line in lines if line.strip()]
    except ValueError as exc:
        raise StoreError("local log has a broken line") from exc


def log_event(event: dict) -> None:
    """Record one chat turn. Never raises: a lost log line must not cost the user an answer."""
    row = {name: event.get(name) for name in EVENT_COLUMNS}
    if row["best_cosine"] is not None:
        row["best_cosine"] = float(row["best_cosine"])  # numpy floats from retrieval can't be adapted
    meta = {key: value for key, value in event.items() if key not in EVENT_COLUMNS}
    if not config.DATABASE_URL:
        try:
            _append_local({"type": "event", "ts": _now(), **row, "meta": meta})
        except OSError as exc:
            log.warning("event not logged (local file): %s", exc)
        return
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO events (ip_hash, outcome, language, message, standalone_question,"
                " best_cosine, latency_ms, meta) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
                (*row.values(), json.dumps(meta, ensure_ascii=False, default=str)),  # EVENT_COLUMNS order
            )
    except psycopg.Error as exc:
        log.warning("event not logged (database): %s", exc)


def _count(sql: str, params: tuple) -> int:
    """Run a `count(*) AS n` query for a limit. 0 when unknown: limits fail open."""
    if not config.DATABASE_URL:
        return 0  # local development has no limits
    try:
        with _connect() as conn:
            return conn.execute(sql, params).fetchone()["n"]
    except psycopg.Error as exc:
        # Better to serve a possible abuser than to lock everyone out while the database is down.
        log.warning("limit not checked (database): %s", exc)
        return 0


def recent_count(ip_hash: str, seconds: int) -> int:
    """Chat turns from this visitor in the last `seconds` (rate limit). Blocked turns don't count,
    so a visitor who keeps retrying while blocked is let back in when the window passes.
    IS DISTINCT FROM, not <>: a NULL outcome still counts."""
    return _count(
        "SELECT count(*) AS n FROM events WHERE ip_hash = %s AND ts > now() - %s"
        " AND outcome IS DISTINCT FROM 'rate_limited'",
        (ip_hash, timedelta(seconds=seconds)),
    )


def leads_today(ip_hash: str) -> int:
    """Quote requests from this visitor today. ip_hash changes at UTC midnight, so no date filter."""
    return _count("SELECT count(*) AS n FROM leads WHERE ip_hash = %s", (ip_hash,))


def save_lead(lead: LeadRequest, ip_hash: str) -> None:
    """Store a quote request. Raises StoreError, because the visitor must know it was not sent."""
    if not config.DATABASE_URL:
        try:
            _append_local({"type": "lead", "ts": _now(), **lead.model_dump(), "ip_hash": ip_hash})
        except OSError as exc:
            log.error("lead not saved (local file): %s", type(exc).__name__)
            raise StoreError("lead not saved") from exc
        return
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO leads (name, email, consent, question, language, ip_hash)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (lead.name, lead.email, lead.consent, lead.question, lead.language, ip_hash),
            )
    except psycopg.Error as exc:
        # Only the error class and SQL state: a database message can quote the email.
        log.error("lead not saved (database): %s %s", type(exc).__name__, exc.sqlstate)
        raise StoreError("lead not saved") from exc


def list_gaps(limit: int = 50) -> list[dict]:
    """Latest questions the bot could not answer, newest first: the content gaps. Raises StoreError."""
    if not config.DATABASE_URL:
        events = [r for r in _read_local() if r.get("type") == "event" and r.get("outcome") in GAP_OUTCOMES]
        return [{field: r.get(field) for field in GAP_FIELDS} for r in reversed(events)][:limit]
    try:
        with _connect() as conn:
            return conn.execute(
                "SELECT ts, message, standalone_question, outcome, best_cosine FROM events"
                " WHERE outcome = ANY(%s) ORDER BY ts DESC LIMIT %s",
                (list(GAP_OUTCOMES), limit),
            ).fetchall()
    except psycopg.Error as exc:
        raise StoreError("gaps unreadable") from exc


if __name__ == "__main__":
    # Content-gap report: what visitors asked that the 7 pages could not answer.
    for gap in list_gaps():
        print(f"{gap['ts']}  {gap['outcome']:<14}  {gap['standalone_question'] or gap['message']}")
