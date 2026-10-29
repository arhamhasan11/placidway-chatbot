"""HTTP API and the chat UI. Vercel imports `app` from this file; locally run
`uvicorn app:app --reload` and open http://127.0.0.1:8000."""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from bot import chat, retrieve, store
from bot.messages import t
from bot.schemas import ChatRequest, ChatResponse, LeadRequest

log = logging.getLogger(__name__)

app = FastAPI(title="PlacidWay Assistant")


def client_ip(request: Request) -> str:
    """The visitor's address. Vercel overwrites x-forwarded-for with the real client IP as the
    first entry, so it can't be spoofed there; locally the socket address is used."""
    first = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if first:
        return first
    return request.client.host if request.client else "unknown"


@app.post("/api/chat")
def api_chat(req: ChatRequest, request: Request) -> ChatResponse:
    return chat.handle_turn(req, client_ip(request))


@app.post("/api/lead")
def api_lead(lead: LeadRequest, request: Request) -> JSONResponse:
    """Quote request from the inline form. LeadRequest already rejects a bad email or no consent (422)."""
    ip_hash = store.hash_ip(client_ip(request))
    # A daily cap per visitor, so a script can't fill the leads table (and the sales inbox) with junk.
    if store.leads_today(ip_hash) >= config.RATE_LIMIT_LEADS_PER_DAY:
        return JSONResponse({"ok": False, "message": t("rate_limited", lead.language)}, status_code=429)
    try:
        store.save_lead(lead, ip_hash)
    except store.StoreError:
        return JSONResponse({"ok": False, "message": t("busy", lead.language)}, status_code=503)
    return JSONResponse({"ok": True, "message": t("lead_thanks", lead.language)})


@app.get("/api/health")
def api_health() -> dict:
    """Status for uptime checks and the UI. Reads local files only; never calls an LLM."""
    index_ok = True
    try:
        index = retrieve.index_info()
    except Exception as exc:  # broad on purpose: health must report a broken index, whatever broke it
        log.error("index not loaded", exc_info=exc)
        index, index_ok = {"error": f"index not loaded ({type(exc).__name__})"}, False
    has_embeddings = bool(index.get("has_embeddings"))
    # hybrid/dense without embeddings loses its "I don't know" gate and its paraphrase matching.
    embeddings_ok = config.RETRIEVAL_MODE == "bm25" or has_embeddings
    return {
        "status": "ok" if config.OPENAI_API_KEY and index_ok and embeddings_ok else "degraded",
        "git_sha": os.getenv("VERCEL_GIT_COMMIT_SHA", "")[:7] or "local",
        "index": index,
        "models": {"chat": config.CHAT_MODEL, "embed": config.EMBED_MODEL},
        "settings": {
            "retrieval_mode": config.RETRIEVAL_MODE,
            "top_k": config.TOP_K,
            "min_cosine": config.MIN_COSINE,
            "cosine_gate": config.RETRIEVAL_MODE != "bm25" and has_embeddings,
            "history_turns": config.HISTORY_TURNS,
            "max_message_chars": config.MAX_MESSAGE_CHARS,  # the chat page sizes its input box from this
        },
        "storage": "postgres" if config.DATABASE_URL else "local",
    }


@app.exception_handler(Exception)
def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: the visitor gets the fixed 'busy' reply, never a stack trace."""
    log.error("unhandled error on %s", request.url.path, exc_info=exc)  # the traceback stays server-side
    body = ChatResponse(answer=t("busy", "en"), sources=[], outcome="busy", language="en")
    return JSONResponse(body.model_dump(), status_code=500)


# Mounted last so the /api routes above win. On Vercel the CDN serves public/ itself and the
# folder may be left out of the function bundle, so a missing folder must not crash the import.
if (config.ROOT / "public").is_dir():
    app.mount("/", StaticFiles(directory=config.ROOT / "public", html=True), name="ui")
