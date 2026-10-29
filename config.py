"""Every setting the bot uses, in one place.

Tunables are plain constants: edit them here and redeploy. Secrets come from
environment variables only (a local .env file, Vercel env vars, GitHub secrets)
and are never committed.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")  # local development only; a missing file is fine

# --- Secrets (environment only) ------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")  # Neon Postgres; empty = log to data/local_log.jsonl
IP_HASH_SALT = os.getenv("IP_HASH_SALT", "")  # keys the visitor hash, so a leaked log can't be reversed to IPs

# --- Models ---------------------------------------------------------------------
# Any OpenAI-compatible endpoint works (Groq, Gemini, ...): change these three lines.
LLM_BASE_URL = None  # None = api.openai.com
CHAT_MODEL = "gpt-6-luna"  # used for both LLM calls (understand + answer)
REASONING_EFFORT = "none"  # "none" keeps latency low; set to None for models without reasoning
EMBED_MODEL = "text-embedding-3-small"  # changing this needs a re-ingest (python -m ingest.build)
LLM_TIMEOUT_S = 15  # per request; up to 4 model calls per turn must fit in vercel.json maxDuration
UNDERSTAND_MAX_TOKENS = 300
ANSWER_MAX_TOKENS = 700

# --- Retrieval ------------------------------------------------------------------
RETRIEVAL_MODE = "hybrid"  # "hybrid" (BM25 + embeddings, merged by RRF) | "dense" | "bm25"
TOP_K = 8  # chunks shown to the answer model; a table answer needs several row chunks
RRF_K = 60  # standard constant from the Reciprocal Rank Fusion paper
MAX_PER_SECTION = 2  # chunks taken from one section of a page; stops near-copies crowding out other evidence
MAX_SEARCHES = 3  # searches per question at most (the understand step splits multi-part questions)
MIN_COSINE = 0.38  # best embedding match below this = "I don't know". Set from eval/run_eval.py --retrieval
# (2026-10-28): answerable questions bottomed at 0.437, unanswerable ones topped at 0.326.
# Reranking: search returns RERANK_CANDIDATES chunks, OpenAI's Decisions API keeps the TOP_K most relevant.
RERANK = True  # False = plain search order (no extra call)
RERANK_CANDIDATES = 30  # wide enough for combined lists (all doctors, all reviews), which rank 10th-25th
RERANK_TIMEOUT_S = 8  # measured ~0.6 s; on timeout the search order is used
DECISIONS_MODEL = "gpt-6-luna"  # the only model the Decisions API supports (Oct 2026)
DECISIONS_URL = "https://api.openai.com/v1/decisions"
MAX_SOURCES = 3  # source links shown under an answer

# --- Conversation ---------------------------------------------------------------
HISTORY_TURNS = 6  # previous messages sent to the understand step; 0 = no history sent to the model
MAX_HISTORY_CHARS = 4000  # longest history message accepted (the UI trims to this)
MAX_MESSAGE_CHARS = 500
# The medical disclaimer is shown only for medical questions, and not repeated within this many
# assistant turns; 0 = on every medical answer.
DISCLAIMER_EVERY_N_TURNS = 3
SUPPORTED_LANGUAGES = ("en", "es", "ar")  # fixed replies in messages.yaml; answers follow the user's language anyway
QUOTE_URL = "https://www.placidway.com/contact-us"

# --- Abuse limits (need DATABASE_URL to work across serverless instances) --------
RATE_LIMIT_PER_MINUTE = 8
RATE_LIMIT_PER_DAY = 80
RATE_LIMIT_LEADS_PER_DAY = 5  # quote requests per visitor; stops a script flooding the leads table

# --- Ingestion ------------------------------------------------------------------
USER_AGENT = "PlacidWayAssessmentBot/1.0 (candidate assessment; respects robots.txt)"
MIN_DELAY_S = 1.0  # the brief asks ~1 request/second; robots.txt Crawl-delay wins if larger (it is 10)
FETCH_TIMEOUT_S = 20
CHUNK_MAX_CHARS = 1200  # long sections are split at paragraph boundaries below this size
MIN_PAGE_CHARS = 400  # a page that extracts to less text than this is treated as broken
MAX_SHRINK = 0.5  # refuse a refresh if a page loses more than half its text

# --- Files ----------------------------------------------------------------------
SOURCES_FILE = ROOT / "sources.yaml"
MESSAGES_FILE = ROOT / "messages.yaml"
PROMPTS_DIR = ROOT / "prompts"
DATA_DIR = ROOT / "data"
CHUNKS_FILE = DATA_DIR / "chunks.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
REPORT_FILE = DATA_DIR / "report.json"
LOCAL_LOG_FILE = DATA_DIR / "local_log.jsonl"
