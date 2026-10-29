"""Data shapes shared by ingestion, retrieval, the two LLM calls and the API."""

from typing import Literal

from pydantic import BaseModel, Field

import config

# NUL characters are rejected at the door: Postgres text columns cannot store them, so a message
# containing one would make the log or lead insert fail.
NO_NUL = r"^[^\x00]*$"

# --- Knowledge base (written by ingest/build.py, read by bot/retrieve.py) --------


class Chunk(BaseModel):
    id: str  # "<slug>#<n>", e.g. "adrenal-package#7"
    url: str
    label: str  # page label from sources.yaml, shown to users as the source title
    section: str  # heading path inside the page, e.g. "Cost of ... > Out-Patient Program"
    kind: Literal["text", "table_row", "table", "faq", "list"]  # "table" = all rows of one table
    text: str  # what is searched and shown to the model; begins with "label | section"
    hash: str  # sha256 of text, first 16 hex chars; unchanged text keeps its embedding


class Hit(BaseModel):
    chunk: Chunk
    ref: str  # "C1".."Ck", the handle the answer model cites this chunk by
    cosine: float  # embedding similarity to the question, 0..1 (0 in bm25-only mode)


# --- LLM call 1: understand the message -------------------------------------------

Intent = Literal["question", "quote_or_booking", "medical_advice", "out_of_scope", "greeting"]


class Understanding(BaseModel):
    language: str  # ISO 639-1 code of the user's message, e.g. "en", "es", "ar"
    standalone_question: str  # in English, self-contained: follow-ups resolved from history
    intents: list[Intent]
    # One short English search per distinct part of the question ("ITC street address", "ITC doctors"),
    # so one topic can't take every retrieval slot. Empty = search the standalone question.
    searches: list[str] = Field(default_factory=list)


# --- LLM call 2: grounded answer -----------------------------------------------------


class Draft(BaseModel):
    answerable: bool  # false when the chunks do not contain the answer
    answer: str  # in the user's language
    citations: list[str]  # refs of the chunks the answer relies on, e.g. ["C2", "C5"]


# --- HTTP API ---------------------------------------------------------------------------


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=config.MAX_HISTORY_CHARS, pattern=NO_NUL)
    disclaimer: bool = False  # the UI echoes back whether this reply carried the disclaimer


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=config.MAX_MESSAGE_CHARS, pattern=NO_NUL)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=40)


class Source(BaseModel):
    title: str
    url: str


Outcome = Literal[
    "answered",  # grounded answer with sources
    "not_found",  # nothing relevant in the pages
    "guard_fallback",  # the draft failed the code checks twice
    "out_of_scope",
    "greeting",
    "booking",  # only asked to book / get a quote
    "busy",  # LLM provider failed
    "rate_limited",
]


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    outcome: Outcome
    language: str
    disclaimer: str | None = None
    show_lead_form: bool = False


class LeadRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120, pattern=NO_NUL)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=200)
    consent: Literal[True]  # the form cannot be sent without ticking consent
    question: str = Field(default="", max_length=1000, pattern=NO_NUL)
    language: str = Field(default="en", max_length=8)
