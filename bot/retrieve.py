"""Finds the chunks most relevant to a question. Called by bot/chat.py once per question.

BM25 catches exact words and numbers ("adrenal", "18995"); embeddings catch paraphrases.
Hybrid mode merges the two rankings with Reciprocal Rank Fusion (RRF).
"""

import json
import logging
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

import config
from bot import llm
from bot.schemas import Chunk, Hit

log = logging.getLogger(__name__)

# Only words that carry no topic. Kept tiny on purpose: "not", "per", "without" can matter.
STOPWORDS = frozenset(
    "a an and are at be by can do does for from how i in is it its me my "
    "of on or the this to what which you your".split()
)
# Words on nearly every chunk of these 7 pages, so they carry no signal here ("s" is what is left of
# "Erika's"). BM25's IDF alone could not demote the video-title list chunks: each repeats "ITC",
# "alternative", "cancer", "treatment" and "Tijuana" in every title, so "cost of alternative cancer
# treatment at ITC" ranked a list of patient-story videos above the price rows.
DOMAIN_STOPWORDS = frozenset("itc alternative cancer treatment treatments tijuana s".split())


@dataclass
class Index:
    chunks: list[Chunk]
    bm25: BM25Okapi
    embeddings: np.ndarray | None  # row i = chunk i; None when built with --no-embed
    report: dict  # data/report.json


_index: Index | None = None  # loaded on first search, kept for the life of the process


def tokenize(text: str) -> list[str]:
    """Lowercase word and number tokens. "18,995" becomes "18995", the price list's spelling."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())
    return [token for token in re.findall(r"\w+", text) if token not in STOPWORDS and token not in DOMAIN_STOPWORDS]


def uses_embeddings() -> bool:
    """True when search embeds the query, so hits carry real cosines: chat.py's "I don't know" gate
    needs them. False in bm25 mode, and for an index built with --no-embed."""
    return config.RETRIEVAL_MODE != "bm25" and _load().embeddings is not None


def search_all(queries: list[str], k: int | None = None) -> list[Hit]:
    """One search per part of a question, merged by taking turns: each part's best chunk first,
    then each part's second, and so on. "Address and doctors?" then gets both, instead of ten
    doctor chunks pushing the address out. One query is just search()."""
    k = k or config.TOP_K
    if len(queries) <= 1:
        return search(queries[0] if queries else "", k)
    per_query = [search(q, k) for q in queries[: config.MAX_SEARCHES]]
    merged: list[Hit] = []
    for rank in range(k):
        for hits in per_query:
            if rank < len(hits) and all(h.chunk.id != hits[rank].chunk.id for h in merged):
                merged.append(hits[rank])
    return [hit.model_copy(update={"ref": f"C{n}"}) for n, hit in enumerate(merged[:k], start=1)]


def search(query: str, k: int | None = None) -> list[Hit]:
    """Top-k chunks for `query`, named C1..Ck in rank order."""
    index = _load()
    mode = config.RETRIEVAL_MODE if uses_embeddings() else "bm25"
    if mode == "bm25":
        cosines = np.zeros(len(index.chunks))  # no embedding call at all, so no cosine to report
    else:
        cosines = index.embeddings @ llm.embed([query])[0]  # rows are unit length: dot = cosine

    if mode == "hybrid":
        order = _fuse([_bm25_ranking(index, query), _ranking(cosines)])
    elif mode == "dense":
        order = _ranking(cosines)
    elif mode == "bm25":
        order = _bm25_ranking(index, query)
    else:
        raise ValueError(f"config.RETRIEVAL_MODE must be hybrid, dense or bm25, not {mode!r}")

    top = _spread(index, order, k or config.TOP_K)
    return [Hit(chunk=index.chunks[i], ref=f"C{n}", cosine=float(cosines[i])) for n, i in enumerate(top, start=1)]


def _spread(index: Index, order: list[int], k: int) -> list[int]:
    """The first k chunks in rank order, skipping repeats so other evidence gets a slot:

    - text identical to a better-ranked chunk (the 3 package pages share their price tables). The
      kept copy is the best-ranked one, which is the page the user named, because every chunk
      starts with its page label (tested in test_identical_package_tables_keep_the_one_the_user_named);
    - more than config.MAX_PER_SECTION chunks from one section of a page.
    Without this, six "doctors" chunks pushed the clinic's address out of the results, and three
    copies of one price table hid the price list.
    """
    seen_bodies: set[str] = set()
    taken: dict[tuple[str, str], int] = {}
    top: list[int] = []
    for i in order:
        chunk = index.chunks[i]
        body = chunk.text.split("\n", 1)[-1]  # without the "page | section" first line
        section = (chunk.url, chunk.section)
        # A whole-table chunk is never capped: two single rows must not block the table holding them all.
        capped = chunk.kind != "table" and taken.get(section, 0) >= config.MAX_PER_SECTION
        if body in seen_bodies or capped:
            continue
        seen_bodies.add(body)
        taken[section] = taken.get(section, 0) + 1
        top.append(i)
        if len(top) == k:
            break
    return top


def _ranking(scores: np.ndarray) -> list[int]:
    """Chunk positions, best score first."""
    return [int(i) for i in np.argsort(-scores, kind="stable")]


def _bm25_ranking(index: Index, query: str) -> list[int]:
    """BM25 ranking without zero scores: a chunk sharing no word with the query is no evidence."""
    scores = index.bm25.get_scores(tokenize(query))
    return [i for i in _ranking(scores) if scores[i] > 0]


def _fuse(rankings: list[list[int]]) -> list[int]:
    """Reciprocal Rank Fusion: each ranking adds 1 / (RRF_K + rank); a missing chunk adds nothing.

    RRF uses ranks, not raw scores, so BM25 and cosine scales never need calibrating.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            fused[i] = fused.get(i, 0.0) + 1 / (config.RRF_K + rank)
    return sorted(fused, key=lambda i: fused[i], reverse=True)


def index_info() -> dict:
    """The build report plus what is actually loaded (health check)."""
    index = _load()
    return {**index.report, "loaded_chunks": len(index.chunks), "has_embeddings": index.embeddings is not None}


def reset() -> None:
    """Forget the loaded index, so the next search reads data/ again (tests, refresh)."""
    global _index
    _index = None


def _load() -> Index:
    global _index
    if _index is None:
        _index = _read_index()
    return _index


def _read_index() -> Index:
    if not config.CHUNKS_FILE.exists():
        raise RuntimeError(f"{config.CHUNKS_FILE} is missing: run python -m ingest.build")
    chunks = [Chunk.model_validate(row) for row in json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))]
    if not chunks:
        raise RuntimeError(f"{config.CHUNKS_FILE} has no chunks: run python -m ingest.build")
    report = json.loads(config.REPORT_FILE.read_text(encoding="utf-8")) if config.REPORT_FILE.exists() else {}

    embeddings = np.load(config.EMBEDDINGS_FILE) if config.EMBEDDINGS_FILE.exists() else None
    if embeddings is not None:
        # Vectors from two different models are not comparable: every cosine would be noise.
        if report.get("embed_model") != config.EMBED_MODEL:
            raise RuntimeError(
                f"embeddings were built with {report.get('embed_model')!r} but config.EMBED_MODEL is "
                f"{config.EMBED_MODEL!r}: run python -m ingest.build"
            )
        if embeddings.shape[0] != len(chunks):
            raise RuntimeError(f"{len(chunks)} chunks but {embeddings.shape[0]} embeddings: run python -m ingest.build")
    elif config.RETRIEVAL_MODE != "bm25":
        log.warning("no %s: searching with BM25 only instead of %s", config.EMBEDDINGS_FILE.name, config.RETRIEVAL_MODE)

    return Index(
        chunks=chunks, bm25=BM25Okapi([tokenize(c.text) for c in chunks]), embeddings=embeddings, report=report
    )
