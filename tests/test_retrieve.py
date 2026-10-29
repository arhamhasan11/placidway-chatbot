"""bot/retrieve.py on a tiny synthetic index: RRF ordering, modes, and refusing a stale index.
One test also ranks the real data/chunks.json, when it has been built."""

import json
import logging
import math

import numpy as np
import pytest

import config
from bot import llm, retrieve
from bot.schemas import Chunk

# Six chunks: four of them must NOT contain "ozone", or BM25's IDF for it drops to zero.
TEXTS = {
    "a": "Page A | Ozone | ozone ozone ozone treatment",
    "b": "Page B | Stem cells | live cell therapy",
    "c": "Page C | Ozone | ozone therapy at Fizikon",
    "d": "Page D | Doctors | medical director",
    "e": "Page E | Location | address in Tijuana",
    "f": "Page F | Reviews | rating from patients",
}
# Cosine of each chunk with the (fake) query embedding. Dense order: b, c, d, a, e, f.
COSINES = {"a": 0.1, "b": 0.9, "c": 0.8, "d": 0.5, "e": 0.05, "f": 0.0}


def unit_row(cos: float) -> list[float]:
    return [cos, math.sqrt(1 - cos**2), 0.0]  # unit length, so its dot with [1, 0, 0] is `cos`


@pytest.fixture
def index_files(tmp_path, monkeypatch):
    chunks = [
        {
            "id": slug,
            "url": f"https://example.com/{slug}",
            "label": f"Page {slug.upper()}",
            "section": "S",
            "kind": "text",
            "text": text,
            "hash": slug,
        }
        for slug, text in TEXTS.items()
    ]
    (tmp_path / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    np.save(tmp_path / "embeddings.npy", np.array([unit_row(COSINES[s]) for s in TEXTS], dtype=np.float32))
    (tmp_path / "report.json").write_text(json.dumps({"embed_model": config.EMBED_MODEL, "total_chunks": 6}))
    monkeypatch.setattr(config, "CHUNKS_FILE", tmp_path / "chunks.json")
    monkeypatch.setattr(config, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy")
    monkeypatch.setattr(config, "REPORT_FILE", tmp_path / "report.json")
    retrieve.reset()
    yield tmp_path
    retrieve.reset()


@pytest.fixture
def embed_calls(monkeypatch):
    calls = []

    def fake_embed(texts):
        calls.append(texts)
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(llm, "embed", fake_embed)
    return calls


@pytest.fixture
def no_embed(monkeypatch):
    def fail(texts):
        raise AssertionError("bm25 search must not call the embeddings API")

    monkeypatch.setattr(llm, "embed", fail)


def ids(hits):
    return [hit.chunk.id for hit in hits]


def test_hybrid_merges_rankings_with_rrf(index_files, embed_calls, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "hybrid")
    hits = retrieve.search("ozone", k=4)
    # BM25: a, c.  Dense: b, c, d, a.  RRF (k=60): c = 1/62 + 1/62 beats a = 1/61 + 1/64,
    # so the chunk that is second in both lists wins over one that is first in only one.
    assert ids(hits) == ["c", "a", "b", "d"]
    assert [hit.ref for hit in hits] == ["C1", "C2", "C3", "C4"]
    assert embed_calls == [["ozone"]]


def test_hits_carry_raw_cosine_not_rrf_score(index_files, embed_calls, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "hybrid")
    hits = retrieve.search("ozone", k=3)
    assert [round(hit.cosine, 3) for hit in hits] == [0.8, 0.1, 0.9]


def test_default_k_is_top_k(index_files, embed_calls, monkeypatch):
    monkeypatch.setattr(config, "TOP_K", 2)
    assert len(retrieve.search("ozone")) == 2


def test_dense_mode_ranks_by_cosine(index_files, embed_calls, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "dense")
    assert ids(retrieve.search("ozone", k=4)) == ["b", "c", "d", "a"]


def test_bm25_mode_makes_no_embedding_call(index_files, no_embed, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "bm25")
    hits = retrieve.search("ozone", k=4)
    assert ids(hits) == ["a", "c"]  # chunks sharing no word with the query are dropped
    assert all(hit.cosine == 0.0 for hit in hits)


def test_bm25_returns_nothing_for_unknown_words(index_files, no_embed, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "bm25")
    assert retrieve.search("weather in cancun") == []


def test_numbers_match_with_or_without_thousands_comma():
    assert retrieve.tokenize("How much is the $18,995 program?") == ["much", "18995", "program"]


def test_words_on_every_page_are_not_search_terms():
    assert retrieve.tokenize("Cost of ITC's alternative cancer treatments in Tijuana") == ["cost"]


@pytest.mark.parametrize(
    ("mode", "has_embeddings", "expected"),
    [("hybrid", True, True), ("dense", True, True), ("bm25", True, False), ("hybrid", False, False)],
)
def test_uses_embeddings(index_files, monkeypatch, mode, has_embeddings, expected):
    if not has_embeddings:
        (index_files / "embeddings.npy").unlink()
    monkeypatch.setattr(config, "RETRIEVAL_MODE", mode)
    assert retrieve.uses_embeddings() is expected


REAL_CHUNKS = config.CHUNKS_FILE  # read at import, before any fixture points config elsewhere


@pytest.mark.skipif(not REAL_CHUNKS.exists(), reason="data/chunks.json not built")
def test_real_index_ranks_a_price_row_above_video_titles(monkeypatch, tmp_path, no_embed):
    # The query that once put a list of patient-story video titles first in BM25 (see DOMAIN_STOPWORDS).
    monkeypatch.setattr(config, "CHUNKS_FILE", REAL_CHUNKS)
    monkeypatch.setattr(config, "EMBEDDINGS_FILE", tmp_path / "none.npy")  # bm25 needs no vectors
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "bm25")
    retrieve.reset()
    try:
        hits = retrieve.search("What is the cost of alternative cancer treatment at ITC in Tijuana?", k=1000)
    finally:
        retrieve.reset()
    kinds = ["videos" if hit.chunk.id.startswith("itc-videos") else hit.chunk.kind for hit in hits]
    assert "table_row" in kinds  # every table row is a price row
    assert "videos" not in kinds[: kinds.index("table_row")]


def test_missing_embeddings_degrade_to_bm25(index_files, no_embed, monkeypatch, caplog):
    (index_files / "embeddings.npy").unlink()
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "hybrid")
    with caplog.at_level(logging.WARNING, logger="bot.retrieve"):
        assert ids(retrieve.search("ozone")) == ["a", "c"]
        retrieve.search("ozone")
    assert len([r for r in caplog.records if "BM25 only" in r.getMessage()]) == 1  # warned once
    assert retrieve.index_info()["has_embeddings"] is False


def test_refuses_embeddings_from_another_model(index_files, embed_calls):
    (index_files / "report.json").write_text(json.dumps({"embed_model": "some-other-model"}))
    with pytest.raises(RuntimeError, match="some-other-model"):
        retrieve.search("ozone")


def test_refuses_embeddings_that_do_not_match_the_chunks(index_files, embed_calls):
    np.save(index_files / "embeddings.npy", np.ones((2, 3), dtype=np.float32))
    with pytest.raises(RuntimeError, match="6 chunks but 2 embeddings"):
        retrieve.search("ozone")


def test_missing_chunks_file_says_how_to_fix(index_files):
    (index_files / "chunks.json").unlink()
    with pytest.raises(RuntimeError, match="ingest.build"):
        retrieve.search("ozone")


def test_unknown_mode_is_an_error(index_files, embed_calls, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "hybird")
    with pytest.raises(ValueError, match="hybird"):
        retrieve.search("ozone")


def test_index_is_cached_until_reset(index_files, embed_calls):
    assert retrieve.index_info()["loaded_chunks"] == 6
    chunks = json.loads((index_files / "chunks.json").read_text())[:5]
    (index_files / "chunks.json").write_text(json.dumps(chunks))
    (index_files / "embeddings.npy").unlink()
    assert retrieve.index_info()["loaded_chunks"] == 6  # still the cached copy
    retrieve.reset()
    info = retrieve.index_info()
    assert info["loaded_chunks"] == 5
    assert info["has_embeddings"] is False
    assert info["total_chunks"] == 6  # report.json contents come through unchanged


def test_spread_skips_copies_and_caps_one_section(monkeypatch):
    monkeypatch.setattr(config, "MAX_PER_SECTION", 2)

    def chunk(n, url, section, body):
        return Chunk(id=f"p#{n}", url=url, label="P", section=section, kind="text",
                     text=f"P | {section}\n{body}", hash=str(n))

    chunks = [chunk(0, "u1", "Doctors", "Dr A"), chunk(1, "u1", "Doctors", "Dr B"),
              chunk(2, "u1", "Doctors", "Dr C"),            # 3rd from the "Doctors" section: skipped
              chunk(3, "u2", "Cost", "Dr A"),               # same text as #0 on another page: skipped
              chunk(4, "u1", "Clinic details", "Address")]
    index = retrieve.Index(chunks=chunks, bm25=None, embeddings=None, report={})
    assert retrieve._spread(index, [0, 1, 2, 3, 4], k=3) == [0, 1, 4]


@pytest.mark.parametrize("cancer", ["anal", "adrenal", "adenocarcinoma"])
def test_identical_package_tables_keep_the_one_the_user_named(cancer, monkeypatch):
    # The 3 package pages share one price table and only one copy is kept per search: it must be
    # the page the user asked about, so the answer cites the right package.
    if not config.CHUNKS_FILE.exists():
        pytest.skip("no data/chunks.json")
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "bm25")
    retrieve.reset()
    hits = retrieve.search(f"What does the in-patient program for {cancer} cancer cost?")
    first_price = next(h for h in hits if "Total Cost" in h.chunk.text)
    assert first_price.chunk.id.startswith(cancer)
    retrieve.reset()


def test_search_all_gives_every_part_of_the_question_a_turn(monkeypatch):
    def hit(cid):
        chunk = Chunk(id=cid, url="u", label="P", section=cid, kind="text", text=f"P | {cid}\n{cid}", hash=cid)
        return retrieve.Hit(chunk=chunk, ref="?", cosine=0.5)

    results = {"doctors": [hit("doc1"), hit("doc2"), hit("doc3")],
               "address": [hit("addr"), hit("doc1")]}  # doc1 found by both: kept once
    monkeypatch.setattr(retrieve, "search", lambda q, k=None: results[q])
    hits = retrieve.search_all(["doctors", "address"], k=4)
    assert [h.chunk.id for h in hits] == ["doc1", "addr", "doc2", "doc3"]
    assert [h.ref for h in hits] == ["C1", "C2", "C3", "C4"]


def test_search_all_with_one_query_is_a_plain_search(monkeypatch):
    monkeypatch.setattr(retrieve, "search", lambda q, k=None: [q, k])
    assert retrieve.search_all(["what is included?"], k=8) == ["what is included?", 8]


def test_rows_never_block_their_whole_table(monkeypatch):
    monkeypatch.setattr(config, "MAX_PER_SECTION", 2)

    def chunk(n, kind, body):
        return Chunk(id=f"p#{n}", url="u", label="P", section="Costs", kind=kind, text=f"P | Costs\n{body}", hash=str(n))

    chunks = [chunk(0, "table_row", "row A"), chunk(1, "table_row", "row B"),
              chunk(2, "table_row", "row C"), chunk(3, "table", "Full table: A B C")]
    index = retrieve.Index(chunks=chunks, bm25=None, embeddings=None, report={})
    assert retrieve._spread(index, [0, 1, 2, 3], k=4) == [0, 1, 3]  # row C capped, the table is not
