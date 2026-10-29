"""The build writes a consistent knowledge base, or nothing at all when a quality gate fails."""

import json

import numpy as np
import pytest

import config
from bot import llm
from ingest import build
from ingest.extract import Block
from ingest.fetch import fetch_pages, load_sources

PAGES = load_sources()["pages"]
_EXTRACTED: dict = {}


@pytest.fixture(autouse=True)
def reuse_extraction(monkeypatch):
    """extract() is pure and parsing 3 MB of HTML takes seconds, so each page is parsed once per run."""
    real_extract = build.extract

    def cached(html, selector, drop_selectors):
        key = (hash(html), selector, tuple(drop_selectors))
        if key not in _EXTRACTED:
            _EXTRACTED[key] = real_extract(html, selector, drop_selectors)
        return _EXTRACTED[key]

    monkeypatch.setattr(build, "extract", cached)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CHUNKS_FILE", tmp_path / "chunks.json")
    monkeypatch.setattr(config, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy")
    monkeypatch.setattr(config, "REPORT_FILE", tmp_path / "report.json")
    return tmp_path


@pytest.fixture
def fake_llm(monkeypatch) -> list[int]:
    """Stands in for llm.embed: returns unit vectors and records how many texts each call embedded."""
    calls = []

    def embed(texts):
        calls.append(len(texts))
        vectors = np.random.default_rng(len(texts)).random((len(texts), 8), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    monkeypatch.setattr(llm, "embed", embed)
    return calls


def serve(monkeypatch, replace: dict[str, str]) -> None:
    """Offline pages, with some slugs' HTML replaced (a broken or changed page)."""
    def fake_fetch(pages, offline=False, client=None):
        return {**fetch_pages(pages, offline=True), **replace}
    monkeypatch.setattr(build, "fetch_pages", fake_fetch)


def snapshot(folder) -> dict[str, bytes]:
    files = [p for p in folder.rglob("*") if p.is_file()]
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in files}


def report(folder) -> dict:
    return json.loads((folder / "report.json").read_text(encoding="utf-8"))


def test_offline_build_writes_consistent_files(data_dir):
    assert build.main(["--offline", "--no-embed"]) == 0
    chunks = json.loads((data_dir / "chunks.json").read_text(encoding="utf-8"))
    rep = report(data_dir)
    assert rep["total_chunks"] == len(chunks) == sum(p["chunks"] for p in rep["pages"].values())
    assert rep["embedded"] is False and rep["embed_model"] == config.EMBED_MODEL
    assert not (data_dir / "embeddings.npy").exists()
    assert sorted(rep["pages"]) == sorted(p["slug"] for p in PAGES)
    assert rep["pages"]["itc-price-list"]["table_rows"] == 7
    assert set(rep["pages"]["itc-profile"]) == {
        "url", "label", "chars", "chunks", "table_rows", "sha"}
    assert sorted(p.name for p in (data_dir / "pages").iterdir()) == sorted(
        f"{p['slug']}.md" for p in PAGES)
    assert "–" in (data_dir / "report.json").read_text(encoding="utf-8")  # ensure_ascii=False
    assert b"\r\n" not in (data_dir / "chunks.json").read_bytes()  # same bytes on every OS


def test_unchanged_rerun_keeps_built_at_and_bytes(data_dir):
    build.main(["--offline", "--no-embed"])
    rep = report(data_dir)
    rep["built_at"] = "2000-01-01T00:00:00+00:00"
    (data_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n",
                                          encoding="utf-8", newline="\n")
    before = snapshot(data_dir)
    assert build.main(["--offline", "--no-embed"]) == 0
    assert snapshot(data_dir) == before  # a no-change refresh produces no git diff


def test_changed_page_gets_a_new_built_at(data_dir, monkeypatch):
    build.main(["--offline", "--no-embed"])
    rep = report(data_dir)
    rep["built_at"] = "2000-01-01T00:00:00+00:00"
    (data_dir / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    html = fetch_pages(PAGES, offline=True)["itc-price-list"].replace("$18995", "$19995")
    serve(monkeypatch, {"itc-price-list": html})
    assert build.main(["--no-embed"]) == 0
    assert report(data_dir)["built_at"] != "2000-01-01T00:00:00+00:00"


def test_empty_page_is_rejected_and_nothing_is_written(data_dir, monkeypatch, capsys):
    serve(monkeypatch, {"adrenal-package": "<html><body>Just a moment...</body></html>"})
    assert build.main(["--no-embed"]) == 1
    out = capsys.readouterr().out
    assert "adrenal-package: expected phrase 'Adrenal' not found" in out
    assert "adrenal-package: only 0 characters extracted" in out
    assert snapshot(data_dir) == {}


def test_shrunk_page_is_rejected_and_last_good_build_stays(data_dir, capsys):
    build.main(["--offline", "--no-embed"])
    rep = report(data_dir)
    rep["pages"]["itc-videos"]["chars"] *= 3  # as if the page used to be three times bigger
    rep["pages"]["itc-videos"]["table_rows"] = 4  # and used to have a table
    (data_dir / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    before = snapshot(data_dir)
    assert build.main(["--offline", "--no-embed"]) == 1
    out = capsys.readouterr().out
    assert "itc-videos: shrank from" in out and "itc-videos: had 4 table rows, now none" in out
    assert snapshot(data_dir) == before


def test_embeddings_are_reused_for_unchanged_chunks(data_dir, monkeypatch, fake_llm):
    assert build.main(["--offline"]) == 0
    total = report(data_dir)["total_chunks"]
    vectors = np.load(data_dir / "embeddings.npy")
    assert vectors.shape == (total, 8) and vectors.dtype == np.float32
    assert fake_llm == [total]

    html = fetch_pages(PAGES, offline=True)["itc-price-list"].replace("$18995", "$19995")
    serve(monkeypatch, {"itc-price-list": html})
    assert build.main([]) == 0
    assert 0 < fake_llm[1] < total  # only the changed chunks were embedded again
    assert report(data_dir)["embedded"] is True

    monkeypatch.setattr(config, "EMBED_MODEL", "another-model")
    assert build.main(["--offline"]) == 0
    assert fake_llm[2] == report(data_dir)["total_chunks"]  # another model's vectors never mix in


def test_no_embed_removes_stale_vectors(data_dir, fake_llm):
    build.main(["--offline"])
    assert (data_dir / "embeddings.npy").exists()
    build.main(["--offline", "--no-embed"])
    assert not (data_dir / "embeddings.npy").exists()
    assert report(data_dir)["embedded"] is False


def test_embedding_failure_writes_nothing(data_dir, monkeypatch, capsys):
    def broken(texts):
        raise llm.LLMError("provider down")
    monkeypatch.setattr(llm, "embed", broken)
    assert build.main(["--offline"]) == 1
    assert "Embedding failed" in capsys.readouterr().out
    assert snapshot(data_dir) == {}


def test_only_reviews_and_sidebar_rows_repeated_from_an_earlier_page_are_dropped():
    pages = [{"slug": "a"}, {"slug": "b"}]
    blocks = {"a": [Block("Reviews > Review by Jana C", "text", "Great  stay."),
                    Block("Clinic details: address, location, specialities, packages, doctors", "list", "- Location: Tijuana"),
                    Block("Cost", "table_row", "Total: $18,995")],
              "b": [Block("Recent Reviews > Review by Jana C", "text", "great stay."),  # same one
                    Block("Clinic details: address, location, specialities, packages, doctors", "list", "- Location: Tijuana"),  # same sidebar row
                    Block("Cost", "table_row", "Total: $18,995"),  # this package's own price
                    Block("Reviews > Review by Chris P", "text", "Great stay.")]}  # another review
    kept = build.drop_repeats(pages, blocks)
    assert kept["a"] == blocks["a"]  # the first page keeps everything
    assert kept["b"] == blocks["b"][2:]


def test_build_keeps_one_copy_of_each_review_and_every_package_price(data_dir):
    assert build.main(["--offline", "--no-embed"]) == 0
    chunks = json.loads((data_dir / "chunks.json").read_text(encoding="utf-8"))
    reviews = {c["id"].split("#")[0] for c in chunks if "Review by" in c["section"]}
    assert reviews == {"itc-profile"}  # pages 5 and 6 repeated the profile's reviews
    for slug in ("adenocarcinoma-package", "adrenal-package", "anal-package"):
        own = [c for c in chunks if c["id"].startswith(f"{slug}#")]
        assert sum(c["kind"] == "table_row" for c in own) == 2, slug
        assert any("Optional Specialized Procedures" in c["section"] for c in own), slug
    details: dict[str, str] = {}  # sidebar lines are one chunk each, so join them per page
    for c in chunks:
        if c["section"].startswith("Clinic details: address, location, specialities, packages, doctors"):
            slug = c["id"].split("#")[0]
            details[slug] = details.get(slug, "") + c["text"]
    assert "Padre Kino" in details["itc-profile"] and "Appendix" in details["itc-profile"]
    assert "Price Range: $18995 - $18995" in details["itc-videos"]  # first said on page 5
    assert "itc-price-list" not in details  # every sidebar row of page 6 is on page 1 or 5

    # The report keeps each page's size as extracted, before repeats are dropped: that is the
    # number the next build's shrink gate compares against.
    page = next(p for p in PAGES if p["slug"] == "itc-price-list")
    sources = load_sources()
    raw = build.extract(fetch_pages([page], offline=True)["itc-price-list"],
                        sources["content_selector"], sources["drop_selectors"])
    assert report(data_dir)["pages"]["itc-price-list"]["chars"] == sum(len(b.text) for b in raw)
