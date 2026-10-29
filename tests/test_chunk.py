"""Chunking: rows and FAQs stand alone, prose merges within a section, chunks name their page."""

import hashlib

import config
from ingest.chunk import make_chunks
from ingest.extract import Block, extract
from ingest.fetch import FIXTURES_DIR, load_sources

PAGE = {"slug": "demo", "url": "https://example.com/demo", "label": "Demo page"}


def bodies(chunks) -> list[str]:
    return [c.text.split("\n", 1)[1] for c in chunks]


def test_rows_and_faqs_are_never_merged():
    blocks = [Block("Prices", "table_row", "Item: A"), Block("Prices", "table_row", "Item: B"),
              Block("Prices", "faq", "Q: x?\nA: y"), Block("Prices", "text", "tiny")]
    chunks = make_chunks(PAGE, blocks)
    assert [(c.kind, b) for c, b in zip(chunks, bodies(chunks))] == [
        ("table_row", "Item: A"), ("table_row", "Item: B"), ("table", "Full table:\nItem: A\nItem: B"),
        ("faq", "Q: x?\nA: y"), ("text", "tiny")]


def test_text_and_lists_merge_within_a_section_only():
    blocks = [Block("S", "text", "Intro."), Block("S", "list", "- a\n- b"),
              Block("T", "text", "Other section.")]
    chunks = make_chunks(PAGE, blocks)
    assert [(c.section, c.kind) for c in chunks] == [("S", "text"), ("T", "text")]
    assert bodies(chunks) == ["Intro.\n- a\n- b", "Other section."]


def test_merging_stops_at_the_size_limit(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MAX_CHARS", 20)
    chunks = make_chunks(PAGE, [Block("S", "list", "- 1234567"), Block("S", "list", "- 1234567"),
                                Block("S", "list", "- 1234567")])
    assert bodies(chunks) == ["- 1234567\n- 1234567", "- 1234567"]
    assert all(c.kind == "list" for c in chunks)


def test_long_text_splits_at_sentences_and_long_lists_at_lines(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MAX_CHARS", 31)
    paragraph = "First sentence here. Second sentence here. Third one."
    lines = "\n".join(f"- item number {i}" for i in range(4))
    chunks = make_chunks(PAGE, [Block("S", "text", paragraph), Block("L", "list", lines)])
    assert bodies(chunks) == [
        "First sentence here.", "Second sentence here.", "Third one.",
        "- item number 0\n- item number 1", "- item number 2\n- item number 3"]
    assert all(len(b) <= 31 for b in bodies(chunks))


def test_an_unsplittable_sentence_is_kept_whole(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MAX_CHARS", 10)
    chunks = make_chunks(PAGE, [Block("S", "text", "Averyveryverylongword")])
    assert bodies(chunks) == ["Averyveryverylongword"]


def test_chunk_fields():
    [chunk] = make_chunks(PAGE, [Block("A > B", "text", "Hello.")])
    assert chunk.id == "demo#1"
    assert chunk.text == "Demo page | A > B\nHello."
    assert chunk.hash == hashlib.sha256(chunk.text.encode()).hexdigest()[:16]
    assert (chunk.url, chunk.label, chunk.section) == (PAGE["url"], "Demo page", "A > B")


def test_real_package_pages_keep_rows_whole_and_labelled():
    sources = load_sources()
    for page in sources["pages"][1:4]:
        html = (FIXTURES_DIR / page["fixture"]).read_text(encoding="utf-8")
        blocks = extract(html, sources["content_selector"], sources["drop_selectors"])
        chunks = make_chunks(page, blocks)
        assert [c.id for c in chunks] == [f"{page['slug']}#{n}" for n in range(1, len(chunks) + 1)]
        assert all(c.text.startswith(f"{page['label']} | ") for c in chunks)
        row = next(c for c in chunks if "In-Patient Program | Duration" in c.text)
        assert row.kind == "table_row" and row.text.endswith("$30,000")
        all_text = "\n".join(c.text for c in chunks)
        assert all(b.text in all_text for b in blocks if b.kind != "text")  # nothing lost


def test_a_table_also_becomes_one_whole_table_chunk():
    page = {"slug": "p", "url": "https://www.placidway.com/x", "label": "Page"}
    blocks = [Block("Prices", "table_row", "Item: A | Price: $1"),
              Block("Prices", "table_row", "Item: B | Price: $2"),
              Block("Notes", "text", "Prices are estimates.")]
    chunks = make_chunks(page, blocks)
    assert [c.kind for c in chunks] == ["table_row", "table_row", "table", "text"]
    assert "Item: A | Price: $1\nItem: B | Price: $2" in chunks[2].text  # every row, in order


def test_a_single_row_table_gets_no_extra_chunk():
    page = {"slug": "p", "url": "https://www.placidway.com/x", "label": "Page"}
    chunks = make_chunks(page, [Block("Prices", "table_row", "Item: A | Price: $1")])
    assert [c.kind for c in chunks] == ["table_row"]


def test_the_starting_from_badge_joins_only_the_table_with_its_amount():
    page = {"slug": "p", "url": "https://www.placidway.com/x", "label": "Page"}
    blocks = [Block("Headline price", "text", "$18,995 Price starting from"),
              Block("Cost", "table_row", "Program: Out-Patient | Total: $18,995"),
              Block("Cost", "table_row", "Program: In-Patient | Total: $30,000"),
              Block("Extras", "table_row", "Item: LAK | Price: $2,600"),
              Block("Extras", "table_row", "Item: MCA | Price: $4,650")]
    tables = [c.text for c in make_chunks(page, blocks) if c.kind == "table"]
    assert tables[0].startswith("Page | Cost\n$18,995 Price starting from\nFull table:")
    assert "starting from" not in tables[1]  # the extras table has no $18,995


def test_badge_amount_must_match_a_whole_price():
    from ingest.chunk import _shares_amount
    assert _shares_amount("$18,995 Price starting from", "Total: $18995")
    assert not _shares_amount("$18,995 Price starting from", "Total: $189,950")
    assert not _shares_amount("$18,995 Price starting from", "Total: $8,995")


def test_a_section_with_three_or_more_parts_also_gets_one_combined_chunk():
    page = {"slug": "p", "url": "https://www.placidway.com/x", "label": "Page"}
    blocks = [Block(f"Doctors > Dr. {name}", "text", f"Dr. {name} treats patients.") for name in "ABC"]
    combined = [c for c in make_chunks(page, blocks) if c.section == "Doctors"]
    assert len(combined) == 1
    assert all(f"Dr. {name}:\nDr. {name} treats patients." in combined[0].text for name in "ABC")


def test_two_parts_get_no_combined_chunk():
    page = {"slug": "p", "url": "https://www.placidway.com/x", "label": "Page"}
    blocks = [Block(f"Doctors > Dr. {name}", "text", "Bio.") for name in "AB"]
    assert all(c.section != "Doctors" for c in make_chunks(page, blocks))
