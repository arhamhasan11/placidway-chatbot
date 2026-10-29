"""Rebuild the knowledge base: fetch -> extract -> gates -> drop repeats -> chunk -> embed -> write.

Run by hand or by the daily GitHub Action: python -m ingest.build [--offline] [--no-embed]
If any gate fails, nothing is written and the bot keeps answering from the last good build.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config
from bot import llm
from bot.schemas import Chunk
from ingest.chunk import make_chunks
from ingest.extract import REVIEW_HEADING, SIDEBAR_SECTION, Block, extract, to_markdown
from ingest.fetch import FetchError, fetch_pages, load_sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ingest.build", description=__doc__)
    parser.add_argument("--offline", action="store_true", help="read tests/fixtures, no network")
    parser.add_argument("--no-embed", action="store_true", help="skip embeddings (no API key)")
    args = parser.parse_args(argv)

    sources = load_sources()
    pages = sources["pages"]
    try:
        html = fetch_pages(pages, offline=args.offline)
    except FetchError as exc:
        print(f"Fetch failed, nothing written: {exc}")
        return 1
    blocks = {p["slug"]: extract(html[p["slug"]], sources["content_selector"],
                                 sources["drop_selectors"]) for p in pages}
    previous = _read_json(config.REPORT_FILE)
    problems = check_gates(pages, blocks, previous)
    if problems:
        print("Quality gates failed, nothing written (the live data stays on the last good build):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    # After the gates, so each page is judged on its own and never fails because of another.
    kept = drop_repeats(pages, blocks)
    page_chunks = {p["slug"]: make_chunks(p, kept[p["slug"]]) for p in pages}
    chunks = [chunk for slug in page_chunks for chunk in page_chunks[slug]]
    vectors = None
    if not args.no_embed:
        try:
            vectors = embed_chunks(chunks, previous)
        except llm.LLMError as exc:
            print(f"Embedding failed, nothing written: {exc}")
            return 1
    markdown = {p["slug"]: to_markdown(p, kept[p["slug"]]) for p in pages}
    # Report chars and rows from `blocks`, not `kept`: they are what the next build's gates compare.
    report = make_report(pages, blocks, page_chunks, markdown, previous, vectors is not None)
    write_outputs(markdown, chunks, vectors, report)
    print_summary(report, previous)
    return 0


def page_text(blocks: list[Block]) -> str:
    """What the gates read: headings and text only, never the label we add ourselves."""
    return "\n".join(f"{b.section}\n{b.text}" for b in blocks)


def check_gates(pages: list[dict], blocks: dict[str, list[Block]], previous: dict) -> list[str]:
    """Every reason to reject this build; [] means publish. No LLM involved."""
    problems = []
    for page in pages:
        slug, found = page["slug"], blocks[page["slug"]]
        chars = sum(len(b.text) for b in found)
        rows = sum(b.kind == "table_row" for b in found)
        old = previous.get("pages", {}).get(slug, {})
        if page["expect"] not in page_text(found):
            problems.append(f"{slug}: expected phrase {page['expect']!r} not found "
                            "(error page, bot challenge or redesign?)")
        if chars < config.MIN_PAGE_CHARS:
            problems.append(f"{slug}: only {chars} characters extracted "
                            f"(minimum {config.MIN_PAGE_CHARS})")
        if old and chars < old["chars"] * (1 - config.MAX_SHRINK):
            problems.append(f"{slug}: shrank from {old['chars']} to {chars} characters")
        if old.get("table_rows") and not rows:
            problems.append(f"{slug}: had {old['table_rows']} table rows, now none")
    return problems


def drop_repeats(pages: list[dict], blocks: dict[str, list[Block]]) -> dict[str, list[Block]]:
    """Each page's blocks minus the reviews and sidebar rows an earlier page already gave."""
    # The clinic's reviews and sidebar facts are copied onto several pages (sources.yaml order
    # decides which copy stays): three copies of one review would take three TOP_K slots. Only
    # these are compared. The package pages share text too (price rows, "Optional Specialized
    # Procedures"), but there each copy is that package's own fact, cited from its own page.
    seen, kept = set(), {}
    for page in pages:
        kept[page["slug"]] = []
        for block in blocks[page["slug"]]:
            heading = block.section.split(" > ")[-1]  # "Review by Jana C, ..." or "Clinic details"
            shared = heading == SIDEBAR_SECTION or heading.startswith(REVIEW_HEADING)
            key = (heading, " ".join(block.text.lower().split()))
            if shared and key in seen:
                continue
            seen.add(key)
            kept[page["slug"]].append(block)
    return kept


def embed_chunks(chunks: list[Chunk], previous: dict) -> np.ndarray:
    """One L2-normalised row per chunk. Unchanged chunks reuse last build's vectors."""
    known = _previous_vectors(previous)
    new = [c for c in chunks if c.hash not in known]
    if new:
        known.update(zip((c.hash for c in new), llm.embed([c.text for c in new])))
    print(f"Embedded {len(new)} new chunks, reused {len(chunks) - len(new)}")
    return np.stack([known[c.hash] for c in chunks]).astype(np.float32)


def _previous_vectors(previous: dict) -> dict[str, np.ndarray]:
    """hash -> vector from the last build; empty if it used another model or has no vectors."""
    if previous.get("embed_model") != config.EMBED_MODEL or not previous.get("embedded"):
        return {}
    if not (config.CHUNKS_FILE.exists() and config.EMBEDDINGS_FILE.exists()):
        return {}
    old_chunks = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))
    matrix = np.load(config.EMBEDDINGS_FILE)
    if len(old_chunks) != len(matrix):
        return {}  # the two files disagree, so trust neither
    return {c["hash"]: row for c, row in zip(old_chunks, matrix)}


def make_report(pages: list[dict], blocks: dict[str, list[Block]],
                page_chunks: dict[str, list[Chunk]], markdown: dict[str, str],
                previous: dict, embedded: bool) -> dict:
    stats = {}
    for page in pages:
        slug = page["slug"]
        stats[slug] = {
            "url": page["url"],
            "label": page["label"],
            "chars": sum(len(b.text) for b in blocks[slug]),
            "chunks": len(page_chunks[slug]),
            "table_rows": sum(b.kind == "table_row" for b in blocks[slug]),
            "sha": hashlib.sha256(markdown[slug].encode("utf-8")).hexdigest()[:16],
        }
    # Same content as last time -> same timestamp, so a daily refresh with no changes
    # produces no git diff and no redeploy.
    old_shas = {slug: p.get("sha") for slug, p in previous.get("pages", {}).items()}
    unchanged = "built_at" in previous and old_shas == {s: p["sha"] for s, p in stats.items()}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {"built_at": previous["built_at"] if unchanged else now,
            "embed_model": config.EMBED_MODEL, "embedded": embedded,
            "total_chunks": sum(s["chunks"] for s in stats.values()), "pages": stats}


def write_outputs(markdown: dict[str, str], chunks: list[Chunk], vectors: np.ndarray | None,
                  report: dict) -> None:
    """Each file is written to a temp file then renamed, so a crash never leaves half a file."""
    pages_dir = config.DATA_DIR / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for slug, text in markdown.items():
        _write_text(pages_dir / f"{slug}.md", text)
    for old in pages_dir.glob("*.md"):
        if old.stem not in markdown:  # page removed from sources.yaml
            old.unlink()
    if vectors is None:
        config.EMBEDDINGS_FILE.unlink(missing_ok=True)  # stale vectors would not match the chunks
    else:
        tmp = config.EMBEDDINGS_FILE.with_name(config.EMBEDDINGS_FILE.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, vectors)
        os.replace(tmp, config.EMBEDDINGS_FILE)
    _write_text(config.CHUNKS_FILE, _to_json([c.model_dump() for c in chunks]))
    _write_text(config.REPORT_FILE, _to_json(report))  # last: it marks the build complete


def print_summary(report: dict, previous: dict) -> None:
    old_pages = previous.get("pages", {})
    print(f"{'page':<26}{'chars':>7}{'chunks':>8}{'rows':>6}  status")
    for slug, page in report["pages"].items():
        old_sha = old_pages.get(slug, {}).get("sha")
        status = "new" if old_sha is None else "unchanged" if old_sha == page["sha"] else "CHANGED"
        print(f"{slug:<26}{page['chars']:>7}{page['chunks']:>8}{page['table_rows']:>6}  {status}")
    print(f"{report['total_chunks']} chunks, embedded={report['embedded']}, "
          f"built_at={report['built_at']}")


def _read_json(path: Path) -> dict:
    """The previous report; {} on the first build."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _to_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1) + "\n"


def _write_text(path: Path, text: str) -> None:
    # newline="\n": the same bytes on Windows and on the Linux refresh job, so no false diffs.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main())
