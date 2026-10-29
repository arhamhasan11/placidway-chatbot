"""Cut one page's blocks into the chunks that retrieval searches (called by ingest/build.py)."""

import hashlib
import re

import config
from bot.schemas import Chunk
from ingest.extract import HEADLINE_SECTION, SIDEBAR_SECTION, Block

# A table row or an FAQ is one fact with its own context, so it is always its own chunk:
# an answer can then cite the exact price row instead of a whole table.
WHOLE_KINDS = ("table_row", "faq")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def make_chunks(page: dict, blocks: list[Block]) -> list[Chunk]:
    """Chunks in page order. Text and lists in the same section merge up to CHUNK_MAX_CHARS.

    No overlap between chunks: they end at headings, so repeating text would add nothing.
    """
    limit = config.CHUNK_MAX_CHARS
    parts: list[tuple[str, str, str]] = []  # (section, kind, body)
    for block in blocks:
        # Sidebar facts (address, doctors, packages) stay one line per chunk: merged, the address
        # was diluted by the package list and "where is ITC?" often missed it.
        if block.section == SIDEBAR_SECTION:  # "- Location: ..." -> section "Clinic details ... > Location"
            label = block.text.lstrip("- ").split(":", 1)[0]
            parts.append((f"{block.section} > {label}", block.kind, block.text))
            continue
        if block.kind in WHOLE_KINDS:
            parts.append((block.section, block.kind, block.text))
            continue
        for piece in _split(block.text, limit):
            if parts and _fits(parts[-1], block.section, piece, limit):
                section, kind, body = parts[-1]
                merged_kind = kind if kind == block.kind else "text"
                parts[-1] = (section, merged_kind, f"{body}\n{piece}")
            else:
                parts.append((block.section, block.kind, piece))
    headline = next((b.text for b in blocks if b.section == HEADLINE_SECTION), "")
    parts = _add_whole_tables(parts, limit, headline)
    parts = _add_section_groups(parts, limit)
    return [_chunk(page, n, *part) for n, part in enumerate(parts, start=1)]


def _add_whole_tables(parts: list[tuple[str, str, str]], limit: int,
                      headline: str = "") -> list[tuple[str, str, str]]:
    """After each run of 2+ rows from one table, add the whole table as one more chunk.

    Row chunks let an answer cite one exact price; the whole-table chunk lets "break down all
    the costs" find every row at once (as single rows they ranked anywhere from 9th to 22nd).
    The page's "Price starting from" badge joins a table that contains the badge's own amount,
    so "$18,995 ... starting from" travels with that price and with no other.
    """
    out: list[tuple[str, str, str]] = []
    run: list[str] = []
    for i, part in enumerate(parts):
        out.append(part)
        if part[1] == "table_row":
            run.append(part[2])
        nxt = parts[i + 1] if i + 1 < len(parts) else None
        run_ends = nxt is None or nxt[1] != "table_row" or nxt[0] != part[0]
        if part[1] == "table_row" and run_ends:
            whole = "\n".join(run)
            if len(run) >= 2 and len(whole) <= 2 * limit:  # a huge table stays rows-only
                badge = f"{headline}\n" if _shares_amount(headline, whole) else ""
                out.append((part[0], "table", f"{badge}Full table:\n{whole}"))
            run = []
    return out


def _add_section_groups(parts: list[tuple[str, str, str]], limit: int) -> list[tuple[str, str, str]]:
    """For a section split into 3+ sub-sections (one per doctor, one per review), add one chunk
    holding all of them, like a whole table.

    Split up, "who are its doctors?" retrieved two of the four and "what do patients say?" saw
    only some reviews. A group longer than 3 x CHUNK_MAX_CHARS is skipped: too broad to be useful.
    """
    children: dict[str, dict[str, list[str]]] = {}  # parent -> {child heading -> bodies}
    for section, _, body in parts:
        if " > " in section:
            parent, child = section.rsplit(" > ", 1)
            children.setdefault(parent, {}).setdefault(child, []).append(body)
    groups = []
    for parent, kids in children.items():
        joined = "\n\n".join(f"{child}:\n" + "\n".join(bodies) for child, bodies in kids.items())
        if len(kids) >= 3 and len(joined) <= 3 * limit:
            # The names come first: a long chunk opening with a generic line ranked 21st for "doctors".
            names = "; ".join(kids)
            groups.append((parent, "text", f"All {len(kids)} listed: {names}.\n\n{joined}"))
    return parts + groups


def _shares_amount(headline: str, table: str) -> bool:
    """True when the badge's amount ("$18,995") is one of the table's prices ("$18,995" or "$18995")."""
    return bool(_dollar_amounts(headline) & _dollar_amounts(table))


def _dollar_amounts(text: str) -> set[int]:
    """Whole "$" amounts as numbers, so $18,995 equals $18995 but never $189,950."""
    return {int(n.replace(",", "")) for n in re.findall(r"\$\s?(\d[\d,]*\d|\d)", text)}


def _fits(last: tuple[str, str, str], section: str, piece: str, limit: int) -> bool:
    section_last, kind_last, body_last = last
    return (section_last == section and kind_last not in WHOLE_KINDS
            and len(body_last) + 1 + len(piece) <= limit)


def _split(text: str, limit: int) -> list[str]:
    """Cut text longer than `limit` at line breaks (lists) or sentence ends (one long paragraph).

    A single line or sentence longer than `limit` stays whole: better a big chunk than lost words.
    """
    if len(text) <= limit:
        return [text]
    separator, units = ("\n", text.split("\n")) if "\n" in text else (" ", SENTENCE_END.split(text))
    pieces = [units[0]]
    for unit in units[1:]:
        if len(pieces[-1]) + len(separator) + len(unit) <= limit:
            pieces[-1] += separator + unit
        else:
            pieces.append(unit)
    return pieces


def _chunk(page: dict, n: int, section: str, kind: str, body: str) -> Chunk:
    # Every chunk names its page: the 3 package pages are near-identical apart from this label.
    text = f"{page['label']} | {section}\n{body}"
    return Chunk(
        id=f"{page['slug']}#{n}",
        url=page["url"],
        label=page["label"],
        section=section,
        kind=kind,
        text=text,
        hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    )
