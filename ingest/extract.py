"""Turn one PlacidWay page's HTML into clean text blocks with their heading path.

Called by ingest/build.py. Reads the content area named in sources.yaml, minus its
drop_selectors (the one boilerplate list), plus two small parts beside it on purpose:
the header's price badge and review count, and the "About ..." sidebar.
"""

import re
from dataclasses import dataclass
from typing import Literal

from bs4 import BeautifulSoup, Comment, Tag

# The "Price starting from" badge sits in the page header, above the content area.
HEADLINE_PRICE = "div.price-social-block-ctm div.actual-price"
HEADLINE_SECTION = "Headline price"
# The "About Medical Center" / "About Package" sidebar is a column beside the content area and
# the only place that states the street address, specialities and the clinic's full package
# list (including Appendix). Pages 1-6 have one; page 7 does not.
SIDEBAR_ROWS = "div.pack-about-aside ul.pack-about-items > li"
SIDEBAR_SECTION = "Clinic details: address, location, specialities, packages, doctors"  # named the way people ask, so it is found
# Rows kept, by label. A whitelist, not a drop list: a row the site adds later could name other
# clinics, so it stays out until someone has read it. Left out today: "Translations" (language
# flags), "Medical Center Videos" / "Medical Center Prices" (links to pages 5 and 6), and
# "Overview" (a marketing line that buried the address when people asked where ITC is).
SIDEBAR_FIELDS = {"Price Range", "Speciality", "Specialty", "Treatment", "Location", "Packages",
                  "Top Packages", "Associated Center", "Associated Doctors", "Focus Area"}
# "6 Reviews" in the header, next to the stars (the 4.3 rating is only an attribute, not text).
PAGE_HEADER = "div.pack-title-price-header"
REVIEW_COUNT = re.compile(r"\s*(\d+) Reviews?\s*")
# Patient reviews: one card per review. Its reviewer and date become a heading, so every
# chunk of the review says who wrote it. Level 6 nests it under the page's "Reviews" heading.
REVIEW_CARD = "div.review-card"
REVIEW_HEADING = "Review by "
REVIEW_LEVEL = 6
# Page 5's grid of 128 videos has no heading of its own (the page's h1 says "Videos").
VIDEO_GRID = "section.related-videos"
VIDEO_TITLE = "h3.title"
VIDEO_SECTION = "Videos"
VIDEO_GROUP = 25  # titles per list block; small enough to read, the chunker splits further
# Page 7's disclaimer has no heading either; without one it would land under the FAQ heading.
DISCLAIMER = "section#disclaimer"
DISCLAIMER_SECTION = "PlacidWay disclaimer"
SKIP_COLUMNS = {"action", "actions"}  # button columns in price tables
FAQ_MARKERS = ("faq", "frequently asked")  # an h2 with these words turns "...?" headings into FAQs
DEFAULT_SECTION = "Overview"  # for text that appears before any heading
# Decorative badges like "01" or "MEDICAL DIRECTOR" sit above the heading they belong to,
# so kept as text they would be glued onto the end of the previous section.
BADGE = re.compile(r"[A-Z0-9 &'-]{1,40}")

HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
BLOCK_TAGS = {"p", "div", "section", "article", "header", "aside", "blockquote", "figure",
              "ul", "ol", "dl", "li", "table", "details", "hr", *HEADINGS}


@dataclass
class Block:
    section: str  # heading path, e.g. "Cost of ... > Out-Patient Program"
    kind: Literal["text", "table_row", "faq", "list"]
    text: str


def extract(html: str, selector: str, drop_selectors: list[str]) -> list[Block]:
    """Blocks of the page in reading order; [] when the content area is missing."""
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one(selector)
    if root is None:
        return []
    for css in drop_selectors:
        for element in root.select(css):
            element.decompose()
    items = _headline_price(soup) + _walk(root) + _sidebar(soup)
    blocks, seen = [], set()
    for block in _assemble(items):
        if block.text not in seen:  # the same paragraph twice on a page adds nothing
            seen.add(block.text)
            blocks.append(block)
    return blocks


def to_markdown(page: dict, blocks: list[Block]) -> str:
    """Human-readable copy of what was extracted (data/pages/<slug>.md)."""
    lines = [f"# {page['label']}", "", f"Source: {page['url']}"]
    section = None
    for block in blocks:
        if block.section != section:
            lines += ["", f"## {block.section}"]
            section = block.section
        lines += ["", f"- {block.text}" if block.kind == "table_row" else block.text]
    return "\n".join(lines) + "\n"


def _clean(text: str) -> str:
    """Collapse whitespace. get_text(" ") also leaves a space before punctuation after a tag."""
    return re.sub(r" ([,.;:!?)])", r"\1", " ".join(text.split()))


def _walk(element: Tag) -> list[tuple]:
    """Items (kind, level, text) for the element's children in page order; level is only used
    by headings. A run of inline content (text, links, bold...) becomes one text item."""
    items, run = [], []
    for child in element.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, Tag) and child.name in BLOCK_TAGS:
            items += _inline_text(run) + _block(child)
            run = []
        else:
            run.append(child)
    return items + _inline_text(run)


def _inline_text(run: list) -> list[tuple]:
    text = _clean(" ".join(n.get_text(" ") if isinstance(n, Tag) else str(n) for n in run))
    # A run that is nothing but links is a button ("Contact PlacidWay for a Free Quote").
    links_only = all(n.name == "a" if isinstance(n, Tag) else not n.strip() for n in run)
    if not text or links_only or BADGE.fullmatch(text):
        return []
    return [("text", 0, text)]


def _block(tag: Tag) -> list[tuple]:
    if tag.name in HEADINGS:
        text = _clean(tag.get_text(" "))
        return [("heading", HEADINGS[tag.name], text)] if text else []
    if tag.name in ("ul", "ol"):
        lines = [_clean(li.get_text(" ")) for li in tag.find_all("li", recursive=False)]
        lines = [f"- {line}" for line in lines if line]
        return [("list", 0, "\n".join(lines))] if lines else []
    if tag.name == "table":
        return _table(tag)
    if tag.name == "dl":
        # Page 7 "Quick Facts": one record of label/value pairs, so one row (dt and dd alternate).
        pairs = zip(tag.find_all("dt"), tag.find_all("dd"))
        parts = [f"{_clean(dt.get_text(' '))}: {_clean(dd.get_text(' '))}" for dt, dd in pairs]
        return [("table_row", 0, " | ".join(parts))] if parts else []
    if tag.name == "details":
        return _details(tag)
    if tag.css.match(REVIEW_CARD):
        return _review(tag)
    if tag.css.match(VIDEO_GRID):
        return _videos(tag)
    if tag.css.match(DISCLAIMER):
        return [("heading", 2, DISCLAIMER_SECTION)] + _walk(tag)
    return _walk(tag)


def _table(table: Tag) -> list[tuple]:
    """One item per data row: "Header1: value1 | Header2: value2"."""
    headers, items = [], []
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if cells and all(cell.name == "th" for cell in cells):
            headers = [_cell_text(cell) for cell in cells]
            continue
        parts = []
        for i, cell in enumerate(cells):
            header = headers[i] if i < len(headers) else ""
            value = _cell_text(cell)
            if value and header.lower() not in SKIP_COLUMNS:
                parts.append(f"{header}: {value}" if header else value)
        if parts:
            items.append(("table_row", 0, " | ".join(parts)))
    return items


def _cell_text(cell: Tag) -> str:
    # Cells on these pages hold either plain text or one list (page 1: the 12 cancer types).
    entries = [_clean(li.get_text(" ")) for li in cell.find_all("li")]
    return "; ".join(e for e in entries if e) if entries else _clean(cell.get_text(" "))


def _details(details: Tag) -> list[tuple]:
    """Page 7 FAQs are <details><summary>question</summary>answer</details>."""
    summary = details.find("summary")
    if summary is None:
        return _walk(details)
    question = _clean(summary.get_text(" "))
    summary.extract()
    answer = _clean(details.get_text(" "))
    return [("faq", 0, f"Q: {question}\nA: {answer}")] if question and answer else []


def _review(card: Tag) -> list[tuple]:
    body = card.select_one(".review-body")
    if body is None:
        return []
    who = [_clean(el.get_text(" ")) for el in card.select(".reviewer-name, .review-date")]
    heading = REVIEW_HEADING + (", ".join(w for w in who if w) or "a patient")
    return [("heading", REVIEW_LEVEL, heading), ("text", 0, _clean(body.get_text(" ")))]


def _videos(grid: Tag) -> list[tuple]:
    titles = [t for t in (_clean(h.get_text(" ")) for h in grid.select(VIDEO_TITLE)) if t]
    items = [("heading", 2, VIDEO_SECTION)] if titles else []
    for start in range(0, len(titles), VIDEO_GROUP):
        group = titles[start:start + VIDEO_GROUP]
        items.append(("list", 0, "\n".join(f"- {title}" for title in group)))
    return items


def _headline_price(soup: BeautifulSoup) -> list[tuple]:
    badge = soup.select_one(HEADLINE_PRICE)
    text = _clean(badge.get_text(" ")) if badge else ""
    return [("heading", 2, HEADLINE_SECTION), ("text", 0, text)] if text else []


def _sidebar(soup: BeautifulSoup) -> list[tuple]:
    """The sidebar's kept rows as "- Label: value" lines, plus the header's review count."""
    lines = []
    for row in soup.select(SIDEBAR_ROWS):
        label = row.find("span")
        name = _clean(label.extract().get_text(" ")).rstrip(":") if label else ""
        if name not in SIDEBAR_FIELDS:
            continue
        links = [_clean(a.get_text(" ")) for a in row.find_all("a")]  # packages, doctors
        value = "; ".join(links) if links else _clean(row.get_text(" "))
        if value:
            lines.append(f"- {name}: {value}")
    if not lines:
        return []  # page 7: no sidebar, and its review count is for testimonials we drop
    header = soup.select_one(PAGE_HEADER)
    count = header.find(string=REVIEW_COUNT.fullmatch) if header else None
    if count:
        lines.append(f"- Patient reviews: {REVIEW_COUNT.fullmatch(count).group(1)}")
    # Level 1 and placed last, so no page heading can nest it and no page text falls under it.
    # One block per row, so build.py can drop a row that an earlier page already gave.
    return [("heading", 1, SIDEBAR_SECTION)] + [("list", 0, line) for line in lines]


def _assemble(items: list[tuple]) -> list[Block]:
    """Give each item its heading path; an FAQ question heading plus its answer is one faq block."""
    blocks, path = [], []  # path: [(level, heading text), ...]
    question, answer = "", []
    for kind, level, text in items:
        if question and kind in ("text", "list"):
            answer.append(text)
            continue
        if question and answer:
            blocks.append(Block(_section(path), "faq", f"Q: {question}\nA: " + "\n".join(answer)))
        question, answer = "", []
        if kind != "heading":
            blocks.append(Block(_section(path), kind, text))
            continue
        path = [h for h in path if h[0] < level]
        in_faq = any(marker in _section(path).lower() for marker in FAQ_MARKERS)
        if in_faq and text.endswith("?"):
            question = text  # the question goes into the block, not into the section path
        else:
            path.append((level, text))
    if question and answer:
        blocks.append(Block(_section(path), "faq", f"Q: {question}\nA: " + "\n".join(answer)))
    return blocks


def _section(path: list[tuple[int, str]]) -> str:
    return " > ".join(text for _, text in path) or DEFAULT_SECTION
