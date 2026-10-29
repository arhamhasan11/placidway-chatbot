"""Extraction against the 7 saved pages: the facts survive, the boilerplate does not."""

import pytest

from ingest.extract import SIDEBAR_FIELDS, Block, extract, to_markdown
from ingest.fetch import FIXTURES_DIR, load_sources

SOURCES = load_sources()
PAGES = {page["slug"]: page for page in SOURCES["pages"]}
PACKAGES = ["adenocarcinoma-package", "adrenal-package", "anal-package"]

# Strings from widgets, other clinics and platform banners that must never reach the bot.
FORBIDDEN = [
    "2024-11-03", "15:30:00", "No slots available", "Request a video call", "Chat with",
    "Related Experiences", "Mommy Makeover", "Gastric Bypass", "Duodenal Switch", "Ibogaine",
    "Eyelid Surgery", "JCI", "TURKEY MOH", "NABH", "DHCC", "KOREA MOHW", "60+", "2k+",
    "$24,000", "$33,400", "$4,500", "Eliava", "$7,850", "$6,500", "$6,949", "Verita",
    "Limphocare", "EBOO", "aryngeal", "hyroid", "Gangnam", "KFSHRC", "Saudi", "Riyadh",
    "New Path", "Hair in 1 Day", "center_data", "@context", "schema.org", "Write a review",
    "Read More", "Load More", "Free Quote",
]


@pytest.fixture(scope="module")
def blocks() -> dict[str, list[Block]]:
    return {slug: extract((FIXTURES_DIR / page["fixture"]).read_text(encoding="utf-8"),
                          SOURCES["content_selector"], SOURCES["drop_selectors"])
            for slug, page in PAGES.items()}


def text_of(found: list[Block]) -> str:
    return "\n".join(f"{b.section}\n{b.text}" for b in found)


def rows(found: list[Block]) -> list[str]:
    return [b.text for b in found if b.kind == "table_row"]


@pytest.mark.parametrize("slug", PAGES)
def test_no_boilerplate_survives(blocks, slug):
    text = text_of(blocks[slug])
    assert [s for s in FORBIDDEN if s in text] == []


@pytest.mark.parametrize("slug", PAGES)
def test_blocks_are_clean(blocks, slug):
    found = blocks[slug]
    assert found and all(b.text.strip() and b.section for b in found)
    assert len({b.text for b in found}) == len(found)  # no duplicates within a page
    assert PAGES[slug]["expect"] in text_of(found)


@pytest.mark.parametrize("slug", PACKAGES)
def test_package_pages_keep_prices_lists_and_faqs(blocks, slug):
    found, text = blocks[slug], text_of(blocks[slug])
    assert rows(found) == [
        "Program Type: Out-Patient Program | Duration: 3 Weeks | Total Cost (USD): $18,995",
        "Program Type: In-Patient Program | Duration: 3 Weeks | Total Cost (USD): $30,000",
    ]
    assert "$18,995 Price starting from" in text
    for fact in ["Included in the Out-Patient Program ($18,995 USD)",
                 "Included in the In-Patient Program ($30,000 USD)",
                 "- Lymphocytes Activated Killer Cells (LAK): $2,600 USD",
                 "- Specific Killer Cell Antibody: $3,950 - $4,250 USD",
                 "- Ethanol Tumor Ablation: $3,800 - $4,250 USD",
                 "- Mononuclear Cell Activation (MCA): $4,650 USD",
                 "- Intratumoral Immunotherapy: $6,200 - $7,200 USD",
                 "$200 per visit", "$462 per pint", "$950", "$300 per area",
                 "3-Week Typical Itinerary at ITC Tijuana", "Week 1", "Week 3"]:
        assert fact in text, fact
    faqs = [b for b in found if b.kind == "faq"]
    assert len(faqs) == 8
    assert all(b.text.startswith("Q: ") and "\nA: " in b.text for b in faqs)
    assert all("FAQ" in b.section for b in faqs)


def test_profile_page(blocks):
    found, text = blocks["itc-profile"], text_of(blocks["itc-profile"])
    [row] = rows(found)
    assert row.startswith("Base Package Price: $18,995 | Cancer Types Treated at ITC: Breast")
    assert row.split(" | ")[1].count(";") == 11  # 12 cancer types
    assert "Standard Inclusions: Physician consultations & daily care;" in row
    faqs = [b.text for b in found if b.kind == "faq"]
    assert len(faqs) == 7
    assert any("Treatment costs start at $18,995" in f for f in faqs)
    for fact in ["Risks and Important Medical Considerations", "Dr. Carlos Bautista",
                 "Dr. David Alvarez", "Dr. Adolfo Carrillo", "Dr. Pablo Orozco"]:
        assert fact in text, fact
    assert "MEDICAL DIRECTOR\n" not in text and "\n01\n" not in text  # decorative badges


def test_videos_page_lists_all_titles_in_groups(blocks):
    lists = [b for b in blocks["itc-videos"] if b.kind == "list" and b.section == "Videos"]
    titles = [line for b in lists for line in b.text.split("\n")]
    assert len(titles) == 128
    assert all(t.startswith("- ") for t in titles)
    assert max(b.text.count("\n") + 1 for b in lists) == 25


def test_price_list_rows(blocks):
    found = rows(blocks["itc-price-list"])
    assert len(found) == 7
    assert "Procedure: Breast Cancer | Details: Breast Cancer | Price in USD: $18995" in found
    assert "Price in USD: $18995" in found  # the row with empty procedure and details
    assert not any("Actions" in r or "Chat" in r for r in found)


def test_comparison_page(blocks):
    found, text = blocks["alt-medicine-prices"], text_of(blocks["alt-medicine-prices"])
    for row in [
        "Procedure: Acupuncture Therapy Package | Low: $200 | High: $800",
        "Procedure: Ayurvedic Panchakarma (14 Days) | Low: $800 | High: $2,500",
        "Procedure: Ozone Therapy Package | Low: $500 | High: $1,500",
        "Procedure: Alternative Stem Cell Therapy | Low: $3,000 | High: $10,000",
        "Procedure: Naturopathic Wellness Retreat | Low: $1,200 | High: $4,000",
        "Component: Initial Holistic Consultation | Low (USD): $50 | High (USD): $250",
        "Clinic / Surgeon: Form Recovery & Wellness in Bangkok Thailand Bangkok, Thailand | "
        "Standard: $64",
        "Clinic / Surgeon: Fizikon Medical Center Konya, Turkey | Standard: $110",
        "Clinic / Surgeon: Stem Health Guadalajara Zapopan, Mexico | Standard: $1000",
        "Standard: $5500",
    ]:
        assert any(r.startswith(row) or row in r for r in rows(found)), row
    assert "Ozone Therapy" in [r for r in rows(found) if "Fizikon" in r][0]
    assert not any("Action" in r or "Quote" in r for r in rows(found))
    assert len([b for b in found if b.kind == "faq"]) == 12
    for fact in ["- Initial holistic health assessment", "- International flights and transfers",
                 "PlacidWay disclaimer", "Prices on our site are estimates only"]:
        assert fact in text, fact
    assert "Mustafa" not in text  # page 7 testimonials name no clinic, so they are dropped


def details(found: list[Block]) -> list[str]:
    return [b.text for b in found if b.section == "Clinic details: address, location, specialities, packages, doctors"]


def test_sidebar_gives_the_profile_address_packages_and_doctor(blocks):
    lines = details(blocks["itc-profile"])
    assert lines == [
        "- Speciality: Alternative Medicine, Detox, Cancer Treatment, Chronic Diseases",
        "- Location: Blvd. Padre Kino 10133, Zona Urbana Rio Tijuana, 22010 Tijuana, B.C., Mexico, "
        "Tijuana 22010, Mexico",
        "- Packages: " + "; ".join(
            f"Alternative {kind} Cancer Treatment Package in Tijuana, Mexico by ITC"
            for kind in ["Adenocarcinoma", "Adrenal", "Anal", "Appendix"]),
        "- Associated Doctors: Dr. Carlos Bautista",
        "- Patient reviews: 6",
    ]


@pytest.mark.parametrize("slug", PAGES)
def test_sidebar_holds_only_itc_facts(blocks, slug):
    rows = dict(line[2:].split(": ", 1) for line in details(blocks[slug]))
    if slug == "alt-medicine-prices":  # no sidebar; its "10 Reviews" are dropped testimonials
        assert rows == {}
        return
    assert "Blvd. Padre Kino 10133" in rows["Location"]
    assert set(rows) <= SIDEBAR_FIELDS | {"Patient reviews"}  # no flags, video or price-page links
    listed = [p for k in ("Packages", "Top Packages") if k in rows for p in rows[k].split("; ")]
    assert all(p.endswith("Package in Tijuana, Mexico by ITC") for p in listed)
    assert rows.get("Associated Doctors", "Dr. Carlos Bautista") == "Dr. Carlos Bautista"
    assert rows.get("Associated Center", "ITC Immunity Therapy Center").startswith("ITC Immunity")
    assert not any(button in str(rows) for button in ["Chat with", "Book Consultation"])


def test_reviews_are_kept_only_under_a_reviews_heading(blocks):
    reviews = [b for found in blocks.values() for b in found if "Review by" in b.section]
    assert len(reviews) == 9  # 3 reviews on pages 1, 5 and 6
    assert all("Reviews" in b.section.split(" > Review by")[0] for b in reviews)
    assert any(b.section.endswith("Review by Guest P, Oct 24 2026") for b in reviews)


# --- Small hand-written pages for the individual rules --------------------------------------

SELECTOR = "main"


def run(body: str, drop: list[str] | None = None) -> list[Block]:
    return extract(f"<html><body><main>{body}</main></body></html>", SELECTOR, drop or [])


def test_missing_content_area_gives_nothing():
    assert extract("<html><body><p>Just a moment...</p></body></html>", SELECTOR, []) == []


def test_table_rows_skip_empty_cells_and_action_columns():
    found = run("<h2>Prices</h2><table><tr><th>Item</th><th>Note</th><th>Actions</th></tr>"
                "<tr><td>A</td><td></td><td><a>Book</a></td></tr></table>")
    assert found == [Block("Prices", "table_row", "Item: A")]


def test_faq_headings_become_question_and_answer():
    found = run("<h2>FAQs</h2><h3>Is it safe?</h3><p>Ask your doctor.</p><p>Always.</p>"
                "<h2>Next</h2><p>Other text.</p>")
    assert found == [Block("FAQs", "faq", "Q: Is it safe?\nA: Ask your doctor.\nAlways."),
                     Block("Next", "text", "Other text.")]


def test_details_summary_faq_and_lists():
    found = run("<h2>Help</h2><details><summary>Cost?</summary><p>About $50.</p></details>"
                "<ul><li>one</li><li> two </li></ul>")
    assert [(b.kind, b.text) for b in found] == [("faq", "Q: Cost?\nA: About $50."),
                                                 ("list", "- one\n- two")]


def test_heading_path_badges_links_drops_and_duplicates():
    found = run("<h2>A</h2><h3>B</h3><div>MEDICAL DIRECTOR</div><p>Text <b>bold</b> , here.</p>"
                "<div><a href='#'>Contact us</a></div><p>Text bold, here.</p>"
                "<div class='junk'>gone</div><h2>C</h2><p>Last.</p>", drop=["div.junk"])
    assert found == [Block("A > B", "text", "Text bold, here."), Block("C", "text", "Last.")]


def test_sidebar_keeps_listed_rows_after_the_content_with_the_review_count():
    html = ("<html><body><div class='pack-title-price-header'><a><span> 6 Reviews </span></a></div>"
            "<main><h1>Page</h1><p>Text.</p></main>"
            "<div class='pack-about-aside'><ul class='pack-about-items'>"
            "<li><span>Translations:</span><span><a>EN</a></span></li>"
            "<li><span>Location:</span> Blvd. Padre Kino 10133 , Tijuana </li>"
            "<li><span>Packages:</span><span><a>One</a>,</span><span><a>Two</a>,</span></li>"
            "<li><span>Similar Centers:</span><span><a>Other Clinic</a></span></li>"
            "</ul><button>Chat with Center</button></div></body></html>")
    assert extract(html, SELECTOR, []) == [
        Block("Page", "text", "Text."),
        Block("Clinic details: address, location, specialities, packages, doctors", "list", "- Location: Blvd. Padre Kino 10133, Tijuana"),
        Block("Clinic details: address, location, specialities, packages, doctors", "list", "- Packages: One; Two"),
        Block("Clinic details: address, location, specialities, packages, doctors", "list", "- Patient reviews: 6")]


def test_review_count_without_a_sidebar_is_ignored():
    html = ("<html><body><div class='pack-title-price-header'><span>10 Reviews</span></div>"
            "<main><p>Text.</p></main></body></html>")
    assert extract(html, SELECTOR, []) == [Block("Overview", "text", "Text.")]


def test_markdown_names_the_page_and_sections():
    page = {"label": "Label", "url": "https://example.com/x"}
    md = to_markdown(page, [Block("S", "text", "Hello"), Block("S", "table_row", "K: v")])
    assert md == "# Label\n\nSource: https://example.com/x\n\n## S\n\nHello\n\n- K: v\n"
