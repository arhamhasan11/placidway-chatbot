"""bot/guard.py: the code check that stops invented or misplaced prices."""

import pytest

from bot import guard
from bot.schemas import Chunk, Draft, Hit


def hit(ref: str, text: str) -> Hit:
    chunk = Chunk(
        id=f"page#{ref}",
        url=f"https://example.com/{ref}",
        label="Page",
        section="S",
        kind="text",
        text=text,
        hash="0",
    )
    return Hit(chunk=chunk, ref=ref, cosine=0.9)


PACKAGE = hit(
    "C1", "Adenocarcinoma package | Cost | Out-Patient Program 3 Weeks $18,995 USD; In-Patient Program $30,000 USD"
)
PRICE_LIST = hit("C2", "ITC Price List | Pricings | Breast Cancer 18995 USD")
COMPARISON = hit("C3", "Compare Prices | Components | Initial Holistic Consultation $50 - $250")
EXTRAS = hit("C4", "Adenocarcinoma package | Optional | Specific Killer Cell Antibody: $3,950 - $4,250 USD")
REVIEW = hit("C5", "Clinic profile | Reviews > Review by Guest P, Oct 24 2026 | They gave me a 50% dose, not 10%.")
LISTING = hit("C6", "Compare Prices | Clinics | Fizikon Medical Center | Standard: $110 Modified date Jul 31, 2023")
FAQ = hit("C7", "Compare Prices | FAQ | A: Patients can save 40% to 70% on alternative medicine abroad.")
HITS = [PACKAGE, PRICE_LIST, COMPARISON, EXTRAS, REVIEW, LISTING, FAQ]


def draft(answer: str, citations: list[str], answerable: bool = True) -> Draft:
    return Draft(answerable=answerable, answer=answer, citations=citations)


# --- money_amounts --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "$18,995",
        "US$18995",
        "US$ 18,995",
        "U$S 18.995",
        "USD 18,995",
        "18,995 USD",
        "18995 usd",
        "18,995 dollars",
        "18.995 dólares",
        "18.995 dolares",
        "١٨٬٩٩٥ دولار",  # Arabic-Indic digits and Arabic thousands separator
        "۱۸۹۹۵ USD",  # Extended Arabic-Indic digits
        "18 995 $",  # no-break space as thousands separator, French-style trailing $
        "18 995 USD",
        "$18,995.00",
        "18.995,00 USD",
    ],
)
def test_spellings_of_the_same_price_are_equal(text):
    assert guard.money_amounts(text) == {18995}


def test_different_amounts_stay_different():
    assert guard.money_amounts("$12,500") == {12500}
    assert guard.money_amounts("$12,500") != {12000}


@pytest.mark.parametrize(
    "text",
    [
        "MXN 18,995",
        "MXN$18,995",
        "EUR 18.995",
        "€18,995",
        "18,995 US dollars",
        "18,995 U.S. dollars",
        "18.995 pesos",
        "18,995 MXN",
        "18.995 EUR",
        "18.995 euros",
        "18.995 €",
        "١٨٬٩٩٥ بيزو",
    ],
)
def test_pesos_and_euros_are_money_too(text):
    # Recognised so that a converted amount is checked like any price, and rejected: no chunk states one.
    assert guard.money_amounts(text) == {18995}


@pytest.mark.parametrize(
    "text",
    [
        "$3,950 - $4,250",
        "$3,950 – $4,250 USD",
        "3,950 - 4,250 USD",
        "3,950 to 4,250 USD",
        "3.950 a 4.250 dólares",
        "between 3,950 and 4,250 USD",
        "entre 3.950 y 4.250 dólares",
        "بين ٣٬٩٥٠ و٤٬٢٥٠ دولار",
    ],
)
def test_ranges_yield_both_ends(text):
    assert guard.money_amounts(text) == {3950, 4250}


@pytest.mark.parametrize("text", ["$18,995 - 3 weeks", "$18,995 a 3-week program", "$18,995 to 3 people"])
def test_a_duration_after_a_price_is_not_a_range(text):
    assert guard.money_amounts(text) == {18995}


def test_years_durations_and_percentages_are_not_money():
    text = "In 2026 the 3 weeks program (save 40% to 70%) costs $18,995, rated 4.3 by 6 reviews."
    assert guard.money_amounts(text) == {18995}


def test_dates_and_dose_percentages_are_not_money():
    assert guard.money_amounts(REVIEW.chunk.text) == set()
    assert guard.money_amounts(LISTING.chunk.text) == {110}  # not 31 from "Jul 31" or 2023


def test_several_prices_in_one_answer():
    text = "Out-Patient: 18,995 USD (3 weeks). In-Patient: 30,000 USD. Extras from $200 per visit."
    assert guard.money_amounts(text) == {18995, 30000, 200}


# --- check ------------------------------------------------------------------------------


def test_correct_cited_price_passes():
    assert guard.check(draft("The Out-Patient Program costs 18,995 USD for 3 weeks.", ["C1"]), HITS) == []


def test_price_matches_the_chunk_written_another_way():
    # The price list writes "18995 USD"; the answer may write it the package-page way, "$18,995".
    assert guard.check(draft("Breast cancer: $18,995.", ["C2"]), HITS) == []


def test_answer_in_arabic_digits_matches_chunk():
    assert guard.check(draft("التكلفة ١٨٬٩٩٥ دولار", ["C1"]), HITS) == []


def test_invented_price_fails():
    problems = guard.check(draft("Yes, ITC's treatment costs $50.", ["C1"]), HITS)
    assert len(problems) == 1
    assert "50" in problems[0]


def test_near_miss_price_fails():
    problems = guard.check(draft("It costs $18,000.", ["C1"]), HITS)
    assert "18,000" in problems[0]


@pytest.mark.parametrize(
    ("answer", "ref"),
    [("A consultation costs $50.", "C5"), ("Ozone therapy costs 31 USD.", "C6")],
)
def test_a_plain_number_in_the_chunk_does_not_vouch_for_a_price(answer, ref):
    # "a 50% dose" and "Jul 31" contain the digits, but neither is an amount of money.
    problems = guard.check(draft(answer, [ref]), HITS)
    assert len(problems) == 1 and "amounts" in problems[0]


def test_converted_amount_fails():
    problems = guard.check(draft("The Out-Patient Program costs about 330,000 pesos.", ["C1"]), HITS)
    assert "330,000" in problems[0]


def test_correct_price_from_an_uncited_chunk_fails():
    # $50 is real, but only on the comparison page (C3). Citing the package page doesn't cover it.
    problems = guard.check(draft("The consultation starts at $50.", ["C1"]), HITS)
    assert len(problems) == 1


def test_price_moved_to_the_right_chunk_passes():
    assert guard.check(draft("The Initial Holistic Consultation is $50 - $250.", ["C3"]), HITS) == []


def test_range_needs_both_ends_in_cited_chunks():
    assert guard.check(draft("Specific Killer Cell Antibody: $3,950 - $4,250 USD.", ["C4"]), HITS) == []
    problems = guard.check(draft("Specific Killer Cell Antibody: $3,950 - $4,500 USD.", ["C4"]), HITS)
    assert "4,500" in problems[0] and "3,950" not in problems[0]


def test_citation_outside_hits_fails():
    problems = guard.check(draft("The Out-Patient Program costs 18,995 USD.", ["C1", "C9"]), HITS)
    assert len(problems) == 1
    assert "C9" in problems[0]


def test_unknown_citation_does_not_vouch_for_a_price():
    problems = guard.check(draft("It costs 18,995 USD.", ["C9"]), HITS)
    assert len(problems) == 2  # bad citation, and the price is in no cited chunk


def test_answerable_draft_must_cite():
    problems = guard.check(draft("ITC is in Tijuana.", []), HITS)
    assert problems == ["The answer cites no chunk; list every chunk it uses in citations."]


def test_unanswerable_draft_without_citations_passes():
    assert guard.check(draft("These pages don't list a recovery time.", [], answerable=False), HITS) == []


def test_unanswerable_draft_still_cannot_state_unsupported_prices():
    assert guard.check(draft("I can't confirm $50.", [], answerable=False), HITS) != []


def test_years_in_answer_are_not_checked():
    assert guard.check(draft("Prices last verified in 2026: 18,995 USD for 3 weeks.", ["C1"]), HITS) == []


# --- percentages ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("save 40% to 70%", {"40", "70"}),
        ("save 40 to 70 %", {"40", "70"}),
        ("a 4.5% fee", {"4.5"}),
        ("un 4,5 % más", {"4.5"}),
        ("ahorra 30 por ciento", {"30"}),
        ("وفر ٧٠٪", {"70"}),
        ("$18,995 for 3 weeks in 2026", set()),
    ],
)
def test_percentages(text, expected):
    assert guard.percentages(text) == expected


def test_invented_percentage_fails():
    problems = guard.check(draft("ITC's page states the Out-Patient Program has a 70% success rate.", ["C1"]), HITS)
    assert problems == [
        "These percentages are not in any chunk you cited: 70%. Give a percentage only as the cited chunk "
        "states it, or leave it out."
    ]


def test_percentage_from_the_cited_chunk_passes():
    assert guard.check(draft("The page says patients can save 40% to 70% abroad.", ["C7"]), HITS) == []


def test_a_price_does_not_vouch_for_a_percentage():
    # "18995" is in the chunk, but not as a percentage.
    assert guard.check(draft("Breast cancer: 18995%.", ["C2"]), HITS) != []


def test_by_ref_maps_citations_to_chunks():
    assert guard.by_ref(HITS)["C2"] is PRICE_LIST.chunk


def test_currency_named_with_a_nationality_word_is_still_checked():
    # "330,000 Mexican pesos" is a conversion the pages never state, so it must be caught.
    assert 330000 in guard.money_amounts("About 330,000 Mexican pesos.")
    assert 18995 in guard.money_amounts("It costs 18,995 US dollars.")
