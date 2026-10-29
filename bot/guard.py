"""Code checks on the answer model's draft, run by bot/chat.py before anything reaches the user.

A prompt can ask for grounded prices; only code can verify them. Every currency amount and every
percentage in the answer must appear in a chunk the answer cites, and every citation must be a
chunk we sent.
"""

import re

from bot.schemas import Chunk, Draft, Hit

# Arabic-Indic (U+0660-0669) and Extended Arabic-Indic (U+06F0-06F9) digits -> ASCII.
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "0123456789" * 2)

# Thousands separators: comma, Spanish dot, Arabic thousands sign, no-break space, narrow no-break space, space.
_SEP = r"[,.\u066c\u00a0\u202f ]"
# A separator only counts when exactly 3 digits follow ("18,995" "18.995" "18 995"), so
# "5.0" or "3 weeks" never fuse into one number. Otherwise a plain digit run ("18995").
_NUM = rf"(?<!\d)(\d{{1,3}}(?:{_SEP}\d{{3}})+(?!\d)|\d+)(?:[.,]00)?(?!\d)"  # ".00" cents dropped
# Pesos and euros are recognised so that a converted amount is caught: no chunk states one.
_PREFIX = r"(?:US\$|U\$S|USD|MXN\$?|EUR|€|\$)\s?"
# "Mexican pesos" / "US dollars": a nationality word may sit between the number and the currency.
_SUFFIX = r"\s?(?:(?:Mexican|American|US|U\.S\.)\s)?(?:USD|US\$|MXN|EUR|€|\$|dollars?|d[oó]lares|pesos?|euros?|دولار|بيزو|يورو)"
# Range connectors: hyphen, en dash, em dash, and the word for "to"/"and" in English, Spanish, Arabic.
# "3,950 - 4,250 USD", "3.950 a 4.250 dólares", "between 3,950 and 4,250 USD".
_TO = r"\s*(?:-|\u2013|\u2014|to|and|a|y|إلى|و)\s*"

# "$3,950 - $4,250" needs no range rule: each end has its own tag. A range only matters when one
# tag follows both ends. After a prefix tag, "$18,995 - 3 weeks" must not make "3" a price.
_MONEY = re.compile(rf"{_PREFIX}{_NUM}|{_NUM}(?:{_TO}{_NUM})?{_SUFFIX}", re.IGNORECASE)

# Percentages keep their decimals ("4.5%" is not "5%"); "40% to 70%" and "40 to 70%" give both ends.
# The sign is %, the Arabic percent sign U+066A, or the words "por ciento" / "percent".
_PCT_NUM = r"(?<![\d.,])(\d+(?:[.,]\d+)?)"
_PERCENT = re.compile(rf"{_PCT_NUM}(?:{_TO}{_PCT_NUM})?\s?(?:%|\u066a|por\s?ciento|per\s?cent)", re.IGNORECASE)


def money_amounts(text: str) -> set[int]:
    """Currency-tagged amounts in `text`, separators and cents removed. Years, durations, dates and
    percentages ("2026", "3 weeks", "Jul 31", "50%") carry no currency tag, so they are not money."""
    text = text.translate(_DIGITS)
    return {int(re.sub(r"\D", "", n)) for m in _MONEY.finditer(text) for n in m.groups() if n}


def percentages(text: str) -> set[str]:
    """Percent figures in `text`, e.g. {"40", "70"} for "save 40% to 70%"; "4,5 %" is written "4.5"."""
    text = text.translate(_DIGITS)
    return {n.replace(",", ".") for m in _PERCENT.finditer(text) for n in m.groups() if n}


def by_ref(hits: list[Hit]) -> dict[str, Chunk]:
    """{"C1": chunk, ...}: how a citation in a draft finds its chunk (here and in chat._sources)."""
    return {hit.ref: hit.chunk for hit in hits}


def check(draft: Draft, hits: list[Hit]) -> list[str]:
    """Problems with the draft, phrased so they can be fed back to the model. [] means it passes."""
    chunks = by_ref(hits)
    problems = []

    unknown = [ref for ref in draft.citations if ref not in chunks]
    if unknown:
        problems.append(f"Citations {', '.join(unknown)} are not chunks you were given; cite only {', '.join(chunks)}.")

    if draft.answerable and not draft.citations:
        problems.append("The answer cites no chunk; list every chunk it uses in citations.")

    # Compared money to money: a "$50" answer is not vouched for by "a 50% dose" or "Jul 31".
    cited_text = " ".join(chunks[ref].text for ref in draft.citations if ref in chunks)
    missing = sorted(money_amounts(draft.answer) - money_amounts(cited_text))
    if missing:
        amounts = ", ".join(f"{n:,}" for n in missing)
        # The usual cause is a correction that repeats the user's wrong figure ("I can't confirm $50"),
        # so the feedback says exactly how to correct it without the number.
        problems.append(
            f"These amounts are not in any chunk you cited: {amounts}. Copy each price exactly from "
            "the chunk that states it and cite that chunk, or leave the amount out. If an amount is "
            "the user's own figure, do not repeat it: write 'that price' and give the listed price instead."
        )

    invented = sorted(percentages(draft.answer) - percentages(cited_text))
    if invented:
        figures = ", ".join(f"{p}%" for p in invented)
        problems.append(
            f"These percentages are not in any chunk you cited: {figures}. Give a percentage only as "
            "the cited chunk states it, or leave it out."
        )
    return problems
