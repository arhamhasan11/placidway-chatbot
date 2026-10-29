"""The deterministic checks behind the golden-question report (rules in the header of questions.yaml).

Pure functions: no HTTP, no LLM. eval/run_eval.py calls check_item() once per conversation.
"""

import re

from bot.guard import money_amounts  # the same money parser the answer guard uses


NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6"}


def _squash(text: str) -> str:
    """Lowercase, then drop hyphens and all whitespace: 'Out Patient' and 'out-patient' both become 'outpatient'.

    Durations are normalised too, so "3 weeks", "3-week" and "three weeks" all match.
    """
    text = text.lower()
    text = re.sub(r"\b(one|two|three|four|five|six)\b", lambda m: NUMBER_WORDS[m.group(1)], text)
    text = re.sub(r"\b(\d+)[\s-]*(day|week|month)s?\b", r"\1\2", text)
    return re.sub(r"[\s-]+", "", text)


def text_match(expected: str, answer: str) -> bool:
    """Case-insensitive substring that ignores hyphens and spaces, so the model's spelling choices don't fail an item."""
    # Word gaps vanish too, so "in-patient" would also match "within patient": keep items specific.
    return _squash(expected) in _squash(answer)


def money_match(expected: str, answer: str) -> bool:
    """'$3,950 - $4,250' needs both amounts in the answer; '$200 per visit' also needs the words."""
    amounts = {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", expected)}
    words = re.sub(r"[$\d,\s-]+", " ", expected).strip()
    # Parsed amounts, not substrings: "$18,995" must not match "$189,950", and must match "18.995 USD".
    return amounts <= money_amounts(answer) and (not words or text_match(words, answer))


def found(expected: str, answer: str) -> bool:
    """Money items ('$18,995', bare '30,000') compare parsed amounts; everything else is text."""
    if expected.startswith("$") or re.fullmatch(r"[\d,]+", expected):
        return money_match(expected, answer)
    return text_match(expected, answer)


def route(expected: str, answers: list[str]) -> tuple[str, str]:
    """'T1: $18,995' is checked against turn 1's answer; anything else against the final answer."""
    m = re.match(r"T(\d+): (.*)", expected)
    if m:
        return m.group(2), answers[int(m.group(1)) - 1]
    return expected, answers[-1]


def cite_problems(item: dict, response: dict) -> list[str]:
    """Apply the item's cite rule (all / any / optional / none) to the answer's source links."""
    expected = {u.rstrip("/") for u in item.get("source_urls", [])}
    cited = {s["url"].rstrip("/") for s in response["sources"]}
    rule = item.get("cite", "optional")
    if rule == "all" and not expected <= cited:
        return [f"no link to {'/'.join(u.split('/')[3:5])}" for u in sorted(expected - cited)]
    if rule == "any" and expected and not expected & cited:
        return ["links none of the expected pages"]
    if rule == "none" and cited:
        return ["links a page where no link was expected"]
    return []


def check_item(item: dict, responses: list[dict]) -> dict:
    """Run the deterministic checks; return failed checks, warnings and what a human must judge."""
    answers = [r["answer"] for r in responses]
    final = responses[-1]
    failed, warnings, manual = [], [], []
    for expected in item.get("must_include", []):
        if "(semantic)" in expected:
            manual.append(f"must include: {expected}")
        elif not found(*route(expected, answers)):
            failed.append(f"missing {expected!r}")
    for banned in item.get("must_not_include", []):
        if found(*route(banned, answers)):
            failed.append(f"contains {banned!r}")
    failed += cite_problems(item, final)
    # lead_capture belongs to the conversation, not one turn (H29 offers the form on turn 2 of 2).
    if item.get("lead_capture") and not any(r["show_lead_form"] for r in responses):
        failed.append("no lead form")
    if item.get("disclaimer") == "required" and not final["disclaimer"]:
        failed.append("no disclaimer")
    if item.get("disclaimer") == "not_expected" and final["disclaimer"]:
        warnings.append("disclaimer shown where none was expected (small deduction, not a failure)")
    language = item.get("language")
    for n, r in enumerate(responses, start=1):
        if "error" in r:  # says why every other check failed
            failed.insert(0, f"T{n} got no answer ({r['error'][:80]})")
        elif language and r["language"] != language:
            failed.append(f"T{n} language {r['language']!r}, expected {language!r}")
    manual += [f"must not claim: {c}" for c in item.get("must_not_claim", [])]
    manual += [f"should include: {s}" for s in item.get("should_include", [])]
    return {"failed": failed, "warnings": warnings, "manual": manual}
