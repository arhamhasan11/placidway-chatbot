"""Unit tests for the evaluation runner (eval/run_eval.py, checks.py, retrieval.py). No network, no LLM."""

import json
from types import SimpleNamespace

import httpx

from eval import retrieval, run_eval
import pytest

from eval.checks import check_item, found, route, text_match

ADENO = ("https://www.placidway.com/package/7903/"
         "Alternative-Adenocarcinoma-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC")
PRICES = "https://www.placidway.com/price-list/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center"


def reply(answer: str, urls: tuple[str, ...] = (), language: str = "en", disclaimer: str | None = None,
          lead: bool = False) -> dict:
    """A /api/chat response body as the runner sees it."""
    return {"answer": answer, "sources": [{"title": "page", "url": u} for u in urls], "outcome": "answered",
            "language": language, "disclaimer": disclaimer, "show_lead_form": lead}


def item(**fields) -> dict:
    return {"id": "X01", "type": "test", "turns": ["q1", "q2"], **fields}


# --- text matching -----------------------------------------------------------------


def test_text_ignores_hyphens_and_case():
    assert text_match("out-patient", "The Outpatient Program")
    assert text_match("Out-Patient", "out-patient")
    assert text_match("X-ray, ultrasound, or CT", "one X-ray, ultrasound, or CT scan")


def test_text_ignores_spaces_inside_words():
    # The questions.yaml header promises "out-patient" = "Out Patient" = "outpatient", both ways round.
    for spelling in ("out-patient", "Out Patient", "outpatient"):
        assert text_match(spelling, "The Out Patient Program")
        assert text_match(spelling, "the outpatient program")
        assert text_match(spelling, "an out-patient stay")
    assert not text_match("out-patient", "the in-patient program")


def test_text_collapses_whitespace():
    assert text_match("3 weeks", "lasts 3\n  weeks")
    assert text_match("3 weeks", "three weeks")  # number words count too


# --- money matching (uses bot.guard.money_amounts) ---------------------------------


def test_money_formats():
    assert found("$18,995", "It starts at $18,995 USD.")
    assert found("$18,995", "Desde 18.995 USD por 3 semanas.")
    assert found("$18995", "costs $18,995")
    assert found("30,000", "البرنامج الداخلي ٣٠٬٠٠٠ دولار أمريكي")


def test_money_is_not_a_substring_match():
    assert not found("$18,995", "about $189,950")
    assert not found("$1000", "$10,000 for laryngeal cancer")


def test_money_range_needs_both_ends():
    assert found("$3,950 - $4,250", "between $3,950 and $4,250")
    assert not found("$3,950 - $4,250", "from $3,950")


def test_money_with_words_needs_the_words():
    assert found("$200 per visit", "Consultations cost $200 per visit.")
    assert not found("$200 per visit", "Blood costs $200 per pint.")


def test_numbers_with_dots_are_text_not_money():
    assert found("4.3", "Rated 4.3 from 6 reviews")
    assert not found("4.3", "Rated 43 from 6 reviews")


# --- routing and item checks -------------------------------------------------------------


def test_route_by_turn_prefix():
    answers = ["first", "second"]
    assert route("T1: $18,995", answers) == ("$18,995", "first")
    assert route("after 3 months", answers) == ("after 3 months", "second")


def test_turn_prefix_checks_the_earlier_answer():
    q = item(must_include=["T1: $18,995", "after 3 months"])
    good = [reply("Out-patient is $18,995."), reply("Follow-up after 3 months.")]
    swapped = [reply("Follow-up after 3 months."), reply("Out-patient is $18,995.")]
    assert check_item(q, good)["failed"] == []
    assert len(check_item(q, swapped)["failed"]) == 2


def test_must_not_include_money():
    q = item(must_not_include=["$24,000"])
    assert check_item(q, [reply("Verita charges 24,000 USD.")])["failed"] == ["contains '$24,000'"]
    assert check_item(q, [reply("ITC charges $18,995.")])["failed"] == []


def test_semantic_items_go_to_manual_review():
    q = item(must_include=["cannot book (semantic)"], must_not_claim=["A slot is free."])
    result = check_item(q, [reply("I can't book.")])
    assert result["failed"] == []
    assert result["manual"] == ["must include: cannot book (semantic)", "must not claim: A slot is free."]


def test_cite_rules():
    one = [reply("x", urls=(ADENO,))]

    def failed(urls: list[str], cite: str, responses: list[dict] = one) -> list[str]:
        return check_item(item(source_urls=urls, cite=cite), responses)["failed"]

    assert failed([ADENO, PRICES], "any") == []
    assert failed([ADENO, PRICES], "all") == ["no link to price-list/5"]
    assert failed([PRICES], "any") == ["links none of the expected pages"]
    assert failed([PRICES], "optional") == []
    assert failed([], "none") == ["links a page where no link was expected"]
    assert failed([], "none", [reply("x")]) == []


def test_lead_form_on_any_turn_counts():
    q = item(lead_capture=True)
    assert check_item(q, [reply("a", lead=True), reply("b")])["failed"] == []
    assert check_item(q, [reply("a"), reply("b")])["failed"] == ["no lead form"]


def test_disclaimer_required_fails_but_unexpected_only_warns():
    assert check_item(item(disclaimer="required"), [reply("a")])["failed"] == ["no disclaimer"]
    result = check_item(item(disclaimer="not_expected"), [reply("a", disclaimer="Not medical advice.")])
    assert result["failed"] == [] and len(result["warnings"]) == 1


def test_language_checked_on_every_turn():
    q = item(language="es")
    spanish = reply("hola", language="es")
    assert check_item(q, [spanish, spanish])["failed"] == []
    assert check_item(q, [reply("hi"), spanish])["failed"] == ["T1 language 'en', expected 'es'"]


def test_golden_file_loads_with_resolved_urls():
    data = run_eval.load_questions()
    by_id = {q["id"]: q for q in data["questions"]}
    assert len(by_id) == 38
    assert by_id["T03"]["source_urls"] == [ADENO]
    assert run_eval.select(data["questions"], None, smoke=True)[0]["id"] == "T01"


# --- conversation loop, against a fake transport (no network) ----------------------------


def test_run_item_carries_history_and_retries_once_when_rate_limited(monkeypatch):
    monkeypatch.setattr(run_eval, "RATE_LIMIT_WAIT_S", 0)
    replies = [{**reply("slow down"), "outcome": "rate_limited"}, reply("$18,995", disclaimer="d"),
               reply("after 3 months")]
    sent = []

    def fake_server(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=replies[len(sent) - 1])

    client = httpx.Client(transport=httpx.MockTransport(fake_server))
    result = run_eval.run_item(client, "http://bot", item(must_include=["T1: $18,995", "after 3 months"]))
    assert len(sent) == 3  # turn 1 was retried once
    assert sent[2] == {"message": "q2", "history": [
        {"role": "user", "content": "q1", "disclaimer": False},
        {"role": "assistant", "content": "$18,995", "disclaimer": True},
    ]}
    assert result["result"]["failed"] == []


def test_full_run_warning_only_for_a_deployed_bot():
    assert run_eval.remote_full_run("https://placidway-bot.vercel.app", smoke=False, only=None)
    assert not run_eval.remote_full_run("https://placidway-bot.vercel.app", smoke=True, only=None)
    assert not run_eval.remote_full_run("https://placidway-bot.vercel.app", smoke=False, only="T01")
    for local in ("http://127.0.0.1:8000", "http://localhost:8000", "http://[::1]:8000"):
        assert not run_eval.remote_full_run(local, smoke=False, only=None)


def test_cli_prints_the_warning_before_a_full_remote_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_eval, "PAUSE_S", 0)
    no_checks = {"failed": [], "warnings": [], "manual": []}
    monkeypatch.setattr(run_eval, "run_item", lambda client, base_url, item: {
        "item": item, "responses": [], "latencies": [], "result": no_checks})
    out = ["--out", str(tmp_path / "results.md")]
    run_eval.main(["--base-url", "https://placidway-bot.vercel.app", *out])
    assert "use --smoke against production" in capsys.readouterr().out
    run_eval.main(["--base-url", "https://placidway-bot.vercel.app", "--smoke", *out])
    assert "Warning" not in capsys.readouterr().out


def test_server_error_is_reported_not_raised():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(422, json={"detail": "bad"})))
    result = run_eval.run_item(client, "http://bot", item(turns=["q1"], must_include=["$18,995"]))
    assert result["responses"][0]["outcome"] == "error"
    assert result["result"]["failed"][0].startswith("T1 got no answer (HTTP 422")


# --- TEST_RESULTS.md (fake /api/health and transcripts, no HTTP) -------------------------


def test_test_results_header_and_brief_items_only(tmp_path):
    t01 = item(id="T01", type="cost", brief_test=1, turns=["How much?"], expected_behaviour="States $18,995.")
    h01 = {**t01, "id": "H01"}
    responses = [reply("It is $18,995.\nPer program.", urls=(ADENO,))]
    results = [{"item": q, "responses": responses, "latencies": [1.0], "result": check_item(q, responses)}
               for q in (t01, h01)]
    health = {"git_sha": "abc1234", "models": {"chat": "gpt-test", "embed": "embed-test"}}  # /api/health shape
    path = tmp_path / "TEST_RESULTS.md"
    run_eval.write_test_results(results, path, "http://bot", health)
    text = path.read_text(encoding="utf-8")
    assert "- Git commit: abc1234" in text and "- Model: gpt-test" in text
    assert "## Test 1: cost (T01)" in text and "H01" not in text  # H items are ours, not the brief's
    assert "> It is $18,995.\n> Per program." in text
    assert "**Result: PASS (automatic checks)**" in text
    run_eval.write_test_results(results, path, "http://bot", {"git_sha": "abc1234"})  # health had no models
    assert "- Model: unknown" in path.read_text(encoding="utf-8")


# --- retrieval report tables (fake hits, no index) ---------------------------------------


def fake_hit(url: str, cosine: float) -> SimpleNamespace:
    """Only the two fields the tables read (hit.chunk.url, hit.cosine), so a Hit rename can't break this."""
    return SimpleNamespace(chunk=SimpleNamespace(url=url), cosine=cosine)


def test_hit_table_and_cosine_table():
    questions = [{"id": "A", "turns": ["q"], "cite": "all", "source_urls": [ADENO], "log_unanswered": False},
                 {"id": "B", "turns": ["q"], "cite": "none", "source_urls": [], "log_unanswered": True}]
    found_page = {"A": [fake_hit(ADENO, 0.61)], "B": [fake_hit(PRICES, 0.22)]}
    wrong_page = {"A": [fake_hit(PRICES, 0.40)], "B": []}
    lines, totals = retrieval.hit_table(questions, {"hybrid": found_page, "dense": found_page, "bm25": wrong_page})
    assert totals == "hybrid 1/1, dense 1/1, bm25 0/1"  # B has cite none, so it is not scored
    summary = retrieval.cosine_table(questions, found_page)[-1]
    assert summary == "Lowest best cosine where a link is required: 0.610; highest where no link is expected: 0.220"


@pytest.mark.parametrize("answer", ["a 3-week program", "for 3 weeks", "three weeks long", "Three-week stay"])
def test_durations_match_however_they_are_written(answer):
    assert text_match("3 weeks", answer)


def test_different_durations_still_differ():
    assert not text_match("3 weeks", "a 2-week program")
    assert not text_match("after 3 months", "after 3 weeks")
