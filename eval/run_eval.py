"""Golden-question evaluation: sends eval/questions.yaml to a running bot and writes a markdown report.

Run by hand against a local or deployed bot; --retrieval scores the search step alone, without HTTP.
"""

import argparse
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # `python eval/run_eval.py` only puts eval/ on the path

import config  # noqa: E402
from eval import retrieval  # noqa: E402  (the --retrieval report)
from eval.checks import check_item  # noqa: E402  (the matching rules; this file is CLI, HTTP and reports)

QUESTIONS_FILE = ROOT / "eval" / "questions.yaml"
SMOKE_IDS = ["T01", "T03", "T05", "T06", "T07", "T08"]  # a fast post-deploy pass: 6 items, 7 messages
PAUSE_S = 1.0  # between questions; the server allows 8 messages a minute per IP
RATE_LIMIT_WAIT_S = 60  # one full rate-limit window
HTTP_TIMEOUT_S = 150  # above vercel.json maxDuration (120 s), so a server-side timeout shows up as itself
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


# --- Talking to the bot ------------------------------------------------------------------


def post_chat(client: httpx.Client, base_url: str, message: str, history: list[dict]) -> tuple[dict, float]:
    """One POST /api/chat. A failure becomes an 'error' response so the report still shows it."""
    start = time.perf_counter()
    try:
        r = client.post(f"{base_url}/api/chat", json={"message": message, "history": history})
        body = r.json()
        if "answer" not in body:  # e.g. a 422 validation error
            body = {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    except (httpx.HTTPError, ValueError) as exc:  # server down, timeout, or a non-JSON error page
        body = {"error": f"{type(exc).__name__}: {exc}"}
    seconds = time.perf_counter() - start
    if "error" in body:
        body.update(answer="", sources=[], outcome="error", language="", disclaimer=None, show_lead_form=False)
    return body, seconds


def ask(client: httpx.Client, base_url: str, message: str, history: list[dict]) -> tuple[dict, float]:
    """Send one turn; a rate-limited reply waits one window and retries once."""
    body, seconds = post_chat(client, base_url, message, history)
    if body["outcome"] == "rate_limited":
        print(f"  rate limited; waiting {RATE_LIMIT_WAIT_S}s and retrying once", flush=True)
        time.sleep(RATE_LIMIT_WAIT_S)
        body, seconds = post_chat(client, base_url, message, history)
    return body, seconds


def run_item(client: httpx.Client, base_url: str, item: dict) -> dict:
    """Send the item's turns in order, carrying the history the way the web UI does."""
    history, responses, latencies = [], [], []
    for message in item["turns"]:
        body, seconds = ask(client, base_url, message, history)
        responses.append(body)
        latencies.append(seconds)
        history.append({"role": "user", "content": message, "disclaimer": False})
        history.append({"role": "assistant", "content": body["answer"][:config.MAX_HISTORY_CHARS],
                        "disclaimer": body["disclaimer"] is not None})
    return {"item": item, "responses": responses, "latencies": latencies,
            "result": check_item(item, responses)}


# --- Reports ----------------------------------------------------------------------------------


def turn_lines(item: dict, responses: list[dict]) -> list[str]:
    """Each question and the bot's verbatim reply, quoted so its markdown can't break the report."""
    lines = []
    for n, (message, r) in enumerate(zip(item["turns"], responses), start=1):
        reply = r["answer"] or r.get("error", "")
        sources = "; ".join(f"[{s['title']}]({s['url']})" for s in r["sources"]) or "none"
        lines += [f"**T{n} user:** {message}", "", f"**T{n} bot** (outcome: `{r['outcome']}`):", ""]
        lines += [f"> {line}" for line in reply.splitlines()] or [">"]
        lines += ["", f"- Sources: {sources}", f"- Disclaimer: {r['disclaimer'] or 'none'}",
                  f"- Lead form: {'shown' if r['show_lead_form'] else 'not shown'}", ""]
    return lines


def check_lines(result: dict) -> list[str]:
    return ([f"- FAILED: {f}" for f in result["failed"]] + [f"- warning: {w}" for w in result["warnings"]]
            + [f"- [ ] {m}" for m in result["manual"]])


def write_report(results: list[dict], path: Path, base_url: str, global_claims: list[str]) -> None:
    lines = [f"# Evaluation run {date.today()}", "", f"Base URL: {base_url}", "",
             "| id | type | result | failed checks | latency |", "|---|---|---|---|---|"]
    for r in results:
        verdict = "FAIL" if r["result"]["failed"] else "pass"
        latency = " / ".join(f"{s:.1f}s" for s in r["latencies"])
        lines.append(f"| {r['item']['id']} | {r['item']['type']} | {verdict} | "
                     f"{'; '.join(r['result']['failed'])} | {latency} |")
    lines += ["", "## Checked by hand on every answer", ""] + [f"- [ ] must not claim: {c}" for c in global_claims]
    for r in results:
        item = r["item"]
        lines += ["", f"## {item['id']} ({item['type']})", "", f"Expected: {item['expected_behaviour']}", ""]
        lines += turn_lines(item, r["responses"]) + ["**Checks**", ""] + check_lines(r["result"])
    write_lines(path, lines)


def git_sha(health: dict) -> str:
    """Prefer the deployed commit the server reports; fall back to this checkout."""
    if health.get("git_sha"):
        return health["git_sha"]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    except OSError:  # git not installed
        return "unknown"
    return f"{out.stdout.strip()} (local checkout)" if out.returncode == 0 else "unknown"


def write_test_results(results: list[dict], path: Path, base_url: str, health: dict) -> None:
    """The brief's deliverable: the 8 required tests with the bot's verbatim answers and the verdict."""
    lines = ["# Test results", "", f"- Date: {date.today()}", f"- Base URL: {base_url}",
             f"- Git commit: {git_sha(health)}", f"- Model: {health.get('models', {}).get('chat', 'unknown')}", "",
             "Sent to the running bot by `python eval/run_eval.py --test-results`. Answers are verbatim. "
             "FAILED and warning lines are automatic checks; lines marked [ ] need a human read."]
    for r in results:
        item = r["item"]
        if not item["id"].startswith("T"):  # H items are our own; the brief asks for its 8 tests only
            continue
        verdict = "FAIL" if r["result"]["failed"] else "PASS (automatic checks)"
        lines += ["", f"## Test {item['brief_test']}: {item['type'].replace('_', ' ')} ({item['id']})", "",
                  f"Expected: {item['expected_behaviour']}", ""]
        lines += turn_lines(item, r["responses"]) + [f"**Result: {verdict}**", ""] + check_lines(r["result"])
    write_lines(path, lines)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- CLI -----------------------------------------------------------------------------------------


def load_questions(path: Path = QUESTIONS_FILE) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))  # resolves the U1..U7 URL anchors


def remote_full_run(base_url: str, smoke: bool, only: str | None) -> bool:
    """A full run sends every turn (47), most of a deployed bot's RATE_LIMIT_PER_DAY for one IP."""
    return not (smoke or only) and httpx.URL(base_url).host not in LOCAL_HOSTS


def select(questions: list[dict], only: str | None, smoke: bool) -> list[dict]:
    if smoke:
        ids = SMOKE_IDS
    elif only:
        ids = [i.strip().upper() for i in only.split(",")]
    else:
        return questions
    unknown = set(ids) - {q["id"] for q in questions}
    if unknown:
        sys.exit(f"Unknown question ids: {', '.join(sorted(unknown))}")
    return [q for q in questions if q["id"] in ids]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--only", help="comma-separated ids, e.g. T01,T08")
    p.add_argument("--smoke", action="store_true", help="only " + ",".join(SMOKE_IDS))
    p.add_argument("--out", type=Path, help="report path (default eval/out/results.md or retrieval.md)")
    p.add_argument("--test-results", type=Path, help="also write the brief's T01-T08 deliverable here")
    p.add_argument("--retrieval", action="store_true", help="score search only, no HTTP")
    p.add_argument("--strict", action="store_true", help="exit 1 if any item fails")
    args = p.parse_args(argv)
    data = load_questions()
    questions = select(data["questions"], args.only, args.smoke)
    if args.retrieval:
        return retrieval.retrieval_report(questions, args.out or ROOT / "eval" / "out" / "retrieval.md")
    base_url, out = args.base_url.rstrip("/"), args.out or ROOT / "eval" / "out" / "results.md"
    if remote_full_run(base_url, args.smoke, args.only):
        turns = sum(len(q["turns"]) for q in questions)
        print(f"Warning: {turns} messages to {base_url}, which allows {config.RATE_LIMIT_PER_DAY} a day per IP. "
              "Full runs belong on a local server; use --smoke against production.", flush=True)
    results = []
    with httpx.Client(timeout=HTTP_TIMEOUT_S) as client:
        for n, item in enumerate(questions):
            if n:
                time.sleep(PAUSE_S)
            print(f"{item['id']} ({len(item['turns'])} turn(s))", flush=True)
            results.append(run_item(client, base_url, item))
        health = {}
        if args.test_results:
            try:
                health = client.get(f"{base_url}/api/health").json()
            except (httpx.HTTPError, ValueError) as exc:  # the header then says "unknown"
                print(f"  /api/health unavailable: {exc}")
    write_report(results, out, base_url, data.get("global_checks", {}).get("must_not_claim", []))
    if args.test_results:
        write_test_results(results, args.test_results, base_url, health)
    passed = sum(not r["result"]["failed"] for r in results)
    manual = sum(any("(semantic)" in m for m in r["result"]["manual"]) for r in results)
    print(f"{passed}/{len(results)} passed (deterministic checks); {manual} items have manual checks")
    print(f"Report: {out}")
    return 1 if args.strict and passed < len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
