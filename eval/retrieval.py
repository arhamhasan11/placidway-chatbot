"""`run_eval.py --retrieval`: scores the search step alone, with no HTTP and no answer model.

Writes hit@TOP_K per retrieval mode, and each question's best cosine so MIN_COSINE is picked from data.
"""

from datetime import date
from pathlib import Path

import config

MODES = ("hybrid", "dense", "bm25")


def retrieval_report(questions: list[dict], path: Path) -> int:
    """Search every question's first turn in each mode; write the two tables. Returns the exit code."""
    if not config.OPENAI_API_KEY or not config.EMBEDDINGS_FILE.exists():
        print("--retrieval needs OPENAI_API_KEY and data/embeddings.npy (run: python -m ingest.build)")
        return 2
    from bot import retrieve  # imported here: only this mode needs the index and the embedding API
    from bot.llm import LLMError

    hits = {}
    try:
        for mode in MODES:
            config.RETRIEVAL_MODE = mode  # retrieve reads it at call time, so this switches modes
            retrieve.reset()
            hits[mode] = {q["id"]: retrieve.search(q["turns"][0]) for q in questions}
    except LLMError as exc:
        print(f"--retrieval stopped: the embedding call failed ({exc})")
        return 2
    hit_lines, totals = hit_table(questions, hits)
    cosine_lines = cosine_table(questions, hits["dense"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Retrieval report {date.today()}", ""] + hit_lines + [""] + cosine_lines
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"hit@{config.TOP_K}: {totals}")
    print(cosine_lines[-1])
    print(f"Report: {path}")
    return 0


# Both tables read only hit.chunk.url and hit.cosine, so they don't depend on how hits are labelled.


def hit_table(questions: list[dict], hits: dict) -> tuple[list[str], str]:
    """A hit = any expected page among the top-K chunks; scored only for items that must link a page."""
    scored = [q for q in questions if q.get("cite") != "none" and q.get("source_urls")]
    counts = dict.fromkeys(MODES, 0)
    lines = [f"## hit@{config.TOP_K} (first turn only)", "", "| id | " + " | ".join(MODES) + " |",
             "|---|" + "---|" * len(MODES)]
    for q in scored:
        urls = {u.rstrip("/") for u in q["source_urls"]}
        row = []
        for mode in MODES:
            hit = any(h.chunk.url.rstrip("/") in urls for h in hits[mode][q["id"]])
            counts[mode] += hit
            row.append("hit" if hit else "MISS")
        lines.append(f"| {q['id']} | " + " | ".join(row) + " |")
    totals = ", ".join(f"{mode} {counts[mode]}/{len(scored)}" for mode in MODES)
    return lines + ["", f"Totals: {totals}"], totals


def cosine_table(questions: list[dict], dense_hits: dict) -> list[str]:
    """Best embedding match per item, sorted, so MIN_COSINE can sit between answerable and not."""
    best = {q["id"]: max((h.cosine for h in dense_hits[q["id"]]), default=0.0) for q in questions}
    lines = [f"## Best cosine (dense mode); MIN_COSINE is {config.MIN_COSINE}", "",
             "| id | cite | log_unanswered | best cosine |", "|---|---|---|---|"]
    for q in sorted(questions, key=lambda q: -best[q["id"]]):
        lines.append(f"| {q['id']} | {q.get('cite')} | {q.get('log_unanswered')} | {best[q['id']]:.3f} |")
    must_link = [best[q["id"]] for q in questions if q.get("cite") in ("all", "any")]
    no_link = [best[q["id"]] for q in questions if q.get("cite") == "none"]
    return lines + ["", f"Lowest best cosine where a link is required: {min(must_link, default=0):.3f}; "
                        f"highest where no link is expected: {max(no_link, default=0):.3f}"]
