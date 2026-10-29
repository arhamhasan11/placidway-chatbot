"""LLM call 2: writes a draft answer from the retrieved chunks only. Called by bot/chat.py,
at most twice per turn (the second time with the guard's feedback)."""

import config
from bot import llm
from bot.schemas import Draft, Hit


def draft_answer(question: str, language: str, hits: list[Hit], medical: bool, feedback: str | None = None) -> Draft:
    """Ask the model for a grounded answer; bot/guard.py checks it before anyone sees it."""
    system = (config.PROMPTS_DIR / "answer.md").read_text(encoding="utf-8")
    return llm.chat_json(system, render(question, language, hits, medical, feedback), Draft, config.ANSWER_MAX_TOKENS)


def render(question: str, language: str, hits: list[Hit], medical: bool, feedback: str | None) -> str:
    """The user content, in the order prompts/answer.md describes it."""
    lines = [f"Target language: {language}", f"Question: <question>{question}</question>"]
    if medical:
        lines.append("Medical: yes. The question touches health, safety or a treatment's effects (section 6).")
    if feedback:
        lines.append(f"Feedback: your previous draft failed these checks. Fix them: {feedback}")
    lines.append("")
    for hit in hits:
        lines.append(f'<chunk id="{hit.ref}" source="{hit.chunk.label}">\n{hit.chunk.text}\n</chunk>')
    return "\n".join(lines)
