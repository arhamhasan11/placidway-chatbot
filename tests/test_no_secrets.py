"""Nothing that looks like a real key or password is in a file that could be pushed, and .env
stays out of git. Run before every push: python -m pytest tests/test_no_secrets.py"""

import os
import re
from pathlib import Path

import config

# Generated, local-only or third-party folders; none of them is committed. data/ IS scanned:
# the refresh workflow commits data/pages, chunks.json and report.json without a human looking.
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".vercel", "node_modules"}
FIXTURES = Path("tests") / "fixtures"  # saved copies of PlacidWay's own pages
LOCAL_LOG = Path("data") / "local_log.jsonl"  # git-ignored dev log of chat turns, never pushed
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),  # OpenAI API key
    re.compile(r"postgres(ql)?://[^\s'\"]+:[^\s'\"]+@"),  # database URL with a password
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),  # GitHub classic token
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),  # GitHub fine-grained token
]


def is_local_env_file(name: str) -> bool:
    """.env and .env.* hold real secrets and are git-ignored; .env.example is committed."""
    return (name == ".env" or name.startswith(".env.")) and name != ".env.example"


def candidate_files(root: Path = config.ROOT) -> list[Path]:
    files = []
    for folder, dirs, names in os.walk(root):
        here = Path(folder)
        # Pruned in place so the walk never enters a skipped folder (.venv alone has thousands of files).
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and here / d != root / FIXTURES]
        files += [here / name for name in names if not is_local_env_file(name) and here / name != root / LOCAL_LOG]
    return files


def test_scan_includes_committed_data_but_not_local_only_files(tmp_path):
    for name in ("data/pages/itc-profile.md", "data/chunks.json", "data/local_log.jsonl", "tests/fixtures/a.html",
                 ".venv/lib/x.py", ".env", ".env.example", "app.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x", encoding="utf-8")
    found = {path.relative_to(tmp_path).as_posix() for path in candidate_files(tmp_path)}
    assert found == {"data/pages/itc-profile.md", "data/chunks.json", ".env.example", "app.py"}


def test_no_key_like_strings_in_committable_files():
    found = []
    for path in candidate_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary file (e.g. an image): no text to leak
        found += [f"{path.relative_to(config.ROOT)}: {p.pattern}" for p in SECRET_PATTERNS if p.search(text)]
    assert not found, "possible secrets:\n" + "\n".join(found)


def test_env_file_is_git_ignored():
    ignored = (config.ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in ignored]


def test_env_example_has_names_but_no_values():
    for line in (config.ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            name, _, value = line.partition("=")
            assert value.strip() == "", f".env.example must not hold a value for {name}"
