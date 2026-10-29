"""Download the pages listed in sources.yaml, politely (called by ingest/build.py).

Only listed URLs, only what robots.txt allows, one request per Crawl-delay, and a bot
challenge is an error rather than content.
"""

import time
import urllib.robotparser
from urllib.parse import urlsplit

import httpx
import yaml

import config

FIXTURES_DIR = config.ROOT / "tests" / "fixtures"  # saved pages for --offline builds and tests
RETRIES = 3  # DNS failures came and went during recon; three retries rode them out
BACKOFF_S = 2.0  # waits 2, 4, 8 seconds between attempts
# Signs of a bot wall instead of the page. Retrying a challenge or a rate limit makes it worse,
# so these fail at once (a plain 500/502/504 is retried).
CHALLENGE_STATUSES = {403, 429, 503}
CHALLENGE_MARKERS = ("Just a moment...", "Enable JavaScript and cookies")
CHALLENGE_SCAN_CHARS = 3000  # challenge pages say so near the top


class FetchError(Exception):
    """The page could not be fetched as real content; the build keeps the last good version."""


def load_sources() -> dict:
    return yaml.safe_load(config.SOURCES_FILE.read_text(encoding="utf-8"))


def fetch_pages(pages: list[dict], offline: bool = False,
                client: httpx.Client | None = None) -> dict[str, str]:
    """slug -> HTML for each page. offline=True reads tests/fixtures instead of the network."""
    allowed = {page["url"] for page in load_sources()["pages"]}
    for page in pages:
        if page["url"] not in allowed:
            raise FetchError(f"{page['url']} is not listed in sources.yaml")
    if offline:
        return {p["slug"]: (FIXTURES_DIR / p["fixture"]).read_text(encoding="utf-8") for p in pages}
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=config.FETCH_TIMEOUT_S, follow_redirects=True)
    try:
        return _fetch_all(client, pages)
    finally:
        if own_client:
            client.close()


def _fetch_all(client: httpx.Client, pages: list[dict]) -> dict[str, str]:
    robots = {}  # robots.txt per site, read once
    html = {}
    for page in pages:
        url = page["url"]
        site = urlsplit(url).netloc
        if site not in robots:
            robots[site] = _read_robots(client, url)
        if not robots[site].can_fetch(config.USER_AGENT, url):
            raise FetchError(f"robots.txt disallows {url}")
        crawl_delay = robots[site].crawl_delay(config.USER_AGENT) or 0
        time.sleep(max(config.MIN_DELAY_S, float(crawl_delay)))
        response = _get(client, url)
        if response.status_code != 200:
            raise FetchError(f"{url}: HTTP {response.status_code}")
        html[page["slug"]] = response.text
    return html


def _read_robots(client: httpx.Client, url: str) -> urllib.robotparser.RobotFileParser:
    """Fetched with our own User-Agent: RobotFileParser.read() would send Python's default
    one, and it treats a 401/403 (often a bot wall) as "disallow everything"."""
    robots_url = urlsplit(url)._replace(path="/robots.txt", query="", fragment="").geturl()
    response = _get(client, robots_url)
    parser = urllib.robotparser.RobotFileParser(robots_url)
    if response.status_code == 404:
        parser.parse([])  # no robots.txt means no rules
    elif response.status_code == 200:
        parser.parse(response.text.splitlines())
    else:
        raise FetchError(f"{robots_url}: HTTP {response.status_code}, cannot check permissions")
    return parser


def _get(client: httpx.Client, url: str) -> httpx.Response:
    """GET with retries for network errors and 5xx; raises FetchError on a bot challenge."""
    for attempt in range(RETRIES + 1):
        try:
            response = client.get(url, headers={"User-Agent": config.USER_AGENT})
        except httpx.TransportError as exc:  # DNS failure, timeout, connection reset
            problem = f"network error: {exc}"
        else:
            if _is_challenge(response):
                raise FetchError(f"{url}: bot challenge (HTTP {response.status_code})")
            if response.status_code < 500:
                return response
            problem = f"HTTP {response.status_code}"
        if attempt < RETRIES:
            time.sleep(BACKOFF_S * 2**attempt)
    raise FetchError(f"{url}: {problem} after {RETRIES + 1} attempts")


def _is_challenge(response: httpx.Response) -> bool:
    if response.status_code in CHALLENGE_STATUSES:
        return True
    if response.headers.get("cf-mitigated", "").lower() == "challenge":
        return True
    head = response.text[:CHALLENGE_SCAN_CHARS]
    return any(marker in head for marker in CHALLENGE_MARKERS)
