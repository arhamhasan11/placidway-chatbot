"""The fetcher is polite and paranoid: allowlist, robots.txt, crawl delay, retries, bot walls."""

import httpx
import pytest

import config
from ingest import fetch
from ingest.fetch import FIXTURES_DIR, FetchError, fetch_pages, load_sources

PAGES = load_sources()["pages"]
ROBOTS = (FIXTURES_DIR / "robots.txt").read_text(encoding="utf-8")
GOOD_HTML = "<html><body><main>ITC Immunity Therapy Center</main></body></html>"


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    calls = []
    monkeypatch.setattr(fetch.time, "sleep", calls.append)
    return calls


def mock_client(page_responses: list, robots: httpx.Response | None = None):
    """A client whose page requests get `page_responses` in order; `requests` records every call."""
    requests = []
    queue = list(page_responses)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/robots.txt":
            return robots or httpx.Response(200, text=ROBOTS)
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    return httpx.Client(transport=httpx.MockTransport(handler)), requests


def test_offline_reads_fixtures_without_network(sleeps):
    html = fetch_pages(PAGES, offline=True)
    assert list(html) == [p["slug"] for p in PAGES]
    assert "ITC Immunity Therapy Center" in html["itc-profile"]
    assert sleeps == []


def test_refuses_urls_not_in_sources(sleeps):
    client, requests = mock_client([])
    stranger = {"slug": "x", "url": "https://www.placidway.com/other", "fixture": "1.html"}
    with pytest.raises(FetchError, match="not listed in sources.yaml"):
        fetch_pages([stranger], client=client)
    assert requests == []


def test_reads_robots_with_our_user_agent_and_waits_the_crawl_delay(sleeps):
    client, requests = mock_client([httpx.Response(200, text=GOOD_HTML)] * 2)
    html = fetch_pages(PAGES[:2], client=client)
    assert list(html) == [PAGES[0]["slug"], PAGES[1]["slug"]]
    assert [r.url.path for r in requests][0] == "/robots.txt"  # read once for the site
    assert len(requests) == 3
    assert all(r.headers["User-Agent"] == config.USER_AGENT for r in requests)
    assert sleeps == [10.0, 10.0]  # robots.txt Crawl-delay beats MIN_DELAY_S


def test_all_sources_are_allowed_by_the_saved_robots_txt(sleeps):
    client, _ = mock_client([httpx.Response(200, text=GOOD_HTML)] * len(PAGES))
    assert len(fetch_pages(PAGES, client=client)) == len(PAGES)


def test_disallowed_url_is_refused(sleeps):
    robots = httpx.Response(200, text="User-agent: *\nDisallow: /package/\n")
    client, requests = mock_client([httpx.Response(200, text=GOOD_HTML)], robots=robots)
    with pytest.raises(FetchError, match="robots.txt disallows"):
        fetch_pages(PAGES[:2], client=client)  # page 1 is a /profile/, page 2 a /package/
    assert sleeps == [config.MIN_DELAY_S]  # no Crawl-delay in this robots.txt


def test_missing_robots_txt_means_no_rules(sleeps):
    client, _ = mock_client([httpx.Response(200, text=GOOD_HTML)], robots=httpx.Response(404))
    assert fetch_pages(PAGES[:1], client=client)


@pytest.mark.parametrize("response", [
    httpx.Response(403, text="Forbidden"),
    httpx.Response(429, text="Slow down"),
    httpx.Response(503, text="<title>Just a moment...</title>"),
    httpx.Response(200, text=GOOD_HTML, headers={"cf-mitigated": "challenge"}),
    httpx.Response(200, text="<html>Enable JavaScript and cookies to continue</html>"),
])
def test_bot_challenges_fail_at_once(sleeps, response):
    client, requests = mock_client([response])
    with pytest.raises(FetchError, match="bot challenge"):
        fetch_pages(PAGES[:1], client=client)
    assert len(requests) == 2  # robots.txt + one try: no retries against a bot wall


def test_network_errors_and_5xx_are_retried(sleeps):
    client, _ = mock_client([httpx.ConnectError("DNS failure"), httpx.Response(502),
                             httpx.Response(200, text=GOOD_HTML)])
    assert fetch_pages(PAGES[:1], client=client)["itc-profile"] == GOOD_HTML
    assert sleeps == [10.0, 2.0, 4.0]  # crawl delay, then backoff before each retry


def test_gives_up_after_the_last_retry(sleeps):
    client, requests = mock_client([httpx.Response(500)] * (fetch.RETRIES + 1))
    with pytest.raises(FetchError, match="HTTP 500 after 4 attempts"):
        fetch_pages(PAGES[:1], client=client)
    assert len(requests) == 1 + fetch.RETRIES + 1


def test_other_client_errors_are_not_content(sleeps):
    client, _ = mock_client([httpx.Response(404, text="Not found")])
    with pytest.raises(FetchError, match="HTTP 404"):
        fetch_pages(PAGES[:1], client=client)
