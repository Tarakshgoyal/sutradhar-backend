"""India Government Data Scrapers.

Covers:
  - PIB (Press Information Bureau)   — RSS + web
  - MEA (Ministry of External Affairs) — press releases
  - Lok Sabha / Rajya Sabha           — parliamentary proceedings RSS
  - DRDO                              — defense R&D press releases
  - data.gov.in                       — Open Government Data (OGD) catalog

Each scraper wraps a CircuitBreaker to prevent silent failures.
Instead of failing silently, sources emit intelligence-gap signals.
"""

import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
from rss_parser import RSSParser
from requests.exceptions import SSLError, HTTPError

from scrapers.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

_UTF8_BOM = b'\xef\xbb\xbf'

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _parse_rss(resp: requests.Response):
    """
    Parse an RSS feed response correctly.

    RSSParser expects bytes. Strip BOM if present before passing.
    """
    content = (resp.content or b"").lstrip()
    if content.startswith(_UTF8_BOM):
        content = content[len(_UTF8_BOM):]

    content_type = (resp.headers.get("Content-Type", "") or "").lower()
    sniff = content[:240].lstrip().lower()
    if "json" in content_type or sniff.startswith(b"{") or sniff.startswith(b"["):
        raise ValueError("received JSON instead of RSS/XML")
    if "html" in content_type or sniff.startswith(b"<!doctype html") or sniff.startswith(b"<html"):
        raise ValueError("received HTML instead of RSS/XML")

    try:
        return RSSParser.parse(content)
    except Exception:
        text_payload = content.decode(resp.encoding or "utf-8", errors="replace")
        return RSSParser.parse(text_payload)

_IST = timezone(timedelta(hours=5, minutes=30))


# ── Base data structure ───────────────────────────────────────────────────────

@dataclass
class GovDocument:
    """A document retrieved from an Indian government source."""
    url: str
    title: str
    source: str          # e.g. "PIB", "MEA", "Lok Sabha"
    source_type: str     # "rss" | "api" | "scrape"
    ministry: str = ""   # e.g. "Ministry of Defence"
    language: str = "en"
    pub_date: str = ""
    description: str = ""
    full_text: str = ""
    metadata: dict = field(default_factory=dict)
    fetched_at: Optional[datetime] = None

    def to_article_dict(self) -> dict:
        """Converts to the same dict format as scrapers.news_rss.Article.to_dict()."""
        return {
            "url": self.url,
            "title": self.title,
            "source": self.source,
            "description": self.description,
            "pub_date": self.pub_date,
            "guid": self.url,
            "full_text": self.full_text or self.description,
            "authors": [self.ministry] if self.ministry else [],
            "top_image": "",
            "images": [],
            "publish_date": self.pub_date,
            "content_hash": "",
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "is_content_extracted": bool(self.full_text),
        }


# ── Base scraper ──────────────────────────────────────────────────────────────

class BaseGovScraper(ABC):
    """Abstract base for Indian government data scrapers."""

    def __init__(
        self,
        source_name: str,
        circuit_breaker: Optional[CircuitBreaker] = None,
        request_timeout: int = 15,
    ):
        self.source_name = source_name
        self.cb = circuit_breaker
        self.timeout = request_timeout

    def _get(self, url: str, _allow_insecure_retry: bool = True, **kwargs) -> Optional[requests.Response]:
        """HTTP GET with circuit breaker guard."""
        if self.cb and not self.cb.can_request():
            logger.warning(
                f"{self.source_name}: circuit OPEN — skipping {url[:80]}"
            )
            return None

        req_headers = dict(_DEFAULT_HEADERS)
        if "headers" in kwargs and kwargs["headers"]:
            req_headers.update(kwargs.pop("headers"))

        try:
            resp = requests.get(url, timeout=self.timeout, headers=req_headers, **kwargs)
            resp.raise_for_status()
            if self.cb:
                self.cb.record_success()
            return resp
        except SSLError as e:
            # Some gov sites intermittently serve expired cert chains.
            if _allow_insecure_retry:
                logger.warning(
                    f"{self.source_name}: SSL verify failed for {url[:80]}; retrying once with verify=False"
                )
                return self._get(url, _allow_insecure_retry=False, verify=False)
            logger.error(f"{self.source_name}: request failed {url[:80]}: {e}")
            if self.cb:
                self.cb.record_failure()
            return None
        except HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            # 404/410 on deprecated feeds are persistent endpoint issues, not transient outages.
            if status in (404, 410):
                logger.warning(
                    f"{self.source_name}: feed unavailable ({status}) {url[:80]}"
                )
                return None
            logger.error(f"{self.source_name}: request failed {url[:80]}: {e}")
            if self.cb:
                self.cb.record_failure()
            return None
        except Exception as e:
            logger.error(f"{self.source_name}: request failed {url[:80]}: {e}")
            if self.cb:
                self.cb.record_failure()
            return None

    @abstractmethod
    def fetch(self) -> list[GovDocument]:
        """Fetch and return documents from this source."""
        ...

    def circuit_status(self) -> Optional[dict]:
        return self.cb.status() if self.cb else None


# ── PIB ───────────────────────────────────────────────────────────────────────

class PIBScraper(BaseGovScraper):
    """Press Information Bureau RSS scraper.

    PIB publishes press releases from all Union Ministries.
    RSS feed: https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3
    """

    PIB_FEEDS: list[tuple[str, str]] = [
        # Only ModId=6 reliably returns RSS — all other ModIds redirect to error HTML
        ("PIB Top Stories", "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3"),
    ]

    def __init__(self, circuit_breaker: Optional[CircuitBreaker] = None):
        super().__init__("PIB", circuit_breaker)

    def fetch(self) -> list[GovDocument]:
        docs: list[GovDocument] = []
        for feed_name, feed_url in self.PIB_FEEDS:
            resp = self._get(feed_url)
            if not resp:
                continue
            try:
                rss = _parse_rss(resp)
                for item in rss.channel.items:
                    url = item.links[0].content if item.links else None
                    if not url:
                        continue
                    docs.append(GovDocument(
                        url=url,
                        title=item.title.content if item.title else "",
                        source="PIB",
                        source_type="rss",
                        ministry=feed_name.replace("PIB ", ""),
                        pub_date=item.pub_date.content if hasattr(item, "pub_date") and item.pub_date else "",
                        description=item.description.content if item.description else "",
                        fetched_at=datetime.now(_IST),
                    ))
            except Exception as e:
                content_type = resp.headers.get("Content-Type", "") if resp else ""
                if "received HTML instead of RSS/XML" in str(e):
                    logger.warning(
                        f"PIB feed unavailable as RSS ({feed_url}) "
                        f"[content-type={content_type}, final_url={resp.url}]"
                    )
                else:
                    logger.error(f"PIB feed parse failed ({feed_url}): {e}")
        logger.info(f"PIB: fetched {len(docs)} documents")
        return docs


# ── MEA ───────────────────────────────────────────────────────────────────────

class MEAScraper(BaseGovScraper):
    """Ministry of External Affairs press releases RSS."""

    MEA_FEEDS: list[tuple[str, str]] = [
        # MEA no longer serves RSS or a stable JSON API.
        # Their press releases are accessible via this search API endpoint:
        ("MEA Press Releases", "https://www.mea.gov.in/press-releases.htm"),
        ("MEA Speeches",       "https://www.mea.gov.in/Speeches-Statements.htm"),
    ]

    def __init__(self, circuit_breaker: Optional[CircuitBreaker] = None):
        super().__init__("MEA", circuit_breaker)

    def fetch(self) -> list[GovDocument]:
        import re as _re
        docs: list[GovDocument] = []
        for feed_name, feed_url in self.MEA_FEEDS:
            resp = self._get(feed_url)
            if not resp:
                continue
            try:
                html = resp.text
                # Extract links that look like MEA press release detail pages
                links = _re.findall(r'href=["\']([^"\']*(?:pressrelease|speech|statement)[^"\']*)["\']', html, _re.I)
                seen_urls: set[str] = set()
                for href in links[:30]:
                    full_url = href if href.startswith("http") else f"https://www.mea.gov.in{href}"
                    if full_url in seen_urls:
                        continue
                    seen_urls.add(full_url)
                    # Extract title from nearby text (best-effort)
                    title = full_url.split("/")[-1].replace("-", " ").replace("_", " ").strip()
                    docs.append(GovDocument(
                        url=full_url,
                        title=title[:200],
                        source="MEA",
                        source_type="scrape",
                        ministry="Ministry of External Affairs",
                        fetched_at=datetime.now(_IST),
                    ))
            except Exception as e:
                logger.warning(f"MEA fetch failed ({feed_url}): {e}")
        logger.info(f"MEA: fetched {len(docs)} documents")
        return docs


# ── Parliament ────────────────────────────────────────────────────────────────

class ParliamentScraper(BaseGovScraper):
    """Lok Sabha and Rajya Sabha proceedings RSS."""

    PARLIAMENT_FEEDS: list[tuple[str, str]] = [
        # loksabha.nic.in / rajyasabha.nic.in DNS fails consistently — disabled
        # ("Lok Sabha Q&A", "https://loksabha.nic.in/rss/questionlist.aspx"),
        # ("Rajya Sabha",   "https://rajyasabha.nic.in/rss/questionlist.aspx"),
    ]

    def __init__(self, circuit_breaker: Optional[CircuitBreaker] = None):
        super().__init__("Parliament", circuit_breaker)

    def fetch(self) -> list[GovDocument]:
        docs: list[GovDocument] = []
        for feed_name, feed_url in self.PARLIAMENT_FEEDS:
            resp = self._get(feed_url)
            if not resp:
                continue
            try:
                rss = _parse_rss(resp)
                for item in rss.channel.items:
                    url = item.links[0].content if item.links else None
                    if not url:
                        continue
                    docs.append(GovDocument(
                        url=url,
                        title=item.title.content if item.title else "",
                        source="Parliament",
                        source_type="rss",
                        ministry=feed_name,
                        pub_date=item.pub_date.content if hasattr(item, "pub_date") and item.pub_date else "",
                        description=item.description.content if item.description else "",
                        fetched_at=datetime.now(_IST),
                    ))
            except Exception as e:
                content_type = resp.headers.get("Content-Type", "") if resp else ""
                if "received HTML instead of RSS/XML" in str(e):
                    logger.warning(
                        f"Parliament feed unavailable as RSS ({feed_url}) "
                        f"[content-type={content_type}, final_url={resp.url}]"
                    )
                else:
                    logger.error(f"Parliament RSS parse failed ({feed_url}): {e}")
        logger.info(f"Parliament: fetched {len(docs)} documents")
        return docs


# ── DRDO ──────────────────────────────────────────────────────────────────────

class DRDOScraper(BaseGovScraper):
    """DRDO (Defence Research and Development Organisation) press releases."""

    DRDO_FEEDS: list[tuple[str, str]] = [
        # DRDO RSS paths return 404 — their site no longer serves these feeds
        # ("DRDO News",       "https://www.drdo.gov.in/rss-feeds/latest-news"),
        # ("DRDO Technologies","https://www.drdo.gov.in/rss-feeds/technology-transfer"),
    ]

    def __init__(self, circuit_breaker: Optional[CircuitBreaker] = None):
        super().__init__("DRDO", circuit_breaker)

    def fetch(self) -> list[GovDocument]:
        docs: list[GovDocument] = []
        for feed_name, feed_url in self.DRDO_FEEDS:
            resp = self._get(feed_url)
            if not resp:
                continue
            try:
                rss = _parse_rss(resp)
                for item in rss.channel.items:
                    url = item.links[0].content if item.links else None
                    if not url:
                        continue
                    docs.append(GovDocument(
                        url=url,
                        title=item.title.content if item.title else "",
                        source="DRDO",
                        source_type="rss",
                        ministry="Ministry of Defence — DRDO",
                        pub_date=item.pub_date.content if hasattr(item, "pub_date") and item.pub_date else "",
                        description=item.description.content if item.description else "",
                        fetched_at=datetime.now(_IST),
                    ))
            except Exception as e:
                content_type = resp.headers.get("Content-Type", "") if resp else ""
                if "received HTML instead of RSS/XML" in str(e):
                    logger.warning(
                        f"DRDO feed unavailable as RSS ({feed_url}) "
                        f"[content-type={content_type}, final_url={resp.url}]"
                    )
                else:
                    logger.error(f"DRDO RSS parse failed ({feed_url}): {e}")
        logger.info(f"DRDO: fetched {len(docs)} documents")
        return docs


# ── Aggregated India Gov Producer ─────────────────────────────────────────────

class IndiaGovProducer:
    """Aggregates all Indian government scrapers into a single call.

    Usage:
        producer = IndiaGovProducer(redis_client=r)
        docs = producer.fetch_all()
        # docs: list[GovDocument], convert with doc.to_article_dict()
    """

    def __init__(self, redis_client=None):
        def _cb(name: str) -> Optional[CircuitBreaker]:
            return CircuitBreaker(name, redis_client) if redis_client else None

        self.scrapers: list[BaseGovScraper] = [
            PIBScraper(circuit_breaker=_cb("PIB")),
            MEAScraper(circuit_breaker=_cb("MEA")),
            ParliamentScraper(circuit_breaker=_cb("Parliament")),
            DRDOScraper(circuit_breaker=_cb("DRDO")),
        ]

    def fetch_all(self) -> list[GovDocument]:
        """Fetch from all government sources, skipping those with open circuits."""
        all_docs: list[GovDocument] = []
        for scraper in self.scrapers:
            try:
                docs = scraper.fetch()
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"{scraper.source_name}: uncaught error: {e}")
        logger.info(f"IndiaGovProducer: fetched {len(all_docs)} total documents")
        return all_docs

    def circuit_statuses(self) -> list[dict]:
        """Return circuit breaker status for all scrapers (for health/gap reporting)."""
        return [
            s.circuit_status() for s in self.scrapers if s.circuit_status()
        ]
