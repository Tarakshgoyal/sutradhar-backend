"""
News RSS Scraper

A reusable scraper that takes any RSS feed URL, fetches the latest articles,
and extracts full content using newspaper3k.
"""

import logging
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime

import requests
import newspaper
from rss_parser import RSSParser


logger = logging.getLogger(__name__)

_UTF8_BOM = b"\xef\xbb\xbf"


def _payload_preview(raw: bytes, limit: int = 140) -> str:
    """Return a short, log-safe preview of a response payload."""
    if not raw:
        return ""
    snippet = raw[:limit].decode("utf-8", errors="replace")
    return re.sub(r"\s+", " ", snippet).strip()


def _looks_like_non_rss(raw: bytes, content_type: str) -> bool:
    """Detect obvious non-RSS payloads (HTML/JSON/challenge pages)."""
    ct = (content_type or "").lower()
    sniff = raw[:240].lstrip().lower()
    if "json" in ct or sniff.startswith(b"{") or sniff.startswith(b"["):
        return True
    if "html" in ct or sniff.startswith(b"<!doctype html") or sniff.startswith(b"<html"):
        return True
    return False


def _parse_rss_response(response: requests.Response):
    """Parse RSS response robustly across bytes/text parser behaviors."""
    raw = response.content or b""
    if raw.startswith(_UTF8_BOM):
        raw = raw[len(_UTF8_BOM):]
    raw = raw.lstrip()

    content_type = response.headers.get("Content-Type", "")
    if _looks_like_non_rss(raw, content_type):
        raise ValueError(
            f"non-RSS payload (content-type={content_type!r}, preview={_payload_preview(raw)!r})"
        )

    # Most versions of rss_parser accept bytes.
    try:
        return RSSParser.parse(raw)
    except Exception:
        # Some environments/parsers behave better with text.
        text_payload = raw.decode(response.encoding or "utf-8", errors="replace")
        return RSSParser.parse(text_payload)


def _parse_pub_date(date_str: str) -> Optional[datetime]:
    """Parse an RSS pub_date string into a timezone-aware datetime. Returns None on failure."""
    if not date_str:
        return None
    # Try RFC 2822 (standard RSS format: "Thu, 27 Mar 2026 10:00:00 +0000")
    try:
        dt = parsedate_to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass
    # Fallback: ISO 8601 / other common formats via dateutil
    try:
        from dateutil import parser as dateutil_parser
        dt = dateutil_parser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass
    return None


def _extract_next_page_url(response_text: str) -> Optional[str]:
    """
    Extract the RFC 5005 'next' page URL from raw RSS/Atom XML.
    Handles both <atom:link rel="next" href="..."/> and <link rel="next" href="..."/>.
    """
    patterns = [
        r'<atom:link[^>]+rel=["\']next["\'][^>]+href=["\']([^"\']+)["\']',
        r'<atom:link[^>]+href=["\']([^"\']+)["\'][^>]+rel=["\']next["\']',
        r'<link[^>]+rel=["\']next["\'][^>]+href=["\']([^"\']+)["\']',
        r'<link[^>]+href=["\']([^"\']+)["\'][^>]+rel=["\']next["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


@dataclass
class Article:
    """Represents a fully extracted news article."""

    url: str
    title: str
    source: str  # e.g. "NDTV", "BBC", "CNN"
    description: str = ""
    pub_date: str = ""
    guid: str = ""

    # Fields populated after full content extraction
    full_text: str = ""
    authors: list[str] = field(default_factory=list)
    top_image: str = ""
    images: list[str] = field(default_factory=list)
    publish_date: Optional[datetime] = None

    # Metadata
    content_hash: str = ""
    fetched_at: Optional[datetime] = None
    is_content_extracted: bool = False

    def __post_init__(self):
        if not self.content_hash and self.url:
            self.content_hash = hashlib.sha256(self.url.encode()).hexdigest()[:16]

    # Domains that block newspaper3k — skip content extraction entirely
    _BLOCKED_DOMAINS = {
        "yourstory.com", "washingtonpost.com", "wsj.com",
        "nytimes.com", "ft.com", "bloomberg.com",
        "foreignaffairs.com", "techcrunch.com",
        "warontherocks.com", "brookings.edu",
    }

    def extract_content(self, timeout: int = 10) -> bool:
        """
        Download and parse the full article content using newspaper3k.

        Returns True if extraction succeeded, False otherwise.
        """
        # Skip domains that consistently block scrapers (saves time & log noise)
        try:
            from urllib.parse import urlparse
            domain = urlparse(self.url).netloc.lower().lstrip("www.")
            if any(domain.endswith(blocked) for blocked in self._BLOCKED_DOMAINS):
                self.is_content_extracted = False
                return False
        except Exception:
            pass

        try:
            article = newspaper.Article(self.url, request_timeout=timeout)
            article.download()
            article.parse()

            text = article.text or ""
            # Reject paywalled/empty extractions
            if len(text.strip()) < 80:
                self.is_content_extracted = False
                return False

            self.full_text = text
            self.authors = article.authors
            self.top_image = article.top_image or ""
            self.images = list(article.images) if article.images else []
            self.publish_date = article.publish_date
            self.is_content_extracted = True
            self.fetched_at = datetime.now()

            # Update title from article if RSS title was empty
            if not self.title and article.title:
                self.title = article.title

            logger.info(f"Extracted content for: {self.title} ({len(self.full_text)} chars)")
            return True

        except Exception as e:
            err = str(e)
            # Suppress noisy known-failure messages; log others as warning not error
            if "403" in err or "download()" in err or "Forbidden" in err:
                logger.debug(f"Skipped (blocked): {self.url}")
            else:
                logger.warning(f"Failed to extract content from {self.url}: {e}")
            self.is_content_extracted = False
            return False

    def to_dict(self) -> dict:
        """Serialize article to a dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "source": self.source,
            "description": self.description,
            "pub_date": self.pub_date,
            "guid": self.guid,
            "full_text": self.full_text,
            "authors": self.authors,
            "top_image": self.top_image,
            "images": self.images,
            "publish_date": self.publish_date.isoformat() if self.publish_date else None,
            "content_hash": self.content_hash,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "is_content_extracted": self.is_content_extracted,
        }


class NewsRSSScraper:
    """
    Scrapes news articles from RSS feeds.

    Usage:
        scraper = NewsRSSScraper()

        # Add feeds
        scraper.add_feed("NDTV", "https://feeds.feedburner.com/ndtvnews-top-stories")
        scraper.add_feed("BBC", "https://feeds.bbci.co.uk/news/rss.xml")

        # Fetch latest articles (metadata only, from RSS)
        articles = scraper.fetch_all()

        # Extract full content for all articles
        scraper.extract_all_content(articles)

        # Or do both in one call
        articles = scraper.fetch_and_extract_all()
    """

    def __init__(self, request_timeout: int = 15, dedup_titles: bool = True):
        self._feeds: list[tuple[str, str]] = []  # [(source_name, rss_url), ...]
        self._seen_urls: set[str] = set()
        self._seen_title_keys: set[str] = set()  # normalized title keys for cross-source dedup
        self._dedup_titles = dedup_titles
        self._request_timeout = request_timeout
        # Headers to bypass bot detection
        self._headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml, */*',
            'Accept-Language': 'en-US,en;q=0.9',
        }

    @property
    def feeds(self) -> list[tuple[str, str]]:
        """Return registered feeds."""
        return list(self._feeds)

    @property
    def seen_urls(self) -> set[str]:
        """Return set of already-processed URLs."""
        return set(self._seen_urls)

    def add_feed(self, source_name: str, rss_url: str) -> None:
        """Register an RSS feed source."""
        self._feeds.append((source_name, rss_url))
        logger.info(f"Added feed: {source_name} -> {rss_url}")

    def remove_feed(self, source_name: str) -> None:
        """Remove all feeds for a given source name."""
        self._feeds = [(s, u) for s, u in self._feeds if s != source_name]
        logger.info(f"Removed feeds for: {source_name}")

    def mark_seen(self, urls: set[str] | list[str]) -> None:
        """Mark URLs as already processed (won't be returned by fetch)."""
        self._seen_urls.update(urls)

    def clear_seen(self) -> None:
        """Clear the set of seen URLs."""
        self._seen_urls.clear()
        self._seen_title_keys.clear()

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Normalize title for dedup: lowercase, strip punctuation, collapse whitespace."""
        import re
        t = title.lower().strip()
        t = re.sub(r'[^\w\s]', '', t)      # remove punctuation
        t = re.sub(r'\s+', ' ', t).strip()  # collapse whitespace
        # Drop very short words that vary across sources (a, an, the, etc.)
        words = [w for w in t.split() if len(w) > 2]
        # Use first 8 meaningful words as the key (headlines vary in length)
        return ' '.join(words[:8])

    # ── Fetching (RSS metadata only) ──────────────────────────────────────

    def fetch_feed(
        self,
        source_name: str,
        rss_url: str,
        *,
        include_seen: bool = False,
        max_per_feed: int = 0,
        cutoff_date: Optional[datetime] = None,
    ) -> list[Article]:
        """
        Fetch articles from a single RSS feed, following pagination links (RFC 5005)
        until all pages within the cutoff window have been retrieved.

        Args:
            source_name: Name of the source (e.g. "BBC")
            rss_url: The RSS feed URL
            include_seen: If True, include articles that were already fetched before
            max_per_feed: Max articles to take from this feed (0 = unlimited)
            cutoff_date: Only include articles published on or after this date.
                         Pagination stops once a page contains only older articles.
        """
        all_articles: list[Article] = []
        current_url: Optional[str] = rss_url
        pages_fetched = 0
        max_pages = 50  # safety cap to prevent infinite loops

        while current_url and pages_fetched < max_pages:
            pages_fetched += 1

            try:
                response = requests.get(current_url, headers=self._headers, timeout=self._request_timeout)
                response.raise_for_status()
            except requests.RequestException as e:
                logger.error(f"Failed to fetch RSS feed from {source_name} ({current_url}): {e}")
                break

            try:
                rss = _parse_rss_response(response)
            except Exception as e:
                logger.error(
                    f"Failed to parse RSS feed from {source_name}: {e} "
                    f"(status={response.status_code}, url={response.url})"
                )
                break

            page_had_fresh_items = False

            for item in rss.channel.items:
                url = item.links[0].content if item.links else None
                if not url:
                    continue

                pub_date_str = item.pub_date.content if hasattr(item, "pub_date") and item.pub_date else ""

                # Apply date cutoff: skip articles older than the window.
                # RSS feeds are newest-first; once we hit an old article we can stop
                # fetching further pages too.
                if cutoff_date and pub_date_str:
                    article_date = _parse_pub_date(pub_date_str)
                    if article_date is not None and article_date < cutoff_date:
                        # This article is outside the window. Since feeds are sorted
                        # newest-first, everything that follows is also older — stop.
                        logger.debug(
                            f"{source_name}: article '{pub_date_str}' is before cutoff, "
                            f"stopping pagination."
                        )
                        current_url = None  # signal to exit outer loop
                        break

                page_had_fresh_items = True

                # Skip already-seen URLs unless explicitly requested
                if not include_seen and url in self._seen_urls:
                    continue

                title = item.title.content if item.title else ""

                # Cross-source title dedup
                if self._dedup_titles and title:
                    title_key = self._normalize_title(title)
                    if title_key and title_key in self._seen_title_keys:
                        logger.debug(f"Title dedup: skipping '{title[:60]}' from {source_name}")
                        continue
                    if title_key:
                        self._seen_title_keys.add(title_key)

                description = item.description.content if item.description else ""
                guid = item.guid.content if hasattr(item, "guid") and item.guid else ""

                article = Article(
                    url=url,
                    title=title,
                    source=source_name,
                    description=description,
                    pub_date=pub_date_str,
                    guid=guid,
                )
                all_articles.append(article)
                self._seen_urls.add(url)

                if max_per_feed and len(all_articles) >= max_per_feed:
                    current_url = None  # signal to exit outer loop
                    break

            # Only follow the next-page link if:
            # - we haven't been told to stop (current_url still set)
            # - the page actually contained fresh items (avoid looping on stale feeds)
            if current_url is not None:
                next_url = _extract_next_page_url(response.text)
                if next_url and next_url != current_url and page_had_fresh_items:
                    current_url = next_url
                else:
                    break  # no next page or no fresh items — done

        if pages_fetched > 1:
            logger.info(f"Fetched {len(all_articles)} articles from {source_name} ({pages_fetched} pages)")
        else:
            logger.info(f"Fetched {len(all_articles)} articles from {source_name}")
        return all_articles

    def fetch_all(
        self,
        *,
        include_seen: bool = False,
        max_per_feed: int = 0,
        days: int = 28,
    ) -> list[Article]:
        """
        Fetch articles from all registered RSS feeds.
        Returns Article objects with metadata only (no full text yet).

        Args:
            max_per_feed: Max articles per feed (0 = unlimited)
            days: Only return articles published within the last N days (0 = no filter).
                  Defaults to 28 days. Also controls how far back pagination will go.
        """
        cutoff_date: Optional[datetime] = None
        if days > 0:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        all_articles = []
        for source_name, rss_url in self._feeds:
            articles = self.fetch_feed(
                source_name,
                rss_url,
                include_seen=include_seen,
                max_per_feed=max_per_feed,
                cutoff_date=cutoff_date,
            )
            all_articles.extend(articles)

        logger.info(f"Fetched {len(all_articles)} total articles from {len(self._feeds)} feeds")
        return all_articles

    # ── Content Extraction (full article text) ────────────────────────────

    def extract_content(self, article: Article) -> Article:
        """Extract full content for a single article. Returns the same article (mutated)."""
        article.extract_content(timeout=self._request_timeout)
        return article

    def extract_all_content(
        self, articles: list[Article], *, max_workers: int = 5
    ) -> list[Article]:
        """
        Extract full content for a list of articles in parallel.

        Args:
            articles: List of Article objects to extract content for
            max_workers: Number of parallel download threads
        """
        succeeded = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(article.extract_content, self._request_timeout): article
                for article in articles
            }
            for future in as_completed(futures):
                article = futures[future]
                try:
                    result = future.result()
                    if result:
                        succeeded += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Extraction failed for {article.url}: {e}")
                    failed += 1

        logger.info(
            f"Content extraction complete: {succeeded} succeeded, {failed} failed "
            f"out of {len(articles)} articles"
        )
        return articles

    # ── Combined fetch + extract ──────────────────────────────────────────

    def fetch_and_extract_all(
        self,
        *,
        include_seen: bool = False,
        max_workers: int = 5,
        days: int = 28,
    ) -> list[Article]:
        """
        Fetch all RSS feeds and extract full content for every article.

        Args:
            include_seen: If True, re-process previously seen URLs
            max_workers: Number of parallel download threads for content extraction
            days: Only include articles published within the last N days (0 = no filter)
        """
        articles = self.fetch_all(include_seen=include_seen, days=days)
        if articles:
            self.extract_all_content(articles, max_workers=max_workers)
        return articles

    def fetch_and_extract_feed(
        self,
        source_name: str,
        rss_url: str,
        *,
        include_seen: bool = False,
        max_workers: int = 5,
    ) -> list[Article]:
        """
        Fetch a single RSS feed and extract full content for every article.
        The feed does not need to be pre-registered.
        """
        articles = self.fetch_feed(source_name, rss_url, include_seen=include_seen)
        if articles:
            self.extract_all_content(articles, max_workers=max_workers)
        return articles

    # ── Utilities ─────────────────────────────────────────────────────────

    def get_new_articles(self, articles: list[Article]) -> list[Article]:
        """Filter articles to only those with successfully extracted content."""
        return [a for a in articles if a.is_content_extracted and a.full_text]

    def summary(self, articles: list[Article]) -> dict:
        """Return a summary of the fetched articles."""
        by_source: dict[str, int] = {}
        for a in articles:
            by_source[a.source] = by_source.get(a.source, 0) + 1

        return {
            "total": len(articles),
            "with_content": sum(1 for a in articles if a.is_content_extracted),
            "by_source": by_source,
        }


# ── Convenience: pre-configured scraper with common news feeds ────────────

def create_default_scraper() -> NewsRSSScraper:
    """Create a scraper pre-loaded with common news RSS feeds."""
    scraper = NewsRSSScraper()

    # ── India — General ───────────────────────────────────────────────────────
    scraper.add_feed("NDTV", "https://feeds.feedburner.com/ndtvnews-top-stories")
    scraper.add_feed("NDTV", "https://feeds.feedburner.com/ndtvnews-india-news")
    scraper.add_feed("NDTV", "https://feeds.feedburner.com/ndtvnews-world-news")

    scraper.add_feed("India Today", "https://www.indiatoday.in/rss/1206514")   # Nation
    scraper.add_feed("India Today", "https://www.indiatoday.in/rss/1206577")   # World
    scraper.add_feed("India Today", "https://www.indiatoday.in/rss/1206550")   # Business

    scraper.add_feed("The Hindu", "https://www.thehindu.com/news/feeder/default.rss")
    scraper.add_feed("The Hindu", "https://www.thehindu.com/news/international/feeder/default.rss")
    scraper.add_feed("The Hindu", "https://www.thehindu.com/news/states/feeder/default.rss")
    scraper.add_feed("The Hindu", "https://www.thehindu.com/business/Economy/feeder/default.rss")

    scraper.add_feed("Live Mint", "https://www.livemint.com/rss/news")
    scraper.add_feed("Live Mint", "https://www.livemint.com/rss/companies")
    scraper.add_feed("Live Mint", "https://www.livemint.com/rss/money")
    scraper.add_feed("Live Mint", "https://www.livemint.com/rss/politics")
    scraper.add_feed("Live Mint", "https://www.livemint.com/rss/AI")

    scraper.add_feed("Economic Times", "https://economictimes.indiatimes.com/rssfeedstopstories.cms")
    scraper.add_feed("Economic Times", "https://economictimes.indiatimes.com/news/politics-and-nation/rssfeeds/1052732854.cms")
    scraper.add_feed("Economic Times", "https://economictimes.indiatimes.com/news/international/rssfeeds/2147477.cms")
    scraper.add_feed("Economic Times", "https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms")

    scraper.add_feed("India TV", "https://www.indiatvnews.com/rssnews/topstory.xml")
    scraper.add_feed("India TV", "https://www.indiatvnews.com/rssnews/topstory-world.xml")
    scraper.add_feed("India TV", "https://www.indiatvnews.com/rssnews/topstory-business.xml")

    scraper.add_feed("Hindustan Times", "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml")
    scraper.add_feed("Hindustan Times", "https://www.hindustantimes.com/feeds/rss/world-news/rssfeed.xml")
    scraper.add_feed("Hindustan Times", "https://www.hindustantimes.com/feeds/rss/business/rssfeed.xml")

    scraper.add_feed("Hindu Business Line", "https://www.thehindubusinessline.com/feeder/default.rss")
    scraper.add_feed("Hindu Business Line", "https://www.thehindubusinessline.com/economy/feeder/default.rss")
    scraper.add_feed("Hindu Business Line", "https://www.thehindubusinessline.com/markets/feeder/default.rss")
    scraper.add_feed("Hindu Business Line", "https://www.thehindubusinessline.com/news/world/feeder/default.rss")

    scraper.add_feed("Times Now", "https://www.timesnownews.com/feeds/gns-en-latest.xml")

    scraper.add_feed("ANI News", "https://www.aninews.in/rss/feed/category/world/general-news.xml")
    scraper.add_feed("ANI News", "https://www.aninews.in/rss/feed/category/national/general-news.xml")

    scraper.add_feed("Scroll", "https://feeds.feedburner.com/ScrollinArticles.rss")

    scraper.add_feed("The Quint", "https://www.thequint.com/feed")

    # ORF has no public RSS feed anymore — removed

    # ── International ─────────────────────────────────────────────────────────
    scraper.add_feed("Al Jazeera",   "https://www.aljazeera.com/xml/rss/all.xml")
    scraper.add_feed("The Diplomat", "https://thediplomat.com/feed/")
    scraper.add_feed("DW News",      "https://rss.dw.com/xml/rss-en-all")
    scraper.add_feed("AP News",      "https://feeds.feedburner.com/APWorldNews")

    # ── Geopolitics & Policy ──────────────────────────────────────────────────
    scraper.add_feed("Foreign Affairs",  "https://www.foreignaffairs.com/rss.xml")
    scraper.add_feed("War on the Rocks", "https://warontherocks.com/feed/")

    # ── Technology ────────────────────────────────────────────────────────────
    scraper.add_feed("TechCrunch", "https://techcrunch.com/feed/")

    # ── Climate / Energy ──────────────────────────────────────────────────────
    scraper.add_feed("Climate Home News", "https://www.climatechangenews.com/feed/")

    return scraper


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    scraper = create_default_scraper()

    # Fetch RSS metadata for all feeds
    print("Fetching RSS feeds...")
    articles = scraper.fetch_all()
    print(f"\nFound {len(articles)} articles across all feeds\n")

    # Show what we got from RSS
    for article in articles[:5]:
        print(f"[{article.source}] {article.title}")
        print(f"  URL: {article.url}")
        print(f"  Published: {article.pub_date}")
        print()

    # Extract full content in parallel
    print("Extracting full article content...")
    scraper.extract_all_content(articles, max_workers=5)

    # Show results
    print(f"\n{'='*60}")
    print(scraper.summary(articles))
    print(f"{'='*60}\n")

    # Show a sample extracted article
    extracted = scraper.get_new_articles(articles)
    if extracted:
        sample = extracted[0]
        print(f"Sample article: {sample.title}")
        print(f"Source: {sample.source}")
        print(f"Authors: {sample.authors}")
        print(f"Text preview: {sample.full_text[:500]}...")