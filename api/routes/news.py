"""REST endpoints for querying scraped and prioritised news articles."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select, func, desc, or_

from models.database import SessionLocal
from models.scraped_article import ScrapedArticle

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/news")


def _article_to_dict(a: ScrapedArticle, *, full: bool = False) -> dict:
    base = {
        "id": a.id,
        "url": a.url,
        "title": a.title,
        "source": a.source,
        "description": a.description,
        "pub_date": a.pub_date,
        "top_image": a.top_image,
        "importance_score": a.importance_score,
        "domain": a.domain or "other",
        "cluster_id": a.topic_cluster_id,
        "cluster_label": a.cluster_label or "",
        "scraped_at": a.scraped_at.isoformat() if a.scraped_at else None,
    }
    if full:
        base["full_text"] = a.full_text
        base["authors"] = a.authors
    return base


# ── GET /api/news ─────────────────────────────────────────────────────────────

@router.get("")
def list_news(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    source: Optional[str] = Query(None, description="Comma-separated source names, e.g. BBC,NDTV"),
    domain: Optional[str] = Query(None, description="Comma-separated domains, e.g. geopolitics,defense"),
    min_score: Optional[float] = Query(None, ge=0.0, le=10.0),
    from_date: Optional[str] = Query(None, description="ISO date, e.g. 2026-03-01"),
    to_date: Optional[str] = Query(None, description="ISO date, e.g. 2026-03-07"),
    q: Optional[str] = Query(None, description="Search in title and description"),
):
    """Paginated, filterable list of scraped articles sorted by importance then recency."""
    db = SessionLocal()
    try:
        stmt = select(ScrapedArticle)

        if source:
            sources = [s.strip() for s in source.split(",") if s.strip()]
            stmt = stmt.where(ScrapedArticle.source.in_(sources))
        if domain:
            domains = [d.strip() for d in domain.split(",") if d.strip()]
            stmt = stmt.where(ScrapedArticle.domain.in_(domains))
        if min_score is not None:
            stmt = stmt.where(ScrapedArticle.importance_score >= min_score)
        if from_date:
            stmt = stmt.where(ScrapedArticle.scraped_at >= datetime.fromisoformat(from_date))
        if to_date:
            stmt = stmt.where(ScrapedArticle.scraped_at <= datetime.fromisoformat(to_date))
        if q:
            pattern = f"%{q}%"
            stmt = stmt.where(
                or_(
                    ScrapedArticle.title.ilike(pattern),
                    ScrapedArticle.description.ilike(pattern),
                )
            )

        total = db.scalar(select(func.count()).select_from(stmt.subquery()))
        articles = db.scalars(
            stmt
            .order_by(desc(ScrapedArticle.importance_score), desc(ScrapedArticle.scraped_at))
            .offset((page - 1) * per_page)
            .limit(per_page)
        ).all()

        return {
            "total": total,
            "page": page,
            "per_page": per_page,
            "articles": [_article_to_dict(a) for a in articles],
        }
    finally:
        db.close()


# ── GET /api/news/top ─────────────────────────────────────────────────────────

@router.get("/top")
def top_news(
    limit: int = Query(10, ge=1, le=100),
    hours: int = Query(24, ge=1, le=168),
    domain: Optional[str] = Query(None, description="Domain filter, e.g. geopolitics"),
):
    """Top N articles by importance score in the last N hours."""
    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        stmt = (
            select(ScrapedArticle)
            .where(ScrapedArticle.scraped_at >= cutoff)
            .where(ScrapedArticle.importance_score.isnot(None))
        )
        if domain:
            stmt = stmt.where(ScrapedArticle.domain == domain)

        articles = db.scalars(
            stmt
            .order_by(desc(ScrapedArticle.importance_score), desc(ScrapedArticle.scraped_at))
            .limit(limit)
        ).all()
        return {"articles": [_article_to_dict(a) for a in articles]}
    finally:
        db.close()


# ── GET /api/news/topics ──────────────────────────────────────────────────────

@router.get("/topics")
def trending_topics(
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(50, ge=1, le=200),
):
    """Trending topic clusters with human-readable labels and article counts."""
    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Aggregate per cluster
        cluster_agg = (
            select(
                ScrapedArticle.topic_cluster_id,
                func.max(ScrapedArticle.importance_score).label("max_score"),
                func.count(ScrapedArticle.id).label("article_count"),
                func.min(ScrapedArticle.scraped_at).label("first_seen"),
            )
            .where(ScrapedArticle.scraped_at >= cutoff)
            .where(ScrapedArticle.topic_cluster_id.isnot(None))
            .group_by(ScrapedArticle.topic_cluster_id)
            .subquery()
        )

        # Join to fetch representative article metadata (the one with max score)
        rep = ScrapedArticle.__table__.alias("rep")
        rows = db.execute(
            select(cluster_agg, rep)
            .join(
                rep,
                (rep.c.topic_cluster_id == cluster_agg.c.topic_cluster_id)
                & (rep.c.importance_score == cluster_agg.c.max_score),
            )
            .order_by(desc(cluster_agg.c.max_score))
            .limit(limit)
        ).mappings().all()

        topics = []
        seen: set[str] = set()
        for row in rows:
            cid = row["topic_cluster_id"]
            if cid in seen:
                continue
            seen.add(cid)
            topics.append({
                "cluster_id": cid,
                "cluster_label": row["cluster_label"] or "",
                "domain": row["domain"] or "other",
                "importance_score": row["max_score"],
                "article_count": row["article_count"],
                "first_seen": row["first_seen"].isoformat() if row["first_seen"] else None,
                "top_article": {
                    "id": row["id"],
                    "title": row["title"],
                    "source": row["source"],
                    "url": row["url"],
                    "top_image": row["top_image"],
                    "pub_date": row["pub_date"],
                },
            })

        return {"topics": topics}
    finally:
        db.close()


# ── GET /api/news/sources ─────────────────────────────────────────────────────

@router.get("/sources")
def news_sources():
    """Active news sources with article counts and average importance score."""
    db = SessionLocal()
    try:
        rows = db.execute(
            select(
                ScrapedArticle.source,
                func.count(ScrapedArticle.id).label("article_count"),
                func.avg(ScrapedArticle.importance_score).label("avg_importance"),
                func.max(ScrapedArticle.scraped_at).label("last_seen"),
            )
            .group_by(ScrapedArticle.source)
            .order_by(desc(func.count(ScrapedArticle.id)))
        ).all()

        return {
            "sources": [
                {
                    "source": row.source,
                    "article_count": row.article_count,
                    "avg_importance_score": (
                        round(float(row.avg_importance), 2) if row.avg_importance else None
                    ),
                    "last_seen": row.last_seen.isoformat() if row.last_seen else None,
                }
                for row in rows
            ]
        }
    finally:
        db.close()


# ── GET /api/news/domains ─────────────────────────────────────────────────────

@router.get("/domains")
def list_domains():
    """All distinct domains present in the DB with article counts."""
    db = SessionLocal()
    try:
        rows = db.execute(
            select(
                ScrapedArticle.domain,
                func.count(ScrapedArticle.id).label("article_count"),
            )
            .where(ScrapedArticle.domain != "")
            .group_by(ScrapedArticle.domain)
            .order_by(desc(func.count(ScrapedArticle.id)))
        ).all()
        return {
            "domains": [
                {"domain": row.domain, "article_count": row.article_count}
                for row in rows
            ]
        }
    finally:
        db.close()


# ── GET /api/news/{article_id} ────────────────────────────────────────────────

@router.get("/{article_id}")
def get_article(article_id: int):
    """Full article detail including full text."""
    db = SessionLocal()
    try:
        article = db.get(ScrapedArticle, article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        return _article_to_dict(article, full=True)
    finally:
        db.close()
