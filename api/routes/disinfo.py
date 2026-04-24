"""Disinformation intelligence API routes.

Endpoints:
    GET  /api/disinfo/signals     — Active disinformation signals
    GET  /api/disinfo/narratives  — Coordinated narrative clusters
    GET  /api/disinfo/sources     — Source credibility with flags
    POST /api/disinfo/report      — Manually report a disinformation signal
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, field_validator
from sqlalchemy import select, desc

from models.database import SessionLocal
from models.disinfo_signal import DisinfoSignal
from models.source_config import SourceConfig
from models.scraped_article import ScrapedArticle
from agents.coordination_analyzer import CoordinationAnalyzer

logger = logging.getLogger(__name__)

# Initialize coordination analyzer (lazy loaded)
_coordination_analyzer: Optional[CoordinationAnalyzer] = None

def _get_coordination_analyzer() -> CoordinationAnalyzer:
    global _coordination_analyzer
    if _coordination_analyzer is None:
        _coordination_analyzer = CoordinationAnalyzer()
    return _coordination_analyzer

VALID_SIGNAL_TYPES = [
    "coordinated_narrative", "sentiment_manipulation",
    "source_network", "targeted_operation",
]
VALID_SEVERITIES = ["high", "medium", "low"]
router = APIRouter(prefix="/api/disinfo", tags=["disinfo"])

_IST = timezone(timedelta(hours=5, minutes=30))



class ReportDisinfoRequest(BaseModel):
    signal_type: str
    severity: str = "medium"
    confidence: float
    actor_attribution: Optional[str] = None
    target_entity: Optional[str] = None
    target_domain: Optional[str] = None
    narrative_summary: Optional[str] = None
    evidence_articles: Optional[list[str]] = None
    flagged_sources: Optional[list[str]] = None
    expires_hours: Optional[int] = None  # how many hours until signal expires (default: 72h)

    @field_validator("signal_type")
    @classmethod
    def validate_signal_type(cls, v: str) -> str:
        if v not in VALID_SIGNAL_TYPES:
            raise ValueError(f"signal_type must be one of {VALID_SIGNAL_TYPES}")
        return v

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        if v not in VALID_SEVERITIES:
            raise ValueError(f"severity must be one of {VALID_SEVERITIES}")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v


class DisinfoSignalOut(BaseModel):
    id: int
    signal_type: str
    severity: str
    confidence: float
    actor_attribution: Optional[str]
    target_entity: Optional[str]
    target_domain: Optional[str]
    cluster_id: Optional[str]
    narrative_summary: Optional[str]
    coordination_score: float
    evidence_articles: Optional[list]
    flagged_sources: Optional[list]
    detected_at: str


@router.get(
    "/signals",
    response_model=list[DisinfoSignalOut],
    summary="Active disinformation signals",
)
def get_disinfo_signals(
    signal_type: Optional[str] = Query(None, description="Filter by signal_type"),
    severity: Optional[str] = Query(None, description="Filter by severity (high/medium/low)"),
    domain: Optional[str] = Query(None, description="Filter by target domain, e.g. geopolitics"),
    hours: Optional[int] = Query(None, ge=1, le=720, description="Only include signals detected in the last N hours"),
    limit: int = Query(50, ge=1, le=200),
):
    """Return active disinformation and information operation signals."""
    db = SessionLocal()
    try:
        now = datetime.now(_IST)
        query = select(DisinfoSignal).where(
            (DisinfoSignal.expires_at > now) | DisinfoSignal.expires_at.is_(None)
        )
        if hours is not None:
            cutoff = now - timedelta(hours=hours)
            query = query.where(DisinfoSignal.detected_at >= cutoff)
        if signal_type:
            query = query.where(DisinfoSignal.signal_type == signal_type)
        if severity:
            query = query.where(DisinfoSignal.severity == severity)
        if domain:
            query = query.where(DisinfoSignal.target_domain == domain)
        query = query.order_by(desc(DisinfoSignal.detected_at)).limit(limit)
        rows = db.scalars(query).all()
        return [
            DisinfoSignalOut(
                id=r.id,
                signal_type=r.signal_type,
                severity=r.severity,
                confidence=r.confidence,
                actor_attribution=r.actor_attribution,
                target_entity=r.target_entity,
                target_domain=r.target_domain,
                cluster_id=r.cluster_id,
                narrative_summary=r.narrative_summary,
                coordination_score=r.coordination_score,
                evidence_articles=r.evidence_articles,
                flagged_sources=r.flagged_sources,
                detected_at=r.detected_at.isoformat() if r.detected_at else "",
            )
            for r in rows
        ]
    finally:
        db.close()


@router.get(
    "/narratives",
    summary="Coordinated narrative clusters",
)
def get_coordinated_narratives(
    limit: int = Query(20, ge=1, le=100),
):
    """Return coordinated narrative clusters with article counts and sources."""
    db = SessionLocal()
    try:
        now = datetime.now(_IST)
        rows = db.scalars(
            select(DisinfoSignal).where(
                DisinfoSignal.signal_type == "coordinated_narrative",
                (DisinfoSignal.expires_at > now) | DisinfoSignal.expires_at.is_(None),
            ).order_by(desc(DisinfoSignal.coordination_score)).limit(limit)
        ).all()
        return [
            {
                "cluster_id": r.cluster_id,
                "narrative": r.narrative_summary,
                "coordination_score": r.coordination_score,
                "confidence": r.confidence,
                "flagged_sources": r.flagged_sources or [],
                "article_count": len(r.evidence_articles) if r.evidence_articles else 0,
                "target_domain": r.target_domain,
                "detected_at": r.detected_at.isoformat() if r.detected_at else "",
            }
            for r in rows
        ]
    finally:
        db.close()


@router.get(
    "/sources",
    summary="Source credibility with disinformation flags",
)
def get_source_credibility(
    flagged_only: bool = Query(False, description="Return only sources flagged for coordination"),
    limit: int = Query(50, ge=1, le=200),
):
    """Return news source credibility scores with any disinformation flags."""
    db = SessionLocal()
    try:
        # Flagged sources from recent disinfo signals
        flagged_sources: dict[str, int] = {}
        now = datetime.now(_IST)
        signals = db.scalars(
            select(DisinfoSignal).where(
                (DisinfoSignal.expires_at > now) | DisinfoSignal.expires_at.is_(None)
            )
        ).all()
        for s in signals:
            for src in (s.flagged_sources or []):
                flagged_sources[src] = flagged_sources.get(src, 0) + 1

        sources = db.scalars(select(SourceConfig).limit(limit)).all()
        result = []
        for src in sources:
            flag_count = flagged_sources.get(src.source_name, 0)
            if flagged_only and flag_count == 0:
                continue
            result.append({
                "source_name": src.source_name,
                "credibility_score": src.credibility_score,
                "disinfo_flag_count": flag_count,
                "is_flagged": flag_count > 0,
            })

        return sorted(result, key=lambda x: x["disinfo_flag_count"], reverse=True)
    finally:
        db.close()


@router.post(
    "/report",
    response_model=DisinfoSignalOut,
    status_code=201,
    summary="Manually report a disinformation signal",
)
def report_disinfo_signal(request: ReportDisinfoRequest):
    """Persist a manually-reported disinformation signal.

    Used by analysts to flag coordinated operations that the automated
    detector may have missed. The signal will appear immediately in
    GET /api/disinfo/signals.

    Signal types:
    - **coordinated_narrative** — Multiple sources pushing the same narrative in a short window
    - **sentiment_manipulation** — Asymmetric negative framing of a specific entity
    - **source_network** — Suspected network of low-credibility sources amplifying each other
    - **targeted_operation** — Suspected information operation targeting a specific entity or event
    """
    now = datetime.now(_IST)
    expires_at = (
        now + timedelta(hours=request.expires_hours)
        if request.expires_hours
        else now + timedelta(hours=72)  # default 72h TTL
    )

    db = SessionLocal()
    try:
        # Calculate coordination score if evidence articles are provided
        coordination_score = request.confidence  # Default to confidence

        if request.evidence_articles and len(request.evidence_articles) >= 2:
            # Fetch article data from database
            article_urls = request.evidence_articles[:50]  # Limit to 50 articles
            articles_data = db.execute(
                select(ScrapedArticle).where(ScrapedArticle.url.in_(article_urls))
            ).scalars().all()

            if len(articles_data) >= 2:
                # Calculate real coordination score
                analyzer = _get_coordination_analyzer()
                articles_for_analysis = [
                    {
                        "title": a.title,
                        "description": a.description,
                        "source": a.source,
                        "pub_date": a.pub_date,
                        "url": a.url,
                    }
                    for a in articles_data
                ]
                coordination_score = analyzer.analyze_coordination(
                    articles_for_analysis,
                    time_window_hours=24
                )
                logger.info(
                    f"Calculated coordination score: {coordination_score} "
                    f"from {len(articles_for_analysis)} articles"
                )

        signal = DisinfoSignal(
            signal_type=request.signal_type,
            severity=request.severity,
            confidence=request.confidence,
            actor_attribution=request.actor_attribution,
            target_entity=request.target_entity,
            target_domain=request.target_domain,
            narrative_summary=request.narrative_summary,
            coordination_score=coordination_score,  # Real coordination score or confidence fallback
            evidence_articles=request.evidence_articles or [],
            flagged_sources=request.flagged_sources or [],
            detected_at=now,
            expires_at=expires_at,
        )
        db.add(signal)
        db.commit()
        db.refresh(signal)
        logger.info(f"Manual disinfo signal reported: {signal.signal_type} → {signal.target_entity}")
        return DisinfoSignalOut(
            id=signal.id,
            signal_type=signal.signal_type,
            severity=signal.severity,
            confidence=signal.confidence,
            actor_attribution=signal.actor_attribution,
            target_entity=signal.target_entity,
            target_domain=signal.target_domain,
            cluster_id=signal.cluster_id,
            narrative_summary=signal.narrative_summary,
            coordination_score=signal.coordination_score,
            evidence_articles=signal.evidence_articles,
            flagged_sources=signal.flagged_sources,
            detected_at=signal.detected_at.isoformat() if signal.detected_at else "",
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save disinfo signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
