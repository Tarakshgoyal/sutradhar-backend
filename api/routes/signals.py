"""Anomaly signals API.

Signals are pre-computed by the background signal_worker every 15 minutes.
The /refresh endpoint can trigger on-demand detection synchronously.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select, desc

from models.database import SessionLocal
from models.detected_signal import DetectedSignal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/signals")


def _signal_to_dict(s: DetectedSignal) -> dict:
    return {
        "id": s.id,
        "type": s.signal_type,
        "severity": s.severity,
        "entity_name": s.entity_name or None,
        "entity_type": s.entity_type or None,
        "cluster_id": s.cluster_id or None,
        "cluster_label": s.cluster_label or None,
        "domain": s.domain or None,
        "spike_ratio": round(s.spike_ratio, 2),
        "current_count": s.current_count,
        "baseline_count": round(s.baseline_count, 2),
        "detected_at": s.detected_at.isoformat() if s.detected_at else None,
        "expires_at": s.expires_at.isoformat() if s.expires_at else None,
    }


@router.get("")
def list_signals(
    signal_type: Optional[str] = Query(
        None, description="Filter by type: entity_spike, new_entity, topic_spike"
    ),
    severity: Optional[str] = Query(None, description="Filter by severity: high, medium"),
    domain: Optional[str] = Query(None, description="Filter by domain: geopolitics, defense, economics, technology, climate, etc."),
    limit: int = Query(50, ge=1, le=200),
):
    """Active anomaly signals, sorted by spike ratio descending.

    Only returns signals where expires_at > now().
    Signals are refreshed by the background signal_worker every 15 minutes.
    """
    now = datetime.now(timezone.utc)
    db = SessionLocal()
    try:
        stmt = (
            select(DetectedSignal)
            .where(DetectedSignal.expires_at > now)
            .order_by(desc(DetectedSignal.spike_ratio))
            .limit(limit)
        )
        if signal_type:
            stmt = stmt.where(DetectedSignal.signal_type == signal_type)
        if severity:
            stmt = stmt.where(DetectedSignal.severity == severity)
        if domain:
            stmt = stmt.where(DetectedSignal.domain == domain)

        signals = db.scalars(stmt).all()
        return {
            "count": len(signals),
            "signals": [_signal_to_dict(s) for s in signals],
        }
    finally:
        db.close()


@router.post("/refresh")
def refresh_signals():
    """Run signal detection synchronously and return the new count.

    This triggers the same detection logic used by the background worker,
    useful for on-demand refreshes when the DB has freshly ingested articles.
    """
    try:
        from scheduler.signal_worker import run_detection_cycle
        count = run_detection_cycle()
        return {"status": "ok", "signals_detected": count}
    except Exception as exc:
        logger.error(f"Signal refresh failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
