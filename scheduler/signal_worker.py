"""Signal / Anomaly Detection Worker.

Runs as a standalone background process. Detects three types of anomalies:

  A. entity_spike   — Entity mentioned significantly more than its same-hour
                      7-day baseline (Neo4j-based).
  B. new_entity     — Newly appearing entity that immediately has high
                      connectivity in the graph (Neo4j-based, no created_at scan).
  C. topic_spike    — Topic cluster receiving far more articles than its
                      same-hour 7-day baseline (Postgres-based).

Key design choices that address known failure modes:
  - Same-hour baseline (not daily avg): avoids time-of-day news-cycle bias.
  - MIN_BASELINE_MENTIONS guard: skips obscure entities with insufficient history.
  - Laplace smoothing on denominator: prevents division-by-zero and dampens
    small-number false positives.
  - Minimum current-count guard: a single article can never trigger a spike.
  - Topic spike uses 6h windows (not 2h) to smooth over scraper batch cadence.
  - Signal B uses relationship scan (no created_at property scan) to avoid
    full index-less node scans.

Run:
    python -m scheduler.signal_worker
"""

import json
import logging
import signal as os_signal
import time
from datetime import datetime, timedelta, timezone

import redis
from neo4j import GraphDatabase
from sqlalchemy.dialects.postgresql import insert as pg_insert

from agents.disinfo_detector import DisinfoDetector
from config import (
    NEO4J_URI, NEO4J_AUTH,
    REDIS_HOST, REDIS_PORT,
    SIGNAL_WORKER_INTERVAL_SECONDS,
)
from models.database import Base, engine, SessionLocal
from models.detected_signal import DetectedSignal
from models.scraped_article import ScrapedArticle
from sqlalchemy import select, func, and_

# SignalSwarm augments the statistical detectors
_SIGNAL_SWARM_AVAILABLE = False
try:
    from swarm.signal_swarm import SignalSwarm
    _SIGNAL_SWARM_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ── pub_date parser (mirrors graph.py _PARSE_DATE_CYPHER) ────────────────────
# Article nodes in Neo4j store pub_date as an RSS string ("Thu, 06 Mar 2026 ...").
# This fragment parses it to "YYYY-MM-DD" for date comparisons.
# Usage: _PARSE_PUB_DATE.format(src="a") inside a Cypher query.
_PARSE_PUB_DATE = """
    CASE
        WHEN {src}.pub_date CONTAINS ','
        THEN split(split({src}.pub_date, ', ')[1], ' ')[2] + '-' +
             CASE split(split({src}.pub_date, ', ')[1], ' ')[1]
                 WHEN 'Jan' THEN '01' WHEN 'Feb' THEN '02' WHEN 'Mar' THEN '03'
                 WHEN 'Apr' THEN '04' WHEN 'May' THEN '05' WHEN 'Jun' THEN '06'
                 WHEN 'Jul' THEN '07' WHEN 'Aug' THEN '08' WHEN 'Sep' THEN '09'
                 WHEN 'Oct' THEN '10' WHEN 'Nov' THEN '11' WHEN 'Dec' THEN '12'
                 ELSE '01' END + '-' +
             CASE WHEN size(split(split({src}.pub_date, ', ')[1], ' ')[0]) = 1
                  THEN '0' + split(split({src}.pub_date, ', ')[1], ' ')[0]
                  ELSE split(split({src}.pub_date, ', ')[1], ' ')[0]
             END
        ELSE substring({src}.pub_date, 0, 10)
    END
"""

# ── Tuning constants ──────────────────────────────────────────────────────────

# Signal A / B: entity mention thresholds
ENTITY_SPIKE_WINDOW_HOURS = 2        # current window size
ENTITY_SPIKE_RATIO_THRESHOLD = 3.0   # (smoothed) ratio to call it a spike
ENTITY_MIN_BASELINE_MENTIONS = 2     # skip entities with thin history
ENTITY_MIN_CURRENT_MENTIONS = 1      # single article never fires
LAPLACE_SMOOTH = 1.0                 # denominator smoothing

# Signal B: new entity connectivity
NEW_ENTITY_LOOKBACK_HOURS = 24       # entity is "new" if only seen within this window
NEW_ENTITY_MIN_DEGREE = 5            # min graph connections to be notable

# Signal C: topic cluster thresholds
TOPIC_SPIKE_WINDOW_HOURS = 6         # wider window smooths scraper-batch noise
TOPIC_SPIKE_RATIO_THRESHOLD = 2.5
TOPIC_MIN_BASELINE_ARTICLES = 1      # skip low-volume clusters
TOPIC_MIN_CURRENT_ARTICLES = 1
TOPIC_MIN_CLUSTER_AGE_HOURS = 3      # brand-new cluster can't "spike"

# How long a detected signal stays visible in the API
SIGNAL_TTL_HOURS = 6

# IST offset (UTC+5:30) — normalise to local news cycle time
IST = timezone(timedelta(hours=5, minutes=30))

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info(f"Received signal {signum}, shutting down.")
    _shutdown = True


# ── Signal A: Entity mention spike ───────────────────────────────────────────

def _detect_entity_spikes(driver) -> list[dict]:
    """Compare entity article counts today vs the same calendar day over 7 days.

    Article nodes in Neo4j only carry pub_date (RSS string) — no scraped_at.
    We parse pub_date to YYYY-MM-DD using _PARSE_PUB_DATE and compare at
    day granularity. This still eliminates the flat-average bias because each
    baseline day is a discrete observation rather than a smeared mean.
    """
    now_ist = datetime.now(IST)
    today_str = now_ist.strftime("%Y-%m-%d")
    baseline_dates = [
        (now_ist - timedelta(days=d)).strftime("%Y-%m-%d")
        for d in range(1, 8)
    ]

    parse_a = _PARSE_PUB_DATE.format(src="a")
    signals = []

    with driver.session() as session:
        # Step 1: entities mentioned today
        current_rows = session.run(f"""
            MATCH (a:Article)-[:EVIDENCES]->(e:Entity)
            WITH e, a, {parse_a} AS pub
            WHERE pub = $today
            WITH e.name AS name, e.type AS etype, count(DISTINCT a) AS cnt
            WHERE cnt >= $min_current
            RETURN name, etype, cnt
            ORDER BY cnt DESC
            LIMIT 100
        """, today=today_str, min_current=ENTITY_MIN_CURRENT_MENTIONS).data()

        if not current_rows:
            return []

        entity_names = [r["name"] for r in current_rows]
        current_by_name = {r["name"]: (r["cnt"], r["etype"]) for r in current_rows}

        # Step 2: for each of the past 7 days, count mentions per entity
        baseline_counts: dict[str, list[int]] = {n: [] for n in entity_names}

        for day_str in baseline_dates:
            day_rows = session.run(f"""
                MATCH (a:Article)-[:EVIDENCES]->(e:Entity)
                WHERE e.name IN $names
                WITH e, a, {parse_a} AS pub
                WHERE pub = $day
                WITH e.name AS name, count(DISTINCT a) AS cnt
                RETURN name, cnt
            """, names=entity_names, day=day_str).data()

            day_map = {r["name"]: r["cnt"] for r in day_rows}
            for name in entity_names:
                baseline_counts[name].append(day_map.get(name, 0))

        # Step 3: compute ratios and emit signals
        for name in entity_names:
            current_cnt, etype = current_by_name[name]
            history = baseline_counts[name]
            total_baseline = sum(history)

            if total_baseline < ENTITY_MIN_BASELINE_MENTIONS:
                continue  # not enough history — skip to avoid false positives

            baseline_avg = total_baseline / len(history)
            ratio = (current_cnt + LAPLACE_SMOOTH) / (baseline_avg + LAPLACE_SMOOTH)

            if ratio < ENTITY_SPIKE_RATIO_THRESHOLD:
                continue

            severity = "high" if ratio >= 5.0 else "medium"
            signals.append({
                "signal_type": "entity_spike",
                "severity": severity,
                "entity_name": name,
                "entity_type": etype or "",
                "cluster_id": "",
                "cluster_label": "",
                "domain": "",
                "spike_ratio": ratio,
                "current_count": current_cnt,
                "baseline_count": baseline_avg,
            })

    logger.info(f"entity_spike: {len(signals)} signals detected")
    return signals


# ── Signal B: New high-connectivity entity ────────────────────────────────────

def _detect_new_entities(driver) -> list[dict]:
    """Find entities first seen within the last 24h that already have high
    graph connectivity.

    Entity nodes carry e.first_seen = pub_date (set ON CREATE in graph_builder).
    We use _PARSE_PUB_DATE on first_seen to get a comparable date string,
    then check it falls within the lookback window. This avoids a full node
    scan on a non-existent scraped_at property.

    NOTE: For production scale, add a Neo4j index on Entity.first_seen:
          CREATE INDEX entity_first_seen IF NOT EXISTS
          FOR (e:Entity) ON (e.first_seen)
    """
    now_ist = datetime.now(IST)
    # Build set of date strings covering the lookback window
    lookback_dates = {
        (now_ist - timedelta(hours=h)).strftime("%Y-%m-%d")
        for h in range(NEW_ENTITY_LOOKBACK_HOURS + 1)
    }
    parse_e = _PARSE_PUB_DATE.format(src="e")

    signals = []
    with driver.session() as session:
        rows = session.run(f"""
            MATCH (e:Entity)
            WHERE e.first_seen IS NOT NULL
            WITH e, {parse_e} AS first_seen_date
            WHERE first_seen_date IN $lookback_dates
            MATCH (e)-[r:RELATES_TO]-()
            WITH e, first_seen_date, count(r) AS degree
            WHERE degree >= $min_degree
            MATCH (a:Article)-[:EVIDENCES]->(e)
            RETURN e.name AS name, e.type AS etype,
                   degree, count(DISTINCT a) AS article_count
            ORDER BY degree DESC
            LIMIT 30
        """, lookback_dates=list(lookback_dates), min_degree=NEW_ENTITY_MIN_DEGREE).data()

        for row in rows:
            signals.append({
                "signal_type": "new_entity",
                "severity": "high" if row["degree"] >= 10 else "medium",
                "entity_name": row["name"],
                "entity_type": row["etype"] or "",
                "cluster_id": "",
                "cluster_label": "",
                "domain": "",
                "spike_ratio": float(row["degree"]),  # ratio = degree for new entities
                "current_count": row["article_count"],
                "baseline_count": 0.0,
            })

    logger.info(f"new_entity: {len(signals)} signals detected")
    return signals


# ── Signal C: Topic cluster spike (Postgres) ─────────────────────────────────

def _detect_topic_spikes(db) -> list[dict]:
    """Compare topic cluster article counts in current 6h window vs same 6h
    window across past 7 days (same-hour baseline).

    Uses a 6h window (not 2h) to smooth over scraper batch cadence. Clusters
    younger than TOPIC_MIN_CLUSTER_AGE_HOURS are excluded — a brand-new cluster
    cannot spike.
    """
    now_utc = datetime.now(timezone.utc)
    window_end = now_utc.replace(minute=0, second=0, microsecond=0)
    window_start = window_end - timedelta(hours=TOPIC_SPIKE_WINDOW_HOURS)
    cluster_age_cutoff = now_utc - timedelta(hours=TOPIC_MIN_CLUSTER_AGE_HOURS)

    # Current window: article count per cluster
    current_rows = db.execute(
        select(
            ScrapedArticle.topic_cluster_id,
            ScrapedArticle.cluster_label,
            ScrapedArticle.domain,
            func.count(ScrapedArticle.id).label("cnt"),
        )
        .where(ScrapedArticle.scraped_at >= window_start)
        .where(ScrapedArticle.scraped_at < window_end)
        .where(ScrapedArticle.topic_cluster_id.isnot(None))
        .group_by(
            ScrapedArticle.topic_cluster_id,
            ScrapedArticle.cluster_label,
            ScrapedArticle.domain,
        )
        .having(func.count(ScrapedArticle.id) >= TOPIC_MIN_CURRENT_ARTICLES)
    ).all()

    if not current_rows:
        return []

    cluster_ids = [r.topic_cluster_id for r in current_rows]

    # Baseline: same 6h slot over the past 7 days, grouped by cluster
    baseline_slots = []
    for days_back in range(1, 8):
        slot_start = window_start - timedelta(days=days_back)
        slot_end = window_end - timedelta(days=days_back)
        baseline_slots.append((slot_start, slot_end))

    # Fetch all baseline counts in one query per slot
    baseline_sums: dict[str, list[int]] = {c: [] for c in cluster_ids}
    for slot_start, slot_end in baseline_slots:
        slot_rows = db.execute(
            select(
                ScrapedArticle.topic_cluster_id,
                func.count(ScrapedArticle.id).label("cnt"),
            )
            .where(ScrapedArticle.topic_cluster_id.in_(cluster_ids))
            .where(ScrapedArticle.scraped_at >= slot_start)
            .where(ScrapedArticle.scraped_at < slot_end)
            .group_by(ScrapedArticle.topic_cluster_id)
        ).all()
        slot_map = {r.topic_cluster_id: r.cnt for r in slot_rows}
        for cid in cluster_ids:
            baseline_sums[cid].append(slot_map.get(cid, 0))

    # Filter out clusters younger than MIN_CLUSTER_AGE by checking first article
    first_seen_rows = db.execute(
        select(
            ScrapedArticle.topic_cluster_id,
            func.min(ScrapedArticle.scraped_at).label("first_seen"),
        )
        .where(ScrapedArticle.topic_cluster_id.in_(cluster_ids))
        .group_by(ScrapedArticle.topic_cluster_id)
    ).all()
    first_seen_map = {r.topic_cluster_id: r.first_seen for r in first_seen_rows}

    signals = []
    for row in current_rows:
        cid = row.topic_cluster_id
        first_seen = first_seen_map.get(cid)
        if first_seen and first_seen.replace(tzinfo=timezone.utc) > cluster_age_cutoff:
            continue  # cluster is too new

        history = baseline_sums[cid]
        total_baseline = sum(history)

        if total_baseline < TOPIC_MIN_BASELINE_ARTICLES:
            continue  # not enough history

        baseline_avg = total_baseline / len(history)
        ratio = (row.cnt + LAPLACE_SMOOTH) / (baseline_avg + LAPLACE_SMOOTH)

        if ratio < TOPIC_SPIKE_RATIO_THRESHOLD:
            continue

        severity = "high" if ratio >= 4.0 else "medium"
        signals.append({
            "signal_type": "topic_spike",
            "severity": severity,
            "entity_name": "",
            "entity_type": "",
            "cluster_id": cid,
            "cluster_label": row.cluster_label or "",
            "domain": row.domain or "",
            "spike_ratio": ratio,
            "current_count": row.cnt,
            "baseline_count": baseline_avg,
        })

    logger.info(f"topic_spike: {len(signals)} signals detected")
    return signals


# ── Persistence ───────────────────────────────────────────────────────────────

def _persist_signals(db, signals: list[dict]) -> int:
    """Upsert signals. The unique constraint on (signal_type, entity_name,
    cluster_id, detected_at) prevents duplicate rows within a single run.
    Returns count of inserted/updated rows.
    """
    if not signals:
        return 0

    now_utc = datetime.now(timezone.utc)
    expires_at = now_utc + timedelta(hours=SIGNAL_TTL_HOURS)

    # Delete expired signals first
    db.query(DetectedSignal).filter(DetectedSignal.expires_at <= now_utc).delete()

    inserted = 0
    for sig in signals:
        stmt = (
            pg_insert(DetectedSignal)
            .values(
                **sig,
                detected_at=now_utc,
                expires_at=expires_at,
            )
            .on_conflict_do_update(
                constraint="uq_signal_subject_detected_at",
                set_={
                    "spike_ratio": sig["spike_ratio"],
                    "current_count": sig["current_count"],
                    "expires_at": expires_at,
                },
            )
        )
        db.execute(stmt)
        inserted += 1

    db.commit()
    return inserted


def _publish_signals(r: redis.Redis, signals: list[dict]) -> None:
    """Publish high-severity signals to the live-feed Redis channel so the
    SSE stream picks them up without polling.
    """
    high = [s for s in signals if s["severity"] == "high"]
    if not high:
        return
    try:
        r.publish("india-innovates:live-feed", json.dumps({
            "type": "signal",
            "count": len(high),
            "signals": high[:5],  # cap payload size
            "timestamp": time.time(),
        }))
    except Exception as e:
        logger.debug(f"Failed to publish signal notification: {e}")


# ── On-demand detection (used by API /signals/refresh) ───────────────────────

def run_detection_cycle() -> int:
    """Run a single detection cycle and persist results. Returns count of signals."""
    from neo4j import GraphDatabase
    import redis as redis_lib
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    redis_client = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    db = SessionLocal()
    try:
        all_signals: list[dict] = []
        try:
            all_signals += _detect_entity_spikes(neo4j_driver)
        except Exception as e:
            logger.warning(f"entity_spike: {e}")
        try:
            all_signals += _detect_new_entities(neo4j_driver)
        except Exception as e:
            logger.warning(f"new_entity: {e}")
        try:
            all_signals += _detect_topic_spikes(db)
        except Exception as e:
            logger.warning(f"topic_spike: {e}")
        count = _persist_signals(db, all_signals)
        _publish_signals(redis_client, all_signals)
        return count
    finally:
        db.close()
        neo4j_driver.close()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    os_signal.signal(os_signal.SIGINT, _handle_signal)
    os_signal.signal(os_signal.SIGTERM, _handle_signal)

    Base.metadata.create_all(engine)
    logger.info("Database tables ensured.")

    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    # Initialize signal swarm if available
    signal_swarm = None
    if _SIGNAL_SWARM_AVAILABLE:
        try:
            signal_swarm = SignalSwarm(neo4j_driver, redis_client)
            logger.info("[SignalSwarm] Bee-algorithm swarm initialized (scouts + workers + reporter)")
        except Exception as e:
            logger.warning(f"[SignalSwarm] Failed to initialize: {e}")

    logger.info("=" * 60)
    logger.info("Starting Signal Worker")
    logger.info(f"  Mode: {'Swarm (bee-algorithm) + Statistical' if signal_swarm else 'Statistical only'}")
    logger.info(f"  Interval: {SIGNAL_WORKER_INTERVAL_SECONDS}s")
    logger.info(f"  Signal TTL: {SIGNAL_TTL_HOURS}h")
    logger.info("=" * 60)

    try:
        while not _shutdown:
            cycle_start = time.time()
            db = SessionLocal()
            try:
                all_signals: list[dict] = []

                # ── Statistical detectors (existing) ─────────────────────────
                try:
                    all_signals += _detect_entity_spikes(neo4j_driver)
                except Exception as e:
                    logger.error(f"entity_spike detection failed: {e}", exc_info=True)

                try:
                    all_signals += _detect_new_entities(neo4j_driver)
                except Exception as e:
                    logger.error(f"new_entity detection failed: {e}", exc_info=True)

                try:
                    all_signals += _detect_topic_spikes(db)
                except Exception as e:
                    logger.error(f"topic_spike detection failed: {e}", exc_info=True)

                try:
                    detector = DisinfoDetector()
                    disinfo_sigs = detector.run()
                    logger.info(f"disinfo_detector: {len(disinfo_sigs)} signals")
                except Exception as e:
                    logger.error(f"disinfo detection failed: {e}", exc_info=True)

                # ── Swarm detectors (bee-algorithm augmentation) ──────────────
                if signal_swarm:
                    try:
                        logger.info("=" * 60)
                        logger.info("SWARM C: Signal Detection Swarm (bee-algorithm)")
                        logger.info("=" * 60)
                        swarm_signals = signal_swarm.run()
                        # Merge: swarm signals that aren't already covered by stats
                        existing_keys = {
                            f"{s['signal_type']}::{s.get('entity_name') or s.get('cluster_id', '')}"
                            for s in all_signals
                        }
                        new_from_swarm = [
                            s for s in swarm_signals
                            if f"{s['signal_type']}::{s.get('entity_name') or s.get('cluster_id', '')}"
                            not in existing_keys
                        ]
                        all_signals += new_from_swarm
                        logger.info(
                            f"[SignalSwarm] +{len(new_from_swarm)} new signals "
                            f"({len(swarm_signals) - len(new_from_swarm)} duplicates merged)"
                        )
                    except Exception as e:
                        logger.error(f"SignalSwarm failed: {e}", exc_info=True)

                count = _persist_signals(db, all_signals)
                _publish_signals(redis_client, all_signals)

                elapsed = time.time() - cycle_start
                logger.info(
                    f"Signal cycle complete in {elapsed:.1f}s — "
                    f"{count} signals persisted"
                )
            except Exception as e:
                db.rollback()
                logger.error(f"Signal cycle failed: {e}", exc_info=True)
            finally:
                db.close()

            sleep_end = time.time() + SIGNAL_WORKER_INTERVAL_SECONDS
            while not _shutdown and time.time() < sleep_end:
                time.sleep(5)
    finally:
        if signal_swarm:
            signal_swarm.close()
        neo4j_driver.close()
        redis_client.close()
        logger.info("Signal worker shut down.")


if __name__ == "__main__":
    main()
