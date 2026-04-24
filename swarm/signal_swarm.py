"""Swarm C — Signal Detection Swarm (Bee Algorithm).

Upgrades the statistical signal worker with a bee-swarm pattern:

  Scout Agents (N=3):
      Randomly sample subgraphs of Neo4j looking for anomalous patterns.
      On finding something interesting, they publish a "waggle dance" message
      to Redis Pub/Sub channel swarm:signals:dance — signalling workers to
      investigate the flagged region.

  Worker Agents (N=5):
      Listen on the dance channel. When a scout flags a region, workers
      deep-dive: run precise statistical tests on the flagged entities/clusters
      and determine if a genuine signal exists.

  Reporter Agent (N=1):
      Synthesizes all worker findings into final Signal dicts.
      Deduplicates and ranks by severity.

The "dance" (communication) happens via Redis Pub/Sub, mirroring the
bee-hive communication pattern where scouts recruit workers proportional
to the quality of the food source (anomaly strength).

This augments (not replaces) the existing statistical detectors:
  - Scouts find NEW anomaly regions the statistics might miss
  - Workers confirm or reject with deep statistical checks
  - Reporter adds LLM context for severity classification

Returns same format as signal_worker.py (_detect_entity_spikes etc.) so
it integrates cleanly into the existing persistence layer.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from datetime import datetime, timedelta, timezone

import redis
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_AUTH, REDIS_HOST, REDIS_PORT
from swarm.blackboard import SwarmBlackboard, DANCE_CHANNEL

logger = logging.getLogger(__name__)

# ── Swarm config ──────────────────────────────────────────────────────────────
N_SCOUTS = 3
N_WORKERS = 5
SCOUT_TIMEOUT = 60      # seconds per scout run
WORKER_TIMEOUT = 90     # seconds per worker deep-dive
DANCE_LISTEN_TIMEOUT = 30  # seconds workers listen for dances after scouts done
LAPLACE_SMOOTH = 1.0

# Sampling parameters
SCOUT_SAMPLE_SIZE = 20   # entities to sample per scout
SCOUT_SPIKE_THRESHOLD = 2.5  # quick spike ratio filter in scouts

# IST offset
IST = timezone(timedelta(hours=5, minutes=30))


# ── Scout Agent ───────────────────────────────────────────────────────────────

class _ScoutAgent:
    """Randomly samples Neo4j subgraph for anomalous patterns.

    Scouts look for quick signals using lightweight heuristics:
    - Entity degree spikes (many new relations in short time)
    - Cluster densification (sudden clique formation)
    - Bridge entity emergence (new cross-domain connectors)

    On finding something, publishes a dance message to Redis.
    """

    def __init__(self, scout_id: int, driver, bb: SwarmBlackboard):
        self.scout_id = scout_id
        self.driver = driver
        self.bb = bb

    def run(self) -> list[dict]:
        """Sample subgraph and publish dance messages for anything interesting."""
        now_ist = datetime.now(IST)
        today_str = now_ist.strftime("%Y-%m-%d")
        yesterday_str = (now_ist - timedelta(days=1)).strftime("%Y-%m-%d")

        dances_published = []

        # Sampling strategy varies per scout for diversity
        if self.scout_id == 0:
            findings = self._sample_entity_spikes(today_str, yesterday_str)
        elif self.scout_id == 1:
            findings = self._sample_new_connectors(today_str)
        else:
            findings = self._sample_cluster_burst(now_ist)

        for finding in findings:
            dance_msg = {
                "scout_id": self.scout_id,
                "finding_type": finding["type"],
                "subject": finding["subject"],
                "preliminary_score": finding["score"],
                "metadata": finding.get("metadata", {}),
                "ts": time.time(),
            }
            self.bb.publish_dance(dance_msg)
            dances_published.append(dance_msg)
            logger.debug(
                f"[Scout{self.scout_id}] dance published: "
                f"{finding['type']} — {finding['subject']} (score={finding['score']:.1f})"
            )

        logger.info(f"[Scout{self.scout_id}] {len(dances_published)} dances published")
        return dances_published

    def _sample_entity_spikes(self, today: str, yesterday: str) -> list[dict]:
        """Sample random entities and check for quick mention spikes."""
        findings = []
        try:
            from scheduler.signal_worker import _PARSE_PUB_DATE
            parse_a = _PARSE_PUB_DATE.format(src="a")
            with self.driver.session() as session:
                # Random sample of recently active entities
                rows = session.run(f"""
                    MATCH (a:Article)-[:EVIDENCES]->(e:Entity)
                    WITH e, a, {parse_a} AS pub
                    WHERE pub >= $yesterday
                    WITH e.name AS name, e.type AS etype, count(DISTINCT a) AS today_cnt
                    WHERE today_cnt >= 2
                    RETURN name, etype, today_cnt
                    ORDER BY rand()
                    LIMIT $sample_size
                """, yesterday=yesterday, sample_size=SCOUT_SAMPLE_SIZE).data()

                for row in rows:
                    # Quick baseline: count from 7-day window
                    week_ago = (datetime.now(IST) - timedelta(days=7)).strftime("%Y-%m-%d")
                    baseline_row = session.run(f"""
                        MATCH (a:Article)-[:EVIDENCES]->(e:Entity {{name: $name}})
                        WITH a, {parse_a} AS pub
                        WHERE pub >= $week_ago AND pub < $yesterday
                        RETURN count(DISTINCT a) AS cnt
                    """, name=row["name"], week_ago=week_ago, yesterday=yesterday).single()

                    baseline = (baseline_row["cnt"] if baseline_row else 0)
                    ratio = (row["today_cnt"] + LAPLACE_SMOOTH) / (baseline / 6 + LAPLACE_SMOOTH)

                    if ratio >= SCOUT_SPIKE_THRESHOLD and baseline >= 3:
                        findings.append({
                            "type": "entity_spike_candidate",
                            "subject": row["name"],
                            "score": ratio,
                            "metadata": {"etype": row["etype"], "today_cnt": row["today_cnt"]},
                        })
        except Exception as e:
            logger.debug(f"[Scout{self.scout_id}] entity spike sampling failed: {e}")
        return findings

    def _sample_new_connectors(self, today: str) -> list[dict]:
        """Find entities that recently became cross-domain connectors."""
        findings = []
        try:
            from scheduler.signal_worker import _PARSE_PUB_DATE
            parse_e = _PARSE_PUB_DATE.format(src="e")
            with self.driver.session() as session:
                rows = session.run(f"""
                    MATCH (e:Entity)
                    WHERE e.first_seen IS NOT NULL
                    WITH e, {parse_e} AS first_seen
                    WHERE first_seen = $today
                    MATCH (e)-[r:RELATES_TO]-(neighbor:Entity)
                    WITH e, first_seen, count(DISTINCT r) AS degree,
                         collect(DISTINCT neighbor.type) AS neighbor_types
                    WHERE degree >= 3 AND size(neighbor_types) >= 2
                    RETURN e.name AS name, e.type AS etype,
                           degree, size(neighbor_types) AS domain_count
                    ORDER BY rand()
                    LIMIT $sample_size
                """, today=today, sample_size=SCOUT_SAMPLE_SIZE).data()

                for row in rows:
                    score = float(row["degree"]) * row["domain_count"]
                    findings.append({
                        "type": "new_connector_candidate",
                        "subject": row["name"],
                        "score": score,
                        "metadata": {
                            "etype": row["etype"],
                            "degree": row["degree"],
                            "domain_count": row["domain_count"],
                        },
                    })
        except Exception as e:
            logger.debug(f"[Scout{self.scout_id}] new connector sampling failed: {e}")
        return findings

    def _sample_cluster_burst(self, now_ist: datetime) -> list[dict]:
        """Find topic clusters with unusual recent growth from Postgres."""
        findings = []
        try:
            from models.database import SessionLocal
            from models.scraped_article import ScrapedArticle
            from sqlalchemy import select, func

            now_utc = datetime.now(timezone.utc)
            window_start = now_utc - timedelta(hours=6)
            week_ago = now_utc - timedelta(days=7)

            db = SessionLocal()
            try:
                # Random sample of active clusters
                rows = db.execute(
                    select(
                        ScrapedArticle.topic_cluster_id,
                        ScrapedArticle.cluster_label,
                        ScrapedArticle.domain,
                        func.count(ScrapedArticle.id).label("recent_cnt"),
                    )
                    .where(ScrapedArticle.scraped_at >= window_start)
                    .where(ScrapedArticle.topic_cluster_id.isnot(None))
                    .group_by(
                        ScrapedArticle.topic_cluster_id,
                        ScrapedArticle.cluster_label,
                        ScrapedArticle.domain,
                    )
                    .having(func.count(ScrapedArticle.id) >= 2)
                    .order_by(func.random())
                    .limit(SCOUT_SAMPLE_SIZE)
                ).all()

                for row in rows:
                    # Quick baseline
                    baseline_row = db.execute(
                        select(func.count(ScrapedArticle.id))
                        .where(ScrapedArticle.topic_cluster_id == row.topic_cluster_id)
                        .where(ScrapedArticle.scraped_at >= week_ago)
                        .where(ScrapedArticle.scraped_at < window_start)
                    ).scalar() or 0

                    ratio = (row.recent_cnt + LAPLACE_SMOOTH) / (baseline_row / 7 + LAPLACE_SMOOTH)
                    if ratio >= SCOUT_SPIKE_THRESHOLD:
                        findings.append({
                            "type": "cluster_burst_candidate",
                            "subject": row.cluster_label or row.topic_cluster_id,
                            "score": ratio,
                            "metadata": {
                                "cluster_id": row.topic_cluster_id,
                                "domain": row.domain,
                                "recent_cnt": row.recent_cnt,
                            },
                        })
            finally:
                db.close()

        except Exception as e:
            logger.debug(f"[Scout{self.scout_id}] cluster burst sampling failed: {e}")
        return findings


# ── Worker Agent ──────────────────────────────────────────────────────────────

class _WorkerAgent:
    """Deep-dives on scout-flagged anomaly candidates.

    Listens on the Redis dance channel, picks up dance messages, runs full
    statistical validation on the flagged entity/cluster.
    """

    def __init__(self, worker_id: int, driver):
        self.worker_id = worker_id
        self.driver = driver

    def deep_dive(self, dance_msg: dict) -> dict | None:
        """Run precise statistical test on a scout-flagged candidate."""
        finding_type = dance_msg.get("finding_type", "")
        subject = dance_msg.get("subject", "")
        metadata = dance_msg.get("metadata", {})

        try:
            if finding_type == "entity_spike_candidate":
                return self._validate_entity_spike(subject, metadata)
            elif finding_type == "new_connector_candidate":
                return self._validate_new_connector(subject, metadata)
            elif finding_type == "cluster_burst_candidate":
                return self._validate_cluster_burst(subject, metadata)
        except Exception as e:
            logger.debug(f"[Worker{self.worker_id}] deep-dive failed for {subject}: {e}")
        return None

    def _validate_entity_spike(self, entity_name: str, metadata: dict) -> dict | None:
        """Full statistical spike validation matching signal_worker standards."""
        from scheduler.signal_worker import (
            _PARSE_PUB_DATE, ENTITY_SPIKE_RATIO_THRESHOLD,
            ENTITY_MIN_BASELINE_MENTIONS, ENTITY_MIN_CURRENT_MENTIONS,
        )
        now_ist = datetime.now(IST)
        today_str = now_ist.strftime("%Y-%m-%d")
        baseline_dates = [
            (now_ist - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(1, 8)
        ]
        parse_a = _PARSE_PUB_DATE.format(src="a")

        with self.driver.session() as session:
            current = session.run(f"""
                MATCH (a:Article)-[:EVIDENCES]->(e:Entity {{name: $name}})
                WITH a, {parse_a} AS pub
                WHERE pub = $today
                RETURN count(DISTINCT a) AS cnt
            """, name=entity_name, today=today_str).single()
            current_cnt = current["cnt"] if current else 0

            if current_cnt < ENTITY_MIN_CURRENT_MENTIONS:
                return None

            baseline_counts = []
            for day_str in baseline_dates:
                row = session.run(f"""
                    MATCH (a:Article)-[:EVIDENCES]->(e:Entity {{name: $name}})
                    WITH a, {parse_a} AS pub
                    WHERE pub = $day
                    RETURN count(DISTINCT a) AS cnt
                """, name=entity_name, day=day_str).single()
                baseline_counts.append(row["cnt"] if row else 0)

            total_baseline = sum(baseline_counts)
            if total_baseline < ENTITY_MIN_BASELINE_MENTIONS:
                return None

            baseline_avg = total_baseline / len(baseline_counts)
            ratio = (current_cnt + LAPLACE_SMOOTH) / (baseline_avg + LAPLACE_SMOOTH)

            if ratio < ENTITY_SPIKE_RATIO_THRESHOLD:
                return None

            return {
                "signal_type": "entity_spike",
                "severity": "high" if ratio >= 5.0 else "medium",
                "entity_name": entity_name,
                "entity_type": metadata.get("etype", ""),
                "cluster_id": "",
                "cluster_label": "",
                "domain": "",
                "spike_ratio": ratio,
                "current_count": current_cnt,
                "baseline_count": baseline_avg,
                "_swarm_source": f"scout+worker{self.worker_id}",
            }

    def _validate_new_connector(self, entity_name: str, metadata: dict) -> dict | None:
        """Validate a new cross-domain connector entity."""
        from scheduler.signal_worker import NEW_ENTITY_MIN_DEGREE
        with self.driver.session() as session:
            row = session.run("""
                MATCH (e:Entity {name: $name})-[r:RELATES_TO]-()
                WITH e, count(r) AS degree
                MATCH (a:Article)-[:EVIDENCES]->(e)
                RETURN degree, count(DISTINCT a) AS article_count
            """, name=entity_name).single()

            if not row or row["degree"] < NEW_ENTITY_MIN_DEGREE:
                return None

            return {
                "signal_type": "new_entity",
                "severity": "high" if row["degree"] >= 10 else "medium",
                "entity_name": entity_name,
                "entity_type": metadata.get("etype", ""),
                "cluster_id": "",
                "cluster_label": "",
                "domain": "",
                "spike_ratio": float(row["degree"]),
                "current_count": row["article_count"],
                "baseline_count": 0.0,
                "_swarm_source": f"scout+worker{self.worker_id}",
            }

    def _validate_cluster_burst(self, cluster_label: str, metadata: dict) -> dict | None:
        """Validate a topic cluster burst from Postgres."""
        from scheduler.signal_worker import (
            TOPIC_SPIKE_RATIO_THRESHOLD, TOPIC_MIN_BASELINE_ARTICLES,
            TOPIC_MIN_CURRENT_ARTICLES, TOPIC_MIN_CLUSTER_AGE_HOURS,
        )
        cluster_id = metadata.get("cluster_id")
        if not cluster_id:
            return None

        from models.database import SessionLocal
        from models.scraped_article import ScrapedArticle
        from sqlalchemy import select, func

        now_utc = datetime.now(timezone.utc)
        window_end = now_utc.replace(minute=0, second=0, microsecond=0)
        window_start = window_end - timedelta(hours=6)

        db = SessionLocal()
        try:
            current_cnt = db.execute(
                select(func.count(ScrapedArticle.id))
                .where(ScrapedArticle.topic_cluster_id == cluster_id)
                .where(ScrapedArticle.scraped_at >= window_start)
                .where(ScrapedArticle.scraped_at < window_end)
            ).scalar() or 0

            if current_cnt < TOPIC_MIN_CURRENT_ARTICLES:
                return None

            # 7-day baseline
            baseline_counts = []
            for days_back in range(1, 8):
                slot_start = window_start - timedelta(days=days_back)
                slot_end = window_end - timedelta(days=days_back)
                cnt = db.execute(
                    select(func.count(ScrapedArticle.id))
                    .where(ScrapedArticle.topic_cluster_id == cluster_id)
                    .where(ScrapedArticle.scraped_at >= slot_start)
                    .where(ScrapedArticle.scraped_at < slot_end)
                ).scalar() or 0
                baseline_counts.append(cnt)

            total_baseline = sum(baseline_counts)
            if total_baseline < TOPIC_MIN_BASELINE_ARTICLES:
                return None

            baseline_avg = total_baseline / len(baseline_counts)
            ratio = (current_cnt + LAPLACE_SMOOTH) / (baseline_avg + LAPLACE_SMOOTH)

            if ratio < TOPIC_SPIKE_RATIO_THRESHOLD:
                return None

            return {
                "signal_type": "topic_spike",
                "severity": "high" if ratio >= 4.0 else "medium",
                "entity_name": "",
                "entity_type": "",
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "domain": metadata.get("domain", ""),
                "spike_ratio": ratio,
                "current_count": current_cnt,
                "baseline_count": baseline_avg,
                "_swarm_source": f"scout+worker{self.worker_id}",
            }
        finally:
            db.close()


# ── Reporter Agent ────────────────────────────────────────────────────────────

class _ReporterAgent:
    """Synthesizes all worker findings into final, deduplicated Signal dicts.

    Optionally uses LLM to re-classify severity based on broader context.
    """

    def __init__(self):
        self._llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.1, num_predict=512)

    def synthesize(self, worker_findings: list[dict]) -> list[dict]:
        """Deduplicate, rank, and optionally enrich signals."""
        if not worker_findings:
            return []

        # Deduplicate by (signal_type, entity_name OR cluster_id)
        seen: dict[str, dict] = {}
        for sig in worker_findings:
            dedup_key = f"{sig['signal_type']}::{sig.get('entity_name') or sig.get('cluster_id', '')}"
            if dedup_key not in seen or sig["spike_ratio"] > seen[dedup_key]["spike_ratio"]:
                seen[dedup_key] = sig

        final = list(seen.values())

        # Sort by spike_ratio descending
        final.sort(key=lambda s: s["spike_ratio"], reverse=True)

        # Remove internal metadata key before returning
        for sig in final:
            sig.pop("_swarm_source", None)

        logger.info(f"[Reporter] synthesized {len(final)} unique signals from {len(worker_findings)} findings")
        return final


# ── SignalSwarm orchestrator ──────────────────────────────────────────────────

class SignalSwarm:
    """Bee-algorithm signal detection swarm.

    Usage:
        swarm = SignalSwarm(neo4j_driver, redis_client)
        new_signals = swarm.run()
        # Returns list[dict] in same format as signal_worker.py

    Designed to AUGMENT the existing statistical detectors, not replace them.
    Integrate in signal_worker main():
        all_signals += swarm.run()
    """

    def __init__(self, neo4j_driver, redis_client: redis.Redis):
        self.driver = neo4j_driver
        self.redis = redis_client
        self._bb: SwarmBlackboard | None = None
        try:
            self._bb = SwarmBlackboard()
        except Exception as e:
            logger.warning(f"[SignalSwarm] Redis blackboard unavailable: {e}")

        self._scouts = [
            _ScoutAgent(i, neo4j_driver, self._bb or _DummyBlackboard())
            for i in range(N_SCOUTS)
        ]
        self._workers = [_WorkerAgent(i, neo4j_driver) for i in range(N_WORKERS)]
        self._reporter = _ReporterAgent()
        self._executor = ThreadPoolExecutor(max_workers=N_SCOUTS + N_WORKERS)

    def run(self) -> list[dict]:
        """Run one full bee-algorithm cycle. Returns validated signals."""
        logger.info(
            f"[SignalSwarm] Starting bee-algorithm cycle: "
            f"{N_SCOUTS} scouts → dance channel → {N_WORKERS} workers → reporter"
        )
        cycle_start = time.time()

        # Phase 1: Scouts sample subgraph and publish dance messages
        logger.info("[SignalSwarm] Phase 1: Scouts sampling Neo4j...")
        all_dances: list[dict] = []

        scout_futures = {
            self._executor.submit(s.run): s.scout_id
            for s in self._scouts
        }
        for future in as_completed(scout_futures, timeout=SCOUT_TIMEOUT * 1.5):
            scout_id = scout_futures[future]
            try:
                dances = future.result(timeout=SCOUT_TIMEOUT)
                all_dances.extend(dances)
                logger.info(f"[Scout{scout_id}] completed: {len(dances)} dances")
            except Exception as e:
                logger.warning(f"[Scout{scout_id}] failed: {e}")

        if not all_dances:
            logger.info("[SignalSwarm] No dances published — no candidate anomalies found")
            return []

        logger.info(f"[SignalSwarm] Phase 1 complete: {len(all_dances)} dance messages published")

        # Phase 2: Workers deep-dive on dance-flagged candidates
        # Distribute dances across workers (each worker gets ≈ len/N_WORKERS dances)
        logger.info(f"[SignalSwarm] Phase 2: {N_WORKERS} workers deep-diving on {len(all_dances)} candidates...")

        # Assign dances round-robin
        dance_assignments: dict[int, list[dict]] = {i: [] for i in range(N_WORKERS)}
        for i, dance in enumerate(all_dances):
            dance_assignments[i % N_WORKERS].append(dance)

        worker_findings: list[dict] = []
        worker_futures = {
            self._executor.submit(self._run_worker, worker, dance_assignments[worker.worker_id]): worker.worker_id
            for worker in self._workers
        }
        for future in as_completed(worker_futures, timeout=WORKER_TIMEOUT * 1.5):
            worker_id = worker_futures[future]
            try:
                findings = future.result(timeout=WORKER_TIMEOUT)
                worker_findings.extend(findings)
                logger.info(f"[Worker{worker_id}] completed: {len(findings)} validated signals")
            except Exception as e:
                logger.warning(f"[Worker{worker_id}] failed: {e}")

        logger.info(f"[SignalSwarm] Phase 2 complete: {len(worker_findings)} raw signals from workers")

        # Phase 3: Reporter synthesizes
        logger.info("[SignalSwarm] Phase 3: Reporter synthesizing...")
        final_signals = self._reporter.synthesize(worker_findings)

        elapsed = time.time() - cycle_start
        logger.info(
            f"[SignalSwarm] Cycle complete in {elapsed:.1f}s — "
            f"{len(final_signals)} validated signals"
        )

        # Pheromone reinforcement for detected entities
        if self._bb:
            for sig in final_signals:
                subject = sig.get("entity_name") or sig.get("cluster_label", "")
                if subject:
                    self._bb.reinforce_pheromone(subject, delta=sig["spike_ratio"] / 5.0)

        return final_signals

    def _run_worker(self, worker: _WorkerAgent, dances: list[dict]) -> list[dict]:
        """Run a worker against its assigned dance messages."""
        findings = []
        for dance in dances:
            result = worker.deep_dive(dance)
            if result:
                findings.append(result)
        return findings

    def close(self):
        self._executor.shutdown(wait=False)
        if self._bb:
            self._bb.close()


class _DummyBlackboard:
    """No-op blackboard when Redis is unavailable."""
    def publish_dance(self, message: dict) -> None:
        pass
    def reinforce_pheromone(self, topic: str, delta: float = 1.0) -> float:
        return 0.0
