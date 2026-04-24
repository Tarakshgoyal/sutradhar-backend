"""Disinformation and Adversarial Narrative Detection Agent.

Detects coordinated inauthentic behavior, propaganda patterns, and
information operations targeting India — from both state and non-state actors.

Three detection layers:
  1. Narrative Cluster Analysis — temporal burst of identical narratives
  2. Sentiment Manipulation — sudden shifts not matching real events
  3. India-Targeted Threat Classification — LLM classifier
"""

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from models.database import SessionLocal
from models.scraped_article import ScrapedArticle
from models.disinfo_signal import DisinfoSignal

logger = logging.getLogger(__name__)
_IST = timezone(timedelta(hours=5, minutes=30))

# Minimum articles in a cluster to consider coordinated
_COORDINATION_MIN_ARTICLES = 5
# Minimum sources to deem "spread" (< threshold → suspicious single-source)
_COORDINATION_MIN_SOURCES = 2
# Cosine similarity threshold (reuses existing cluster_id from NewsPriorityAgent)
_BURST_WINDOW_HOURS = 6
# Confidence threshold above which we save a signal
_SAVE_THRESHOLD = 0.55


class ThreatAssessment(BaseModel):
    is_coordinated: bool = Field(description="True if coordination patterns are detected")
    threat_type: str = Field(
        description="one of: state_sponsored, domestic_political, commercial_spam, organic, unknown"
    )
    actor_attribution: str = Field(description="Best guess of actor, e.g. 'state-sponsored (China)' or 'unknown'")
    target_domain: str = Field(description="Domain being targeted, e.g. 'defense', 'elections', 'economy'")
    confidence: float = Field(description="0.0–1.0 confidence in the assessment")
    reasoning: str = Field(description="2-3 sentence justification")


class DisinfoDetector:
    """Detects coordinated narratives and manipulation patterns.

    Runs as a layer in the signal_worker and optionally as a pre-screen
    in the consumer pipeline for high-coordination clusters.
    """

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.2, num_predict=1024)
        logger.info("DisinfoDetector initialized")

    # ── Layer 1: Coordinated narrative detection ───────────────────────────────

    def detect_coordinated_narratives(
        self, window_hours: int = _BURST_WINDOW_HOURS
    ) -> list[dict]:
        """Detect clusters published in a tight time window from few sources.

        Uses existing cluster_id + source metadata from scraped_articles table.
        Coordination score = temporal density × narrative homogeneity × source concentration.
        """
        db = SessionLocal()
        signals = []
        try:
            cutoff = datetime.now(_IST) - timedelta(hours=window_hours)
            rows = db.scalars(
                select(ScrapedArticle).where(
                    ScrapedArticle.topic_cluster_id.is_not(None),
                    ScrapedArticle.scraped_at >= cutoff,
                ).order_by(ScrapedArticle.topic_cluster_id, ScrapedArticle.scraped_at)
            ).all()

            # Group by cluster
            clusters: dict[str, list[ScrapedArticle]] = defaultdict(list)
            for row in rows:
                if row.topic_cluster_id:
                    clusters[row.topic_cluster_id].append(row)

            for cluster_id, articles in clusters.items():
                if len(articles) < _COORDINATION_MIN_ARTICLES:
                    continue

                unique_sources = {a.source for a in articles}
                # Suspicious: many articles from few sources
                if len(unique_sources) >= _COORDINATION_MIN_SOURCES * 2:
                    continue

                # Time burst: all published within 2h of each other
                dates = [a.scraped_at for a in articles if a.scraped_at]
                if not dates:
                    continue
                time_spread_hours = (max(dates) - min(dates)).total_seconds() / 3600

                coordination_score = (
                    (len(articles) / 10.0)              # volume factor
                    * max(0, 1 - time_spread_hours / 6) # burst factor
                    * (1 - len(unique_sources) / max(len(articles), 1))  # concentration
                )
                coordination_score = min(coordination_score, 1.0)

                if coordination_score < 0.4:
                    continue

                label = articles[0].cluster_label or cluster_id
                signals.append({
                    "signal_type": "coordinated_narrative",
                    "severity": "high" if coordination_score > 0.7 else "medium",
                    "confidence": coordination_score,
                    "cluster_id": cluster_id,
                    "narrative_summary": label,
                    "coordination_score": coordination_score,
                    "evidence_articles": [a.url for a in articles[:10]],
                    "flagged_sources": list(unique_sources),
                    "target_entity": None,
                    "target_domain": articles[0].domain if articles else None,
                    "article_count": len(articles),
                })

        except Exception as e:
            logger.error(f"DisinfoDetector coordination detection failed: {e}")
        finally:
            db.close()

        logger.info(f"DisinfoDetector: {len(signals)} coordinated narrative signals")
        return signals

    # ── Layer 2: Sentiment manipulation detection ──────────────────────────────

    def detect_sentiment_manipulation(self) -> list[dict]:
        """Detect sudden entity sentiment shifts in Indic-language content.

        Compares sentiment polarity of articles mentioning the same entity
        in a 24h window vs the prior 24h. A sudden swing (>0.4) not explained
        by real events is flagged as potential sentiment manipulation.

        Uses IndicNLPAgent for Hindi/Urdu/Bengali sentiment scoring.
        """
        from agents.indic_nlp import IndicNLPAgent
        from sqlalchemy import select, desc

        agent = IndicNLPAgent()
        signals = []
        db = SessionLocal()
        try:
            now = datetime.now(_IST)
            window_24h = now - timedelta(hours=24)
            window_48h = now - timedelta(hours=48)

            # Fetch articles with Indic content (description/title has Indic chars)
            recent_articles = db.scalars(
                select(ScrapedArticle).where(
                    ScrapedArticle.scraped_at >= window_48h,
                ).order_by(desc(ScrapedArticle.scraped_at)).limit(500)
            ).all()

            # Filter to Indic-language articles
            indic_articles = [
                a for a in recent_articles
                if a.title and agent.is_indic_text(a.title)
            ]

            if len(indic_articles) < 10:
                # Not enough Indic content in window
                return []

            # Split into current (0-24h) and baseline (24-48h) windows
            current_articles = [
                a for a in indic_articles if a.scraped_at and a.scraped_at >= window_24h.replace(tzinfo=None)
            ]
            baseline_articles = [
                a for a in indic_articles if a.scraped_at and a.scraped_at < window_24h.replace(tzinfo=None)
            ]

            if not current_articles or not baseline_articles:
                return []

            # Analyze sentiment for both windows
            current_results = agent.analyze_batch([a.title for a in current_articles])
            baseline_results = agent.analyze_batch([a.title for a in baseline_articles])

            current_indic = [r for r in current_results if r.is_indic]
            baseline_indic = [r for r in baseline_results if r.is_indic]

            if not current_indic or not baseline_indic:
                return []

            current_avg = sum(r.sentiment_score for r in current_indic) / len(current_indic)
            baseline_avg = sum(r.sentiment_score for r in baseline_indic) / len(baseline_indic)

            swing = abs(current_avg - baseline_avg)
            if swing > 0.4:
                direction = "positive" if current_avg > baseline_avg else "negative"
                signals.append({
                    "signal_type": "sentiment_manipulation",
                    "severity": "high" if swing > 0.6 else "medium",
                    "confidence": min(swing, 1.0),
                    "actor_attribution": None,
                    "target_entity": None,
                    "target_domain": "indic_media",
                    "cluster_id": None,
                    "narrative_summary": (
                        f"Sudden {direction} sentiment swing in Indic-language media: "
                        f"+{swing:.2f} shift over 24h "
                        f"({len(current_indic)} articles analyzed)"
                    ),
                    "coordination_score": swing,
                    "evidence_articles": [a.url for a in current_articles[:5]],
                    "flagged_sources": list({a.source for a in current_articles}),
                })
                logger.info(
                    f"Sentiment manipulation signal: swing={swing:.3f} "
                    f"({baseline_avg:.2f} → {current_avg:.2f})"
                )

        except Exception as e:
            logger.error(f"Sentiment manipulation detection failed: {e}")
        finally:
            db.close()

        logger.info(f"sentiment_manipulation: {len(signals)} signals")
        return signals


    # ── Layer 3: LLM threat classification ────────────────────────────────────

    def classify_threat(self, cluster_label: str, sample_titles: list[str]) -> Optional[ThreatAssessment]:
        """Use LLM to classify a suspicious cluster as a threat type."""
        if not sample_titles:
            return None

        system = """You are an information operations analyst specializing in threats to India.
You analyze news article clusters for signs of coordinated inauthentic behavior.
Classify the cluster and return JSON matching the ThreatAssessment schema.
Be conservative — only flag as coordinated if there are clear signals."""

        titles_str = "\n".join(f"- {t}" for t in sample_titles[:10])
        user = f"""Cluster topic: {cluster_label}

Sample article titles:
{titles_str}

Assess whether this cluster shows signs of coordinated information operations.
Return JSON: {{is_coordinated, threat_type, actor_attribution, target_domain, confidence, reasoning}}"""

        try:
            resp = self.llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            import json
            data = json.loads(resp.content)
            return ThreatAssessment(**data)
        except Exception as e:
            logger.error(f"DisinfoDetector LLM classify failed: {e}")
            return None

    # ── Run full detection cycle ───────────────────────────────────────────────

    def run(self) -> list[dict]:
        """Run all detection layers and persist signals to Postgres."""
        all_signals = []
        all_signals.extend(self.detect_coordinated_narratives())
        all_signals.extend(self.detect_sentiment_manipulation())

        if all_signals:
            self._persist_signals(all_signals)

        logger.info(f"DisinfoDetector: {len(all_signals)} total signals")
        return all_signals

    def _persist_signals(self, signals: list[dict]) -> None:
        db = SessionLocal()
        try:
            expires = datetime.now(_IST) + timedelta(hours=24)
            for s in signals:
                if s.get("confidence", 0) < _SAVE_THRESHOLD:
                    continue
                stmt = (
                    pg_insert(DisinfoSignal)
                    .values(
                        signal_type=s["signal_type"],
                        severity=s.get("severity", "medium"),
                        confidence=s["confidence"],
                        actor_attribution=s.get("actor_attribution"),
                        target_entity=s.get("target_entity"),
                        target_domain=s.get("target_domain"),
                        cluster_id=s.get("cluster_id"),
                        narrative_summary=s.get("narrative_summary"),
                        coordination_score=s.get("coordination_score", 0.0),
                        evidence_articles=s.get("evidence_articles", []),
                        flagged_sources=s.get("flagged_sources", []),
                        expires_at=expires,
                    )
                    .on_conflict_do_nothing()
                )
                db.execute(stmt)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"DisinfoDetector persist failed: {e}")
        finally:
            db.close()
