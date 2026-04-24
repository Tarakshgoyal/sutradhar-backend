"""News Priority Agent — cross-batch topic clustering and importance scoring.

Runs inside the Kafka consumer BEFORE articles are sent to GraphBuilder.

Responsibilities:
1. Embed article title+description using MiniLM (same model as ResolutionAgent)
2. Assign each article to a topic cluster, comparing against ALL active clusters
   stored in Redis (24-hour rolling window) — solves the cross-batch blindspot.
3. Pick the best representative per cluster (credibility × log(content_length)).
4. Score new cluster representatives via a single LLM call each — structured
   output with strict Pydantic schema avoids JSON truncation risk.
5. Persist ALL articles to Postgres (with scores) and return only the
   high-importance subset to send to GraphBuilder.
"""

import base64
import json
import logging
import math
import time
import uuid
from typing import Optional

import numpy as np
import redis
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config import REDIS_HOST, REDIS_PORT
from models.database import SessionLocal
from models.scraped_article import ScrapedArticle
from models.source_config import SourceConfig
from scrapers.news_rss import Article
from agents.scoring_config import ScoringConfig

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
CLUSTER_SIMILARITY_THRESHOLD = 0.82   # cosine sim to assign to existing cluster
GRAPH_IMPORTANCE_THRESHOLD = 5.0       # articles below this skip GraphBuilder
CLUSTER_TTL_SECONDS = 86400            # 24-hour rolling window in Redis
DEFAULT_CREDIBILITY = 0.70             # for unknown sources

# ── Redis keys ────────────────────────────────────────────────────────────────
CLUSTERS_ACTIVE_KEY = "india-innovates:clusters:active"   # ZSET uuid → last_seen
SOURCE_CREDIBILITY_KEY = "india-innovates:source_credibility"   # HASH source → score
SOURCE_CREDIBILITY_TTL = 3600   # refresh from Postgres hourly


# ── Pydantic schema for LLM scoring ──────────────────────────────────────────

class ArticleImportance(BaseModel):
    impact_score: int = Field(
        ge=0, le=10,
        description=(
            "Structural scale of impact — how many people, institutions, or countries "
            "are MATERIALLY affected. Score the real-world consequence, not the emotional "
            "shock value or sensationalism of the headline. "
            "0 = single individual (one person killed, one company fined). "
            "1-2 = local community or single neighbourhood. "
            "3-4 = single city or district. "
            "5-6 = national-level policy, economy, or public-health change. "
            "7-8 = multi-country impact, major alliance shift, or continental policy. "
            "9-10 = global systemic change: war escalation, nuclear posture shift, "
            "trade-regime collapse, pandemic declaration. "
            "\n\nCALIBRATION EXAMPLES — use these as anchors:\n"
            "• International military strike (US-Israel on Iran) → 9-10\n"
            "• Country lifts decades-old nuclear-weapons ban (Finland/NATO) → 8-9\n"
            "• National election result (Nepal, India state elections) → 7-8\n"
            "• Major tech antitrust ruling (Google monopoly) → 7-8\n"
            "• Domestic violent crime (murder, stabbing, assault) → 1-2\n"
            "• Celebrity / influencer incident → 1-2\n"
            "• Scheduled awareness day (World Obesity Day) → 3-4"
        ),
    )
    novelty_score: int = Field(
        ge=0, le=10,
        description=(
            "How NEW or UNPRECEDENTED the development is — not merely how recently "
            "the article was published. Score the historical novelty of the event itself. "
            "0 = routine, scheduled, or fully expected event. "
            "1-2 = ongoing known situation with no new dimension. "
            "3-4 = incremental update on a known story. "
            "5-6 = significant policy change or notable escalation. "
            "7-8 = first-time event or historic precedent being set. "
            "9-10 = completely unprecedented — nothing comparable in recent history. "
            "\n\nCALIBRATION EXAMPLES:\n"
            "• Scheduled awareness day (World X Day, International Y Day) → 0-1\n"
            "• Recurring seasonal weather pattern → 1-2\n"
            "• Country lifting a decades-old weapons ban → 8-9\n"
            "• First-ever military strike between two specific nations → 9-10\n"
            "• A well-known political figure making a routine statement → 1-2"
        ),
    )
    india_relevance: int = Field(
        ge=0, le=10,
        description=(
            "Direct relevance to India's strategic, economic, or security interests. "
            "0 = no connection to India whatsoever. "
            "1-2 = tangential — involves Indian nationals abroad with no policy dimension, "
            "or a domestic crime with no systemic implications. "
            "3-4 = indirect thematic link (global trend that may eventually affect India). "
            "5-6 = directly involves India or a key bilateral partner. "
            "7-8 = materially impacts India's security, economy, or trade. "
            "9-10 = existential or immediate threat to Indian sovereignty or stability. "
            "\n\nCALIBRATION EXAMPLES:\n"
            "• Domestic crime (Indian victim, no policy angle) → 1-2\n"
            "• India-specific trade war or tariff dispute → 8-9\n"
            "• India military asset crash (Su-30MKI) → 8-9\n"
            "• Foreign policy shift by a non-partner nation → 2-3\n"
            "• Global pandemic declaration → 6-7"
        ),
    )
    domain: str = Field(
        description=(
            "Lowercase topic domain for the article. You MUST pick from this exact list: "
            "geopolitics, defense, economics, technology, energy, health, politics, elections, "
            "crime, human_interest, environment, science, sports, education, infrastructure, "
            "judiciary, diplomacy. "
            "Only if the article genuinely does not fit ANY of these, use a concise single-word "
            "or hyphenated label of your own. Do NOT use synonyms — e.g. use 'defense' not "
            "'military', 'geopolitics' not 'geo-politics', 'economics' not 'economy'."
        )
    )
    cluster_label: str = Field(
        description=(
            "3-6 word title-case label for the topic cluster. "
            "Write the TOPIC, not the headline. "
            "Good: 'India US Tariff Dispute'. Bad: 'Modi Slams Washington Over New Levies'."
        )
    )


# ── System prompt (static, separated from article data) ──────────────────────
SCORING_SYSTEM_PROMPT = (
    "You are a news-importance scorer for an India-focused geopolitical intelligence "
    "platform. Your sole job is to fill in the structured scoring fields for one "
    "article at a time. Assess OBJECTIVE IMPORTANCE — the structural, geopolitical, "
    "and policy significance of the event — never its emotional shock value or "
    "sensationalism. The field descriptions contain detailed rubrics and calibration "
    "examples; follow them precisely."
)


class NewsPriorityAgent:
    """Cross-batch topic clustering + LLM importance scoring."""

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        embedding_model: str = "all-MiniLM-L6-v2",
        db_session=None,
    ):
        self.embedder = SentenceTransformer(embedding_model)
        self.llm = ChatOllama(model="gpt-oss:120b-cloud").with_structured_output(ArticleImportance)
        self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        # Initialize database session and scoring config
        if db_session:
            self.db = db_session
            self._owns_db = False
        else:
            self.db = SessionLocal()
            self._owns_db = True

        self.scoring_config = ScoringConfig(self.db)
        logger.info("NewsPriorityAgent initialized")

    # ── Source credibility ────────────────────────────────────────────────────

    def _load_source_credibility(self) -> dict[str, float]:
        """Return source → credibility map. Redis cache, fallback to Postgres."""
        cached = self.r.hgetall(SOURCE_CREDIBILITY_KEY)
        if cached:
            return {k: float(v) for k, v in cached.items()}

        db = SessionLocal()
        try:
            rows = db.query(SourceConfig).filter(SourceConfig.active.is_(True)).all()
            cred_map = {row.source_name: row.credibility_score for row in rows}
            if cred_map:
                self.r.hset(SOURCE_CREDIBILITY_KEY, mapping={k: str(v) for k, v in cred_map.items()})
                self.r.expire(SOURCE_CREDIBILITY_KEY, SOURCE_CREDIBILITY_TTL)
            return cred_map
        finally:
            db.close()

    def _article_score(self, article: Article, cred_map: dict[str, float]) -> float:
        """Selection score for choosing the best cluster representative."""
        credibility = cred_map.get(article.source, DEFAULT_CREDIBILITY)
        content_len = max(len(article.full_text), 1)
        return credibility * math.log(content_len)

    # ── Redis cluster helpers ─────────────────────────────────────────────────

    def _load_active_clusters(self) -> list[tuple[str, np.ndarray, dict]]:
        """Load all active cluster centroids from Redis, expiring stale ones."""
        now = time.time()
        cutoff = now - CLUSTER_TTL_SECONDS
        self.r.zremrangebyscore(CLUSTERS_ACTIVE_KEY, "-inf", cutoff)

        cluster_uuids = self.r.zrangebyscore(CLUSTERS_ACTIVE_KEY, cutoff, "+inf")
        result: list[tuple[str, np.ndarray, dict]] = []
        for cluster_uuid in cluster_uuids:
            meta = self.r.hgetall(f"cluster:{cluster_uuid}")
            if not meta or "centroid_b64" not in meta:
                continue
            try:
                centroid = np.frombuffer(
                    base64.b64decode(meta["centroid_b64"]), dtype=np.float32
                ).copy()
                result.append((cluster_uuid, centroid, meta))
            except Exception as e:
                logger.warning(f"Skipping corrupt cluster {cluster_uuid}: {e}")
        return result

    def _create_cluster(
        self, cluster_uuid: str, article: Article, article_score: float, centroid: np.ndarray
    ) -> None:
        now = time.time()
        self.r.hset(
            f"cluster:{cluster_uuid}",
            mapping={
                "centroid_b64": base64.b64encode(centroid.astype(np.float32).tobytes()).decode(),
                "best_article_url": article.url,
                "best_article_score": str(article_score),
                "importance_score": "5.0",  # neutral until LLM scores
                "cluster_label": "",
                "domain": "other",
                "article_count": "1",
                "first_seen": str(now),
                "last_seen": str(now),
                "graphed": "0",
            },
        )
        self.r.zadd(CLUSTERS_ACTIVE_KEY, {cluster_uuid: now})

    def _update_cluster(
        self, cluster_uuid: str, article: Article, article_score: float, centroid: np.ndarray
    ) -> None:
        now = time.time()
        meta = self.r.hgetall(f"cluster:{cluster_uuid}")
        if not meta:
            return

        current_best = float(meta.get("best_article_score", "0"))
        count = int(meta.get("article_count", "1"))

        updates: dict[str, str] = {
            "article_count": str(count + 1),
            "last_seen": str(now),
        }

        if article_score > current_best:
            updates["best_article_url"] = article.url
            updates["best_article_score"] = str(article_score)
            # Rolling centroid average (keeps centroid drifting toward cluster core)
            old_centroid = np.frombuffer(
                base64.b64decode(meta["centroid_b64"]), dtype=np.float32
            ).copy()
            new_centroid = (old_centroid * count + centroid) / (count + 1)
            norm = np.linalg.norm(new_centroid)
            if norm > 0:
                new_centroid /= norm
            updates["centroid_b64"] = base64.b64encode(
                new_centroid.astype(np.float32).tobytes()
            ).decode()

        self.r.hset(f"cluster:{cluster_uuid}", mapping=updates)
        self.r.zadd(CLUSTERS_ACTIVE_KEY, {cluster_uuid: now})

    # Domain-based weight adjustments are now loaded from database via ScoringConfig
    # See agents/scoring_config.py and scoring_weights table

    def _store_cluster_score(
        self, cluster_uuid: str, importance: ArticleImportance, article_count: int = 1
    ) -> None:
        # Get formula weights from database
        formula_weights = self.scoring_config.get_formula_weights()

        # Base score: impact-heavy formula (impact is the primary signal)
        base = (
            formula_weights.get("impact_score", 0.5) * importance.impact_score
            + formula_weights.get("novelty_score", 0.2) * importance.novelty_score
            + formula_weights.get("india_relevance", 0.3) * importance.india_relevance
        )

        # Domain modifier from database
        domain_mult = self.scoring_config.get_domain_weight(importance.domain)

        # Coverage density bonus: more sources → higher importance (diminishing returns)
        coverage_params = self.scoring_config.get_coverage_params()
        log_base = coverage_params.get("log_base", 6.0)
        multiplier = coverage_params.get("multiplier", 0.5)
        coverage_bonus = min(
            math.log(max(article_count, 1) + 1) / math.log(log_base), 1.0
        ) * multiplier

        score = round(min((base * domain_mult) + coverage_bonus, 10.0), 2)
        self.r.hset(
            f"cluster:{cluster_uuid}",
            mapping={
                "importance_score": str(score),
                "cluster_label": importance.cluster_label,
                "domain": importance.domain,
            },
        )

    # ── Post-LLM label merge ─────────────────────────────────────────────────

    def _merge_clusters_by_label(
        self, new_cluster_uuids: set[str], assignments: dict[int, str]
    ) -> dict[int, str]:
        """Second-pass merge: collapse new clusters that share a label with an existing one.

        Cosine similarity catches near-identical phrasing. This pass catches
        same-topic articles where the LLM independently assigns the same label
        but the embedding distance was below threshold (different specific wording,
        same broad topic — e.g. 'India Iran Relations' from two different events).

        Strategy:
        - Build label → canonical_uuid map from ALL active clusters (new + old).
        - For each new cluster whose label already exists in the map, merge it
          into the canonical cluster (higher importance_score wins).
        - Merged cluster is deleted from Redis so future batches don't match its
          centroid and re-create the split.
        - Returns updated assignments so articles in this batch point to canonical.
        """
        active = self._load_active_clusters()

        # label (normalized) → (uuid, importance_score)
        label_to_canonical: dict[str, tuple[str, float]] = {}
        for cluster_uuid, _, meta in active:
            raw_label = meta.get("cluster_label", "").strip()
            if not raw_label:
                continue
            norm_label = raw_label.lower()
            score = float(meta.get("importance_score", 5.0))
            existing = label_to_canonical.get(norm_label)
            if existing is None or score > existing[1]:
                label_to_canonical[norm_label] = (cluster_uuid, score)

        # Determine which new clusters need merging
        merge_map: dict[str, str] = {}  # new_uuid → canonical_uuid
        for new_uuid in new_cluster_uuids:
            meta = self.r.hgetall(f"cluster:{new_uuid}")
            raw_label = meta.get("cluster_label", "").strip()
            if not raw_label:
                continue
            norm_label = raw_label.lower()
            canonical_uuid, _ = label_to_canonical.get(norm_label, (new_uuid, 0.0))
            if canonical_uuid != new_uuid:
                merge_map[new_uuid] = canonical_uuid

        if not merge_map:
            return assignments

        # Execute merges
        for merged_uuid, canonical_uuid in merge_map.items():
            merged_meta = self.r.hgetall(f"cluster:{merged_uuid}")
            canonical_meta = self.r.hgetall(f"cluster:{canonical_uuid}")

            merged_count = int(merged_meta.get("article_count", "1"))
            canonical_count = int(canonical_meta.get("article_count", "1"))
            merged_score = float(merged_meta.get("importance_score", "5.0"))
            canonical_score = float(canonical_meta.get("importance_score", "5.0"))

            self.r.hset(
                f"cluster:{canonical_uuid}",
                mapping={
                    "article_count": str(canonical_count + merged_count),
                    "importance_score": str(max(merged_score, canonical_score)),
                },
            )
            # Remove merged cluster so its centroid is never loaded again
            self.r.zrem(CLUSTERS_ACTIVE_KEY, merged_uuid)
            self.r.delete(f"cluster:{merged_uuid}")
            logger.info(
                f"Label merge: '{merged_meta.get('cluster_label')}' "
                f"{merged_uuid[:8]} → {canonical_uuid[:8]}"
            )

        # Redirect all article assignments from merged UUIDs to their canonical
        return {i: merge_map.get(uid, uid) for i, uid in assignments.items()}

    # ── LLM scoring ───────────────────────────────────────────────────────────

    @staticmethod
    def _truncate_at_word(text: str, max_chars: int = 500) -> str:
        """Truncate text at the last word boundary within *max_chars*."""
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars].rsplit(" ", 1)[0]
        return cut.rstrip(".,;:!?-") + " …"

    def _score_article(self, article: Article) -> Optional[ArticleImportance]:
        """Score a single cluster representative via system + human message pair."""
        user_text = (
            f"Title: {article.title}\n"
            f"Source: {article.source}\n"
            f"Description: {self._truncate_at_word(article.description, 500)}"
        )
        try:
            return self.llm.invoke([
                SystemMessage(content=SCORING_SYSTEM_PROMPT),
                HumanMessage(content=user_text),
            ])
        except Exception as e:
            logger.error(f"Importance scoring failed for '{article.title[:60]}': {e}")
            return None

    # ── Persistence & routing ─────────────────────────────────────────────────

    def _persist_and_route(
        self,
        batch: list[Article],
        assignments: dict[int, str],
        new_cluster_uuids: set[str],
    ) -> list[Article]:
        """Upsert all articles to Postgres; return subset to send to GraphBuilder."""
        db = SessionLocal()
        try:
            to_graph: list[Article] = []
            graphed_cluster_uuids: set[str] = set()

            for i, article in enumerate(batch):
                cluster_uuid = assignments[i]
                meta = self.r.hgetall(f"cluster:{cluster_uuid}")

                importance_score = float(meta.get("importance_score", 5.0))
                cluster_label = meta.get("cluster_label", "")
                domain = meta.get("domain", "other")
                already_graphed = meta.get("graphed", "0") == "1"

                stmt = (
                    pg_insert(ScrapedArticle)
                    .values(
                        url=article.url,
                        content_hash=article.content_hash,
                        title=article.title,
                        source=article.source,
                        description=article.description,
                        pub_date=article.pub_date,
                        guid=article.guid,
                        full_text=article.full_text,
                        authors=json.dumps(article.authors),
                        top_image=article.top_image,
                        is_content_extracted=article.is_content_extracted,
                        importance_score=importance_score,
                        topic_cluster_id=cluster_uuid,
                        cluster_label=cluster_label,
                        domain=domain,
                    )
                    .on_conflict_do_update(
                        index_elements=["url"],
                        set_={
                            "importance_score": importance_score,
                            "topic_cluster_id": cluster_uuid,
                            "cluster_label": cluster_label,
                            "domain": domain,
                        },
                    )
                )
                db.execute(stmt)

                if importance_score >= GRAPH_IMPORTANCE_THRESHOLD and not already_graphed:
                    to_graph.append(article)
                    graphed_cluster_uuids.add(cluster_uuid)

            db.commit()

            for cluster_uuid in graphed_cluster_uuids:
                self.r.hset(f"cluster:{cluster_uuid}", "graphed", "1")

            logger.info(
                f"Priority: {len(batch)} in → {len(set(assignments.values()))} clusters "
                f"({len(new_cluster_uuids)} new) → {len(to_graph)} to graph"
            )
            return to_graph

        except Exception as e:
            db.rollback()
            logger.error(f"Persist failed, routing all to graph as fallback: {e}", exc_info=True)
            return batch
        finally:
            db.close()

    # ── Public entry point ────────────────────────────────────────────────────

    def process(self, batch: list[Article]) -> list[Article]:
        """
        Cluster + score a batch of articles.

        Returns the subset that should be forwarded to GraphBuilder.
        All articles (including non-graphed ones) are saved to Postgres.
        """
        if not batch:
            return []

        cred_map = self._load_source_credibility()

        # Embed all articles in one forward pass
        texts = [f"{a.title} {a.description}".strip() for a in batch]
        vectors = self.embedder.encode(texts, normalize_embeddings=True).astype(np.float32)

        # Load existing clusters from Redis (cross-batch state)
        active = self._load_active_clusters()
        live_uuids: list[str] = [u for u, _, _ in active]
        live_centroids: list[np.ndarray] = [v for _, v, _ in active]

        assignments: dict[int, str] = {}
        new_cluster_uuids: set[str] = set()

        for i, article in enumerate(batch):
            vec = vectors[i]
            matched_uuid: Optional[str] = None

            if live_centroids:
                centroid_matrix = np.stack(live_centroids)      # (K, 384)
                sims = centroid_matrix @ vec                     # (K,) cosine (normalized)
                best_idx = int(np.argmax(sims))
                if float(sims[best_idx]) >= CLUSTER_SIMILARITY_THRESHOLD:
                    matched_uuid = live_uuids[best_idx]

            article_score = self._article_score(article, cred_map)

            if matched_uuid:
                assignments[i] = matched_uuid
                self._update_cluster(matched_uuid, article, article_score, vec)
            else:
                new_uuid = str(uuid.uuid4())
                assignments[i] = new_uuid
                new_cluster_uuids.add(new_uuid)
                self._create_cluster(new_uuid, article, article_score, vec)
                # Add to working list so later articles in the same batch can match it
                live_uuids.append(new_uuid)
                live_centroids.append(vec)

        # Score each new cluster representative (1 LLM call each)
        for cluster_uuid in new_cluster_uuids:
            meta = self.r.hgetall(f"cluster:{cluster_uuid}")
            best_url = meta.get("best_article_url", "")
            article_count = int(meta.get("article_count", "1"))
            rep = next((a for a in batch if a.url == best_url), None)
            if rep:
                importance = self._score_article(rep)
                if importance:
                    self._store_cluster_score(cluster_uuid, importance, article_count=article_count)

        # Post-LLM label merge: collapse clusters the LLM labelled identically
        # (catches same broad topic with different specific wording in embeddings)
        assignments = self._merge_clusters_by_label(new_cluster_uuids, assignments)

        return self._persist_and_route(batch, assignments, new_cluster_uuids)
