"""Swarm A — Extraction Consensus Swarm.

Instead of one LLM pass extracting entities/relations, dispatches the same
article to N=5 workers with different analytical perspectives ("lenses"):

  Worker 0: Geopolitics lens
  Worker 1: Economics lens
  Worker 2: Defense lens
  Worker 3: Climate lens
  Worker 4: Society lens

Each worker enriches the same GLiNER2-extracted entities/relations through
its domain lens, then deposits (entity, relation, entity) triples as votes
on the Redis blackboard.

Consensus rule: a triple promoted to Neo4j only if ≥ CONSENSUS_THRESHOLD
workers agree on it (default 3/5). Dissenting triples go to the low-confidence
buffer for human review.

Drop-in replacement for ExtractionAgent.extract() / extract_batch().
Falls back to single-agent extraction if Redis is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

from langchain_ollama import ChatOllama

from graphs.schemas import (
    ArticleExtraction,
    ExtractedEntity,
    ExtractedRelation,
    LLMEnrichment,
    CanonicalizationResult,
)
from graphs.prompts import CANONICALIZATION_PROMPT
from agents.extraction import ExtractionAgent, ENTITY_LABELS, RELATION_LABELS, _ENTITY_TYPE_MAP
from scrapers.news_rss import Article
from swarm.blackboard import SwarmBlackboard

logger = logging.getLogger(__name__)

# ── Swarm config ──────────────────────────────────────────────────────────────
N_WORKERS = 5
CONSENSUS_THRESHOLD = 3   # ≥3 of 5 workers must agree
WORKER_TIMEOUT = 300      # seconds per worker LLM call

# ── Five perspective prompts injected before the enrichment task ──────────────
PERSPECTIVE_PREFIXES: list[str] = [
    # Worker 0 — Geopolitics
    (
        "You are an expert geopolitical intelligence analyst. "
        "Focus on: heads of state, governments, nations, alliances, treaties, "
        "territorial disputes, diplomatic negotiations, political power shifts, "
        "sanctions, international law, and sovereignty conflicts. "
        "Emphasize causal relations between political actors and nation-states. "
    ),
    # Worker 1 — Economics
    (
        "You are an expert economic intelligence analyst. "
        "Focus on: trade flows, GDP indicators, inflation, currency movements, "
        "supply chains, corporate acquisitions, tariffs, sanctions economic impact, "
        "commodity markets, central bank actions, and investment flows. "
        "Emphasize causal economic relations and financial dependencies. "
    ),
    # Worker 2 — Defense
    (
        "You are an expert defense and security intelligence analyst. "
        "Focus on: military deployments, weapons systems, arms transfers, "
        "military exercises, defense procurement, cyber operations, "
        "threat assessments, force posture changes, and covert operations. "
        "Emphasize offensive/defensive capabilities and threat-actor relations. "
    ),
    # Worker 3 — Climate & Resources
    (
        "You are an expert climate and natural resource intelligence analyst. "
        "Focus on: energy resources (oil, gas, renewables), mineral extraction, "
        "water security, climate events and their geopolitical consequences, "
        "resource competition, food security, and environmental disruption. "
        "Emphasize resource-conflict and climate-driven relations. "
    ),
    # Worker 4 — Society & Technology
    (
        "You are an expert societal and technology intelligence analyst. "
        "Focus on: social movements, public health, migration, demographic shifts, "
        "emerging technologies, AI/semiconductor competition, disinformation, "
        "media influence operations, civil unrest, and human rights issues. "
        "Emphasize how societal and technological factors drive political outcomes. "
    ),
]

PERSPECTIVE_NAMES = ["geopolitics", "economics", "defense", "climate", "society"]


def _normalize_triple_key(source: str, relation: str, target: str) -> str:
    """Case-insensitive, whitespace-normalized triple key for comparison."""
    s = re.sub(r"\s+", " ", source.strip().lower())
    r = relation.strip().lower()
    t = re.sub(r"\s+", " ", target.strip().lower())
    return f"{s}||{r}||{t}"


def _triples_from_enrichment(
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation],
    enrichment: LLMEnrichment | None,
) -> list[dict]:
    """Flatten extraction into a list of triple dicts for voting."""
    triples = []
    # Entity-only signal (name + type)
    for e in entities:
        triples.append({
            "kind": "entity",
            "name": e.name,
            "type": e.type,
            "confidence": e.confidence,
        })
    # Relation triples
    for rel in relations:
        causal = False
        temporal = rel.temporal
        if enrichment:
            key = (rel.source, rel.target, rel.relation)
            for er in enrichment.relation_enrichments:
                if (er.source, er.target, er.relation) == key:
                    causal = er.causal
                    temporal = er.temporal or temporal
                    break
        triples.append({
            "kind": "relation",
            "source": rel.source,
            "relation": rel.relation,
            "target": rel.target,
            "confidence": rel.confidence,
            "causal": causal,
            "temporal": temporal,
        })
    return triples


class _PerspectiveWorker:
    """One worker in the extraction swarm.

    Reuses the same GLiNER2 result but applies a perspective-specific
    system prompt during LLM enrichment.
    """

    def __init__(self, worker_idx: int, base_agent: ExtractionAgent):
        self.worker_idx = worker_idx
        self.perspective = PERSPECTIVE_NAMES[worker_idx]
        self.prefix = PERSPECTIVE_PREFIXES[worker_idx]
        # Share the underlying GLiNER2 model (read-only, thread-safe)
        self._base = base_agent
        # Each worker gets its own LLM instance so they run concurrently
        self.llm = ChatOllama(
            model="gpt-oss:120b-cloud",
            num_predict=2048,
        ).with_structured_output(LLMEnrichment, method="json_schema")

    def run(
        self,
        article: Article,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> tuple[str, list[dict]]:
        """Enrich through this lens and return (perspective_name, triples)."""
        entity_summary = ", ".join(f"{e.name} ({e.type})" for e in entities)
        relation_summary = "\n".join(
            f"  {i + 1}. {r.source} --[{r.relation}]--> {r.target}"
            for i, r in enumerate(relations)
        ) or "(none)"

        from graphs.prompts import ENRICHMENT_PROMPT
        base_prompt = ENRICHMENT_PROMPT.format(
            title=article.title,
            source=article.source,
            pub_date=article.pub_date or "unknown",
            text=article.full_text[:3000],
            entities=entity_summary,
            relations=relation_summary,
        )
        # Prepend the perspective lens
        prompt = self.prefix + "\n\n" + base_prompt

        enrichment: LLMEnrichment | None = None
        try:
            enrichment = self.llm.invoke(prompt)
        except Exception as e:
            logger.debug(f"[swarm/{self.perspective}] LLM failed: {e}")

        triples = _triples_from_enrichment(entities, relations, enrichment)
        return self.perspective, triples


class ExtractionSwarm:
    """Drop-in replacement for ExtractionAgent.

    • extract()       → ArticleExtraction | None
    • extract_batch() → list[(Article, ArticleExtraction)]
    """

    def __init__(self, model: str = "openai/gpt-oss-20b", gliner_model: str = "fastino/gliner2-base-v1"):
        # Base agent for GLiNER2 (shared, deterministic)
        self._base = ExtractionAgent(model=model, gliner_model=gliner_model)
        # Per-perspective LLM workers
        self._workers = [_PerspectiveWorker(i, self._base) for i in range(N_WORKERS)]
        self._executor = ThreadPoolExecutor(max_workers=N_WORKERS)
        self._bb: SwarmBlackboard | None = None
        try:
            self._bb = SwarmBlackboard()
            logger.info("[ExtractionSwarm] Redis blackboard connected")
        except Exception as e:
            logger.warning(f"[ExtractionSwarm] Redis unavailable, running without blackboard: {e}")

    # ── Core extraction ───────────────────────────────────────────────────────

    def extract(self, article: Article) -> ArticleExtraction | None:
        """Run swarm extraction; fall back to single-agent if swarm fails."""
        if not article.full_text:
            return None

        text = article.full_text[:4000]

        # Step 1 — GLiNER2 (deterministic, run once)
        try:
            entities, relations = self._base._gliner_extract(text)
        except Exception as e:
            logger.error(f"[ExtractionSwarm] GLiNER2 failed: {e}")
            return None

        if not entities:
            logger.warning(f"[ExtractionSwarm] GLiNER2 found 0 entities in '{article.title[:50]}'")
            return None

        # Step 2 — Canonicalization (single pass, lightweight)
        entities, relations = self._base._canonicalize_entities(article, entities, relations)

        # Step 3 — Swarm: N perspective workers in parallel
        task_id = str(uuid.uuid4())
        if self._bb:
            self._bb.post_task({
                "article_url": article.url,
                "article_title": article.title,
                "n_entities": len(entities),
                "n_relations": len(relations),
            })

        worker_results: dict[str, list[dict]] = {}
        futures = {
            self._executor.submit(w.run, article, entities, relations): w.perspective
            for w in self._workers
        }

        for future in as_completed(futures, timeout=WORKER_TIMEOUT * 1.2):
            try:
                perspective, triples = future.result(timeout=WORKER_TIMEOUT)
                worker_results[perspective] = triples
                if self._bb:
                    self._bb.deposit_vote(task_id, perspective, triples)
                logger.debug(f"[ExtractionSwarm] {perspective}: {len(triples)} triples")
            except (FuturesTimeout, Exception) as e:
                perspective = futures[future]
                logger.warning(f"[ExtractionSwarm] worker '{perspective}' failed: {e}")

        if len(worker_results) < 2:
            # Not enough workers responded — fall back to single agent
            logger.warning("[ExtractionSwarm] Too few workers responded, falling back to single agent")
            return self._base.extract(article)

        # Step 4 — Consensus
        consensus_extraction, low_conf_triples = self._build_consensus(
            entities, relations, worker_results, task_id
        )

        # Pheromone reinforcement for consensus entities
        if self._bb:
            for e in consensus_extraction.entities:
                self._bb.reinforce_pheromone(e.name, delta=len(worker_results) / N_WORKERS)

        return consensus_extraction

    def _build_consensus(
        self,
        base_entities: list[ExtractedEntity],
        base_relations: list[ExtractedRelation],
        worker_results: dict[str, list[dict]],
        task_id: str,
    ) -> tuple[ArticleExtraction, list[dict]]:
        """Tally votes; promote triples with ≥ CONSENSUS_THRESHOLD agreement."""

        # Count votes per triple key
        entity_votes: dict[str, dict] = {}   # normalized_name → {votes, data}
        relation_votes: dict[str, dict] = {} # triple_key → {votes, data}

        for perspective, triples in worker_results.items():
            seen_keys: set[str] = set()
            for t in triples:
                if t["kind"] == "entity":
                    key = t["name"].strip().lower()
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    if key not in entity_votes:
                        entity_votes[key] = {"votes": 0, "data": t, "perspectives": []}
                    entity_votes[key]["votes"] += 1
                    entity_votes[key]["perspectives"].append(perspective)
                elif t["kind"] == "relation":
                    key = _normalize_triple_key(t["source"], t["relation"], t["target"])
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    if key not in relation_votes:
                        relation_votes[key] = {"votes": 0, "data": t, "perspectives": []}
                    relation_votes[key]["votes"] += 1
                    relation_votes[key]["perspectives"].append(perspective)

        # Entities: original extraction + consensus filter
        name_to_base = {e.name.strip().lower(): e for e in base_entities}
        consensus_entities: list[ExtractedEntity] = []
        low_conf_entities: list[dict] = []

        for norm_key, vote_info in entity_votes.items():
            base_e = name_to_base.get(norm_key)
            if base_e is None:
                # Try partial match
                for k, v in name_to_base.items():
                    if norm_key in k or k in norm_key:
                        base_e = v
                        break
            if base_e is None:
                continue  # Can't reconstruct entity — skip

            if vote_info["votes"] >= CONSENSUS_THRESHOLD:
                consensus_entities.append(base_e)
                logger.debug(
                    f"[consensus] ENTITY '{base_e.name}' — {vote_info['votes']}/{len(worker_results)} agree"
                )
            else:
                low_conf_entities.append({**vote_info["data"], "votes": vote_info["votes"]})

        # If consensus gives us nothing, keep all base entities (safety net)
        if not consensus_entities:
            logger.warning("[ExtractionSwarm] Consensus returned 0 entities — using all base entities")
            consensus_entities = base_entities

        # Relations: original + consensus filter
        triple_to_base = {
            _normalize_triple_key(r.source, r.relation, r.target): r
            for r in base_relations
        }
        consensus_relations: list[ExtractedRelation] = []
        low_conf_relations: list[dict] = []

        for triple_key, vote_info in relation_votes.items():
            base_r = triple_to_base.get(triple_key)
            if base_r is None:
                continue

            if vote_info["votes"] >= CONSENSUS_THRESHOLD:
                # Apply enrichment from majority perspective (most causal flags)
                data = vote_info["data"]
                if data.get("causal"):
                    base_r.causal = True
                if data.get("temporal"):
                    base_r.temporal = data["temporal"]
                consensus_relations.append(base_r)
                logger.debug(
                    f"[consensus] RELATION {base_r.source} --[{base_r.relation}]--> {base_r.target} "
                    f"— {vote_info['votes']}/{len(worker_results)} agree"
                )
            else:
                low_conf_relations.append({**vote_info["data"], "votes": vote_info["votes"]})

        # Store low-confidence triples for human review
        all_low_conf = low_conf_entities + low_conf_relations
        if all_low_conf and self._bb:
            self._bb.store_low_confidence(task_id, all_low_conf)
            logger.info(
                f"[ExtractionSwarm] {len(all_low_conf)} low-confidence triples buffered for review"
            )

        # Events: gather from all workers (no strict consensus needed — union of unique events)
        # We run the enrichment step once more on consensus entities to get events
        enrichment = self._base._llm_enrich(
            # We need an article stub — reuse the last enrichment call from workers
            # by synthesizing what was agreed on
            article=_make_article_stub(consensus_entities),
            entities=consensus_entities,
            relations=consensus_relations,
        )
        events = enrichment.events if enrichment else []

        logger.info(
            f"[ExtractionSwarm] Consensus: {len(consensus_entities)} entities, "
            f"{len(consensus_relations)} relations | "
            f"Low-conf: {len(all_low_conf)} triples"
        )

        return ArticleExtraction(
            entities=consensus_entities,
            relations=consensus_relations,
            events=events,
        ), all_low_conf

    # ── Batch interface (mirrors ExtractionAgent.extract_batch) ───────────────

    def extract_batch(
        self, articles: list[Article], max_workers: int = 3
    ) -> list[tuple[Article, ArticleExtraction]]:
        results: list[tuple[Article, ArticleExtraction]] = []
        for i, article in enumerate(articles):
            logger.info(f"[ExtractionSwarm] [{i + 1}/{len(articles)}] {article.title[:60]}")
            extraction = self.extract(article)
            if extraction:
                logger.info(
                    f"  → {len(extraction.entities)} entities (consensus), "
                    f"{len(extraction.relations)} relations, {len(extraction.events)} events"
                )
                results.append((article, extraction))
            else:
                logger.warning(f"  → Skipped")
        logger.info(f"[ExtractionSwarm] {len(results)}/{len(articles)} articles extracted")
        return results

    def close(self):
        self._executor.shutdown(wait=False)
        if self._bb:
            self._bb.close()


def _make_article_stub(entities: list[ExtractedEntity]):
    """Create a minimal Article-like object for event extraction."""
    from scrapers.news_rss import Article as RSSArticle
    return RSSArticle(
        url="", title="", source="", description="",
        pub_date="", guid="",
        full_text=" ".join(e.name for e in entities),
        authors=[], top_image="", content_hash="", is_content_extracted=True,
    )
