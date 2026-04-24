"""Autonomous domain report generator.

Pulls structured data from Neo4j (entities, relations, events) and article
full-text from Postgres, then synthesizes a domain-specific intelligence
briefing via Groq LLM.

Domain filtering uses LLM-generated weights that are cached daily in Postgres.
The LLM scores entity types and relation types for domain relevance based on
a sample of what the knowledge graph actually contains — weights drift with
the data rather than relying on static config.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from sqlalchemy import select

from config import NEO4J_URI, NEO4J_AUTH
from models.database import SessionLocal
from models.scraped_article import ScrapedArticle
from models.domain_weight_cache import DomainWeightCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static fallback: used only when LLM weight generation fails
# ---------------------------------------------------------------------------

DOMAIN_CONFIG: dict[str, list[str]] = {
    "climate":     ["Resource", "Technology", "Policy", "Organization"],
    "defence":     ["Military_Asset", "Country", "Person", "Organization"],
    "economics":   ["Economic_Indicator", "Organization", "Country"],
    "geopolitics": ["Country", "Person", "Organization"],
    "society":     ["Person", "Organization", "Location"],
}


# ---------------------------------------------------------------------------
# Domain weights model (LLM structured output)
# ---------------------------------------------------------------------------

class DomainWeights(BaseModel):
    """LLM-generated relevance weights for a specific domain."""
    entity_weights: dict[str, float] = Field(
        description="Entity type → relevance score (0.0 = irrelevant, 1.0 = core to this domain)"
    )
    relation_weights: dict[str, float] = Field(
        description="Relation type → relevance score (0.0 = irrelevant, 1.0 = core to this domain)"
    )


# ---------------------------------------------------------------------------
# Pydantic output model (used for JsonOutputParser)
# ---------------------------------------------------------------------------

class KeyDevelopment(BaseModel):
    title: str = Field(description="Short headline for this development")
    details: str = Field(description="2-3 sentence analysis of this development")
    entities: list[str] = Field(description="Key entities involved")
    date: str = Field(description="Date or period, e.g. 'March 2026', 'ongoing'")


class KeyActor(BaseModel):
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type (Person, Country, etc.)")
    role: str = Field(description="Role in this domain, 1-2 sentences")


class CriticalRelationship(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation_type: str = Field(description="Type of relationship")
    analysis: str = Field(description="Why this relationship matters, 1-2 sentences")


class DomainBriefing(BaseModel):
    executive_summary: str = Field(description="2-3 paragraph overview of domain situation")
    key_developments: list[KeyDevelopment] = Field(description="5-8 important developments")
    key_actors: list[KeyActor] = Field(description="Top 5-10 key actors in this domain")
    critical_relationships: list[CriticalRelationship] = Field(description="5-8 important relationships")
    trends: str = Field(description="2-3 paragraph analysis of trends and outlook")


# ---------------------------------------------------------------------------
# Structured result for multi-agent orchestration
# ---------------------------------------------------------------------------

@dataclass
class ReportResult:
    """Structured output from ReportAgent.generate_with_context().

    Cleanly separates the final briefing from intermediate data that
    downstream agents (e.g. IndiaImpactAgent) may need.
    """
    briefing: dict                          # DomainBriefing output dict
    graph_data: dict                        # Raw graph data (entities, relations, events)
    articles: list = field(default_factory=list)  # Fetched article excerpts


# ---------------------------------------------------------------------------
# Cypher date-parse fragment (same as graph.py)
# ---------------------------------------------------------------------------

_PARSE_DATE = """
    CASE
        WHEN a.pub_date CONTAINS ','
        THEN split(split(a.pub_date, ', ')[1], ' ')[2] + '-' +
             CASE split(split(a.pub_date, ', ')[1], ' ')[1]
                 WHEN 'Jan' THEN '01' WHEN 'Feb' THEN '02' WHEN 'Mar' THEN '03'
                 WHEN 'Apr' THEN '04' WHEN 'May' THEN '05' WHEN 'Jun' THEN '06'
                 WHEN 'Jul' THEN '07' WHEN 'Aug' THEN '08' WHEN 'Sep' THEN '09'
                 WHEN 'Oct' THEN '10' WHEN 'Nov' THEN '11' WHEN 'Dec' THEN '12'
                 ELSE '01' END + '-' +
             split(split(a.pub_date, ', ')[1], ' ')[0]
        ELSE a.pub_date
    END
"""


# ---------------------------------------------------------------------------
# LLM Retry helper
# ---------------------------------------------------------------------------

def _llm_invoke_with_retry(llm, messages, max_retries: int = 3, base_delay: float = 2.0):
    """Invoke LLM with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err_name = type(e).__name__
            is_retryable = any(kw in err_name for kw in
                               ["Connection", "Timeout", "RateLimit", "ServiceUnavailable"])
            if not is_retryable or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"LLM call failed ({err_name}), retry in {delay}s ({attempt + 1}/{max_retries})")
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Report Agent
# ---------------------------------------------------------------------------

class ReportAgent:
    """Generates domain-specific intelligence briefings.

    Uses LLM-generated domain weights (cached daily in Postgres) to score
    entities and relations for domain relevance. All scoring happens in
    Cypher via map parameter lookups — no data pulled to Python for filtering.
    """

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.3, num_predict=4096)
        self.weight_llm = ChatOllama(
            model="gpt-oss:120b-cloud", temperature=0.1, num_predict=2048,
        ).with_structured_output(DomainWeights, method="json_schema")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        self.parser = JsonOutputParser(pydantic_object=DomainBriefing)
        logger.info("ReportAgent initialized")

    def close(self):
        self.driver.close()

    # ── Stage 0: Dynamic Domain Weights ─────────────────────────────────────

    def _sample_graph_types(self, date_cutoff: str) -> dict:
        """Sample entity types and relation types from recent graph data."""
        with self.driver.session() as session:
            entity_result = session.run(f"""
                MATCH (e:Entity)<-[:EVIDENCES]-(a:Article)
                WHERE a.pub_date IS NOT NULL AND a.pub_date <> ''
                  AND ({_PARSE_DATE}) >= $date_cutoff
                WITH e.type AS etype, count(DISTINCT e) AS cnt,
                     collect(DISTINCT e.name)[0..5] AS samples
                ORDER BY cnt DESC
                RETURN collect({{type: etype, count: cnt, samples: samples}}) AS entity_types
            """, date_cutoff=date_cutoff).single()

            rel_result = session.run("""
                MATCH ()-[r:RELATES_TO]->()
                WITH r.type AS rtype, count(r) AS cnt
                ORDER BY cnt DESC LIMIT 25
                RETURN collect({type: rtype, count: cnt}) AS relation_types
            """).single()

        return {
            "entity_types": entity_result["entity_types"] if entity_result else [],
            "relation_types": rel_result["relation_types"] if rel_result else [],
        }

    def _get_domain_weights(self, domain: str, date_cutoff: str) -> dict:
        """Get domain relevance weights — cached per domain per day in Postgres.

        Flow: Postgres cache → LLM structured output → static DOMAIN_CONFIG fallback.
        """
        ist = timezone(timedelta(hours=5, minutes=30))
        today = datetime.now(ist).strftime("%Y-%m-%d")

        # 1. Check Postgres cache
        db = SessionLocal()
        try:
            cached = db.execute(
                select(DomainWeightCache).where(
                    DomainWeightCache.domain == domain,
                    DomainWeightCache.cache_date == today,
                )
            ).scalar_one_or_none()

            if cached:
                logger.info(f"  Domain weights cache HIT for {domain} ({today})")
                return {
                    "entity_weights": json.loads(cached.entity_weights),
                    "relation_weights": json.loads(cached.relation_weights),
                }
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
        finally:
            db.close()

        # 2. Sample graph + LLM generation
        logger.info(f"  Domain weights cache MISS for {domain} — generating via LLM")
        try:
            graph_sample = self._sample_graph_types(date_cutoff)

            prompt = f"""You are a domain classifier for a geopolitical intelligence system.
Given the domain "{domain.upper()}" and the entity types and relation types
currently in the knowledge graph, score each for relevance to this domain.

ENTITY TYPES (with counts and example entities from the graph):
{json.dumps(graph_sample['entity_types'], indent=2)}

RELATION TYPES (with counts):
{json.dumps(graph_sample['relation_types'], indent=2)}

Score each type from 0.0 (completely irrelevant to {domain}) to 1.0 (core to {domain}).
You MUST include ALL types listed above in your output.

Examples for the "{domain}" domain:
- For climate: Resource=1.0, disrupts=0.9, attacks=0.0, Military_Asset=0.1
- For defence: Military_Asset=1.0, attacks=1.0, Economic_Indicator=0.1
- For economics: Economic_Indicator=1.0, trades_with=0.9, deployed_to=0.0"""

            weights_obj: DomainWeights = self.weight_llm.invoke(prompt)
            weights = {
                "entity_weights": weights_obj.entity_weights,
                "relation_weights": weights_obj.relation_weights,
            }

            # 3. Persist to Postgres
            db = SessionLocal()
            try:
                entry = DomainWeightCache(
                    domain=domain,
                    cache_date=today,
                    entity_weights=json.dumps(weights["entity_weights"]),
                    relation_weights=json.dumps(weights["relation_weights"]),
                )
                db.merge(entry)
                db.commit()
                logger.info(f"  Cached {domain} weights for {today}")
            except Exception as e:
                db.rollback()
                logger.warning(f"Failed to cache weights: {e}")
            finally:
                db.close()

            return weights

        except Exception as e:
            logger.error(f"LLM weight generation failed for {domain}: {e}")

        # 4. Static fallback
        logger.warning(f"  Falling back to static DOMAIN_CONFIG for {domain}")
        entity_types = DOMAIN_CONFIG.get(domain, [])
        return {
            "entity_weights": {t: 1.0 for t in entity_types},
            "relation_weights": {},  # no relation filtering in fallback
        }

    # ── Stage 1: Graph Collection ──────────────────────────────────────────

    def _collect_graph_data(self, domain: str, date_cutoff: str) -> dict:
        """Query Neo4j for domain-relevant entities scored by LLM-generated weights.

        Scoring happens entirely in Cypher via map parameter lookups:
        score = type_weight × 0.4 + avg(rel_weight) × 0.6
        """
        weights = self._get_domain_weights(domain, date_cutoff)
        type_weights = weights["entity_weights"]
        rel_weights = weights["relation_weights"]

        with self.driver.session() as session:
            # Query 1: Scored entities — map lookups in Cypher, no Python filtering
            entities_result = session.run(f"""
                MATCH (e:Entity)<-[:EVIDENCES]-(a:Article)
                WHERE a.pub_date IS NOT NULL AND a.pub_date <> ''
                  AND ({_PARSE_DATE}) >= $date_cutoff
                  AND COALESCE($type_weights[e.type], 0.0) > 0.0
                WITH e, count(DISTINCT a) AS article_count,
                     $type_weights[e.type] AS type_score
                OPTIONAL MATCH (e)-[r:RELATES_TO]-()
                WITH e, article_count, type_score,
                     avg(COALESCE($rel_weights[r.type], 0.1)) AS rel_score,
                     count(r) AS degree
                WITH e, article_count, degree,
                     (type_score * 0.4 + rel_score * 0.6) AS domain_score
                ORDER BY domain_score * log(degree + 1) DESC
                LIMIT 30
                RETURN collect({{
                    name: e.name, type: e.type,
                    degree: degree, article_count: article_count,
                    domain_score: domain_score
                }}) AS entities
            """, type_weights=type_weights, rel_weights=rel_weights,
                 date_cutoff=date_cutoff).single()

            entities = entities_result["entities"] if entities_result else []
            entity_names = [e["name"] for e in entities]

            if not entity_names:
                return {"entities": [], "relations": [], "events": [], "article_urls": []}

            # Query 2: Relations between these entities
            relations_result = session.run("""
                MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                WHERE a.name IN $names AND b.name IN $names
                RETURN collect(DISTINCT {
                    source: a.name, target: b.name,
                    type: r.type, causal: r.causal,
                    confidence: r.confidence
                }) AS relations
            """, names=entity_names).single()
            relations = relations_result["relations"] if relations_result else []

            # Query 3: Events involving these entities
            events_result = session.run("""
                MATCH (e:Entity)-[:INVOLVED_IN]->(ev:Event)
                WHERE e.name IN $names
                RETURN collect(DISTINCT {
                    name: ev.name, date: ev.date, status: ev.status,
                    entities: [(e2:Entity)-[:INVOLVED_IN]->(ev) | e2.name]
                }) AS events
            """, names=entity_names).single()
            events = events_result["events"] if events_result else []

            # Query 4: Article URLs linked to these entities
            urls_result = session.run(f"""
                MATCH (a:Article)-[:EVIDENCES]->(e:Entity)
                WHERE e.name IN $names
                  AND a.pub_date IS NOT NULL AND a.pub_date <> ''
                  AND ({_PARSE_DATE}) >= $date_cutoff
                RETURN collect(DISTINCT a.url) AS urls
            """, names=entity_names, date_cutoff=date_cutoff).single()
            article_urls = urls_result["urls"] if urls_result else []

        return {
            "entities": entities,
            "relations": relations,
            "events": events[:20],
            "article_urls": article_urls,
        }

    # ── Stage 2: Article Enrichment ────────────────────────────────────────

    def _fetch_articles(self, urls: list[str], max_articles: int = 15,
                        max_chars: int = 3000) -> list[dict]:
        """Fetch article full_text from Postgres."""
        if not urls:
            return []

        db = SessionLocal()
        try:
            result = db.execute(
                select(
                    ScrapedArticle.url,
                    ScrapedArticle.title,
                    ScrapedArticle.source,
                    ScrapedArticle.full_text,
                    ScrapedArticle.pub_date,
                ).where(ScrapedArticle.url.in_(urls[:max_articles]))
            ).all()

            articles = []
            for row in result:
                text = (row.full_text or "")[:max_chars]
                articles.append({
                    "title": row.title,
                    "source": row.source,
                    "url": row.url,
                    "pub_date": row.pub_date,
                    "excerpt": text,
                })
            return articles
        finally:
            db.close()

    # ── Stage 3: LLM Synthesis ─────────────────────────────────────────────

    def _synthesize(self, domain: str, graph_data: dict,
                    articles: list[dict]) -> dict:
        """Synthesize a structured domain briefing via Groq LLM."""

        # Build context
        entity_lines = "\n".join(
            f"- {e['name']} ({e['type']}) — {e['degree']} connections, {e['article_count']} articles"
            for e in graph_data["entities"][:20]
        )

        relation_lines = "\n".join(
            f"- {r['source']} → [{r['type']}] → {r['target']}"
            + (f" (causal)" if r.get("causal") else "")
            for r in graph_data["relations"][:30]
        )

        event_lines = "\n".join(
            f"- {ev['name']} (date: {ev.get('date', '?')}, status: {ev.get('status', '?')})"
            for ev in graph_data["events"][:15]
        )

        article_lines = "\n\n".join(
            f"### {a['title']} ({a['source']}, {a['pub_date']})\n{a['excerpt']}"
            for a in articles[:10]
        )

        system_prompt = f"""You are a strategic intelligence analyst. Generate a structured {domain.upper()} domain briefing based on the knowledge graph data and source articles provided.

Your audience is senior analysts and policymakers. Be analytical, not just descriptive. Identify patterns, implications, and connections.

IMPORTANT:
- Only reference entities and facts that appear in the provided data
- Do not hallucinate entities, events, or relationships
- Be specific with dates, names, and numbers
- Highlight causal relationships and temporal patterns
- Focus ONLY on {domain}-related insights; ignore unrelated data

{self.parser.get_format_instructions()}
"""

        user_prompt = f"""Generate a {domain.upper()} intelligence briefing.

## Key Entities
{entity_lines or "No entities found"}

## Relationships
{relation_lines or "No relationships found"}

## Events
{event_lines or "No events found"}

## Source Articles
{article_lines or "No articles available"}
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = _llm_invoke_with_retry(self.llm, messages)
        try:
            report = self.parser.parse(response.content)
        except Exception as e:
            logger.error(f"Failed to parse LLM output for {domain}: {e}")
            # Return raw text as executive_summary fallback
            report = {
                "executive_summary": response.content,
                "key_developments": [],
                "key_actors": [],
                "critical_relationships": [],
                "trends": "",
            }

        return report

    # ── Main entry points ──────────────────────────────────────────────────

    def generate_with_context(self, domain: str, date_range: str = "7d") -> "ReportResult":
        """Generate a domain briefing and return structured result with context.

        Returns a ReportResult dataclass so the orchestrator can access
        graph_data and articles without polluting the briefing dict.
        """
        if domain not in DOMAIN_CONFIG:
            raise ValueError(f"Unknown domain: {domain}. Must be one of {list(DOMAIN_CONFIG.keys())}")

        # Compute date cutoff
        days_map = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}
        days = days_map.get(date_range, 7)
        ist = timezone(timedelta(hours=5, minutes=30))
        cutoff = (datetime.now(ist).date() - timedelta(days=days)).isoformat()

        logger.info(f"Generating {domain} report (date_range={date_range}, cutoff={cutoff})")

        # Stage 1: Collect graph data
        graph_data = self._collect_graph_data(domain, cutoff)
        logger.info(
            f"  Graph data: {len(graph_data['entities'])} entities, "
            f"{len(graph_data['relations'])} relations, "
            f"{len(graph_data['events'])} events, "
            f"{len(graph_data['article_urls'])} articles"
        )

        if not graph_data["entities"]:
            logger.warning(f"No graph data found for domain={domain}, skipping synthesis")
            empty_briefing = {
                "domain": domain,
                "date_range": date_range,
                "generated_at": datetime.now(ist).isoformat(),
                "executive_summary": f"No significant {domain} data found in the knowledge graph for the past {date_range}.",
                "key_developments": [],
                "key_actors": [],
                "critical_relationships": [],
                "trends": "Insufficient data for trend analysis.",
                "sources": [],
            }
            return ReportResult(briefing=empty_briefing, graph_data=graph_data, articles=[])

        # Stage 2: Fetch articles
        articles = self._fetch_articles(graph_data["article_urls"])
        logger.info(f"  Fetched {len(articles)} articles from Postgres")

        # Stage 3: Synthesize
        report = self._synthesize(domain, graph_data, articles)
        logger.info(f"  Report synthesized for {domain}")

        # Add metadata
        report["domain"] = domain
        report["date_range"] = date_range
        report["generated_at"] = datetime.now(ist).isoformat()
        report["sources"] = [
            {"title": a["title"], "url": a["url"], "source": a["source"]}
            for a in articles
        ]

        return ReportResult(briefing=report, graph_data=graph_data, articles=articles)

    def generate(self, domain: str, date_range: str = "7d") -> dict:
        """Generate a full domain briefing.

        Args:
            domain: One of climate, defence, economics, geopolitics, society
            date_range: "7d", "30d", etc.

        Returns:
            Structured report dict matching DomainBriefing schema.
        """
        return self.generate_with_context(domain, date_range).briefing
