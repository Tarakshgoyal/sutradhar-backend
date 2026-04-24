"""Inference Agent — multi-hop causal chain discovery, impact propagation,
and weak link detection over the knowledge graph.

Design principle: graph compute first, LLM narrate last.
Steps 1-3 are pure Cypher. Step 4 is a single LLM call for narrative.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from config import NEO4J_URI, NEO4J_AUTH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain → Entity type mapping (for cross-domain detection)
# ---------------------------------------------------------------------------

DOMAIN_ENTITY_TYPES: dict[str, str] = {
    "Person": "geopolitics",
    "Organization": "economics",
    "Country": "geopolitics",
    "Location": "geopolitics",
    "Policy": "geopolitics",
    "Technology": "technology",
    "Economic_Indicator": "economics",
    "Military_Asset": "defence",
    "Resource": "climate",
}


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------

class ChainLink(BaseModel):
    entity: str = Field(description="Entity name at this step")
    entity_type: str = Field(description="Entity type (Person, Country, etc.)")
    relation: str = Field(description="Relation to the next entity in the chain")


class CausalChain(BaseModel):
    chain: list[ChainLink] = Field(description="Ordered sequence of entity→relation→entity links")
    final_entity: str = Field(description="Last entity in the chain")
    score: float = Field(description="Chain confidence score (0-1)")
    narrative: str = Field(description="2-3 sentence analyst explanation of why this chain matters")
    source_articles: list[str] = Field(default_factory=list, description="Article titles providing evidence")


class AffectedEntity(BaseModel):
    name: str = Field(description="Affected entity name")
    type: str = Field(description="Entity type")
    hop_distance: int = Field(description="Hops from the trigger event")
    via_relation: str = Field(description="Relation type through which impact propagates")


class ImpactResult(BaseModel):
    trigger_event: str = Field(description="The event triggering the downstream impact")
    event_date: str = Field(description="Date of the trigger event")
    event_status: str = Field(description="Status: ongoing, concluded, announced, rumored")
    affected: list[AffectedEntity] = Field(description="Downstream affected entities")
    narrative: str = Field(description="2-3 sentence explanation of the impact cascade")


class WeakLink(BaseModel):
    entity: str = Field(description="Bridge entity name")
    type: str = Field(description="Entity type")
    domains_bridged: list[str] = Field(description="Domains this entity connects")
    connection_count: int = Field(description="Number of relationships (low = fragile)")
    risk_narrative: str = Field(description="Why this entity is a critical vulnerability")


class InferenceAnalysis(BaseModel):
    executive_summary: str = Field(description="2-3 paragraph overview of key inferences")
    causal_chains: list[CausalChain] = Field(description="Top 3-5 multi-hop causal chains")
    impact_propagations: list[ImpactResult] = Field(description="2-4 event impact cascades")
    weak_links: list[WeakLink] = Field(description="3-5 critical bridge entities")


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
# Inference Agent
# ---------------------------------------------------------------------------

class InferenceAgent:
    """Discovers multi-hop causal chains, propagates event impact,
    and detects critical weak links in the knowledge graph.

    All graph analysis is done in pure Cypher. A single LLM call at the
    end produces analyst-readable narratives from the structured results.
    """

    def __init__(self, model: str = "openai/gpt-oss-20b", entity_classifier=None):
        self.llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.3, num_predict=4096)
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        self.parser = JsonOutputParser(pydantic_object=InferenceAnalysis)
        self.entity_classifier = entity_classifier
        logger.info("InferenceAgent initialized")

    def close(self):
        self.driver.close()

    # ── Step 1: Causal Chain Discovery ─────────────────────────────────────

    def _discover_causal_chains(self, entity_names: list[str],
                                 max_hops: int = 5,
                                 top_k: int = 5) -> list[dict]:
        """Find multi-hop causal chains across entity types via Neo4j.

        Traverses variable-length RELATES_TO paths, prioritizes causal edges,
        and scores by confidence × evidence × type diversity.
        """
        if not entity_names:
            return []

        chains = []
        with self.driver.session() as session:
            # Query 1: Causal-flag chains (most valuable)
            result = session.run("""
                MATCH path = (start:Entity)-[:RELATES_TO*2..5]->(end_node:Entity)
                WHERE start.name IN $names
                  AND start <> end_node
                  AND ALL(r IN relationships(path) WHERE r.causal = true)
                  AND start.type <> end_node.type
                WITH path, start, end_node,
                     reduce(s = 1.0, r IN relationships(path) |
                         s * coalesce(r.confidence, 0.5)) AS conf_score,
                     reduce(s = 0, r IN relationships(path) |
                         s + coalesce(r.evidence_count, 1)) AS evidence_total,
                     [n IN nodes(path) | n.type] AS type_list
                UNWIND type_list AS t
                WITH path, start, end_node, conf_score, evidence_total,
                     count(DISTINCT t) AS type_diversity
                ORDER BY conf_score * (1 + log(evidence_total + 1)) * type_diversity DESC
                LIMIT $top_k
                RETURN [n IN nodes(path) | {name: n.name, type: n.type}] AS entities,
                       [r IN relationships(path) |
                           {type: r.type, causal: r.causal,
                            confidence: r.confidence,
                            evidence_count: r.evidence_count}] AS relations,
                       conf_score, evidence_total, type_diversity
            """, names=entity_names, top_k=top_k)

            for record in result:
                chains.append({
                    "entities": record["entities"],
                    "relations": record["relations"],
                    "conf_score": record["conf_score"],
                    "evidence_total": record["evidence_total"],
                    "type_diversity": record["type_diversity"],
                })

            # Query 2: Fallback — mixed causal/non-causal if pure causal found < top_k
            if len(chains) < top_k:
                remaining = top_k - len(chains)
                # Get names already found to exclude duplicates
                found_chains = {
                    (c["entities"][0]["name"], c["entities"][-1]["name"])
                    for c in chains
                }

                fallback = session.run("""
                    MATCH path = (start:Entity)-[:RELATES_TO*2..4]->(end_node:Entity)
                    WHERE start.name IN $names
                      AND start <> end_node
                      AND start.type <> end_node.type
                      AND ANY(r IN relationships(path) WHERE r.causal = true)
                    WITH path, start, end_node,
                         reduce(s = 1.0, r IN relationships(path) |
                             s * coalesce(r.confidence, 0.5)) AS conf_score,
                         reduce(s = 0, r IN relationships(path) |
                             s + coalesce(r.evidence_count, 1)) AS evidence_total,
                         [n IN nodes(path) | n.type] AS type_list
                    UNWIND type_list AS t
                    WITH path, start, end_node, conf_score, evidence_total,
                         count(DISTINCT t) AS type_diversity
                    ORDER BY conf_score * (1 + log(evidence_total + 1)) * type_diversity DESC
                    LIMIT $remaining
                    RETURN [n IN nodes(path) | {name: n.name, type: n.type}] AS entities,
                           [r IN relationships(path) |
                               {type: r.type, causal: r.causal,
                                confidence: r.confidence,
                                evidence_count: r.evidence_count}] AS relations,
                           conf_score, evidence_total, type_diversity
                """, names=entity_names, remaining=remaining)

                for record in fallback:
                    key = (record["entities"][0]["name"], record["entities"][-1]["name"])
                    if key not in found_chains:
                        chains.append({
                            "entities": record["entities"],
                            "relations": record["relations"],
                            "conf_score": record["conf_score"],
                            "evidence_total": record["evidence_total"],
                            "type_diversity": record["type_diversity"],
                        })

            # Fetch source articles for chain entities
            if chains:
                all_chain_entities = set()
                for c in chains:
                    for e in c["entities"]:
                        all_chain_entities.add(e["name"])

                articles_result = session.run("""
                    MATCH (a:Article)-[:EVIDENCES]->(e:Entity)
                    WHERE e.name IN $names
                    RETURN e.name AS entity, collect(DISTINCT a.title)[0..3] AS articles
                """, names=list(all_chain_entities))

                entity_articles = {}
                for record in articles_result:
                    entity_articles[record["entity"]] = record["articles"]

                for c in chains:
                    c["source_articles"] = []
                    for e in c["entities"]:
                        c["source_articles"].extend(
                            entity_articles.get(e["name"], [])
                        )
                    # Deduplicate
                    c["source_articles"] = list(dict.fromkeys(c["source_articles"]))[:5]

        logger.info(f"  Causal chains: {len(chains)} discovered")
        return chains

    # ── Step 2: Impact Propagation ─────────────────────────────────────────

    def _propagate_impact(self, date_cutoff: str,
                           max_hops: int = 3,
                           max_events: int = 5) -> list[dict]:
        """Find downstream entities affected by recent events."""
        impacts = []

        with self.driver.session() as session:
            result = session.run("""
                MATCH (ev:Event)<-[:INVOLVED_IN]-(seed:Entity)
                WHERE (ev.date >= $date_cutoff OR ev.status = 'ongoing')
                WITH ev, collect(DISTINCT seed.name) AS seed_names
                ORDER BY ev.date DESC
                LIMIT $max_events
                UNWIND seed_names AS start_name
                MATCH (start:Entity {name: start_name})
                OPTIONAL MATCH path = (start)-[:RELATES_TO*1..3]->(affected:Entity)
                WHERE affected <> start
                  AND NOT affected.name IN seed_names
                WITH ev, start,
                     collect(DISTINCT {
                         name: affected.name,
                         type: affected.type,
                         hop_distance: length(path),
                         via_relation: [r IN relationships(path) | r.type][-1]
                     }) AS downstream
                RETURN ev.name AS event_name,
                       ev.date AS event_date,
                       ev.status AS event_status,
                       collect(DISTINCT start.name) AS seed_entities,
                       reduce(acc = [], d IN collect(downstream) |
                           acc + d) AS affected_entities
            """, date_cutoff=date_cutoff, max_events=max_events)

            for record in result:
                affected = record["affected_entities"] or []
                # Deduplicate by entity name, keep lowest hop_distance
                seen = {}
                for a in affected:
                    if a and a.get("name"):
                        name = a["name"]
                        if name not in seen or a.get("hop_distance", 99) < seen[name].get("hop_distance", 99):
                            seen[name] = a
                deduped = sorted(seen.values(), key=lambda x: x.get("hop_distance", 99))

                if deduped:
                    impacts.append({
                        "event_name": record["event_name"],
                        "event_date": record["event_date"] or "unknown",
                        "event_status": record["event_status"] or "unknown",
                        "seed_entities": record["seed_entities"],
                        "affected": deduped[:15],
                    })

        logger.info(f"  Impact propagation: {len(impacts)} event cascades found")
        return impacts

    # ── Step 3: Weak Link Detection ────────────────────────────────────────

    def _detect_weak_links(self, entity_names: list[str],
                            max_links: int = 10) -> list[dict]:
        """Find bridge entities connecting different domain clusters."""
        if not entity_names:
            return []

        weak_links = []
        with self.driver.session() as session:
            result = session.run("""
                MATCH (bridge:Entity)-[r:RELATES_TO]-(neighbor:Entity)
                WHERE bridge.name IN $names
                WITH bridge,
                     collect(DISTINCT neighbor.type) AS neighbor_types,
                     count(DISTINCT r) AS degree,
                     collect(DISTINCT {
                         name: neighbor.name,
                         type: neighbor.type,
                         relation: r.type
                     })[0..5] AS sample_connections
                WHERE size(neighbor_types) >= 2
                RETURN bridge.name AS entity,
                       bridge.type AS type,
                       neighbor_types,
                       degree,
                       sample_connections
                ORDER BY size(neighbor_types) DESC, degree ASC
                LIMIT $max_links
            """, names=entity_names, max_links=max_links)

            for record in result:
                # Map entity types to domain names
                domains = set()
                for nt in record["neighbor_types"]:
                    if self.entity_classifier:
                        # Use database-backed classifier
                        try:
                            domain = self.entity_classifier.get_primary_domain(nt)
                            domains.add(domain)
                        except Exception as e:
                            logger.warning(f"Failed to classify entity type '{nt}': {e}")
                            domains.add(DOMAIN_ENTITY_TYPES.get(nt, "other"))
                    else:
                        # Fallback to hardcoded mapping
                        domains.add(DOMAIN_ENTITY_TYPES.get(nt, "other"))

                weak_links.append({
                    "entity": record["entity"],
                    "type": record["type"],
                    "domains_bridged": sorted(domains),
                    "connection_count": record["degree"],
                    "neighbor_types": record["neighbor_types"],
                    "sample_connections": record["sample_connections"],
                })

        logger.info(f"  Weak links: {len(weak_links)} bridge entities found")
        return weak_links

    # ── Step 4: LLM Narrative Synthesis ────────────────────────────────────

    def _synthesize_narrative(self, chains: list[dict],
                               impacts: list[dict],
                               weak_links: list[dict]) -> dict:
        """Single LLM call to produce analyst-readable narratives."""

        # Format chains compactly
        chain_text = ""
        for i, c in enumerate(chains[:5], 1):
            path_str = ""
            for j, ent in enumerate(c["entities"]):
                path_str += ent["name"]
                if j < len(c["relations"]):
                    path_str += f" →[{c['relations'][j]['type']}]→ "
            chain_text += f"\n{i}. {path_str}"
            chain_text += f"\n   Score: {c.get('conf_score', 0):.2f}, "
            chain_text += f"Evidence: {c.get('evidence_total', 0)}"
            if c.get("source_articles"):
                chain_text += f"\n   Sources: {', '.join(c['source_articles'][:3])}"

        # Format impacts compactly
        impact_text = ""
        for imp in impacts[:4]:
            impact_text += f"\n- {imp['event_name']} ({imp['event_date']}, {imp['event_status']})"
            for a in imp["affected"][:5]:
                impact_text += f"\n  → {a['name']} ({a['type']}) at hop {a.get('hop_distance', '?')} via {a.get('via_relation', '?')}"

        # Format weak links compactly
        wl_text = ""
        for wl in weak_links[:5]:
            wl_text += f"\n- {wl['entity']} ({wl['type']}): bridges {', '.join(wl['domains_bridged'])}, {wl['connection_count']} connections"

        system_prompt = f"""You are a strategic intelligence analyst. You have been given structured graph analysis results showing:
1. Multi-hop causal chains connecting entities across domains
2. Impact cascades from recent events to downstream entities
3. Critical bridge entities (weak links) between domain clusters

For each result, provide a clear narrative explaining:
- WHY this chain/impact/vulnerability matters for decision-makers
- What ACTIONS or MONITORING should be considered
- What RISKS are implied

Be specific — reference entity names, relation types, and events.
Do NOT fabricate facts not present in the data.

{self.parser.get_format_instructions()}
"""

        user_prompt = f"""Analyze these graph inference results and provide strategic narratives.

## Multi-Hop Causal Chains
{chain_text or "No causal chains discovered"}

## Event Impact Cascades
{impact_text or "No recent event impacts detected"}

## Critical Bridge Entities (Weak Links)
{wl_text or "No weak links detected"}
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = _llm_invoke_with_retry(self.llm, messages)
        try:
            analysis = self.parser.parse(response.content)
        except Exception as e:
            logger.error(f"Failed to parse inference narrative: {e}")
            analysis = {
                "executive_summary": response.content if response else "Inference analysis failed.",
                "causal_chains": [],
                "impact_propagations": [],
                "weak_links": [],
            }

        return analysis

    # ── Main entry point ───────────────────────────────────────────────────

    def analyze(self, report_result) -> dict:
        """Run full inference analysis: chains, impacts, weak links, narrative.

        Args:
            report_result: ReportResult from ReportAgent.generate_with_context()

        Returns:
            InferenceAnalysis dict.
        """
        logger.info("InferenceAgent: starting graph inference analysis")

        # Gather entity names from the domain report
        entity_names = [
            e["name"] for e in report_result.graph_data.get("entities", [])
        ]

        if not entity_names:
            logger.warning("No entities in report data, skipping inference")
            return {
                "executive_summary": "No entity data available for inference analysis.",
                "causal_chains": [],
                "impact_propagations": [],
                "weak_links": [],
            }

        # Compute date cutoff for recent events (last 30 days)
        ist = timezone(timedelta(hours=5, minutes=30))
        date_cutoff = (datetime.now(ist).date() - timedelta(days=30)).isoformat()

        # Step 1: Causal chain discovery
        logger.info("  Step 1: Discovering causal chains...")
        chains = self._discover_causal_chains(entity_names, top_k=5)

        # Step 2: Impact propagation from recent events
        logger.info("  Step 2: Propagating event impacts...")
        impacts = self._propagate_impact(date_cutoff, max_events=5)

        # Step 3: Weak link detection
        logger.info("  Step 3: Detecting weak links...")
        weak_links = self._detect_weak_links(entity_names, max_links=5)

        # Step 4: LLM narrative synthesis
        logger.info("  Step 4: Synthesizing narratives via LLM...")
        analysis = self._synthesize_narrative(chains, impacts, weak_links)

        logger.info("InferenceAgent: analysis complete")
        return analysis
