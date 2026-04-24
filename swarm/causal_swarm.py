"""Swarm B — Causal-Chain Hypothesis Swarm.

Replaces the linear InferenceAgent._discover_causal_chains() with a
competitive multi-worker swarm:

  Hypothesis Workers (N=7): Each traverses Neo4j from the same seed
      entities using a different strategy, generating one causal-chain
      hypothesis (3–5 hops). Workers are "blind" to each other; they only
      read the shared graph.

  Critic Workers (N=3): Score every hypothesis on three dimensions:
      - evidence_support    (0–10): chain backed by article evidence
      - historical_precedent (0–10): mirrors documented historical patterns
      - logical_soundness   (0–10): causal logic is coherent

  Consensus: top-scoring chain enters the intelligence report.
      All other chains are stored in Redis as "alternative futures".

This mirrors ant-colony optimization: many paths explored in parallel,
the best path reinforced via pheromones, weaker paths evaporate.

Drop-in replacement for InferenceAgent._discover_causal_chains() and
InferenceAgent.analyze() — call CausalChainSwarm.analyze(report_result).
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from agents.inference import InferenceAgent, _llm_invoke_with_retry
from config import NEO4J_URI, NEO4J_AUTH
from swarm.blackboard import SwarmBlackboard

logger = logging.getLogger(__name__)

# ── Swarm config ──────────────────────────────────────────────────────────────
N_HYPOTHESIS_WORKERS = 7
N_CRITIC_WORKERS = 3
WORKER_TIMEOUT = 240     # seconds per worker
CRITIC_TIMEOUT = 120     # seconds per critic


# ── Seven traversal strategies ────────────────────────────────────────────────
# Each defines how a hypothesis worker traverses Neo4j.

TRAVERSAL_STRATEGIES: list[dict] = [
    {
        "name": "deep_causal",
        "description": "Follow causal-flagged edges 4–5 hops deep",
        "min_hops": 4, "max_hops": 5, "require_causal": True, "order": "confidence",
    },
    {
        "name": "cross_domain",
        "description": "Find shortest path bridging two different entity types",
        "min_hops": 2, "max_hops": 4, "require_causal": False, "order": "type_diversity",
    },
    {
        "name": "high_evidence",
        "description": "Prefer edges with highest evidence counts",
        "min_hops": 2, "max_hops": 4, "require_causal": False, "order": "evidence",
    },
    {
        "name": "weak_link_path",
        "description": "Route through low-degree bridge entities",
        "min_hops": 3, "max_hops": 5, "require_causal": False, "order": "low_degree",
    },
    {
        "name": "mixed_causal",
        "description": "At least one causal edge, rest can be any relation",
        "min_hops": 2, "max_hops": 4, "require_causal": False, "order": "mixed",
    },
    {
        "name": "reverse_chain",
        "description": "Traverse in reverse direction (what caused this?)",
        "min_hops": 2, "max_hops": 4, "require_causal": False, "order": "reverse",
    },
    {
        "name": "recent_only",
        "description": "Prefer edges and entities updated most recently",
        "min_hops": 2, "max_hops": 3, "require_causal": False, "order": "recent",
    },
]

STRATEGY_SYSTEM_PROMPTS: dict[str, str] = {
    "deep_causal": "You are tracing deep causal chains — explain how one event directly CAUSES a cascade across 4–5 steps.",
    "cross_domain": "You are identifying cross-domain transmission — how does an event in one domain (e.g. economics) cause effects in another (e.g. security)?",
    "high_evidence": "You are following the most evidence-supported path — focus on well-documented causal connections.",
    "weak_link_path": "You are identifying fragile dependency chains — paths that run through single points of failure.",
    "mixed_causal": "You are tracing a mix of direct and indirect causation — explain both confirmed and inferred links.",
    "reverse_chain": "You are working backwards — explain what historical factors and prior causes led to the current situation.",
    "recent_only": "You are analyzing the most recent developments — focus on what has changed in the past week and immediate downstream effects.",
}


class _HypothesisWorker:
    """Generates one causal chain hypothesis using a specific traversal strategy."""

    def __init__(self, strategy: dict, driver):
        self.strategy = strategy
        self.driver = driver
        self.llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.4, num_predict=1024)

    def run(self, entity_names: list[str], date_cutoff: str) -> dict | None:
        """Traverse Neo4j and generate a chain hypothesis."""
        strategy = self.strategy
        chain_data = self._traverse(entity_names, date_cutoff)
        if not chain_data:
            return None

        # LLM: generate narrative for this specific chain
        path_text = self._format_path(chain_data)
        system = STRATEGY_SYSTEM_PROMPTS.get(strategy["name"], "You are a strategic intelligence analyst.")

        try:
            response = self.llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=(
                    f"Analyze this causal chain and provide:\n"
                    f"1. A 2-sentence narrative explaining why this chain matters\n"
                    f"2. Key risk or opportunity implied\n\n"
                    f"Chain:\n{path_text}\n\n"
                    f"Respond in plain text, under 150 words."
                )),
            ])
            narrative = response.content.strip()
        except Exception as e:
            logger.debug(f"[CausalSwarm/{strategy['name']}] LLM narrative failed: {e}")
            narrative = f"Chain discovered via {strategy['name']} traversal."

        return {
            "strategy": strategy["name"],
            "entities": chain_data.get("entities", []),
            "relations": chain_data.get("relations", []),
            "conf_score": chain_data.get("conf_score", 0.5),
            "evidence_total": chain_data.get("evidence_total", 1),
            "type_diversity": chain_data.get("type_diversity", 1),
            "source_articles": chain_data.get("source_articles", []),
            "narrative": narrative,
            "critic_scores": {},
            "total_score": 0.0,
        }

    def _traverse(self, entity_names: list[str], date_cutoff: str) -> dict | None:
        s = self.strategy
        # Shuffle seed entities for diversity across workers
        names = entity_names[:]
        random.shuffle(names)
        names = names[:20]  # cap to avoid slow queries

        with self.driver.session() as session:
            try:
                if s["order"] == "reverse":
                    result = session.run("""
                        MATCH path = (end_node:Entity)-[:RELATES_TO*2..4]->(start:Entity)
                        WHERE start.name IN $names AND start <> end_node
                          AND start.type <> end_node.type
                        WITH path, start, end_node,
                             reduce(s=1.0, r IN relationships(path) |
                                 s * coalesce(r.confidence, 0.5)) AS conf_score,
                             reduce(s=0, r IN relationships(path) |
                                 s + coalesce(r.evidence_count,1)) AS evidence_total,
                             [n IN nodes(path) | n.type] AS type_list
                        UNWIND type_list AS t
                        WITH path, start, end_node, conf_score, evidence_total,
                             count(DISTINCT t) AS type_diversity
                        ORDER BY conf_score DESC LIMIT 3
                        RETURN [n IN nodes(path) | {name: n.name, type: n.type}] AS entities,
                               [r IN relationships(path) | {type: r.type, causal: r.causal,
                                confidence: r.confidence, evidence_count: r.evidence_count}] AS relations,
                               conf_score, evidence_total, type_diversity
                        LIMIT 1
                    """, names=names)
                elif s["require_causal"]:
                    result = session.run(f"""
                        MATCH path = (start:Entity)-[:RELATES_TO*{s['min_hops']}..{s['max_hops']}]->(end_node:Entity)
                        WHERE start.name IN $names AND start <> end_node
                          AND ALL(r IN relationships(path) WHERE r.causal = true)
                        WITH path, start, end_node,
                             reduce(s=1.0, r IN relationships(path) |
                                 s * coalesce(r.confidence,0.5)) AS conf_score,
                             reduce(s=0, r IN relationships(path) |
                                 s + coalesce(r.evidence_count,1)) AS evidence_total,
                             [n IN nodes(path) | n.type] AS type_list
                        UNWIND type_list AS t
                        WITH path, start, end_node, conf_score, evidence_total,
                             count(DISTINCT t) AS type_diversity
                        ORDER BY conf_score * (1 + log(evidence_total+1)) * type_diversity DESC
                        LIMIT 1
                        RETURN [n IN nodes(path) | {{name: n.name, type: n.type}}] AS entities,
                               [r IN relationships(path) | {{type: r.type, causal: r.causal,
                                confidence: r.confidence, evidence_count: r.evidence_count}}] AS relations,
                               conf_score, evidence_total, type_diversity
                    """, names=names)
                elif s["order"] == "type_diversity":
                    result = session.run(f"""
                        MATCH path = (start:Entity)-[:RELATES_TO*{s['min_hops']}..{s['max_hops']}]->(end_node:Entity)
                        WHERE start.name IN $names AND start <> end_node
                          AND start.type <> end_node.type
                        WITH path, start, end_node,
                             reduce(s=1.0, r IN relationships(path) |
                                 s * coalesce(r.confidence,0.5)) AS conf_score,
                             reduce(s=0, r IN relationships(path) |
                                 s + coalesce(r.evidence_count,1)) AS evidence_total,
                             [n IN nodes(path) | n.type] AS type_list
                        UNWIND type_list AS t
                        WITH path, start, end_node, conf_score, evidence_total,
                             count(DISTINCT t) AS type_diversity
                        ORDER BY type_diversity DESC, conf_score DESC
                        LIMIT 1
                        RETURN [n IN nodes(path) | {{name: n.name, type: n.type}}] AS entities,
                               [r IN relationships(path) | {{type: r.type, causal: r.causal,
                                confidence: r.confidence, evidence_count: r.evidence_count}}] AS relations,
                               conf_score, evidence_total, type_diversity
                    """, names=names)
                elif s["order"] == "evidence":
                    result = session.run(f"""
                        MATCH path = (start:Entity)-[:RELATES_TO*{s['min_hops']}..{s['max_hops']}]->(end_node:Entity)
                        WHERE start.name IN $names AND start <> end_node
                        WITH path, start, end_node,
                             reduce(s=1.0, r IN relationships(path) |
                                 s * coalesce(r.confidence,0.5)) AS conf_score,
                             reduce(s=0, r IN relationships(path) |
                                 s + coalesce(r.evidence_count,1)) AS evidence_total,
                             [n IN nodes(path) | n.type] AS type_list
                        UNWIND type_list AS t
                        WITH path, start, end_node, conf_score, evidence_total,
                             count(DISTINCT t) AS type_diversity
                        ORDER BY evidence_total DESC, conf_score DESC
                        LIMIT 1
                        RETURN [n IN nodes(path) | {{name: n.name, type: n.type}}] AS entities,
                               [r IN relationships(path) | {{type: r.type, causal: r.causal,
                                confidence: r.confidence, evidence_count: r.evidence_count}}] AS relations,
                               conf_score, evidence_total, type_diversity
                    """, names=names)
                else:
                    # mixed / low_degree / recent fallback — use a general query with randomized start
                    result = session.run(f"""
                        MATCH path = (start:Entity)-[:RELATES_TO*{s['min_hops']}..{s['max_hops']}]->(end_node:Entity)
                        WHERE start.name IN $names AND start <> end_node
                          AND ANY(r IN relationships(path) WHERE r.causal = true)
                        WITH path, start, end_node,
                             reduce(sc=1.0, r IN relationships(path) |
                                 sc * coalesce(r.confidence,0.5)) AS conf_score,
                             reduce(ev=0, r IN relationships(path) |
                                 ev + coalesce(r.evidence_count,1)) AS evidence_total,
                             [n IN nodes(path) | n.type] AS type_list
                        UNWIND type_list AS t
                        WITH path, start, end_node, conf_score, evidence_total,
                             count(DISTINCT t) AS type_diversity
                        ORDER BY rand()
                        LIMIT 1
                        RETURN [n IN nodes(path) | {{name: n.name, type: n.type}}] AS entities,
                               [r IN relationships(path) | {{type: r.type, causal: r.causal,
                                confidence: r.confidence, evidence_count: r.evidence_count}}] AS relations,
                               conf_score, evidence_total, type_diversity
                    """, names=names)

                record = result.single()
                if not record:
                    return None

                chain = {
                    "entities": record["entities"],
                    "relations": record["relations"],
                    "conf_score": record["conf_score"],
                    "evidence_total": record["evidence_total"],
                    "type_diversity": record["type_diversity"],
                    "source_articles": [],
                }

                # Fetch source articles
                entity_names_in_chain = [e["name"] for e in chain["entities"]]
                art_result = session.run("""
                    MATCH (a:Article)-[:EVIDENCES]->(e:Entity)
                    WHERE e.name IN $names
                    RETURN e.name AS entity, collect(DISTINCT a.title)[0..3] AS articles
                """, names=entity_names_in_chain)
                for ar in art_result:
                    chain["source_articles"].extend(ar["articles"])
                chain["source_articles"] = list(dict.fromkeys(chain["source_articles"]))[:5]

                return chain

            except Exception as e:
                logger.debug(f"[CausalSwarm/{s['name']}] traversal failed: {e}")
                return None

    def _format_path(self, chain: dict) -> str:
        entities = chain.get("entities", [])
        relations = chain.get("relations", [])
        parts = []
        for i, ent in enumerate(entities):
            parts.append(f"{ent['name']} ({ent['type']})")
            if i < len(relations):
                parts.append(f"→[{relations[i].get('type','?')}]→")
        return " ".join(parts)


class _CriticWorker:
    """Scores a set of chain hypotheses on evidence, precedent, and logic."""

    def __init__(self, critic_id: int):
        self.critic_id = critic_id
        self.llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.2, num_predict=512)

    def score(self, chains: list[dict]) -> dict[int, dict]:
        """Returns {chain_index: {evidence_support, historical_precedent, logical_soundness}}."""
        scores: dict[int, dict] = {}
        for idx, chain in enumerate(chains):
            try:
                path_text = self._format_path(chain)
                response = self.llm.invoke([
                    SystemMessage(content=(
                        "You are a critical intelligence analyst scoring causal chain hypotheses. "
                        "Score each dimension 0-10 (integers only). Be strict.\n"
                        "Return ONLY valid JSON: "
                        "{\"evidence_support\": N, \"historical_precedent\": N, \"logical_soundness\": N}"
                    )),
                    HumanMessage(content=(
                        f"Score this causal chain:\n{path_text}\n\n"
                        f"Narrative: {chain.get('narrative', 'N/A')}\n"
                        f"Evidence count: {chain.get('evidence_total', 0)}\n"
                        f"Confidence: {chain.get('conf_score', 0):.2f}"
                    )),
                ])
                import json
                raw = response.content.strip()
                # Extract JSON from response
                json_start = raw.find("{")
                json_end = raw.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    score_data = json.loads(raw[json_start:json_end])
                    scores[idx] = {
                        "evidence_support": int(score_data.get("evidence_support", 5)),
                        "historical_precedent": int(score_data.get("historical_precedent", 5)),
                        "logical_soundness": int(score_data.get("logical_soundness", 5)),
                    }
                else:
                    scores[idx] = {"evidence_support": 5, "historical_precedent": 5, "logical_soundness": 5}
            except Exception as e:
                logger.debug(f"[CausalSwarm/critic{self.critic_id}] scoring failed: {e}")
                scores[idx] = {"evidence_support": 5, "historical_precedent": 5, "logical_soundness": 5}
        return scores

    def _format_path(self, chain: dict) -> str:
        entities = chain.get("entities", [])
        relations = chain.get("relations", [])
        parts = []
        for i, ent in enumerate(entities):
            parts.append(f"{ent['name']} ({ent['type']})")
            if i < len(relations):
                parts.append(f"→[{relations[i].get('type','?')}]→")
        return " ".join(parts)


class CausalChainSwarm:
    """Swarm B — competitive causal-chain hypothesis generation.

    Drop-in replacement for InferenceAgent — call analyze(report_result).
    """

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self._base_inference = InferenceAgent(model=model)
        self._driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        self._hypothesis_workers = [
            _HypothesisWorker(TRAVERSAL_STRATEGIES[i], self._driver)
            for i in range(N_HYPOTHESIS_WORKERS)
        ]
        self._critic_workers = [_CriticWorker(i) for i in range(N_CRITIC_WORKERS)]
        self._executor = ThreadPoolExecutor(max_workers=N_HYPOTHESIS_WORKERS + N_CRITIC_WORKERS)
        self._bb: SwarmBlackboard | None = None
        try:
            self._bb = SwarmBlackboard()
        except Exception as e:
            logger.warning(f"[CausalSwarm] Redis unavailable: {e}")
        logger.info("[CausalChainSwarm] initialized with 7 hypothesis workers + 3 critics")

    def _discover_causal_chains_swarm(
        self, entity_names: list[str], domain: str = "unknown"
    ) -> list[dict]:
        """Run the full swarm: hypotheses → critics → ranked chains."""
        if not entity_names:
            return []

        ist = timezone(timedelta(hours=5, minutes=30))
        date_cutoff = (datetime.now(ist).date() - timedelta(days=30)).isoformat()

        # Phase 1: hypothesis workers in parallel
        logger.info(f"[CausalSwarm] Phase 1: {N_HYPOTHESIS_WORKERS} hypothesis workers generating chains...")
        hypotheses: list[dict] = []
        futures = {
            self._executor.submit(w.run, entity_names, date_cutoff): w.strategy["name"]
            for w in self._hypothesis_workers
        }
        for future in as_completed(futures, timeout=WORKER_TIMEOUT * 1.5):
            strategy_name = futures[future]
            try:
                result = future.result(timeout=WORKER_TIMEOUT)
                if result:
                    hypotheses.append(result)
                    logger.debug(f"[CausalSwarm] {strategy_name}: chain found ({len(result['entities'])} nodes)")
                else:
                    logger.debug(f"[CausalSwarm] {strategy_name}: no chain found")
            except (FuturesTimeout, Exception) as e:
                logger.warning(f"[CausalSwarm] {strategy_name} failed: {e}")

        if not hypotheses:
            logger.warning("[CausalSwarm] No hypotheses generated — falling back to single agent")
            return self._base_inference._discover_causal_chains(entity_names, top_k=5)

        # Deduplicate identical chains
        seen_starts = set()
        unique_hypotheses = []
        for h in hypotheses:
            if h["entities"]:
                key = (h["entities"][0].get("name", ""), h["entities"][-1].get("name", ""))
                if key not in seen_starts:
                    seen_starts.add(key)
                    unique_hypotheses.append(h)
        hypotheses = unique_hypotheses

        logger.info(f"[CausalSwarm] Phase 1 complete: {len(hypotheses)} unique hypotheses")

        # Phase 2: critic workers score all hypotheses
        logger.info(f"[CausalSwarm] Phase 2: {N_CRITIC_WORKERS} critics scoring {len(hypotheses)} chains...")
        critic_futures = {
            self._executor.submit(c.score, hypotheses): c.critic_id
            for c in self._critic_workers
        }
        all_critic_scores: list[dict[int, dict]] = []
        for future in as_completed(critic_futures, timeout=CRITIC_TIMEOUT * 2):
            critic_id = critic_futures[future]
            try:
                scores = future.result(timeout=CRITIC_TIMEOUT)
                all_critic_scores.append(scores)
                logger.debug(f"[CausalSwarm] critic {critic_id}: scored {len(scores)} chains")
            except Exception as e:
                logger.warning(f"[CausalSwarm] critic {critic_id} failed: {e}")

        # Aggregate critic scores
        for idx, hypothesis in enumerate(hypotheses):
            scores_for_chain = [
                critic_scores[idx]
                for critic_scores in all_critic_scores
                if idx in critic_scores
            ]
            if scores_for_chain:
                avg_evidence = sum(s["evidence_support"] for s in scores_for_chain) / len(scores_for_chain)
                avg_precedent = sum(s["historical_precedent"] for s in scores_for_chain) / len(scores_for_chain)
                avg_logic = sum(s["logical_soundness"] for s in scores_for_chain) / len(scores_for_chain)
                total = (avg_evidence + avg_precedent + avg_logic) / 30.0  # normalize to 0-1
                hypothesis["critic_scores"] = {
                    "evidence_support": round(avg_evidence, 1),
                    "historical_precedent": round(avg_precedent, 1),
                    "logical_soundness": round(avg_logic, 1),
                }
                hypothesis["total_score"] = round(total, 3)
            else:
                # No critic scores — use graph confidence as fallback
                hypothesis["total_score"] = hypothesis.get("conf_score", 0.5)

        # Rank by critic total score
        hypotheses.sort(key=lambda h: h["total_score"], reverse=True)

        logger.info(
            f"[CausalSwarm] Phase 2 complete. Best chain: "
            f"strategy={hypotheses[0]['strategy']}, score={hypotheses[0]['total_score']:.3f}"
        )

        # Store alternatives in blackboard
        if self._bb and len(hypotheses) > 1:
            self._bb.store_alternative_chains(domain, hypotheses[1:])
            logger.info(f"[CausalSwarm] {len(hypotheses)-1} alternative futures stored in Redis")
            # Reinforce pheromone for best chain entities
            for ent in hypotheses[0].get("entities", []):
                self._bb.reinforce_pheromone(ent.get("name", ""), delta=2.0)

        return hypotheses[:5]  # return top 5 for InferenceAgent compatibility

    def analyze(self, report_result) -> dict:
        """Full inference analysis using swarm for chain discovery."""
        logger.info("[CausalChainSwarm] starting swarm-enhanced inference analysis")

        entity_names = [
            e["name"] for e in report_result.graph_data.get("entities", [])
        ]
        domain = getattr(report_result, "domain", "unknown")

        if not entity_names:
            logger.warning("[CausalChainSwarm] No entities — falling back to base InferenceAgent")
            return self._base_inference.analyze(report_result)

        ist = timezone(timedelta(hours=5, minutes=30))
        date_cutoff = (datetime.now(ist).date() - timedelta(days=30)).isoformat()

        # Step 1: Swarm causal chain discovery
        logger.info("[CausalChainSwarm] Step 1: Swarm causal chain discovery...")
        chains = self._discover_causal_chains_swarm(entity_names, domain=domain)

        # Steps 2 & 3: Reuse base agent for impact propagation and weak links
        logger.info("[CausalChainSwarm] Step 2: Impact propagation (base agent)...")
        impacts = self._base_inference._propagate_impact(date_cutoff, max_events=5)

        logger.info("[CausalChainSwarm] Step 3: Weak link detection (base agent)...")
        weak_links = self._base_inference._detect_weak_links(entity_names, max_links=5)

        # Step 4: Narrative synthesis
        logger.info("[CausalChainSwarm] Step 4: Narrative synthesis...")
        analysis = self._base_inference._synthesize_narrative(chains, impacts, weak_links)

        # Attach swarm metadata to top chain
        if chains and isinstance(analysis, dict):
            analysis["swarm_metadata"] = {
                "n_hypotheses_generated": len(chains),
                "winning_strategy": chains[0].get("strategy", "unknown"),
                "winning_score": chains[0].get("total_score", 0),
                "alternatives_stored": self._bb is not None,
            }

        logger.info("[CausalChainSwarm] analysis complete")
        return analysis

    def close(self):
        self._executor.shutdown(wait=False)
        self._base_inference.close()
        self._driver.close()
        if self._bb:
            self._bb.close()
