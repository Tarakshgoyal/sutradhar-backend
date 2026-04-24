"""India Impact Analysis Agent.

Analyzes domain briefings from India's strategic perspective.
Uses graph-driven entity discovery (not keyword grep) to find
India-connected entities, then synthesizes strategic insights via LLM.
"""

from __future__ import annotations

import json
import logging
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from config import NEO4J_URI, NEO4J_AUTH
from models.database import SessionLocal
from agents.india_entity_service import IndiaEntityService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------

class StrategicAssessment(BaseModel):
    summary: str = Field(description="2-3 sentence strategic assessment")
    implications: list[str] = Field(description="3-5 key implications for India")


class TransparencyInsight(BaseModel):
    area: str = Field(description="Area of governance/transparency, e.g. 'Defence procurement'")
    assessment: str = Field(description="1-2 sentence assessment of transparency/accountability")
    recommendation: str = Field(description="Actionable recommendation")


class NationalAdvantage(BaseModel):
    domain: str = Field(description="Domain of advantage, e.g. 'Technology', 'Trade'")
    opportunity: str = Field(description="The opportunity India can leverage")
    current_status: str = Field(description="Where India currently stands")
    action_needed: str = Field(description="What India should do to capitalize")


class RiskAssessment(BaseModel):
    threat: str = Field(description="Nature of the threat or risk")
    severity: str = Field(description="One of: high, medium, low")
    affected_sectors: list[str] = Field(description="Sectors/domains affected")
    mitigation: str = Field(description="Suggested mitigation strategy")


class GlobalPositioning(BaseModel):
    dimension: str = Field(description="Dimension of positioning, e.g. 'Military power', 'Trade influence'")
    india_position: str = Field(description="India's current position/standing")
    key_competitors: list[str] = Field(description="Countries competing with India in this dimension")
    trajectory: str = Field(description="One of: improving, stable, declining, uncertain")


class IndiaImpactAnalysis(BaseModel):
    executive_summary: str = Field(description="2-3 paragraph overview of how this domain affects India")
    strategic_assessment: StrategicAssessment = Field(description="High-level strategic assessment")
    transparency_insights: list[TransparencyInsight] = Field(description="2-4 transparency and governance insights")
    national_advantages: list[NationalAdvantage] = Field(description="3-5 national advantage opportunities")
    risks: list[RiskAssessment] = Field(description="3-5 risk assessments")
    global_positioning: list[GlobalPositioning] = Field(description="2-4 global positioning dimensions")
    recommendations: list[str] = Field(description="5-8 actionable policy/strategy recommendations")


# ---------------------------------------------------------------------------
# LLM Retry helper (same as report.py)
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
# India Impact Agent
# ---------------------------------------------------------------------------

class IndiaImpactAgent:
    """Analyzes domain briefings from India's strategic perspective."""

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.3, num_predict=8192)
        self.parser = JsonOutputParser(pydantic_object=IndiaImpactAnalysis)
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        logger.info("IndiaImpactAgent initialized")

    def close(self):
        self.driver.close()

    # ── Phase A: Graph-driven India entity discovery ───────────────────────

    def _discover_india_entities(self) -> set[str]:
        """Find all India-connected entities from database.

        Retrieves pre-discovered India entities from the PostgreSQL database.
        The database is periodically refreshed via graph traversal by a background worker.
        Falls back to direct graph query if database is empty.
        """
        # Get database session
        db = SessionLocal()
        try:
            service = IndiaEntityService(db, neo4j_driver=self.driver)

            # Try to get entities from database first
            entities = service.get_india_entities(min_relevance_score=0.3)

            if entities:
                logger.info(f"  India entity discovery: {len(entities)} entities from database")
                return entities

            # Fallback: discover from graph directly and refresh database
            logger.warning("No India entities in database, discovering from graph")
            discovered = service.discover_from_graph(max_hops=3, min_connection_count=2)

            if discovered:
                # Store discovered entities
                result = service.refresh_database(max_hops=3, min_connection_count=2)
                logger.info(f"  Discovered and stored {result.get('total', 0)} India entities")

                # Return entity names
                entities = {e["entity_name"] for e in discovered}
                return entities

            # Ultimate fallback: return empty set (no silent defaults)
            logger.error("Failed to discover any India entities from graph")
            return set()

        finally:
            db.close()

    # ── Phase A.2: Extract India-relevant subgraph ─────────────────────────

    def _extract_india_subgraph(self, india_entities: set[str],
                                 graph_data: dict) -> dict:
        """Filter domain graph_data for India-connected relations and entities."""
        india_lower = {name.lower() for name in india_entities}

        # Filter relations where source or target is India-connected
        india_relations = [
            r for r in graph_data.get("relations", [])
            if r["source"].lower() in india_lower or r["target"].lower() in india_lower
        ]

        # Collect entities involved in India relations
        involved_names = set()
        for r in india_relations:
            involved_names.add(r["source"])
            involved_names.add(r["target"])

        india_entities_data = [
            e for e in graph_data.get("entities", [])
            if e["name"].lower() in india_lower or e["name"] in involved_names
        ]

        # Filter events involving India-connected entities
        india_events = [
            ev for ev in graph_data.get("events", [])
            if any(ent.lower() in india_lower for ent in ev.get("entities", []))
        ]

        return {
            "entities": india_entities_data[:10],
            "relations": india_relations[:20],
            "events": india_events[:10],
        }

    # ── Phase A.3: Filter India-relevant articles ──────────────────────────

    def _filter_india_articles(self, india_entities: set[str],
                                articles: list[dict],
                                max_articles: int = 5,
                                max_chars: int = 1500) -> list[dict]:
        """Filter articles by graph-discovered India entities.

        Checks article titles and excerpts for mentions of India-connected
        entity names. This catches metonyms because entity resolution
        already linked 'New Delhi', 'Modi', etc. to India.
        """
        india_lower = {name.lower() for name in india_entities}
        scored = []

        for article in articles:
            text = (article.get("title", "") + " " + article.get("excerpt", "")).lower()
            # Count how many India entities appear in this article
            hits = sum(1 for name in india_lower if name in text)
            if hits > 0:
                scored.append((hits, article))

        # Sort by relevance (most India entity mentions first)
        scored.sort(key=lambda x: x[0], reverse=True)

        result = []
        for _, article in scored[:max_articles]:
            truncated = {**article}
            if "excerpt" in truncated:
                truncated["excerpt"] = truncated["excerpt"][:max_chars]
            result.append(truncated)

        return result

    # ── Phase B: Compact prompt construction ───────────────────────────────

    def _build_compact_prompt(self, domain: str, briefing: dict,
                               india_subgraph: dict,
                               india_articles: list[dict]) -> tuple[str, str]:
        """Build a token-budgeted system + user prompt for India analysis."""

        system_prompt = f"""You are a senior strategic intelligence analyst specializing in India's national interests. You are analyzing a {domain.upper()} domain intelligence briefing.

Your task: Assess the strategic impact on India — identify opportunities, risks, transparency implications, and India's global positioning within this domain.

ANALYSIS FRAMEWORK:
1. STRATEGIC ASSESSMENT — What does this mean for India's national interests?
2. TRANSPARENCY INSIGHTS — Governance, accountability, and open-data implications
3. NATIONAL ADVANTAGES — Where can India gain strategic, economic, or technological advantage?
4. RISK ASSESSMENT — What threatens Indian interests? Severity and mitigation.
5. GLOBAL POSITIONING — India's position vs key competitors, trajectory
6. RECOMMENDATIONS — Actionable policy and strategy recommendations

RULES:
- Ground all analysis in the provided data — do NOT fabricate facts
- Be specific: name entities, cite relationships, reference events
- Focus on actionable insights, not generic observations
- Consider India's relationships with all mentioned countries/entities
- Assess both direct and indirect (second-order) effects on India
- Keep each recommendation as a SEPARATE string in the recommendations list
- Keep each field value concise to avoid hitting token limits

{self.parser.get_format_instructions()}
"""

        # Build compact user prompt pieces
        # 1. Executive summary from domain briefing (~400 tokens)
        exec_summary = briefing.get("executive_summary", "No summary available.")

        # 2. Key developments as bullet titles only (~200 tokens)
        dev_bullets = "\n".join(
            f"- {d['title']} ({d.get('date', '?')})"
            for d in briefing.get("key_developments", [])[:8]
        )

        # 3. India subgraph relations (~500 tokens)
        relation_lines = "\n".join(
            f"- {r['source']} →[{r['type']}]→ {r['target']}"
            + (f" (causal)" if r.get("causal") else "")
            for r in india_subgraph.get("relations", [])[:20]
        )

        entity_lines = "\n".join(
            f"- {e['name']} ({e['type']})"
            for e in india_subgraph.get("entities", [])[:10]
        )

        event_lines = "\n".join(
            f"- {ev['name']} ({ev.get('date', '?')}, {ev.get('status', '?')})"
            for ev in india_subgraph.get("events", [])[:10]
        )

        # 4. India-relevant article excerpts (~2000 tokens)
        article_blocks = "\n\n".join(
            f"### {a['title']} ({a.get('source', '?')}, {a.get('pub_date', '?')})\n{a.get('excerpt', '')}"
            for a in india_articles[:5]
        )

        user_prompt = f"""Analyze the following {domain.upper()} intelligence briefing from India's strategic perspective.

## Domain Overview
{exec_summary}

## Key Developments
{dev_bullets or "No developments available"}

## India-Connected Entities
{entity_lines or "No India-connected entities found in this domain"}

## India-Connected Relationships
{relation_lines or "No India-connected relationships found"}

## India-Connected Events
{event_lines or "No India-connected events found"}

## India-Relevant Source Articles
{article_blocks or "No India-relevant articles available"}
"""

        return system_prompt, user_prompt

    # ── Main entry point ───────────────────────────────────────────────────

    def analyze(self, domain: str, report_result) -> dict:
        """Analyze a domain briefing from India's strategic perspective.

        Args:
            domain: Domain name (climate, defence, economics, geopolitics, society)
            report_result: ReportResult from ReportAgent.generate_with_context()

        Returns:
            IndiaImpactAnalysis dict with strategic insights.
        """
        logger.info(f"IndiaImpactAgent: analyzing {domain} from India's perspective")

        # Phase A: Discover India-connected entities from database
        india_entities = self._discover_india_entities()

        if not india_entities:
            logger.error("No India-connected entities found in database or graph")
            # Return empty analysis instead of using hardcoded fallback
            return {
                "executive_summary": "No India-related entities could be discovered for this analysis.",
                "strategic_assessment": {
                    "summary": "Entity discovery failed - cannot perform India impact analysis.",
                    "implications": ["Database refresh needed"]
                },
                "transparency_insights": [],
                "national_advantages": [],
                "risks": [],
                "global_positioning": [],
            }

        # Phase A.2: Extract India-relevant subgraph from domain data
        india_subgraph = self._extract_india_subgraph(
            india_entities, report_result.graph_data
        )
        logger.info(
            f"  India subgraph: {len(india_subgraph['entities'])} entities, "
            f"{len(india_subgraph['relations'])} relations, "
            f"{len(india_subgraph['events'])} events"
        )

        # Phase A.3: Filter articles for India relevance
        india_articles = self._filter_india_articles(
            india_entities, report_result.articles
        )
        logger.info(f"  India-relevant articles: {len(india_articles)}")

        # Phase B: Build compact prompt and call LLM
        system_prompt, user_prompt = self._build_compact_prompt(
            domain, report_result.briefing, india_subgraph, india_articles
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = _llm_invoke_with_retry(self.llm, messages)
        
        try:
            analysis = self.parser.parse(response.content)
        except Exception as e:
            logger.warning(f"Structured parser failed for {domain}, falling back to manual extract: {e}")
            try:
                analysis = self._extract_json(response.content)
            except Exception as e2:
                logger.error(f"Failed to parse India impact analysis for {domain}: {e2}")
                analysis = {
                    "executive_summary": "India impact analysis could not be generated.",
                    "strategic_assessment": {"summary": "", "implications": []},
                    "transparency_insights": [],
                    "national_advantages": [],
                    "risks": [],
                    "global_positioning": [],
                    "recommendations": [],
                }

        logger.info(f"  India impact analysis complete for {domain}")
        return analysis

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Best-effort JSON extraction from LLM response text."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting JSON block from markdown fences
        match = re.search(r'```(?:json)?\s*\n?(\{.*?})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try finding the outermost { ... }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError("Could not extract valid JSON from LLM response")
