"""Policy Brief Agent — auto-generates ministry-ready intelligence documents.

Produces four document types:
  1. intelligence_summary  — 1-page BLUF brief
  2. policy_brief          — 2-3 page brief with policy options
  3. options_memo          — Full strategic analysis with risk matrices
  4. sitrep                — Daily/weekly situation report from graph + signals

All documents include source attribution and entities referenced.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase
from sqlalchemy import select, desc

from config import NEO4J_URI, NEO4J_AUTH
from models.database import SessionLocal
from models.policy_brief import PolicyBrief
from models.scraped_article import ScrapedArticle
from models.detected_signal import DetectedSignal
from sqlalchemy.dialects.postgresql import insert as pg_insert

logger = logging.getLogger(__name__)
_IST = timezone(timedelta(hours=5, minutes=30))


class PolicyBriefAgent:
    """Generates structured intelligence documents for decision-makers."""

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.4, num_predict=4096)
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        logger.info("PolicyBriefAgent initialized")

    def close(self):
        self.driver.close()

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_intelligence_summary(
        self, domain: str, topic: Optional[str] = None
    ) -> dict:
        """Generate a 1-page intelligence summary using BLUF format."""
        context = self._gather_domain_context(domain, hours=48)
        return self._generate_document(
            brief_type="intelligence_summary",
            domain=domain,
            topic=topic,
            context=context,
            instructions="""Write a concise intelligence summary using BLUF format:

BOTTOM LINE UP FRONT (2 sentences): The single most important thing a minister needs to know.

SITUATION: Current state (3-4 bullet points, each citing a source).

KEY DEVELOPMENTS: What changed in the last 48 hours (chronological, 4-5 bullets).

ANALYSIS: What this means — cause, consequence, trajectory (2-3 sentences).

OUTLOOK: Most likely next development and timeframe (1-2 sentences).

SOURCE RELIABILITY: Note confidence level (High/Medium/Low) and any intelligence gaps.

Keep the entire brief under 400 words. Be specific — reference entity names and dates.""",
        )

    def generate_policy_brief(self, domain: str, topic: Optional[str] = None) -> dict:
        """Generate a 2-3 page policy brief with options."""
        context = self._gather_domain_context(domain, hours=168)  # 7 days
        return self._generate_document(
            brief_type="policy_brief",
            domain=domain,
            topic=topic,
            context=context,
            instructions="""Write a structured policy brief for a senior government official:

EXECUTIVE SUMMARY (3-4 sentences): Situation and recommended action.

BACKGROUND: Historical context and key actors (1-2 paragraphs).

CURRENT SITUATION: Detailed analysis of recent developments (2-3 paragraphs).

POLICY OPTIONS:
  Option A: [Conservative approach] — Pros: … Cons: … Risk: …
  Option B: [Moderate approach]    — Pros: … Cons: … Risk: …
  Option C: [Aggressive approach]  — Pros: … Cons: … Risk: …

RECOMMENDATION: Preferred option with rationale (1 paragraph).

IMPLEMENTATION: Immediate actions needed in next 30/60/90 days.

MONITORING INDICATORS: How to know if the situation is improving/deteriorating.""",
        )

    def generate_sitrep(self, hours: int = 24) -> dict:
        """Generate an auto-daily situation report from recent signals and graph activity."""
        context = self._gather_sitrep_context(hours=hours)
        now = datetime.now(_IST)
        from_dt = now - timedelta(hours=hours)

        brief = self._generate_document(
            brief_type="sitrep",
            domain="all",
            topic=f"Situation Report — {now.strftime('%Y-%m-%d')}",
            context=context,
            instructions=f"""Generate a comprehensive situation report for {now.strftime('%Y-%m-%d')}.
Covering the period: {from_dt.strftime('%Y-%m-%d %H:%M')} to {now.strftime('%H:%M')} IST.

EXECUTIVE OVERVIEW (2-3 sentences): The overall strategic picture today.

TOP DEVELOPMENTS (ranked by importance):
  1. [Geopolitics] …
  2. [Defence] …
  3. [Economics] …
  4. [Technology] …
  5. [Climate/Society] …

ANOMALIES & SIGNALS: Which entities or topics spiked unexpectedly.

INDIA IMPLICATIONS: Direct impact on Indian strategic interests.

WATCH LIST: 3 things to monitor closely in the next 24-48 hours.

INTELLIGENCE GAPS: What data sources are unavailable or stale.""",
        )

        # Add time range metadata
        brief["period_from"] = (now - timedelta(hours=hours)).isoformat()
        brief["period_to"] = now.isoformat()
        return brief

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _gather_domain_context(self, domain: str, hours: int = 48) -> str:
        """Pull recent articles, entities, and signals for a domain."""
        db = SessionLocal()
        try:
            cutoff = datetime.now(_IST) - timedelta(hours=hours)

            # Top articles
            articles = db.scalars(
                select(ScrapedArticle).where(
                    ScrapedArticle.domain == domain,
                    ScrapedArticle.scraped_at >= cutoff,
                    ScrapedArticle.importance_score >= 5.0,
                ).order_by(desc(ScrapedArticle.importance_score)).limit(10)
            ).all()

            # Active signals
            signals = db.scalars(
                select(DetectedSignal).where(
                    DetectedSignal.domain == domain,
                    DetectedSignal.expires_at >= datetime.now(_IST),
                ).limit(5)
            ).all()

        finally:
            db.close()

        # Neo4j: domain entities
        neo4j_context = ""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity)-[r:RELATES_TO]-(other:Entity)
                    WHERE e.last_updated >= $cutoff
                    WITH e, count(r) AS degree
                    ORDER BY degree DESC LIMIT 15
                    RETURN e.name AS name, e.type AS type, degree
                """, cutoff=(datetime.now(_IST) - timedelta(hours=hours)).isoformat())
                entities = [f"{r['name']} ({r['type']}, {r['degree']} relations)" for r in result]
                neo4j_context = "Key entities: " + ", ".join(entities[:10])
        except Exception as e:
            logger.error(f"Neo4j context fetch failed: {e}")

        article_context = "\n".join(
            f"- [{a.source}] {a.title} (score={a.importance_score:.1f})"
            for a in articles
        )
        signal_context = "\n".join(
            f"- SIGNAL: {s.signal_type} — {s.entity_name or s.cluster_label} (ratio={s.spike_ratio:.1f}×)"
            for s in signals
        )

        return f"""Domain: {domain}
Time period: last {hours} hours

{neo4j_context}

Top articles:
{article_context or 'No significant articles found.'}

Active signals:
{signal_context or 'No anomaly signals.'}"""

    def _gather_sitrep_context(self, hours: int = 24) -> str:
        """Pull broad cross-domain context for the sitrep."""
        db = SessionLocal()
        try:
            cutoff = datetime.now(_IST) - timedelta(hours=hours)

            articles = db.scalars(
                select(ScrapedArticle).where(
                    ScrapedArticle.scraped_at >= cutoff,
                    ScrapedArticle.importance_score >= 6.0,
                ).order_by(desc(ScrapedArticle.importance_score)).limit(20)
            ).all()

            signals = db.scalars(
                select(DetectedSignal).where(
                    DetectedSignal.expires_at >= datetime.now(_IST),
                ).order_by(desc(DetectedSignal.spike_ratio)).limit(10)
            ).all()
        finally:
            db.close()

        article_context = "\n".join(
            f"- [{a.domain or 'general'}][{a.source}] {a.title}"
            for a in articles
        )
        signal_context = "\n".join(
            f"- {s.severity.upper()} {s.signal_type}: {s.entity_name or s.cluster_label} ({s.domain})"
            for s in signals
        )
        return f"""Sitrep period: last {hours} hours

Top stories:
{article_context or 'No high-importance articles.'}

Intelligence signals:
{signal_context or 'No active signals.'}"""

    def _generate_document(
        self, brief_type: str, domain: str, topic: Optional[str],
        context: str, instructions: str
    ) -> dict:
        """Call LLM and persist the brief to Postgres."""
        system = f"""You are a senior strategic intelligence analyst advising the Government of India.
Your audience: ministers, senior bureaucrats, and national security advisors.
Domain expertise: geopolitics, defence, economics, technology, climate, and Indian strategic affairs.
Style: precise, evidence-based, actionable. Always cite specific entities and dates.
Never speculate without grounding. Distinguish between confirmed and rumored information."""

        user = f"""Generate a {brief_type.replace('_', ' ')} on the following topic.

TOPIC: {topic or domain}

CONTEXT DATA:
{context}

INSTRUCTIONS:
{instructions}"""

        content_md = ""
        try:
            resp = self.llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            content_md = resp.content
        except Exception as e:
            logger.error(f"PolicyBriefAgent LLM call failed: {e}")
            content_md = f"Brief generation failed: {e}"

        # Extract entities mentioned (simple heuristic: capitalized 2+ word sequences)
        import re
        entities = list(set(re.findall(r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+ ?)+', content_md)))[:20]

        brief_data = {
            "brief_type": brief_type,
            "domain": domain,
            "topic": topic or domain,
            "content": {"raw": content_md},
            "markdown_content": content_md,
            "entities": entities,
        }

        # Persist
        db = SessionLocal()
        try:
            brief = PolicyBrief(
                brief_type=brief_type,
                domain=domain,
                topic=topic or domain,
                content={"raw": content_md},
                markdown_content=content_md,
                entities=entities,
            )
            db.add(brief)
            db.commit()
            db.refresh(brief)
            brief_data["id"] = brief.id
            brief_data["created_at"] = brief.created_at.isoformat() if brief.created_at else None
        except Exception as e:
            db.rollback()
            logger.error(f"PolicyBriefAgent persist failed: {e}")
        finally:
            db.close()

        return brief_data
