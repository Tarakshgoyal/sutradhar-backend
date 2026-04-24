"""LangGraph-based RAG chat agent for querying the knowledge graph.

Architecture:
  User question
       │
       ▼
  ┌─────────────┐
  │  Router LLM │  Decide: need graph query, or can answer directly?
  └──────┬──────┘
         │
    ┌────┴────┐
    ▼         ▼
 [cypher]  [answer]
    │
    ▼
 Generate 1..N Cypher queries (LLM decomposes complex questions)
    │
    ▼
 Execute all queries against Neo4j, combine results
    │
    ▼
 [synthesize]  ──▶  Final answer with citations
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from langchain_core.output_parsers import JsonOutputParser

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from neo4j import GraphDatabase, Query
from sqlalchemy import select

from config import NEO4J_URI, NEO4J_AUTH
from models.database import SessionLocal
from models.scraped_article import ScrapedArticle

logger = logging.getLogger(__name__)


def _llm_invoke_with_retry(llm, messages, max_retries: int = 3, base_delay: float = 2.0):
    """Invoke an LLM with exponential backoff retry on transient errors."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err_name = type(e).__name__
            is_retryable = any(kw in err_name for kw in ["Connection", "Timeout", "RateLimit", "ServiceUnavailable"])
            if not is_retryable or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"LLM call failed ({err_name}), retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)

# ---------------------------------------------------------------------------
# Neo4j schema description (injected into prompts so LLM knows the structure)
# ---------------------------------------------------------------------------

NEO4J_SCHEMA = """
Node labels and properties:
- (:Entity {name, type, first_seen, last_updated})
    type is one of: Person, Organization, Country, Location, Policy, Technology,
                    Economic_Indicator, Military_Asset, Resource
- (:Event {name, date, status})
    status is one of: ongoing, concluded, announced, rumored
- (:Article {url, title, source, pub_date})

Relationships:
- (Entity)-[:RELATES_TO {type, since, confidence, causal, current}]->(Entity)
    The `type` property holds the relation verb. Possible values:
    sanctions, allied_with, opposes, trades_with, supplies_to, invaded,
    leads, founded, acquired, located_in, manufactures, funds,
    threatens, negotiates_with, member_of, disrupts, signed_agreement_with,
    deployed_to, develops, exports_to, imports_from, cooperates_with,
    competes_with, attacks, blocks, supports
    `causal` is a boolean. `since` and `current` are optional.

- (Entity)-[:INVOLVED_IN]->(Event)
- (Article)-[:EVIDENCES]->(Entity)
- (Article)-[:EVIDENCES]->(Event)
- (Article)-[:EVIDENCES_REL {relation_type}]->(Entity)

IMPORTANT: Entity-to-Entity relationships are ALL stored as :RELATES_TO edges
with a `type` property. Do NOT use dynamic relationship types like [:sanctions].
Always filter with `r.type = "sanctions"` instead.
"""

CYPHER_SYSTEM = f"""You are a Neo4j Cypher expert for a geopolitical knowledge graph.

{NEO4J_SCHEMA}

Query patterns for common question types:

1. "Tell me about X" / "What do you know about X":
   IMPORTANT: Use SEPARATE queries for each aspect to avoid cartesian product explosion.
   Do NOT combine multiple OPTIONAL MATCH clauses in one query — this creates
   a massive cross-product that will hang the database.

   Query 1a — outgoing relationships:
   MATCH (e:Entity)
   WHERE lower(e.name) = lower("X")
   OPTIONAL MATCH (e)-[r:RELATES_TO]->(t:Entity)
   RETURN e.name AS entity, e.type AS type,
          collect(DISTINCT {{relation: r.type, target: t.name, target_type: t.type}}) AS outgoing
   LIMIT 5

   Query 1b — incoming relationships:
   MATCH (e:Entity)
   WHERE lower(e.name) = lower("X")
   OPTIONAL MATCH (e)<-[r2:RELATES_TO]-(s:Entity)
   RETURN e.name AS entity, e.type AS type,
          collect(DISTINCT {{relation: r2.type, source: s.name, source_type: s.type}}) AS incoming
   LIMIT 5

   Query 1c — events and articles:
   MATCH (e:Entity)
   WHERE lower(e.name) = lower("X")
   OPTIONAL MATCH (e)-[:INVOLVED_IN]->(ev:Event)
   WITH e, collect(DISTINCT {{event: ev.name, date: ev.date, status: ev.status}}) AS events
   OPTIONAL MATCH (a:Article)-[:EVIDENCES]->(e)
   RETURN e.name AS entity, e.type AS type, events,
          collect(DISTINCT {{title: a.title, source: a.source, url: a.url, pub_date: a.pub_date}}) AS articles
   LIMIT 5

2. "How are X and Y related":
   MATCH (a:Entity), (b:Entity)
   WHERE lower(a.name) CONTAINS lower("X") AND lower(b.name) CONTAINS lower("Y")
   OPTIONAL MATCH p = (a)-[:RELATES_TO*1..3]-(b)
   RETURN a.name, b.name, [r IN relationships(p) | {{type: r.type, causal: r.causal}}] AS rels,
          [n IN nodes(p) | n.name] AS path
   LIMIT 10

3. "Who/what does X sanction/trade with/etc":
   MATCH (e:Entity)-[r:RELATES_TO]->(t:Entity)
   WHERE lower(e.name) CONTAINS lower("X") AND r.type = "sanctions"
   RETURN e.name, r.type, t.name, t.type
   LIMIT 25

4. "What events involve X":
   MATCH (e:Entity)-[:INVOLVED_IN]->(ev:Event)
   WHERE lower(e.name) CONTAINS lower("X")
   OPTIONAL MATCH (a:Article)-[:EVIDENCES]->(ev)
   RETURN ev.name, ev.date, ev.status, e.name,
          collect(DISTINCT a.title) AS source_articles
   LIMIT 25

Rules:
- Return ONLY valid Cypher READ queries (no mutations).
- Use case-insensitive matching with lower() or CONTAINS for entity names.
- For exact entity lookups (e.g. a country name), prefer `=` over `CONTAINS` to avoid matching too many entities.
  GOOD: lower(e.name) = lower("india")
  BAD:  lower(e.name) CONTAINS lower("india")  — this also matches "Indian Oil Corporation", "Indian Ocean", etc.
  Use CONTAINS only when you need substring/fuzzy matching.
- ALWAYS use :RELATES_TO with r.type for relationship filtering. Never use dynamic edge labels.
- NEVER combine more than 2 OPTIONAL MATCH clauses in a single query. Multiple OPTIONAL MATCHes
  create cartesian products that hang the database. Split into separate queries instead.
- For general "tell me about" questions, use pattern #1 (multiple queries).
- Always include source articles (via :EVIDENCES) when possible — users want provenance.
- Limit results to 25 rows unless the user asks for more.
- Always alias return columns clearly.
- If the question cannot be answered with a Cypher query, return an empty list of queries.
- Do NOT wrap the query in markdown code fences.

Search term rules (CRITICAL):
- Use INDIVIDUAL KEYWORDS in CONTAINS, not compound phrases.
  GOOD: lower(e.name) CONTAINS lower("iran")
  BAD:  lower(e.name) CONTAINS lower("iran war")
  GOOD: lower(e.name) CONTAINS lower("oil")
  BAD:  lower(e.name) CONTAINS lower("oil price")
- Entity names in the graph are short labels like "Iran", "United States", "OPEC", "Crude Oil".
  They do NOT contain full phrases like "iran war" or "oil price impact".
- When a user says "iran war impact on oil prices", search for "iran" and "oil" separately.
- If needed, combine conditions with OR to catch variations (e.g. "oil" OR "crude" OR "petroleum").

Multiple queries:
- You SHOULD generate multiple Cypher queries when the question involves ANY of:
  1. Causal or impact questions (e.g. "X impact on Y") — query X and Y separately
  2. Multiple distinct entities or topics (e.g. "iran" + "oil")
  3. Comparing or contrasting entities (e.g. "A vs B")
  4. Questions that span different node types (entities + events + articles)
  5. Broad questions where a single query is unlikely to capture all relevant data
- For simple, focused questions (single entity lookup), a single query is fine.
- Each query should be self-contained and independently executable.
- IMPORTANT: cast a wide net. It is MUCH better to return multiple queries that
  cover different angles than a single narrow query that misses relevant data.
- If the question cannot be answered with a Cypher query, return an empty list of queries.
"""


# ---------------------------------------------------------------------------
# Structured output models for Cypher generation
# ---------------------------------------------------------------------------

class CypherQuery(BaseModel):
    """A single Cypher query with its purpose."""
    purpose: str = Field(description="Brief description of what this query retrieves (e.g. 'Iran war events', 'oil-related entities')")
    cypher: str = Field(description="A valid Cypher READ query. Do NOT wrap in code fences.")


class CypherQueryPlan(BaseModel):
    """One or more Cypher queries to answer the user's question."""
    queries: list[CypherQuery] = Field(
        description="List of Cypher queries to execute. Use multiple queries for complex questions. Empty list if no query can answer the question."
    )

SYNTHESIZE_SYSTEM = """You are a senior geopolitical intelligence analyst advising India's decision-makers \
on strategy, policy, and national advantage. Your job is to deliver clear, actionable intelligence \
— never to deflect or plead ignorance.

Data sourcing hierarchy:
1. **Live graph data** (provided below): if the knowledge graph contains relevant entities, \
relationships, or events, lead with that — it reflects the most recent scraped intelligence.
2. **Expert knowledge**: when the graph data is absent, sparse, or only partially answers the \
question, draw on your deep knowledge of geopolitics, international relations, India's strategic \
interests, economic policy, and global affairs to fill the gaps. Clearly label graph-sourced \
claims with "(per graph)" and knowledge-based analysis with "(analyst assessment)".
3. **Never refuse to engage**: even with zero graph data you have expert knowledge — use it.

Style rules:
- Write in flowing prose organized by theme (diplomacy, military, economy, geopolitics, etc.).
- Synthesize and connect the dots: explain *why* relationships matter, what caused what, and \
what the strategic implications are for India and global stability.
- For policy questions: identify the 2-3 most vital policy levers, the stakeholders involved, \
the risks of inaction, and the India-specific angle.
- Lead with the most significant/recent developments. Skip trivial or redundant details.
- Use bullet points sparingly — only for short reference lists (e.g. key policy recommendations).
  The main body should be narrative paragraphs.
- Cite source articles by title when referencing graph-sourced claims.
- Keep the total answer to 300-600 words. Be dense with insight, not verbose.
- Do NOT reproduce raw tables, lists of IDs, or data dumps from the graph.
- Do NOT say "I don't have data" or "the knowledge graph returned no results" — if the graph \
is empty, simply provide expert analysis without mentioning the graph at all.
"""

ROUTER_SYSTEM = """You are a router for a geopolitical knowledge graph assistant.
Given a user message, decide whether it requires querying the Neo4j graph database
or can be answered directly (e.g. greetings, meta-questions about the system, clarifications).

The graph contains Entity nodes (persons, countries, organizations, locations, etc.),
Event nodes, Article nodes, and typed relationships between them.

Respond with exactly one word:
- "graph" if the question needs data from the knowledge graph
- "direct" if you can answer without querying (greetings, system questions, clarifications)
"""


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    """State flowing through the LangGraph."""
    messages: list                  # conversation history (HumanMessage / AIMessage)
    question: str                   # current user question
    route: str                      # "graph" or "direct"
    cypher_queries: list[str]       # one or more generated Cypher queries
    graph_results: list[str]        # formatted results per query
    graph_context: str              # combined context string for synthesis
    article_urls: list[str]         # article URLs found in graph results (for Postgres lookup)
    article_content: str            # fetched article full text for deeper context
    answer: str                     # final answer to return


# ---------------------------------------------------------------------------
# Graph Agent
# ---------------------------------------------------------------------------

class GraphChatAgent:
    """LangGraph agent that answers questions using the Neo4j knowledge graph."""

    def __init__(self, model: str = "openai/gpt-oss-120b"):
        self.llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0, num_predict=4096)
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

        # Cypher generation: LLM -> JSON parser
        self.cypher_parser = JsonOutputParser(pydantic_object=CypherQueryPlan)
        self.cypher_llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0, num_predict=4096)
        self.cypher_system = CYPHER_SYSTEM + "\n" + self.cypher_parser.get_format_instructions()

        self.graph = self._build_graph()

    def close(self):
        self.driver.close()

    # --- Node functions ---

    def _route_node(self, state: ChatState) -> ChatState:
        """Decide whether we need a graph query or can answer directly."""
        logger.info(f"[STEP: router] Question: {state['question']}")
        messages = [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=state["question"]),
        ]
        resp = _llm_invoke_with_retry(self.llm, messages)
        route = resp.content.strip().lower()
        if "graph" in route:
            state["route"] = "graph"
        else:
            state["route"] = "direct"
        logger.info(f"[STEP: router] Route decided: {state['route']}")
        return state

    def _generate_cypher_node(self, state: ChatState) -> ChatState:
        """Generate one or more Cypher queries from the user question."""
        logger.info(f"[STEP: generate_cypher] Generating Cypher for: {state['question']}")
        # Build input with conversation context
        history = _format_history(state["messages"][-6:])  # last 3 turns
        if history:
            user_input = f"Conversation so far:\n{history}\n\nNew question: {state['question']}"
        else:
            user_input = state["question"]

        messages = [
            SystemMessage(content=self.cypher_system),
            HumanMessage(content=user_input),
        ]

        try:
            resp = _llm_invoke_with_retry(self.cypher_llm, messages)
            data = self.cypher_parser.parse(resp.content)
            plan = CypherQueryPlan(**data)
        except Exception as e:
            logger.error(f"Cypher generation failed: {e}")
            state["cypher_queries"] = []
            state["route"] = "direct"  # fallback to direct answer
            return state

        queries = [_clean_cypher(q.cypher) for q in plan.queries]
        queries = [q for q in queries if q]

        state["cypher_queries"] = queries
        logger.info(f"Generated {len(queries)} Cypher query/queries")
        for i, q in enumerate(plan.queries):
            logger.info(f"  Query {i+1} ({q.purpose}): {q.cypher}")
        return state

    def _execute_cypher_node(self, state: ChatState) -> ChatState:
        """Execute all Cypher queries against Neo4j and combine results."""
        queries = state.get("cypher_queries", [])

        if not queries:
            state["graph_results"] = []
            state["graph_context"] = ""
            state["article_urls"] = []
            state["route"] = "direct"  # fallback to direct answer
            return state

        all_results: list[str] = []
        all_entity_names: list[str] = []
        all_article_urls: list[str] = []

        with self.driver.session() as session:
            for i, cypher in enumerate(queries):
                label = f"Query {i+1}/{len(queries)}"
                logger.info(f"{label}: Executing Cypher: {cypher[:200]}")

                # Safety: reject mutations
                upper = cypher.upper()
                if any(kw in upper for kw in ["CREATE", "MERGE", "DELETE", "SET ", "REMOVE ", "DROP "]):
                    all_results.append(f"[{label} rejected: only read queries are allowed]")
                    continue

                try:
                    import time as _t
                    _q_start = _t.time()
                    result = session.run(Query(cypher, timeout=30))
                    records = [dict(r) for r in result]
                    logger.info(f"{label}: Got {len(records)} rows in {_t.time()-_q_start:.2f}s")

                    if not records:
                        all_results.append(f"{label}: No results.")
                    else:
                        context = _format_records(records, cypher)
                        all_results.append(f"{label}:\n{context}")
                        logger.info(f"{label} returned {len(records)} rows")

                        # Collect article URLs from raw records
                        all_article_urls.extend(_extract_article_urls(records))

                        # Collect entity names for neighborhood enrichment
                        if _is_thin_result(records):
                            all_entity_names.extend(_extract_entity_names(records))

                except Exception as e:
                    logger.error(f"{label} failed: {e}")
                    all_results.append(f"{label} failed with error: {e}")

            # Neighborhood enrichment across all thin results
            if all_entity_names:
                unique_names = list(dict.fromkeys(all_entity_names))
                extra = self._fetch_neighborhood(session, unique_names)
                if extra:
                    all_results.append(f"--- Additional context (entity neighborhood) ---\n{extra}")

        state["graph_results"] = all_results
        state["graph_context"] = "\n\n".join(all_results)
        state["article_urls"] = list(dict.fromkeys(all_article_urls))  # dedup, preserve order
        return state

    def _fetch_neighborhood(self, session, entity_names: list[str]) -> str:
        """Fetch relationships, events, and articles for a list of entity names."""
        all_lines = []
        for name in entity_names[:3]:  # cap at 3 entities
            try:
                # Fetch outgoing relationships
                r_out = session.run(
                    'MATCH (e:Entity {name: $name}) '
                    'OPTIONAL MATCH (e)-[r:RELATES_TO]->(t:Entity) '
                    'RETURN e.name AS entity, e.type AS type, '
                    'collect(DISTINCT {relation: r.type, target: t.name, target_type: t.type}) AS outgoing '
                    'LIMIT 1',
                    name=name,
                )
                rec_out = r_out.single()
                if not rec_out:
                    continue

                lines = [f"\nEntity: {rec_out['entity']} ({rec_out['type']})"]

                outgoing = [r for r in rec_out["outgoing"] if r.get("target")]
                if outgoing:
                    lines.append("  Outgoing relationships:")
                    for r in outgoing[:20]:
                        lines.append(f"    → {r['relation']} → {r['target']} ({r.get('target_type', '')})")

                # Fetch incoming relationships
                r_in = session.run(
                    'MATCH (e:Entity {name: $name}) '
                    'OPTIONAL MATCH (e)<-[r2:RELATES_TO]-(s:Entity) '
                    'RETURN collect(DISTINCT {relation: r2.type, source: s.name, source_type: s.type}) AS incoming '
                    'LIMIT 1',
                    name=name,
                )
                rec_in = r_in.single()
                if rec_in:
                    incoming = [r for r in rec_in["incoming"] if r.get("source")]
                    if incoming:
                        lines.append("  Incoming relationships:")
                        for r in incoming[:20]:
                            lines.append(f"    ← {r['relation']} ← {r['source']} ({r.get('source_type', '')})")

                # Fetch events (separate query to avoid cartesian product)
                r_ev = session.run(
                    'MATCH (e:Entity {name: $name})-[:INVOLVED_IN]->(ev:Event) '
                    'RETURN DISTINCT ev.name AS event, ev.date AS date, ev.status AS status '
                    'ORDER BY ev.date DESC LIMIT 15',
                    name=name,
                )
                events = [dict(r) for r in r_ev]
                if events:
                    lines.append("  Events:")
                    for e in events:
                        lines.append(f"    - {e['event']} (date: {e.get('date', '?')}, status: {e.get('status', '?')})")

                # Fetch source articles (separate query)
                r_art = session.run(
                    'MATCH (a:Article)-[:EVIDENCES]->(e:Entity {name: $name}) '
                    'RETURN DISTINCT a.title AS title, a.source AS source, a.url AS url '
                    'ORDER BY a.pub_date DESC LIMIT 10',
                    name=name,
                )
                articles = [dict(r) for r in r_art]
                if articles:
                    lines.append(f"  Source articles ({len(articles)}):")
                    for a in articles:
                        lines.append(f"    - \"{a['title']}\" ({a.get('source', '')})")

                if len(lines) > 1:
                    all_lines.extend(lines)

            except Exception as e:
                logger.debug(f"Neighborhood fetch for '{name}' failed: {e}")

        return "\n".join(all_lines)

    def _fetch_articles_node(self, state: ChatState) -> ChatState:
        """Fetch full article content from Postgres for articles found in graph results."""
        urls = state.get("article_urls", [])
        logger.info(f"[STEP: fetch_articles] {len(urls)} article URLs to fetch")
        if not urls:
            state["article_content"] = ""
            return state

        db = SessionLocal()
        try:
            rows = db.execute(
                select(ScrapedArticle.url, ScrapedArticle.title, ScrapedArticle.full_text)
                .where(ScrapedArticle.url.in_(urls[:5]))
            ).all()

            parts = []
            for url, title, text in rows:
                if text:
                    snippet = text[:2000]
                    if len(text) > 2000:
                        snippet += "\n... [truncated]"
                    parts.append(f"### {title}\nURL: {url}\n{snippet}")

            state["article_content"] = "\n\n".join(parts)
            logger.info(f"Fetched content for {len(parts)} articles from Postgres")
        except Exception as e:
            logger.error(f"Article content fetch failed: {e}")
            state["article_content"] = ""
        finally:
            db.close()

        return state

    def _synthesize_node(self, state: ChatState) -> ChatState:
        """Synthesize a natural-language answer from graph results."""
        graph_ctx = state.get("graph_context", "")
        article_ctx = state.get("article_content", "")

        # Cap context size to avoid overwhelming the LLM
        MAX_GRAPH_CTX = 12000
        MAX_ARTICLE_CTX = 6000
        if len(graph_ctx) > MAX_GRAPH_CTX:
            graph_ctx = graph_ctx[:MAX_GRAPH_CTX] + "\n... [truncated — data continues]"
        if len(article_ctx) > MAX_ARTICLE_CTX:
            article_ctx = article_ctx[:MAX_ARTICLE_CTX] + "\n... [truncated]"

        logger.info(f"[STEP: synthesize] Graph context: {len(graph_ctx)} chars, article content: {len(article_ctx)} chars")
        user_content = (
            f"User question: {state['question']}\n\n"
            f"Knowledge graph data:\n{graph_ctx}"
        )

        if article_ctx:
            user_content += f"\n\nFull article content for reference:\n{article_ctx}"

        messages = [
            SystemMessage(content=SYNTHESIZE_SYSTEM),
            HumanMessage(content=user_content),
        ]
        resp = _llm_invoke_with_retry(self.llm, messages)
        state["answer"] = resp.content
        return state

    def _direct_answer_node(self, state: ChatState) -> ChatState:
        """Answer without querying the graph (greetings, meta questions, policy questions, etc.)."""
        history = _format_history(state["messages"][-6:])
        messages = [
            SystemMessage(content=(
                "You are a senior geopolitical intelligence analyst advising India's decision-makers "
                "on strategy, policy, and national advantage. "
                "For policy, strategy, or analytical questions: provide a clear, expert-level "
                "intelligence briefing — identify key policy levers, stakeholders, risks, and "
                "India's strategic angle. Be direct and actionable. "
                "For greetings or system questions: briefly explain you are an intelligence assistant "
                "that can answer questions about entities, events, geopolitical trends, and policy strategy. "
                "Never deflect with 'I don't have data' — you have deep expert knowledge, use it."
            )),
        ]
        if history:
            messages.append(HumanMessage(content=f"Conversation:\n{history}\n\nUser: {state['question']}"))
        else:
            messages.append(HumanMessage(content=state["question"]))

        resp = _llm_invoke_with_retry(self.llm, messages)
        state["answer"] = resp.content
        return state

    # --- Routing edge ---

    def _route_edge(self, state: ChatState) -> Literal["generate_cypher", "direct_answer"]:
        if state.get("route") == "graph":
            return "generate_cypher"
        return "direct_answer"

    # --- Build the graph ---

    def _build_graph(self) -> StateGraph:
        g = StateGraph(ChatState)

        # Nodes
        g.add_node("router", self._route_node)
        g.add_node("generate_cypher", self._generate_cypher_node)
        g.add_node("execute_cypher", self._execute_cypher_node)
        g.add_node("fetch_articles", self._fetch_articles_node)
        g.add_node("synthesize", self._synthesize_node)
        g.add_node("direct_answer", self._direct_answer_node)

        # Edges
        g.add_edge(START, "router")
        g.add_conditional_edges("router", self._route_edge)
        g.add_edge("generate_cypher", "execute_cypher")
        g.add_edge("execute_cypher", "fetch_articles")
        g.add_edge("fetch_articles", "synthesize")
        g.add_edge("synthesize", END)
        g.add_edge("direct_answer", END)

        return g.compile()

    # --- Public API ---

    def chat(self, question: str, history: list | None = None) -> dict:
        """Run the agent on a question. Returns dict with answer + metadata.

        Args:
            question: The user's question.
            history: List of {"role": "user"|"assistant", "content": "..."} dicts.

        Returns:
            {
                "answer": str,
                "cypher": str | None,
                "graph_context": str | None,
                "route": str,
            }
        """
        messages = []
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

        state: ChatState = {
            "messages": messages,
            "question": question,
            "route": "",
            "cypher_queries": [],
            "graph_results": [],
            "graph_context": "",
            "article_urls": [],
            "article_content": "",
            "answer": "",
        }

        result = self.graph.invoke(state)
        return {
            "answer": result["answer"],
            "cypher": result.get("cypher_queries") or None,
            "graph_context": result.get("graph_context") or None,
            "route": result.get("route", ""),
        }


import re

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_cypher(raw: str) -> str:
    """Strip code fences from LLM-generated Cypher."""
    s = raw.strip()
    # Remove markdown code fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:cypher)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _format_history(messages: list) -> str:
    """Format message list into a readable conversation string."""
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines)


def _format_neo4j_value(v) -> str:
    """Format a Neo4j value for display."""
    if v is None:
        return "null"
    if isinstance(v, dict):
        # Node or relationship properties
        parts = [f"{k}={_format_neo4j_value(val)}" for k, val in v.items()]
        return "{" + ", ".join(parts) + "}"
    if isinstance(v, list):
        return "[" + ", ".join(_format_neo4j_value(i) for i in v) + "]"
    return str(v)


def _dedup_list_values(items: list) -> list:
    """Deduplicate a list of dicts by their string representation."""
    seen = set()
    result = []
    for item in items:
        key = str(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _format_records(records: list[dict], cypher: str) -> str:
    """Format Neo4j records as readable text with deduplication and truncation."""
    lines = []
    for i, rec in enumerate(records[:25]):
        parts = []
        for k, v in rec.items():
            # Deduplicate and truncate list values
            if isinstance(v, list):
                v = _dedup_list_values(v)
                if len(v) > 25:
                    v = v[:25]  # cap long lists
            parts.append(f"{k}: {_format_neo4j_value(v)}")
        lines.append(f"  {i+1}. {', '.join(parts)}")
    return f"Results ({len(records)} rows):\n" + "\n".join(lines)


def _is_thin_result(records: list[dict]) -> bool:
    """Check if query results are 'thin' — just entity name/type with no context."""
    if len(records) > 5:
        return False
    # Check if the result only has simple scalar values (no lists, no nested dicts)
    total_values = 0
    non_null_values = 0
    list_values = 0
    for rec in records:
        for v in rec.values():
            total_values += 1
            if v is not None:
                non_null_values += 1
            if isinstance(v, list) and len(v) > 0:
                list_values += 1
    # Thin if few columns, no list values (no aggregated data)
    return list_values == 0 and non_null_values <= 6


def _extract_entity_names(records: list[dict]) -> list[str]:
    """Extract entity name strings from query results."""
    names = []
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, str) and k.lower() in ("name", "entity", "entity_name", "e.name"):
                names.append(v)
    return list(dict.fromkeys(names))  # dedup preserving order


def _extract_article_urls(records: list[dict]) -> list[str]:
    """Extract article URLs from raw Neo4j records.

    Handles both flat records (with a column named 'url' or 'a.url')
    and nested dicts/lists (e.g. collected article objects like
    {url: ..., title: ...}).
    """
    urls: list[str] = []
    for rec in records:
        for key, val in rec.items():
            # Direct url column (e.g. RETURN a.url)
            if isinstance(val, str) and key.lower() in ("url", "a.url", "article_url"):
                urls.append(val)
            # Nested dict with a 'url' key (e.g. collected article object)
            elif isinstance(val, dict) and "url" in val and val["url"]:
                urls.append(val["url"])
            # List of dicts (e.g. collect(DISTINCT {url: a.url, title: a.title}))
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and "url" in item and item["url"]:
                        urls.append(item["url"])
    return urls
