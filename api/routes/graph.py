from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_AUTH

router = APIRouter(prefix="/api")
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

# Cypher fragment: parse RSS pub_date ("Thu, 06 Mar 2026 ...") into "YYYY-MM-DD"
_PARSE_DATE_CYPHER = """
    CASE
        WHEN {src}.pub_date CONTAINS ','
        THEN split(split({src}.pub_date, ', ')[1], ' ')[2] + '-' +
             CASE split(split({src}.pub_date, ', ')[1], ' ')[1]
                 WHEN 'Jan' THEN '01' WHEN 'Feb' THEN '02' WHEN 'Mar' THEN '03'
                 WHEN 'Apr' THEN '04' WHEN 'May' THEN '05' WHEN 'Jun' THEN '06'
                 WHEN 'Jul' THEN '07' WHEN 'Aug' THEN '08' WHEN 'Sep' THEN '09'
                 WHEN 'Oct' THEN '10' WHEN 'Nov' THEN '11' WHEN 'Dec' THEN '12'
                 ELSE '01' END + '-' +
             split(split({src}.pub_date, ', ')[1], ' ')[0]
        ELSE {src}.pub_date
    END
"""


def _compute_date_cutoff(date_range: str | None) -> str | None:
    """Return ISO date string cutoff for a given date range, or None."""
    if not date_range or date_range == "all":
        return None
    days_map = {"today": 0, "7d": 7, "30d": 30, "90d": 90}
    days = days_map.get(date_range, 7)
    # Use IST (UTC+5:30) since user is in India
    ist = timezone(timedelta(hours=5, minutes=30))
    cutoff = datetime.now(ist).date() - timedelta(days=days)
    return cutoff.isoformat()


@router.get("/graph")
def get_graph(
    limit: int = Query(100, ge=1, le=1000, description="Max entity/event nodes to return"),
    min_connections: int = Query(1, ge=0, description="Minimum relationships to include a node"),
    entity_type: str | None = Query(None, description="Filter by entity type (comma-separated, e.g. Country,Person)"),
    search: str | None = Query(None, description="Search for entity by name (case-insensitive)"),
    date_range: str | None = Query(None, description="Date filter: 'today', '7d', '30d', '90d', or 'all'"),
):
    """Return a filtered subgraph for visualization.

    By default returns the top-N most-connected Entity and Event nodes
    (no Article nodes — they're only used for provenance on click).
    """
    with driver.session() as session:
        # Build the WHERE clauses dynamically
        where_clauses = []
        params: dict = {"limit": limit, "min_connections": min_connections}

        if entity_type:
            types = [t.strip() for t in entity_type.split(",") if t.strip()]
            where_clauses.append("e.type IN $entity_types")
            params["entity_types"] = types

        if search:
            where_clauses.append("lower(e.name) CONTAINS lower($search)")
            params["search"] = search

        # Date range filter — only include entities evidenced by recent articles
        date_subquery = ""
        date_cutoff = _compute_date_cutoff(date_range)
        if date_cutoff:
            parse_expr = _PARSE_DATE_CYPHER.format(src="a")
            date_subquery = f"""
                MATCH (e)<-[:EVIDENCES]-(a:Article)
                WHERE a.pub_date IS NOT NULL AND a.pub_date <> ''
                WITH e, a, {parse_expr} AS parsed_date
                WHERE parsed_date >= $date_cutoff
                WITH DISTINCT e
            """
            params["date_cutoff"] = date_cutoff

        where_str = " AND ".join(where_clauses)
        if where_str:
            where_str = "WHERE " + where_str

        # Step 1: Get top-N entities ranked by connection count
        entity_query = f"""
            MATCH (e:Entity)
            {where_str}
            {'WITH e ' + date_subquery if date_subquery else ''}
            OPTIONAL MATCH (e)-[r:RELATES_TO]-()
            WITH e, count(r) AS degree
            WHERE degree >= $min_connections
            ORDER BY degree DESC
            LIMIT $limit
            RETURN collect({{
                id: elementId(e),
                labels: labels(e),
                props: properties(e),
                degree: degree
            }}) AS entities
        """
        entity_result = session.run(entity_query, **params).single()
        entity_nodes = entity_result["entities"] if entity_result else []

        # Collect entity IDs for edge query
        entity_ids = [n["id"] for n in entity_nodes]

        if not entity_ids:
            return {"nodes": [], "edges": []}

        # Step 2: Get events connected to these entities
        event_result = session.run("""
            MATCH (e:Entity)-[:INVOLVED_IN]->(ev:Event)
            WHERE elementId(e) IN $entity_ids
            RETURN collect(DISTINCT {
                id: elementId(ev),
                labels: labels(ev),
                props: properties(ev),
                degree: 0
            }) AS events
        """, entity_ids=entity_ids).single()
        event_nodes = event_result["events"] if event_result else []

        # Step 3: Get edges between these nodes
        all_ids = entity_ids + [n["id"] for n in event_nodes]

        edge_result = session.run("""
            MATCH (a)-[r]->(b)
            WHERE elementId(a) IN $all_ids AND elementId(b) IN $all_ids
              AND type(r) IN ['RELATES_TO', 'INVOLVED_IN']
            RETURN collect(DISTINCT {
                from: elementId(a),
                to: elementId(b),
                type: type(r),
                props: properties(r)
            }) AS edges
        """, all_ids=all_ids).single()
        edges = edge_result["edges"] if edge_result else []

        # Step 3b: Article co-occurrence edges — Person/Org ↔ Country
        # When no direct RELATES_TO edge exists, articles that mention both
        # a non-country entity and a country imply a real-world association.
        cooc_result = session.run("""
            MATCH (art:Article)-[:EVIDENCES]->(p:Entity)
            MATCH (art)-[:EVIDENCES]->(c:Entity {type: 'Country'})
            WHERE elementId(p) IN $entity_ids
              AND elementId(c) IN $entity_ids
              AND p <> c
              AND NOT p.type = 'Country'
            WITH p, c, count(DISTINCT art) AS co_count
            WHERE co_count >= 1
            RETURN elementId(p) AS from_id,
                   elementId(c) AS to_id,
                   co_count
        """, entity_ids=entity_ids).data()

        cooc_edges = [
            {
                "from": row["from_id"],
                "to": row["to_id"],
                "type": "CO_MENTIONED",
                "props": {"type": "co-mentioned", "co_count": row["co_count"]},
            }
            for row in cooc_result
        ]
        edges = edges + cooc_edges

        # Step 4: Attach source articles to nodes (single batched query)
        all_nodes = entity_nodes + event_nodes

        # Deduplicate
        seen = set()
        unique_nodes = []
        for n in all_nodes:
            if n["id"] not in seen:
                seen.add(n["id"])
                unique_nodes.append(n)

        # Batch-fetch sources for all nodes in one query
        node_names = [
            {"name": n["props"].get("name", ""), "label": n["labels"][0]}
            for n in unique_nodes
        ]

        # Build date filter clause for source articles
        source_date_filter = ""
        source_params: dict = {"nodes": node_names}
        if date_cutoff:
            parse_a = _PARSE_DATE_CYPHER.format(src="a")
            source_date_filter = f"AND a.pub_date IS NOT NULL AND ({parse_a}) >= $date_cutoff"
            source_params["date_cutoff"] = date_cutoff

        sources_result = session.run(f"""
            UNWIND $nodes AS node
            CALL {{
                WITH node
                WITH node WHERE node.label = 'Entity'
                MATCH (a:Article)-[:EVIDENCES]->(e:Entity {{name: node.name}})
                WHERE true {source_date_filter}
                RETURN node.name AS name, a.title AS title, a.url AS url,
                       a.source AS source, a.pub_date AS pub_date
                LIMIT 20
                UNION
                WITH node
                WITH node WHERE node.label = 'Event'
                MATCH (a:Article)-[:EVIDENCES]->(ev:Event {{name: node.name}})
                WHERE true {source_date_filter}
                RETURN node.name AS name, a.title AS title, a.url AS url,
                       a.source AS source, a.pub_date AS pub_date
                LIMIT 20
            }}
            RETURN name, collect({{title: title, url: url, source: source, pub_date: pub_date}}) AS sources
        """, **source_params).data()

        # Build lookup: name -> sources
        sources_by_name = {r["name"]: r["sources"] for r in sources_result}

        for node in unique_nodes:
            name = node["props"].get("name", "")
            node["sources"] = sources_by_name.get(name, [])

    return {"nodes": unique_nodes, "edges": edges}


@router.get("/entities/{entity_name}/timeline")
def entity_timeline(
    entity_name: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Chronological event timeline for an entity.

    Returns all events the entity participated in, with co-entities and
    source articles per event, ordered by date descending.

    Pagination via ?limit=N&offset=N.

    Source deduplication: we collect by url (unique identifier) and carry
    title/source/pub_date alongside it — avoids the unpredictable behaviour
    of collect(DISTINCT {map}) in older Neo4j versions where map equality
    is structural and can produce duplicates.
    """
    with driver.session() as session:
        # Total count for pagination metadata (cheap — no source fetch)
        count_result = session.run("""
            MATCH (e:Entity {name: $name})-[:INVOLVED_IN]->(ev:Event)
            RETURN count(ev) AS total
        """, name=entity_name).single()

        total = count_result["total"] if count_result else 0
        if total == 0:
            return {"entity": {"name": entity_name}, "total": 0, "timeline": []}

        # Fetch entity metadata
        entity_result = session.run("""
            MATCH (e:Entity {name: $name})
            RETURN e.type AS etype LIMIT 1
        """, name=entity_name).single()
        entity_type = entity_result["etype"] if entity_result else None

        # Main timeline query — paginated with SKIP / LIMIT
        # Sources: collect by url to guarantee uniqueness across Neo4j versions.
        rows = session.run("""
            MATCH (e:Entity {name: $name})-[:INVOLVED_IN]->(ev:Event)
            OPTIONAL MATCH (co:Entity)-[:INVOLVED_IN]->(ev)
            WHERE co.name <> $name
            WITH ev, collect(DISTINCT co.name) AS co_entities
            OPTIONAL MATCH (a:Article)-[:EVIDENCES]->(ev)
            WITH ev, co_entities,
                 collect(DISTINCT a.url)   AS src_urls,
                 collect(DISTINCT a.title) AS src_titles,
                 collect(DISTINCT a.source) AS src_sources,
                 collect(DISTINCT a.pub_date) AS src_dates
            ORDER BY ev.date DESC, ev.name ASC
            SKIP toInteger($offset)
            LIMIT toInteger($limit)
            RETURN ev.name        AS event_name,
                   ev.type        AS event_type,
                   ev.date        AS event_date,
                   ev.description AS event_description,
                   co_entities,
                   src_urls, src_titles, src_sources, src_dates
        """, name=entity_name, offset=offset, limit=limit).data()

        timeline = []
        for row in rows:
            # Zip source arrays back into dicts; filter out None url rows
            urls = row["src_urls"] or []
            titles = row["src_titles"] or []
            sources_names = row["src_sources"] or []
            pub_dates = row["src_dates"] or []

            # Pad shorter lists so zip is safe
            max_len = max(len(urls), len(titles), len(sources_names), len(pub_dates), 1)
            urls       = (urls       + [None] * max_len)[:max_len]
            titles     = (titles     + [None] * max_len)[:max_len]
            sources_names = (sources_names + [None] * max_len)[:max_len]
            pub_dates  = (pub_dates  + [None] * max_len)[:max_len]

            article_sources = [
                {"url": u, "title": t, "source": s, "pub_date": d}
                for u, t, s, d in zip(urls, titles, sources_names, pub_dates)
                if u  # url is our unique key — skip rows where it's None
            ]

            timeline.append({
                "event_name": row["event_name"],
                "event_type": row["event_type"],
                "date": row["event_date"],
                "description": row["event_description"],
                "co_entities": [c for c in (row["co_entities"] or []) if c],
                "sources": article_sources,
            })

        return {
            "entity": {"name": entity_name, "type": entity_type},
            "total": total,
            "offset": offset,
            "limit": limit,
            "timeline": timeline,
        }


@router.get("/graph/stats")
def get_graph_stats():
    """Return aggregate stats about the graph (for display)."""
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[r:RELATES_TO]-()
            WITH e.type AS type, count(DISTINCT e) AS count, count(r) AS rels
            RETURN type, count, rels
            ORDER BY count DESC
        """).data()

        total_entities = sum(r["count"] for r in result)
        total_rels = sum(r["rels"] for r in result) // 2  # counted twice

        events = session.run("MATCH (ev:Event) RETURN count(ev) AS count").single()
        articles = session.run("MATCH (a:Article) RETURN count(a) AS count").single()

        return {
            "total_entities": total_entities,
            "total_relationships": total_rels,
            "total_events": events["count"] if events else 0,
            "total_articles": articles["count"] if articles else 0,
            "by_type": result,
        }
