"""Fast Neo4j seed: GLiNER2 only — skips all LLM steps.

Reads scraped articles from Postgres, runs GLiNER2 entity+relation extraction,
writes directly to Neo4j. No Ollama required. Runs in ~5-10 min for 50 articles.

Usage:
    python -m scripts.seed_neo4j_fast [--limit N]
"""

import argparse
import logging
import re
import sys

from neo4j import GraphDatabase
from sqlalchemy import select

sys.path.insert(0, ".")

from config import NEO4J_URI, NEO4J_AUTH
from models.database import SessionLocal
from models.scraped_article import ScrapedArticle
from agents.extraction import ENTITY_LABELS, RELATION_LABELS, _ENTITY_TYPE_MAP, ExtractionAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_ACRONYMS = {
    "us", "usa", "uk", "eu", "un", "nato", "who", "imf", "gdp", "uae",
    "opec", "asean", "brics", "cia", "fbi", "nsa", "isro", "drdo", "ai",
    "bjp", "aap", "rbi", "sebi", "iit", "nit", "isi", "isis",
    "nasa", "gst", "pm", "loc", "pla", "ccp", "prc", "dprk", "rok",
}


def _normalize(name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip())
    return " ".join(
        word.upper() if word.lower() in _ACRONYMS else word.title()
        for word in name.split()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    # Load articles from Postgres
    db = SessionLocal()
    try:
        articles = db.scalars(
            select(ScrapedArticle)
            .where(ScrapedArticle.full_text.isnot(None))
            .order_by(ScrapedArticle.scraped_at.desc())
            .limit(args.limit)
        ).all()
    finally:
        db.close()

    logger.info(f"Loaded {len(articles)} articles from Postgres")
    if not articles:
        logger.error("No articles found — run the scraper first")
        return

    # Load GLiNER2
    logger.info("Loading GLiNER2 model…")
    from gliner2 import GLiNER2
    gliner = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    schema = (
        gliner.create_schema()
        .entities(ENTITY_LABELS)
        .relations(RELATION_LABELS)
    )
    logger.info("GLiNER2 ready")

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    total_entities = 0
    total_relations = 0

    for i, article in enumerate(articles):
        text = (article.full_text or "")[:4000]
        if not text.strip():
            continue

        try:
            raw = gliner.extract(text, schema, include_confidence=True)
        except Exception as e:
            logger.warning(f"[{i+1}] GLiNER2 failed on '{article.title[:50]}': {e}")
            continue

        # Parse entities
        entities_data = []
        seen = set()
        for etype, items in raw.get("entities", {}).items():
            canonical_type = _ENTITY_TYPE_MAP.get(etype, etype.title())
            for item in items:
                name = item["text"] if isinstance(item, dict) else str(item)
                conf = item.get("confidence", 1.0) if isinstance(item, dict) else 1.0
                norm = _normalize(name)
                if norm not in seen and conf >= 0.3:
                    seen.add(norm)
                    entities_data.append({
                        "name": norm,
                        "type": canonical_type,
                        "pub_date": str(article.pub_date) if article.pub_date else None,
                        "url": article.url,
                    })

        # Parse relations
        relations_data = []
        for rel_type, pairs in raw.get("relation_extraction", {}).items():
            for pair in pairs:
                src, tgt, conf = ExtractionAgent._parse_relation_pair(pair)
                if src and tgt:
                    relations_data.append({
                        "source": _normalize(src),
                        "target": _normalize(tgt),
                        "relation": rel_type,
                        "confidence": conf,
                        "url": article.url,
                    })

        if not entities_data:
            logger.info(f"[{i+1}/{len(articles)}] No entities: {article.title[:60]}")
            continue

        # Write to Neo4j
        with driver.session() as s:
            s.run("""
                MERGE (a:Article {url: $url})
                SET a.title = $title, a.source = $source, a.pub_date = $pub_date
            """, url=article.url, title=article.title,
                 source=article.source,
                 pub_date=str(article.pub_date) if article.pub_date else None)

            if entities_data:
                s.run("""
                    UNWIND $rows AS row
                    MERGE (e:Entity {name: row.name})
                    ON CREATE SET e.type = row.type, e.first_seen = row.pub_date
                    SET e.last_updated = row.pub_date
                    WITH e, row
                    MATCH (a:Article {url: row.url})
                    MERGE (a)-[:EVIDENCES]->(e)
                """, rows=entities_data)

            if relations_data:
                s.run("""
                    UNWIND $rows AS row
                    MATCH (src:Entity {name: row.source}), (tgt:Entity {name: row.target})
                    MERGE (src)-[r:RELATES_TO {type: row.relation}]->(tgt)
                    ON CREATE SET r.confidence = row.confidence, r.evidence_count = 1
                    ON MATCH SET
                        r.evidence_count = coalesce(r.evidence_count, 1) + 1
                """, rows=relations_data)

        total_entities += len(entities_data)
        total_relations += len(relations_data)
        logger.info(
            f"[{i+1}/{len(articles)}] {len(entities_data)} entities, "
            f"{len(relations_data)} relations — {article.title[:55]}"
        )

    driver.close()
    logger.info(f"Done: {total_entities} entities, {total_relations} relations written to Neo4j")


if __name__ == "__main__":
    main()
