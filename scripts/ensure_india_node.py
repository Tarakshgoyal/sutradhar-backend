"""Ensure a canonical India country node exists in Neo4j.

Usage:
    python scripts/ensure_india_node.py
"""

from pathlib import Path
import sys

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import NEO4J_AUTH, NEO4J_URI


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    try:
        with driver.session() as session:
            session.run(
                """
                MERGE (i:Entity {name: 'India'})
                ON CREATE SET i.type = 'Country', i.first_seen = toString(date())
                SET i.last_updated = toString(date())
                """
            )
        print("india_node_ready")
    finally:
        driver.close()


if __name__ == "__main__":
    main()



