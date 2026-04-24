"""Merge duplicate country/entity nodes in Neo4j.

Transfers all relationships from the duplicate node to the canonical node,
then deletes the duplicate.

Usage:
    python -m scripts.merge_duplicate_nodes
"""
import sys
sys.path.insert(0, ".")

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_AUTH

# (duplicate_name, canonical_name)
MERGES = [
    ("Indian National Developmental Inclusive Alliance", "India"),
    ("Democratic People'S Republic Of Korea",            "North Korea"),
    ("Democratic People's Republic Of Korea",            "North Korea"),
    ("People'S Republic Of Bangladesh",                  "Bangladesh"),
    ("People's Republic Of Bangladesh",                  "Bangladesh"),
    ("Republic Of Bulgaria",                             "Bulgaria"),
    ("Republic Of Iraq",                                 "Iraq"),
    ("Arab Republic Of Egypt",                           "Egypt"),
    ("Syrian Arab Republic",                             "Syria"),
    ("Kingdom Of Cambodia",                              "Cambodia"),
    ("Kingdom Of Thailand",                              "Thailand"),
    ("Russian Federation",                               "Russia"),
    ("United States Of America",                         "United States"),
    ("Peoples Republic Of China",                        "China"),
    ("People'S Republic Of China",                       "China"),
    ("People's Republic Of China",                       "China"),
    ("Islamic Republic Of Iran",                         "Iran"),
    ("Islamic Republic Of Pakistan",                     "Pakistan"),
    ("Republic Of Korea",                                "South Korea"),
    ("Democratic Socialist Republic Of Sri Lanka",       "Sri Lanka"),
    ("Federal Republic Of Germany",                      "Germany"),
    ("French Republic",                                  "France"),
    ("State Of Israel",                                  "Israel"),
    ("Hashemite Kingdom Of Jordan",                      "Jordan"),
    ("Republic Of India",                                "India"),
    ("Bharat",                                           "India"),
]

TRANSFER_RELS = """
MATCH (dup:Entity {name: $dup})
MATCH (can:Entity {name: $can})
WHERE dup <> can
WITH dup, can
// outgoing rels from dup → merge onto can
OPTIONAL MATCH (dup)-[r:RELATES_TO]->(other)
WHERE other <> can
FOREACH (_ IN CASE WHEN other IS NOT NULL THEN [1] ELSE [] END |
  MERGE (can)-[r2:RELATES_TO {type: r.type}]->(other)
  ON CREATE SET r2 = properties(r)
)
WITH dup, can
// incoming rels to dup → merge onto can
OPTIONAL MATCH (other2)-[r3:RELATES_TO]->(dup)
WHERE other2 <> can
FOREACH (_ IN CASE WHEN other2 IS NOT NULL THEN [1] ELSE [] END |
  MERGE (other2)-[r4:RELATES_TO {type: r3.type}]->(can)
  ON CREATE SET r4 = properties(r3)
)
WITH dup, can
// EVIDENCES from articles
OPTIONAL MATCH (a:Article)-[:EVIDENCES]->(dup)
FOREACH (_ IN CASE WHEN a IS NOT NULL THEN [1] ELSE [] END |
  MERGE (a)-[:EVIDENCES]->(can)
)
WITH dup
DETACH DELETE dup
"""

driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
merged = 0
with driver.session() as s:
    for dup, can in MERGES:
        # check if dup exists
        exists = s.run("MATCH (n:Entity {name:$n}) RETURN count(n) as c", n=dup).single()["c"]
        if exists == 0:
            continue
        # ensure canonical exists (create if needed)
        s.run("MERGE (n:Entity {name:$n}) ON CREATE SET n.type='Country'", n=can)
        s.run(TRANSFER_RELS, dup=dup, can=can)
        print(f"  Merged '{dup}' -> '{can}'")
        merged += 1

driver.close()
print(f"\nDone - merged {merged} duplicate nodes.")
