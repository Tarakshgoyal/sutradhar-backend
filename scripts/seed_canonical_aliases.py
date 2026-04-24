"""Seed canonical entity aliases into the EntityAlias table.

Fixes entity fragmentation where the same real-world entity appears as
multiple nodes (e.g. "India" vs "Republic of India" vs "Bharat").

Also type-gates political coalitions (INDIA Alliance = Organization, not Country)
so they never get merged with the country node.

Usage:
    python -m scripts.seed_canonical_aliases
"""
import sys
sys.path.insert(0, ".")

from models.database import SessionLocal
from models.entity_alias import EntityAlias

# Format: (alias, canonical, entity_type, context_type)
# context_type=None  → always apply regardless of tagged type
# context_type="Country" → only apply when entity is tagged Country
ALIASES = [
    # ── India ────────────────────────────────────────────────────────────────
    ("republic of india",             "India",          "Country",      "Country"),
    ("bharat",                        "India",          "Country",      "Country"),
    ("indian",                        "India",          "Country",      "Country"),
    ("govt of india",                 "India",          "Country",      "Country"),
    ("government of india",           "India",          "Country",      "Country"),
    ("new delhi",                     "New Delhi",      "Location",     "Location"),
    # INDIA Alliance — all forms collapse to India (country-level entity)
    ("india alliance",                        "India", "Country", None),
    ("india bloc",                            "India", "Country", None),
    ("i.n.d.i.a.",                            "India", "Country", None),
    ("i.n.d.i.a",                             "India", "Country", None),
    ("indian national developmental inclusive alliance", "India", "Country", None),
    # NDA
    ("nda",                           "National Democratic Alliance",   "Organization", "Organization"),
    ("national democratic alliance",  "National Democratic Alliance",   "Organization", "Organization"),

    # ── United States ────────────────────────────────────────────────────────
    ("us",                            "United States",  "Country",      "Country"),
    ("usa",                           "United States",  "Country",      "Country"),
    ("america",                       "United States",  "Country",      "Country"),
    ("u.s.",                          "United States",  "Country",      "Country"),
    ("u.s.a.",                        "United States",  "Country",      "Country"),
    ("the united states",             "United States",  "Country",      "Country"),
    ("united states of america",      "United States",  "Country",      "Country"),
    ("washington",                    "United States",  "Country",      "Country"),
    ("washington dc",                 "Washington D.C.", "Location",    "Location"),

    # ── United Kingdom ───────────────────────────────────────────────────────
    ("uk",                            "United Kingdom", "Country",      "Country"),
    ("u.k.",                          "United Kingdom", "Country",      "Country"),
    ("britain",                       "United Kingdom", "Country",      "Country"),
    ("great britain",                 "United Kingdom", "Country",      "Country"),
    ("england",                       "United Kingdom", "Country",      "Country"),
    ("london",                        "London",         "Location",     "Location"),

    # ── Russia ───────────────────────────────────────────────────────────────
    ("russia",                        "Russia",         "Country",      "Country"),
    ("russian federation",            "Russia",         "Country",      "Country"),
    ("ussr",                          "Russia",         "Country",      "Country"),
    ("moscow",                        "Russia",         "Country",      "Country"),
    ("the kremlin",                   "Russia",         "Country",      "Country"),
    ("kremlin",                       "Russia",         "Country",      "Country"),

    # ── China ────────────────────────────────────────────────────────────────
    ("china",                         "China",          "Country",      "Country"),
    ("prc",                           "China",          "Country",      "Country"),
    ("people's republic of china",    "China",          "Country",      "Country"),
    ("peoples republic of china",     "China",          "Country",      "Country"),
    ("beijing",                       "China",          "Country",      "Country"),
    ("ccp",                           "Chinese Communist Party", "Organization", "Organization"),

    # ── Pakistan ─────────────────────────────────────────────────────────────
    ("pakistan",                      "Pakistan",       "Country",      "Country"),
    ("islamic republic of pakistan",  "Pakistan",       "Country",      "Country"),
    ("islamabad",                     "Pakistan",       "Country",      "Country"),

    # ── Iran ─────────────────────────────────────────────────────────────────
    ("iran",                          "Iran",           "Country",      "Country"),
    ("islamic republic of iran",      "Iran",           "Country",      "Country"),
    ("tehran",                        "Iran",           "Country",      "Country"),
    ("persia",                        "Iran",           "Country",      "Country"),

    # ── European Union ────────────────────────────────────────────────────────
    ("eu",                            "European Union", "Organization", "Organization"),
    ("e.u.",                          "European Union", "Organization", "Organization"),
    ("europe",                        "European Union", "Organization", "Organization"),
    ("brussels",                      "European Union", "Organization", "Organization"),

    # ── Israel ───────────────────────────────────────────────────────────────
    ("israel",                        "Israel",         "Country",      "Country"),
    ("state of israel",               "Israel",         "Country",      "Country"),
    ("idf",                           "Israel Defense Forces", "Organization", "Organization"),
    ("israel defense forces",         "Israel Defense Forces", "Organization", "Organization"),

    # ── Bangladesh ───────────────────────────────────────────────────────────
    ("bangladesh",                    "Bangladesh",     "Country",      "Country"),
    ("peoples republic of bangladesh","Bangladesh",     "Country",      "Country"),
    ("people's republic of bangladesh","Bangladesh",    "Country",      "Country"),
    ("dhaka",                         "Bangladesh",     "Country",      "Country"),

    # ── Bulgaria ─────────────────────────────────────────────────────────────
    ("bulgaria",                      "Bulgaria",       "Country",      "Country"),
    ("republic of bulgaria",          "Bulgaria",       "Country",      "Country"),
    ("sofia",                         "Bulgaria",       "Country",      "Country"),

    # ── Iraq ─────────────────────────────────────────────────────────────────
    ("iraq",                          "Iraq",           "Country",      "Country"),
    ("republic of iraq",              "Iraq",           "Country",      "Country"),
    ("baghdad",                       "Iraq",           "Country",      "Country"),

    # ── Egypt ────────────────────────────────────────────────────────────────
    ("egypt",                         "Egypt",          "Country",      "Country"),
    ("arab republic of egypt",        "Egypt",          "Country",      "Country"),
    ("cairo",                         "Egypt",          "Country",      "Country"),

    # ── Syria ────────────────────────────────────────────────────────────────
    ("syria",                         "Syria",          "Country",      "Country"),
    ("syrian arab republic",          "Syria",          "Country",      "Country"),
    ("damascus",                      "Syria",          "Country",      "Country"),

    # ── Cambodia ─────────────────────────────────────────────────────────────
    ("cambodia",                      "Cambodia",       "Country",      "Country"),
    ("kingdom of cambodia",           "Cambodia",       "Country",      "Country"),

    # ── Thailand ─────────────────────────────────────────────────────────────
    ("thailand",                      "Thailand",       "Country",      "Country"),
    ("kingdom of thailand",           "Thailand",       "Country",      "Country"),
    ("bangkok",                       "Thailand",       "Country",      "Country"),

    # ── Sri Lanka ────────────────────────────────────────────────────────────
    ("sri lanka",                     "Sri Lanka",      "Country",      "Country"),
    ("democratic socialist republic of sri lanka", "Sri Lanka", "Country", "Country"),
    ("colombo",                       "Sri Lanka",      "Country",      "Country"),

    # ── Nepal ────────────────────────────────────────────────────────────────
    ("nepal",                         "Nepal",          "Country",      "Country"),
    ("federal democratic republic of nepal", "Nepal",  "Country",      "Country"),
    ("kathmandu",                     "Nepal",          "Country",      "Country"),

    # ── Myanmar ──────────────────────────────────────────────────────────────
    ("myanmar",                       "Myanmar",        "Country",      "Country"),
    ("burma",                         "Myanmar",        "Country",      "Country"),

    # ── Saudi Arabia ─────────────────────────────────────────────────────────
    ("saudi arabia",                  "Saudi Arabia",   "Country",      "Country"),
    ("ksa",                           "Saudi Arabia",   "Country",      "Country"),
    ("riyadh",                        "Saudi Arabia",   "Country",      "Country"),

    # ── Turkey ───────────────────────────────────────────────────────────────
    ("turkey",                        "Turkey",         "Country",      "Country"),
    ("türkiye",                       "Turkey",         "Country",      "Country"),
    ("turkiye",                       "Turkey",         "Country",      "Country"),
    ("ankara",                        "Turkey",         "Country",      "Country"),

    # ── North Korea ──────────────────────────────────────────────────────────
    ("north korea",                   "North Korea",    "Country",      "Country"),
    ("dprk",                          "North Korea",    "Country",      "Country"),
    ("democratic peoples republic of korea", "North Korea", "Country", "Country"),
    ("democratic people's republic of korea", "North Korea", "Country", "Country"),

    # ── South Korea ──────────────────────────────────────────────────────────
    ("south korea",                   "South Korea",    "Country",      "Country"),
    ("rok",                           "South Korea",    "Country",      "Country"),
    ("republic of korea",             "South Korea",    "Country",      "Country"),
    ("seoul",                         "South Korea",    "Country",      "Country"),

    # ── Japan ────────────────────────────────────────────────────────────────
    ("japan",                         "Japan",          "Country",      "Country"),
    ("tokyo",                         "Japan",          "Country",      "Country"),

    # ── France ───────────────────────────────────────────────────────────────
    ("france",                        "France",         "Country",      "Country"),
    ("french republic",               "France",         "Country",      "Country"),
    ("paris",                         "France",         "Country",      "Country"),

    # ── Germany ──────────────────────────────────────────────────────────────
    ("germany",                       "Germany",        "Country",      "Country"),
    ("federal republic of germany",   "Germany",        "Country",      "Country"),
    ("berlin",                        "Germany",        "Country",      "Country"),

    # ── International orgs ───────────────────────────────────────────────────
    ("un",                            "United Nations", "Organization", "Organization"),
    ("u.n.",                          "United Nations", "Organization", "Organization"),
    ("nato",                          "NATO",           "Organization", "Organization"),
    ("north atlantic treaty organization", "NATO",      "Organization", "Organization"),
    ("imf",                           "International Monetary Fund", "Organization", "Organization"),
    ("world bank",                    "World Bank",     "Organization", "Organization"),
    ("who",                           "World Health Organization", "Organization", "Organization"),
    ("wto",                           "World Trade Organization",  "Organization", "Organization"),
    ("scо",                           "Shanghai Cooperation Organisation", "Organization", "Organization"),
    ("sco",                           "Shanghai Cooperation Organisation", "Organization", "Organization"),
    ("brics",                         "BRICS",          "Organization", "Organization"),
    ("asean",                         "ASEAN",          "Organization", "Organization"),
    ("g20",                           "G20",            "Organization", "Organization"),
    ("g7",                            "G7",             "Organization", "Organization"),
    ("quad",                          "Quad",           "Organization", "Organization"),

    # ── Indian institutions ───────────────────────────────────────────────────
    ("bjp",                           "Bharatiya Janata Party",     "Organization", "Organization"),
    ("bharatiya janata party",        "Bharatiya Janata Party",     "Organization", "Organization"),
    ("congress",                      "Indian National Congress",   "Organization", "Organization"),
    ("inc",                           "Indian National Congress",   "Organization", "Organization"),
    ("indian national congress",      "Indian National Congress",   "Organization", "Organization"),
    ("rbi",                           "Reserve Bank of India",      "Organization", "Organization"),
    ("reserve bank of india",         "Reserve Bank of India",      "Organization", "Organization"),
    ("sebi",                          "Securities and Exchange Board of India", "Organization", "Organization"),
    ("isro",                          "ISRO",           "Organization", "Organization"),
    ("drdo",                          "DRDO",           "Organization", "Organization"),
    ("indian army",                   "Indian Army",    "Organization", "Organization"),
    ("indian navy",                   "Indian Navy",    "Organization", "Organization"),
    ("indian air force",              "Indian Air Force","Organization","Organization"),
    ("iaf",                           "Indian Air Force","Organization","Organization"),
    ("pib",                           "Press Information Bureau",   "Organization", "Organization"),
    ("mea",                           "Ministry of External Affairs", "Organization", "Organization"),
]


def seed_canonical_aliases():
    db = SessionLocal()
    added = 0
    skipped = 0
    try:
        for alias_raw, canonical, entity_type, context_type in ALIASES:
            # Check if already exists
            existing = db.query(EntityAlias).filter_by(
                alias=alias_raw, context_type=context_type
            ).first()
            if existing:
                skipped += 1
                continue
            db.add(EntityAlias(
                alias=alias_raw,
                canonical=canonical,
                entity_type=entity_type,
                context_type=context_type,
                confidence=1.0,
                resolved_by="seed",
            ))
            added += 1

        db.commit()
        print(f"Seeded {added} aliases, skipped {skipped} already existing.")
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_canonical_aliases()
