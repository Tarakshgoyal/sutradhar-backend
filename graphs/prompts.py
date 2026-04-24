ENRICHMENT_PROMPT = """You are an intelligence analyst. Entities and relationships have already been extracted from this article by an NER model. Your tasks:

1. ENRICH RELATIONS — for each relation listed below, determine:
   - causal: Is it a cause-effect relationship? (true/false)
   - temporal: What is the time marker? A date ("2024-03"), period ("Q1 2025"), status ("ongoing", "announced", "concluded"), or null if unknown.
   Copy source, target, and relation EXACTLY as given.

2. EXTRACT EVENTS — identify up to 5 named events with:
   - name: short descriptive name
   - date: when it occurred or "ongoing"
   - status: one of: ongoing / concluded / announced / rumored
   - entities: which of the pre-extracted entity names are involved (use exact names from the list)

Pre-extracted entities: {entities}

Pre-extracted relations:
{relations}

Article:
Title: {title}
Source: {source}
Published: {pub_date}

{text}

IMPORTANT: You MUST respond with ONLY valid JSON. No markdown, no bullet points, no prose.
Use this exact JSON structure:
{{
  "relation_enrichments": [
    {{"source": "...", "target": "...", "relation": "...", "causal": false, "temporal": null}}
  ],
  "events": [
    {{"name": "...", "date": "...", "status": "...", "entities": ["..."]}}
  ]
}}"""


RESOLUTION_PROMPT = """You are an entity resolution expert. Given these candidate pairs of entities that might refer to the same real-world entity, determine which ones should be merged.

Rules:
- Only merge if they DEFINITELY refer to the same real-world entity. If unsure, do NOT merge.
- NEVER merge a country with a political party, alliance, or coalition (e.g. "India" ≠ "INDIA Alliance").
- NEVER merge across entity types (Country ≠ Organization ≠ Person).
- When merging, the canonical name should be the SHORT, well-known form (e.g. "India" not "Republic of India", "United States" not "United States of America").
- Acronyms of organizations should merge INTO their full name (e.g. "BJP" → "Bharatiya Janata Party").

Respond with ONLY valid JSON:
{{"merges": [{{"merge_into": "...", "canonical": "...", "confidence": 0.95}}]}}
If nothing should be merged, respond: {{"merges": []}}

Candidate pairs:
{pairs}
"""


CANONICALIZATION_PROMPT = """You are a geopolitical knowledge expert. Given a list of raw entity names extracted from a news article, produce the canonical (full, standard) name, correct the entity type if wrong, and list any known aliases for each.

Rules:
- canonical: The most widely recognized full name.
- corrected_type: If the NER model assigned the WRONG type, provide the correct one. Use null if the type is already correct.
  For entities with confidence < 0.80, ALWAYS verify the type carefully — the NER model is uncertain.
  For entities with confidence >= 0.80, the type is usually correct — only change it if clearly wrong.
  IMPORTANT type distinctions:
  - Country: ONLY sovereign nations. NOT cities, regions, or bodies of water.
  - Location: Cities, regions, geographic features, bodies of water.
  - Organization: Institutions, agencies, legislative bodies.
  - Valid types: Person, Organization, Country, Location, Policy, Technology, Economic_Indicator, Military_Asset, Resource.
- aliases: Common short forms, acronyms, or alternate names. Include the original name if it differs from canonical. Use an empty list [] if there are no aliases.
- Do NOT invent entities. Only canonicalize what is given.
- Keep the same number of entities in your output as in the input. Match each "original" field EXACTLY to the input name.
- You MUST return valid JSON with all entities. Never return empty output.

Example input:
- Trump (type: Person, confidence: 0.95)
- the US (type: Country, confidence: 0.88)
- PLA (type: Organization, confidence: 0.72)
- Indian Ocean (type: Country, confidence: 0.55)
- United States Senate (type: Country, confidence: 0.61)

Example output (note: low-confidence types corrected):
{{"entities": [{{"original": "Trump", "canonical": "Donald Trump", "corrected_type": null, "aliases": ["Trump"]}}, {{"original": "the US", "canonical": "United States", "corrected_type": null, "aliases": ["the US", "USA", "US", "America"]}}, {{"original": "PLA", "canonical": "People's Liberation Army", "corrected_type": null, "aliases": ["PLA"]}}, {{"original": "Indian Ocean", "canonical": "Indian Ocean", "corrected_type": "Location", "aliases": []}}, {{"original": "United States Senate", "canonical": "United States Senate", "corrected_type": "Organization", "aliases": ["US Senate", "Senate"]}}]}}

Article context (for disambiguation):
Title: {title}
Source: {source}

Extracted entities:
{entities}
"""
