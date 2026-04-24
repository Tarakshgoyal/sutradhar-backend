"""Hybrid extraction pipeline: GLiNER2 (NER + RE) → LLM (events + enrichment).

GLiNER2 handles the mechanical extraction — entities and relations grounded in
text spans, ~50ms/article on CPU.  The LLM only handles what requires reasoning:
causal flag classification, temporal marker attachment, and event extraction.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from gliner2 import GLiNER2
from langchain_ollama import ChatOllama


LLM_TIMEOUT_SECONDS = 300  # allow up to 5 min for local Ollama models

from scrapers.news_rss import Article
from graphs.schemas import (
    ArticleExtraction,
    CanonicalizationResult,
    ExtractedEntity,
    ExtractedRelation,
    LLMEnrichment,
)
from graphs.prompts import CANONICALIZATION_PROMPT, ENRICHMENT_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GLiNER2 label definitions
# ---------------------------------------------------------------------------

ENTITY_LABELS: dict[str, str] = {
    "person": "Individual people, political leaders, government officials, executives, military commanders",
    "organization": "Companies, government agencies, institutions, NGOs, military branches, terrorist groups",
    "country": "Sovereign nations and states",
    "location": "Cities, regions, geographic features, territories, military bases, waterways",
    "policy": "Laws, regulations, agreements, treaties, sanctions, trade deals, executive orders",
    "technology": "Technologies, weapons systems, platforms, software, chips, AI models",
    "economic_indicator": "GDP figures, inflation rates, oil prices, stock indices, trade volumes, currency values",
    "military_asset": "Weapons, warships, aircraft, missile systems, military bases, defense systems",
    "resource": "Natural resources, commodities, energy sources, rare earth minerals, oil, gas",
}

RELATION_LABELS: dict[str, str] = {
    "sanctions": "Imposes economic or political sanctions on",
    "allied_with": "Has diplomatic or military alliance with",
    "opposes": "Politically, militarily, or ideologically opposes",
    "trades_with": "Has commercial or trade relationship with",
    "supplies_to": "Provides goods, weapons, technology, or resources to",
    "invaded": "Militarily invaded or occupied",
    "leads": "Leads, heads, governs, or commands",
    "founded": "Created, established, or launched",
    "acquired": "Purchased, merged with, or took over",
    "located_in": "Is geographically located in or headquartered in",
    "manufactures": "Produces, manufactures, or builds",
    "funds": "Financially supports, funds, or invests in",
    "threatens": "Threatens militarily, economically, or politically",
    "negotiates_with": "Engages in diplomatic negotiations with",
    "member_of": "Is a member or part of",
    "disrupts": "Disrupts operations, supply chains, or activities of",
    "signed_agreement_with": "Signed a formal agreement or treaty with",
    "deployed_to": "Deployed military assets or personnel to",
    "develops": "Researches, develops, or advances",
    "exports_to": "Exports goods or resources to",
    "imports_from": "Imports goods or resources from",
    "cooperates_with": "Cooperates or collaborates with",
    "competes_with": "Competes commercially or strategically with",
    "attacks": "Launched military attack, cyber attack, or strike against",
    "blocks": "Blocks, restricts, or bans",
    "supports": "Politically or materially supports",
}

# Auto-derive canonical type names from ENTITY_LABELS keys
# e.g. "person" → "Person", "economic_indicator" → "Economic_Indicator"
_ENTITY_TYPE_MAP: dict[str, str] = {
    k: "_".join(part.capitalize() for part in k.split("_"))
    for k in ENTITY_LABELS
}


class ExtractionAgent:
    """Hybrid GLiNER2 + LLM extraction pipeline.

    • GLiNER2  → fast entity + relation extraction (text-span grounded, no hallucination)
    • LLM     → event extraction + causal/temporal enrichment (reasoning)
    """

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        gliner_model: str = "fastino/gliner2-base-v1",
    ):
        logger.info("Loading GLiNER2 model …")
        self.gliner = GLiNER2.from_pretrained(gliner_model)
        self._schema = (
            self.gliner.create_schema()
            .entities(ENTITY_LABELS)
            .relations(RELATION_LABELS)
        )

        logger.info("Loading LLM for enrichment …")
        self.llm = ChatOllama(
            model="gpt-oss:120b-cloud",
            num_predict=2048,
        ).with_structured_output(LLMEnrichment, method="json_schema")
        self.canon_llm = ChatOllama(
            model="gpt-oss:120b-cloud",
            num_predict=2048,
        ).with_structured_output(CanonicalizationResult, method="json_schema")
        self._executor = ThreadPoolExecutor(max_workers=1)

    # ------------------------------------------------------------------
    # GLiNER2: entities + relations
    # ------------------------------------------------------------------

    def _gliner_extract(
        self, text: str
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        """Run GLiNER2 combined schema, return parsed entities + relations."""

        raw = self.gliner.extract(text, self._schema, include_confidence=True)

        # --- entities ---
        entities: list[ExtractedEntity] = []
        seen: set[str] = set()

        for etype, items in raw.get("entities", {}).items():
            canonical_type = _ENTITY_TYPE_MAP.get(etype, etype.title())
            for item in items:
                if isinstance(item, dict):
                    name = item["text"]
                    conf = item.get("confidence", 1.0)
                else:
                    name = str(item)
                    conf = 1.0
                if name not in seen and conf >= 0.3:
                    seen.add(name)
                    entities.append(
                        ExtractedEntity(name=name, type=canonical_type, aliases=[], confidence=conf)
                    )

        # --- relations ---
        relations: list[ExtractedRelation] = []
        for rel_type, pairs in raw.get("relation_extraction", {}).items():
            for pair in pairs:
                source, target, conf = self._parse_relation_pair(pair)
                if source and target:
                    relations.append(
                        ExtractedRelation(
                            source=source,
                            target=target,
                            relation=rel_type,
                            confidence=conf,
                            temporal=None,
                            causal=False,
                        )
                    )

        return entities, relations

    @staticmethod
    def _parse_relation_pair(pair) -> tuple[str, str, float]:
        """Handle the various output formats GLiNER2 may return."""
        if isinstance(pair, dict):
            head = pair.get("head", {})
            tail = pair.get("tail", {})
            src = head.get("text", "") if isinstance(head, dict) else str(head)
            tgt = tail.get("text", "") if isinstance(tail, dict) else str(tail)
            conf = min(
                head.get("confidence", 1.0) if isinstance(head, dict) else 1.0,
                tail.get("confidence", 1.0) if isinstance(tail, dict) else 1.0,
            )
            return src, tgt, conf
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            return str(pair[0]), str(pair[1]), 1.0
        return "", "", 0.0

    # ------------------------------------------------------------------
    # LLM: entity canonicalization (lightweight — just name normalization)
    # ------------------------------------------------------------------

    def _canonicalize_entities(
        self,
        article: Article,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        """Use LLM to convert raw GLiNER2 spans into canonical names + aliases.

        This is cheap: the prompt is just a list of short names + article title
        for context. Typically <500 tokens round-trip.
        """
        entity_lines = "\n".join(
            f"- {e.name} (type: {e.type}, confidence: {e.confidence:.2f})" for e in entities
        )
        prompt = CANONICALIZATION_PROMPT.format(
            title=article.title,
            source=article.source,
            entities=entity_lines,
        )

        try:
            result: CanonicalizationResult | None = None
            last_err = None
            for attempt in range(3):
                try:
                    future = self._executor.submit(self.canon_llm.invoke, prompt)
                    result = future.result(timeout=LLM_TIMEOUT_SECONDS)
                    break
                except TimeoutError:
                    logger.warning("Canonicalization timed out — using raw spans")
                    return entities, relations
                except Exception as e:
                    last_err = e
                    if attempt < 2:
                        wait = 1.5 ** attempt
                        logger.debug(f"Canonicalization attempt {attempt+1} failed, retrying in {wait:.1f}s: {e}")
                        time.sleep(wait)
            if result is None:
                logger.error(f"Canonicalization failed after 3 attempts: {last_err} — using raw spans")
                return entities, relations
        except Exception as e:
            logger.error(f"Canonicalization failed: {e} — using raw spans")
            return entities, relations

        # Build mapping: original_name → (canonical, aliases)
        canon_map: dict[str, str] = {}
        for ce in result.entities:
            canon_map[ce.original] = ce.canonical

        # Update entities
        for entity in entities:
            canonical = canon_map.get(entity.name)
            if canonical and canonical != entity.name:
                # Find aliases from the LLM result
                for ce in result.entities:
                    if ce.original == entity.name:
                        # Merge: original span + LLM-provided aliases (dedup)
                        all_aliases = set(ce.aliases)
                        all_aliases.add(entity.name)  # keep raw span as alias
                        all_aliases.discard(canonical)  # don't alias to self
                        entity.aliases = list(all_aliases)
                        # Apply type correction if LLM flagged it
                        if ce.corrected_type:
                            logger.debug(f"Type fix: '{entity.name}' {entity.type} → {ce.corrected_type}")
                            entity.type = ce.corrected_type
                        break
                logger.debug(f"Canon: '{entity.name}' → '{canonical}' aliases={entity.aliases}")
                entity.name = canonical
            elif canonical is None:
                # LLM didn't return this entity — keep as-is
                pass
            else:
                # canonical == entity.name — still grab aliases + type fix
                for ce in result.entities:
                    if ce.original == entity.name:
                        if ce.aliases:
                            entity.aliases = list(set(ce.aliases) - {entity.name})
                        if ce.corrected_type:
                            logger.debug(f"Type fix: '{entity.name}' {entity.type} → {ce.corrected_type}")
                            entity.type = ce.corrected_type
                        break

        # Update relation source/target to match new canonical names
        for rel in relations:
            if rel.source in canon_map:
                rel.source = canon_map[rel.source]
            if rel.target in canon_map:
                rel.target = canon_map[rel.target]

        return entities, relations

    # ------------------------------------------------------------------
    # LLM: events + causal / temporal enrichment
    # ------------------------------------------------------------------

    def _llm_enrich(
        self,
        article: Article,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> LLMEnrichment | None:
        entity_summary = ", ".join(f"{e.name} ({e.type})" for e in entities)
        relation_summary = "\n".join(
            f"  {i + 1}. {r.source} --[{r.relation}]--> {r.target}"
            for i, r in enumerate(relations)
        )

        prompt = ENRICHMENT_PROMPT.format(
            title=article.title,
            source=article.source,
            pub_date=article.pub_date or "unknown",
            text=article.full_text[:3000],
            entities=entity_summary,
            relations=relation_summary or "(none extracted)",
        )
        last_err = None
        for attempt in range(3):
            try:
                future = self._executor.submit(self.llm.invoke, prompt)
                return future.result(timeout=LLM_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.warning(f"LLM enrichment timed out ({LLM_TIMEOUT_SECONDS}s) for '{article.title[:50]}' — skipping enrichment")
                return None
            except Exception as e:
                last_err = e
                if attempt < 2:
                    wait = 1.5 ** attempt
                    logger.debug(f"LLM enrichment attempt {attempt+1} failed, retrying in {wait:.1f}s: {e}")
                    time.sleep(wait)
        logger.error(f"LLM enrichment failed after 3 attempts for '{article.title}': {last_err}")
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, article: Article) -> ArticleExtraction | None:
        """Full hybrid extraction: GLiNER2 → LLM enrichment → merged."""
        if not article.full_text:
            return None

        text = article.full_text[:4000]

        # Step 1 — GLiNER2: entities + relations (fast, grounded)
        try:
            entities, relations = self._gliner_extract(text)
        except Exception as e:
            logger.error(f"GLiNER2 failed for '{article.title}': {e}")
            return None

        if not entities:
            logger.warning(f"GLiNER2 found 0 entities in '{article.title[:50]}'")
            return None

        # Step 2 — LLM: canonicalize entity names + generate aliases (lightweight)
        entities, relations = self._canonicalize_entities(article, entities, relations)

        # Step 3 — LLM: events + enrichment (reasoning)
        enrichment = self._llm_enrich(article, entities, relations)

        events = []
        if enrichment:
            # Apply causal flags + temporal markers back to relations
            enrich_lookup = {
                (er.source, er.target, er.relation): er
                for er in enrichment.relation_enrichments
            }
            for rel in relations:
                key = (rel.source, rel.target, rel.relation)
                if key in enrich_lookup:
                    rel.causal = enrich_lookup[key].causal
                    rel.temporal = enrich_lookup[key].temporal
            events = enrichment.events

        return ArticleExtraction(
            entities=entities,
            relations=relations,
            events=events,
        )

    def extract_batch(
        self, articles: list[Article], max_workers: int = 3
    ) -> list[tuple[Article, ArticleExtraction]]:
        """Extract from multiple articles. Returns (article, extraction) pairs."""
        results: list[tuple[Article, ArticleExtraction]] = []

        for i, article in enumerate(articles):
            logger.info(f"[{i + 1}/{len(articles)}] Extracting: {article.title[:60]}")
            extraction = self.extract(article)
            if extraction:
                logger.info(
                    f"  → {len(extraction.entities)} entities, "
                    f"{len(extraction.relations)} relations, "
                    f"{len(extraction.events)} events"
                )
                results.append((article, extraction))
            else:
                logger.warning(f"  → Skipped (no extraction)")

        logger.info(f"Extracted {len(results)}/{len(articles)} articles")
        return results
