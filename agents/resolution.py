import logging
import re
from collections import defaultdict

import numpy as np
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from graphs.schemas import ArticleExtraction, ExtractedEntity, ResolutionBatch
from graphs.prompts import RESOLUTION_PROMPT
from models.database import SessionLocal
from models.entity_alias import EntityAlias

logger = logging.getLogger(__name__)


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower().strip()
    name = re.sub(r"[''`]", "'", name)
    name = re.sub(r"\s+", " ", name)
    return name


class ResolutionAgent:
    """3-tier entity resolution: deterministic → embeddings → LLM.

    The merge table is type-gated:
      (alias, context_type) → canonical
      context_type=None means "always apply" (type-independent)
      context_type="Country" means "only if entity was tagged Country"
    """

    def __init__(self, model: str = "openai/gpt-oss-20b", embedding_model: str = "all-MiniLM-L6-v2"):
        self.llm = ChatOllama(model="gpt-oss:120b-cloud").with_structured_output(ResolutionBatch)
        self.embedder = SentenceTransformer(embedding_model)
        self._load_merge_table()

    def _load_merge_table(self):
        """Load persistent merge table from Postgres.

        Two-level dict:
          typed_table[(norm_alias, context_type)] → canonical
          untyped_table[norm_alias] → canonical   (context_type IS NULL)
        """
        self.typed_table: dict[tuple[str, str], str] = {}
        self.untyped_table: dict[str, str] = {}

        db = SessionLocal()
        try:
            rows = db.execute(
                select(EntityAlias.alias, EntityAlias.canonical, EntityAlias.context_type)
            ).all()
            for alias, canonical, ctx in rows:
                norm = _normalize(alias)
                if ctx is None:
                    self.untyped_table[norm] = canonical
                else:
                    self.typed_table[(norm, ctx)] = canonical
            logger.info(
                f"Loaded {len(self.untyped_table)} untyped + "
                f"{len(self.typed_table)} typed aliases from DB"
            )
        finally:
            db.close()

    def _save_alias(
        self,
        alias: str,
        canonical: str,
        entity_type: str,
        tier: str,
        confidence: float = 1.0,
        context_type: str | None = None,
    ):
        """Persist a merge decision to the DB and in-memory tables."""
        norm = _normalize(alias)

        # Check if already known
        if context_type is None:
            if norm in self.untyped_table:
                return
            self.untyped_table[norm] = canonical
        else:
            key = (norm, context_type)
            if key in self.typed_table:
                return
            self.typed_table[key] = canonical

        db = SessionLocal()
        try:
            db.add(EntityAlias(
                alias=norm,
                canonical=canonical,
                entity_type=entity_type,
                context_type=context_type,
                confidence=confidence,
                resolved_by=tier,
            ))
            db.commit()
        except Exception as e:
            db.rollback()
            logger.debug(f"Alias save skipped (likely duplicate): {e}")
        finally:
            db.close()

    def canonicalize(self, name: str, entity_type: str | None = None) -> str:
        """Look up canonical name from merge table.

        Lookup order:
          1. (name, entity_type) — exact type match
          2. (name, NULL)        — type-independent alias
          3. No match            → keep original
        """
        norm = _normalize(name)

        # 1. Typed lookup
        if entity_type:
            typed = self.typed_table.get((norm, entity_type))
            if typed:
                return typed

        # 2. Untyped (always-apply) lookup
        untyped = self.untyped_table.get(norm)
        if untyped:
            return untyped

        return name

    # --- Tier 1: Deterministic ---

    def _tier1_resolve(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        """Rule-based normalization. Mutates entity names in place."""
        resolved = []
        for entity in entities:
            # Check merge table with type context
            canonical = self.canonicalize(entity.name, entity.type)
            if canonical != entity.name:
                logger.debug(f"Tier1: '{entity.name}' [{entity.type}] → '{canonical}'")
                entity.name = canonical

            # Register LLM-provided aliases (type-independent — they came from same context)
            for alias in entity.aliases:
                norm = _normalize(alias)
                if norm != _normalize(entity.name):
                    self._save_alias(alias, entity.name, entity.type, "tier1")

            resolved.append(entity)

        return resolved

    # --- Tier 2: Embedding Similarity ---

    def _tier2_resolve(self, entities: list[ExtractedEntity]) -> tuple[list[ExtractedEntity], list[tuple[str, str, str, float]]]:
        """Embedding-based similarity with type blocking.
        Returns (entities, candidate_pairs_for_tier3)."""

        by_type: dict[str, list[ExtractedEntity]] = defaultdict(list)
        for e in entities:
            by_type[e.type].append(e)

        candidates: list[tuple[str, str, str, float]] = []

        for etype, group in by_type.items():
            if len(group) < 2:
                continue

            unique_names = list({e.name for e in group})
            if len(unique_names) < 2:
                continue

            embeddings = self.embedder.encode(unique_names, normalize_embeddings=True)
            sim_matrix = embeddings @ embeddings.T

            for i in range(len(unique_names)):
                for j in range(i + 1, len(unique_names)):
                    sim = float(sim_matrix[i][j])
                    if sim > 0.95:
                        # Canonical = shorter name (e.g. "India" wins over "Republic of India")
                        # Exception: if one name is a known acronym keep the longer full form
                        a, b = unique_names[i], unique_names[j]
                        canonical = a if len(a) <= len(b) else b
                        alias    = b if canonical == a else a
                        self._save_alias(alias, canonical, etype, "tier2", confidence=sim)
                        for e in group:
                            if e.name == alias:
                                e.name = canonical
                        logger.debug(f"Tier2 auto-merge: '{alias}' → '{canonical}' (sim={sim:.3f})")
                    elif sim > 0.80:
                        candidates.append((unique_names[i], unique_names[j], etype, sim))

        return entities, candidates

    # --- Tier 3: LLM Disambiguation ---

    def _tier3_resolve(self, candidates: list[tuple[str, str, str, float]]) -> None:
        """LLM-based disambiguation for ambiguous pairs."""
        if not candidates:
            return

        batch_size = 20
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start:batch_start + batch_size]

            pairs_text = "\n".join(
                f'{i+1}. "{name1}" ({etype}) vs "{name2}" ({etype}) [similarity: {sim:.2f}]'
                for i, (name1, name2, etype, sim) in enumerate(batch)
            )

            try:
                result = self.llm.invoke(
                    RESOLUTION_PROMPT.format(pairs=pairs_text)
                )
                for merge in result.merges:
                    pair_type = "Unknown"
                    for name1, name2, etype, _ in batch:
                        if (merge.canonical in (name1, name2)) and (merge.merge_into in (name1, name2)):
                            pair_type = etype
                            break
                    self._save_alias(merge.merge_into, merge.canonical, pair_type, "tier3", merge.confidence)
                    logger.info(f"Tier3: '{merge.merge_into}' → '{merge.canonical}'")
            except Exception as e:
                logger.error(f"Tier3 LLM resolution failed: {e}")

    # --- Orchestrator ---

    def resolve(
        self, extractions: list[tuple[object, ArticleExtraction]]
    ) -> list[tuple[object, ArticleExtraction]]:
        """Run full 3-tier resolution on a batch of extractions.
        Mutates entity/relation names in place and returns the same list."""

        all_entities: list[ExtractedEntity] = []
        for _, extraction in extractions:
            all_entities.extend(extraction.entities)

        total_before = len({e.name for e in all_entities})
        logger.info(f"=== Resolution: {total_before} unique entity names ===")

        # Tier 1: Deterministic
        logger.info("--- Tier 1: Deterministic normalization ---")
        all_entities = self._tier1_resolve(all_entities)
        after_t1 = len({e.name for e in all_entities})
        logger.info(f"  {total_before} → {after_t1} unique names")

        # Tier 2: Embeddings
        logger.info("--- Tier 2: Embedding similarity ---")
        all_entities, candidates = self._tier2_resolve(all_entities)
        after_t2 = len({e.name for e in all_entities})
        logger.info(f"  {after_t1} → {after_t2} unique names, {len(candidates)} candidates for Tier 3")

        # Tier 3: LLM (only if there are ambiguous pairs)
        if candidates:
            logger.info(f"--- Tier 3: LLM disambiguation ({len(candidates)} pairs) ---")
            self._tier3_resolve(candidates)

        # Build entity type map for type-aware canonicalization in relations/events
        entity_type_map: dict[str, str] = {}
        for e in all_entities:
            entity_type_map[e.name] = e.type

        # Apply final merge table to all names
        for _, extraction in extractions:
            for entity in extraction.entities:
                entity.name = self.canonicalize(entity.name, entity.type)
            for rel in extraction.relations:
                rel.source = self.canonicalize(rel.source, entity_type_map.get(rel.source))
                rel.target = self.canonicalize(rel.target, entity_type_map.get(rel.target))
            for event in extraction.events:
                event.entities = [
                    self.canonicalize(e, entity_type_map.get(e)) for e in event.entities
                ]

        final = len({e.name for _, ext in extractions for e in ext.entities})
        logger.info(f"=== Resolution complete: {total_before} → {final} unique entities ===")

        return extractions
