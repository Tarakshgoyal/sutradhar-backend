"""Entity type to domain classification service.

Provides dynamic entity-to-domain mapping with database-backed configuration
and LLM fallback for unknown entity types.
"""

import logging
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from models.entity_type_domain_mapping import EntityTypeDomainMapping

logger = logging.getLogger(__name__)


class EntityClassifier:
    """
    Entity type to domain classifier with fallback hierarchy:
    1. Database exact match
    2. LLM classification for unknown types
    3. Store new classification in database
    """

    def __init__(self, db_session: Session, llm_model: str = "llama-3.3-70b-versatile"):
        self.db = db_session
        self.llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.1, num_predict=256)
        self._cache: Dict[str, Dict] = {}

    def get_classification(self, entity_type: str) -> Dict[str, any]:
        """
        Get domain classification for an entity type.

        Args:
            entity_type: Type of entity (e.g., "Person", "Organization")

        Returns:
            Dict with keys:
                - primary_domain: str (main domain)
                - secondary_domains: List[str] (additional relevant domains)
                - confidence: float (0-1)

        Raises:
            Exception if classification fails
        """
        # Check cache first
        if entity_type in self._cache:
            logger.debug(f"Entity type '{entity_type}' found in cache")
            return self._cache[entity_type]

        # Database lookup
        result = self.db.execute(
            select(EntityTypeDomainMapping).where(
                EntityTypeDomainMapping.entity_type == entity_type
            )
        ).scalar_one_or_none()

        if result:
            classification = {
                "primary_domain": result.primary_domain,
                "secondary_domains": result.secondary_domains,
                "confidence": result.confidence,
            }
            self._cache[entity_type] = classification
            logger.info(f"Entity type '{entity_type}' found in database: {result.primary_domain}")
            return classification

        # LLM fallback for unknown entity types
        logger.warning(f"Entity type '{entity_type}' not in database, using LLM fallback")
        classification = self._classify_with_llm(entity_type)

        if classification:
            # Store in database for future use
            self._store_classification(entity_type, classification)
            self._cache[entity_type] = classification
            logger.info(f"Entity type '{entity_type}' classified via LLM: {classification['primary_domain']}")
            return classification

        # Ultimate fallback
        logger.error(f"Failed to classify entity type '{entity_type}', using default")
        return {
            "primary_domain": "other",
            "secondary_domains": [],
            "confidence": 0.3,
        }

    def _classify_with_llm(self, entity_type: str) -> Optional[Dict]:
        """Use LLM to classify unknown entity types."""
        try:
            system_prompt = """You are a domain classification expert for intelligence analysis.

Valid domains are:
- geopolitics: Political entities, leaders, countries, diplomatic relations
- defense: Military assets, armed forces, defense systems, security
- economics: Economic indicators, markets, financial institutions, trade
- technology: Technologies, innovations, tech companies, R&D
- climate: Environmental resources, climate factors, natural disasters
- society: Social movements, civil organizations, cultural entities
- other: Everything else

Given an entity type, classify it to the MOST relevant primary domain and optionally 1-2 secondary domains.
Respond with ONLY a JSON object in this exact format:
{"primary_domain": "domain_name", "secondary_domains": ["optional_domain1", "optional_domain2"], "confidence": 0.9}

Do not include any other text or explanation."""

            user_prompt = f"Classify this entity type: {entity_type}"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = self.llm.invoke(messages)
            content = response.content.strip()

            # Parse JSON response
            import json
            classification = json.loads(content)

            # Validate fields
            if "primary_domain" not in classification:
                logger.error(f"LLM response missing primary_domain: {content}")
                return None

            # Set defaults
            classification.setdefault("secondary_domains", [])
            classification.setdefault("confidence", 0.7)

            return classification

        except Exception as e:
            logger.error(f"LLM classification failed for '{entity_type}': {e}")
            return None

    def _store_classification(self, entity_type: str, classification: Dict) -> None:
        """Store newly discovered classification in database."""
        try:
            new_mapping = EntityTypeDomainMapping(
                entity_type=entity_type,
                primary_domain=classification["primary_domain"],
                secondary_domains=classification.get("secondary_domains", []),
                confidence=classification.get("confidence", 0.7),
            )
            self.db.add(new_mapping)
            self.db.commit()
            logger.info(f"Stored new classification for '{entity_type}'")
        except Exception as e:
            self.db.rollback()
            logger.warning(f"Failed to store classification for '{entity_type}': {e}")

    def get_primary_domain(self, entity_type: str) -> str:
        """
        Get just the primary domain for an entity type.
        Convenience method for simple lookups.
        """
        classification = self.get_classification(entity_type)
        return classification["primary_domain"]

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
        logger.debug("Entity classifier cache cleared")

    def get_all_mappings(self) -> List[Dict]:
        """Get all entity type mappings from database."""
        results = self.db.execute(select(EntityTypeDomainMapping)).scalars().all()
        return [
            {
                "entity_type": m.entity_type,
                "primary_domain": m.primary_domain,
                "secondary_domains": m.secondary_domains,
                "confidence": m.confidence,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            }
            for m in results
        ]
