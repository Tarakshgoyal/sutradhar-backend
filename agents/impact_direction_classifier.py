"""Impact direction classification service for scenario analysis.

Classifies whether an entity's involvement in a scenario represents:
- risk: Negative/threatening impact
- opportunity: Positive/beneficial impact
- neutral: Unclear or balanced impact
"""

import logging
from typing import Dict, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import json

logger = logging.getLogger(__name__)


# Rule-based fallback: (entity_type, domain, relation_type) -> impact_direction
# These are heuristic rules based on common patterns
IMPACT_RULES = {
    # Military/Defense patterns
    ("Military_Asset", "defense", "THREATENS"): "risk",
    ("Military_Asset", "defense", "DEPLOYS"): "risk",
    ("Military_Asset", "geopolitics", "TARGETS"): "risk",
    ("Country", "defense", "ARMS_RACE"): "risk",

    # Economic patterns
    ("Economic_Indicator", "economics", "DECLINES"): "risk",
    ("Economic_Indicator", "economics", "GROWS"): "opportunity",
    ("Organization", "economics", "INVESTS_IN"): "opportunity",
    ("Organization", "economics", "COLLAPSES"): "risk",

    # Technology patterns
    ("Technology", "technology", "INNOVATES"): "opportunity",
    ("Technology", "technology", "BREAKTHROUGH"): "opportunity",
    ("Technology", "defense", "DEVELOPS"): "neutral",  # Could be defensive or offensive

    # Climate patterns
    ("Resource", "climate", "DEPLETES"): "risk",
    ("Resource", "climate", "SCARCITY"): "risk",
    ("Location", "climate", "DISASTER"): "risk",

    # Geopolitical patterns
    ("Country", "geopolitics", "SANCTIONS"): "risk",
    ("Country", "geopolitics", "ALLIANCE"): "opportunity",
    ("Person", "geopolitics", "LEADS"): "neutral",
    ("Policy", "geopolitics", "REFORM"): "opportunity",
    ("Policy", "geopolitics", "RESTRICTION"): "risk",
}

# Default rules based on entity type alone
DEFAULT_ENTITY_RULES = {
    "Military_Asset": "risk",      # Military developments typically signal risk
    "Economic_Indicator": "neutral",  # Could go either way
    "Technology": "opportunity",    # Innovation is generally positive
    "Resource": "risk",            # Resource discussions often about scarcity
    "Policy": "neutral",           # Policies can be restrictive or beneficial
    "Country": "neutral",          # Countries are context-dependent
    "Person": "neutral",           # People are context-dependent
    "Organization": "neutral",     # Organizations vary widely
    "Location": "neutral",         # Locations are context-dependent
}


class ImpactDirectionClassifier:
    """
    Classify impact direction for entities in scenarios.

    Uses a three-tier fallback system:
    1. LLM classification with context (most accurate)
    2. Rule-based classification using entity type + domain + relation
    3. Default classification based on entity type alone
    """

    def __init__(self, llm_model: str = "llama-3.3-70b-versatile", enable_llm: bool = True):
        self.enable_llm = enable_llm
        if enable_llm:
            self.llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.2, num_predict=256)
        self._cache: Dict[str, str] = {}

    def classify(
        self,
        entity_name: str,
        entity_type: str,
        domain: Optional[str] = None,
        relation_type: Optional[str] = None,
        scenario_context: Optional[str] = None,
    ) -> str:
        """
        Classify the impact direction for an entity in a scenario.

        Args:
            entity_name: Name of the entity
            entity_type: Type of entity (Person, Organization, etc.)
            domain: Domain context (geopolitics, economics, etc.)
            relation_type: Type of relationship to scenario trigger
            scenario_context: Optional scenario description for context

        Returns:
            One of: "risk", "opportunity", "neutral"
        """
        # Generate cache key
        cache_key = f"{entity_name}:{entity_type}:{domain}:{relation_type}"
        if cache_key in self._cache:
            logger.debug(f"Impact direction for '{entity_name}' found in cache")
            return self._cache[cache_key]

        # Try LLM classification first (if enabled)
        if self.enable_llm:
            direction = self._classify_with_llm(
                entity_name, entity_type, domain, relation_type, scenario_context
            )
            if direction:
                self._cache[cache_key] = direction
                logger.info(f"Entity '{entity_name}' classified as '{direction}' via LLM")
                return direction

        # Fallback to rule-based classification
        direction = self._classify_with_rules(entity_type, domain, relation_type)
        self._cache[cache_key] = direction
        logger.info(f"Entity '{entity_name}' classified as '{direction}' via rules")
        return direction

    def _classify_with_llm(
        self,
        entity_name: str,
        entity_type: str,
        domain: Optional[str],
        relation_type: Optional[str],
        scenario_context: Optional[str],
    ) -> Optional[str]:
        """Use LLM to classify impact direction with context."""
        try:
            system_prompt = """You are an intelligence analyst classifying the impact direction of entities in geopolitical scenarios.

Classify whether an entity's involvement represents:
- "risk": Negative, threatening, or destabilizing impact
- "opportunity": Positive, beneficial, or stabilizing impact
- "neutral": Unclear, balanced, or insufficient information

Consider:
- Entity type and role
- Domain context (geopolitics, defense, economics, etc.)
- Relationship type (how the entity is connected)
- Overall scenario context

Respond with ONLY a JSON object in this exact format:
{"direction": "risk", "reasoning": "brief explanation"}

Do not include any other text."""

            # Build context string
            context_parts = [
                f"Entity: {entity_name} (Type: {entity_type})",
            ]
            if domain:
                context_parts.append(f"Domain: {domain}")
            if relation_type:
                context_parts.append(f"Connected via: {relation_type}")
            if scenario_context:
                context_parts.append(f"Scenario: {scenario_context[:200]}")  # Truncate long scenarios

            user_prompt = "Classify the impact direction:\n" + "\n".join(context_parts)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = self.llm.invoke(messages)
            content = response.content.strip()

            # Parse JSON response
            result = json.loads(content)
            direction = result.get("direction", "").lower()

            # Validate direction
            if direction not in ["risk", "opportunity", "neutral"]:
                logger.warning(f"LLM returned invalid direction: {direction}")
                return None

            logger.debug(f"LLM reasoning: {result.get('reasoning', 'N/A')}")
            return direction

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return None

    def _classify_with_rules(
        self,
        entity_type: str,
        domain: Optional[str],
        relation_type: Optional[str],
    ) -> str:
        """Use rule-based classification as fallback."""

        # Try specific rule: (entity_type, domain, relation_type)
        if domain and relation_type:
            key = (entity_type, domain, relation_type)
            if key in IMPACT_RULES:
                logger.debug(f"Matched specific rule: {key} -> {IMPACT_RULES[key]}")
                return IMPACT_RULES[key]

        # Try entity type + domain
        if domain:
            for (e_type, d, r_type), direction in IMPACT_RULES.items():
                if e_type == entity_type and d == domain:
                    logger.debug(f"Matched domain rule: ({e_type}, {d}) -> {direction}")
                    return direction

        # Try entity type default
        if entity_type in DEFAULT_ENTITY_RULES:
            direction = DEFAULT_ENTITY_RULES[entity_type]
            logger.debug(f"Using entity type default: {entity_type} -> {direction}")
            return direction

        # Ultimate fallback
        logger.warning(f"No rules matched for ({entity_type}, {domain}, {relation_type}), defaulting to neutral")
        return "neutral"

    def classify_batch(
        self,
        entities: list[Dict],
        scenario_context: Optional[str] = None,
    ) -> list[str]:
        """
        Classify multiple entities at once.

        Args:
            entities: List of dicts with keys: name, type, domain, relation_type
            scenario_context: Optional scenario description

        Returns:
            List of directions in same order as input
        """
        return [
            self.classify(
                entity_name=e.get("name", ""),
                entity_type=e.get("type", "Unknown"),
                domain=e.get("domain"),
                relation_type=e.get("relation_type"),
                scenario_context=scenario_context,
            )
            for e in entities
        ]

    def clear_cache(self) -> None:
        """Clear the classification cache."""
        self._cache.clear()
        logger.debug("Impact direction classifier cache cleared")
