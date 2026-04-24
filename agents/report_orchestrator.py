"""Multi-agent report orchestrator.

Coordinates ReportAgent (domain briefing), IndiaImpactAgent (India
strategic analysis), and InferenceAgent (causal chains, impact propagation,
weak link detection) to produce enriched intelligence reports.
"""

from __future__ import annotations

import logging

from agents.report import ReportAgent
from agents.india_impact import IndiaImpactAgent
from agents.inference import InferenceAgent

logger = logging.getLogger(__name__)

# Use causal-chain swarm when available, fall back to base InferenceAgent
_CAUSAL_SWARM_AVAILABLE = False
try:
    from swarm.causal_swarm import CausalChainSwarm
    _CAUSAL_SWARM_AVAILABLE = True
except ImportError:
    pass


class ReportOrchestrator:
    """Orchestrates multi-agent report generation.

    Pipeline:
        1. ReportAgent.generate_with_context() → domain briefing + raw data
        2. IndiaImpactAgent.analyze()           → India impact analysis
        3. CausalChainSwarm.analyze()           → 7 hypothesis workers + 3 critics
           (falls back to InferenceAgent if swarm unavailable)
        4. Merge into unified report
    """

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.report_agent = ReportAgent(model=model)
        self.india_impact_agent = IndiaImpactAgent(model=model)
        if _CAUSAL_SWARM_AVAILABLE:
            self.inference_agent = CausalChainSwarm(model=model)
            logger.info("ReportOrchestrator initialized (ReportAgent + IndiaImpactAgent + CausalChainSwarm)")
        else:
            self.inference_agent = InferenceAgent(model=model)
            logger.info("ReportOrchestrator initialized (ReportAgent + IndiaImpactAgent + InferenceAgent)")

    def close(self):
        self.report_agent.close()
        self.india_impact_agent.close()
        self.inference_agent.close()

    def generate(self, domain: str, date_range: str = "7d") -> dict:
        """Generate an enriched domain report with India impact and inference.

        Args:
            domain: One of climate, defence, economics, geopolitics, society
            date_range: "7d", "30d", etc.

        Returns:
            Dict with all DomainBriefing fields + 'india_impact' + 'inference'.
        """
        # Step 1: Domain briefing via ReportAgent
        logger.info(f"{'=' * 60}")
        logger.info(f"ORCHESTRATOR Step 1: ReportAgent — {domain.upper()} briefing")
        logger.info(f"{'=' * 60}")

        report_result = self.report_agent.generate_with_context(domain, date_range)

        # Step 2: India impact analysis
        logger.info(f"{'=' * 60}")
        logger.info(f"ORCHESTRATOR Step 2: IndiaImpactAgent — {domain.upper()} India analysis")
        logger.info(f"{'=' * 60}")

        try:
            india_impact = self.india_impact_agent.analyze(domain, report_result)
        except Exception as e:
            logger.error(f"IndiaImpactAgent failed for {domain}: {e}", exc_info=True)
            india_impact = {
                "executive_summary": f"India impact analysis unavailable for {domain}.",
                "strategic_assessment": {"summary": "", "implications": []},
                "transparency_insights": [],
                "national_advantages": [],
                "risks": [],
                "global_positioning": [],
                "recommendations": [],
            }

        # Step 3: Inference analysis (causal chains, impact, weak links)
        logger.info(f"{'=' * 60}")
        logger.info(f"ORCHESTRATOR Step 3: InferenceAgent — {domain.upper()} graph inference")
        logger.info(f"{'=' * 60}")

        try:
            inference = self.inference_agent.analyze(report_result)
        except Exception as e:
            logger.error(f"InferenceAgent failed for {domain}: {e}", exc_info=True)
            inference = {
                "executive_summary": f"Inference analysis unavailable for {domain}.",
                "causal_chains": [],
                "impact_propagations": [],
                "weak_links": [],
            }

        # Step 4: Merge
        logger.info(f"{'=' * 60}")
        logger.info(f"ORCHESTRATOR Step 4: Merging report — {domain.upper()}")
        logger.info(f"{'=' * 60}")

        merged = {
            **report_result.briefing,
            "india_impact": india_impact,
            "inference": inference,
        }

        logger.info(f"Enriched {domain} report complete (india_impact + inference included)")
        return merged

