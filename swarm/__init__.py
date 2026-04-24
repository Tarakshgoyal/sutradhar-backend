"""Swarm Intelligence package.

Three swarms built on a shared Redis blackboard (stigmergy substrate):

  A. ExtractionSwarm  — Consensus extraction (5 perspective workers)
  B. CausalChainSwarm — Competitive causal-chain hypothesis generation
  C. SignalSwarm      — Bee-algorithm anomaly detection
"""

from swarm.blackboard import SwarmBlackboard
from swarm.extraction_swarm import ExtractionSwarm
from swarm.causal_swarm import CausalChainSwarm
from swarm.signal_swarm import SignalSwarm

__all__ = ["SwarmBlackboard", "ExtractionSwarm", "CausalChainSwarm", "SignalSwarm"]
