"""Redis Blackboard — shared environment / stigmergy substrate.

Key schema:
  swarm:task:{task_id}          → JSON task payload (TTL 1h)
  swarm:vote:{task_id}          → JSON list of worker vote bundles (TTL 1h)
  swarm:pheromone:{topic}       → float score reinforced by consensus (TTL 24h)
  swarm:causal:alternatives:{d} → JSON list of alternative causal chains (TTL 24h)
  swarm:signals:dance           → Redis Pub/Sub channel for bee-algorithm communication

Workers read shared context from the blackboard, perform independent analysis,
then deposit results ("pheromones") back. Emergence happens in the consensus layer.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import redis

from config import REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)

TASK_TTL = 3600          # 1 hour
VOTE_TTL = 3600          # 1 hour
PHEROMONE_TTL = 86400    # 24 hours
CAUSAL_ALT_TTL = 86400   # 24 hours
DANCE_CHANNEL = "swarm:signals:dance"


class SwarmBlackboard:
    """Redis-backed shared environment for swarm workers.

    Instantiate once per swarm run; the connection is lightweight (pooled).
    """

    def __init__(self):
        self._r = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=5,
        )

    # ── Task management ───────────────────────────────────────────────────────

    def post_task(self, payload: dict) -> str:
        """Post a new task and return its task_id."""
        task_id = str(uuid.uuid4())
        key = f"swarm:task:{task_id}"
        self._r.set(key, json.dumps(payload), ex=TASK_TTL)
        logger.debug(f"[blackboard] task posted: {task_id}")
        return task_id

    def get_task(self, task_id: str) -> dict | None:
        raw = self._r.get(f"swarm:task:{task_id}")
        return json.loads(raw) if raw else None

    # ── Vote / pheromone deposits ─────────────────────────────────────────────

    def deposit_vote(self, task_id: str, worker_id: str, triples: list[dict]) -> None:
        """Worker deposits its extracted triples as votes on the blackboard."""
        key = f"swarm:vote:{task_id}"
        bundle = {"worker_id": worker_id, "triples": triples, "ts": time.time()}
        # Use a list per task; RPUSH then expire (reset TTL each push)
        self._r.rpush(key, json.dumps(bundle))
        self._r.expire(key, VOTE_TTL)
        logger.debug(f"[blackboard] vote deposited: task={task_id} worker={worker_id} triples={len(triples)}")

    def get_votes(self, task_id: str) -> list[dict]:
        """Return all vote bundles for a task."""
        key = f"swarm:vote:{task_id}"
        raw_list = self._r.lrange(key, 0, -1)
        return [json.loads(r) for r in raw_list]

    # ── Pheromone reinforcement ───────────────────────────────────────────────

    def reinforce_pheromone(self, topic: str, delta: float = 1.0) -> float:
        """Increment pheromone strength for a topic. Returns new value."""
        key = f"swarm:pheromone:{topic}"
        new_val = self._r.incrbyfloat(key, delta)
        self._r.expire(key, PHEROMONE_TTL)
        return float(new_val)

    def get_pheromone(self, topic: str) -> float:
        key = f"swarm:pheromone:{topic}"
        raw = self._r.get(key)
        return float(raw) if raw else 0.0

    def evaporate_pheromones(self, factor: float = 0.9) -> int:
        """Decay all pheromone values by factor. Returns number of keys updated."""
        keys = self._r.keys("swarm:pheromone:*")
        updated = 0
        for key in keys:
            raw = self._r.get(key)
            if raw:
                new_val = float(raw) * factor
                if new_val < 0.01:
                    self._r.delete(key)
                else:
                    ttl = self._r.ttl(key)
                    self._r.set(key, new_val, ex=max(ttl, 60))
                updated += 1
        return updated

    # ── Causal chain alternative futures ─────────────────────────────────────

    def store_alternative_chains(self, domain: str, chains: list[dict]) -> None:
        """Store losing causal chains as alternative futures."""
        key = f"swarm:causal:alternatives:{domain}"
        self._r.set(key, json.dumps({
            "domain": domain,
            "alternatives": chains,
            "ts": time.time(),
        }), ex=CAUSAL_ALT_TTL)

    def get_alternative_chains(self, domain: str) -> list[dict]:
        key = f"swarm:causal:alternatives:{domain}"
        raw = self._r.get(key)
        if raw:
            return json.loads(raw).get("alternatives", [])
        return []

    # ── Bee-algorithm dance channel ───────────────────────────────────────────

    def publish_dance(self, message: dict) -> None:
        """Scout publishes 'waggle dance' message to worker pool."""
        try:
            self._r.publish(DANCE_CHANNEL, json.dumps(message))
        except Exception as e:
            logger.debug(f"[blackboard] dance publish failed: {e}")

    def subscribe_dance(self) -> redis.client.PubSub:
        """Worker subscribes to scout dance messages."""
        ps = self._r.pubsub(ignore_subscribe_messages=True)
        ps.subscribe(DANCE_CHANNEL)
        return ps

    # ── Low-confidence buffer ─────────────────────────────────────────────────

    def store_low_confidence(self, task_id: str, triples: list[dict]) -> None:
        """Store dissenting triples for human review."""
        key = f"swarm:lowconf:{task_id}"
        self._r.set(key, json.dumps(triples), ex=VOTE_TTL * 6)

    def close(self):
        self._r.close()
