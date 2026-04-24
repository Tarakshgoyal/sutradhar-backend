"""Circuit breaker for data source scrapers.

Prevents cascading failures from a single bad source.
Uses Redis for state persistence across restarts.

States:
    CLOSED   — Normal operation.  Failures counted.
    OPEN     — Source is failing.  All calls rejected for `cooldown_seconds`.
    HALF_OPEN — Testing recovery.  One test call allowed.

Usage:
    breaker = CircuitBreaker(source_name="PIB", redis_client=r)

    if breaker.can_request():
        try:
            data = scrape_pib()
            breaker.record_success()
        except Exception as e:
            breaker.record_failure()
            raise
    else:
        # Source is open, log an intelligence gap signal
        logger.warning(f"Circuit open for {breaker.source_name}, skipping")
"""

import logging
import time
from enum import Enum
from typing import Optional

import redis as redis_lib

logger = logging.getLogger(__name__)


def _as_text(value) -> Optional[str]:
    """Normalize Redis values to text for both bytes and decode_responses=True clients."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


class CBState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Redis-backed circuit breaker for an individual data source."""

    def __init__(
        self,
        source_name: str,
        redis_client: redis_lib.Redis,
        failure_threshold: int = 3,
        cooldown_seconds: int = 300,   # 5 minutes
        half_open_after: int = 60,     # try recovery after 1 minute
    ):
        self.source_name = source_name
        self.redis = redis_client
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_after = half_open_after

        self._key_state = f"cb:{source_name}:state"
        self._key_failures = f"cb:{source_name}:failures"
        self._key_opened_at = f"cb:{source_name}:opened_at"

    # ── State access ──────────────────────────────────────────────────────────

    @property
    def state(self) -> CBState:
        raw = self.redis.get(self._key_state)
        if raw is None:
            return CBState.CLOSED
        state = CBState(_as_text(raw))
        if state == CBState.OPEN:
            # Check if we should transition to HALF_OPEN
            opened_at = self.redis.get(self._key_opened_at)
            if opened_at and time.time() - float(opened_at) >= self.half_open_after:
                self._set_state(CBState.HALF_OPEN)
                return CBState.HALF_OPEN
        return state

    @property
    def failure_count(self) -> int:
        val = self.redis.get(self._key_failures)
        return int(val) if val else 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def can_request(self) -> bool:
        """Return True if a request should proceed."""
        s = self.state
        return s in (CBState.CLOSED, CBState.HALF_OPEN)

    def record_success(self) -> None:
        """Call after a successful request."""
        self.redis.delete(self._key_failures)
        self.redis.delete(self._key_opened_at)
        self._set_state(CBState.CLOSED)
        logger.debug(f"Circuit {self.source_name}: success → CLOSED")

    def record_failure(self) -> None:
        """Call after a failed request."""
        failures = self.redis.incr(self._key_failures)
        logger.warning(
            f"Circuit {self.source_name}: failure #{failures} "
            f"(threshold={self.failure_threshold})"
        )
        if int(failures) >= self.failure_threshold or self.state == CBState.HALF_OPEN:
            self._open()

    def _open(self) -> None:
        self._set_state(CBState.OPEN)
        self.redis.set(self._key_opened_at, str(time.time()))
        logger.error(
            f"Circuit {self.source_name}: OPEN — will retry after "
            f"{self.half_open_after}s"
        )

    def _set_state(self, state: CBState) -> None:
        self.redis.set(
            self._key_state, state.value, ex=self.cooldown_seconds * 10
        )

    # ── Status dict (for /api/signals intelligence gap reporting) ─────────────

    def status(self) -> dict:
        s = self.state
        opened_at = self.redis.get(self._key_opened_at)
        return {
            "source": self.source_name,
            "state": s.value,
            "failure_count": self.failure_count,
            "opened_at": float(opened_at) if opened_at else None,
            "is_healthy": s == CBState.CLOSED,
        }
