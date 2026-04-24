"""Flush the Redis seen-URLs set so the scraper re-fetches all articles.

Run this when the scraper is returning 0 articles because all feed URLs
are already marked as "seen" in Redis from a previous seed/run.

Usage:
    python -m scripts.flush_redis_seen
"""
import sys
sys.path.insert(0, ".")

import redis
from config import REDIS_HOST, REDIS_PORT

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

KEY = "india-innovates:seen_urls"
count = r.scard(KEY)
r.delete(KEY)
print(f"Flushed {count} URLs from Redis key '{KEY}'")
print("The scraper will now re-fetch all articles on the next cycle.")
