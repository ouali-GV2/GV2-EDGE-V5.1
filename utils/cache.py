import time
import threading


class TTLCache:
    """
    Simple in-memory cache with TTL (time to live)
    Ultra fast + perfect for API responses & features
    """

    def __init__(self, default_ttl=300):
        self.default_ttl = default_ttl
        self._store = {}
        self._lock = threading.Lock()

    def set(self, key, value, ttl=None):
        expire_at = time.time() + (ttl or self.default_ttl)
        with self._lock:
            self._store[key] = (value, expire_at)

    def get(self, key):
        with self._lock:
            item = self._store.get(key)

            if not item:
                return None

            value, expire_at = item

            if time.time() > expire_at:
                del self._store[key]
                return None

            return value

    def delete(self, key):
        with self._lock:
            if key in self._store:
                del self._store[key]

    def clear(self):
        with self._lock:
            self._store.clear()

    def cleanup(self):
        """Remove expired items"""
        now = time.time()
        with self._lock:
            keys_to_delete = [
                k for k, (_, exp) in self._store.items()
                if now > exp
            ]
            for k in keys_to_delete:
                del self._store[k]


# ============================
# GLOBAL CACHES
# ============================

# Prices cache (short TTL)
prices_cache = TTLCache(default_ttl=60)

# Events cache (longer TTL)
events_cache = TTLCache(default_ttl=600)

# Features cache
features_cache = TTLCache(default_ttl=120)

# ============================
# CACHE WRAPPER FOR COMPATIBILITY
# ============================

class Cache(TTLCache):
    """Wrapper for backward compatibility with ttl parameter"""
    def __init__(self, ttl=300):
        super().__init__(default_ttl=ttl)
