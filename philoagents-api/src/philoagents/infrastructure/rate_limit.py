"""Fixed-window rate limiting for the chat endpoints.

In-process on purpose: the API runs as a single uvicorn worker, so a shared
counter is enough to stop one player from spending the whole Qwen budget.
"""

import time
from collections import OrderedDict
from threading import Lock

from philoagents.config import settings

_lock = Lock()
# Oldest seen key first, so the pruning loop can stop as soon as it finds a live entry.
_recent: "OrderedDict[str, list[float]]" = OrderedDict()
_MAX_TRACKED_KEYS = 10_000


def _prune(now: float, window: int) -> None:
    """Forget players idle for a whole window so the table cannot grow without bound."""
    cutoff = now - window
    for stale in [
        key for key, stamps in _recent.items() if not stamps or stamps[-1] <= cutoff
    ]:
        del _recent[stale]
    while len(_recent) >= _MAX_TRACKED_KEYS:
        _recent.popitem(last=False)


def allow(key: str) -> tuple[bool, int]:
    """Consume a slot for `key`.

    Returns:
        (allowed, retry_after_seconds). retry_after is 0 when the request is allowed.
    """
    now = time.monotonic()
    window = settings.RATE_LIMIT_WINDOW_SECONDS
    limit = settings.RATE_LIMIT_MAX_MESSAGES

    with _lock:
        stamps = _recent.get(key)
        if stamps is None:
            _prune(now, window)
            stamps = []
            _recent[key] = stamps
            _recent.move_to_end(key)

        cutoff = now - window
        while stamps and stamps[0] <= cutoff:
            stamps.pop(0)

        if len(stamps) >= limit:
            return False, max(1, int(stamps[0] - cutoff) + 1)

        stamps.append(now)
        return True, 0
