"""Replay a simulated match as a live event feed.

Events are released in match time, scaled by `speed` (speed=60 plays 90
minutes in about 90 seconds). The sink is a plain callback, so the same
feed can print to stdout, push to a queue (e.g. Azure Event Hubs) or go
over a WebSocket.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Iterable, Iterator


def paced(events: Iterable[dict], speed: float = 60.0, max_gap_s: float = 3.0,
          sleep: Callable[[float], None] = time.sleep) -> Iterator[dict]:
    """Yield events with real-time gaps between them (divided by speed, capped at max_gap_s)."""
    prev = None
    for ev in events:
        t = ev.get("elapsed_s", 0.0)
        if prev is not None and speed > 0:
            gap = max(0.0, (t - prev) / speed)
            if gap > 0:
                sleep(min(gap, max_gap_s))
        prev = t
        yield ev


def replay(events: Iterable[dict], sink: Callable[[dict], None], speed: float = 60.0, **kw) -> int:
    n = 0
    for ev in paced(events, speed=speed, **kw):
        sink(ev)
        n += 1
    return n


def stdout_sink(ev: dict) -> None:
    print(json.dumps(ev, separators=(",", ":"), default=str), flush=True)
