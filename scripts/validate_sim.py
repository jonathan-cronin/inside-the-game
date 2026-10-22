"""Simulate a batch of matches and compare headline stats with real top-flight ranges.

    python scripts/validate_sim.py --n 50
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from itg.sim import CLUBS, simulate_match  # noqa: E402

# Approximate per-match (both teams combined) ranges for a modern top division.
TARGETS = {
    "goals": (2.4, 3.3),
    "shots": (21, 29),
    "shots_on_target_pct": (30, 40),
    "xg_total": (2.4, 3.3),
    "passes": (800, 1100),
    "pass_completion_pct": (78, 86),
    "pressures": (250, 420),
    "fouls": (18, 26),
    "yellow_cards": (2.5, 5),
    "corners": (8, 12),
    "dribbles": (25, 50),
    "carries": (200, 500),  # deliberate carries only, not every receipt
    "home_win_pct": (40, 48),
    "draw_pct": (20, 28),
    "dist_km_outfield": (9.0, 12.0),
}


def match_stats(m) -> dict:
    ev = m.events
    c = Counter(e["type"] for e in ev)
    passes = [e for e in ev if e["type"] == "Pass"]
    shots = [e for e in ev if e["type"] == "Shot"]
    corners = sum(1 for e in passes if e["pass"]["type"] == "Corner")
    on_t = sum(1 for e in shots if e["shot"]["outcome"] in ("Goal", "Saved"))
    xg = sum(t.get("xg", 0) for t in m.truth["events"])
    fs = m.meta["final_score"]
    dists = [v["distance_m"] for pid, v in m.truth["players"].items() if v["distance_m"] > 8000]
    return {
        "goals": fs["home"] + fs["away"],
        "shots": len(shots),
        "shots_on_target_pct": 100 * on_t / max(1, len(shots)),
        "xg_total": xg,
        "passes": len(passes),
        "pass_completion_pct": 100 * sum(e["pass"]["outcome"] == "Complete" for e in passes) / max(1, len(passes)),
        "pressures": c["Pressure"],
        "fouls": c["Foul Committed"],
        "yellow_cards": sum(1 for e in ev if e["type"] == "Foul Committed" and e["foul_committed"]["card"] in ("Yellow Card", "Second Yellow")),
        "corners": corners,
        "dribbles": c["Dribble"],
        "carries": c["Carry"],
        "home_win_pct": 100.0 * (fs["home"] > fs["away"]),
        "draw_pct": 100.0 * (fs["home"] == fs["away"]),
        "dist_km_outfield": np.mean(dists) / 1000 if dists else 0.0,
        "events": len(ev),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    pairs = list(itertools.permutations([c["id"] for c in CLUBS], 2))
    rows = []
    for i in range(args.n):
        h, a = pairs[i % len(pairs)]
        rows.append(match_stats(simulate_match(h, a, seed=1000 + i)))
    print(f"{'metric':24s} {'sim mean':>10s} {'sim sd':>8s}   target range")
    for k in rows[0]:
        vals = np.array([r[k] for r in rows])
        tgt = TARGETS.get(k)
        flag = "" if tgt is None else ("  ok" if tgt[0] <= vals.mean() <= tgt[1] else "  <-- off")
        tg = "" if tgt is None else f"{tgt[0]}-{tgt[1]}"
        print(f"{k:24s} {vals.mean():10.2f} {vals.std():8.2f}   {tg}{flag}")


if __name__ == "__main__":
    main()
