"""Minute-by-minute match states for training the win-probability model."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .data import Timeline


def team_ratings(timelines: list[Timeline], shot_xg: dict[str, float]) -> dict[str, dict]:
    """Pre-match strength per club: average xG for and against per match."""
    agg = defaultdict(lambda: {"xg_for": 0.0, "xg_against": 0.0, "n": 0})
    for tl in timelines:
        xg = {tl.home: 0.0, tl.away: 0.0}
        for _, team, eid, _ in tl.shots:
            xg[team] += shot_xg.get(eid, 0.0)
        for t, o in ((tl.home, tl.away), (tl.away, tl.home)):
            agg[t]["xg_for"] += xg[t]
            agg[t]["xg_against"] += xg[o]
            agg[t]["n"] += 1
    return {t: {"xg_for": v["xg_for"] / v["n"], "xg_against": v["xg_against"] / v["n"], "matches": v["n"]}
            for t, v in agg.items()}


def recent_share(tl: Timeline, shot_xg: dict[str, float], minute: int, team: str, window: int = 15) -> float:
    mine = sum(shot_xg.get(e, 0.0) for mm, t, e, _ in tl.shots if minute - window <= mm < minute and t == team)
    tot = sum(shot_xg.get(e, 0.0) for mm, t, e, _ in tl.shots if minute - window <= mm < minute)
    return (mine + 0.15) / (tot + 0.3)


def state_rows(timelines: list[Timeline], shot_xg: dict[str, float], ratings: dict) -> pd.DataFrame:
    rows = []
    for tl in timelines:
        goals = [(mm, t) for mm, t, _, g in tl.shots if g]
        for minute in range(0, min(tl.end_minute, 94)):
            for side, opp, is_home in ((tl.home, tl.away, 1), (tl.away, tl.home, 0)):
                gf = sum(1 for mm, t in goals if mm < minute and t == side)
                ga = sum(1 for mm, t in goals if mm < minute and t == opp)
                after = sum(1 for mm, t in goals if mm >= minute and t == side)
                rd = (sum(1 for mm, t in tl.reds if mm < minute and t == opp)
                      - sum(1 for mm, t in tl.reds if mm < minute and t == side))
                rows.append({
                    "match_id": tl.match_id, "minute": minute, "is_home": is_home,
                    "attack": ratings[side]["xg_for"], "opp_defence": ratings[opp]["xg_against"],
                    "score_diff": gf - ga, "red_diff": rd,
                    "recent_xg_share": recent_share(tl, shot_xg, minute, side),
                    "goals_after": after, "t_left": float(tl.end_minute - minute),
                })
    return pd.DataFrame(rows)


def outcome(tl: Timeline) -> str:
    fs = tl.final_score
    return "home" if fs["home"] > fs["away"] else ("away" if fs["away"] > fs["home"] else "draw")


def live_state(tl: Timeline, minute: int, shot_xg: dict[str, float]) -> dict:
    goals = [(mm, t) for mm, t, _, g in tl.shots if g]
    return {
        "home": tl.home, "away": tl.away, "minute": minute,
        "home_goals": sum(1 for mm, t in goals if mm < minute and t == tl.home),
        "away_goals": sum(1 for mm, t in goals if mm < minute and t == tl.away),
        "red_diff": (sum(1 for mm, t in tl.reds if mm < minute and t == tl.away)
                     - sum(1 for mm, t in tl.reds if mm < minute and t == tl.home)),
        "recent_xg_share": recent_share(tl, shot_xg, minute, tl.home),
    }


def log_loss(probs: list[dict], outcomes: list[str]) -> float:
    return float(-np.mean([np.log(max(p[o], 1e-9)) for p, o in zip(probs, outcomes)]))


def brier(probs: list[dict], outcomes: list[str]) -> float:
    return float(np.mean([sum((p[k] - (k == o)) ** 2 for k in ("home", "draw", "away")) for p, o in zip(probs, outcomes)]))
