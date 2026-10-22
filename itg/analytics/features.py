"""Features from events and freeze frames.

The same functions build features for training (on actions that happened)
and for counterfactuals (on actions that could have happened), so the
models score real and hypothetical options the same way.

Only geometry is shared with the simulator. Its hidden *_truth models are
never imported here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from itg.sim.pitch import GOAL, dist_m, dists_m, goal_angle, in_box, mirror, segment_dists, shot_blockers

SET_PIECES = ("Throw-in", "Goal Kick", "Corner", "Free Kick", "Kick Off")

PASS_FEATURES = ["length_m", "dx", "start_x", "start_yc", "end_x", "end_yc", "lane_m", "mark_m", "press_m",
                 "crowd_end", "abs_angle", "high", "cross", "through", "set_piece"]
SHOT_FEATURES = ["dist_m", "angle", "header", "penalty", "free_kick", "pressure_m", "blockers",
                 "gk_dist_m", "gk_offline_m", "crowd_3m"]
CARRY_FEATURES = ["press_m", "start_x", "start_yc", "opps_5m", "opps_ahead_10m", "start_in_box"]


@dataclass
class Frame:
    """A freeze frame split into arrays, in the actor's attacking frame."""
    actor: np.ndarray
    mates: np.ndarray
    mate_ids: list[str]
    opps: np.ndarray        # all opponents, keeper included
    opp_out: np.ndarray     # outfield opponents only
    gk: np.ndarray | None

    @classmethod
    def from_event(cls, ev: dict) -> "Frame":
        ff = ev["freeze_frame"]
        actor = np.array(ev["location"], float)
        mates, ids, opps, out, gk = [], [], [], [], None
        for f in ff:
            loc = np.array(f["location"], float)
            if f["teammate"]:
                if not f["actor"]:
                    mates.append(loc)
                    ids.append(f["player_id"])
            else:
                opps.append(loc)
                if f["keeper"]:
                    gk = loc
                else:
                    out.append(loc)
        arr = lambda xs: np.array(xs, float).reshape(-1, 2)  # noqa: E731
        return cls(actor=actor, mates=arr(mates), mate_ids=ids, opps=arr(opps), opp_out=arr(out), gk=gk)


def _min(a: np.ndarray, default: float = 40.0) -> float:
    return float(a.min()) if len(a) else default


def pass_features(start, end, fr: Frame, *, high: bool, cross: bool = False, through: bool = False,
                  set_piece: bool = False) -> dict:
    start, end = np.asarray(start, float), np.asarray(end, float)
    lane, t = segment_dists(start, end, fr.opps)
    lane = np.where((t > 0.03) & (t < 0.97), lane, 99.0) if len(lane) else lane
    return {
        "length_m": dist_m(start, end),
        "dx": float(end[0] - start[0]),
        "start_x": float(start[0]),
        "start_yc": abs(float(start[1]) - 40.0),
        "end_x": float(end[0]),
        "end_yc": abs(float(end[1]) - 40.0),
        "lane_m": min(_min(lane, 40.0), 40.0),
        "mark_m": min(_min(dists_m(end, fr.opps)), 40.0),
        "press_m": min(_min(dists_m(start, fr.opp_out)), 40.0),
        "crowd_end": int((dists_m(end, fr.opps) < 5.0).sum()) if len(fr.opps) else 0,
        "abs_angle": abs(math.atan2(end[1] - start[1], end[0] - start[0])),
        "high": int(high),
        "cross": int(cross),
        "through": int(through),
        "set_piece": int(set_piece),
    }


def shot_features(loc, fr: Frame, *, header: bool, set_piece: str | None = None) -> dict:
    loc = np.asarray(loc, float)
    gk = fr.gk if fr.gk is not None else np.array([120.0, 40.0])
    return {
        "dist_m": dist_m(loc, GOAL),
        "angle": goal_angle(loc),
        "header": int(header),
        "penalty": int(set_piece == "Penalty"),
        "free_kick": int(set_piece == "Free Kick"),
        "pressure_m": min(_min(dists_m(loc, fr.opp_out)), 40.0),
        "blockers": shot_blockers(loc, fr.opp_out) if len(fr.opp_out) else 0,
        "gk_dist_m": dist_m(loc, gk),
        "gk_offline_m": dist_m(gk, GOAL),
        "crowd_3m": int((dists_m(loc, fr.opp_out) < 3.0).sum()) if len(fr.opp_out) else 0,
    }


def carry_features(start, fr: Frame) -> dict:
    """Chance of keeping the ball on a carry, from the situation at the start only.

    (The recorded end of a failed carry is where it was lost, so using it would leak the outcome.)
    """
    start = np.asarray(start, float)
    rel = fr.opps - start if len(fr.opps) else np.zeros((0, 2))
    d = dists_m(start, fr.opps) if len(fr.opps) else np.array([])
    ahead = (rel[:, 0] > 0) & (d < 10.0) if len(rel) else np.array([], bool)
    return {
        "press_m": min(_min(dists_m(start, fr.opp_out)), 40.0),
        "start_x": float(start[0]),
        "start_yc": abs(float(start[1]) - 40.0),
        "opps_5m": int((d < 5.0).sum()) if len(d) else 0,
        "opps_ahead_10m": int(ahead.sum()) if len(ahead) else 0,
        "start_in_box": int(in_box(start)),
    }


def turnover_point(start, end, fr: Frame) -> np.ndarray:
    """Where a failed pass would most likely be won: the opponent closest to the lane, else the target."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    if not len(fr.opps):
        return end
    lane, t = segment_dists(start, end, fr.opps)
    lane = np.where(t > 0.03, lane, 99.0)
    i = int(np.argmin(lane))
    if lane[i] < 4.0:
        return start + (end - start) * float(t[i])
    return end


def to_opponent_frame(p) -> np.ndarray:
    return mirror(np.asarray(p, float))
