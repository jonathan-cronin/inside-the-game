"""Counterfactual decision engine.

For any on-ball moment we rebuild the options the player had from the
freeze frame (every pass, a shot, a carry) and value each one with the
learned models:

    EV(option) = P(success) * xT(where the ball ends up)
               - P(failure) * xT for the opponent at the turnover point

    EV(shot)   = xG

The chosen action is valued the same way, so

    decision value = EV(chosen) - EV(best alternative)

is in expected-goal units. The win-probability model turns that into
"this decision cost 2.1% win probability".
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from itg.sim.pitch import GOAL, clip_pitch, dist_m, dists_m

from .features import Frame, carry_features, pass_features, shot_features, to_opponent_frame, turnover_point
from .models import ModelBundle

ON_BALL = ("Pass", "Shot", "Carry", "Dribble")


@dataclass
class Option:
    kind: str                     # pass | shot | carry | dribble
    end: list[float]
    p_success: float
    value_if_success: float
    cost_if_failed: float
    ev: float
    target_id: str | None = None
    target_name: str | None = None
    xg: float | None = None
    detail: dict = field(default_factory=dict)

    def short(self) -> dict:
        d = asdict(self)
        for k in ("p_success", "value_if_success", "cost_if_failed", "ev", "xg"):
            if d[k] is not None:
                d[k] = round(d[k], 4)
        d["end"] = [round(v, 1) for v in d["end"]]
        return d


class DecisionEngine:
    def __init__(self, bundle: ModelBundle, names: dict[str, str] | None = None, dribble_success: float = 0.5):
        self.b = bundle
        self.names = names or {}
        self.dribble_success = dribble_success

    # ----------------------------------------------------------- valuation
    def xt(self, p) -> float:
        return self.b.xt.value(float(p[0]), float(p[1]))

    def _pass(self, start, end, fr: Frame, *, high=None, cross=False, through=False, set_piece=False,
              target_id=None) -> Option:
        length = dist_m(start, end)
        high = length > 32 if high is None else high
        f = pass_features(start, end, fr, high=high, cross=cross, through=through, set_piece=set_piece)
        p = float(self.b.pass_success.predict(f)[0])
        v = self.xt(end)
        tp = turnover_point(start, end, fr)
        c = self.xt(to_opponent_frame(tp))
        return Option("pass", list(map(float, end)), p, v, c, p * v - (1 - p) * c, target_id,
                      self.names.get(target_id), detail={"length_m": round(length, 1), "lane_m": round(f["lane_m"], 1),
                                                         "receiver_space_m": round(f["mark_m"], 1),
                                                         "difficulty": round(100 * (1 - p))})

    def _shot(self, loc, fr: Frame, *, header=False, set_piece=None) -> Option:
        f = shot_features(loc, fr, header=header, set_piece=set_piece)
        xg = float(self.b.xg.predict(f)[0])
        c = self.xt([6.0, 40.0])  # a miss usually means a goal kick: their ball, at (6, 40) in their frame
        return Option("shot", [120.0, 40.0], xg, 1.0, c, xg - (1 - xg) * c, xg=xg,
                      detail={"distance_m": round(f["dist_m"], 1), "blockers": f["blockers"],
                              "pressure_m": round(f["pressure_m"], 1)})

    def _carry(self, start, end, fr: Frame) -> Option:
        p = float(self.b.carry.predict(carry_features(start, fr))[0])
        v = self.xt(end)
        c = self.xt(to_opponent_frame(start))
        return Option("carry", list(map(float, end)), p, v, c, p * v - (1 - p) * c,
                      detail={"distance_m": round(dist_m(start, end), 1)})

    # ------------------------------------------------------------ options
    def options(self, ev: dict) -> list[Option]:
        """All realistic alternatives, scored in as few model calls as possible."""
        fr = Frame.from_event(ev)
        start = fr.actor
        opts: list[Option] = []
        # passes to every teammate, one batched model call
        targets = [(pid, loc) for pid, loc in zip(fr.mate_ids, fr.mates) if 3.0 <= dist_m(start, loc) <= 60.0]
        if targets:
            feats = [pass_features(start, loc, fr, high=dist_m(start, loc) > 32) for _, loc in targets]
            ps = self.b.pass_success.predict(feats)
            for (pid, loc), f, p in zip(targets, feats, ps):
                p = float(p)
                v = self.xt(loc)
                c = self.xt(to_opponent_frame(turnover_point(start, loc, fr)))
                opts.append(Option("pass", list(map(float, loc)), p, v, c, p * v - (1 - p) * c, pid,
                                   self.names.get(pid), detail={"length_m": round(f["length_m"], 1),
                                                                "lane_m": round(f["lane_m"], 1),
                                                                "receiver_space_m": round(f["mark_m"], 1),
                                                                "difficulty": round(100 * (1 - p))}))
        if dist_m(start, GOAL) < 36:
            opts.append(self._shot(start, fr))
        # carries: straight ahead and towards goal; keep-ball chance depends only on the start
        p_keep = float(self.b.carry.predict(carry_features(start, fr))[0])
        c = self.xt(to_opponent_frame(start))
        for dvec in (np.array([1.0, 0.0]), (GOAL - start) / (np.linalg.norm(GOAL - start) + 1e-9)):
            dvec = dvec / np.linalg.norm(dvec)
            ahead = fr.opps[((fr.opps - start) @ dvec) > 0] if len(fr.opps) else fr.opps
            space = float(dists_m(start, ahead).min()) if len(ahead) else 30.0
            L = float(np.clip(space * 0.5, 3.0, 10.0)) / 0.875
            end = clip_pitch(start + dvec * L)
            v = self.xt(end)
            opts.append(Option("carry", list(map(float, end)), p_keep, v, c, p_keep * v - (1 - p_keep) * c,
                               detail={"distance_m": round(dist_m(start, end), 1)}))
        return opts

    def actual(self, ev: dict) -> Option | None:
        fr = Frame.from_event(ev)
        start = fr.actor
        t = ev["type"]
        if t == "Pass":
            p = ev["pass"]
            target = np.array(p["end_location"], float)
            rid = p["recipient"]["id"]
            if not p["through_ball"]:
                for pid, loc in zip(fr.mate_ids, fr.mates):
                    if pid == rid:
                        target = loc
            o = self._pass(start, target, fr, high=p["height"] == "High Pass", cross=p["cross"],
                           through=p["through_ball"], set_piece=p["type"] != "Open Play", target_id=rid)
            o.detail.update(cross=p["cross"], through_ball=p["through_ball"], outcome=p["outcome"])
            return o
        if t == "Shot":
            s = ev["shot"]
            o = self._shot(start, fr, header=s["body_part"] == "Head",
                           set_piece=None if s["type"] == "Open Play" else s["type"])
            o.detail.update(outcome=s["outcome"], body_part=s["body_part"], shot_speed_kmh=s["shot_speed_kmh"])
            return o
        if t == "Carry":
            return self._carry(start, np.array(ev["carry"]["end_location"], float), fr)
        if t == "Dribble":
            end = clip_pitch(start + np.array([5.0, 0.0]))
            v, c = self.xt(end), self.xt(to_opponent_frame(start))
            p = self.dribble_success
            return Option("dribble", list(map(float, end)), p, v, c, p * v - (1 - p) * c,
                          detail={"outcome": ev["dribble"]["outcome"]})
        return None

    # ----------------------------------------------------------- evaluate
    def evaluate(self, ev: dict, goal_wp_swing: float | None = None) -> dict | None:
        """Score the decision behind an on-ball event.

        goal_wp_swing: how much a goal for the acting team right now would add to its win
        probability, used to turn expected-goal differences into win-probability terms.
        """
        if ev["type"] not in ON_BALL or "freeze_frame" not in ev:
            return None
        if ev["type"] == "Shot" and (ev["shot"]["body_part"] == "Head" or ev["shot"]["type"] == "Penalty"):
            return None  # a header off a cross or a penalty is not a choice between options
        if ev["type"] == "Pass" and (ev["pass"]["type"] in ("Kick Off", "Goal Kick", "Throw-in")
                                     or ev["pass"]["length_m"] < 3.0):
            return None
        if ev["type"] == "Carry" and ev["carry"]["distance_m"] < 5.0:
            return None  # a short carry is a touch to set up the next action, not a choice in itself
        chosen = self.actual(ev)
        if chosen is None:
            return None
        alts = [o for o in self.options(ev)
                if not (o.kind == chosen.kind and (o.target_id == chosen.target_id or o.kind in ("shot", "carry")))]
        alts.sort(key=lambda o: -o.ev)
        best = alts[0] if alts else None
        dv = chosen.ev - best.ev if best else 0.0
        out = {
            "event_id": ev["id"],
            "chosen": chosen.short(),
            "best_alternative": best.short() if best else None,
            "top_alternatives": [o.short() for o in alts[:3]],
            "n_options": len(alts) + 1,
            "decision_value": round(dv, 4),           # >0: better than every alternative
            "rank": 1 + sum(1 for o in alts if o.ev > chosen.ev),
        }
        if goal_wp_swing is not None:
            out["decision_wp"] = round(dv * goal_wp_swing, 4)
        return out
