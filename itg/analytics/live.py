"""Live match analyst: consumes events one at a time and emits insights.

This is the Interpret + Explain stage of the pipeline. It works the same on
a live feed and on a finished match, and every number an agent narrates
comes from here, so the narration is backed by computed stats.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from itg.sim.pitch import dist_m

from .decisions import ON_BALL, DecisionEngine
from .models import ModelBundle

SPRINT_KMH = 30.0


@dataclass
class PlayerStats:
    id: str
    name: str
    team_id: str
    passes: int = 0
    passes_completed: int = 0
    difficult_completed: int = 0
    shots: int = 0
    goals: int = 0
    xg: float = 0.0
    xt_added: float = 0.0
    decision_value: float = 0.0
    decisions: int = 0
    best_decisions: int = 0
    distance_m: float = 0.0
    top_speed_kmh: float = 0.0
    touches: int = 0
    _last_pos: tuple | None = None
    _last_t: float | None = None

    def public(self) -> dict:
        d = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items() if not k.startswith("_")}
        d["pass_completion"] = round(self.passes_completed / self.passes, 3) if self.passes else None
        return d


class LiveAnalyst:
    def __init__(self, bundle: ModelBundle, meta: dict):
        self.b = bundle
        self.meta = meta
        self.home, self.away = meta["home"]["id"], meta["away"]["id"]
        self.names = {p["id"]: p["name"] for p in meta["players"]}
        self.team_of = {p["id"]: p["team_id"] for p in meta["players"]}
        self.engine = DecisionEngine(bundle, self.names)
        self.players = {p["id"]: PlayerStats(p["id"], p["name"], p["team_id"]) for p in meta["players"]}
        self.score = {self.home: 0, self.away: 0}
        self.reds = {self.home: 0, self.away: 0}
        self.minute = 0
        self.elapsed = 0.0
        self.xg_log: list[tuple[int, str, float]] = []
        self.xt_window: deque = deque()  # (elapsed_s, team, xt_added) for momentum
        self.team = {t: defaultdict(float) for t in (self.home, self.away)}
        self.possession_time = {self.home: 0.0, self.away: 0.0}
        self._last_poss: tuple[str, float] | None = None
        self.wp_timeline: list[dict] = []
        self._wp_cache: dict = {}
        self._momentum_leader: str | None = None
        self.insights: list[dict] = []

    # ---------------------------------------------------------------- state
    def state(self) -> dict:
        mine = sum(x for m, t, x in self.xg_log if self.minute - 15 <= m <= self.minute and t == self.home)
        tot = sum(x for m, t, x in self.xg_log if self.minute - 15 <= m <= self.minute)
        return {"home": self.home, "away": self.away, "minute": self.minute,
                "home_goals": self.score[self.home], "away_goals": self.score[self.away],
                "red_diff": self.reds[self.away] - self.reds[self.home],
                "recent_xg_share": (mine + 0.15) / (tot + 0.3)}

    def win_probability(self, extra_home=0, extra_away=0) -> dict:
        s = self.state()
        key = (s["minute"], s["home_goals"], s["away_goals"], s["red_diff"], round(s["recent_xg_share"], 2),
               extra_home, extra_away)
        if key not in self._wp_cache:
            self._wp_cache[key] = self.b.wp.probabilities(s, extra_home, extra_away)
        return self._wp_cache[key]

    def team_win(self, team_id: str, wp: dict) -> float:
        return wp["home"] if team_id == self.home else wp["away"]

    def other(self, team_id: str) -> str:
        return self.away if team_id == self.home else self.home

    def goal_swing(self, team_id: str) -> float:
        now = self.team_win(team_id, self.win_probability())
        after = self.team_win(team_id, self.win_probability(*(1, 0) if team_id == self.home else (0, 1)))
        return after - now

    def summary(self) -> dict:
        wp = self.win_probability()
        total_poss = sum(self.possession_time.values()) or 1.0
        return {
            "minute": self.minute,
            "score": {"home": self.score[self.home], "away": self.score[self.away]},
            "win_probability": {k: round(v, 3) for k, v in wp.items()},
            "teams": {t: {"xg": round(self.team[t]["xg"], 2), "shots": int(self.team[t]["shots"]),
                          "passes": int(self.team[t]["passes"]),
                          "pass_completion": round(self.team[t]["passes_completed"] / max(1, self.team[t]["passes"]), 3),
                          "xt_added": round(self.team[t]["xt_added"], 3),
                          "possession": round(self.possession_time[t] / total_poss, 3)}
                      for t in (self.home, self.away)},
            "momentum": self.momentum(),
        }

    def momentum(self, window_s: float = 600.0) -> dict:
        """Share of threat (xT added) created by each team in the last 10 minutes."""
        while self.xt_window and self.xt_window[0][0] < self.elapsed - window_s:
            self.xt_window.popleft()
        h = sum(max(0.0, x) for _, t, x in self.xt_window if t == self.home)
        a = sum(max(0.0, x) for _, t, x in self.xt_window if t == self.away)
        share = (h + 0.02) / (h + a + 0.04)
        return {"home_share": round(share, 3), "away_share": round(1 - share, 3)}

    # ------------------------------------------------------------- process
    def process(self, ev: dict) -> list[dict]:
        out: list[dict] = []
        self.minute = ev["minute"]
        self.elapsed = ev["elapsed_s"]
        t = ev["type"]
        team = ev["team"]["id"]
        pid = ev["player"]["id"] if ev.get("player") else None
        ps = self.players.get(pid) if pid else None

        if self._last_poss is not None:
            self.possession_time[self._last_poss[0]] += max(0.0, self.elapsed - self._last_poss[1])
        self._last_poss = (ev["possession_team"]["id"], self.elapsed)

        if "freeze_frame" in ev:
            out += self._track_movement(ev)

        if t in ON_BALL and ps:
            ps.touches += 1
            swing = self.goal_swing(team)
            dec = self.engine.evaluate(ev, goal_wp_swing=swing)
            if dec:
                ps.decisions += 1
                ps.decision_value += dec["decision_value"]
                ps.best_decisions += int(dec["rank"] == 1)
                out += self._decision_insight(ev, dec, swing)
            if t == "Pass":
                out += self._pass(ev, ps, dec)
            elif t == "Carry":
                self._credit_move(ev, ps, ev["location"], ev["carry"]["end_location"])
            elif t == "Shot":
                out += self._shot(ev, ps, dec)
        elif t == "Foul Committed" and (ev.get("foul_committed") or {}).get("card") in ("Red Card", "Second Yellow"):
            before = self.win_probability()
            self.reds[team] += 1
            after = self.win_probability()
            out.append(self._insight(ev, "red_card", 0.8, {"wp_before": before, "wp_after": after}))
        elif t == "Substitution":
            out.append(self._insight(ev, "substitution", 0.15, {"replacement": ev["substitution"]["replacement"]["name"]}))

        if not self.wp_timeline or self.wp_timeline[-1]["minute"] != self.minute:
            self.wp_timeline.append({"minute": self.minute, **{k: round(v, 4) for k, v in self.win_probability().items()}})
        out += self._momentum_check(ev)
        self.insights += out
        return out

    # ---------------------------------------------------------- handlers
    def _credit_move(self, ev, ps: PlayerStats, start, end, ok=True):
        if not ok:
            return 0.0
        gain = self.engine.xt(end) - self.engine.xt(start)
        ps.xt_added += gain
        self.team[ps.team_id]["xt_added"] += gain
        self.xt_window.append((self.elapsed, ps.team_id, gain))
        return gain

    def _pass(self, ev, ps: PlayerStats, dec) -> list[dict]:
        p = ev["pass"]
        ok = p["outcome"] == "Complete"
        ps.passes += 1
        ps.passes_completed += int(ok)
        self.team[ps.team_id]["passes"] += 1
        self.team[ps.team_id]["passes_completed"] += int(ok)
        gain = self._credit_move(ev, ps, ev["location"], p["end_location"], ok)
        out = []
        difficulty = dec["chosen"]["detail"]["difficulty"] if dec else None
        if ok and difficulty is not None and difficulty >= 60:
            ps.difficult_completed += 1
            out.append(self._insight(ev, "difficult_pass", 0.25 + difficulty / 400 + max(0.0, gain) * 2, {
                "difficulty": difficulty, "length_m": p["length_m"], "ball_speed_kmh": p["ball_speed_kmh"],
                "recipient": p["recipient"]["name"], "xt_added": round(gain, 4),
                "through_ball": p["through_ball"], "cross": p["cross"]}))
        if ok and ps.passes_completed in (50, 100):
            out.append(self._insight(ev, "milestone", 0.15, {"what": f"{ps.passes_completed} completed passes",
                                                               "pass_completion": round(ps.passes_completed / ps.passes, 3)}))
        return out

    def _shot(self, ev, ps: PlayerStats, dec) -> list[dict]:
        s = ev["shot"]
        xg = dec["chosen"]["xg"] if dec else 0.0
        team = ps.team_id
        ps.shots += 1
        ps.xg += xg
        self.team[team]["shots"] += 1
        self.team[team]["xg"] += xg
        self.xg_log.append((self.minute, team, xg))
        facts = {"xg": round(xg, 3), "outcome": s["outcome"], "body_part": s["body_part"],
                 "shot_speed_kmh": s["shot_speed_kmh"], "distance_m": dec["chosen"]["detail"]["distance_m"] if dec else None,
                 "type": s["type"]}
        if s["outcome"] == "Goal":
            # the event already carries the updated score; compute WP before and after
            before = self.win_probability()
            self.score[team] += 1
            after = self.win_probability()
            ps.goals += 1
            xpts = lambda wp: 3 * self.team_win(team, wp) + (wp["draw"])  # noqa: E731
            facts.update(wp_before=_r(before), wp_after=_r(after),
                         wp_swing=round(self.team_win(team, after) - self.team_win(team, before), 4),
                         xpts_swing=round(xpts(after) - xpts(before), 3),
                         situation=_situation(self.score[team] - 1, self.score[self.other(team)]),
                         score={"home": self.score[self.home], "away": self.score[self.away]},
                         player_goals=ps.goals)
            return [self._insight(ev, "goal", 1.0, facts)]
        if xg >= 0.3:
            facts["wp_lost"] = round(xg * self.goal_swing(team), 4)
            return [self._insight(ev, "big_chance_missed", 0.65 + 0.3 * xg, facts)]
        return [self._insight(ev, "shot", 0.15 + xg, facts)]

    def _decision_insight(self, ev, dec, swing) -> list[dict]:
        dv = dec["decision_value"]
        best = dec["best_alternative"]
        if best is None:
            return []
        speculative_shot = dec["chosen"]["kind"] == "shot" and dv <= -0.025 and best["kind"] == "pass"
        if (dv <= -0.05 and best["ev"] >= 0.06) or speculative_shot:
            imp = min(0.9, 0.35 + abs(dv) * 6)
            return [self._insight(ev, "missed_option", imp, {
                "chosen": dec["chosen"], "better_option": best, "decision_value": dv,
                "decision_wp": dec.get("decision_wp"), "n_options": dec["n_options"],
                "rank": dec["rank"], "goal_wp_swing": round(swing, 4)})]
        # a "great decision" is finding the best pass when others were on, not just shooting from close range
        if (dec["rank"] == 1 and dec["chosen"]["kind"] == "pass" and best["kind"] == "pass"
                and dv >= 0.02 and dec["chosen"]["ev"] >= 0.04):
            return [self._insight(ev, "great_decision", min(0.7, 0.3 + dv * 5), {
                "chosen": dec["chosen"], "next_best": best, "decision_value": dv,
                "decision_wp": dec.get("decision_wp"), "n_options": dec["n_options"]})]
        return []

    def _track_movement(self, ev) -> list[dict]:
        out = []
        actor_home = ev["team"]["id"] == self.home
        for f in ev["freeze_frame"]:
            ps = self.players.get(f["player_id"])
            if not ps:
                continue
            x, y = f["location"]
            # back to one fixed frame (home attacking right) so positions are comparable over time
            pos = (x, y) if actor_home else (120 - x, 80 - y)
            if ps._last_pos is not None and self.elapsed > (ps._last_t or 0):
                ps.distance_m += dist_m(pos, ps._last_pos)
            ps._last_pos, ps._last_t = pos, self.elapsed
            kmh = f["speed_ms"] * 3.6
            if kmh > ps.top_speed_kmh:
                first_sprint = ps.top_speed_kmh < SPRINT_KMH <= kmh
                ps.top_speed_kmh = kmh
                if first_sprint:
                    out.append(self._insight(ev, "sprint", 0.2, {"player": ps.name, "player_id": ps.id,
                                                                  "speed_kmh": round(kmh, 1),
                                                                  "team_id": ps.team_id}))
        return out

    def _momentum_check(self, ev) -> list[dict]:
        if ev["type"] not in ON_BALL or self.minute < 10:
            return []
        m = self.momentum()
        leader = self.home if m["home_share"] >= 0.7 else (self.away if m["away_share"] >= 0.7 else None)
        if leader and leader != self._momentum_leader:
            self._momentum_leader = leader
            name = self.meta["home"]["name"] if leader == self.home else self.meta["away"]["name"]
            return [self._insight(ev, "momentum_shift", 0.5, {"team_id": leader, "team_name": name, **m})]
        if leader is None and 0.4 <= m["home_share"] <= 0.6:
            self._momentum_leader = None
        return []

    def _insight(self, ev, kind: str, importance: float, facts: dict) -> dict:
        return {
            "id": f"ins-{ev['id']}-{kind}",
            "event_id": ev["id"], "kind": kind, "importance": round(float(min(1.0, importance)), 3),
            "minute": ev["minute"], "second": ev["second"], "elapsed_s": ev["elapsed_s"],
            "team": ev["team"], "team_is_home": ev["team"]["id"] == self.home, "player": ev.get("player"),
            "score": {"home": self.score[self.home], "away": self.score[self.away]},
            "location": ev.get("location"),
            "facts": facts,
        }


def _situation(before_for: int, before_against: int) -> str:
    """What a goal did to the scoreline, from the scorer's point of view."""
    diff = before_for - before_against
    return {-1: "equaliser", 0: "go_ahead", 1: "extend_lead"}.get(diff, "consolation" if diff < -1 else "extend_lead")


def _r(d: dict) -> dict:
    return {k: round(v, 4) for k, v in d.items()}


def analyse_match(bundle: ModelBundle, meta: dict, events: list[dict]) -> LiveAnalyst:
    la = LiveAnalyst(bundle, meta)
    for ev in events:
        la.process(ev)
    return la
