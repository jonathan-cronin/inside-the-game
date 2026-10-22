"""Agent-based synthetic match engine.

Each on-ball moment works like this:
  1. Both teams move towards a shape that depends on ball position, phase,
     team style, score and fatigue. Movement is limited by each player's top
     speed, so pressing teams really do cover more ground and tire.
  2. The ball carrier sees a set of options (every pass, through balls,
     crosses, carries, take-ons, shots, clearances), values each with the
     hidden truth models, and picks one with a softmax whose temperature
     depends on the player's vision. So players sometimes make the wrong
     choice, and those are the moments a counterfactual engine can explain.
  3. The outcome is sampled, and events are emitted with a freeze frame of
     all 22 players (positions and speeds) in the actor's frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .pitch import (BOX_X, GOAL, X_M, Y_M, clip_pitch, dist_m, dists_m, in_box, mirror,
                    pass_success_truth, segment_dists, shot_blockers, sigmoid, xg_truth, xt_truth)
from .teams import FORMATIONS, ROLE_GROUP, Player, Team

CORNER_SPOTS = [(110, 36), (109, 44), (111, 40), (107, 33), (107, 47), (104, 40)]


@dataclass
class PState:
    pos: np.ndarray            # absolute frame: home attacks towards x = 120
    stamina: float = 1.0
    distance_m: float = 0.0
    speed_ms: float = 0.0
    yellow: int = 0
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class Match:
    meta: dict
    events: list[dict]
    truth: dict  # hidden generator information, for evaluation only


class MatchSimulator:
    def __init__(self, home: Team, away: Team, seed: int | None = None, match_id: str | None = None,
                 kickoff: str = "2026-10-24T15:00:00Z", include_truth: bool = False):
        assert home.id != away.id
        self.home, self.away = home, away
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.match_id = match_id or f"SYN-{home.id}-{away.id}-{seed if seed is not None else 0}"
        self.kickoff_time = kickoff
        self.include_truth = include_truth
        self.is_home = {home.id: True, away.id: False}
        self.on_pitch: dict[str, list[Player]] = {home.id: list(home.starters), away.id: list(away.starters)}
        self.bench: dict[str, list[Player]] = {home.id: list(home.bench), away.id: list(away.bench)}
        self.slot: dict[str, str] = {p.id: p.role for p in home.starters + away.starters}
        self.st: dict[str, PState] = {p.id: PState(pos=np.array([60.0, 40.0])) for p in home.all_players() + away.all_players()}
        self.score = {home.id: 0, away.id: 0}
        self.events: list[dict] = []
        self.idx = 0
        self.period = 1
        self.t = 0.0
        self.period_offset = 0.0
        self.poss_id = 0
        self.poss_team: Team = home
        self.carrier: Player = home.starters[-1]
        self.ball = np.array([60.0, 40.0])
        self.pattern = "From Kick Off"
        self.restart: str | None = None
        self.mode = "open"
        self.force_header = False
        self.last_change_t = -99.0
        self.subs_done = {home.id: 0, away.id: 0}
        self.sub_minutes = {tid: sorted(self.rng.integers(56, 86, size=3).tolist()) for tid in (home.id, away.id)}
        self.truth_log: list[dict] = []
        self.press_roll = 1.0
        self.by_id: dict[str, Player] = {p.id: p for p in home.all_players() + away.all_players()}

    # ------------------------------------------------------------------ utils
    def other(self, team: Team) -> Team:
        return self.away if team is self.home else self.home

    def f(self, team: Team, p) -> np.ndarray:
        """Convert between the absolute frame and team's attacking frame (an involution)."""
        p = np.asarray(p, float)
        return p.copy() if self.is_home[team.id] else np.array([120.0 - p[0], 80.0 - p[1]])

    def a(self, player: Player, attr: str) -> float:
        v = self.home_adv(player) + player.attrs[attr]
        if attr not in ("aerial", "goalkeeping"):
            v -= 0.12 * (1.0 - self.st[player.id].stamina)
        return v

    def home_adv(self, player: Player) -> float:
        return 0.025 if self.is_home[player.team_id] else 0.0

    def team_of(self, player: Player) -> Team:
        return self.home if player.team_id == self.home.id else self.away

    @property
    def minute(self) -> int:
        return int(self.t // 60) + (45 if self.period == 2 else 0)

    def style(self, team: Team) -> dict:
        s = dict(team.style)
        diff = self.score[team.id] - self.score[self.other(team).id]
        m = self.minute
        if diff < 0 and m >= 60:
            k = min(1.0, (m - 60) / 25)
            s["directness"] += 0.25 * k
            s["block_height"] += 0.2 * k
            s["press"] += 0.2 * k
            s["shoot_bias"] = 0.006 * k
        elif diff > 0 and m >= 65:
            k = min(1.0, (m - 65) / 20)
            s["block_height"] -= 0.25 * k
            s["possession"] += 0.15 * k
            s["press"] -= 0.15 * k
        if len(self.on_pitch[team.id]) < 11:
            s["press"] -= 0.15
            s["block_height"] -= 0.15
        for k_ in ("possession", "directness", "press", "block_height", "width"):
            s[k_] = float(np.clip(s[k_], 0.0, 1.0))
        s.setdefault("shoot_bias", 0.0)
        return s

    def positions(self, team: Team, frame_team: Team) -> tuple[list[Player], np.ndarray]:
        players = self.on_pitch[team.id]
        return players, np.array([self.f(frame_team, self.st[p.id].pos) for p in players])

    def outfield(self, team: Team, frame_team: Team):
        players, pos = self.positions(team, frame_team)
        keep = [i for i, p in enumerate(players) if p.role != "GK"]
        return [players[i] for i in keep], pos[keep]

    def keeper(self, team: Team) -> Player | None:
        for p in self.on_pitch[team.id]:
            if p.role == "GK":
                return p
        return None

    # --------------------------------------------------------------- events
    def emit(self, etype: str, team: Team, player: Player | None, loc_abs=None, data: dict | None = None,
             under_pressure: bool = False, freeze: bool = False, duration: float = 0.0,
             truth: dict | None = None) -> dict:
        loc = None if loc_abs is None else self.f(team, loc_abs)
        ev = {
            "id": f"{self.match_id}-{self.idx:05d}",
            "index": self.idx,
            "match_id": self.match_id,
            "period": self.period,
            "timestamp": _fmt(self.t),
            "minute": self.minute,
            "second": int(self.t % 60),
            "elapsed_s": round(self.period_offset + self.t, 2),
            "type": etype,
            "team": team.public(),
            "player": None if player is None else {"id": player.id, "name": player.name,
                                                   "position": self.slot.get(player.id, player.role)},
            "possession": self.poss_id,
            "possession_team": self.poss_team.public(),
            "play_pattern": self.pattern,
            "location": None if loc is None else [round(float(loc[0]), 1), round(float(loc[1]), 1)],
            "under_pressure": bool(under_pressure),
            "duration": round(duration, 2),
            "score": {"home": self.score[self.home.id], "away": self.score[self.away.id]},
        }
        if data:
            ev[etype.lower().replace(" ", "_")] = data
        if freeze:
            ev["freeze_frame"] = self.freeze_frame(team, player)
        self.events.append(ev)
        if truth is not None:
            self.truth_log.append({"event_id": ev["id"], **truth})
            if self.include_truth:
                ev["_truth"] = truth
        self.idx += 1
        return ev

    def freeze_frame(self, team: Team, actor: Player | None) -> list[dict]:
        out = []
        for tm in (team, self.other(team)):
            for p in self.on_pitch[tm.id]:
                loc = self.f(team, self.st[p.id].pos)
                out.append({"player_id": p.id, "teammate": tm is team, "actor": actor is not None and p.id == actor.id,
                            "keeper": p.role == "GK", "location": [round(float(loc[0]), 1), round(float(loc[1]), 1)],
                            "speed_ms": round(self.st[p.id].speed_ms, 2)})
        return out

    # ------------------------------------------------------------- movement
    def targets(self, att: Team, dfn: Team, mode: str = "open") -> dict[str, np.ndarray]:
        """Target positions (absolute frame) for every player on the pitch."""
        b = self.f(att, self.ball)
        sa, sd = self.style(att), self.style(dfn)
        tg: dict[str, np.ndarray] = {}

        # defending team, in its own frame (own goal at x = 0)
        bd = mirror(b)
        line_base = 18 + sd["block_height"] * 42          # deepest line, own frame
        line = min(line_base, max(bd[0] - 7, 16))
        danger = float(np.clip((40 - bd[0]) / 25, 0, 1))  # 0 in midfield -> 1 when ball is near own box
        compact = (24 + 8 * sd["block_height"]) * (1 - 0.4 * danger)
        width_d = 80 * (0.55 + 0.1 * sd["width"]) * (1 - 0.3 * danger)
        for p in self.on_pitch[dfn.id]:
            u, v = FORMATIONS[dfn.formation][self.slot[p.id]]
            if p.role == "GK":
                x, y = float(np.clip(line * 0.22, 2.5, 16)), 40 + (bd[1] - 40) * 0.2
            else:
                x = line + u * compact
                y = 40 + (v - 0.5) * width_d + (bd[1] - 40) * 0.45
            tg[p.id] = np.array([x, y])
        # pressing: nearest outfield defenders step out to the ball
        out_p, out_pos = self.outfield(dfn, dfn)
        if len(out_p):
            d = dists_m(bd, out_pos)
            order = np.argsort(d)
            counter = (self.t - self.last_change_t) < 5.0 and self.pattern != "From Kick Off"
            press = sd["press"] + (0.25 if counter else 0.0)
            goal_dir = np.array([0.0, 40.0]) - bd
            goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-9)
            engage = 0.25 + 0.5 * press + 0.35 * danger
            p1 = out_p[order[0]]
            if self.press_roll < engage:
                tg[p1.id] = bd + goal_dir * (2.5 + (1 - press) * 4.5) * (1 - 0.6 * danger)
                if len(order) > 1 and self.press_roll < engage * max(press * 0.5, danger * 0.7):
                    p2 = out_p[order[1]]
                    tg[p2.id] = 0.6 * tg[p2.id] + 0.4 * (bd + goal_dir * 5.0)
            else:
                tg[p1.id] = 0.7 * tg[p1.id] + 0.3 * (bd + goal_dir * 8.0)
        offside_line_att = 120 - min(tg[p.id][0] for p in self.on_pitch[dfn.id] if p.role != "GK")

        # attacking team, in its own frame
        back = float(np.clip(b[0] - 40 + 10 * sa["block_height"], 16, 64))
        length = 52.0
        width_a = 80 * (0.62 + 0.25 * sa["width"])
        for p in self.on_pitch[att.id]:
            role = self.slot[p.id]
            u, v = FORMATIONS[att.formation][role]
            if p.role == "GK":
                x, y = float(np.clip(4 + b[0] * 0.12, 3, 18)), 40.0
            else:
                if ROLE_GROUP[role] == "FB":
                    u += 0.2 + 0.15 * sa["width"]
                x = back + u * length
                y = 40 + (v - 0.5) * width_a + (b[1] - 40) * 0.2
                x = min(x, max(offside_line_att - 0.8, b[0]))
            tg[p.id] = np.array([x, y])

        # man-marking: when the ball is near goal, defenders pick up attackers in and around the box
        if danger > 0.25 and mode == "open":
            pressers = set()
            if len(out_p):
                pressers.add(out_p[order[0]].id)
            free_def = [p for p in self.on_pitch[dfn.id] if p.role != "GK" and p.id not in pressers]
            threats = [p for p in self.on_pitch[att.id] if p.id != self.carrier.id and p.role != "GK"
                       and self.f(att, self.st[p.id].pos)[0] > 96]
            threats.sort(key=lambda p: dist_m(self.f(att, self.st[p.id].pos), GOAL))
            for th in threats:
                if not free_def:
                    break
                tp = self.f(att, self.st[th.id].pos)       # attacker, attacking frame
                tp_own = mirror(tp)                         # same point, defenders' frame
                j = int(np.argmin([dist_m(tg[d.id], tp_own) for d in free_def]))
                d = free_def.pop(j)
                goal_side = np.array([0.0, 40.0]) - tp_own
                goal_side = goal_side / (np.linalg.norm(goal_side) + 1e-9)
                tightness = 1.0 + 1.5 * (1 - danger)
                tg[d.id] = (1 - danger) * tg[d.id] + danger * (tp_own + goal_side * tightness)

        if mode in ("corner", "box_setpiece"):
            self._setpiece_targets(att, dfn, tg, b)
        elif mode == "kickoff":
            for p in self.on_pitch[att.id]:
                tg[p.id][0] = min(tg[p.id][0], 58.5)
            for p in self.on_pitch[dfn.id]:
                tg[p.id][0] = min(tg[p.id][0], 58.5)  # own frame for defenders
        elif mode == "penalty":
            for p in self.on_pitch[att.id] + self.on_pitch[dfn.id]:
                if p.role != "GK":
                    tg[p.id][0] = min(tg[p.id][0], 98.0) if p in self.on_pitch[att.id] else max(tg[p.id][0], 22.0)

        # convert to absolute frame, add per-player noise
        out = {}
        for p in self.on_pitch[att.id]:
            out[p.id] = self.f(att, clip_pitch(tg[p.id] + self.st[p.id].offset))
        for p in self.on_pitch[dfn.id]:
            out[p.id] = self.f(att, mirror(clip_pitch(tg[p.id] + self.st[p.id].offset)))
        out[self.carrier.id] = self.ball.copy()
        return out

    def _setpiece_targets(self, att: Team, dfn: Team, tg: dict, b: np.ndarray):
        att_out = [p for p in self.on_pitch[att.id] if p.role != "GK" and p.id != self.carrier.id]
        att_out.sort(key=lambda p: -p.attrs["aerial"])
        spots = [np.array(s, float) for s in CORNER_SPOTS]
        for p, s in zip(att_out[:5], spots):
            tg[p.id] = s + self.rng.normal(0, 1.0, 2)
        for p, s in zip(att_out[5:7], [np.array([98.0, 30.0]), np.array([98.0, 50.0])]):
            tg[p.id] = s
        for p in att_out[7:]:
            tg[p.id] = np.array([62.0, 40.0 + self.rng.normal(0, 10)])
        dfn_out = [p for p in self.on_pitch[dfn.id] if p.role != "GK"]
        dfn_out.sort(key=lambda p: -p.attrs["aerial"])
        for i, p in enumerate(dfn_out):
            if i < len(spots):
                s = spots[i]
                tg[p.id] = mirror(s + np.array([1.5, 0.0])) + self.rng.normal(0, 0.8, 2)
            elif i < 9:
                tg[p.id] = mirror(np.array([100.0, 30.0 + 10 * (i - len(spots))]))
            else:
                tg[p.id] = np.array([45.0, 40.0])

    def move(self, dt: float, mode: str = "open", teleport: bool = False):
        if dt <= 0:
            return
        self.press_roll = self.rng.random()
        tg = self.targets(self.poss_team, self.other(self.poss_team), mode)
        for pid, target in tg.items():
            stt = self.st[pid]
            p = self._player(pid)
            stt.offset = 0.85 * stt.offset + self.rng.normal(0, 0.9, 2)
            vec = target - stt.pos
            dist_units = float(np.hypot(vec[0], vec[1]))
            if teleport:
                stt.pos = target.copy()
                stt.speed_ms = 0.0
                continue
            vmax_ms = (6.3 + 2.8 * p.attrs["pace"]) * (0.82 + 0.18 * stt.stamina)
            max_units = vmax_ms * dt / 0.86
            step = vec if dist_units <= max_units else vec * (max_units / dist_units)
            moved_m = math.hypot(step[0] * X_M, step[1] * Y_M)
            # off-ball movement between events is never perfectly straight
            moved_m += 0.15 * dt * (1.0 + 0.8 * self.rng.random())
            stt.pos = clip_pitch(stt.pos + step)
            stt.speed_ms = min(moved_m / dt, vmax_ms)
            stt.distance_m += moved_m
            endurance = 0.55 + 0.7 * p.attrs["stamina"]
            stt.stamina = max(0.35, stt.stamina - moved_m / (10500 * endurance) * 0.42)
        self.hold_onside()

    def hold_onside(self):
        """Attackers without the ball stay level with the second-last defender (or behind the ball)."""
        att, dfn = self.poss_team, self.other(self.poss_team)
        _, dpos = self.positions(dfn, att)
        if len(dpos) < 2:
            return
        line = float(np.sort(dpos[:, 0])[-2])  # second-last opponent, keeper included
        bx = self.f(att, self.ball)[0]
        limit = max(line, bx)
        for p in self.on_pitch[att.id]:
            if p.id == self.carrier.id:
                continue
            q = self.f(att, self.st[p.id].pos)
            if q[0] > limit:
                q[0] = limit - 0.3
                self.st[p.id].pos = self.f(att, q)

    def _player(self, pid: str) -> Player:
        return self.by_id[pid]

    # -------------------------------------------------------------- control
    def set_possession(self, team: Team, carrier: Player, ball_abs, pattern: str | None = None):
        if team is not self.poss_team:
            self.poss_id += 1
            self.last_change_t = self.t
        self.poss_team = team
        self.carrier = carrier
        self.ball = clip_pitch(np.asarray(ball_abs, float))
        self.st[carrier.id].pos = self.ball.copy()
        if pattern:
            self.pattern = pattern

    def advance(self, dt: float, mode: str = "open"):
        self.t += dt
        self.move(dt, mode)

    # -------------------------------------------------------------- the match
    def run(self) -> Match:
        added = {1: 60 * int(self.rng.integers(1, 4)) + int(self.rng.integers(0, 50)),
                 2: 60 * int(self.rng.integers(3, 7)) + int(self.rng.integers(0, 50))}
        first = self.home if self.rng.random() < 0.5 else self.away
        for tm in (self.home, self.away):
            self.emit("Starting XI", tm, None, data={
                "formation": tm.formation,
                "lineup": [{**p.public(), "position": self.slot[p.id]} for p in self.on_pitch[tm.id]],
                "bench": [p.public() for p in self.bench[tm.id]],
            })
        for period in (1, 2):
            self.period = period
            self.t = 0.0
            self.emit("Half Start", self.home, None)
            self.kickoff(first if period == 1 else self.other(first))
            length = 45 * 60 + added[period]
            guard = 0
            while self.t < length and guard < 20000:
                self.step()
                guard += 1
            self.emit("Half End", self.home, None)
            self.emit("Half End", self.away, None)
            self.period_offset += self.t
        meta = {
            "match_id": self.match_id,
            "competition": "Synthetic League",
            "kickoff": self.kickoff_time,
            "seed": self.seed,
            "home": {**self.home.public(), "short": self.home.short, "formation": self.home.formation,
                     "colours": list(self.home.colours)},
            "away": {**self.away.public(), "short": self.away.short, "formation": self.away.formation,
                     "colours": list(self.away.colours)},
            "final_score": {"home": self.score[self.home.id], "away": self.score[self.away.id]},
            "players": [{**p.public(), "team_id": p.team_id} for p in self.home.all_players() + self.away.all_players()],
            "coordinate_system": "StatsBomb-style 120x80; every location is in the acting team's frame, attacking towards x=120",
        }
        truth = {
            "events": self.truth_log,
            "players": {p.id: {"attrs": p.attrs, "distance_m": round(self.st[p.id].distance_m, 1),
                               "final_stamina": round(self.st[p.id].stamina, 3)}
                        for p in self.home.all_players() + self.away.all_players()},
            "styles": {self.home.id: self.home.style, self.away.id: self.away.style},
        }
        return Match(meta=meta, events=self.events, truth=truth)

    def kickoff(self, team: Team):
        self.mode = "kickoff"
        self.restart = "kickoff"
        kicker = next((p for p in self.on_pitch[team.id] if ROLE_GROUP[self.slot[p.id]] == "ST"),
                      self.on_pitch[team.id][-1])
        self.set_possession(team, kicker, np.array([60.0, 40.0]), "From Kick Off")
        self.last_change_t = -99.0
        self.move(1.0, "kickoff", teleport=True)
        for p in self.on_pitch[team.id] + self.on_pitch[self.other(team).id]:
            self.st[p.id].speed_ms = 0.0

    # ------------------------------------------------------------------ step
    def step(self):
        if self.restart:
            self.maybe_substitute()
        T, D, c = self.poss_team, self.other(self.poss_team), self.carrier
        b = self.f(T, self.ball)
        d_out, d_pos = self.outfield(D, T)
        press_d = dists_m(b, d_pos) if len(d_pos) else np.array([99.0])
        pi = int(np.argmin(press_d))
        press_m = float(press_d[pi])
        presser = d_out[pi]
        under_pressure = press_m < 3.5 and self.restart is None

        if under_pressure and self.rng.random() < 0.75:
            counter = (self.t - self.last_change_t) < 5.0
            self.emit("Pressure", D, presser, self.st[presser.id].pos,
                      data={"counterpress": bool(counter)}, duration=0.5)
            box_factor = 0.08 if in_box(b) else 1.0
            if self.rng.random() < 0.014 * (1 + D.style["press"]) * box_factor:
                self.foul(D, presser, T, c)
                return

        if self.force_header:
            self.force_header = False
            if self.rng.random() < 0.65:
                self.shoot(T, D, c, b, press_m, header=True)
                return

        restart = self.restart
        self.restart = None
        if restart == "penalty":
            self.shoot(T, D, c, b, 10.0, set_piece="penalty")
            return

        opts = self.options(T, D, c, b, press_m, restart)
        choice = self.choose(c, T, opts, restart)
        truth = {"chosen": choice["kind"], "ev": round(choice["ev"], 5),
                 "best_ev": round(max(o["ev"] for o in opts), 5), "n_options": len(opts),
                 "p_success": round(choice.get("p", 1.0), 4)}
        mode = self.mode
        self.mode = "open"
        k = choice["kind"]
        if k in ("pass", "through_ball", "cross"):
            self.do_pass(T, D, c, b, choice, under_pressure, truth, restart, mode)
        elif k == "carry":
            self.do_carry(T, D, c, b, choice, under_pressure, truth)
        elif k == "dribble":
            self.do_dribble(T, D, c, b, choice, under_pressure, truth, presser)
        elif k == "shot":
            self.shoot(T, D, c, b, press_m, truth=truth, set_piece="free_kick" if restart == "free_kick" else None)
        elif k == "clearance":
            self.do_clearance(T, D, c, b, choice, under_pressure, truth)

    # --------------------------------------------------------------- options
    def options(self, T: Team, D: Team, c: Player, b: np.ndarray, press_m: float, restart: str | None) -> list[dict]:
        mates, m_pos = self.positions(T, T)
        d_out, d_pos = self.outfield(D, T)
        _, d_all = self.positions(D, T)  # includes the keeper
        st_c = self.st[c.id].stamina
        opts: list[dict] = []
        max_len = {"throw_in": 28, "kickoff": 20}.get(restart, 65)

        keep_value = 0.012 + 0.012 * T.style["possession"]  # what losing the ball forfeits

        def turnover_cost(pt):
            return xt_truth(mirror(pt)) + keep_value

        # passes to feet
        for r, rp in zip(mates, m_pos):
            if r.id == c.id:
                continue
            if restart == "kickoff" and rp[0] > 60:
                continue
            if restart in ("corner", "box_free_kick") and in_box(rp):
                continue  # deliveries into the box are modelled as crosses
            length = dist_m(b, rp)
            if length < 3 or length > max_len:
                continue
            lane, t = segment_dists(b, rp, d_all)
            lane = np.where((t > 0.03) & (t < 0.97), lane, 99.0)
            li = int(np.argmin(lane)) if len(lane) else 0
            lane_m = float(lane[li]) if len(lane) else 99.0
            mark_m = float(dists_m(rp, d_all).min()) if len(d_all) else 99.0
            high = length > 32 or (restart == "goal_kick" and length > 28)
            p = pass_success_truth(length_m=length, lane_m=lane_m, mark_m=mark_m,
                                   press_m=press_m if restart is None else 12.0,
                                   passing=self.a(c, "passing"), stamina=st_c,
                                   forward_units=rp[0] - b[0], high=high)
            if restart == "throw_in":
                p = min(0.97, p + 0.05)
            value = xt_truth(rp) * (0.7 + 0.3 * min(1.0, mark_m / 6.0))
            ip = b + (rp - b) * (float(t[li]) if len(t) and lane_m < 4 else 0.5)
            ev = p * value - (1 - p) * turnover_cost(ip)
            opts.append(dict(kind="pass", target=r, end=rp.copy(), p=p, ev=ev, length=length, high=high,
                             lane_m=lane_m, mark_m=mark_m, ip=ip))

        # through balls into space behind the line
        if restart in (None, "free_kick") and len(d_pos):
            line_x = float(d_pos[:, 0].max())
            for r, rp in zip(mates, m_pos):
                if r.id == c.id or r.role == "GK" or ROLE_GROUP[self.slot[r.id]] not in ("ST", "W", "AM", "FB"):
                    continue
                if line_x > 112 or rp[0] < b[0] - 5 or rp[0] < 70:
                    continue
                end = clip_pitch(np.array([min(line_x + 5 + 4 * r.attrs["pace"], 109.0),
                                           rp[1] + (40 - rp[1]) * 0.25]))
                length = dist_m(b, end)
                if length < 8 or length > 45:
                    continue
                lane, t = segment_dists(b, end, d_all)
                lane = np.where((t > 0.03) & (t < 0.97), lane, 99.0)
                lane_m = float(lane.min())
                p_pass = pass_success_truth(length_m=length, lane_m=lane_m, mark_m=8.0, press_m=press_m,
                                            passing=self.a(c, "passing") + 0.4 * (self.a(c, "vision") - 0.65),
                                            stamina=st_c, forward_units=end[0] - b[0], through=True)
                race = float(dists_m(end, d_all).min()) - dist_m(rp, end) * (1.05 - 0.25 * r.attrs["pace"])
                p = p_pass * sigmoid(race / 2.5)
                ev = p * xt_truth(end) - (1 - p) * turnover_cost(end) * 0.6
                opts.append(dict(kind="through_ball", target=r, end=end, p=p, ev=ev, length=length,
                                 high=False, lane_m=lane_m, mark_m=8.0, ip=end))

        # crosses from wide areas (and set-piece deliveries)
        wide = b[0] >= 96 and (b[1] < 20 or b[1] > 60)
        if wide or restart in ("corner", "box_free_kick"):
            for r, rp in zip(mates, m_pos):
                if r.id == c.id or not in_box(rp):
                    continue
                near = dists_m(rp, d_pos)
                nd = int(np.argmin(near)) if len(near) else 0
                best_def = d_out[nd].attrs["aerial"] if len(near) else 0.5
                crowd = int((near < 3.0).sum()) if len(near) else 0
                p = float(np.clip(0.32 + 0.5 * (r.attrs["aerial"] - best_def) + 0.25 * (self.a(c, "passing") - 0.65)
                                  - 0.06 * crowd + (0.06 if restart else 0.0), 0.05, 0.7))
                hx = xg_truth(rp, header=True, pressure_m=float(near[nd]) if len(near) else 10.0,
                              blockers=shot_blockers(rp, d_pos), finishing=r.attrs["finishing"])
                ev = p * (0.65 * hx + 0.35 * xt_truth(rp)) - (1 - p) * 0.006
                opts.append(dict(kind="cross", target=r, end=rp.copy(), p=p, ev=ev, length=dist_m(b, rp),
                                 high=True, lane_m=2.0, mark_m=float(near[nd]) if len(near) else 9.0, ip=rp))

        if restart in ("throw_in", "goal_kick", "kickoff", "corner"):
            return opts or self._fallback_option(b)

        # shots
        dgoal = dist_m(b, GOAL)
        if dgoal < 36 and c.role != "GK" and (restart is None or restart == "free_kick"):
            blockers = shot_blockers(b, d_pos)
            gk = self.keeper(D)
            xg = xg_truth(b, pressure_m=press_m, blockers=blockers, finishing=self.a(c, "finishing"),
                          keeper=gk.attrs["goalkeeping"] if gk else 0.3,
                          set_piece="free_kick" if restart == "free_kick" else None)
            # a missed shot usually hands the ball back, so it forfeits possession too
            ev = xg - (1 - xg) * keep_value * 0.5
            opts.append(dict(kind="shot", p=1.0, ev=ev, xg=xg, blockers=blockers))

        if c.role != "GK" and restart is None:
            # carries
            ahead = [np.array([1.0, 0.0]), (GOAL - b) / (np.linalg.norm(GOAL - b) + 1e-9)]
            if b[1] < 30 or b[1] > 50:
                ahead.append(np.array([0.7, 0.7 * (1 if b[1] < 40 else -1)]))
            for dvec in ahead:
                dvec = dvec / np.linalg.norm(dvec)
                rel = d_all - b
                norm = np.linalg.norm(rel, axis=1) + 1e-9
                in_cone = (rel @ dvec) / norm > 0.6        # within ~53 degrees of the carry direction
                space_m = float(dists_m(b, d_all[in_cone]).min()) if in_cone.any() else 30.0
                L = float(np.clip(space_m * 0.5, 3.0, 12.0 if b[0] < 96 else 6.0)) / X_M
                end = clip_pitch(b + dvec * L)
                end[0] = min(end[0], 112.0)
                path, _ = segment_dists(b, end, d_all)
                blockers = int((path < 2.0).sum()) if len(path) else 0
                crowd = int((dists_m(end, d_all) < 5.0).sum()) if len(d_all) else 0
                p = float(np.clip(0.96 - 0.3 * math.exp(-press_m / 2.5) - 0.18 * blockers - 0.08 * crowd
                                  - (0.2 if in_box(end) else 0.0)
                                  + 0.2 * (self.a(c, "dribbling") - 0.6), 0.15, 0.99))
                ev = p * xt_truth(end) - (1 - p) * turnover_cost(b + dvec * 2)
                opts.append(dict(kind="carry", end=end, p=p, ev=ev, length=dist_m(b, end)))
            # take-on
            if press_m < 4.0 and len(d_pos):
                p = float(np.clip(0.42 + 0.7 * (self.a(c, "dribbling") - 0.6), 0.12, 0.75))
                end = clip_pitch(b + np.array([5.0, self.rng.normal(0, 3)]))
                end[0] = min(end[0], 110.0)
                if b[0] > 96:
                    p *= 0.6  # far harder to beat a man in a crowded box
                ev = p * xt_truth(end) - (1 - p) * turnover_cost(b)
                opts.append(dict(kind="dribble", end=end, p=p, ev=ev, length=dist_m(b, end)))
            # clearance under pressure deep in own half
            if b[0] < 35 and press_m < 3.5:
                land = clip_pitch(np.array([b[0] + 45 + self.rng.normal(0, 8), 40 + self.rng.normal(0, 18)]))
                ev = 0.4 * xt_truth(land) - 0.6 * turnover_cost(land)
                opts.append(dict(kind="clearance", end=land, p=0.4, ev=ev, length=dist_m(b, land)))
        return opts or self._fallback_option(b)

    def _fallback_option(self, b):
        land = clip_pitch(np.array([b[0] + 35, 40 + self.rng.normal(0, 15)]))
        return [dict(kind="clearance", end=land, p=0.4, ev=0.0, length=dist_m(b, land))]

    def choose(self, c: Player, T: Team, opts: list[dict], restart: str | None) -> dict:
        s = self.style(T)
        temp = float(np.clip(0.006 * (1.55 - self.a(c, "vision")), 0.0025, 0.008))
        utils = []
        for o in opts:
            u = o["ev"]
            if o["kind"] in ("pass", "through_ball"):
                fwd = (o["end"][0] - self.f(T, self.ball)[0]) / 40.0
                u += 0.008 * s["directness"] * float(np.clip(fwd, -1, 1))
                if o["length"] < 22 and o["p"] > 0.85:
                    u += 0.006 * s["possession"]
                if o["length"] > 35:
                    u += 0.006 * (s["directness"] - 0.5)
                if restart == "goal_kick":
                    u += 0.01 * (s["directness"] - 0.5) * (1 if o["length"] > 35 else -1)
            elif o["kind"] == "shot":
                u += 0.002 + s["shoot_bias"] + 0.003 * (c.attrs["finishing"] - 0.6)
                if dist_m(self.f(T, self.ball), GOAL) > 16.5:
                    u += 0.009 * c.attrs["finishing"]  # players back themselves from range
            elif o["kind"] == "carry":
                u -= 0.001
            utils.append(u)
        z = np.array(utils) / temp
        z -= z.max()
        w = np.exp(z)
        return opts[int(self.rng.choice(len(opts), p=w / w.sum()))]

    # --------------------------------------------------------------- actions
    def do_pass(self, T, D, c, b, o, under_pressure, truth, restart, mode):
        r: Player = o["target"]
        end = o["end"]
        success = self.rng.random() < o["p"]
        high = o["high"]
        speed_kmh = float(np.clip((48 if not high else 62) + 0.55 * o["length"] + self.rng.normal(0, 6), 25, 110))
        travel = o["length"] / (speed_kmh / 3.6)
        ptype = {"throw_in": "Throw-in", "goal_kick": "Goal Kick", "corner": "Corner", "free_kick": "Free Kick",
                 "box_free_kick": "Free Kick", "kickoff": "Kick Off"}.get(restart, "Open Play")
        data = {"recipient": {"id": r.id, "name": r.name}, "length_m": round(o["length"], 1),
                "angle": round(math.atan2(end[1] - b[1], end[0] - b[0]), 3),
                "height": "High Pass" if high else "Ground Pass", "type": ptype,
                "cross": o["kind"] == "cross", "through_ball": o["kind"] == "through_ball",
                "switch": bool(abs(end[1] - b[1]) > 40), "ball_speed_kmh": round(speed_kmh, 1)}
        if success:
            land = clip_pitch(end + self.rng.normal(0, 0.6, 2))
            data["end_location"] = _r(land)
            data["outcome"] = "Complete"
            self.emit("Pass", T, c, self.ball, data, under_pressure, freeze=True, duration=travel, truth=truth)
            self.set_possession(T, r, self.f(T, land))
            if o["kind"] == "cross":
                self.force_header = True
            self.advance(travel + self.rng.uniform(1.6, 4.4))
            return
        # failure: deflected behind, out of play, or won by a defender
        d_out, d_pos = self.outfield(D, T)
        if (o["kind"] == "cross" or end[0] > 100) and self.rng.random() < 0.2:
            data["end_location"] = _r(end)
            data["outcome"] = "Incomplete"
            self.emit("Pass", T, c, self.ball, data, under_pressure, freeze=True, duration=travel, truth=truth)
            blocker = d_out[int(np.argmin(dists_m(end, d_pos)))]
            self.emit("Clearance", D, blocker, self.f(T, end), {"end_location": None, "outcome": "Out for corner"})
            self.advance(travel)
            self.setup_restart("corner", T, self.f(T, np.array([120.0, 0.5 if end[1] < 40 else 79.5])))
            return
        if self.rng.random() < (0.35 if high else 0.16):
            over = clip_pitch(end + (end - b) / (np.linalg.norm(end - b) + 1e-9) * self.rng.uniform(4, 15), 0.0)
            data["end_location"] = _r(over)
            data["outcome"] = "Out"
            self.emit("Pass", T, c, self.ball, data, under_pressure, freeze=True, duration=travel, truth=truth)
            self.advance(travel)
            self.out_of_play(T, over)
            return
        lane, t = segment_dists(b, end, d_pos)
        lane = np.where(t > 0.03, lane, 99.0)
        di = int(np.argmin(lane))
        winner = d_out[di]
        ip = clip_pitch(b + (end - b) * float(t[di])) if lane[di] < 4 else clip_pitch(d_pos[di] + (end - d_pos[di]) * 0.3)
        data["end_location"] = _r(ip)
        data["outcome"] = "Incomplete"
        self.emit("Pass", T, c, self.ball, data, under_pressure, freeze=True, duration=travel, truth=truth)
        self.advance(travel * float(t[di]) if lane[di] < 4 else travel)
        ip_abs = self.f(T, ip)
        etype = "Interception" if lane[di] < 4 else "Ball Recovery"
        self.set_possession(D, winner, ip_abs, self._regain_pattern(D, ip_abs))
        self.emit(etype, D, winner, ip_abs, {"outcome": "Won"})
        self.advance(self.rng.uniform(0.6, 1.6))

    def do_carry(self, T, D, c, b, o, under_pressure, truth):
        end = o["end"]
        speed = (3.8 + 2.6 * c.attrs["pace"]) * (0.8 + 0.2 * self.st[c.id].stamina)
        dur = o["length"] / speed
        if self.rng.random() < o["p"]:
            self.emit("Carry", T, c, self.ball, {"end_location": _r(end), "distance_m": round(o["length"], 1),
                                                 "speed_ms": round(speed, 2)}, under_pressure,
                      freeze=True, duration=dur, truth=truth)
            self.ball = self.f(T, end)
            self.st[c.id].pos = self.ball.copy()
            self.st[c.id].distance_m += o["length"]
            self.advance(dur)
            return
        loss = clip_pitch(b + (end - b) * self.rng.uniform(0.15, 0.6))
        self.emit("Carry", T, c, self.ball, {"end_location": _r(loss), "distance_m": round(dist_m(b, loss), 1),
                                             "speed_ms": round(speed, 2)}, under_pressure,
                  freeze=True, duration=dur * 0.4, truth=truth)
        self.ball = self.f(T, loss)
        self.advance(dur * 0.4)
        self.contest_lost(T, D, c, loss)

    def do_dribble(self, T, D, c, b, o, under_pressure, truth, defender):
        won = self.rng.random() < o["p"] + 0.4 * (self.a(c, "dribbling") - self.a(defender, "defending")) * 0.5
        self.emit("Dribble", T, c, self.ball, {"outcome": "Complete" if won else "Incomplete",
                                               "opponent": {"id": defender.id, "name": defender.name}},
                  under_pressure, freeze=True, duration=1.2, truth=truth)
        if won:
            self.ball = self.f(T, o["end"])
            self.st[c.id].pos = self.ball.copy()
            self.advance(1.4)
        else:
            self.advance(0.8)
            self.contest_lost(T, D, c, b, defender)

    def contest_lost(self, T, D, c, loc, defender=None):
        d_out, d_pos = self.outfield(D, T)
        if defender is None:
            defender = d_out[int(np.argmin(dists_m(loc, d_pos)))]
        foul_p = 0.14 + 0.08 * D.style["press"] + 0.08 * (1 - self.st[defender.id].stamina)
        if in_box(loc):
            foul_p *= 0.08
        if self.rng.random() < foul_p:
            self.foul(D, defender, T, c)
            return
        self.emit("Duel", D, defender, self.f(T, loc), {"type": "Tackle", "outcome": "Won"})
        self.emit("Dispossessed", T, c, self.f(T, loc))
        loc_abs = self.f(T, loc)
        self.set_possession(D, defender, loc_abs, self._regain_pattern(D, loc_abs))
        self.advance(self.rng.uniform(0.6, 1.5))

    def foul(self, D: Team, offender: Player, T: Team, victim: Player):
        loc_abs = self.st[victim.id].pos.copy() if victim.id != self.carrier.id else self.ball.copy()
        card = None
        r = self.rng.random()
        if r < 0.004:
            card = "Red Card"
        elif r < 0.16:
            card = "Yellow Card"
            self.st[offender.id].yellow += 1
            if self.st[offender.id].yellow >= 2:
                card = "Second Yellow"
        self.emit("Foul Committed", D, offender, loc_abs, {"card": card} if card else {"card": None})
        self.emit("Foul Won", T, victim, loc_abs)
        if card in ("Red Card", "Second Yellow") and offender.role != "GK":
            self.on_pitch[D.id] = [p for p in self.on_pitch[D.id] if p.id != offender.id]
        loc_t = self.f(T, loc_abs)
        if in_box(loc_t):
            self.setup_restart("penalty", T, self.f(T, np.array([108.0, 40.0])))
        elif loc_t[0] > 92 and 20 < loc_t[1] < 60:
            self.setup_restart("free_kick", T, loc_abs, shooter=True)
        elif loc_t[0] > 82:
            self.setup_restart("box_free_kick", T, loc_abs)
        else:
            self.setup_restart("free_kick", T, loc_abs)

    def do_clearance(self, T, D, c, b, o, under_pressure, truth):
        land = o["end"]
        self.emit("Clearance", T, c, self.ball, {"end_location": _r(land)}, under_pressure, freeze=True,
                  duration=2.5, truth=truth)
        self.advance(2.5)
        self.loose_ball(T, D, land, p_keep=0.4)

    def loose_ball(self, T: Team, D: Team, loc_t: np.ndarray, p_keep: float):
        """Contested ball at loc (T's frame)."""
        keep = self.rng.random() < p_keep
        team = T if keep else D
        loc_abs = self.f(T, clip_pitch(loc_t))
        players, pos = self.outfield(team, team)
        here = self.f(team, loc_abs)
        winner = players[int(np.argmin(dists_m(here, pos)))]
        self.emit("Ball Recovery", team, winner, loc_abs, {"outcome": "Won"})
        self.set_possession(team, winner, loc_abs, self._regain_pattern(team, loc_abs) if not keep else None)
        self.advance(self.rng.uniform(0.8, 2.0))

    def shoot(self, T, D, c, b, press_m, header=False, truth=None, set_piece=None):
        d_out, d_pos = self.outfield(D, T)
        blockers = 0 if set_piece == "penalty" else shot_blockers(b, d_pos)
        if set_piece == "free_kick":
            blockers = max(blockers, 2)
        gk = self.keeper(D)
        xg = xg_truth(b, header=header, pressure_m=press_m, blockers=blockers, finishing=self.a(c, "finishing"),
                      keeper=gk.attrs["goalkeeping"] if gk else 0.3, set_piece=set_piece)
        truth = {**(truth or {}), "xg": round(xg, 4)}
        speed = float(np.clip((48 if header else 82) + 22 * c.attrs["finishing"] + self.rng.normal(0, 8), 30, 130))
        goal = self.rng.random() < xg
        if goal:
            outcome = "Goal"
        else:
            r = self.rng.random()
            block_p = 0.0 if set_piece == "penalty" else min(0.55, 0.12 + 0.14 * blockers)
            if r < block_p:
                outcome = "Blocked"
            elif r < block_p + (1 - block_p) * 0.30:
                outcome = "Saved"
            elif r < block_p + (1 - block_p) * 0.34:
                outcome = "Post"
            else:
                outcome = "Off T"
        gy = 40 + self.rng.normal(0, 2.0) if outcome in ("Goal", "Saved") else 40 + self.rng.choice([-1, 1]) * self.rng.uniform(4.2, 12)
        data = {"outcome": outcome, "body_part": "Head" if header else ("Right Foot" if self.rng.random() < 0.7 else "Left Foot"),
                "type": {"penalty": "Penalty", "free_kick": "Free Kick"}.get(set_piece, "Open Play"),
                "shot_speed_kmh": round(speed, 1), "end_location": [120.0, round(float(gy), 1)]}
        dur = dist_m(b, GOAL) / (speed / 3.6)
        self.emit("Shot", T, c, self.ball, data, press_m < 4.5, freeze=True, duration=dur, truth=truth)
        if goal:
            self.score[T.id] += 1
            self.events[-1]["score"] = {"home": self.score[self.home.id], "away": self.score[self.away.id]}
            self.advance(dur)
            self.t += self.rng.uniform(45, 80)
            self.kickoff(D)
            return
        self.advance(dur)
        if outcome == "Saved":
            if gk:
                self.emit("Goal Keeper", D, gk, self.f(T, np.array([118.0, 40.0])), {"type": "Shot Saved"})
            if self.rng.random() < 0.3:
                self.setup_restart("corner", T, self.f(T, np.array([120.0, 0.5 if self.rng.random() < 0.5 else 79.5])))
            elif gk:
                self.set_possession(D, gk, self.f(T, np.array([116.0, 40.0])), "From Keeper")
                self.advance(self.rng.uniform(3, 8))
        elif outcome == "Blocked":
            self.emit("Block", D, d_out[int(np.argmin(dists_m(b, d_pos)))], self.ball)
            r = self.rng.random()
            if r < 0.3:
                self.setup_restart("corner", T, self.f(T, np.array([120.0, 0.5 if self.rng.random() < 0.5 else 79.5])))
            else:
                self.loose_ball(T, D, b + np.array([-6.0, self.rng.normal(0, 6)]), p_keep=0.3)
        else:
            self.setup_restart("goal_kick", D, self.f(D, np.array([6.0, 40.0])))

    def out_of_play(self, T: Team, over_t: np.ndarray):
        D = self.other(T)
        if over_t[0] >= 119.5:
            self.setup_restart("goal_kick", D, self.f(D, np.array([6.0, 40.0])))
        elif over_t[0] <= 0.5:
            self.setup_restart("corner", D, self.f(D, np.array([120.0, 0.5 if over_t[1] > 40 else 79.5])))
        else:
            y = 0.2 if over_t[1] < 40 else 79.8
            self.setup_restart("throw_in", D, self.f(T, np.array([over_t[0], y])))

    def setup_restart(self, kind: str, team: Team, loc_abs: np.ndarray, shooter: bool = False):
        loc_t = self.f(team, loc_abs)
        players = self.on_pitch[team.id]
        if kind == "goal_kick":
            taker = self.keeper(team) or players[0]
        elif kind == "penalty":
            taker = max((p for p in players if p.role != "GK"), key=lambda p: p.attrs["finishing"])
        elif kind in ("corner", "box_free_kick") or (kind == "free_kick" and shooter):
            pool = [p for p in players if ROLE_GROUP[self.slot[p.id]] not in ("GK", "CB")]
            taker = max(pool, key=lambda p: p.attrs["passing"] + (p.attrs["finishing"] if shooter else 0))
        else:
            outs, pos = self.outfield(team, team)
            taker = outs[int(np.argmin(dists_m(loc_t, pos)))]
        pattern = {"goal_kick": "From Goal Kick", "corner": "From Corner", "throw_in": "From Throw In",
                   "free_kick": "From Free Kick", "box_free_kick": "From Free Kick", "penalty": "From Free Kick"}[kind]
        self.set_possession(team, taker, loc_abs, pattern)
        self.restart = kind
        self.mode = {"corner": "corner", "box_free_kick": "box_setpiece", "penalty": "penalty"}.get(kind, "open")
        dead = {"goal_kick": (14, 30), "corner": (18, 35), "throw_in": (6, 18), "free_kick": (12, 30),
                "box_free_kick": (20, 40), "penalty": (40, 70)}[kind]
        self.advance(float(self.rng.uniform(*dead)), self.mode)

    def _regain_pattern(self, team: Team, loc_abs) -> str:
        loc = self.f(team, loc_abs)
        opp = self.other(team)
        _, opp_pos = self.outfield(opp, team)
        caught = int((opp_pos[:, 0] < loc[0]).sum()) if len(opp_pos) else 0
        return "From Counter" if loc[0] < 70 and caught >= 5 else "Regular Play"

    def maybe_substitute(self):
        for team in (self.home, self.away):
            n = self.subs_done[team.id]
            if n >= 3 or not self.bench[team.id] or self.minute < self.sub_minutes[team.id][n]:
                continue
            candidates = [p for p in self.on_pitch[team.id] if p.role != "GK"]
            if not candidates:
                continue
            off = min(candidates, key=lambda p: self.st[p.id].stamina)
            group = ROLE_GROUP[self.slot[off.id]]
            bench = self.bench[team.id]
            on = next((p for p in bench if p.group == group), None) or next((p for p in bench if p.role != "GK"), None)
            if on is None:
                continue
            bench.remove(on)
            self.slot[on.id] = self.slot[off.id]
            self.st[on.id].pos = self.st[off.id].pos.copy()
            self.on_pitch[team.id] = [on if p.id == off.id else p for p in self.on_pitch[team.id]]
            if self.carrier.id == off.id:
                self.carrier = on
            self.subs_done[team.id] += 1
            self.emit("Substitution", team, off, data={"replacement": {"id": on.id, "name": on.name},
                                                       "outcome": "Tactical"})


def _fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"00:{int(m):02d}:{s:06.3f}"


def _r(p) -> list[float]:
    return [round(float(p[0]), 1), round(float(p[1]), 1)]


def simulate_match(home_id: str, away_id: str, seed: int | None = None, **kw) -> Match:
    from .teams import build_team
    return MatchSimulator(build_team(home_id), build_team(away_id), seed=seed, **kw).run()
