"""Load simulated matches and turn them into model-ready tables."""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .features import Frame, carry_features, pass_features, shot_features


@dataclass
class MatchData:
    match_id: str
    meta: dict
    events: list[dict]
    truth: dict | None


def load_match(events_path: Path) -> MatchData:
    events_path = Path(events_path)
    opener = gzip.open if events_path.suffix == ".gz" else open
    with opener(events_path, "rt") as fh:
        events = [json.loads(line) for line in fh]
    mid = events[0]["match_id"]
    base = events_path.parent / mid
    meta = json.loads(Path(f"{base}.meta.json").read_text())
    truth_p = Path(f"{base}.truth.json")
    truth = json.loads(truth_p.read_text()) if truth_p.exists() else None
    return MatchData(mid, meta, events, truth)


def match_files(directory: Path, limit: int | None = None) -> list[Path]:
    files = sorted(Path(directory).glob("*.events.jsonl*"))
    return files[:limit] if limit else files


def iter_matches(directory: Path, limit: int | None = None):
    """Yield matches one at a time (a match with freeze frames is large in memory)."""
    for f in match_files(directory, limit):
        yield load_match(f)


def load_dir(directory: Path, limit: int | None = None) -> list[MatchData]:
    return list(iter_matches(directory, limit))


@dataclass
class Timeline:
    """The light-weight record of a match needed for win probability and ratings."""
    match_id: str
    home: str
    away: str
    shots: list[tuple[int, str, str, int]]   # (minute, team, event_id, goal)
    reds: list[tuple[int, str]]
    end_minute: int
    final_score: dict

    @classmethod
    def from_match(cls, m: "MatchData") -> "Timeline":
        shots, reds = [], []
        for e in m.events:
            if e["type"] == "Shot":
                shots.append((e["minute"], e["team"]["id"], e["id"], int(e["shot"]["outcome"] == "Goal")))
            elif e["type"] == "Foul Committed" and (e.get("foul_committed") or {}).get("card") in ("Red Card", "Second Yellow"):
                reds.append((e["minute"], e["team"]["id"]))
        return cls(m.match_id, m.meta["home"]["id"], m.meta["away"]["id"], shots, reds,
                   max(e["minute"] for e in m.events) + 1, m.meta["final_score"])


def truth_by_event(m: MatchData) -> dict[str, dict]:
    if not m.truth:
        return {}
    return {t["event_id"]: t for t in m.truth["events"]}


def shots_table(matches: list[MatchData]) -> pd.DataFrame:
    rows = []
    for m in matches:
        tr = truth_by_event(m)
        for e in m.events:
            if e["type"] != "Shot":
                continue
            fr = Frame.from_event(e)
            sp = e["shot"]["type"] if e["shot"]["type"] != "Open Play" else None
            f = shot_features(e["location"], fr, header=e["shot"]["body_part"] == "Head", set_piece=sp)
            f.update(match_id=m.match_id, event_id=e["id"], goal=int(e["shot"]["outcome"] == "Goal"),
                     truth_xg=tr.get(e["id"], {}).get("xg"))
            rows.append(f)
    return pd.DataFrame(rows)


def passes_table(matches: list[MatchData]) -> pd.DataFrame:
    rows = []
    for m in matches:
        tr = truth_by_event(m)
        for e in m.events:
            if e["type"] != "Pass" or "freeze_frame" not in e:
                continue
            p = e["pass"]
            fr = Frame.from_event(e)
            # the intended target: the recipient's position at the moment of the pass
            target = p["end_location"]
            for ffp in e["freeze_frame"]:
                if ffp["player_id"] == p["recipient"]["id"] and not p["through_ball"]:
                    target = ffp["location"]
            f = pass_features(e["location"], target, fr, high=p["height"] == "High Pass", cross=p["cross"],
                              through=p["through_ball"], set_piece=p["type"] != "Open Play")
            f.update(match_id=m.match_id, event_id=e["id"], complete=int(p["outcome"] == "Complete"),
                     truth_p=tr.get(e["id"], {}).get("p_success"))
            rows.append(f)
    return pd.DataFrame(rows)


def carries_table(matches: list[MatchData]) -> pd.DataFrame:
    rows = []
    for m in matches:
        ev = m.events
        for i, e in enumerate(ev):
            if e["type"] != "Carry":
                continue
            lost = any(x["type"] == "Dispossessed" and x["player"] and x["player"]["id"] == e["player"]["id"]
                       for x in ev[i + 1:i + 4])
            f = carry_features(e["location"], Frame.from_event(e))
            f.update(match_id=m.match_id, event_id=e["id"], kept=int(not lost))
            rows.append(f)
    return pd.DataFrame(rows)


def moves_for_xt(matches: list[MatchData]) -> pd.DataFrame:
    """Every attempted move (pass or carry) and shot, with start/end and outcome, in the actor's frame."""
    rows = []
    for m in matches:
        ev = m.events
        for i, e in enumerate(ev):
            t = e["type"]
            if t == "Pass":
                rows.append(dict(kind="move", x=e["location"][0], y=e["location"][1],
                                 ex=e["pass"]["end_location"][0], ey=e["pass"]["end_location"][1],
                                 ok=int(e["pass"]["outcome"] == "Complete")))
            elif t == "Carry":
                lost = any(x["type"] == "Dispossessed" for x in ev[i + 1:i + 4])
                rows.append(dict(kind="move", x=e["location"][0], y=e["location"][1],
                                 ex=e["carry"]["end_location"][0], ey=e["carry"]["end_location"][1], ok=int(not lost)))
            elif t == "Shot":
                rows.append(dict(kind="shot", x=e["location"][0], y=e["location"][1], ex=None, ey=None,
                                 ok=int(e["shot"]["outcome"] == "Goal"), event_id=e["id"], match_id=m.match_id))
    return pd.DataFrame(rows)
