"""Command line for the match simulator.

    python -m itg.sim match --home RVS --away NGT --seed 7
    python -m itg.sim match --home RVS --away NGT --seed 7 --stream --speed 60
    python -m itg.sim batch --n 200 --out data/batch      (gzipped, ~0.6 MB per match)
    python -m itg.sim clubs
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
from pathlib import Path

from .engine import Match, simulate_match
from .stream import replay, stdout_sink
from .teams import CLUBS, build_team


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def save(match: Match, out_dir: Path, with_truth: bool = True, compress: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    mid = match.meta["match_id"]
    path = out_dir / (f"{mid}.events.jsonl.gz" if compress else f"{mid}.events.jsonl")
    opener = (lambda p: gzip.open(p, "wt")) if compress else (lambda p: open(p, "w"))
    with opener(path) as fh:
        for ev in match.events:
            fh.write(json.dumps(ev, separators=(",", ":"), default=_json_default) + "\n")
    (out_dir / f"{mid}.meta.json").write_text(json.dumps(match.meta, indent=2, default=_json_default))
    if with_truth:
        (out_dir / f"{mid}.truth.json").write_text(json.dumps(match.truth, default=_json_default))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m itg.sim")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("match", help="simulate one match")
    m.add_argument("--home", default="RVS")
    m.add_argument("--away", default="NGT")
    m.add_argument("--seed", type=int, default=7)
    m.add_argument("--out", default="data/matches")
    m.add_argument("--stream", action="store_true", help="print events to stdout in (scaled) real time")
    m.add_argument("--speed", type=float, default=60.0)

    b = sub.add_parser("batch", help="simulate many matches (training data for the analytics models)")
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--seed", type=int, default=10_000)
    b.add_argument("--out", default="data/batch")

    sub.add_parser("clubs", help="list the fictional clubs and squads")

    args = ap.parse_args(argv)
    if args.cmd == "clubs":
        for c in CLUBS:
            t = build_team(c["id"])
            print(f"{c['id']}  {c['name']:20s} {c['formation']:8s} style={c['style']}")
            for p in t.starters:
                print(f"     {p.number:>2} {p.role:4s} {p.name}")
        return
    if args.cmd == "match":
        match = simulate_match(args.home, args.away, seed=args.seed)
        path = save(match, Path(args.out))
        if args.stream:
            replay(match.events, stdout_sink, speed=args.speed)
        else:
            fs = match.meta["final_score"]
            print(f"{match.meta['home']['name']} {fs['home']}-{fs['away']} {match.meta['away']['name']}"
                  f"  ({len(match.events)} events) -> {path}")
        return
    if args.cmd == "batch":
        ids = [c["id"] for c in CLUBS]
        pairs = list(itertools.permutations(ids, 2))
        out = Path(args.out)
        for i in range(args.n):
            h, a = pairs[i % len(pairs)]
            match = simulate_match(h, a, seed=args.seed + i)
            save(match, out, compress=True)
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{args.n} matches")
        print(f"done -> {out}")


if __name__ == "__main__":
    main()
