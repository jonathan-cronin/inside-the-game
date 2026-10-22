"""Analyse a match with the trained models.

    python -m itg.analytics analyse data/matches/SYN-RVS-NGT-7.events.jsonl
    python -m itg.analytics analyse <events file> --top 15 --mode casual --out reports/insights.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_match
from .explain import headline
from .live import analyse_match
from .models import ModelBundle


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m itg.analytics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyse", help="run the live analyst over a match and print the key moments")
    a.add_argument("events")
    a.add_argument("--bundle", default="models/bundle.pkl")
    a.add_argument("--top", type=int, default=12)
    a.add_argument("--mode", choices=["analyst", "casual"], default="analyst")
    a.add_argument("--out", default=None, help="write every insight, the WP timeline and stats as JSON")
    args = ap.parse_args(argv)

    bundle = ModelBundle.load(Path(args.bundle))
    m = load_match(Path(args.events))
    la = analyse_match(bundle, m.meta, m.events)
    fs = m.meta["final_score"]
    print(f"{m.meta['home']['name']} {fs['home']}-{fs['away']} {m.meta['away']['name']}\n")
    top = sorted(la.insights, key=lambda i: -i["importance"])[: args.top]
    for ins in sorted(top, key=lambda i: i["elapsed_s"]):
        print(" ", headline(ins, args.mode))
    s = la.summary()
    print("\nxG", {t: v["xg"] for t, v in s["teams"].items()}, " possession",
          {t: v["possession"] for t, v in s["teams"].items()})
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "meta": {k: v for k, v in m.meta.items() if k != "players"},
            "insights": la.insights, "wp_timeline": la.wp_timeline, "summary": s,
            "players": {pid: p.public() for pid, p in la.players.items() if p.touches},
        }, indent=1, default=str))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
