"""Train every analytics model on simulated matches and check each against the generator's hidden truth.

    python -m itg.sim batch --n 400 --out data/batch     # once (or two processes with different --seed)
    python scripts/train_models.py --data data/batch --out models/bundle.pkl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from itg.analytics.data import Timeline, carries_table, iter_matches, match_files, moves_for_xt, passes_table, shots_table  # noqa: E402
from itg.analytics.models import ModelBundle, WinProbability, XTGrid, carry_model, pass_model, xg_model  # noqa: E402
from itg.analytics.winprob_data import brier, live_state, outcome, state_rows, team_ratings  # noqa: E402
from itg.analytics.winprob_data import log_loss as wp_log_loss  # noqa: E402
from itg.sim.pitch import xt_truth  # noqa: E402  (evaluation only: never used to train)


def is_test(match_id: str, frac: float = 0.15) -> bool:
    return int(hashlib.md5(match_id.encode()).hexdigest(), 16) % 1000 < frac * 1000


def calibration(pred, y, bins=(0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0)):
    out = []
    pred, y = np.asarray(pred), np.asarray(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (pred >= lo) & (pred < hi)
        if k.sum():
            out.append({"bin": f"{lo:.2f}-{hi:.2f}", "n": int(k.sum()), "pred": round(float(pred[k].mean()), 3),
                        "actual": round(float(y[k].mean()), 3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/batch")
    ap.add_argument("--out", default="models/bundle.pkl")
    ap.add_argument("--report", default="reports/model_report.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # one streaming pass over the data: matches with freeze frames are too big to hold at once
    tabs = {"shots": [], "passes": [], "carries": [], "moves": []}
    timelines: list[Timeline] = []
    files = match_files(Path(args.data), limit=args.limit)
    for i, m in enumerate(iter_matches(Path(args.data), limit=args.limit)):
        tabs["shots"].append(shots_table([m]))
        tabs["passes"].append(passes_table([m]))
        tabs["carries"].append(carries_table([m]))
        mv = moves_for_xt([m])
        mv["match_id"] = m.match_id
        tabs["moves"].append(mv)
        timelines.append(Timeline.from_match(m))
        if (i + 1) % 50 == 0:
            print(f"  featurised {i + 1}/{len(files)} matches")
    shots, passes, carries, all_moves = (pd.concat(tabs[k], ignore_index=True) for k in ("shots", "passes", "carries", "moves"))
    train = [t for t in timelines if not is_test(t.match_id)]
    test = [t for t in timelines if is_test(t.match_id)]
    print(f"{len(timelines)} matches: {len(train)} train / {len(test)} test")
    report: dict = {"n_train_matches": len(train), "n_test_matches": len(test)}

    # ------------------------------------------------------------------ xG
    s_tr, s_te = shots[~shots.match_id.map(is_test)], shots[shots.match_id.map(is_test)]
    xg = xg_model().fit(s_tr, "goal")
    p = xg.predict(s_te)
    base = np.full(len(s_te), s_tr.goal.mean())
    report["xg"] = {
        "train_shots": len(s_tr), "test_shots": len(s_te),
        "log_loss": round(log_loss(s_te.goal, p), 4), "log_loss_baseline": round(log_loss(s_te.goal, base), 4),
        "brier": round(brier_score_loss(s_te.goal, p), 4), "auc": round(roc_auc_score(s_te.goal, p), 3),
        "corr_with_truth": round(float(np.corrcoef(p, s_te.truth_xg)[0, 1]), 3),
        "mae_vs_truth": round(float(np.abs(p - s_te.truth_xg).mean()), 4),
        "sum_pred": round(float(p.sum()), 1), "sum_goals": int(s_te.goal.sum()),
        "sum_truth": round(float(s_te.truth_xg.sum()), 1),
        "calibration": calibration(p, s_te.goal),
    }
    print("xG", {k: v for k, v in report["xg"].items() if k != "calibration"})

    # ------------------------------------------------------------- passes
    p_tr, p_te = passes[~passes.match_id.map(is_test)], passes[passes.match_id.map(is_test)]
    pm = pass_model().fit(p_tr, "complete")
    pp = pm.predict(p_te)
    known = p_te.truth_p.notna().to_numpy()
    report["pass_success"] = {
        "train_passes": len(p_tr), "test_passes": len(p_te),
        "log_loss": round(log_loss(p_te.complete, pp), 4),
        "log_loss_baseline": round(log_loss(p_te.complete, np.full(len(p_te), p_tr.complete.mean())), 4),
        "auc": round(roc_auc_score(p_te.complete, pp), 3),
        "corr_with_truth": round(float(np.corrcoef(pp[known], p_te.truth_p[known])[0, 1]), 3),
        "mae_vs_truth": round(float(np.abs(pp[known] - p_te.truth_p[known]).mean()), 4),
        "calibration": calibration(pp, p_te.complete, bins=(0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.01)),
    }
    print("pass", {k: v for k, v in report["pass_success"].items() if k != "calibration"})

    # ------------------------------------------------------------- carries
    c_tr, c_te = carries[~carries.match_id.map(is_test)], carries[carries.match_id.map(is_test)]
    cm = carry_model().fit(c_tr, "kept")
    cp = cm.predict(c_te)
    report["carry_retention"] = {"train": len(c_tr), "test": len(c_te),
                                 "auc": round(roc_auc_score(c_te.kept, cp), 3),
                                 "base_rate": round(float(c_tr.kept.mean()), 3)}
    print("carry", report["carry_retention"])

    # ------------------------------------------------------------------ xT
    moves = all_moves[~all_moves.match_id.map(is_test)]
    shot_rows = moves[moves.kind == "shot"]
    xg_by_event = dict(zip(shots.event_id, xg.predict(shots)))
    xt = XTGrid().fit(moves, shot_xg=shot_rows.event_id.map(xg_by_event).to_numpy())
    centres = [((i + 0.5) * 120 / xt.nx, (j + 0.5) * 80 / xt.ny) for i in range(xt.nx) for j in range(xt.ny)]
    learned = np.array([xt.grid[i, j] for i in range(xt.nx) for j in range(xt.ny)])
    truth = np.array([xt_truth(c) for c in centres])
    from scipy.stats import spearmanr
    report["xt"] = {"grid": f"{xt.nx}x{xt.ny}", "iterations": xt.iterations,
                    "max": round(float(learned.max()), 4), "own_half_mean": round(float(learned[: len(learned) // 2].mean()), 4),
                    "spearman_vs_truth_surface": round(float(spearmanr(learned, truth).statistic), 3)}
    print("xT", report["xt"])

    # ----------------------------------------------------- win probability
    ratings = team_ratings(train, xg_by_event)
    rows = state_rows(train, xg_by_event, ratings)
    wp = WinProbability(ratings=ratings).fit(rows)
    by_minute = {}
    for minute in (0, 30, 60, 75, 85):
        probs = [wp.probabilities(live_state(m, minute, xg_by_event)) for m in test]
        outs = [outcome(m) for m in test]
        by_minute[minute] = {"log_loss": round(wp_log_loss(probs, outs), 4), "brier": round(brier(probs, outs), 4)}
    freq = {o: np.mean([outcome(m) == o for m in train]) for o in ("home", "draw", "away")}
    report["win_probability"] = {
        "ratings": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in ratings.items()},
        "test_log_loss_by_minute": by_minute,
        "baseline_log_loss": round(wp_log_loss([freq] * len(test), [outcome(m) for m in test]), 4),
        "coefficients": dict(zip(["is_home", "log_attack", "log_opp_defence", "score_diff", "score_diff_x_time",
                                  "red_diff", "recent_xg_share"], np.round(wp.model.coef_, 3).tolist())),
    }
    print("WP", {k: v for k, v in report["win_probability"].items() if k != "ratings"})

    bundle = ModelBundle(xg=xg, pass_success=pm, carry=cm, xt=xt, wp=wp, report=report)
    bundle.save(Path(args.out))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"saved -> {args.out}, report -> {args.report}")


if __name__ == "__main__":
    main()
