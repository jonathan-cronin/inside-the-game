"""Analytics tests: train tiny models on a few simulated matches and check the pipeline end to end."""
import numpy as np
import pandas as pd
import pytest

from itg.analytics.data import Timeline, carries_table, moves_for_xt, passes_table, shots_table
from itg.analytics.decisions import DecisionEngine
from itg.analytics.explain import headline
from itg.analytics.features import Frame, pass_features
from itg.analytics.live import analyse_match
from itg.analytics.models import ModelBundle, WinProbability, XTGrid, carry_model, pass_model, xg_model
from itg.analytics.winprob_data import state_rows, team_ratings
from itg.sim import simulate_match
from itg.analytics.data import MatchData


@pytest.fixture(scope="module")
def tiny():
    sims = [simulate_match(h, a, seed=s) for h, a, s in
            [("RVS", "NGT", 1), ("NGT", "RVS", 2), ("RVS", "NGT", 3), ("NGT", "RVS", 4)]]
    ms = [MatchData(m.meta["match_id"], m.meta, m.events, m.truth) for m in sims]
    shots, passes, carries, moves = shots_table(ms), passes_table(ms), carries_table(ms), moves_for_xt(ms)
    xg = xg_model().fit(shots, "goal")
    pm = pass_model().fit(passes, "complete")
    cm = carry_model().fit(carries, "kept")
    sx = dict(zip(shots.event_id, xg.predict(shots)))
    xt = XTGrid().fit(moves, moves[moves.kind == "shot"].event_id.map(sx).to_numpy())
    tls = [Timeline.from_match(m) for m in ms]
    ratings = team_ratings(tls, sx)
    wp = WinProbability(ratings=ratings).fit(state_rows(tls, sx, ratings))
    return ms, ModelBundle(xg, pm, cm, xt, wp)


def test_pass_features_see_blocking_defender():
    ev = {"location": [60, 40], "freeze_frame": [
        {"player_id": "a", "teammate": True, "actor": True, "keeper": False, "location": [60, 40], "speed_ms": 0},
        {"player_id": "b", "teammate": True, "actor": False, "keeper": False, "location": [80, 40], "speed_ms": 0},
        {"player_id": "c", "teammate": False, "actor": False, "keeper": False, "location": [70, 40.5], "speed_ms": 0},
    ]}
    fr = Frame.from_event(ev)
    blocked = pass_features([60, 40], [80, 40], fr, high=False)
    open_lane = pass_features([60, 40], [60, 70], fr, high=False)
    assert blocked["lane_m"] < 1.0 < open_lane["lane_m"]


def test_xt_rises_towards_goal(tiny):
    _, b = tiny
    assert b.xt.value(110, 40) > b.xt.value(80, 40) > b.xt.value(30, 40)


def test_win_probability_is_coherent(tiny):
    _, b = tiny
    s = {"home": "RVS", "away": "NGT", "minute": 80, "home_goals": 1, "away_goals": 0,
         "red_diff": 0, "recent_xg_share": 0.5}
    p = b.wp.probabilities(s)
    assert abs(sum(p.values()) - 1) < 1e-6
    assert p["home"] > p["away"]
    assert b.wp.probabilities(s, extra_away=1)["home"] < p["home"]


def test_decision_engine_scores_alternatives(tiny):
    ms, b = tiny
    eng = DecisionEngine(b)
    evs = [e for e in ms[0].events if e["type"] == "Pass" and e["pass"]["type"] == "Open Play"][:50]
    out = [eng.evaluate(e, goal_wp_swing=0.3) for e in evs]
    out = [o for o in out if o]
    assert out
    for o in out:
        assert o["n_options"] >= 2
        assert 0 <= o["chosen"]["p_success"] <= 1
        if o["best_alternative"]:
            assert o["decision_value"] == pytest.approx(o["chosen"]["ev"] - o["best_alternative"]["ev"], abs=1e-3)


def test_live_analyst_end_to_end(tiny):
    ms, b = tiny
    la = analyse_match(b, ms[1].meta, ms[1].events)
    kinds = {i["kind"] for i in la.insights}
    goals = sum(ms[1].meta["final_score"].values())
    assert (goals == 0) or ("goal" in kinds)
    assert la.summary()["score"] == ms[1].meta["final_score"]
    for ins in la.insights[:200]:
        assert isinstance(headline(ins), str) and isinstance(headline(ins, "casual"), str)
