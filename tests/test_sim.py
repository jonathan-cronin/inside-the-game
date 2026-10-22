import math

from itg.sim import build_team, simulate_match
from itg.sim.pitch import xg_truth, xt_truth


def _match(seed=11):
    return simulate_match("RVS", "NGT", seed=seed)


def test_deterministic_for_same_seed():
    a, b = _match(5), _match(5)
    assert a.meta["final_score"] == b.meta["final_score"]
    assert [e["type"] for e in a.events] == [e["type"] for e in b.events]


def test_squads_are_stable_and_fictional():
    t1, t2 = build_team("RVS"), build_team("RVS")
    assert [p.name for p in t1.starters] == [p.name for p in t2.starters]
    assert len(t1.starters) == 11 and len(t1.bench) == 5
    assert len({p.id for p in t1.all_players()}) == 16


def test_event_schema_and_clock():
    m = _match()
    ev = m.events
    assert ev[0]["type"] == "Starting XI"
    times = [e["elapsed_s"] for e in ev]
    assert times == sorted(times)
    assert 5200 < times[-1] < 6300  # ~90 minutes plus added time
    for e in ev:
        if e["location"] is not None:
            x, y = e["location"]
            assert 0 <= x <= 120 and 0 <= y <= 80


def test_freeze_frames_include_everyone_and_actor_on_ball():
    m = _match()
    on_ball = [e for e in m.events if e["type"] in ("Pass", "Shot", "Carry", "Dribble")]
    assert on_ball
    for e in on_ball[:300]:
        ff = e["freeze_frame"]
        assert 19 <= len(ff) <= 22  # red cards can remove players
        actor = [f for f in ff if f["actor"]]
        assert len(actor) == 1
        assert math.dist(actor[0]["location"], e["location"]) < 1.0


def test_score_matches_goals():
    m = _match()
    goals = [e for e in m.events if e["type"] == "Shot" and e["shot"]["outcome"] == "Goal"]
    home = sum(1 for g in goals if g["team"]["id"] == "RVS")
    assert m.meta["final_score"] == {"home": home, "away": len(goals) - home}


def test_truth_models_behave_sensibly():
    assert xg_truth((108, 40)) > xg_truth((95, 40)) > xg_truth((80, 40))
    assert xg_truth((108, 40), header=True) < xg_truth((108, 40))
    assert xg_truth((108, 40), set_piece="penalty") > 0.6
    assert xt_truth((110, 40)) > xt_truth((90, 40)) > xt_truth((40, 40))
