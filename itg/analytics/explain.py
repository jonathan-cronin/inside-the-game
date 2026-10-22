"""Deterministic, template-based explanations of insights.

These do two jobs: they show the numbers behind each insight in plain
English, and they are the fallback narration if a model call fails, so
the overlay never goes blank.
"""
from __future__ import annotations


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _opt(o: dict) -> str:
    if o["kind"] == "shot":
        return f"a shot (xG {o['xg']:.2f})"
    if o["kind"] == "pass":
        who = o.get("target_name") or "a teammate"
        return f"the pass to {who} ({_pct(o['p_success'])} to come off, worth {o['value_if_success']:.2f} xT)"
    if o["kind"] == "carry":
        return f"carrying it forward {o['detail'].get('distance_m', 0):.0f} m"
    return o["kind"]


def headline(ins: dict, mode: str = "analyst") -> str:
    k, f = ins["kind"], ins["facts"]
    who = (ins.get("player") or {}).get("name", "")
    m = f"{ins['minute']}'"
    if k == "goal":
        sit = f.get("situation")
        if mode == "casual":
            lead = {"equaliser": "It's all square!", "go_ahead": f"{ins['team']['name']} lead!",
                    "extend_lead": f"{ins['team']['name']} extend the lead.",
                    "consolation": "One back."}.get(sit, "")
            return f"{m} GOAL! {who} scores. {lead} {ins['team']['name']} now {_pct(_team_wp(ins, f['wp_after']))} to win."
        return (f"{m} Goal, {who} ({sit.replace('_', ' ') if sit else 'goal'}): {f['distance_m']} m, xG {f['xg']:.2f}, "
                f"{f['shot_speed_kmh']:.0f} km/h. Worth {f['xpts_swing']:+.2f} expected points "
                f"(win {_pct(_team_wp(ins, f['wp_before']))} -> {_pct(_team_wp(ins, f['wp_after']))}, "
                f"draw {_pct(f['wp_before']['draw'])} -> {_pct(f['wp_after']['draw'])}).")
    if k == "big_chance_missed":
        if mode == "casual":
            return f"{m} So close! {who} misses a big chance."
        return f"{m} Big chance missed by {who}: xG {f['xg']:.2f} ({f['outcome']}). Cost ~{_pct(f.get('wp_lost'))} win probability."
    if k == "missed_option":
        ch, bt = f["chosen"], f["better_option"]
        if mode == "casual":
            return f"{m} {who} went for {ch['kind']}, but {_opt_casual(bt)} was on."
        cost = f.get("decision_wp")
        tail = f" Cost: {abs(f['decision_value']):.2f} xG" + (f", {_pct(abs(cost))} win probability." if cost is not None else ".")
        return f"{m} {who} chose {_opt(ch)}. Better: {_opt(bt)}.{tail}"
    if k == "great_decision":
        ch, nb = f["chosen"], f["next_best"]
        if mode == "casual":
            return f"{m} Brilliant from {who}: picked the best option on the pitch."
        return f"{m} {who} picked the best of {f['n_options']} options: {_opt(ch)}, {f['decision_value']:.2f} xG better than {_opt(nb)}."
    if k == "difficult_pass":
        if mode == "casual":
            return f"{m} What a pass from {who} to {f['recipient']}!"
        return (f"{m} {who} completes a pass rated {f['difficulty']}/100 for difficulty to {f['recipient']} "
                f"({f['length_m']:.0f} m, {f['ball_speed_kmh']:.0f} km/h, +{f['xt_added']:.3f} xT).")
    if k == "sprint":
        return f"{m} {f['player']} hits {f['speed_kmh']:.1f} km/h."
    if k == "momentum_shift":
        return f"{m} Momentum: {f['team_name']} have created {_pct(max(f['home_share'], f['away_share']))} of the threat in the last 10 minutes."
    if k == "red_card":
        return f"{m} Red card for {who}."
    if k == "substitution":
        return f"{m} Substitution: {f['replacement']} on for {who}."
    if k == "milestone":
        return f"{m} {who}: {f['what']} ({_pct(f['pass_completion'])} completion)."
    if k == "shot":
        return f"{m} {who} shoots: xG {f['xg']:.2f}, {f['outcome']}."
    return f"{m} {k}"


def _opt_casual(o: dict) -> str:
    if o["kind"] == "shot":
        return "a shot"
    if o["kind"] == "pass":
        return f"a pass to {o.get('target_name') or 'a teammate'}"
    return "running with it"


def _team_wp(ins: dict, wp: dict) -> float:
    return wp["home"] if ins.get("team_is_home", True) else wp["away"]
