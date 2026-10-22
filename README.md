# Inside the Game: decision intelligence for live football

Built for the Microsoft x Premier League *Inside the Game* developer hackathon (Oct 2026).

**Main idea:** each key moment is explained as a *decision*. We compare what the player did with
the options they had, and measure how much it moved the match.

> "He shot from 25 yards (xG 0.04). The square ball to the free winger was worth 0.15.
> Win probability dropped 2%."

## Pipeline

| Stage | Component | Status |
|---|---|---|
| Data | Synthetic match engine with 22-player freeze frames (`itg/sim`) | Built |
| Ingest | Live event feed (`itg/sim/stream.py`), later Azure Event Hubs | Local version built |
| Interpret | Learned xG, xT, pass difficulty, carry retention, in-play win probability (`itg/analytics`) | Built |
| Explain | Counterfactual decision engine plus a live analyst that emits ranked insights | Built |
| Agents | Moment Detector, Tactical Explainer, Narrator, Personalizer, Recap (Microsoft Foundry + Agent Framework, stats served over MCP) | Planned |
| Render | Timed, machine-readable overlay JSON plus a web front end | Planned |

## The synthetic match engine

All clubs, players and the competition are fictional. Each club has a formation, a playing style
(possession, directness, pressing, block height, width) and a fixed squad with attributes
(passing, vision, finishing, dribbling, pace, stamina, defending, aerial, goalkeeping).

On every on-ball moment:

1. **Both teams move into shape.** Their positions depend on where the ball is, which team has
   it, each team's style, the score, red cards and fatigue. That covers pressing, a defensive
   line, offside, the defence collapsing into the box near goal, man-marking in the box and
   set-piece routines. Players can only move as fast as their top speed allows, so pressing
   teams cover more distance and tire.
2. **The ball carrier picks an action.** They consider every pass, through balls, crosses,
   carries, take-ons, shots and clearances. Each option is valued with the engine's hidden
   models, and one is chosen at random weighted by value. How sharply the best option is
   favoured depends on the player's vision, so players sometimes make the wrong choice.
   Those are the moments the counterfactual engine explains.
3. **The outcome is sampled.** Events are recorded StatsBomb-style, with a freeze frame
   (position and speed of all 22 players) on every on-ball event.

**Hidden truth vs. learned models.** The engine's own xG, xT and pass-success functions
(`itg/sim/pitch.py`, `*_truth`) are never used by the analytics layer. The analytics layer
has to learn its models from generated events. Because the truth is known, we can show how
well each learned model recovers it.

### Realism check (40 simulated matches, both teams combined)

| Metric | Simulated | Real top-flight range |
|---|---|---|
| Goals | 2.9 | 2.4–3.3 |
| Shots | 23.8 | 21–29 |
| Shots on target | 33% | 30–40% |
| xG | 2.6 | 2.4–3.3 |
| Passes | 1,096 | 800–1,100 |
| Pass completion | 82% | 78–86% |
| Fouls | 19.2 | 18–26 |
| Yellow cards | 3.3 | 2.5–5 |
| Corners | 8.8 | 8–12 |
| Distance per outfield player | 10.5 km | 9–12 km |

Known gap: a higher share of shots come from 6–11 m than in real football, and there are
fewer long shots. To reproduce: `python scripts/validate_sim.py --n 40`.

## Analytics layer

All models are trained only on simulated events (`python scripts/train_models.py`). Each one is then
checked against the generator's hidden truth on held-out matches. Results from 400 matches
(340 train / 60 test) are in `reports/model_report.json`:

| Model | What it is | Held-out result |
|---|---|---|
| xG | Gradient boosting on distance, angle, header, set piece, pressure, blockers, keeper position (from the freeze frame) | Correlation with true xG **0.94**, mean error 0.027; 158 predicted vs 159 true xG on 1,454 shots; well calibrated |
| Pass success / difficulty | Gradient boosting on length, direction, nearest opponent to the passing lane, receiver's space, pressure on the passer | AUC 0.78; correlation with true success probability 0.77; calibrated in every bin |
| Carry retention | Chance of keeping the ball, from the situation at the start of the carry | AUC 0.75 |
| xT | Markov-chain expected threat (Singh, 2018) on a 16 × 12 grid, using the learned xG | Rank correlation with the generator's hidden value surface **0.90** |
| Win probability | Two Poisson goal processes: rates from pre-match strength, scoreline × time, red cards and recent xG share | Log loss falls from 0.93 at kick-off to 0.45 by 85' (frequency baseline 1.08) |

**Decision engine (`itg/analytics/decisions.py`).** For each on-ball moment it rebuilds every
option from the freeze frame and values each with the learned models:
`EV = P(success) × xT(end) − P(failure) × opponent xT at the turnover point`, and a shot is worth its xG.
Decision value = EV(chosen) − EV(best alternative), in expected goals. Multiplying by how much a goal
would be worth right now gives the cost in win probability.

**Live analyst (`itg/analytics/live.py`).** Processes events one at a time, the same way on a live
feed or a finished match. It keeps team and player state (xG, xT added, pass completion, decision
quality, distance, top speed, possession, momentum, win probability) and emits ranked insights:
`goal`, `big_chance_missed`, `missed_option`, `great_decision`, `difficult_pass`, `momentum_shift`,
`sprint`, `red_card`, `milestone`. Each insight carries the numbers behind it, so agents narrate
facts rather than make them up. `itg/analytics/explain.py` turns insights into template text in
analyst and casual modes, which also serves as the fallback narration if a model call fails.

```text
$ python -m itg.analytics analyse data/matches/SYN-HBR-KGM-99.events.jsonl
  33' Darius Whitfield chose the pass to Arlo Ramos (95% to come off, worth 0.03 xT). Better: a shot
      (xG 0.15). Cost: 0.12 xG, 4% win probability.
  40' Goal, Darius Whitfield (go ahead): 7.0 m, xG 0.20, 103 km/h. Worth +0.75 expected points
      (win 46% -> 77%, draw 36% -> 18%).
  50' Aaron Mansour completes a pass rated 60/100 for difficulty to Darius Whitfield (30 m, 69 km/h, +0.102 xT).
  60' Goal, Oscar Gallagher (equaliser): 7.9 m, xG 0.17, 105 km/h. Worth +0.81 expected points
      (win 9% -> 28%, draw 26% -> 50%).
```

One finding is consistent across matches: by the learned models, the simulated players pass too
often when a shot from 15–20 m was worth more. That is a real tactical takeaway the system surfaces
on its own.

## Event format

Each line of `*.events.jsonl` is one event:

```json
{"id": "SYN-RVS-NGT-7-00412", "period": 1, "timestamp": "00:23:41.250", "minute": 23,
 "elapsed_s": 1421.25, "type": "Pass", "team": {"id": "RVS", "name": "Riverside Athletic"},
 "player": {"id": "RVS-08", "name": "...", "position": "LCM"}, "possession": 41,
 "play_pattern": "Regular Play", "location": [71.3, 30.2], "under_pressure": true,
 "score": {"home": 1, "away": 0},
 "pass": {"recipient": {...}, "end_location": [88.0, 12.4], "length_m": 20.1,
          "height": "Ground Pass", "type": "Open Play", "cross": false, "through_ball": false,
          "switch": false, "ball_speed_kmh": 61.2, "outcome": "Complete"},
 "freeze_frame": [{"player_id": "...", "teammate": true, "actor": true, "keeper": false,
                   "location": [71.3, 30.2], "speed_ms": 3.1}, ...]}
```

Every location is given from the acting team's point of view, attacking towards x = 120 on a
120 × 80 pitch (StatsBomb convention). `score` is the score *after* the event.

Event types: Starting XI, Half Start/End, Pass, Carry, Dribble, Shot, Pressure, Interception,
Ball Recovery, Duel, Dispossessed, Clearance, Block, Goal Keeper, Foul Committed/Won,
Substitution.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m itg.sim clubs                                    # the fictional clubs and squads
python -m itg.sim match --home RVS --away NGT --seed 7     # one match -> data/matches/
python -m itg.sim match --home RVS --away NGT --stream --speed 60   # live feed, 60x speed
python -m itg.sim batch --n 400                            # training data -> data/batch/ (~250 MB)
python scripts/train_models.py                             # -> models/bundle.pkl + reports/model_report.json
python -m itg.analytics analyse data/matches/SYN-RVS-NGT-7.events.jsonl --mode casual
python -m pytest -q
```
