# Handover: Inside the Game hackathon entry

Read this first. It sets out what has been built, the decisions already made, and what's next.
This project was started in a separate Claude session, and this file is the only handover.

## The hackathon
- Microsoft x Premier League "Inside the Game: Developer Hackathon". Official rules are in the
  GitHub repo microsoft/insidethegamehackathon (OFFICIAL RULES.md).
- **Submission deadline: Tue 27 Oct 2026, 11:59 PM Pacific = Wed 28 Oct, about 07:00 UK.** Treat
  Tuesday evening UK time as the real deadline. Jono is registered and entering solo.
- The submission needs:
  - a public GitHub repo
  - a demo video under 2 minutes on YouTube or Vimeo, with no third-party trademarks or
    copyrighted music
  - a short pitch naming the Microsoft/Azure technologies used
  - a working link judges can test, which must stay up until judging ends on 10 Nov
  - all materials in English
- Only synthetic data is allowed, and the project must be new work (built after 29 Sep).
- Judging: five criteria at 20% each:
  - technical implementation, including use of the "hero" tech: Microsoft Foundry, Agent
    Framework, Azure MCP, GitHub Copilot agent mode, Azure services
  - agentic design (multi-agent, MCP)
  - real-world impact
  - UX and demo presentation
  - fit to category
- Prize targets: Grand Prize, plus the **Best Multi-Agent System** or Best Use of Foundry category prize.
- The brief describes one pipeline in five stages: ingest → interpret → explain → render →
  personalise (analyst vs casual fan, favourite club or player, language).

## The pitch
Every key moment is explained as a decision: what the player did compared with the options they
had, and how much that moved the match, measured in expected goals and win probability.
Most entries will be agents narrating a scoreline. Ours has real football analytics underneath, and
every number is checked against the simulator's known true values.

## What's built (all tested, `python -m pytest -q` → 11 passed)
1. **`itg/sim`: synthetic match engine.**
   - Six fictional clubs with styles, formations and squads (player names unique across the league).
   - StatsBomb-style events, each on-ball event with a freeze frame of all 22 players (position
     and speed).
   - Players choose between options with a softmax, so they sometimes make the wrong choice.
   - The engine's own xG/xT/pass models (`*_truth` in `pitch.py`) are hidden ground truth.
   - Headline stats match real top-flight ranges (`scripts/validate_sim.py`).
   - Live feed: `python -m itg.sim match --stream --speed 60`.
2. **`itg/analytics`: learned models + counterfactual engine + live analyst.**
   - `features.py`: shared features for training and for hypothetical options.
   - `models.py`: xG, pass success, carry retention (sklearn gradient boosting); Markov xT; a
     Poisson in-play win-probability model.
   - `decisions.py`: rebuilds every option from the freeze frame and values each one;
     decision value = EV(chosen) − EV(best alternative), turned into win-probability cost.
   - `live.py`: `LiveAnalyst.process(event)` emits ranked insights with the numbers attached:
     goal, big_chance_missed, missed_option, great_decision, difficult_pass, momentum_shift,
     sprint, red_card, milestone. It also keeps team and player state and the win-probability
     timeline.
   - `explain.py`: template sentences in analyst and casual modes. **This is also the fallback
     narration if an LLM call fails.**
   - Trained bundle: `models/bundle.pkl`. Metrics: `reports/model_report.json`; xG correlation
     with truth 0.94, xT 0.90, pass model AUC 0.78; full table in README.
   - Commands: `python scripts/train_models.py` (needs `python -m itg.sim batch --n 400` first),
     then `python -m itg.analytics analyse <events.jsonl> [--mode casual]`.

## Rules to keep
- The analytics code must never import the `*_truth` functions, except in evaluation scripts.
- Fictional clubs and players only. No real names, logos or Premier League branding in the demo.
- Agents narrate facts that come out of `LiveAnalyst` insights and state. They must not invent
  numbers. A grounding check should verify that numbers in the narration match the facts.
- Secrets go in `.env`, never committed and never pasted into chat. Make the model provider a
  config setting: Foundry for the demo and submission, GitHub Models as a free backup for development.
- Model bundles are sklearn pickles, so retrain locally if the sklearn version differs
  (it takes a few minutes).

## Known issues
- In the simulator, more shots come from 6–11 m and fewer from long range than in real football.
- Most flagged `missed_option` moments are "should have shot". The moment-picking agent should
  choose a varied mix, and favour pass-vs-pass and speculative-shot cases.
- Pressures are slightly high and take-ons slightly low compared with real ranges. Fine for now.

## Next steps, in order
1. **Azure/Foundry.** Jono has a free Azure subscription. Still to do: budget alert, a Foundry
   project, one small "mini" chat model deployed, endpoint and deployment name in `.env`.
2. **MCP server** (`itg/mcp`) exposing the analyst as tools: `get_match_state`,
   `get_player_stats`, `get_recent_insights`, `explain_decision(event_id)`,
   `what_if(event_id)` (win probability if the best option had been taken).
3. **Agents (Microsoft Agent Framework, Python) on Foundry:**
   - Moment Detector: picks which insights go to air, with variety and rate limits.
   - Tactical Explainer: control vs chaos, pressure shifts, rhythm.
   - Narrator: live lines.
   - Personalizer: analyst vs casual, favourite club or player, language.
   - Recap: post-match summary in several languages.
   - Orchestrator: shared state, retries, fallback to `explain.py` templates.
   The multi-agent prize explicitly judges distinct roles, handoffs and failure recovery.
4. **Render.** A timed overlay JSON schema, plus a FastAPI + WebSocket front end:
   - live feed with overlay cards
   - decision overlay (pitch with the player's option arrows and their values)
   - win-probability line
   - mode, club, player and language toggles
5. **Azure deploy and polish.** Container Apps scaled to zero, Event Hubs for the ingest step if
   time allows, tracing (App Insights / Foundry), CI, README architecture diagram.
6. **Demo video.** One 20-second "money shot": the same moment as an analyst breakdown, then as a
   Spanish-language story for a casual fan following one player.

## Jono's time this week
Long build sessions on **Thu, Sat and Mon**, evenings on Wed, Fri and Sun (he's working days),
and Tue is buffer before submitting. Aim for one thin end-to-end path working first:
event → analyst → one agent → overlay in the browser. Then build out from there.

## Housekeeping
- Public repo: https://github.com/jonathan-cronin/inside-the-game (branch `main`). Commits use the
  GitHub noreply email (set in local git config).
- `_to_delete/` holds old setup bundles and is safe to delete (it's in .gitignore).
- `pip install -r requirements.txt` (numpy, pandas, scikit-learn, scipy, pytest).
