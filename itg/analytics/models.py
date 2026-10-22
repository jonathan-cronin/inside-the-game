"""Learned models: xG, pass success, carry retention, xT and win probability."""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor

from .features import CARRY_FEATURES, PASS_FEATURES, SHOT_FEATURES


# --------------------------------------------------------------------- classifiers
@dataclass
class Classifier:
    features: list[str]
    name: str
    model: HistGradientBoostingClassifier | None = None

    def fit(self, df: pd.DataFrame, target: str) -> "Classifier":
        self.model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=24,
                                                    min_samples_leaf=40, l2_regularization=1.0,
                                                    early_stopping=True, validation_fraction=0.15,
                                                    random_state=0)
        self.model.fit(df[self.features].to_numpy(float), df[target].to_numpy(int))
        return self

    def predict(self, rows) -> np.ndarray:
        if isinstance(rows, dict):
            rows = [rows]
        X = np.array([[r[f] for f in self.features] for r in rows], float) if isinstance(rows, list) \
            else rows[self.features].to_numpy(float)
        return self.model.predict_proba(X)[:, 1]


def xg_model() -> Classifier:
    return Classifier(SHOT_FEATURES, "xg")


def pass_model() -> Classifier:
    return Classifier(PASS_FEATURES, "pass_success")


def carry_model() -> Classifier:
    return Classifier(CARRY_FEATURES, "carry_retention")


# ------------------------------------------------------------------------------ xT
@dataclass
class XTGrid:
    """Expected threat (Singh, 2018), learned as a Markov chain over a 16 x 12 grid of the pitch."""
    nx: int = 16
    ny: int = 12
    grid: np.ndarray | None = None
    iterations: int = 0

    def cell(self, x, y):
        cx = np.clip((np.asarray(x) / 120.0 * self.nx).astype(int), 0, self.nx - 1)
        cy = np.clip((np.asarray(y) / 80.0 * self.ny).astype(int), 0, self.ny - 1)
        return cx, cy

    def fit(self, moves: pd.DataFrame, shot_xg: np.ndarray | None = None, tol: float = 1e-6) -> "XTGrid":
        nx, ny = self.nx, self.ny
        shots = moves[moves.kind == "shot"]
        mv = moves[moves.kind == "move"]
        sx, sy = self.cell(shots.x.to_numpy(), shots.y.to_numpy())
        mx, my = self.cell(mv.x.to_numpy(), mv.y.to_numpy())
        n_shot = np.zeros((nx, ny))
        n_goal = np.zeros((nx, ny))
        n_move = np.zeros((nx, ny))
        np.add.at(n_shot, (sx, sy), 1)
        # use the learned xG (less noisy than raw goals) for shot quality per cell
        np.add.at(n_goal, (sx, sy), shot_xg if shot_xg is not None else shots.ok.to_numpy())
        np.add.at(n_move, (mx, my), 1)
        total = n_shot + n_move
        s = np.divide(n_shot, total, out=np.zeros_like(total), where=total > 0)
        m = np.divide(n_move, total, out=np.zeros_like(total), where=total > 0)
        g = np.divide(n_goal, n_shot, out=np.zeros_like(n_shot), where=n_shot > 0)
        # transition matrix of *successful* moves; failures carry no value
        ok = mv[mv.ok == 1]
        ox, oy = self.cell(ok.x.to_numpy(), ok.y.to_numpy())
        ex, ey = self.cell(ok.ex.to_numpy(), ok.ey.to_numpy())
        T = np.zeros((nx * ny, nx * ny))
        np.add.at(T, (ox * ny + oy, ex * ny + ey), 1)
        denom = n_move.reshape(-1)[:, None]
        T = np.divide(T, denom, out=np.zeros_like(T), where=denom > 0)
        xt = np.zeros(nx * ny)
        sg = (s * g).reshape(-1)
        mm = m.reshape(-1)
        for it in range(200):
            new = sg + mm * (T @ xt)
            if np.abs(new - xt).max() < tol:
                xt = new
                break
            xt = new
        self.iterations = it + 1
        self.grid = xt.reshape(nx, ny)
        return self

    def value(self, x, y) -> float:
        """Bilinear interpolation between cell centres, for smooth values."""
        gx = float(np.clip(x / 120.0 * self.nx - 0.5, 0, self.nx - 1))
        gy = float(np.clip(y / 80.0 * self.ny - 0.5, 0, self.ny - 1))
        x0, y0 = int(gx), int(gy)
        x1, y1 = min(x0 + 1, self.nx - 1), min(y0 + 1, self.ny - 1)
        fx, fy = gx - x0, gy - y0
        g = self.grid
        return float((1 - fx) * (1 - fy) * g[x0, y0] + fx * (1 - fy) * g[x1, y0]
                     + (1 - fx) * fy * g[x0, y1] + fx * fy * g[x1, y1])


# ---------------------------------------------------------------- win probability
WP_FEATURES = ["is_home", "attack", "opp_defence", "score_diff", "red_diff", "minute", "recent_xg_share"]
MATCH_MINUTES = 95.0


@dataclass
class WinProbability:
    """In-play win probability from two Poisson goal processes.

    The scoring rate for each team in the time left depends on pre-match
    strength (attacking xG for, opponent xG conceded), the scoreline (teams
    behind attack more), red cards and recent dominance. The final result is
    then the current score plus independent Poisson goals for each side,
    which makes "what if this had gone in?" a one-line calculation.
    """
    ratings: dict[str, dict] = field(default_factory=dict)
    model: PoissonRegressor | None = None

    def fit(self, rows: pd.DataFrame) -> "WinProbability":
        self.model = PoissonRegressor(alpha=1e-4, max_iter=500)
        X = self._design(rows)
        rate = rows["goals_after"] / rows["t_left"]
        self.model.fit(X, rate, sample_weight=rows["t_left"])
        return self

    @staticmethod
    def _design(df: pd.DataFrame) -> np.ndarray:
        sd = df["score_diff"].clip(-3, 3)
        return np.column_stack([
            df["is_home"], np.log(df["attack"]), np.log(df["opp_defence"]),
            sd, sd * df["minute"] / 90.0, df["red_diff"], df["recent_xg_share"] - 0.5,
        ])

    def rates(self, state: dict) -> tuple[float, float]:
        """Expected goals in the time remaining for home and away."""
        t_left = max(1.0, MATCH_MINUTES - state["minute"])
        avg = {k: float(np.mean([r[k] for r in self.ratings.values()])) for k in ("xg_for", "xg_against")}
        rating = lambda team: self.ratings.get(team, avg)  # noqa: E731  (unknown club -> league average)
        rows = []
        for side, opp, is_home, sd, rd, share in (
            ("home", "away", 1, state["home_goals"] - state["away_goals"], state.get("red_diff", 0),
             state.get("recent_xg_share", 0.5)),
            ("away", "home", 0, state["away_goals"] - state["home_goals"], -state.get("red_diff", 0),
             1 - state.get("recent_xg_share", 0.5)),
        ):
            rows.append({"is_home": is_home, "attack": rating(state[side])["xg_for"],
                         "opp_defence": rating(state[opp])["xg_against"], "score_diff": sd,
                         "red_diff": rd, "minute": state["minute"], "recent_xg_share": share})
        lam = self.model.predict(self._design(pd.DataFrame(rows))) * t_left
        return float(lam[0]), float(lam[1])

    def probabilities(self, state: dict, extra_home: int = 0, extra_away: int = 0) -> dict:
        lh, la = self.rates(state)
        dh = state["home_goals"] + extra_home
        da = state["away_goals"] + extra_away
        ks = np.arange(0, 11)
        ph = np.exp(-lh) * lh ** ks / np.array([math.factorial(int(k)) for k in ks])
        pa = np.exp(-la) * la ** ks / np.array([math.factorial(int(k)) for k in ks])
        joint = np.outer(ph, pa)
        diff = (dh + ks)[:, None] - (da + ks)[None, :]
        out = {"home": float(joint[diff > 0].sum()), "draw": float(joint[diff == 0].sum()),
               "away": float(joint[diff < 0].sum())}
        z = sum(out.values())
        return {k: v / z for k, v in out.items()}


# ----------------------------------------------------------------------- bundle
@dataclass
class ModelBundle:
    xg: Classifier
    pass_success: Classifier
    carry: Classifier
    xt: XTGrid
    wp: WinProbability
    report: dict = field(default_factory=dict)

    def save(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "ModelBundle":
        with open(path, "rb") as fh:
            return pickle.load(fh)
