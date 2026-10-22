"""Pitch geometry and the simulator's hidden "ground truth" models.

Coordinates follow the StatsBomb convention: a 120 x 80 pitch, and every
event is expressed from the acting team's point of view, attacking towards
x = 120. Distances used by the models are converted to metres
(105 m x 68 m pitch).

The *_truth functions are the generator's hidden laws of football. The
analytics layer never imports them: it has to learn xG, xT and pass
difficulty from the generated events, and these functions are then used
to check how well it recovered them.
"""
from __future__ import annotations

import math

import numpy as np

PITCH_X, PITCH_Y = 120.0, 80.0
X_M, Y_M = 105.0 / 120.0, 68.0 / 80.0
GOAL = np.array([120.0, 40.0])
GOAL_HALF_WIDTH_M = 3.66
BOX_X = 102.0
BOX_Y = (18.0, 62.0)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def dist_m(a, b) -> float:
    return math.hypot((a[0] - b[0]) * X_M, (a[1] - b[1]) * Y_M)


def dists_m(p, pts: np.ndarray) -> np.ndarray:
    """Distances in metres from point p to each row of pts."""
    if len(pts) == 0:
        return np.array([])
    return np.hypot((pts[:, 0] - p[0]) * X_M, (pts[:, 1] - p[1]) * Y_M)


def mirror(p) -> np.ndarray:
    return np.array([PITCH_X - p[0], PITCH_Y - p[1]])


def clip_pitch(p, margin: float = 0.5) -> np.ndarray:
    return np.array([
        min(max(p[0], margin), PITCH_X - margin),
        min(max(p[1], margin), PITCH_Y - margin),
    ])


def in_box(p) -> bool:
    return p[0] >= BOX_X and BOX_Y[0] <= p[1] <= BOX_Y[1]


def goal_angle(loc) -> float:
    """Angle (radians) subtended by the goal mouth from loc."""
    x = max((PITCH_X - loc[0]) * X_M, 0.05)
    y = (loc[1] - 40.0) * Y_M
    return abs(math.atan2(GOAL_HALF_WIDTH_M - y, x) - math.atan2(-GOAL_HALF_WIDTH_M - y, x))


def segment_dists(a, b, pts: np.ndarray):
    """Distance (m) from each point to segment a->b, plus projection t in [0, 1]."""
    if len(pts) == 0:
        return np.array([]), np.array([])
    a = np.asarray(a, float)
    ab = np.asarray(b, float) - a
    l2 = float(ab @ ab)
    if l2 < 1e-9:
        return dists_m(a, pts), np.zeros(len(pts))
    t = np.clip(((pts - a) @ ab) / l2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = np.hypot((pts[:, 0] - proj[:, 0]) * X_M, (pts[:, 1] - proj[:, 1]) * Y_M)
    return d, t


def in_triangle(pts: np.ndarray, a, b, c) -> np.ndarray:
    if len(pts) == 0:
        return np.zeros(0, bool)

    def cross(p1, p2, p3):
        return (p1[:, 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[:, 1] - p3[1])

    d1 = cross(pts, a, b)
    d2 = cross(pts, b, c)
    d3 = cross(pts, c, a)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def shot_blockers(loc, defenders: np.ndarray) -> int:
    """Number of outfield defenders inside the triangle from the ball to the posts."""
    posts = (np.array([120.0, 40.0 - 4.3]), np.array([120.0, 40.0 + 4.3]))
    return int(in_triangle(defenders, np.asarray(loc), posts[0], posts[1]).sum())


# ---------------------------------------------------------------------------
# Hidden ground-truth models
# ---------------------------------------------------------------------------

def xt_truth(loc) -> float:
    """Value of holding the ball at loc: chance the possession ends in a goal."""
    d = dist_m(loc, GOAL)
    lateral = (loc[1] - 40.0) * Y_M
    width = 9.0 + 0.35 * d
    return 0.006 + 0.45 * math.exp(-d / 12.0) * math.exp(-((lateral / width) ** 2))


def xg_truth(loc, *, header=False, pressure_m=10.0, blockers=0, finishing=0.65,
             keeper=0.7, set_piece=None) -> float:
    if set_piece == "penalty":
        return float(np.clip(0.76 + 0.15 * (finishing - 0.65) - 0.15 * (keeper - 0.7), 0.6, 0.9))
    d = dist_m(loc, GOAL)
    z = (-1.7 + 1.45 * goal_angle(loc) - 0.105 * d - 0.5 * blockers
         - 1.4 * math.exp(-pressure_m / 2.2) + 1.4 * (finishing - 0.65) - 1.0 * (keeper - 0.7))
    if header:
        z -= 1.5
    if set_piece == "free_kick":
        z -= 0.35
    return sigmoid(z)


def pass_success_truth(*, length_m, lane_m, mark_m, press_m, passing, stamina,
                       forward_units, high=False, through=False) -> float:
    z = (3.7 - 0.06 * length_m
         - 1.7 * math.exp(-lane_m / 1.8)
         - 0.9 * math.exp(-mark_m / 2.0)
         - 0.6 * math.exp(-press_m / 2.5)
         + 3.0 * (passing - 0.65)
         - 1.2 * (1.0 - stamina)
         - 0.012 * max(0.0, forward_units))
    if high:
        z -= 0.35
    if through:
        z -= 0.4
    return sigmoid(z)
