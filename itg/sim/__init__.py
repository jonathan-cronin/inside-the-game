"""Synthetic, football-realistic match data generator."""
from .engine import Match, MatchSimulator, simulate_match
from .teams import CLUBS, build_team

__all__ = ["Match", "MatchSimulator", "simulate_match", "CLUBS", "build_team"]
