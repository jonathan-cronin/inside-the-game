"""Fictional clubs and players.

Every club, player and competition here is invented, so demos never use
real trademarks or real people. Squads are generated deterministically
from the club id, so "Riverside Athletic" always has the same players
across matches (needed for favourite-player personalisation).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

# (u, v): u = depth from own goal (0 = back line, 1 = centre-forward),
#         v = width (0 = left touchline, 1 = right), in the team's own frame.
FORMATIONS: dict[str, dict[str, tuple[float, float]]] = {
    "4-3-3": {
        "GK": (0, 0.5), "LB": (0.05, 0.1), "LCB": (0.0, 0.37), "RCB": (0.0, 0.63), "RB": (0.05, 0.9),
        "CDM": (0.33, 0.5), "LCM": (0.48, 0.3), "RCM": (0.48, 0.7),
        "LW": (0.85, 0.12), "ST": (1.0, 0.5), "RW": (0.85, 0.88),
    },
    "4-4-2": {
        "GK": (0, 0.5), "LB": (0.05, 0.1), "LCB": (0.0, 0.37), "RCB": (0.0, 0.63), "RB": (0.05, 0.9),
        "LM": (0.55, 0.1), "LCM": (0.42, 0.38), "RCM": (0.42, 0.62), "RM": (0.55, 0.9),
        "LS": (1.0, 0.4), "RS": (1.0, 0.6),
    },
    "4-2-3-1": {
        "GK": (0, 0.5), "LB": (0.05, 0.1), "LCB": (0.0, 0.37), "RCB": (0.0, 0.63), "RB": (0.05, 0.9),
        "LDM": (0.32, 0.4), "RDM": (0.32, 0.6),
        "LAM": (0.75, 0.15), "CAM": (0.72, 0.5), "RAM": (0.75, 0.85), "ST": (1.0, 0.5),
    },
    "3-5-2": {
        "GK": (0, 0.5), "LCB": (0.0, 0.25), "CB": (0.0, 0.5), "RCB": (0.0, 0.75),
        "LWB": (0.45, 0.05), "RWB": (0.45, 0.95),
        "CDM": (0.32, 0.5), "LCM": (0.5, 0.33), "RCM": (0.5, 0.67),
        "LS": (1.0, 0.4), "RS": (1.0, 0.6),
    },
}

ROLE_GROUP = {
    "GK": "GK",
    "LCB": "CB", "RCB": "CB", "CB": "CB",
    "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
    "CDM": "DM", "LDM": "DM", "RDM": "DM",
    "LCM": "CM", "RCM": "CM",
    "LM": "W", "RM": "W", "LW": "W", "RW": "W", "LAM": "AM", "RAM": "AM", "CAM": "AM",
    "ST": "ST", "LS": "ST", "RS": "ST",
}

# Attribute offsets relative to club strength, per role group.
ROLE_PROFILE = {
    "GK": dict(passing=-0.12, vision=-0.1, finishing=-0.45, dribbling=-0.35, pace=-0.15, stamina=-0.05, defending=-0.2, aerial=0.05, goalkeeping=0.12),
    "CB": dict(passing=-0.05, vision=-0.08, finishing=-0.25, dribbling=-0.15, pace=-0.04, stamina=0.0, defending=0.12, aerial=0.12, goalkeeping=-0.6),
    "FB": dict(passing=0.0, vision=-0.03, finishing=-0.2, dribbling=0.02, pace=0.1, stamina=0.1, defending=0.03, aerial=-0.05, goalkeeping=-0.6),
    "DM": dict(passing=0.07, vision=0.05, finishing=-0.12, dribbling=-0.02, pace=-0.03, stamina=0.1, defending=0.08, aerial=0.02, goalkeeping=-0.6),
    "CM": dict(passing=0.1, vision=0.08, finishing=-0.06, dribbling=0.04, pace=0.0, stamina=0.1, defending=0.0, aerial=-0.03, goalkeeping=-0.6),
    "AM": dict(passing=0.1, vision=0.12, finishing=0.02, dribbling=0.1, pace=0.04, stamina=0.0, defending=-0.12, aerial=-0.1, goalkeeping=-0.6),
    "W":  dict(passing=0.04, vision=0.05, finishing=0.0, dribbling=0.12, pace=0.14, stamina=0.03, defending=-0.12, aerial=-0.1, goalkeeping=-0.6),
    "ST": dict(passing=-0.02, vision=0.02, finishing=0.14, dribbling=0.05, pace=0.07, stamina=-0.02, defending=-0.18, aerial=0.08, goalkeeping=-0.6),
}

SHIRT = {"GK": 1, "RB": 2, "LB": 3, "RWB": 2, "LWB": 3, "LCB": 5, "RCB": 4, "CB": 6, "CDM": 6, "LDM": 6,
         "RDM": 4, "LCM": 8, "RCM": 10, "CAM": 10, "LAM": 11, "RAM": 7, "LM": 11, "RM": 7, "LW": 11,
         "RW": 7, "ST": 9, "LS": 9, "RS": 19}

FIRST_NAMES = [
    "Aaron", "Adebayo", "Alfie", "Amadou", "Andre", "Arlo", "Ben", "Callum", "Carlos", "Cian",
    "Dani", "Darius", "Diego", "Elias", "Emeka", "Enzo", "Ethan", "Felipe", "Finn", "Gabriel",
    "Hamza", "Harvey", "Hugo", "Ibrahim", "Isaac", "Jaden", "Jamal", "Javier", "Joao", "Jonah",
    "Kai", "Kenji", "Kofi", "Leo", "Lucas", "Luka", "Marco", "Mateo", "Max", "Mohammed",
    "Nathan", "Niko", "Noah", "Oisin", "Omar", "Oscar", "Pablo", "Rafael", "Reece", "Rio",
    "Ryo", "Sami", "Sebastian", "Tariq", "Theo", "Tomas", "Vinicius", "Yusuf", "Zak", "Zion",
]
LAST_NAMES = [
    "Abara", "Achterberg", "Adeyemi", "Almeida", "Andersson", "Banda", "Barros", "Bello", "Brennan", "Calloway",
    "Castellano", "Chukwu", "Costa", "Dalgaard", "Diallo", "Doyle", "Eke", "Ferreira", "Fofana", "Gallagher",
    "Garrido", "Hollis", "Hartley", "Ibarra", "Igwe", "Jansen", "Kamara", "Kovac", "Lindqvist", "Lopes",
    "Mansour", "Mbeki", "Moreau", "Nakamura", "Ndiaye", "Novak", "Okafor", "Olsen", "Ortega", "Petrov",
    "Quinn", "Ramos", "Reyes", "Sato", "Silva", "Sorensen", "Tanaka", "Traore", "Udeh", "Valdez",
    "Varga", "Wakefield", "Whitfield", "Yamamoto", "Yilmaz", "Zelenko", "Zeller", "Brightwell", "Ashworth", "Kettering",
]
_BLOCKED: set[tuple[str, str]] = set()  # add (first, last) pairs here to avoid real-player clashes

CLUBS = [
    dict(id="RVS", name="Riverside Athletic", short="Riverside", formation="4-3-3", strength=0.72,
         colours=("#1B4F9C", "#FFFFFF"),
         style=dict(possession=0.8, directness=0.3, press=0.75, block_height=0.75, width=0.85)),
    dict(id="NGT", name="Northgate Rovers", short="Northgate", formation="4-4-2", strength=0.66,
         colours=("#C8102E", "#111111"),
         style=dict(possession=0.35, directness=0.75, press=0.45, block_height=0.4, width=0.9)),
    dict(id="KGM", name="Kingsmere FC", short="Kingsmere", formation="4-2-3-1", strength=0.7,
         colours=("#6A1B9A", "#F2C94C"),
         style=dict(possession=0.6, directness=0.5, press=0.6, block_height=0.55, width=0.8)),
    dict(id="HBR", name="Harbour City", short="Harbour", formation="3-5-2", strength=0.64,
         colours=("#00897B", "#FFFFFF"),
         style=dict(possession=0.45, directness=0.55, press=0.4, block_height=0.3, width=0.95)),
    dict(id="ASV", name="Ashford Vale", short="Ashford", formation="4-4-2", strength=0.6,
         colours=("#2E7D32", "#FFEB3B"),
         style=dict(possession=0.3, directness=0.8, press=0.35, block_height=0.3, width=0.85)),
    dict(id="WBK", name="Westbrook United", short="Westbrook", formation="4-3-3", strength=0.68,
         colours=("#212121", "#FF6F00"),
         style=dict(possession=0.65, directness=0.45, press=0.85, block_height=0.8, width=0.8)),
]
CLUB_BY_ID = {c["id"]: c for c in CLUBS}
ATTRS = ["passing", "vision", "finishing", "dribbling", "pace", "stamina", "defending", "aerial", "goalkeeping"]


@dataclass
class Player:
    id: str
    name: str
    short_name: str
    number: int
    role: str
    team_id: str
    attrs: dict[str, float]

    @property
    def group(self) -> str:
        return ROLE_GROUP[self.role]

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "number": self.number, "position": self.role}


@dataclass
class Team:
    id: str
    name: str
    short: str
    formation: str
    style: dict[str, float]
    colours: tuple[str, str]
    starters: list[Player]
    bench: list[Player] = field(default_factory=list)

    def public(self) -> dict:
        return {"id": self.id, "name": self.name}

    def all_players(self) -> list[Player]:
        return self.starters + self.bench


def _seed_for(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def build_team(club_id: str) -> Team:
    club = CLUB_BY_ID[club_id]
    rng = np.random.default_rng(_seed_for(club_id))
    used_names: set[str] = set()
    used_numbers: set[int] = set()

    def make(role: str, idx: int, star: bool = False) -> Player:
        group = ROLE_GROUP[role]
        prof = ROLE_PROFILE[group]
        boost = 0.07 if star else 0.0
        attrs = {a: float(np.clip(club["strength"] + prof[a] + boost + rng.normal(0, 0.05), 0.2, 0.97)) for a in ATTRS}
        while True:
            first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
            full = f"{first} {last}"
            if full not in used_names and (first, last) not in _BLOCKED:
                used_names.add(full)
                break
        num = SHIRT.get(role, 12)
        while num in used_numbers:
            num = int(rng.integers(12, 40))
        used_numbers.add(num)
        return Player(id=f"{club_id}-{num:02d}", name=full, short_name=last, number=num,
                      role=role, team_id=club_id, attrs=attrs)

    roles = list(FORMATIONS[club["formation"]].keys())
    star_idx = int(rng.integers(5, len(roles)))  # one standout outfield player
    starters = [make(r, i, star=(i == star_idx)) for i, r in enumerate(roles)]
    bench_roles = ["GK", _first(roles, "CB"), _first(roles, ("CM", "DM")), _first(roles, ("W", "AM")), _first(roles, "ST")]
    bench = [make(r, 20 + i) for i, r in enumerate(bench_roles)]
    for p in bench:  # bench is slightly weaker
        for a in p.attrs:
            p.attrs[a] = max(0.2, p.attrs[a] - 0.03)
    # names are assigned league-wide so no two players share a name (needed for "follow my player")
    for p, (full, last) in zip(starters + bench, _league_names()[club_id]):
        p.name, p.short_name = full, last
    return Team(id=club_id, name=club["name"], short=club["short"], formation=club["formation"],
                style=dict(club["style"]), colours=club["colours"], starters=starters, bench=bench)


_NAMES_CACHE: dict[str, list[tuple[str, str]]] = {}


def _league_names() -> dict[str, list[tuple[str, str]]]:
    """Deterministic, league-wide unique names: 16 per club, in CLUBS order."""
    if not _NAMES_CACHE:
        rng = np.random.default_rng(20261027)
        used: set[str] = set()
        for club in CLUBS:
            names = []
            while len(names) < 16:
                first, last = str(rng.choice(FIRST_NAMES)), str(rng.choice(LAST_NAMES))
                full = f"{first} {last}"
                if full in used or (first, last) in _BLOCKED:
                    continue
                used.add(full)
                names.append((full, last))
            _NAMES_CACHE[club["id"]] = names
    return _NAMES_CACHE


def _first(roles, groups) -> str:
    groups = (groups,) if isinstance(groups, str) else groups
    for r in roles:
        if ROLE_GROUP[r] in groups:
            return r
    return roles[-1]
