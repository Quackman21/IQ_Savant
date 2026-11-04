"""Scoreboard logic: which games to show, how important each one is, and who
can clinch. Pure Python/numpy (no Streamlit) so the nightly pipeline, tests
and any future front end can all reuse it.

Importance
----------
Regular season: simulate the rest of the season many times. For each game,
replay every simulation twice -- once with the home team winning, once with
the away team winning -- using the *same* random draws. The importance is the
total change in playoff odds across every team in MLB between those two
worlds. A September game between two wild-card rivals moves several teams'
odds; a game between two eliminated teams moves nothing.

Postseason: the swing in each team's chance of winning the World Series
(series odds at 50/50 per game, scaled by the rounds still to come). Clinch
and elimination games rank highest; the World Series outranks earlier rounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np

from core.mlb_api import AL, NL, POSTSEASON, REGULAR

ET = ZoneInfo("America/New_York")
SWITCH_TIME = (10, 30)      # before 10:30 AM ET show yesterday, after show today
TOP_N = 5
N_SIMS = 4000
REGRESS_GAMES = 40          # pull records toward .500 early in the season
HOME_EDGE = 0.04
ROUND_DIGITS = 3            # importance rounded to 0.1 pt before tie-breaking
VOID_STATES = {"Postponed", "Cancelled", "Canceled"}
ROUNDS_AFTER = {"W": 0, "L": 1, "D": 2, "F": 3}
DEFAULT_LENGTH = {"F": 3, "D": 5, "L": 7, "W": 7}
ROUND_LABEL = {"F": "WC", "D": "DS", "L": "LCS", "W": "WS"}


# ----------------------------------------------------------------------------
# Game parsing
# ----------------------------------------------------------------------------
@dataclass
class Game:
    pk: int
    gtype: str
    day: str                      # officialDate, YYYY-MM-DD
    start_utc: str
    home: int
    away: int
    state: str                    # pre | live | final | void
    winner: int | None = None
    home_score: int | None = None
    away_score: int | None = None
    series_game: int | None = None
    series_len: int | None = None
    inning: int | None = None
    inning_state: str | None = None
    sched_innings: int = 9
    # filled in by the ranking step
    importance: float = 0.0
    max_pct: float = 0.0
    clinch: dict = field(default_factory=dict)      # team id -> what a win clinches
    series_wins: dict = field(default_factory=dict)  # team id -> wins to display
    series_need: int | None = None

    @property
    def played(self) -> bool:
        return self.state == "final" and self.winner is not None


def parse_game(g: dict) -> Game:
    st = g.get("status", {})
    detailed = st.get("detailedState", "")
    abstract = st.get("abstractGameState", "")
    home, away = g["teams"]["home"], g["teams"]["away"]
    if detailed in VOID_STATES:
        state = "void"
    elif abstract == "Final":
        state = "final"
    elif abstract == "Live":
        state = "live"
    else:
        state = "pre"
    winner = None
    if home.get("isWinner"):
        winner = home["team"]["id"]
    elif away.get("isWinner"):
        winner = away["team"]["id"]
    if state == "final" and winner is None:      # tie / suspended: not a result
        state = "void" if detailed in VOID_STATES else "pre"
    ls = g.get("linescore", {}) or {}
    return Game(
        pk=g["gamePk"], gtype=g["gameType"], day=g.get("officialDate", ""),
        start_utc=g.get("gameDate", ""),
        home=home["team"]["id"], away=away["team"]["id"], state=state,
        winner=winner, home_score=home.get("score"), away_score=away.get("score"),
        series_game=g.get("seriesGameNumber"), series_len=g.get("gamesInSeries"),
        inning=ls.get("currentInning"), inning_state=ls.get("inningState"),
        sched_innings=ls.get("scheduledInnings") or g.get("scheduledInnings") or 9,
    )


def dedupe(games: list[Game]) -> list[Game]:
    """A postponed game can appear on two dates with one gamePk; keep the real one."""
    best: dict[int, Game] = {}
    rank = {"final": 3, "live": 2, "pre": 1, "void": 0}
    for g in games:
        cur = best.get(g.pk)
        if cur is None or (rank[g.state], g.day) > (rank[cur.state], cur.day):
            best[g.pk] = g
    return list(best.values())


# ----------------------------------------------------------------------------
# Which day to show
# ----------------------------------------------------------------------------
def display_date(now: datetime | None = None) -> date:
    now = (now or datetime.now(ET)).astimezone(ET)
    switch = now.replace(hour=SWITCH_TIME[0], minute=SWITCH_TIME[1], second=0, microsecond=0)
    return now.date() if now >= switch else now.date() - timedelta(days=1)


def pick_day(games: list[Game], target: date) -> str | None:
    """The target day if it has games, else the most recent day that had any
    (offseason / off-days fall back to the last games played)."""
    t = target.isoformat()
    days = {g.day for g in games if g.state != "void"}
    if t in days:
        return t
    earlier = sorted(d for d in days if d < t)
    return earlier[-1] if earlier else None


# ----------------------------------------------------------------------------
# Regular season: standings, simulation, importance, clinching
# ----------------------------------------------------------------------------
def standings_before(games: list[Game], day: str) -> dict[int, list[int]]:
    """team -> [wins, losses] from regular-season games played before `day`."""
    rec: dict[int, list[int]] = {}
    for g in games:
        if g.gtype != REGULAR:
            continue
        for t in (g.home, g.away):
            rec.setdefault(t, [0, 0])
        if g.played and g.day < day:
            loser = g.away if g.winner == g.home else g.home
            rec[g.winner][0] += 1
            rec[loser][1] += 1
    return rec


def remaining_before(games: list[Game], day: str) -> list[Game]:
    """Regular-season games not yet counted as of the start of `day`."""
    return [g for g in games
            if g.gtype == REGULAR and g.state != "void" and not (g.played and g.day < day)]


def _playoffs(wins: np.ndarray, noise: np.ndarray, teams: dict, idx: dict) -> np.ndarray:
    """N x T bool: which teams make the 12-team field in each simulation."""
    score = wins + noise
    made = np.zeros(score.shape, dtype=bool)
    for league in (AL, NL):
        cols = [idx[t] for t in idx if teams.get(t, {}).get("league") == league]
        if not cols:
            continue
        divs: dict[int, list[int]] = {}
        for t, c in idx.items():
            if teams.get(t, {}).get("league") == league:
                divs.setdefault(teams[t]["division"], []).append(c)
        for dcols in divs.values():
            winner = np.array(dcols)[np.argmax(score[:, dcols], axis=1)]
            made[np.arange(len(score)), winner] = True
        wc = np.where(made[:, cols], -np.inf, score[:, cols])
        k = min(3, max(len(cols) - len(divs), 0))
        if k:
            top = np.argpartition(-wc, k - 1, axis=1)[:, :k]
            made[np.arange(len(score))[:, None], np.array(cols)[top]] = True
    return made


def regular_importance(games: list[Game], show: list[Game], teams: dict, day: str) -> None:
    rec = standings_before(games, day)
    remaining = remaining_before(games, day)
    team_ids = sorted(t for t in rec if t in teams)
    if not remaining or not team_ids:
        return
    idx = {t: i for i, t in enumerate(team_ids)}
    T, G = len(team_ids), len(remaining)
    W = np.array([rec[t][0] for t in team_ids], dtype=np.float32)
    L = np.array([rec[t][1] for t in team_ids], dtype=np.float32)
    strength = (W + 0.5 * REGRESS_GAMES) / (W + L + REGRESS_GAMES)

    H = np.zeros((G, T), dtype=np.float32)
    A = np.zeros((G, T), dtype=np.float32)
    for j, g in enumerate(remaining):
        if g.home in idx and g.away in idx:
            H[j, idx[g.home]] = 1
            A[j, idx[g.away]] = 1
    ph, pa = strength @ H.T, strength @ A.T
    p_home = np.where(ph + pa > 0, ph * (1 - pa) / np.maximum(ph * (1 - pa) + pa * (1 - ph), 1e-9), 0.5)
    p_home = np.clip(p_home + HOME_EDGE, 0.05, 0.95)

    rng = np.random.default_rng(int(day.replace("-", "")))   # stable within a day
    home_won = rng.random((N_SIMS, G)) < p_home
    wins = W + home_won.astype(np.float32) @ H + (~home_won).astype(np.float32) @ A
    noise = rng.random((N_SIMS, T)) * 0.5                       # random tie-breaks
    col_of = {g.pk: j for j, g in enumerate(remaining)}

    for g in show:
        j = col_of.get(g.pk)
        if j is None or g.home not in idx or g.away not in idx:
            continue
        h, a = idx[g.home], idx[g.away]
        base = wins.copy()
        base[:, h] -= home_won[:, j]
        base[:, a] -= ~home_won[:, j]
        w_home = base.copy(); w_home[:, h] += 1
        w_away = base;        w_away[:, a] += 1
        swing = _playoffs(w_home, noise, teams, idx).mean(0) - _playoffs(w_away, noise, teams, idx).mean(0)
        g.importance = float(np.abs(swing).sum()) / 2   # = total playoff odds moved

    # clinch chances use the same pre-game standings
    left = {t: 0 for t in team_ids}
    for g in remaining:
        if g.home in left: left[g.home] += 1
        if g.away in left: left[g.away] += 1
    wins_now = {t: rec[t][0] for t in team_ids}
    for g in show:
        if g.home in idx and g.away in idx:
            g.clinch = {}
            for t in (g.home, g.away):
                label = can_clinch_with_win(t, g.away if t == g.home else g.home, wins_now, left, teams)
                if label:
                    g.clinch[t] = label


def _clinched(t: int, floor: int, ceil: dict[int, int], teams: dict) -> tuple[bool, bool, bool]:
    """(division, postseason berth, best record in league) clinched if `t` finishes with
    `floor` wins and every rival wins all their remaining games. Ties go
    against `t`; rivals playing each other is ignored, so this never claims a
    clinch that isn't real."""
    league, div = teams[t]["league"], teams[t]["division"]
    above_by_div: dict[int, int] = {}
    best = True
    for o, c in ceil.items():
        if o == t or teams.get(o, {}).get("league") != league:
            continue
        if c >= floor:
            best = False
            d = teams[o]["division"]
            above_by_div[d] = above_by_div.get(d, 0) + 1
    div_clinched = above_by_div.get(div, 0) == 0
    wildcard_ahead = sum(max(0, n - 1) for n in above_by_div.values())
    # if t doesn't win its division, its division winner is one of the teams above it
    berth = div_clinched or wildcard_ahead < 3
    return div_clinched, berth, best and div_clinched


# Clinch labels (same letters as MLB's standings)
CLINCH_LABEL = {"z": "Division & Best Record", "y": "Division", "w": "Wild Card"}


def clinch_code(div: bool, berth: bool, best: bool) -> str | None:
    return "z" if best else "y" if div else "w" if berth else None


def can_clinch_with_win(t: int, opp: int, wins: dict, left: dict, teams: dict) -> str | None:
    """What a win today would newly clinch for `t` ('z', 'y', 'w'), or None."""
    ceil_now = {o: wins[o] + left[o] for o in wins}
    now = _clinched(t, wins[t], ceil_now, teams)
    ceil_after = dict(ceil_now)
    ceil_after[opp] -= 1                                    # opponent loses this game
    after = _clinched(t, wins[t] + 1, ceil_after, teams)
    # report the biggest thing newly clinched
    for code, i in (("z", 2), ("y", 0), ("w", 1)):
        if after[i] and not now[i]:
            return code
    return None


# ----------------------------------------------------------------------------
# Postseason: series state, leverage, clinching
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def series_odds(need_a: int, need_b: int) -> float:
    """P(team A wins the series) at 50/50 per game."""
    if need_a == 0:
        return 1.0
    if need_b == 0:
        return 0.0
    return 0.5 * series_odds(need_a - 1, need_b) + 0.5 * series_odds(need_a, need_b - 1)


def postseason_state(games: list[Game]) -> dict[int, tuple[dict, int]]:
    """gamePk -> ({team: wins before this game}, wins needed)."""
    series: dict[tuple, list[Game]] = {}
    for g in games:
        if g.gtype in POSTSEASON and g.state != "void":
            series.setdefault((g.gtype, frozenset((g.home, g.away))), []).append(g)
    out = {}
    for (gtype, pair), gs in series.items():
        gs.sort(key=lambda g: (g.day, g.start_utc, g.series_game or 0))
        length = next((g.series_len for g in gs if g.series_len), DEFAULT_LENGTH[gtype])
        need = length // 2 + 1
        wins = {t: 0 for t in pair}
        for g in gs:
            out[g.pk] = (dict(wins), need)
            if g.played:
                wins[g.winner] += 1
    return out


def postseason_importance(games: list[Game], show: list[Game]) -> None:
    state = postseason_state(games)
    for g in show:
        if g.pk not in state:
            continue
        pre, need = state[g.pk]
        h_need, a_need = need - pre[g.home], need - pre[g.away]
        if h_need <= 0 or a_need <= 0:
            continue
        lev = series_odds(h_need - 1, a_need) - series_odds(h_need, a_need - 1)
        g.importance = lev * 0.5 ** ROUNDS_AFTER[g.gtype]   # title odds moved
        g.series_need = need
        g.clinch = {t: ROUND_LABEL[g.gtype] for t in (g.home, g.away) if pre[t] == need - 1}
        post = dict(pre)
        if g.played:
            post[g.winner] += 1
        g.series_wins = post


def postseason_strip(season_games: list[Game], target: str, n: int = TOP_N):
    """During the postseason: the last `n` postseason games up to `target`
    (oldest -> newest) and whether `target` itself is an off day.
    Returns None when the strip should use the regular-season view."""
    games = dedupe(season_games)
    post = sorted((g for g in games if g.gtype in POSTSEASON and g.state != "void" and g.day <= target),
                  key=lambda g: (g.day, g.start_utc))
    if not post:
        return None
    on_target = [g for g in games if g.day == target and g.state != "void"]
    if on_target and not any(g.gtype in POSTSEASON for g in on_target):
        return None                                   # still regular season
    show = post[-n:]
    postseason_importance(games, show)
    off_day = not on_target and champion(games, target) is None
    return show, off_day


def series_results(games: list[Game], upto: str) -> list[dict]:
    """Every postseason series decided on or before `upto`, in order decided:
    {round, winner, wins, played, day}."""
    games = dedupe(games)
    state = postseason_state(games)
    out = []
    for g in sorted(games, key=lambda g: (g.day, g.start_utc)):
        if g.gtype not in POSTSEASON or not g.played or g.day > upto or g.pk not in state:
            continue
        pre, need = state[g.pk]
        if pre[g.winner] + 1 == need:                       # this game ended the series
            out.append({"round": g.gtype, "winner": g.winner, "wins": need,
                        "played": sum(pre.values()) + 1, "day": g.day})
    return out


def congrats_for(games: list[Game], upto: str) -> list[dict]:
    """Series clinched on the most recent game day up to `upto`; after the
    World Series, the champion's card stays up all winter."""
    done = series_results(games, upto)
    if not done:
        return []
    ws = [r for r in done if r["round"] == "W"]
    if ws:
        return ws[-1:]
    last_day = max(g.day for g in dedupe(games)
                   if g.gtype in POSTSEASON and g.played and g.day <= upto)
    return [r for r in done if r["day"] == last_day]


def champion(games: list[Game], upto: str) -> int | None:
    """World Series winner, once the clinching game is on or before `upto`."""
    ws = [g for g in games if g.gtype == "W" and g.played and g.day <= upto]
    if not ws:
        return None
    need = (next((g.series_len for g in ws if g.series_len), 7)) // 2 + 1
    tally: dict[int, int] = {}
    for g in ws:
        tally[g.winner] = tally.get(g.winner, 0) + 1
    best = max(tally, key=tally.get)
    return best if tally[best] >= need else None


# ----------------------------------------------------------------------------
# Putting it together
# ----------------------------------------------------------------------------
def rank_games(season_games: list[Game], day: str, teams: dict, top_n: int = TOP_N) -> list[Game]:
    games = dedupe(season_games)
    show = [g for g in games if g.day == day and g.state != "void"]
    reg = [g for g in show if g.gtype == REGULAR]
    post = [g for g in show if g.gtype in POSTSEASON]
    if reg:
        regular_importance(games, reg, teams, day)
    if post:
        postseason_importance(games, post)

    # tie-break: best regular-season winning % of the two teams (as of that day)
    rec = standings_before(games, day if reg else "9999-12-31")
    pct = {t: (w / (w + l) if w + l else 0.5) for t, (w, l) in rec.items()}
    for g in show:
        g.max_pct = max(pct.get(g.home, 0.5), pct.get(g.away, 0.5))
    show.sort(key=lambda g: (-round(g.importance, ROUND_DIGITS), -g.max_pct, g.start_utc))
    return show[:top_n]
