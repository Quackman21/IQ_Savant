"""Standings from MLB's feed, flattened to one dict per team.

Clinch codes follow MLB's legend:
    w = Clinched Wild Card   y = Clinched Division
    z = Clinched Division and Best Record in League
"""
from __future__ import annotations

from core.mlb_api import AL, NL

# MLB division ids, in display order (East, Central, West)
DIVISIONS = {
    AL: [(201, "AL East"), (202, "AL Central"), (200, "AL West")],
    NL: [(204, "NL East"), (205, "NL Central"), (203, "NL West")],
}
LEAGUE_NAME = {AL: "American League", NL: "National League"}
CLINCH_MAP = {"x": "w", "w": "w", "y": "y", "z": "z"}
CLINCH_TEXT = {"w": "Clinched Wild Card", "y": "Clinched Division",
               "z": "Clinched Division and Best Record in League"}
WC_SPOTS = 3
EARLIEST = "2015-04-05"        # Statcast era opening day; same 3-division format since


def wc_spots(season: int) -> int:
    """Wild-card berths per league that year (2020's 16-team field: top 2 in each
    division + 2 wild cards, so the cut sits 5 deep in the non-leader race)."""
    return 3 if season >= 2022 else 5 if season == 2020 else 2


def _split(rec: dict, kind: str, wanted: str) -> str:
    for r in rec.get("records", {}).get(kind, []) or []:
        if r.get("type") == wanted:
            return f'{r.get("wins", 0)}-{r.get("losses", 0)}'
    return "-"


def _num(x) -> str:
    return "-" if x in (None, "", "-") else str(x)


def parse(payload: dict) -> list[dict]:
    rows = []
    for block in payload.get("records", []):
        league = block.get("league", {}).get("id")
        div = block.get("division", {}).get("id")
        for r in block.get("teamRecords", []):
            rs, ra = r.get("runsScored", 0) or 0, r.get("runsAllowed", 0) or 0
            diff = r.get("runDifferential", rs - ra)
            rows.append({
                "team": r["team"]["id"],
                "league": league, "division": div,
                "w": r.get("wins", 0), "l": r.get("losses", 0),
                "pct": r.get("winningPercentage") or r.get("leagueRecord", {}).get("pct", ".000"),
                "gb": _num(r.get("divisionGamesBack", r.get("gamesBack"))),
                "e": _num(r.get("eliminationNumber")),
                "wcgb": _num(r.get("wildCardGamesBack")),
                "wc_e": _num(r.get("wildCardEliminationNumber")),
                "l10": _split(r, "splitRecords", "lastTen"),
                "strk": (r.get("streak") or {}).get("streakCode", "-"),
                "rs": rs, "ra": ra, "diff": f"{diff:+d}" if isinstance(diff, int) else str(diff),
                "xwl": _split(r, "expectedRecords", "xWinLoss"),
                "home": _split(r, "splitRecords", "home"),
                "away": _split(r, "splitRecords", "away"),
                "winners": _split(r, "splitRecords", "winners"),
                "clinch": CLINCH_MAP.get((r.get("clinchIndicator") or "").lower()),
                "div_rank": int(r.get("divisionRank", 99) or 99),
                "wc_rank": int(r["wildCardRank"]) if str(r.get("wildCardRank", "")).isdigit() else 99,
            })
    return rows


def _pct(row) -> float:
    try:
        return float(row["pct"])
    except (TypeError, ValueError):
        return 0.0


def division_tables(rows: list[dict], league: int) -> list[tuple[str, list[dict]]]:
    out = []
    for div_id, name in DIVISIONS[league]:
        teams = sorted((r for r in rows if r["division"] == div_id),
                       key=lambda r: (r["div_rank"], -_pct(r)))
        out.append((name, teams))
    return out


def wildcard_table(rows: list[dict], league: int) -> tuple[list[dict], list[dict]]:
    """(division leaders, wild-card race). The first WC_SPOTS of the race are in."""
    lg = [r for r in rows if r["league"] == league]
    leaders = sorted((r for r in lg if r["div_rank"] == 1), key=lambda r: -_pct(r))
    race = sorted((r for r in lg if r["div_rank"] != 1), key=lambda r: (r["wc_rank"], -_pct(r)))
    return leaders, race


def next_games(games, as_of: str, teams: dict) -> dict[int, str]:
    """team -> 'vs NYY · Apr 3' / '@ NYY · Apr 3': each team's next game after
    the standings date (today's unfinished games count when viewing today)."""
    from datetime import date as _date
    out: dict[int, str] = {}
    upcoming = sorted((g for g in games if g.state != "void" and
                       (g.day > as_of or (g.day == as_of and g.state != "final"))),
                      key=lambda g: (g.day, g.start_utc))
    for g in upcoming:
        d = _date.fromisoformat(g.day)
        when = f"{d:%b} {d.day}"
        for t, opp, sign in ((g.home, g.away, "vs"), (g.away, g.home, "@")):
            if t not in out:
                out[t] = f'{sign} {teams.get(opp, {}).get("abbr", "?")} · {when}'
    return out
