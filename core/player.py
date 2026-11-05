"""Player header: bio fields and the ENTRY line (how the player came into pro ball).

ENTRY examples
    2019 by BAL, Round 1, 1st Overall, Oregon State, Signing Bonus $8.1M
    2020 by NYM, Round 1, 19th Overall, Harvard-Westlake HS (CA), Signing Bonus $3.4M
    Signed by CLE out of Dominican Republic
(MLB's public data has draft bonuses but not international signing bonuses.)
"""
from __future__ import annotations

from datetime import date


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 13:
        return f"{n}th"
    return f'{n}{ {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th") }'


def money(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if v <= 0:
        return ""
    if v >= 1_000_000:
        return f"${v / 1_000_000:.1f}M".replace(".0M", "M")
    if v >= 1_000:
        return f"${v / 1_000:.0f}K"
    return f"${v:,.0f}"


def age_ym(dob: str, today: date | None = None) -> str:
    """Age as years.months, e.g. 23.04 = 23 years, 4 months."""
    try:
        b = date.fromisoformat(dob)
    except (TypeError, ValueError):
        return "-"
    t = today or date.today()
    months = (t.year - b.year) * 12 + (t.month - b.month) - (t.day < b.day)
    return f"{months // 12}.{months % 12:02d}"


def short_date(iso: str | None) -> str:
    try:
        d = date.fromisoformat(iso)
    except (TypeError, ValueError):
        return "-"
    return f"{d.month}/{d.day}/{d.year % 100:02d}"


def height_weight(p: dict) -> str:
    h = (p.get("height") or "").replace(" ", "").replace('"', "")
    w = p.get("weight")
    return f"{h}/{w}" if h and w else (h or "-")


def _is_hs(school: dict) -> bool:
    name, cls = school.get("name", ""), school.get("schoolClass", "")
    return "HS" in cls.upper() or " HS" in name or "High School" in name


def draft_entry(pick: dict, year: int, team_abbr: str) -> str:
    school = pick.get("school") or {}
    sname = school.get("name", "")
    if sname and _is_hs(school) and school.get("state"):
        if not sname.endswith(("HS", "High School")):
            sname = f"{sname} HS"
        sname = f'{sname} ({school["state"]})'
    parts = [f"{year} by {team_abbr}"]
    if pick.get("pickRound"):
        rnd = str(pick["pickRound"])
        parts.append(f"Round {rnd}" if rnd.isdigit() else rnd)
    if pick.get("pickNumber"):
        parts.append(f'{ordinal(int(pick["pickNumber"]))} Overall')
    if sname:
        parts.append(sname)
    bonus = money(pick.get("signingBonus"))
    if bonus:
        parts.append(f"Signing Bonus {bonus}")
    return ", ".join(parts)


def signing_entry(transactions: list[dict], country: str, abbr_for) -> str | None:
    """'Signed by CLE out of Dominican Republic' from the first free-agent signing."""
    signs = sorted((t for t in transactions if t.get("typeCode") == "SFA"),
                   key=lambda t: t.get("date", ""))
    if not signs:
        return None
    team = signs[0].get("toTeam", {})
    who = abbr_for(team.get("id")) or team.get("name", "")
    where = f" out of {country}" if country and country != "USA" else ""
    return f"Signed by {who}{where}"


def bio(p: dict, today: date | None = None) -> dict:
    """Header fields from an MLB 'people' record."""
    bats = p.get("batSide", {}).get("code", "")
    throws = p.get("pitchHand", {}).get("code", "")
    place = ", ".join(x for x in (p.get("birthCity"), p.get("birthStateProvince") or p.get("birthCountry")) if x)
    return {
        "first": p.get("useName") or p.get("firstName", ""),
        "last": p.get("useLastName") or p.get("lastName", ""),
        "pos": p.get("primaryPosition", {}).get("abbreviation", "-"),
        "hw": height_weight(p),
        "status": "ACTIVE" if p.get("active") else "INACTIVE",
        "bt": f"{bats}/{throws}" if bats or throws else "-",
        "age": age_ym(p.get("birthDate"), today),
        "dob": short_date(p.get("birthDate")),
        "debut": short_date(p.get("mlbDebutDate")),
        "born": place or "-",
        "country": p.get("birthCountry", ""),
        "draft_year": p.get("draftYear"),
        "team_id": p.get("currentTeam", {}).get("id"),
        "team_name": p.get("currentTeam", {}).get("name", ""),
        "is_pitcher": p.get("primaryPosition", {}).get("abbreviation") == "P",
    }


# ----------------------------------------------------------------------------
# Which Summary tabs a player gets
# ----------------------------------------------------------------------------
INFIELD = {"1B", "2B", "3B", "SS"}
OUTFIELD = {"LF", "CF", "RF", "OF"}
MIN_GAMES = 10            # games at a spot (last 3 seasons) to earn its tab


def position_games(fielding_splits: list[dict], since: int) -> dict[str, int]:
    """Games played at each position since `since` (MLB + MiLB fielding splits)."""
    out: dict[str, int] = {}
    for s in fielding_splits:
        try:
            if int(s.get("season", 0)) < since:
                continue
        except ValueError:
            continue
        pos = s.get("position", {}).get("abbreviation", "")
        g = s.get("stat", {}).get("gamesPlayed", s.get("stat", {}).get("games", 0)) or 0
        if pos:
            out[pos] = out.get(pos, 0) + int(g)
    return out


def summary_tabs(primary: str, games: dict[str, int], career_pa: int) -> list[str]:
    """Hitting / Core 4 / Defense / Baserunning / Pitching,
    depending on where the player actually plays."""
    def plays(group: set[str], primary_hit: bool) -> bool:
        return primary_hit or sum(games.get(p, 0) for p in group) >= MIN_GAMES

    pitcher = primary == "P" or games.get("P", 0) >= MIN_GAMES or primary == "TWP"
    hitter = primary != "P" or career_pa >= 50 or primary == "TWP"
    tabs: list[str] = []
    if pitcher and primary in ("P", "TWP"):
        tabs.append("Pitching")
    if hitter:
        tabs += ["Hitting", "Core 4"]
        if plays({"C"}, primary == "C") or plays(INFIELD, primary in INFIELD) or plays(OUTFIELD, primary in OUTFIELD):
            tabs.append("Defense")          # Catching / Infield / Outfield live under Defense
        tabs.append("Baserunning")
    if pitcher and "Pitching" not in tabs:
        tabs.append("Pitching")
    return tabs
