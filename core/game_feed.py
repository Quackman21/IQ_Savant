"""Turn MLB's live game feed (api/v1.1/game/{pk}/feed/live) into plain tables
for the Box Score, Lineup and Play by Play views."""
from __future__ import annotations


def _pid(x) -> str:
    return f"ID{x}"


def header(feed: dict) -> dict:
    gd, ld = feed.get("gameData", {}), feed.get("liveData", {})
    ls = ld.get("linescore", {})
    status = gd.get("status", {})
    return {
        "pk": gd.get("game", {}).get("pk"),
        "gtype": gd.get("game", {}).get("type"),
        "date": gd.get("datetime", {}).get("officialDate", ""),
        "start": gd.get("datetime", {}).get("dateTime", ""),
        "away": gd.get("teams", {}).get("away", {}).get("id"),
        "home": gd.get("teams", {}).get("home", {}).get("id"),
        "away_runs": ls.get("teams", {}).get("away", {}).get("runs"),
        "home_runs": ls.get("teams", {}).get("home", {}).get("runs"),
        "state": status.get("abstractGameState", ""),       # Preview / Live / Final
        "detail": status.get("detailedState", ""),
        "inning": ls.get("currentInning"),
        "inning_state": ls.get("inningState"),
        "venue": gd.get("venue", {}).get("name", ""),
    }


def linescore(feed: dict) -> dict:
    ls = feed.get("liveData", {}).get("linescore", {})
    final = feed.get("gameData", {}).get("status", {}).get("abstractGameState") == "Final"
    played = ls.get("innings", [])
    n = max(ls.get("scheduledInnings", 9), len(played))
    out = {}
    for side in ("away", "home"):
        runs = ["" for _ in range(n)]
        for inn in played:
            r = inn.get(side, {}).get("runs")
            if r is not None:
                runs[inn["num"] - 1] = r
            elif side == "home" and final and inn["num"] == len(played):
                runs[inn["num"] - 1] = "X"                   # home didn't need to bat
        t = ls.get("teams", {}).get(side, {})
        out[side] = {"innings": runs, "R": t.get("runs", ""), "H": t.get("hits", ""),
                     "E": t.get("errors", "")}
    out["n"] = n
    return out


def _players(feed: dict, side: str) -> dict:
    return feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {}).get("players", {})


def batting(feed: dict, side: str) -> list[dict]:
    rows = []
    for p in _players(feed, side).values():
        order = p.get("battingOrder")
        if not order:
            continue
        b = p.get("stats", {}).get("batting", {})
        rows.append({
            "order": int(order), "sub": int(order) % 100 != 0,
            "id": p["person"]["id"], "name": p["person"].get("fullName", ""),
            "pos": p.get("position", {}).get("abbreviation", ""),
            "AB": b.get("atBats", 0), "R": b.get("runs", 0), "H": b.get("hits", 0),
            "RBI": b.get("rbi", 0), "BB": b.get("baseOnBalls", 0), "SO": b.get("strikeOuts", 0),
            "HR": b.get("homeRuns", 0),
            "AVG": p.get("seasonStats", {}).get("batting", {}).get("avg", "-"),
        })
    return sorted(rows, key=lambda r: r["order"])


def pitching(feed: dict, side: str) -> list[dict]:
    team = feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {})
    players = team.get("players", {})
    rows = []
    for pid in team.get("pitchers", []):
        p = players.get(_pid(pid), {})
        s = p.get("stats", {}).get("pitching", {})
        rows.append({
            "id": pid, "name": p.get("person", {}).get("fullName", ""),
            "IP": s.get("inningsPitched", "0.0"), "H": s.get("hits", 0), "R": s.get("runs", 0),
            "ER": s.get("earnedRuns", 0), "BB": s.get("baseOnBalls", 0), "SO": s.get("strikeOuts", 0),
            "HR": s.get("homeRuns", 0), "P": s.get("numberOfPitches", s.get("pitchesThrown", 0)),
            "ERA": p.get("seasonStats", {}).get("pitching", {}).get("era", "-"),
        })
    return rows


def lineup(feed: dict, side: str) -> dict:
    """Starting nine (with bats) and starting / probable pitcher."""
    gd = feed.get("gameData", {})
    people = gd.get("players", {})
    team = feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {})
    players = team.get("players", {})
    nine = []
    for i, pid in enumerate(team.get("battingOrder", []), 1):
        p = players.get(_pid(pid), {})
        info = people.get(_pid(pid), {})
        nine.append({"spot": i, "id": pid, "name": p.get("person", {}).get("fullName", info.get("fullName", "")),
                     "pos": p.get("position", {}).get("abbreviation", ""),
                     "bats": info.get("batSide", {}).get("code", "")})
    starter = None
    if team.get("pitchers"):
        sp = team["pitchers"][0]
        starter = {"id": sp, "name": people.get(_pid(sp), {}).get("fullName", ""),
                   "throws": people.get(_pid(sp), {}).get("pitchHand", {}).get("code", "")}
    else:
        pp = gd.get("probablePitchers", {}).get(side)
        if pp:
            starter = {"id": pp["id"], "name": pp.get("fullName", ""),
                       "throws": people.get(_pid(pp["id"]), {}).get("pitchHand", {}).get("code", ""),
                       "probable": True}
    return {"nine": nine, "pitcher": starter}


def plays(feed: dict) -> list[dict]:
    """Half-innings in order, each with its plays (and every pitch of each play)."""
    halves: list[dict] = []
    for p in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        about, res = p.get("about", {}), p.get("result", {})
        key = (about.get("inning"), about.get("halfInning"))
        if not halves or (halves[-1]["inning"], halves[-1]["half"]) != key:
            halves.append({"inning": key[0], "half": key[1], "plays": []})
        pitches = []
        for e in p.get("playEvents", []):
            if not e.get("isPitch"):
                continue
            d = e.get("details", {})
            pitches.append({
                "n": e.get("pitchNumber"),
                "call": d.get("description", ""),
                "type": d.get("type", {}).get("description", ""),
                "mph": e.get("pitchData", {}).get("startSpeed"),
                "count": f'{e.get("count", {}).get("balls", 0)}-{e.get("count", {}).get("strikes", 0)}',
            })
        halves[-1]["plays"].append({
            "event": res.get("event", ""),
            "desc": res.get("description", ""),
            "batter": p.get("matchup", {}).get("batter", {}).get("fullName", ""),
            "pitcher": p.get("matchup", {}).get("pitcher", {}).get("fullName", ""),
            "scoring": bool(about.get("isScoringPlay")),
            "score": (res.get("awayScore"), res.get("homeScore")),
            "outs": p.get("count", {}).get("outs"),
            "pitches": pitches,
        })
    return halves
