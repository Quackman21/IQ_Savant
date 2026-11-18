"""ABS challenges by player (2026+): who challenged (hitter / catcher / pitcher), how it went, where the pitch was,
and what it was worth.

Per challenge
  Role          Hitting = the batter challenged, Catching = the catcher (Statcast fielder_2), Pitching = the pitcher.
  Result        Successful = overturned, Failed = the call stood.
  Margin        inches the ball was outside the ABS zone (+) or inside it (-). The zone counts a strike when any part
                of the ball touches it, so the edge is the plate's edge (8.5") / the hitter's zone top + bottom, plus
                the ball's radius (1.45"). Corners: straight-line distance.
  Not counted   the LAST PITCH OF THE GAME, the call stood, and the call was right by more than 3 inches (a ball more
                than 3" outside the zone, or a strike more than 3" inside it). With nothing left to save the challenge
                for, players burn it on pitches that weren't close -- those don't count against them. An overturn
                always counts.
  ABS Runs      what the overturn was worth to the challenger's side: a ball turned into a strike (or the reverse)
                in that count = the run value of a ball minus a strike in that count, from that season's pitches
                (Statcast delta_run_exp: average after a called ball minus after a called strike). Failed = 0.

  Challenge cost  a FAILED challenge costs what a challenge was worth in that game: the average value of the missed
                calls that went AGAINST his team that game (a ball that touched the ABS zone called a ball on his
                team's pitcher, or a called strike outside it on his team's hitters -- the umpire's original call),
                blended toward the season's average missed call (SHRINK_K missed calls' worth) so a clean game still
                costs something and one call doesn't set it. Free on the last pitch of the game. Overturns cost 0.
  Net ABS Runs  ABS Runs - Challenge cost. Graded on this.

Per player x role x season (saved small, data/league/abs_players_<season>.csv.gz): games in that role, challenges,
overturns, not counted, runs. MLB averages / spreads by role -> the colour grades (as in Core 4).

    python -m core.abs_stats 2026        # reads data/raw/statcast_2026.csv.gz + the ABS challenge file
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LEAGUE = ROOT / "data" / "league"
ROLES = {"Hitting": "batter", "Catching": "fielder_2", "Pitching": "pitcher"}
FINAL_MARGIN_IN = 3.0
BALL_R_IN = 1.45
MIN_GRADE = 5                     # counted challenges before a player's rates go into the MLB spread
MIN_GAMES = 20                    # games in the role before his runs / game go into the spread
SHRINK_K = 3.0                    # missed calls' worth of the season average blended into each game's average
# fallback count values (runs, ball minus called strike) -- replaced by the season's own when it's built
DEFAULT_VALUES = {"0-0": .087, "1-0": .108, "2-0": .145, "3-0": .175, "0-1": .072, "1-1": .092, "2-1": .132,
                  "3-1": .245, "0-2": .118, "1-2": .162, "2-2": .261, "3-2": .555}
COLS = ["game_date", "game_pk", "at_bat_number", "pitch_number", "challenger", "role", "overturned", "result",
        "counted", "final_pitch", "margin_in", "from_call", "to_call", "value", "runs", "balls", "strikes", "inning",
        "inning_topbot", "outs_when_up", "batter", "pitcher", "fielder_2", "stand", "p_throws", "pitch_type",
        "release_speed", "plate_x", "plate_z", "sz_top", "sz_bot", "home_team", "away_team", "play_id", "team", "team_misses",
        "game_miss_value", "cost", "net"]


def _n(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)


def margin_in(df: pd.DataFrame) -> pd.Series:
    """Inches the ball's centre is outside the ABS zone's touching edge (+), or inside its nearest edge (-)."""
    px, pz = _n(df, "plate_x") * 12, _n(df, "plate_z") * 12
    top, bot = _n(df, "sz_top") * 12, _n(df, "sz_bot") * 12
    x = px.abs() - (8.5 + BALL_R_IN)
    up, down = pz - (top + BALL_R_IN), (bot - BALL_R_IN) - pz
    dz = np.maximum(up, down)
    outside = np.hypot(np.clip(x, 0, None), np.clip(dz, 0, None))
    inside = np.maximum(x, dz)
    return pd.Series(np.where(outside > 0, outside, inside), index=df.index).where(px.notna() & pz.notna())


def count_values(raw: pd.DataFrame) -> dict:
    """'b-s' -> runs a ball is worth over a called strike in that count (batting side), from the season itself."""
    if raw is None or raw.empty or "delta_run_exp" not in raw:
        return dict(DEFAULT_VALUES)
    d = raw[raw["description"].astype(str).isin(["ball", "blocked_ball", "called_strike"])]
    cnt = _n(d, "balls").astype("Int64").astype(str) + "-" + _n(d, "strikes").astype("Int64").astype(str)
    rv = _n(d, "delta_run_exp")
    strike = d["description"].astype(str).eq("called_strike")
    out = {}
    for c in DEFAULT_VALUES:
        m = cnt.eq(c)
        b, s = rv[m & ~strike].mean(), rv[m & strike].mean()
        out[c] = round(abs(float(b - s)), 4) if (m & ~strike).sum() > 50 and (m & strike).sum() > 50 else DEFAULT_VALUES[c]
    return out


def value_of(balls, strikes, values: dict | None = None) -> pd.Series:
    v = values or DEFAULT_VALUES
    key = pd.Series(balls).astype("Int64").astype(str) + "-" + pd.Series(strikes).astype("Int64").astype(str)
    return key.map(v).astype(float)


def _final_pitch(raw: pd.DataFrame) -> pd.Series:
    """True on each game's last pitch."""
    ab, pn = _n(raw, "at_bat_number"), _n(raw, "pitch_number")
    order = ab * 1000 + pn
    return order.eq(order.groupby(raw["game_pk"]).transform("max"))


def missed_calls(raw: pd.DataFrame, values: dict | None = None) -> pd.DataFrame:
    """Every called pitch the umpire got wrong by the ABS zone (his ORIGINAL call), who it went against, its value.
    raw should already carry the original calls (abs_calls.original_calls) for 2026+."""
    d = raw[raw["description"].astype(str).isin(["called_strike", "ball", "blocked_ball"])]
    if d.empty:
        return pd.DataFrame(columns=["game_pk", "against", "value"])
    m = margin_in(d)
    strike = d["description"].astype(str).eq("called_strike")
    miss = (strike & (m > 0)) | (~strike & (m <= 0))
    d, strike = d[miss.fillna(False)], strike[miss.fillna(False)]
    col = lambda c: d[c].astype(str) if c in d else pd.Series("", index=d.index)
    top = col("inning_topbot").eq("Top")
    bat = np.where(top, col("away_team"), col("home_team"))
    fld = np.where(top, col("home_team"), col("away_team"))
    return pd.DataFrame({"game_pk": pd.to_numeric(d["game_pk"], errors="coerce"),
                         "against": np.where(strike, bat, fld),              # a bad strike hurts the hitters
                         "value": value_of(_n(d, "balls"), _n(d, "strikes"), values).values}, index=d.index)


def game_costs(misses: pd.DataFrame, k: float = SHRINK_K) -> tuple[dict, float]:
    """((game, team) -> what a challenge was worth to that team that game, season average missed call)."""
    if misses is None or misses.empty:
        avg = float(np.mean(list(DEFAULT_VALUES.values())))
        return {}, avg
    avg = float(misses["value"].mean())
    g = misses.groupby(["game_pk", "against"])["value"].agg(["sum", "count"])
    cost = ((g["sum"] + k * avg) / (g["count"] + k)).to_dict()
    return {(int(a), str(b)): float(v) for (a, b), v in cost.items()}, avg


def table(raw: pd.DataFrame, ch: pd.DataFrame, values: dict | None = None,
          costs: dict | None = None, avg_miss: float | None = None, misses: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per ABS challenge, with everything the profile shows (role, result, counted, margin, runs)."""
    from core import abs_calls
    if raw is None or raw.empty or ch is None or ch.empty:
        return pd.DataFrame(columns=COLS)
    key = ["game_pk", "at_bat_number", "pitch_number"]
    games = set(pd.to_numeric(ch["game_pk"], errors="coerce").dropna().astype(int))
    r = raw[pd.to_numeric(raw["game_pk"], errors="coerce").isin(games)].copy()
    if r.empty:
        return pd.DataFrame(columns=COLS)
    for k in key:
        r[k] = pd.to_numeric(r[k], errors="coerce")
    r = r.dropna(subset=key)
    r[key] = r[key].astype(int)
    r["final_pitch"] = _final_pitch(r)
    stood = abs_calls.call_that_stood(r)
    desc = r["description"].astype(str)
    r["_stood"] = stood.fillna(desc.map({"called_strike": "S", "ball": "B", "blocked_ball": "B"}))
    c = ch.copy()
    for k in key:
        c[k] = pd.to_numeric(c[k], errors="coerce")
    c = c.dropna(subset=key)
    c[key] = c[key].astype(int)
    if "play_id" not in c:
        c["play_id"] = None
    m = c[key + ["overturned", "challenger", "play_id"]].merge(r, on=key, how="inner")
    if m.empty:
        return pd.DataFrame(columns=COLS)
    who = pd.to_numeric(m["challenger"], errors="coerce")
    m["role"] = np.select([who.eq(_n(m, "batter")), who.eq(_n(m, "fielder_2")), who.eq(_n(m, "pitcher"))],
                          ["Hitting", "Catching", "Pitching"], "Other")
    over = m["overturned"].astype(str).str.lower().isin(["true", "1", "1.0"])
    m["overturned"] = over
    m["result"] = np.where(over, "Successful", "Failed")
    m["to_call"] = m["_stood"].map({"S": "Strike", "B": "Ball"}).fillna("")
    flip = {"Strike": "Ball", "Ball": "Strike"}
    m["from_call"] = np.where(over, m["to_call"].map(flip).fillna(""), m["to_call"])
    m["margin_in"] = margin_in(m).round(1)
    right_by = np.where(m["to_call"].eq("Ball"), m["margin_in"], -m["margin_in"])     # how clearly the call was right
    m["counted"] = over | ~(m["final_pitch"].astype(bool) & (pd.Series(right_by, index=m.index) > FINAL_MARGIN_IN))
    m["value"] = value_of(_n(m, "balls"), _n(m, "strikes"), values).values
    m["runs"] = np.where(over, m["value"], 0.0)
    m["challenger"] = who
    # what a failed challenge cost: that game's average missed call against his team (blended to the season)
    if costs is None:
        costs, avg_miss = game_costs(misses if misses is not None else missed_calls(r, values))
    avg_miss = avg_miss if avg_miss is not None else float(np.mean(list(DEFAULT_VALUES.values())))
    top = m["inning_topbot"].astype(str).eq("Top")
    team = np.where(m["role"].eq("Hitting"), np.where(top, m["away_team"], m["home_team"]),
                    np.where(top, m["home_team"], m["away_team"]))
    m["team"] = team
    mc = misses if misses is not None else None
    m["game_miss_value"] = [costs.get((int(g), str(t)), avg_miss) for g, t in zip(m["game_pk"], team)]
    if mc is not None and not mc.empty:
        cnt = mc.groupby(["game_pk", "against"]).size().to_dict()
        m["team_misses"] = [cnt.get((g, t), 0) for g, t in zip(m["game_pk"], team)]
    m["cost"] = np.where(~over & ~m["final_pitch"].astype(bool), m["game_miss_value"], 0.0)
    m["cost"] = np.where(m["counted"], m["cost"], 0.0)
    m["net"] = m["runs"] - m["cost"]
    m["game_date"] = m["game_date"].astype(str).str[:10]
    for col in COLS:
        if col not in m:
            m[col] = np.nan
    return m[COLS].sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def players(raw: pd.DataFrame, tab: pd.DataFrame) -> pd.DataFrame:
    """Every player x role with games in that role, plus his challenge totals."""
    rows = []
    for role, col in ROLES.items():
        if col not in raw:
            continue
        g = raw[["game_pk", col]].dropna().drop_duplicates().groupby(col).size().rename("games")
        t = tab[tab["role"].eq(role)] if not tab.empty else tab
        agg = pd.DataFrame({"games": g})
        if not t.empty:
            cc = t[t["counted"]]
            agg["challenges"] = cc.groupby("challenger").size()
            agg["overturned"] = cc.groupby("challenger")["overturned"].sum()
            agg["not_counted"] = t[~t["counted"]].groupby("challenger").size()
            agg["runs"] = cc.groupby("challenger")["runs"].sum()
            agg["cost"] = cc.groupby("challenger")["cost"].sum()
        for c in ("challenges", "overturned", "not_counted", "runs", "cost"):
            agg[c] = pd.to_numeric(agg.get(c), errors="coerce").fillna(0.0) if c in agg else 0.0
        agg = agg.reset_index().rename(columns={col: "player", "index": "player"})
        agg["role"] = role
        rows.append(agg)
    if not rows:
        return pd.DataFrame(columns=["player", "role", "games", "challenges", "overturned", "not_counted", "runs", "cost"])
    out = pd.concat(rows, ignore_index=True)
    out["player"] = pd.to_numeric(out["player"], errors="coerce").astype("Int64")
    return out[["player", "role", "games", "challenges", "overturned", "not_counted", "runs", "cost"]]


# ----------------------------------------------------------------------------- the numbers the profile shows
STATS = ["G", "Challenges", "Chal/G", "Overturned", "Failed", "Overturn %", "ABS Runs", "Challenge cost",
         "Net ABS Runs", "Net/G", "Not counted"]
GRADED = {"Overturn %": 1, "Net ABS Runs": 1, "Net/G": 1}   # higher = better (coloured like Core 4)


def summarize(games: float, rows: pd.DataFrame) -> dict:
    """rows = his challenges in one role (all of them; 'counted' decides what counts)."""
    c = rows[rows["counted"].astype(bool)] if not rows.empty else rows
    n = len(c)
    o = int(c["overturned"].astype(bool).sum()) if n else 0
    runs = float(c["runs"].sum()) if n else 0.0
    cost = float(pd.to_numeric(c["cost"], errors="coerce").fillna(0).sum()) if n and "cost" in c else 0.0
    net = runs - cost
    return {"G": games, "Challenges": n, "Chal/G": n / games if games else math.nan, "Overturned": o,
            "Failed": n - o, "Overturn %": o / n if n else math.nan, "ABS Runs": runs, "Challenge cost": -cost,
            "Net ABS Runs": net, "Net/G": net / games if games else math.nan,
            "Not counted": int(len(rows) - n)}


def league(pl: pd.DataFrame) -> dict:
    """role -> stat -> (MLB average, spread across players) -- the colour grades."""
    out = {}
    if pl is None or pl.empty:
        return out
    for role, g in pl.groupby("role"):
        g = g.assign(cost=pd.to_numeric(g["cost"], errors="coerce").fillna(0) if "cost" in g else 0.0)
        g = g.assign(net=g["runs"] - g["cost"])
        G, n, o, runs, cost = g["games"].sum(), g["challenges"].sum(), g["overturned"].sum(), g["runs"].sum(), g["cost"].sum()
        q = g[g["challenges"] >= MIN_GRADE]
        qg = g[g["games"] >= MIN_GAMES]
        per = len(g) or 1
        sd = lambda s_, ok: float(s_.std()) if ok else math.nan
        out[role] = {
            "G": (G / per, math.nan), "Challenges": (n / per, math.nan),
            "Chal/G": (n / G if G else math.nan, math.nan), "Overturned": (o / per, math.nan),
            "Failed": ((n - o) / per, math.nan), "Not counted": (g["not_counted"].sum() / per, math.nan),
            "Overturn %": (o / n if n else math.nan, sd(q["overturned"] / q["challenges"], len(q) > 2)),
            "ABS Runs": (runs / per, math.nan), "Challenge cost": (-cost / per, math.nan),
            "Net ABS Runs": ((runs - cost) / per, sd(qg["net"], len(qg) > 2)),
            "Net/G": ((runs - cost) / G if G else math.nan, sd(qg["net"] / qg["games"], len(qg) > 2)),
        }
    return out


# ----------------------------------------------------------------------------- files
def paths(season: int) -> dict:
    return {"tab": LEAGUE / f"abs_{season}.csv.gz", "players": LEAGUE / f"abs_players_{season}.csv.gz",
            "meta": LEAGUE / f"abs_meta_{season}.json"}


def seasons() -> list[int]:
    return sorted(int(p.stem.split("_")[1].split(".")[0]) for p in LEAGUE.glob("abs_[0-9]*.csv.gz"))


def load(season: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    p = paths(season)
    tab = pd.read_csv(p["tab"], dtype={"game_date": str, "play_id": str}) if p["tab"].exists() else pd.DataFrame(columns=COLS)
    pl = pd.read_csv(p["players"]) if p["players"].exists() else pd.DataFrame()
    meta = json.loads(p["meta"].read_text()) if p["meta"].exists() else {}
    if not tab.empty:
        tab["counted"] = tab["counted"].astype(str).str.lower().isin(["true", "1", "1.0"])
        tab["overturned"] = tab["overturned"].astype(str).str.lower().isin(["true", "1", "1.0"])
    return tab, pl, meta


def save(raw: pd.DataFrame, season: int, ch: pd.DataFrame | None = None) -> dict:
    """Build + save the season's challenge table, player totals and count values."""
    from core import abs_calls
    if ch is None:
        games = pd.to_numeric(raw["game_pk"], errors="coerce").dropna().astype(int).unique().tolist()
        ch = abs_calls.season_challenges(season, games)
    values = count_values(raw)
    try:
        orig = abs_calls.original_calls(raw, challenges=ch)          # the umpire's calls, before any overturn
    except Exception:
        orig = raw
    misses = missed_calls(orig, values)
    costs, avg_miss = game_costs(misses)
    tab = table(raw, ch, values, costs, avg_miss, misses)
    pl = players(raw, tab)
    p = paths(season)
    LEAGUE.mkdir(parents=True, exist_ok=True)
    tab.to_csv(p["tab"], index=False)
    pl.to_csv(p["players"], index=False)
    p["meta"].write_text(json.dumps({"values": values, "final_margin_in": FINAL_MARGIN_IN, "avg_miss": avg_miss,
                                     "shrink_k": SHRINK_K, "missed_calls": int(len(misses)),
                                     "last_game": str(tab["game_date"].max()) if len(tab) else None}))
    by = tab["role"].value_counts().to_dict() if len(tab) else {}
    print(f"[{season}] ABS: {len(tab):,} challenges ({by}); "
          f"{int((~tab['counted']).sum()) if len(tab) else 0} not counted (last pitch, call right by 3\"+)",
          flush=True)
    return {"challenges": len(tab), "by_role": by}


def fill_play_ids(rows: pd.DataFrame, get_pbp) -> pd.DataFrame:
    """Savant video ids for challenges saved before the challenge file kept them: from each game's play-by-play."""
    if rows.empty:
        return rows
    rows = rows.copy()
    if "play_id" not in rows:
        rows["play_id"] = None
    miss = rows["play_id"].isna() | rows["play_id"].astype(str).isin(["", "nan", "None"])
    if not miss.any():
        return rows
    from concurrent.futures import ThreadPoolExecutor
    games = sorted(set(pd.to_numeric(rows.loc[miss, "game_pk"], errors="coerce").dropna().astype(int)))

    def ids(pk):
        try:
            pbp = get_pbp(pk)
        except Exception:
            return pk, {}
        out = {}
        for p in pbp.get("allPlays", []):
            ab = int(p.get("about", {}).get("atBatIndex", -1)) + 1
            for e in p.get("playEvents", []):
                if e.get("isPitch") and e.get("playId") and e.get("pitchNumber") is not None:
                    out[(ab, int(e["pitchNumber"]))] = e["playId"]
        return pk, out
    with ThreadPoolExecutor(6) as ex:
        found = dict(ex.map(ids, games))
    rows.loc[miss, "play_id"] = [found.get(int(g), {}).get((int(a), int(n)), None)
                                 for g, a, n in zip(rows.loc[miss, "game_pk"], rows.loc[miss, "at_bat_number"],
                                                    rows.loc[miss, "pitch_number"])]
    return rows


def main(argv: list[str]) -> None:
    from pipeline import build_league_reference as bl
    for a in argv or ["2026"]:
        season = int(a)
        f = bl.RAW_DIR / f"statcast_{season}.csv.gz"
        if not f.exists():
            print(f"[{season}] no raw Statcast file ({f.name}) -- run the league build first.", flush=True)
            continue
        raw = pd.read_csv(f, low_memory=False)
        save(raw, season)


if __name__ == "__main__":
    main(sys.argv[1:])
