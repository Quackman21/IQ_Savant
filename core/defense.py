"""Defense: column layouts for Infield / Outfield / Catching and how each row is filled.

Sources
  * MLB fielding stats (any date range -> months work): chances, PO, A, E, DP, innings, SB/CS against
  * Savant leaderboards (full seasons only, via pybaseball): OAA, fielding run value, outfielder
    jump, catch probability, catcher framing, pop time
  * Blank (our models later): DRS family, OPA / FPA / CPA, every grade, infield Star tab,
    catcher blocking-by-location, Frame Speed Grade (-> SAM 3 glove model)
"""
from __future__ import annotations

import math

import pandas as pd

INFIELD = ["1B", "2B", "3B", "SS"]
OUTFIELD = ["LF", "CF", "RF"]
POS_NUM = {"C": 2, "1B": 3, "2B": 4, "3B": 5, "SS": 6, "LF": 7, "CF": 8, "RF": 9}
STAR_EDGES = {1: "0.9 and 1", 2: "0.75 and 0.9", 3: "0.5 and 0.75", 4: ".25 and 0.5", 5: "0 and .25"}

LEAD = ["Year", "Period", "Age", "Type", "ORG", "LVL >", "Pos >"]

# (key, label, hover definition, format)   format: int num1 num2 pct pctile time mph blank grade
# "|" starts a new column group (thin divider)
B = "blank"
_DRS = [("DRS", "DRS", "Defensive Runs Saved (our DRS model — coming)", B)]
_RATE = ["|", ("DRS/S", "DRS / Season", "DRS per full season (rate)", B),
         ("PCT_MLB", "Percentile (MLB)", "Percentile of DRS / Season vs MLB", B),
         ("PCT_LVL", "Percentile (LVL)", "Percentile of DRS / Season vs the level", B)]
_FIELDING = ["|", ("INN", "Inn", "Innings at the position(s)", "num1"),
             ("PO", "PO", "Putouts", "int"), ("A", "A", "Assists", "int"),
             ("E", "E", "Errors", "int"), ("DP", "DP", "Double plays", "int")]


def _g(label, desc):
    return (label.upper().replace(" ", "_"), label, desc, "grade")


IF_GRADES_TAB = [("N", "#", "Chances (putouts + assists + errors)", "int"), "|", *_DRS,
                 ("RS_FIELD", "RS Field", "Runs Saved on field plays (sum of pre-throw Rv Added)", B),
                 ("RS_1B", "RS 1B Rec", "Runs Saved from first base receiving (sum of DeltaOutCatch * 0.67)", B),
                 *_RATE, "|",
                 ("OPA", "OPA", "Total Out Probability Added (sum of model OPA per play)", B),
                 ("FPA", "FPA", "Total Fielded Probability Added (IsFielded - Prob(Fielded))", B), "|",
                 _g("First Step Grade", "Out Probability Added by fielder speed towards the ball in first second off bat"),
                 _g("Range Grade", "Out Probability Added by fielder between time of field (first contact) and off bat"),
                 _g("Conversion Grade", "On fielded balls, the difference between Out Prob at Throw (or step on base) "
                                        "and Out Prob at Field (1B: Out Prob at Catch)"),
                 _g("Ball Security Grade", "On fielded balls, the difference between Out Prob at Throw (or step on base) "
                                           "and Out Prob at Field"),
                 _g("Arm Strength Grade", "On thrown balls, the Out Probability Added from the fielder's throw speed"),
                 _g("Arm Accuracy Grade", "On thrown balls, the Out Probability Added from the throw accuracy"),
                 _g("1B Catch Grade", "Ability for first baseman to convert outs given the throw"), "|",
                 ("OAA", "OAA", "Statcast Outs Above Average (full seasons)", "int"),
                 ("FRV", "FRV", "Statcast Fielding Run Value (full seasons)", "int"),
                 *_FIELDING]


def _stars(kind: str, measure: str):
    out = []
    for k in range(1, 6):
        out += [(f"S{k}_N", f"{k} Star Plays", f"# of plays with {kind} Probability between {STAR_EDGES[k]}", "int"),
                (f"S{k}", f"{k} Star", f"{measure} on plays with {kind} Probability between {STAR_EDGES[k]}",
                 "pct" if kind == "Catch" else B)]
        if k < 5:
            out.append("|")
    return out


IF_STAR_TAB = [("N", "#", "Chances", "int"), "|", *_DRS,
               ("RS_FIELD", "RS Field", "Runs Saved on field plays", B),
               ("RS_1B", "RS 1B Rec", "Runs Saved from first base receiving", B), *_RATE, "|",
               *_stars("Out", "Out Probability Added")]

OF_GRADES_TAB = [("N", "#", "Chances (putouts + assists + errors)", "int"), "|", *_DRS,
                 ("RS_RANGE", "RS Range", "Outfield Defensive Runs Saved from range", B),
                 ("RS_CONV", "RS Conversion", "Outfield Defensive Runs Saved from converting catchable balls", B),
                 ("RS_ARM", "RS Arm", "Total Outfield Defensive Runs Saved from limiting baserunners", B),
                 *_RATE, "|",
                 ("CPA", "CPA", "Total Catch Probability Added (IsCaught - Prob(Caught))", B),
                 ("CPA_R", "CPA Range", "Catch Probability Added from range", B),
                 ("CPA_C", "CPA Conversion", "Catch Probability Added from conversion", B), "|",
                 _g("Range", "20-80 range grade"),
                 _g("Reaction", "Catch probability added from the fielder's first step"),
                 _g("Acceleration", "Catch probability added from the fielder's average speed over the first two "
                                    "seconds after contact. Combination of Reaction & Acceleration"),
                 _g("Jump", "Jump grade (reaction + acceleration + direction)"),
                 _g("Direction", "The angle between a fielder's starting location, his location after 2 seconds, and "
                                 "the landing location of the ball"),
                 _g("Max Spd", "Catch probability added from the fielder's maximum speed reached on the play "
                               "relative to the expected given hang time & distance to cover"), "|",
                 _g("Conversion", "Fielder's ability to complete the play; the difference between baseline prob "
                                  "caught and tool-adjusted prob caught"),
                 _g("XBH Prev", "XBH prevention grade from the outfield defensive range model"), "|",
                 _g("Arm Str", "Arm strength grade"),
                 _g("Reputation", "Arm reputation grade from the outfield arm run value model"),
                 _g("Transfer", "Transfer (exchange) grade"),
                 _g("Momentum", "Momentum into the throw grade"), "|",
                 ("OAA", "OAA", "Statcast Outs Above Average (full seasons)", "int"),
                 ("REACT_FT", "Reaction (ft)", "Statcast jump: feet covered vs average in the reaction phase", "num1"),
                 ("BURST_FT", "Burst (ft)", "Statcast jump: feet covered vs average in the burst phase", "num1"),
                 ("ROUTE_FT", "Route (ft)", "Statcast jump: feet vs average from route efficiency", "num1"),
                 ("FRV", "FRV", "Statcast Fielding Run Value (full seasons)", "int"),
                 *_FIELDING]

OF_STAR_TAB = [("N", "#", "Chances", "int"), "|", *_DRS,
               ("RS_ARM", "RS Arm", "Outfield Defensive Runs Saved from limiting baserunners", B), *_RATE, "|",
               *_stars("Catch", "Catch rate (Statcast catch probability, full seasons)")]

C_OVERVIEW = [("OPPS", "Opps Total", "# of framing, blocking, and throwing opportunities", B),
              ("F_OPPS", "Framing Opps", "# of framing opportunities (taken pitches he received)", "int"),
              ("B_OPPS", "Blocking Opps", "# of blocking opportunities", B),
              ("T_OPPS", "Throwing Opps", "# of throwing opportunities (stolen base attempts)", "int"), "|",
              ("DRS", "DRS", "Catching DRS", B),
              ("F_RS", "Framing Runs", "Framing runs saved (our framing model, any period): each extra strike x what "
               "a strike instead of a ball is worth in that count", "num1s"),
              ("B_RS", "Blocking RS", "Blocking runs (Statcast, full seasons)", "num1"),
              ("T_RS", "Throwing RS", "Throwing runs (Statcast, full seasons)", "num1"), "|",
              ("DRS/S", "DRS / Season", "Catching DRS Rate", B),
              ("F_RS/S", "Framing RS / Season", "Framing DRS Rate", B),
              ("B_RS/S", "Blocking RS / Season", "Blocking DRS Rate", B),
              ("T_RS/S", "Throwing RS / Season", "Throwing DRS Rate", B)]
C_FRAMING = [("DRS", "DRS", "Catching DRS", B), ("DRS/S", "DRS / Season", "Catching DRS Rate", B), "|",
             ("F_OPPS", "Framing Opps", "# of framing opportunities: every taken pitch (called strike or ball) he "
              "received", "int"),
             ("F_RS", "Framing Runs", "Framing runs saved: each extra strike x what a strike instead of a ball is worth "
              "in that count (about 0.07 runs at 0-0, 0.59 at 3-2)", "num1s"),
             ("F_RS/S", "Framing RS / Season", "Framing DRS Rate", B), "|",
             ("F_XS", "Extra Strikes", "Strikes above (+) / below (-) a league-average catcher on the same pitches.\n"
              "Each taken pitch: (1 if called a strike, 0 if a ball) - its strike chance. A 45% pitch called a "
              "strike = +0.55, called a ball = -0.45. Strike chance = league-average umpire, from location vs the "
              "hitter's zone, count, batter side and pitch type", "num1s"),
             ("F_XS/G", "Extra Strikes / Game", "Extra Strikes per game caught", "num2s"),
             ("F_SLAA", "SLAA", "Strikes Looking Above Average per framing opportunity\n[Extra Strikes / Framing Opps]",
              "pcts"),
             ("F_SLAA+", "SLAA+", "Called strikes vs expected, 100 = average\n[100 x called strikes / expected called "
              "strikes; 105 = 5% more strikes than an average catcher]", "int"), "|",
             ("F_AS", "Framing AS", "Strike rate on edge pitches, arm side — percentile vs MLB catchers", "pctile"),
             ("F_GS", "Framing GS", "Strike rate on edge pitches, glove side — percentile vs MLB catchers", "pctile"),
             ("F_TOZ", "Framing TOZ", "Strike rate on edge pitches, top of zone — percentile vs MLB catchers", "pctile"),
             ("F_BOZ", "Framing BOZ", "Strike rate on edge pitches, bottom of zone — percentile vs MLB catchers",
              "pctile"), "|",
             _g("Frame Speed Grade", "20-80 grade for glove speed towards middle of zone (our SAM 3 glove model)")]
C_BLOCKING = [("DRS", "DRS", "Catching DRS", B), ("DRS/S", "DRS / Season", "Catching DRS Rate", B), "|",
              ("B_OPPS", "Blocking Opps", "# of blocking opportunities", B),
              ("B_RS", "Blocking RS", "Blocking DRS (Statcast blocking runs, full seasons)", "num1"),
              ("B_RS/S", "Blocking RS / Season", "Blocking DRS Rate", B), "|",
              ("B_AIR", "Blocking Air", "Blocking on pitches not in the dirt — percentile", B),
              ("B_DAS", "Blocking Dirt AS", "Blocking on pitches in the dirt, arm side — percentile", B),
              ("B_DMID", "Blocking Dirt Mid", "Blocking on pitches in the dirt, middle — percentile", B),
              ("B_DGS", "Blocking Dirt GS", "Blocking on pitches in the dirt, glove side — percentile", B)]
C_THROWING = [("DRS", "DRS", "Catching DRS", B), ("DRS/S", "DRS / Season", "Catching DRS Rate", B), "|",
              ("T_OPPS", "Throwing Opps", "# of throwing opportunities (stolen base attempts)", "int"),
              ("T_RS", "Throwing RS", "Throwing DRS (Statcast throwing runs, full seasons)", "num1"),
              ("T_RS/S", "Throwing RS / Season", "Throwing DRS Rate", B), "|",
              ("POP", "Pop Time", "Time elapsed between catcher receiving pitch and throw landing at base "
                                  "(to 2B, Statcast, full seasons)", "time"),
              ("XFER", "Xfer", "Exchange time: glove to release (Statcast, full seasons)", "time"),
              ("ARM", "Arm Strength", "Max-effort arm strength (Statcast, full seasons)", "mph"),
              _g("Accuracy Grade", "Throw accuracy grade"), "|",
              ("SB", "SB", "Stolen bases allowed", "int"), ("CS", "CS", "Runners caught stealing", "int"),
              ("CS%", "CS%", "Caught stealing rate", "pct")]

TABS = {
    "Infield": {"Grades": IF_GRADES_TAB, "Star": IF_STAR_TAB},
    "Outfield": {"Grades": OF_GRADES_TAB, "Star": OF_STAR_TAB},
    "Catching": {"Overview": C_OVERVIEW, "Framing": C_FRAMING, "Blocking": C_BLOCKING, "Throwing": C_THROWING},
}
GROUP_POS = {"Infield": INFIELD, "Outfield": OUTFIELD, "Catching": ["C"]}


def defense_tabs(primary: str, games: dict[str, int], min_games: int = 10) -> list[str]:
    """Catching / Infield / Outfield for the positions this player actually plays."""
    out = []
    for name in ("Catching", "Infield", "Outfield"):
        pos = GROUP_POS[name]
        if primary in pos or sum(games.get(p, 0) for p in pos) >= min_games:
            out.append(name)
    return out


# ----------------------------------------------------------------------------
# Filling a row
# ----------------------------------------------------------------------------
def ip_outs(ip) -> int:
    try:
        whole, _, part = str(ip).partition(".")
        return int(whole) * 3 + int(part or 0)
    except ValueError:
        return 0


def mlb_counts(splits: list[dict], positions: list[str]) -> dict:
    """Sum MLB fielding splits for the chosen positions."""
    for s in splits:                                   # game logs keep the position inside "stat"
        if "position" not in s and isinstance(s.get("stat", {}).get("position"), dict):
            s["position"] = s["stat"]["position"]
    keep = [s for s in splits if s.get("position", {}).get("abbreviation") in positions]
    tot = {"N": 0, "PO": 0, "A": 0, "E": 0, "DP": 0, "outs": 0, "SB": 0, "CS": 0, "G": 0}
    played = []
    for s in keep:
        st = s.get("stat", {})
        g = st.get("gamesPlayed", st.get("games", 1 if st else 0)) or 0
        if g:
            played.append(s["position"]["abbreviation"])
        tot["N"] += st.get("chances", 0) or 0
        tot["PO"] += st.get("putOuts", 0) or 0
        tot["A"] += st.get("assists", 0) or 0
        tot["E"] += st.get("errors", 0) or 0
        tot["DP"] += st.get("doublePlays", 0) or 0
        tot["outs"] += ip_outs(st.get("innings", 0))
        tot["SB"] += st.get("stolenBases", 0) or 0
        tot["CS"] += st.get("caughtStealing", 0) or 0
        tot["G"] += g
    tot["INN"] = tot["outs"] / 3
    tot["T_OPPS"] = tot["SB"] + tot["CS"]
    tot["CS%"] = tot["CS"] / tot["T_OPPS"] if tot["T_OPPS"] else math.nan
    tot["Pos"] = ", ".join(p for p in positions if p in played)
    teams = sorted({s.get("team", {}).get("id") for s in keep if s.get("team", {}).get("id")})
    tot["team_ids"] = teams
    return tot


def col(row: pd.Series | None, *needles: str, exclude: tuple = ()) -> float:
    """First numeric value in a Savant row whose column contains every needle (case-insensitive)."""
    if row is None:
        return math.nan
    for c in row.index:
        lc = c.lower()
        if all(n in lc for n in needles) and not any(x in lc for x in exclude):
            v = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(v):
                return float(v)
    return math.nan


def pctile(board: pd.DataFrame, value: float, *needles: str, lower_better: bool = False) -> float:
    if board is None or board.empty or value is None or math.isnan(value):
        return math.nan
    cols = [c for c in board.columns if all(n in c.lower() for n in needles)]
    if not cols:
        return math.nan
    s = pd.to_numeric(board[cols[0]], errors="coerce").dropna()
    if s.empty:
        return math.nan
    p = 100 * (s < value).mean()
    return 100 - p if lower_better else p


def framing_zones(row: pd.Series | None, board: pd.DataFrame, throws: str = "R") -> dict:
    """Arm side / glove side / top / bottom strike-rate percentiles from Savant's framing
    leaderboard (shadow zones 11-19, catcher's view). RHT catcher: glove side = 14, arm side = 16."""
    if row is None or board is None or board.empty:
        return {}
    groups = {"F_TOZ": ["11", "12", "13"], "F_BOZ": ["17", "18", "19"],
              "F_GS": ["14"] if throws == "R" else ["16"], "F_AS": ["16"] if throws == "R" else ["14"]}
    out = {}
    for key, zones in groups.items():
        cols = [c for c in board.columns if "rate" in c.lower() and any(c.endswith(f"_{z}") for z in zones)]
        if not cols:
            continue
        b = board[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
        v = pd.to_numeric(row[cols], errors="coerce").mean()
        if pd.notna(v) and not b.empty:
            out[key] = 100 * (b < v).mean()
    return out



# ----------------------------------------------------------------------------
# Baserunning (same table style; RVA = our run value model, blank for now)
# ----------------------------------------------------------------------------
_RVA = [("RVA", "RVA Total", "Run Value Added on the bases (our model — coming)", B),
        ("RVA_ADV", "RVA Advancing", "Run Value Added advancing on hits / outs (our model — coming)", B),
        ("RVA_SB", "RVA Base Stealing", "Run Value Added stealing bases (our model — coming)", B)]
BR_ADVANCING = [*_RVA, "|",
                ("ADV_OPPS", "Advance Opps", "Opportunities to take an extra base", B),
                ("ATT%", "Attempt%", "Rate of extra-base attempts", B),
                ("ATT_EXP", "Attempt% Over Exp.", "Attempt rate above expected", B), "|",
                ("RVA_TAG", "RVA Tag", "Run value tagging up", B), ("RVA_HIT", "RVA Base Hit", "Run value on base hits", B),
                ("RVA_GB", "RVA GB Out", "Run value on ground-ball outs", B)]
BR_STEALING = [*_RVA, "|",
               ("G", "G", "Games", "int"), ("PA", "PA", "Plate appearances", "int"), "|",
               ("SB_ATT", "SB Attempts", "Stolen base attempts", "int"),
               ("SB%", "SB Success%", "Stolen base success rate", "pct"),
               ("SB", "SB", "Stolen bases", "int"), ("RVA_SB1", "RVA SB", "Run value from stolen bases", B),
               ("CS", "CS", "Caught stealing", "int"), ("RVA_CS", "RVA CS", "Run value lost to caught stealing", B)]
BR_TOOLS = [("SPRINT", "Sprint Speed", "Statcast sprint speed, ft/sec (full seasons)", "num1"),
            ("HP1B", "HP to 1B", "Home to first time, seconds (full seasons)", "num2"),
            ("BOLTS", "Bolts", "Runs of 30+ ft/sec (full seasons)", "int"), "|",
            ("LEAD2_1", "Secondary Lead 1st", "Secondary lead off first (Statcast 2023+ — coming)", B),
            ("LEAD2_2", "Secondary Lead 2nd", "Secondary lead off second (coming)", B),
            ("LEAD1_1", "Primary Lead 1st", "Primary lead off first (coming)", B),
            ("LEAD1_2", "Primary Lead 2nd", "Primary lead off second (coming)", B), "|",
            _g("HTF Grade", "Home-to-first grade"), _g("OOB Grade", "Out-of-the-box grade"),
            _g("Max Speed Grade", "Max speed grade"), _g("Acceleration Grade", "Acceleration grade")]
BR_TABS = {"Advancing": BR_ADVANCING, "Base Stealing": BR_STEALING, "Tools": BR_TOOLS}
