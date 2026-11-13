"""Defense: column layouts for Infield / Outfield / Catching and how each row is filled.

Sources
  * MLB fielding stats (any date range -> months work): chances, PO, A, E, DP, innings, SB/CS against
  * Savant leaderboards (full seasons only, via pybaseball): OAA, fielding run value, outfielder
    jump, catch probability, catcher framing, pop time
  * IQ fielding model (models/fielding.py, any period): DRS, RS Field / Range / Arm, DP runs, OPA / FPA / CPA,
    stars, DRS / Season + percentiles, catcher DRS (framing + Statcast blocking / throwing), blocking opps
  * Tool grades (v20.0): infield Range / Conversion / Ball Security (our model, any period) + Arm Strength
    (Statcast, seasons); outfield Range (ours) + Reaction / Acceleration / Jump / Direction / Arm (Statcast)
  * Blank on purpose (need tracking that isn't public; no stand-ins): First Step, Arm Accuracy, 1B Catch,
    RS 1B receiving, OF conversion / max speed / XBH prev / reputation / transfer / momentum, catcher accuracy,
    Frame Speed Grade (-> SAM 3 glove model)
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
_DRS = [("DRS", "DRS", "Defensive Runs Saved (IQ fielding model, any period)\nInfield: RS Field + double-play runs. "
                       "Outfield: RS Range + RS Arm. Runs vs an average fielder at the position", "num1s")]
_RATE = ["|", ("DRS/S", "DRS / Season", "DRS per 1,200 innings (a full season)", "num1s"),
         ("PCT_MLB", "Percentile (MLB)", "Percentile of DRS / Season vs MLB players at the same position(s) "
                                         "that season (300+ innings)", "pctile"),
         ("PCT_LVL", "Percentile (LVL)", "Percentile of DRS / Season vs the level (MLB for now)", "pctile")]
_FIELDING = ["|", ("INN", "Inn", "Innings at the position(s)", "num1"),
             ("PO", "PO", "Putouts", "int"), ("A", "A", "Assists", "int"),
             ("E", "E", "Errors", "int"), ("DP", "DP", "Double plays", "int")]


def _g(label, desc):
    return (label.upper().replace(" ", "_"), label, desc, "grade")


def _nb(label, why):
    """A tool grade we leave BLANK on purpose: the data isn't public and we don't use stand-ins."""
    return (label.upper().replace(" ", "_"), label, f"Blank on purpose: {why}. No stand-ins.", B)


IF_GRADES_TAB = [("N", "#", "Chances (putouts + assists + errors)", "int"), "|", *_DRS,
                 ("RS_FIELD", "RS Field", "Runs saved turning batted balls into outs: OPA x run value of an out vs "
                  "a hit on that kind of ball", "num1s"),
                 ("RS_1B", "RS 1B Rec", "Runs saved from first base receiving (needs throw tracking -- not public)", B),
                 *_RATE, "|",
                 ("OPA", "OPA", "Out Probability Added: out made = 1 - out chance; not made = -(out chance x his "
                  "share of the play). Out chance from exit velo, launch angle, direction, distance", "num1s"),
                 ("FPA", "FPA", "Fielded Probability Added: fielded by him (1 / 0) - chance he'd be the one to field "
                  "it", "num1s"), "|",
                 _nb("First Step Grade", "how fast he reacts off the bat (needs infielder tracking, which MLB doesn't "
                                         "publish)"),
                 _g("Range Grade", "20-80. Getting to the ball: the part of his OPA from reaching balls (everything "
                                   "except finishing plays he already fielded), per responsible play, regressed for "
                                   "sample size. Hover for his rate."),
                 _g("Conversion Grade", "20-80. Finishing the play once he has the ball: on balls he fielded, outs vs "
                                        "how often an average fielder at his position finishes that same ball (exit "
                                        "velo, angle, direction, distance). Includes throws and bobbles; a first "
                                        "baseman's dropped good throw isn't charged to him. Regressed for sample size."),
                 _g("Ball Security Grade", "20-80. Bobbles and boots: fielding errors vs expected on the balls he "
                                           "fielded (throwing errors excluded). Part of Conversion, shown on its own. "
                                           "Regressed -- errors are rare, so it takes many chances to trust."),
                 _g("Arm Strength Grade", "20-80. Statcast average throw speed at his position (full seasons, 2020+) "
                                          "vs other qualified throwers there"),
                 _nb("Arm Accuracy Grade", "throw accuracy (needs throw tracking; error counts alone are too rare to be "
                                           "more than a stand-in)"),
                 _nb("1B Catch Grade", "first-base receiving (scoops aren't tracked publicly)"), "|",
                 ("OAA", "OAA", "Statcast Outs Above Average (full seasons)", "int"),
                 ("FRV", "FRV", "Statcast Fielding Run Value (full seasons)", "int"),
                 *_FIELDING]


def _stars(kind: str, measure: str):
    out = []
    for k in range(1, 6):
        out += [(f"S{k}_N", f"{k} Star Plays", f"# of plays with {kind} Probability between {STAR_EDGES[k]}", "int"),
                (f"S{k}", f"{k} Star", f"{measure} on plays with {kind} Probability between {STAR_EDGES[k]}",
                 "pct" if kind == "Catch" else "num1s")]
        if k < 5:
            out.append("|")
    return out


IF_STAR_TAB = [("N", "#", "Chances", "int"), "|", *_DRS,
               ("RS_FIELD", "RS Field", "Runs saved turning batted balls into outs", "num1s"),
               ("RS_1B", "RS 1B Rec", "Runs saved from first base receiving (needs throw tracking -- not public)", B),
               *_RATE, "|",
               *_stars("Out", "Out Probability Added")]

OF_GRADES_TAB = [("N", "#", "Chances (putouts + assists + errors)", "int"), "|", *_DRS,
                 ("RS_RANGE", "RS Range", "Runs saved catching balls: CPA x run value of a catch vs a hit. "
                  "Includes conversion (public data can't tell whether he reached the ball)", "num1s"),
                 ("RS_CONV", "RS Conversion", "Runs from converting balls he reached (needs fielder tracking -- "
                  "not public; it's inside RS Range)", B),
                 ("RS_ARM", "RS Arm", "Runs saved limiting baserunners on balls he fielded with runners on: expected "
                  "run value of the play - actual (holds, extra bases, assists)", "num1s"),
                 *_RATE, "|",
                 ("CPA", "CPA", "Catch Probability Added: caught = 1 - catch chance; not caught = -(catch chance x "
                  "his share)", "num1s"),
                 ("CPA_R", "CPA Range", "Catch Probability Added from range (needs fielder tracking -- not public)", B),
                 ("CPA_C", "CPA Conversion", "Catch Probability Added from conversion (needs fielder tracking)", B), "|",
                 _g("Range", "20-80. Catch Probability Added per responsible play (our model, any period), "
                             "regressed for sample size. Includes conversion -- public data can't split them."),
                 _g("Reaction", "20-80. Statcast jump: feet covered vs average in the first 1.5 s (full seasons), "
                                "regressed by his opportunities"),
                 _g("Acceleration", "20-80. Statcast jump, burst: feet vs average from 1.5 to 3 s (full seasons)"),
                 _g("Jump", "20-80. Statcast total jump: feet vs average in the first 3 s = reaction + burst + route"),
                 _g("Direction", "20-80. Statcast jump, route: feet vs average from running a direct path"),
                 _nb("Max Spd", "top speed on the play vs needed (needs fielder tracking)"), "|",
                 _nb("Conversion", "finishing balls he reached (public data can't tell if he reached it)"),
                 _nb("XBH Prev", "extra-base-hit prevention (needs fielder tracking)"), "|",
                 _g("Arm Str", "20-80. Statcast average throw speed at his position (full seasons, 2020+)"),
                 _nb("Reputation", "runners not testing his arm (no public model yet)"),
                 _nb("Transfer", "exchange time (outfield transfers aren't public)"),
                 _nb("Momentum", "momentum into the throw (needs fielder tracking)"), "|",
                 ("OAA", "OAA", "Statcast Outs Above Average (full seasons)", "int"),
                 ("REACT_FT", "Reaction (ft)", "Statcast jump: feet covered vs average in the reaction phase", "num1"),
                 ("BURST_FT", "Burst (ft)", "Statcast jump: feet covered vs average in the burst phase", "num1"),
                 ("ROUTE_FT", "Route (ft)", "Statcast jump: feet vs average from route efficiency", "num1"),
                 ("FRV", "FRV", "Statcast Fielding Run Value (full seasons)", "int"),
                 *_FIELDING]

OF_STAR_TAB = [("N", "#", "Chances", "int"), "|", *_DRS,
               ("RS_ARM", "RS Arm", "Runs saved limiting baserunners", "num1s"), *_RATE, "|",
               *_stars("Catch", "Catch rate (IQ catch chance, any period)")]

C_OVERVIEW = [("OPPS", "Opps Total", "# of framing, blocking, and throwing opportunities", "int"),
              ("F_OPPS", "Framing Opps", "# of framing opportunities (taken pitches he received)", "int"),
              ("B_OPPS", "Blocking Opps", "Genuine blocking opportunities: pitches with a runner on and a 5%+ wild "
               "pitch / passed ball chance", "int"),
              ("T_OPPS", "Throwing Opps", "# of throwing opportunities (stolen base attempts)", "int"), "|",
              ("DRS", "DRS", "Catching DRS: Framing + Blocking + Throwing runs (any period)", "num1s"),
              ("F_RS", "Framing Runs", "Framing runs saved (our framing model, any period): each extra strike x what "
               "a strike instead of a ball is worth in that count", "num1s"),
              ("B_RS", "Blocking RS", "Blocking runs, situational (IQ, any period): every pitch with a runner on, "
               "(WP/PB chance - what happened) x what a WP / PB costs in that base / out situation", "num1s"),
              ("T_RS", "Throwing RS", "Throwing runs (IQ model, any period): (caught - caught-stealing chance) x run "
               "swing of an out vs a stolen base, on every steal attempt", "num1s"), "|",
              ("DRS/S", "DRS / Season", "Catching DRS per 1,200 innings", "num1s"),
              ("F_RS/S", "Framing RS / Season", "Framing runs per 1,200 innings", "num1s"),
              ("B_RS/S", "Blocking RS / Season", "Blocking runs per 1,200 innings", "num1s"),
              ("T_RS/S", "Throwing RS / Season", "Throwing runs per 1,200 innings", "num1s")]
C_FRAMING = [("DRS", "DRS", "Catching DRS: Framing + Blocking + Throwing runs (any period)", "num1s"), ("DRS/S", "DRS / Season", "Catching DRS per 1,200 innings", "num1s"), "|",
             ("F_OPPS", "Framing Opps", "# of framing opportunities: every taken pitch (called strike or ball) he "
              "received", "int"),
             ("F_RS", "Framing Runs", "Framing runs saved: each extra strike x what a strike instead of a ball is worth "
              "in that count (about 0.07 runs at 0-0, 0.59 at 3-2)", "num1s"),
             ("F_RS/S", "Framing RS / Season", "Framing runs per 1,200 innings", "num1s"), "|",
             ("F_XS", "Extra Strikes", "Strikes above (+) / below (-) a league-average catcher on the same pitches.\n"
              "Each taken pitch: (1 if called a strike, 0 if a ball) - its strike chance. A 45% pitch called a "
              "strike = +0.55, called a ball = -0.45. Strike chance = league-average umpire, from location vs the "
              "hitter's zone, count, batter side and pitch type", "num1s"),
             ("F_XS/G", "Extra Strikes / Game", "Extra Strikes per game caught", "num2s"),
             ("F_SLAA", "SLAA", "Strikes Looking Above Average per framing opportunity\n[Extra Strikes / Framing Opps]",
              "pcts"),
             ("F_SLAA+", "SLAA+", "Called strikes vs expected, 100 = average\n[100 x called strikes / expected called "
              "strikes; 105 = 5% more strikes than an average catcher]", "int"),
             ("F_rSLAA", "SLAA (regressed)", "Best read of his framing SKILL: SLAA with league-average pitches added so "
              "small samples don't swing\n[Extra Strikes / (Framing Opps + 720 before 2023, + 1,080 from 2023)]", "pcts"),
             "|",
             ("F_AS", "Framing AS", "Strike rate on edge pitches, arm side — percentile vs MLB catchers", "pctile"),
             ("F_GS", "Framing GS", "Strike rate on edge pitches, glove side — percentile vs MLB catchers", "pctile"),
             ("F_TOZ", "Framing TOZ", "Strike rate on edge pitches, top of zone — percentile vs MLB catchers", "pctile"),
             ("F_BOZ", "Framing BOZ", "Strike rate on edge pitches, bottom of zone — percentile vs MLB catchers",
              "pctile"), "|",
             _g("Frame Speed Grade", "20-80 grade for glove speed towards middle of zone (our SAM 3 glove model)")]
C_BLOCKING = [("DRS", "DRS", "Catching DRS: Framing + Blocking + Throwing runs (any period)", "num1s"),
              ("DRS/S", "DRS / Season", "Catching DRS per 1,200 innings", "num1s"), "|",
              ("B_OPPS", "Blocking Opps", "Genuine blocking opportunities: pitches with a runner on whose wild pitch / "
               "passed ball chance was 5%+ (Statcast's Medium + Tough). Totals below use EVERY pitch with a runner on",
               "int"),
              ("B_BLOCKS", "Blocks", "Genuine opportunities he kept in front", "int"),
              ("B_BLK%", "Block %", "Blocks / genuine opportunities", "pct"),
              ("B_XBLK%", "xBlock %", "Block % an average catcher would have on the same pitches", "pct"),
              ("B_BAA", "Blocks Above Avg", "Statcast's Blocks Above Average: sum over every pitch with a runner on of "
               "(WP/PB chance - what happened). A 10% pitch blocked = +0.10; a 2% pitch that gets by = -0.98",
               "num1s"), "|",
              ("B_RS", "Blocking RS", "Blocking runs, SITUATIONAL (goes into DRS): each pitch's (chance - what happened) "
               "x what a WP / PB would cost in that base / out situation", "num1s"),
              ("B_RS_SC", "Blocking RS (Statcast)", "Statcast's way: Blocks Above Average x 0.25 runs (no situation)",
               "num1s"),
              ("B_RS/S", "Blocking RS / Season", "Situational blocking runs per 1,200 innings", "num1s"),
              ("B_BIG", "Big Stop Runs", "Situational runs saved on TOUGH pitches (15%+ WP/PB chance) he kept in front",
               "num1s"), "|",
              ("B_TOUGH_N", "Tough Opps", "Pitches with a 15%+ WP/PB chance (Statcast: under 85% block probability)",
               "int"),
              ("B_TOUGH%", "Tough Block %", "Share of tough pitches he kept in front", "pct"),
              ("B_MED_N", "Medium Opps", "Pitches with a 5-15% WP/PB chance (Statcast: 85-95% block probability)", "int"),
              ("B_MED%", "Medium Block %", "Share of medium pitches he kept in front", "pct"), "|",
              ("B_AIR", "Blocking Air", "Blocks above average per pitch, pitches not in the dirt -- percentile vs MLB "
               "catchers", "pctile"),
              ("B_DGS", "Dirt Glove Side", "... in the dirt, glove side (catcher's left, left of the plate) -- percentile",
               "pctile"),
              ("B_DMID", "Dirt Middle", "... in the dirt, over the plate -- percentile", "pctile"),
              ("B_DAS", "Dirt Arm Side", "... in the dirt, arm side (catcher's right, right of the plate) -- percentile",
               "pctile")]
C_THROWING = [("DRS", "DRS", "Catching DRS: Framing + Blocking + Throwing runs (any period)", "num1s"), ("DRS/S", "DRS / Season", "Catching DRS per 1,200 innings", "num1s"), "|",
              ("T_OPPS", "Throwing Opps", "# of throwing opportunities (stolen base attempts)", "int"),
              ("T_RS", "Throwing RS", "Throwing runs (IQ model, any period): on every steal attempt, (caught - "
               "caught-stealing chance for an average catcher: base, pitch type, velo, location, pitcher hand, "
               "count, batter side; no runner speed) x run swing of an out vs a stolen base", "num1s"),
              ("T_RS/S", "Throwing RS / Season", "Throwing runs per 1,200 innings", "num1s"), "|",
              ("POP", "Pop Time", "Time elapsed between catcher receiving pitch and throw landing at base "
                                  "(to 2B, Statcast, full seasons)", "time"),
              ("XFER", "Xfer", "Exchange time: glove to release (Statcast, full seasons)", "time"),
              ("ARM", "Arm Strength", "Max-effort arm strength (Statcast, full seasons)", "mph"),
              _nb("Accuracy Grade", "throw accuracy (needs throw tracking)"), "|",
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


# ----------------------------------------------------------------------------
# 20-80 grades from Savant season boards (jump, arm strength)
# ----------------------------------------------------------------------------
JUMP_K = 30          # 2-star+ opportunities before half of a jump number is trusted (until the study measures it)


def _num(board: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(board[c], errors="coerce")


def _find(board: pd.DataFrame, needles: tuple, exclude: tuple = ()) -> str | None:
    for c in board.columns:
        lc = c.lower()
        if all(n in lc for n in needles) and not any(x in lc for x in exclude):
            return c
    return None


def board_grade(board: pd.DataFrame | None, row: pd.Series | None, needles: tuple, n_needles: tuple | None = None,
                k: float = 0, min_n: float = 0, exclude: tuple = ()) -> tuple[float, str]:
    """(20-80 grade, hover text) for one player's value in a Savant board column, vs everyone on the board with
    min_n+ opportunities. Regressed by n / (n + k) when k > 0."""
    if board is None or board.empty or row is None:
        return math.nan, ""
    c = _find(board, needles, exclude)
    if c is None:
        return math.nan, ""
    x = pd.to_numeric(row.get(c), errors="coerce")
    if pd.isna(x):
        return math.nan, ""
    vals = _num(board, c)
    nc = _find(board, n_needles) if n_needles else None
    n = float(pd.to_numeric(row.get(nc), errors="coerce")) if nc else math.nan
    if nc and min_n:
        vals = vals[_num(board, nc) >= min_n]
    vals = vals.dropna()
    if len(vals) < 15 or not vals.std():
        return math.nan, ""
    shrink = n / (n + k) if k and n == n else 1.0
    g = 50 + 10 * (x - vals.mean()) * shrink / vals.std()
    tail = f" | {int(n)} opps, {100 * shrink:.0f}% trusted" if k and n == n else ""
    return float(g), f"{x:+.1f} vs board avg {vals.mean():+.1f}{tail}"


def arm_column(positions: list[str]) -> tuple:
    """Savant arm board column for these positions (one position -> arm_ss; several -> arm_inf / arm_of)."""
    if len(positions) == 1:
        return ("arm_" + positions[0].lower(),)
    return ("arm_inf",) if set(positions) <= set(INFIELD) else ("arm_of",)
