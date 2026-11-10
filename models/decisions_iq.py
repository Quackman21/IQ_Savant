"""IQ Swing Decisions (v1): was each swing / take a good decision for a LEAGUE-AVERAGE hitter?

No hitter strengths: every hitter is judged against the same standard. Judged by expected
runs, never by what happened (a good swing that pops up is still a good decision).

On every pitch the hitter had two options:
  TAKE  value = P(called strike) x [strike cost] + P(ball) x [ball value]
  SWING value = P(miss) x [strike cost] + P(foul) x [foul value] + P(in play) x [contact value]

  - P(called strike): location vs the hitter's own zone, batter side, count (umpires call a
    smaller zone at 0-2 and a bigger one at 3-0). League-average umpire.
  - P(miss / foul / in play): league-average swings at this pitch (location, pitch group, velo,
    movement toward / away from the hitter, same / opposite hand, count).
  - Foul: before 2 strikes a foul IS a strike (no credit). With 2 strikes a foul keeps the
    at-bat alive (worth 0 -- the loss avoided). A caught foul tip is a strike / strikeout.
  - Contact value, measured against the count you're already in (contact at 3-0 has to beat a
    great count; at 0-2 almost any contact helps):
      before 2 strikes -- HUNT DAMAGE: P(damage contact) x [value of damage contact] +
                          P(weak contact) x [value of an out]. Heart pitches (likely damage) say
                          swing; edge pitches (likely weak contact) say take.
      with 2 strikes  -- realistic: expected xwOBA on contact for this pitch, in runs.
  - Ball / strike / contact values come from real run values (delta_run_exp) in each count.

Order of importance (built in, not hoped for):
  1. COUNT     sets the stakes
  2. LOCATION  sets the odds
  3. RUNNERS + OUTS adjust the stakes at BASEOUT_WEIGHT strength, and can only decide CLOSE
     calls: if count + location alone says a decision is clearly good / bad (|z| >= CLEAR),
     the situation can't flip it.
  (4. game state -- planned as the smallest layer, not in v1)

Grade per pitch: decision value = chosen option minus the other, divided by what was at stake
in that count (so a clear mistake at 0-0 counts like one at 3-2). Good (1) / bad (0) /
neutral (NaN, true coin flips within +-NEUTRAL). The run value is kept for impact.

Trained by pipeline/train_decisions.py -> models/artifacts/decisions_iq_v1.joblib
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "decisions_iq_v1.joblib"
SHIFT_BAN = 2023          # 2015-2022 and 2023+ (pitch clock, shift ban, bigger bases) get separate models

BASEOUT_WEIGHT = 0.5      # runners + outs: half strength
CLEAR = 0.25              # count + location alone this sure -> situation can't flip it
NEUTRAL = 0.05            # |z| below this = coin flip, no grade
SHRINK = 200              # pitches needed before a count x runners x outs cell trusts its own value

MISS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}       # foul tip caught = strike
FOUL = {"foul"}
IN_PLAY = {"hit_into_play"}
BALL = {"ball", "blocked_ball", "hit_by_pitch"}
CALLED = {"called_strike"}
# bunts, pitchouts, intentional balls, pitch-clock automatic balls / strikes: not decisions
SKIP = {"foul_bunt", "missed_bunt", "bunt_foul_tip", "foul_pitchout", "pitchout", "intent_ball",
        "automatic_ball", "automatic_strike"}

# IQ zones (coaching tiers, inside -> out). r = how far from the middle of the hitter's zone, where 1.0 =
# the edge of the zone (the larger of the side-to-side and up-and-down distances). Heart / Sub-heart / Edge
# are all INSIDE the zone; "Edge (off)" = just off it with the ball still touching it.
IQ_ZONES = [(0.50, "Heart"), (0.80, "Sub-heart"), (1.00, "Edge"), (1.17, "Edge (off)"), (2.00, "Chase")]
ZONE_ORDER = [z for _, z in IQ_ZONES] + ["Waste"]


def iq_zone(df: pd.DataFrame) -> pd.Series:
    c = lambda n: pd.to_numeric(df[n], errors="coerce") if n in df else pd.Series(np.nan, index=df.index)
    top, bot = c("sz_top"), c("sz_bot")
    nx = c("plate_x").abs() / (17 / 24)
    nz = (c("plate_z") - (top + bot) / 2).abs() / ((top - bot) / 2)
    r = np.maximum(nx, nz)
    out = pd.Series(np.select([r < lim for lim, _ in IQ_ZONES], [z for _, z in IQ_ZONES], "Waste"), index=df.index)
    return out.where(r.notna())


PITCH_FEATURES = ["x_in", "z_norm", "speed", "ivb", "move_in", "group", "same_hand", "balls", "strikes"]
CALLED_FEATURES = ["x_in", "z_norm", "is_left", "balls", "strikes", "group"]   # umpires call pitch types a bit differently
QUALITY_CAP = 1.5
# Before 2 strikes a swing is judged on DAMAGE: contact counts only if it's damage-quality (expected
# xwOBA >= DAMAGE_XWOBA); weak contact is valued as an out (a wasted swing). With 2 strikes contact is
# valued realistically (putting it in play beats striking out).
DAMAGE_XWOBA = 0.500
OUT_EVENTS = {"field_out", "force_out", "grounded_into_double_play", "double_play", "fielders_choice_out",
              "fielders_choice", "sac_fly", "sac_fly_double_play", "triple_play", "other_out"}         # one pitch can't move Decision Quality more than this (x100 on the site)


def action(df: pd.DataFrame) -> pd.Series:
    """swing / take / None (not a decision we grade)."""
    desc = df["description"].astype(str)
    bunt = df["des"].astype(str).str.contains("bunt", case=False, na=False) if "des" in df else False
    out = pd.Series(None, index=df.index, dtype=object)
    out[desc.isin(MISS | FOUL | IN_PLAY)] = "swing"
    out[desc.isin(BALL | CALLED)] = "take"
    out[desc.isin(SKIP) | (desc.isin(IN_PLAY) & bunt)] = None
    return out


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Pitch as the hitter saw it. x_in: + = inside, for both sides. z_norm: 0 = bottom of HIS
    zone, 1 = top. move_in: horizontal movement toward (+) / away from (-) the hitter."""
    from core import hit_metrics as hm
    c = lambda n: pd.to_numeric(df[n], errors="coerce") if n in df else pd.Series(np.nan, index=df.index)
    left = df["stand"].astype(str).eq("L") if "stand" in df else pd.Series(False, index=df.index)
    side = np.where(left, 1.0, -1.0)                       # catcher's view: RHB inside = negative x
    top, bot = c("sz_top"), c("sz_bot")
    span = (top - bot).where((top - bot) > 0.5)
    group = hm.pitch_group(df).map({"Fastball": 0.0, "Breaking": 1.0, "Offspeed": 2.0})
    throws = df["p_throws"].astype(str) if "p_throws" in df else pd.Series("R", index=df.index)
    return pd.DataFrame({
        "x_in": c("plate_x") * side, "z_norm": (c("plate_z") - bot) / span, "speed": c("release_speed"),
        "ivb": c("pfx_z") * 12, "move_in": c("pfx_x") * 12 * side, "group": group,
        "same_hand": (throws.values == np.where(left, "L", "R")).astype(float),
        "is_left": left.astype(float), "balls": c("balls").clip(0, 3), "strikes": c("strikes").clip(0, 2),
    }, index=df.index)


def state(df: pd.DataFrame) -> pd.DataFrame:
    c = lambda n: pd.to_numeric(df[n], errors="coerce") if n in df else pd.Series(np.nan, index=df.index)
    on = lambda n: df[n].notna().astype(int) if n in df else pd.Series(0, index=df.index)
    count = c("balls").clip(0, 3).astype("Int64").astype(str) + "-" + c("strikes").clip(0, 2).astype("Int64").astype(str)
    base = on("on_1b") + 2 * on("on_2b") + 4 * on("on_3b")
    outs = c("outs_when_up").clip(0, 2).fillna(0).astype(int)
    return pd.DataFrame({"count": count, "bo": count + "|" + base.astype(str) + "|" + outs.astype(str),
                         "two_k": c("strikes").ge(2)}, index=df.index)


# ----------------------------------------------------------------------------
_cache: dict = {}


def load() -> dict | None:
    if "m" not in _cache:
        if not ARTIFACT.exists():
            return None
        import joblib
        _cache["m"] = joblib.load(ARTIFACT)
    return _cache["m"]


def _look(table: dict, keys: pd.Series, fallback: pd.Series | float) -> pd.Series:
    v = keys.map(table).astype(float)
    return v.fillna(fallback) if isinstance(fallback, pd.Series) else v.fillna(fallback)


def _values(st: pd.DataFrame, b: dict, layer: str, p_cs, p_miss, p_foul, p_bip, xq, p_dmg=None) -> tuple:
    """(take value, swing value) in runs, from the 'count' or 'bo' (count x runners x outs) tables."""
    key = st[layer]
    t = b["tables"][layer]
    core = b["tables"]["count"]
    fb = lambda name: _look(core[name], st["count"], np.nan)
    ball = _look(t["ball"], key, fb("ball"))
    strike = _look(t["strike"], key, fb("strike"))
    bip = _look(t["bip"], key, fb("bip"))
    foul = strike.where(~st["two_k"], 0.0)                 # pre-2K foul = strike; 2K foul = alive
    contact = bip + b["runs_per_xwoba"] * (xq - b["xq_mean"])          # realistic (2 strikes)
    if p_dmg is not None and "bip_dmg" in t and "bip_out" in t:
        dmg = _look(t["bip_dmg"], key, _look(core["bip_dmg"], st["count"], bip))
        weak = _look(t["bip_out"], key, _look(core["bip_out"], st["count"], bip))
        hunt = p_dmg * dmg + (1 - p_dmg) * weak                        # before 2 strikes: damage or nothing
        contact = contact.where(st["two_k"], hunt)
    take = p_cs * strike + (1 - p_cs) * ball
    swing = p_miss * strike + p_foul * foul + p_bip * contact
    stakes = (fb("ball") - fb("strike")).abs()
    return take, swing, stakes


def score(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Per pitch: action, take / swing value (runs), decision runs, z, good (1 / 0 / NaN), and
    whether runners + outs changed the grade. Rows that aren't gradeable get NaN."""
    b = bundle or load()
    if b is not None and "eras" in b and not df.empty:   # each pitch graded by its own era's model
        yr = pd.to_datetime(df["game_date"]).dt.year if "game_date" in df else pd.Series(9999, index=df.index)
        parts = [score(df[m], b["eras"][e]) for e, m in (("pre", yr < b["split"]), ("post", yr >= b["split"]))
                 if m.any() and b["eras"].get(e)]
        return pd.concat(parts).reindex(df.index) if parts else score(df.iloc[0:0], b["eras"]["post"]).reindex(df.index)
    cols = ["action", "p_called", "p_swing_avg", "take_runs", "swing_runs", "decision_runs", "z", "z_core",
            "good", "quality", "runs_vs_avg", "flipped"]
    out = pd.DataFrame(np.nan, index=df.index, columns=cols)
    if b is None or df.empty:
        return out
    act = action(df)
    f = features(df)
    ok = act.notna() & f[["x_in", "z_norm"]].notna().all(axis=1) & f["balls"].notna() & f["strikes"].notna()
    out["action"] = act
    if not ok.any():
        return out
    d, fd, st = df[ok], f[ok], state(df[ok])
    p_cs = pd.Series(b["called"].predict_proba(fd[CALLED_FEATURES])[:, 1], index=d.index)
    sw = b["swing"].predict_proba(fd[PITCH_FEATURES])
    cls = list(b["swing"].classes_)
    p_miss, p_foul, p_bip = (pd.Series(sw[:, cls.index(k)], index=d.index) for k in ("miss", "foul", "bip"))
    xq = pd.Series(b["contact"].predict(fd[PITCH_FEATURES]), index=d.index)
    p_dmg = pd.Series(b["damage"].predict_proba(fd[PITCH_FEATURES])[:, 1], index=d.index) if "damage" in b else None
    sign = np.where(act[ok].eq("swing"), 1.0, -1.0)

    take_c, swing_c, stakes = _values(st, b, "count", p_cs, p_miss, p_foul, p_bip, xq, p_dmg)
    take_b, swing_b, _ = _values(st, b, "bo", p_cs, p_miss, p_foul, p_bip, xq, p_dmg)
    d_core = sign * (swing_c - take_c)
    d_full = d_core + BASEOUT_WEIGHT * (sign * (swing_b - take_b) - d_core)
    stakes = stakes.where(stakes > 1e-6)
    z_core, z = d_core / stakes, d_full / stakes

    good = pd.Series(np.where(z > 0, 1.0, 0.0), index=d.index)
    good[z.abs() < NEUTRAL] = np.nan
    clear = z_core.abs() >= CLEAR
    good[clear] = np.where(z_core[clear] > 0, 1.0, 0.0)
    core_grade = pd.Series(np.where(z_core > 0, 1.0, 0.0), index=d.index).where(z_core.abs() >= NEUTRAL)
    flipped = ~((good == core_grade) | (good.isna() & core_grade.isna()))

    take_f = take_c + BASEOUT_WEIGHT * (take_b - take_c)
    swing_f = swing_c + BASEOUT_WEIGHT * (swing_b - swing_c)
    p_sw = pd.Series(b["swing_rate"].predict_proba(fd[PITCH_FEATURES])[:, 1], index=d.index)
    chosen = np.where(sign > 0, swing_f, take_f)
    vs_avg = chosen - (p_sw * swing_f + (1 - p_sw) * take_f)
    out.loc[ok, ["p_called", "p_swing_avg", "take_runs", "swing_runs", "decision_runs", "z", "z_core",
                 "quality", "runs_vs_avg"]] = np.column_stack(
        [p_cs, p_sw, take_f, swing_f, d_full, z, z_core, 100 * z.clip(-QUALITY_CAP, QUALITY_CAP), vs_avg])
    out.loc[ok, "good"] = good
    out.loc[ok, "flipped"] = flipped.astype(float)
    return out


# ----------------------------------------------------------------------------
# Used by catcher framing (models/framing.py): same league-umpire strike chance, same count values
# ----------------------------------------------------------------------------
def _by_era(df: pd.DataFrame, b: dict):
    """(rows, bundle) pairs -- era bundles get their own rows."""
    if "eras" not in b:
        return [(pd.Series(True, index=df.index), b)]
    yr = pd.to_datetime(df["game_date"]).dt.year if "game_date" in df else pd.Series(9999, index=df.index)
    return [(m, b["eras"][e]) for e, m in (("pre", yr < b["split"]), ("post", yr >= b["split"])) if b["eras"].get(e)]


def called_strike_chance(df: pd.DataFrame, bundle: dict | None = None) -> pd.Series:
    """League-average-umpire chance each pitch is called a strike if taken (NaN without a location)."""
    b = bundle or load()
    out = pd.Series(np.nan, index=df.index)
    if b is None or df.empty:
        return out
    f = features(df)
    ok = f[["x_in", "z_norm"]].notna().all(axis=1) & f["balls"].notna() & f["strikes"].notna()
    for m, sub in _by_era(df, b):
        rows = m & ok
        if rows.any():
            out[rows] = sub["called"].predict_proba(f.loc[rows, CALLED_FEATURES])[:, 1]
    return out


def strike_value(df: pd.DataFrame, bundle: dict | None = None) -> pd.Series:
    """Runs a strike instead of a ball is worth in each pitch's count (ball value - strike value)."""
    b = bundle or load()
    out = pd.Series(np.nan, index=df.index)
    if b is None or df.empty:
        return out
    cnt = state(df)["count"]
    for m, sub in _by_era(df, b):
        t = sub["tables"]["count"]
        out[m] = (cnt[m].map(t["ball"]) - cnt[m].map(t["strike"])).astype(float)
    return out

