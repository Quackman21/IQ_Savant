import math

import numpy as np
import pandas as pd

from core import hit_metrics as hm


def _bip(ev, la, hc_x=125.42, hc_y=100, stand="R", **kw):
    row = dict(type="X", description="hit_into_play", events="field_out", launch_speed=ev,
               launch_angle=la, hc_x=hc_x, hc_y=hc_y, stand=stand, zone=5, woba_value=0, woba_denom=1,
               estimated_woba_using_speedangle=0.5, launch_speed_angle=1)
    row.update(kw)
    return row


def _pitch(desc, zone, **kw):
    return dict(type="S" if desc != "ball" else "B", description=desc, events=None, zone=zone,
                stand="R", **kw)


def df():
    rows = [
        _pitch("swinging_strike", 5), _pitch("called_strike", 5), _pitch("ball", 12),
        _pitch("swinging_strike", 13), _pitch("foul", 4),
        _bip(105, 25, launch_speed_angle=6) | {"estimated_woba_using_speedangle": 1.5},   # barrel, FB, middle
        _bip(45, 0),              # weak GB (EV <= 50, LA in -30..10)
        _bip(85, 5),              # GB
        _bip(60, 30),             # weak FB (EV <= 70, LA 10..51)
        _bip(90, 60),             # weak FB (LA >= 51)
        _bip(100, 20, hc_x=60, hc_y=120),   # pulled for RHB (left field), FB, airpull
    ]
    d = pd.DataFrame(rows)
    d["p_throws"] = "R"
    return d


def test_batted_ball_types_partition():
    v = hm.compute(hm.prep(df())).iloc[0]
    assert v["TRK"] == 6 and v["BIP"] == 6
    assert math.isclose(v["WEAK GB%"], 1 / 6) and math.isclose(v["GB%"], 1 / 6)
    assert math.isclose(v["FB%"], 2 / 6) and math.isclose(v["WEAK FB%"], 2 / 6)
    assert math.isclose(v["WEAK GB%"] + v["GB%"] + v["FB%"] + v["WEAK FB%"], 1.0)


def test_swing_decisions():
    v = hm.compute(hm.prep(df())).iloc[0]
    # pitches in zone: 5,5,4 + six BIP at zone 5 = 9 ; swings in zone: whiff(5), foul(4), 6 BIP = 8
    assert v["PIT"] == 11 and v["SWG"] == 9
    assert math.isclose(v["Z-SWING%"], 8 / 9)
    assert math.isclose(v["O-SWING%"], 1 / 2)          # 1 swing (zone 13) of 2 out-of-zone pitches
    assert math.isclose(v["WHIFF%"], 2 / 9) and math.isclose(v["O-WHIFF%"], 1.0)


def test_spray_and_air():
    v = hm.compute(hm.prep(df())).iloc[0]
    assert math.isclose(v["PULL%"], 1 / 6)
    assert math.isclose(v["AIRPULL%"], 1 / 6)
    assert math.isclose(v["FB-PULL%"], 1 / 2)          # 1 of 2 fly balls pulled
    assert math.isclose(v["PULL-AIR%"], 1.0)           # the 1 pulled ball is a fly ball
    assert math.isclose(v["BARREL%"], 1 / 6)


def test_ev_stats():
    v = hm.compute(hm.prep(df())).iloc[0]
    evs = [105, 45, 85, 60, 90, 100]
    assert math.isclose(v["EV"], np.mean(evs)) and v["MAX EV"] == 105
    assert v["HHEV"] == 105                            # top 1/8 of 6 -> 1 ball
    assert math.isclose(v["EV>95"], 2 / 6) and math.isclose(v["EV<80"], 2 / 6)


def test_xwoba_model_slot():
    v = hm.compute(hm.prep(df(), model="savant")).iloc[0]
    assert math.isclose(v["xwOBACON"], (1.5 + 5 * 0.5) / 6)
    from models import xwoba
    xwoba.MODELS["flat"] = lambda d: pd.Series(0.3, index=d.index)
    v2 = hm.compute(hm.prep(df(), model="flat")).iloc[0]
    assert math.isclose(v2["xwOBACON"], 0.3)


def test_rows_and_ref_keys():
    d = hm.prep(df())
    rows = hm.player_rows(d, "zone_io", ["platoon"])
    labels = [r[0] for r in rows]
    assert labels[0] == {"ZONE": "All", "platoon": "vs RHP"}
    assert {"ZONE": "In Zone", "platoon": "vs RHP"} in labels
    assert hm.ref_key({"ZONE": "In Zone", "platoon": "vs RHP"}, "zone_io") == ("zone_io+platoon", "In Zone|vs RHP")
    assert hm.ref_key({"ZONE": "All"}, "zone_io") == ("all", "All")


def test_format():
    assert hm.fmt("SWING%", 0.4444) == "44.4%" and hm.fmt("xwOBA", 0.368) == ".368"
    assert hm.fmt("PA", 186800) == "186.8K" and hm.fmt_delta("xwOBA", 0.053) == "+.053"


def test_league_reference_per_hitter_avgs():
    from core import league_ref
    a = df(); a["batter"] = 1
    b = df(); b["batter"] = 2
    b["launch_speed"] = b["launch_speed"] - 10
    ref = league_ref.build(hm.prep(pd.concat([a, b], ignore_index=True)), min_pa=0)
    row = ref[(ref.dims == "all") & (ref.stat == "MAX EV")].iloc[0]
    assert row["avg"] == (105 + 95) / 2                 # average hitter's max, not the league max
    row = ref[(ref.dims == "all") & (ref.stat == "EV>95")].iloc[0]
    assert math.isclose(row["avg"], 2 / 12)             # rates stay pooled
    assert set(ref.dims) >= {"all", "zone_io", "platoon", "zone_io+platoon"}   # (no pitch locations -> no attack zones)


def test_nullable_pybaseball_types():
    """pybaseball returns nullable dtypes (<NA>); the 2015 backfill crashed on these."""
    untracked = _bip(None, None, hc_x=None, hc_y=None, stand="L", events="single", p_throws="R")
    raw = pd.DataFrame(df().to_dict("records") + [untracked])
    nullable = raw.convert_dtypes()
    assert str(nullable["launch_speed"].dtype) in ("Float64", "Int64")   # nullable, like pybaseball
    v = hm.compute(hm.prep(nullable)).iloc[0]
    assert v["BIP"] == 7 and v["TRK"] == 6                         # untracked ball counted, not crashing
    from core import league_ref
    nullable["batter"] = 1
    assert not league_ref.build(hm.prep(nullable), min_pa=0).empty


def _located(desc, px, pz, strikes=0, pitch="FF", **kw):
    """A pitch at a plate location (zone 3.4 / 1.6 ft): center = (0, 2.5)."""
    return dict(type="S" if desc != "ball" else "B", description=desc, events=None, plate_x=px, plate_z=pz,
                sz_top=3.4, sz_bot=1.6, strikes=strikes, balls=0, pitch_type=pitch, stand="R",
                zone=5 if abs(px) < 0.7 and 1.6 < pz < 3.4 else 13, **kw)


def test_decisions_placeholder_and_swing_zones():
    rows = [
        _located("foul", 0, 2.5),                       # heart swing  -> good
        _located("called_strike", 0, 2.5),              # heart take   -> bad
        _located("ball", 0, 3.9),                       # chase take   -> good
        _located("swinging_strike", 0, 3.9, strikes=2), # chase swing  -> bad (2k)
        _located("called_strike", 0.75, 2.5, strikes=2),# shadow take with 2k -> bad
        _located("ball", 0.75, 2.5),                    # shadow take <2k -> fine
    ]
    v = hm.compute(hm.prep(pd.DataFrame(rows))).iloc[0]
    assert v["DECISIONS"] == 6 and math.isclose(v["GD%"], 3 / 6)
    assert v["2K DECISIONS"] == 2 and v["GD% 2K"] == 0
    assert math.isclose(v["H-SWING%"], 0.5) and v["C-SWING%"] == 0.5
    assert v["2K O-SWING%"] == 0.5                      # 2 two-strike pitches out of zone, 1 swung at
    assert math.isclose(v["H-CONT%"], 1.0)              # the heart swing was a foul (contact)


def test_ball_flight_and_barrel_definition():
    rows = [_bip(100, 20, estimated_ba_using_speedangle=0.6, estimated_slg_using_speedangle=1.6),   # barrel (xBA/xSLG)
            _bip(100, 20, estimated_ba_using_speedangle=0.6, estimated_slg_using_speedangle=1.2),   # not a barrel
            _bip(96, 40), _bip(80, 5)]
    v = hm.compute(hm.prep(pd.DataFrame(rows))).iloc[0]
    assert math.isclose(v["BARREL%"], 1 / 4)
    assert math.isclose(v["LD%"], 2 / 4)                # LA 20, 20
    assert math.isclose(v["LAHHB"], (20 + 20 + 40) / 3) # hard-hit = EV >= 95
    assert v["MAX EV (FLY)"] == 100 and math.isclose(v["EV (FLY)"], (100 + 100 + 96) / 3)


def test_bat_tracking():
    rows = [_pitch("foul", 5, bat_speed=70, swing_length=7), _pitch("swinging_strike", 5, bat_speed=76, swing_length=7.4),
            _pitch("called_strike", 5, bat_speed=None)]
    v = hm.compute(hm.prep(pd.DataFrame(rows))).iloc[0]
    assert v["# BAT SPEEDS"] == 2 and v["AVG BAT SPEED"] == 73 and v["TOP 8TH BAT SPEED"] == 76
