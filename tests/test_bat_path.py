import numpy as np
import pandas as pd

from core import batted_ball, hit_metrics as hm


def _swings(n=300, seed=4):
    rng = np.random.default_rng(seed)
    desc = rng.choice(["hit_into_play", "foul", "swinging_strike", "ball"], n, p=[.3, .3, .2, .2])
    df = pd.DataFrame({
        "batter": 1, "game_date": "2025-06-01", "game_pk": 1, "game_type": "R", "stand": rng.choice(["L", "R"], n),
        "p_throws": rng.choice(["L", "R"], n), "pitch_type": rng.choice(["FF", "SL", "CH"], n),
        "description": desc, "events": np.where(desc == "hit_into_play", "single", None),
        "type": np.where(desc == "hit_into_play", "X", "S"), "zone": rng.integers(1, 15, n),
        "bat_speed": np.where(desc != "ball", rng.normal(72, 5, n), np.nan),
        "swing_length": 7.2, "attack_angle": rng.normal(10, 8, n), "attack_direction": rng.normal(-2, 10, n),
        "swing_path_tilt": rng.normal(32, 4, n), "release_speed": rng.normal(91, 4, n),
        "launch_speed": np.where(np.isin(desc, ["hit_into_play", "foul"]), rng.normal(88, 12, n), np.nan),
        "launch_angle": rng.normal(12, 25, n), "hc_x": rng.normal(125, 40, n), "hc_y": rng.normal(120, 30, n),
        "intercept_ball_minus_batter_pos_y_inches": rng.normal(30, 5, n),
        "intercept_ball_minus_batter_pos_x_inches": rng.normal(28, 3, n), "launch_speed_angle": rng.integers(1, 7, n),
    })
    return df


def test_vector_labels_match_row_labels():
    df = _swings()
    vec = batted_ball.spin_labels(df)
    for i, r in df.iterrows():
        one = batted_ball.spin_label(r)
        assert (one["label"] if one else None) == vec[i]


def test_bat_path_metrics():
    df = _swings()
    df["poc_front"] = df["intercept_ball_minus_batter_pos_y_inches"] - 28
    v = hm.compute(hm.prep(df)).iloc[0]
    made = df["description"].isin(["foul", "hit_into_play"])
    assert abs(v["POC DEPTH"] - df.loc[made, "intercept_ball_minus_batter_pos_y_inches"].mean()) < 1e-9
    assert abs(v["POC FRONT"] - (v["POC DEPTH"] - 28)) < 1e-9
    assert 0 <= v["SQ UP%/CON"] <= 1 and v["SQ UP%/SW"] <= v["SQ UP%/CON"]
    assert v["BLAST%/CON"] <= v["SQ UP%/CON"] + 1e-9
    types = ["OPTIMAL%", "BACKSPIN%", "TOPSPIN%", "HOOK%", "SLICE%", "KNUCKLE%"]
    assert abs(sum(v[t] for t in types) - 1) < 1e-9
    for keys in hm.BAT_PATH.values():
        assert all(k in hm.CATALOG for k in keys)


def test_poc_map_renders():
    from ui.bat_path import poc_map_html
    m = pd.DataFrame({"pf": [5.0, -3.0, 10.0], "pxc": [1.0, -4.0, 2.0], "bip": [True, False, True],
                      "ev": [101.0, np.nan, 80.0], "la": [20.0, np.nan, -5.0], "pgroup": ["Fastball"] * 3,
                      "ptype": ["FF"] * 3, "p_throws": ["R"] * 3, "zone_io": ["In Zone"] * 3,
                      "date": ["2025-06-01"] * 3, "result": ["Single", "Foul", "Field out"]})
    h = poc_map_html(m)
    assert h.count("<circle") == 3 and "Where he meets it" in h
    assert "No measured" in poc_map_html(m.assign(pf=np.nan))


def test_poc_uses_stance_depth():
    from core import bat_path
    df = _swings(40).assign(plate_x=0.0, vx0=0.0, vy0=-130.0, ax=0.0, ay=25.0)
    views = {"season": {(1, "R"): 27.0, (1, "L"): 29.0}}
    out = bat_path.poc(df, 1, 2025, views, {})
    made = df["description"].isin(["foul", "hit_into_play"])
    exp = df["intercept_ball_minus_batter_pos_y_inches"] - np.where(df["stand"] == "R", 27.0, 29.0)
    assert np.allclose(out.loc[made, "pf"], exp[made]) and out.loc[~made, "pf"].isna().all()
    assert not out["estimated"].any()


def _tracked(n=400, seed=7):
    df = _swings(n, seed)
    rng = np.random.default_rng(seed)
    df["stand"] = "R"
    df["zone"] = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14], n)
    df["sz_top"], df["sz_bot"] = 3.4, 1.6
    df["description"] = np.where(df["description"] == "ball", "swinging_strike", df["description"])
    df["bat_speed"] = rng.normal(72, 4, n)
    return df


def test_synth_pitch_hits_its_target_and_looks_like_statcast():
    from core import bat_path as bp, swing3d
    for g, (pt, v, ivb, hb, _) in bp.GROUPS.items():
        for hand in ("R", "L"):
            p = bp.synth_pitch(v[hand], ivb, hb, hand, 0.4, 2.2)
            x, z = swing3d.ball_at(p, 17 / 12)
            assert abs(x - 0.4) < 1e-6 and abs(z - 2.2) < 1e-6
            rx, rz = swing3d.ball_at(p, p["release_pos_y"])
            assert abs(rx - p["release_pos_x"]) < 1e-6 and abs(rz - p["release_pos_z"]) < 1e-6
    ff = bp.synth_pitch(94.3, 14, 9, "R", 0, 2.5)
    assert -18 < ff["az"] < -13 and -13 < ff["ax"] < -8                      # typical RHP four-seam


def test_zone_targets():
    from core import bat_path as bp
    assert bp.zone_target(5, 1.6, 3.4) == (0.0, 2.5)
    x1, z1 = bp.zone_target(1, 1.6, 3.4)
    assert x1 < 0 and z1 > 3.0
    x14, z14 = bp.zone_target(14, 1.6, 3.4)
    assert x14 > 0.83 and z14 < 1.6


def test_profile_widens_and_scene_builds():
    from core import bat_path as bp
    sw = bp.swings(_tracked())
    pr = bp.profile(sw, "Fastball", 5, "R", "R")
    assert pr["n"] >= bp.MIN_SWINGS and "cy" in pr
    thin = bp.profile(sw[sw.zone != 5], "Fastball", 5, "R", "R")
    assert thin["level"] in ("this pitch group (all zones)", "all his swings")
    sc = bp.model_scene(sw, "Fastball", 5, "R", 27.0, 26.0, "R")
    assert sc and sc["model"]["ptype"] == "FF" and len(sc["arc"]) > 50 and sc["exit"] is None
    assert abs(sc["read"]["front"] - (pr["cy"] - 27.0)) < 0.06
    from ui.bat_path import model_card_html, path3d_html
    h = path3d_html(sc, "t", "s")
    assert "function build" in h and 'id="stage"' in h and "open3d" not in h
    assert "Bat speed" in model_card_html(sc, "R")


def test_zone_names_follow_the_hitter():
    from core import bat_path as bp
    assert bp.zone_name(5, "R") == "Middle-Middle"
    assert bp.zone_name(1, "R") == "Up-In" and bp.zone_name(1, "L") == "Up-Away"
    assert bp.zone_name(9, "R") == "Down-Away" and bp.zone_name(9, "L") == "Down-In"
    assert bp.zone_name(13, "R") == "Chase Down-In" and bp.zone_name(12, "L") == "Chase Up-In"


def test_cell_spray_and_poc():
    from core import bat_path as bp
    from ui.bat_path import poc_map_html, spray_html
    df = _tracked()
    df["p_throws"] = "R"
    poc = pd.DataFrame({"pf": df["intercept_ball_minus_batter_pos_y_inches"] - 27, "pxc": 0.0}, index=df.index)
    cell = bp.cell_balls(df, poc, "Fastball", 5, "R", "R")
    exp = df["description"].isin(["foul", "hit_into_play"]) & df["pitch_type"].eq("FF") & df["zone"].eq(5)
    assert len(cell) == exp.sum() and cell["pf"].notna().all()
    h = spray_html(cell[cell["bip"]])
    assert h.count("<circle") == cell["bip"].sum()
    p = poc_map_html(cell, summary=False, title="POC · x")
    assert "Where he meets it" not in p and "POC · x" in p


def test_spray_on_home_park():
    from core import aar, bat_path as bp
    from ui.bat_path import spray_html
    df = _tracked()
    df["inning_topbot"] = np.where(np.arange(len(df)) % 3 == 0, "Top", "Bot")
    df["home_team"] = np.where(df["inning_topbot"] == "Bot", "BOS", "NYY")
    assert bp.home_team(df) == "BOS"
    dims = {"leftLine": 310.0, "leftCenter": 379.0, "center": 390.0, "rightCenter": 380.0, "rightLine": 302.0}
    park = {"venue": "Fenway Park", "dims": dims, "outline": aar.park_outline("Boston Red Sox", dims, venue="Fenway Park")}
    b = pd.DataFrame({"hc_x": [100.0, 150.0], "hc_y": [80.0, 100.0], "ev": [101.0, 85.0], "la": [25.0, 10.0],
                      "result": ["Home run", "Single"], "date": ["2025-06-01"] * 2, "ptype": ["FF"] * 2})
    h = spray_html(b, "Spray", park)
    assert "Fenway Park" in h and h.count("<circle") == 2 and ">310<" in h
    assert "generic park" in spray_html(b, "Spray", {})


def test_color_modes_and_heat_maps():
    from core import bat_path as bp
    from ui.bat_path import COLOR_BY, poc_map_html, spray_html
    df = _tracked(800)
    df["p_throws"], df["bb_type"] = "R", np.where(df["description"] == "hit_into_play", "line_drive", None)
    poc = pd.DataFrame({"pf": df["intercept_ball_minus_batter_pos_y_inches"] - 27, "pxc": 0.0}, index=df.index)
    c = bp.contact_rows(df, poc)
    assert set(c["bb"].dropna()) == {"LD"} and c.loc[c["bip"], "ctype"].notna().any()
    for mode in COLOR_BY:
        for disp in ("Points", "Heat Map"):
            p = poc_map_html(c, mode, summary=False, display=disp)
            s = spray_html(c[c["bip"]], "Spray", {}, mode, disp)
            assert ("<rect" in p) == (disp == "Heat Map") and ("<circle" in s) == (disp == "Points")
    assert "Line drive" in spray_html(c[c["bip"]], "S", {}, "Hit Type", "Points")
    assert "average exit velo" in spray_html(c[c["bip"]], "S", {}, "Exit Velo", "Heat Map")
    assert "how often" in poc_map_html(c, "Result", summary=False, display="Heat Map")


def test_framed_chart_is_a_page_that_sizes_itself():
    from ui.bat_path import framed
    h = framed("<svg viewBox='0 0 10 10'></svg>")
    assert h.startswith("<!doctype html>") and "<svg" in h and "setFrameHeight" in h
