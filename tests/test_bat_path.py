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
    pr = bp.profile(sw, ["FF"], "All", "All", 5, "R")
    assert pr["n"] >= bp.MIN_SWINGS and "cy" in pr and pr["level"] == "your filters, here"
    thin = bp.profile(sw[sw.zone != 5], ["FF"], "All", "All", 5, "R")
    assert thin["level"] == "your filters, all locations"
    sc = bp.model_scene(sw, None, "All", "All", 5, "R", 27.0, 26.0)
    assert sc and sc["model"]["ptype"] == "FF" and len(sc["arc"]) > 50 and sc["exit"] is None
    pall = bp.profile(sw, None, "All", "All", 5, "R")
    assert abs(sc["read"]["front"] - (pall["cy"] - 27.0)) < 0.06
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
    cell = bp.cell_balls(df, poc, ["FF"], "RHP", "All", 5, "R")
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


def test_filters_and_demo_pitch():
    from core import bat_path as bp
    df = _tracked()
    df["p_throws"] = np.where(np.arange(len(df)) % 2, "L", "R")
    sw = bp.swings(df)
    alltypes = list(sw["ptype"].unique())
    assert bp.pick(sw).all()
    assert (sw[bp.pick(sw, hand="LHP")]["hand"] == "L").all()
    assert bp.demo_group(None, alltypes) == "Fastball" and bp.demo_group(alltypes, alltypes) == "Fastball"
    assert bp.demo_group(["SL"], alltypes) == "Breaking" and bp.demo_group(["CH"], alltypes) == "Offspeed"
    assert bp.demo_group(["SL", "CH"], alltypes) == "Fastball"            # mixed -> default
    sc = bp.model_scene(sw, ["SL"], "LHP", "All", None, "R", 27.0, 26.0, alltypes)
    assert sc["model"]["ptype"] == "SL" and sc["model"]["hand"] == "L" and sc["model"]["zone"] is None
    d = bp.model_scene(sw, None, "All", "All", None, "R", 27.0, 26.0, alltypes)
    assert d["model"]["ptype"] == "FF" and d["model"]["hand"] == "R"      # dead-zone RHP fastball by default


def test_zone_lists():
    from core import bat_path as bp
    assert bp.zones_of(None) is None and bp.zones_of(list(range(1, 10))) is None and bp.zones_of([]) is None
    assert bp.zones_of(5) == [5] and bp.zones_of([6, 3, 3]) == [3, 6]
    assert bp.zones_label([5], "R") == "Middle-Middle" and bp.zones_label([1, 2], "R") == "Up-In + Up-Middle"
    assert bp.zones_label([1, 2, 3], "R") == "3 zones" and bp.zones_label(None, "L") == "All locations"
    df = _tracked()
    sw = bp.swings(df)
    assert set(sw[bp.pick(sw, zone=[1, 2])]["zone"]) == {1, 2}
    sc = bp.model_scene(sw, None, "All", "All", [1, 3], "R", 27.0, 26.0)       # aimed between up-in and up-away
    assert abs(sc["pitch"][-1][0]) < 3


def test_swing_groups_on_every_chart():
    from core import bat_path as bp
    from ui.bat_path import compare_html, path3d_html, poc_map_html, spray_html
    df = _tracked(1500)
    df["p_throws"] = "R"
    sw = bp.swings(df)
    groups = [dict(bp.DEFAULT_GROUP), {"types": ["SL"], "hand": "All", "count": "All", "zone": 9}]
    scs = bp.group_scenes(sw, groups, "R", lambda h, g: 27.0, 26.0, list(sw.ptype.unique()))
    assert [s["gid"] for s in scs] == [0, 1] and scs[0]["model"]["ptype"] == "FF" and scs[1]["model"]["ptype"] == "SL"
    h = path3d_html(scs, "Bat path", "x")
    assert '"extra"' in h and 'id="grp"' in h and "setOn" in h
    poc = pd.DataFrame({"pf": df["intercept_ball_minus_batter_pos_y_inches"] - 27, "pxc": 0.0}, index=df.index)
    cells = [bp.cell_balls(df, poc, g["types"], g["hand"], g["count"], g["zone"], "R")
             .assign(grp=k, gname=bp.group_label(g, "R"), gcolor=bp.GROUP_COLORS[k]) for k, g in enumerate(groups)]
    cell = pd.concat(cells)
    for disp in ("Points", "Heat Map"):
        p = poc_map_html(cell, "Swing Group", summary=False, display=disp)
        s = spray_html(cell[cell["bip"]], "Spray", {}, "Swing Group", disp)
        assert 'class="chips"' in p and 'class="chips"' in s and 'class="g1"' in p and 'class="g1"' in s
    assert "<rect class=\"g1\"" in poc_map_html(cell, "Exit Velo", summary=False)      # B = squares
    t = compare_html(scs)
    assert "Swing groups side by side" in t and "Attack angle" in t
    assert "No measured" in poc_map_html(pd.DataFrame(), summary=False)
    assert "no balls in play" in spray_html(pd.DataFrame())


def _raw_league(n=6000, seed=11):
    rng = np.random.default_rng(seed)
    bs = rng.normal(72, 5, n)
    return pd.DataFrame({
        "game_type": "R", "batter": rng.integers(1, 40, n), "stand": rng.choice(["L", "R"], n),
        "p_throws": rng.choice(["L", "R"], n), "pitch_type": rng.choice(["FF", "SL", "CH"], n),
        "zone": rng.integers(1, 10, n), "description": rng.choice(["foul", "hit_into_play", "swinging_strike"], n),
        "bat_speed": bs, "attack_angle": rng.normal(10, 7, n), "attack_direction": rng.normal(0, 10, n),
        "swing_path_tilt": rng.normal(32, 4, n), "intercept_ball_minus_batter_pos_y_inches": rng.normal(30, 6, n),
        "vx0": 5.0, "vy0": -135.0, "vz0": -5.0, "ax": -8.0, "ay": 28.0, "az": -15.0})


def test_bat_norms_fast_only(monkeypatch):
    from core import aar, bat_path as bp
    from pipeline import bat_norms
    raw = _raw_league()
    monkeypatch.setattr(aar, "stance_depths", lambda season: {(b, s): 28.0 for b in range(1, 40) for s in "LR"})
    d = bat_norms.swings(raw, 2026)
    assert set(d["description"]) <= {"foul", "hit_into_play"} and (d["pf"] == d["intercept_ball_minus_batter_pos_y_inches"] - 28).all()
    nm = bat_norms.build(d)
    k = nm["keys"]["All|All"]
    fast = d[d["fast"]]
    assert k["n"] == len(fast) and abs(k["contact"]["bat_speed"]["m"] - fast["bat_speed"].mean()) < 0.01
    assert min(fast["bat_speed"]) >= 75 and k["contact"]["bat_speed"]["m"] > 76
    assert "Fastball|5" in nm["keys"] and len(k["depth"]["attack_angle"]["m"]) == len(nm["bins"])
    # the site's lookup: one pitch group + one zone -> that range; mixed -> all
    assert bp.norm_for(nm, ["FF"], [5])["key"] == "Fastball|5"
    assert bp.norm_for(nm, ["FF", "SL"], [5])["key"] == "All|5"
    assert bp.norm_for(nm, None, None)["key"] == "All|All"
    assert bp.norm_for({}, None, None) == {}


def test_plane_gaps_are_measured():
    from pipeline.bat_norms import gaps
    df = pd.DataFrame({"stand": ["R", "L"], "attack_angle": [10.0, 10.0], "attack_direction": [5.0, 5.0],
                       "vx0": [0.0, 0.0], "vy0": [-135.0, -135.0], "vz0": [-5.0, -5.0], "ax": [0.0, 0.0],
                       "ay": [28.0, 28.0], "az": [-15.0, -15.0]})
    gv, gh = gaps(df, pd.Series([0.0, 0.0]))
    assert -10 < gv.iloc[0] < 10 and abs(gh.iloc[0] - 5.0) < 1e-6          # straight pitch: gap_h = attack direction


def test_graph_panel_and_trails_in_3d():
    from core import bat_path as bp
    from ui.bat_path import path3d_html
    df = _tracked(1500)
    df["p_throws"] = "R"
    df = df.assign(vx0=5.0, vy0=-135.0, vz0=-5.0, ax=-8.0, ay=28.0, az=-15.0)
    poc = pd.DataFrame({"pf": df["intercept_ball_minus_batter_pos_y_inches"] - 27, "pxc": 0.0}, index=df.index)
    sw = bp.swings(df, poc)
    assert sw["pf"].notna().any() and sw["gap_v"].notna().any()
    scs = bp.group_scenes(sw, [dict(bp.DEFAULT_GROUP)], "R", lambda h, g: 27.0, 26.0, list(sw.ptype.unique()), {})
    s = scs[0]["series"]
    assert s["cols"][0] == "pf" and len(s["pts"]) > 0 and s["n_fast"] >= 0
    h = path3d_html(scs, "Bat path", "x")
    for bit in ('id="s3trail"', 'id="tv"', "function paint", "gap_v", "gap_h"):
        assert bit in h
    assert "vs MLB" not in h and "Horizontal bat angle" not in h            # plane matching only
    from ui.bat_path import chart_html
    c = chart_html(scs, bp.series(sw, {"types": None, "hand": "All", "count": "All", "zone": None}, "R"))
    for bit in ('id="cm"', 'id="cv"', 'value="self"', 'value="league"', "function draw", '"all"'):
        assert bit in c


def test_on_plane_window_and_hba():
    from core import bat_path as bp
    from ui.bat_path import compare_html
    df = _tracked(1500)
    df["p_throws"] = "R"
    sw = bp.swings(df)
    sc = bp.model_scene(sw, None, "All", "All", [5], "R", 27.0, 26.0)
    p = bp.path_angles(sc)
    assert {"gap_v", "gap_h", "hba"} <= set(p.columns) and len(p) == len(sc["arc"])
    c = p.loc[(p["t"] >= 0).idxmax()]
    assert abs(c["gap_v"] - sc["read"]["plane_gap"]) < 3          # path angle at contact ~ measured attack angle
    w = bp.on_plane(sc, "gap_v", tol=90)
    assert w["on"] and w["ms"] > 0 and 0 <= w["pct"] <= 1
    t = compare_html([sc | {"name": "A", "color": "#2f5fb3", "gid": 0}])
    assert "Horizontal bat angle" in t and "On plane, vertical" in t


def test_bat_angles_match_the_3d_geometry():
    from core import bat_path as bp, swing3d
    row = {"vx0": 5.0, "vy0": -135.0, "vz0": -5.0, "ax": -8.0, "ay": 28.0, "az": -15.0, "plate_x": 0.1, "plate_z": 2.4,
           "bat_speed": 72, "attack_angle": 9.0, "attack_direction": -4.0, "swing_path_tilt": 33.0,
           "intercept_ball_minus_batter_pos_y_inches": 31.0, "intercept_ball_minus_batter_pos_x_inches": 28.0,
           "description": "foul", "stand": "R", "swing_length": 7.2}
    sc = swing3d.scene(row, 27.0, 26.0)
    p = bp.path_angles(sc)
    v, h = bp.bat_angles([9.0], [-4.0], [33.0], ["R"], [sc["read"]["front"]], [27.0], [sc["read"]["side_in"]],
                         [sc["read"]["height"]], {"R": 26.0})
    assert abs(v[0] - sc["read"]["vba"]) < 0.5 and abs(h[0] - p.loc[(p["t"] >= 0).idxmax(), "hba"]) < 0.5


def test_match_angles_are_the_gap_reference():
    from pipeline.bat_norms import plane
    df = pd.DataFrame({"stand": ["R", "L"], "attack_angle": [10.0, 10.0], "attack_direction": [5.0, 5.0],
                       "vx0": [3.0, 3.0], "vy0": [-135.0, -135.0], "vz0": [-5.0, -5.0], "ax": [0.0, 0.0],
                       "ay": [28.0, 28.0], "az": [-15.0, -15.0]})
    pl = plane(df, pd.Series([0.0, 0.0]))
    assert np.allclose(pl["gap_v"], df["attack_angle"] - pl["match_v"])
    assert np.allclose(pl["gap_h"], df["attack_direction"] - pl["match_h"]) and pl["match_h"][0] == -pl["match_h"][1]


def test_time_ranges_league_and_self(monkeypatch):
    from core import aar, bat_path as bp
    from pipeline import bat_norms
    raw = _raw_league().assign(plate_x=0.0, plate_z=2.5, swing_length=7.2,
                               intercept_ball_minus_batter_pos_x_inches=28.0)
    monkeypatch.setattr(aar, "stance_depths", lambda season: {(b, s): 28.0 for b in range(1, 40) for s in "LR"})
    nm = bat_norms.build(bat_norms.swings(raw, 2026))
    k = nm["keys"]["All|All"]
    assert len(nm["tgrid"]) == len(bp.TGRID) and set(k["time"]) == set(bp.TIME_METRICS)
    i0 = nm["tgrid"].index(0.0)
    assert abs(k["time"]["attack_angle"]["m"][i0] - k["contact"]["attack_angle"]["m"]) < 1.0    # at contact = measured
    assert k["time"]["attack_angle"]["m"][0] is None or k["time"]["attack_angle"]["n"][0] < k["time"]["attack_angle"]["n"][i0]
    df = _tracked(800).assign(vx0=5.0, vy0=-135.0, vz0=-5.0, ax=-8.0, ay=28.0, az=-15.0, plate_x=0.0, plate_z=2.5)
    df["p_throws"] = "R"
    poc = pd.DataFrame({"pf": df["intercept_ball_minus_batter_pos_y_inches"] - 27, "pxc": 0.0}, index=df.index)
    sw = bp.swings(df, poc, {"R": 26.0})
    st = bp.self_time(sw, "R", {"R": 26.0})
    assert st["comp"]["hba"]["m"][st["tgrid"].index(0.0)] is not None
    from ui.bat_path import chart_html
    scs = bp.group_scenes(sw, [dict(bp.DEFAULT_GROUP)], "R", lambda h, g: 27.0, 26.0, list(sw.ptype.unique()), nm)
    h = chart_html(scs, bp.series(sw, {"types": None, "hand": "All", "count": "All", "zone": None}, "R"), st)
    assert '"self_time"' in h and '"tgrid"' in h
