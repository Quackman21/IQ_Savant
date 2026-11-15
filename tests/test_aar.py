import numpy as np
import pandas as pd

from core import aar
from ui import aar as aar_ui


def game(seed=1):
    """A fake 4-PA game for one right-handed hitter (Statcast column names)."""
    rng = np.random.default_rng(seed)
    pas = [  # (pitches as (desc, events, type), )
        [("ball", None, "B"), ("called_strike", None, "S"), ("hit_into_play", "field_out", "X")],
        [("ball", None, "B"), ("ball", None, "B"), ("called_strike", None, "S"), ("swinging_strike", None, "S"),
         ("swinging_strike", "strikeout", "S")],
        [("hit_into_play", "single", "X")],
        [("foul", None, "S"), ("ball", None, "B"), ("hit_into_play", "home_run", "X")],
    ]
    rows = []
    for ab, pa in enumerate(pas, start=3):
        b = s = 0
        for pn, (desc, ev, typ) in enumerate(pa, 1):
            bip = typ == "X"
            rows.append({"game_pk": 777, "game_date": "2026-08-22", "game_type": "R", "batter": 1, "pitcher": 50 + ab,
                         "at_bat_number": ab * 9, "pitch_number": pn, "inning": ab - 2, "inning_topbot": "Bot",
                         "balls": b, "strikes": s, "description": desc, "events": ev, "type": typ,
                         "stand": "R", "p_throws": "R" if ab % 2 else "L",
                         "pitch_type": ["FF", "SL", "CH", "SI"][pn % 4], "release_speed": 85 + pn,
                         "pfx_x": -0.8, "pfx_z": 1.2, "plate_x": rng.uniform(-1.2, 1.2),
                         "plate_z": rng.uniform(1.2, 3.8), "sz_top": 3.4, "sz_bot": 1.6, "zone": 5,
                         "bb_type": "line_drive" if bip else None, "launch_speed": 101.5 if bip else None,
                         "launch_angle": 18.0 if bip else None, "hit_distance_sc": 300 if bip else None,
                         "hc_x": 100.0 if bip else None, "hc_y": 90.0 if bip else None,
                         "bat_speed": 74.0 if desc in ("foul", "hit_into_play", "swinging_strike") else None,
                         "intercept_ball_minus_batter_pos_x_inches": 20.0 if desc in ("foul", "hit_into_play") else None,
                         "intercept_ball_minus_batter_pos_y_inches": 25.0 if desc in ("foul", "hit_into_play") else None})
            b, s = b + (desc == "ball"), min(2, s + (desc in ("called_strike", "swinging_strike", "foul")))
    return pd.DataFrame(rows)


def feed():
    plays = []
    for ab in range(3, 7):
        plays.append({"about": {"atBatIndex": ab * 9 - 1},
                      "playEvents": [{"isPitch": True, "pitchNumber": n, "playId": f"vid-{ab}-{n}"} for n in range(1, 6)]})
    return {"liveData": {"plays": {"allPlays": plays},
                         "boxscore": {"teams": {"home": {"players": {"ID1": {"stats": {"batting": {
                             "atBats": 4, "runs": 1, "hits": 2, "doubles": 0, "triples": 0, "homeRuns": 1, "rbi": 2,
                             "baseOnBalls": 0, "strikeOuts": 1}}}}}, "away": {"players": {}}}}},
            "gameData": {"players": {f"ID{50 + ab}": {"id": 50 + ab, "fullName": f"Pitcher {ab}"} for ab in range(3, 7)},
                         "teams": {"home": {"id": 111}, "away": {"id": 147}}}}


def test_pitches_and_plate_appearances():
    p = aar.pitches(game(), feed())
    assert len(p) == 12 and p["n"].tolist() == list(range(1, 13))
    assert p.loc[0, "video"] == "vid-3-1" and p.loc[0, "pitcher"] == "Pitcher 3"
    assert p.loc[0, "hb"] > 0                                  # arm side + for a RHP's -pfx_x
    assert p.loc[0, "inning"] == 1 and p.loc[0, "half"] == "▼"
    pas = aar.plate_appearances(p)
    assert [x["n_pitches"] for x in pas] == [3, 5, 1, 3]
    assert pas[0]["result"] == "Out" and "LD" in pas[0]["extra"] and "101.5 mph" in pas[0]["extra"]
    assert pas[1]["result"] == "Strikeout" and pas[3]["result"] == "Home run"


def test_box_line_and_contact_orientation():
    line = aar.box_line(feed(), 1)
    assert line["H"] == 2 and line["HR"] == 1 and line["K"] == 1 and line["side"] == "home"
    assert aar.box_line(feed(), 999) == {}
    # season where pulled balls are met farther out front and outside pitches farther from the body
    rng = np.random.default_rng(0)
    n = 200
    px = rng.uniform(-1, 1, n)
    spray_pull = rng.uniform(-40, 40, n)                       # degrees toward pull (RHB pull = left = -x)
    hcx = 125.42 - 100 * np.sin(np.radians(spray_pull / 0.75))
    s = pd.DataFrame({"stand": "R", "plate_x": px, "hc_x": hcx, "hc_y": 100.0,
                      aar.CONTACT_Y: 20 + spray_pull * .3 + rng.normal(0, 2, n),
                      aar.CONTACT_X: 18 + px * 6 + rng.normal(0, 1, n)})
    assert aar.contact_orientation(s) == (1, 1)
    s[aar.CONTACT_Y] *= -1
    assert aar.contact_orientation(s)[0] == -1
    assert aar.contact_orientation(s.head(5)) is None


def test_component_renders_every_piece():
    p = aar.pitches(game(), feed())
    p = p.assign(pf=[5.0 if c == c else float("nan") for c in p["cx"]],
                 pxc=[1.0 if c == c else float("nan") for c in p["cx"]])
    pas = aar.plate_appearances(p)
    tiles = {k: [("Last 30 PA", "1"), ("Season", "2"), ("MLB season", "3")] for k in aar.GAME_METRICS}
    html = aar_ui.component_html(p, pas, aar.box_line(feed(), 1), tiles, aar.GAME_METRICS, "No measured contact.",
                                 {"leftLine": 315, "center": 404, "rightLine": 322}, "Fenway Park")
    assert html.count('<tr class="pa-h"') == 4                         # one table, one header row per PA
    assert html.count("<table class=\"p\">") == 1
    assert html.count('<tr class="row') == 12 and "tr.row{display:none" in html     # every PA starts collapsed
    assert html.count('<g class="pt"') == 12 + 3 + 4      # zone 12, spray 3 balls in play, contact 4 (fouls + in play)
    assert 'data-n="1" data-pa="1"' in html and "sporty-videos?playId=vid-3-1" in html
    assert "<svg viewBox=\"0 0 24 24\"" in html                        # camera is an icon, not an emoji
    assert "Last 30 PA" in html and '<option value="GD%" selected>' in html      # Good decision % by default
    assert "allPA.forEach(p => open.add(p))" in html                           # every PA open to start
    assert html.count('class="chip-f on"') == len(set(p["ptype"])) and 'data-h="L"' in html   # filters
    assert ">404<" in html and "Fenway Park" in html                           # this park's fence


def test_decision_labels():
    assert aar.decision_label(0.4) == "Great" and aar.decision_label(0.1) == "Good"
    assert aar.decision_label(0.0) == "Neutral" and aar.decision_label(-0.1) == "Bad"
    assert aar.decision_label(-0.4) == "Poor" and aar.decision_label(float("nan")) == ""


def _season_swings(n=300, X0=-30.0, F0=-28.0, sx=1, seed=4):
    """A RHB whose hips sit 30" off the middle of the plate and 28" behind its front edge (catcher's view)."""
    rng = np.random.default_rng(seed)
    slope = rng.normal(0, 0.06, n)
    vy0, ay = -130.0, 28.0
    a, b, c = 0.5 * ay, vy0, 50 - aar.FRONT_Y
    t = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    vx0 = slope * (vy0 + ay * t)                                  # ax = 0 -> this slope at the plate
    px = rng.uniform(-1, 1, n)                                    # ft, where the ball crosses the front
    cy = rng.normal(31, 5, n)                                     # contact, inches out front of his hips
    f = F0 + cy                                                   # ... = inches in front of the plate
    x_ball = px * 12 + slope * f                                  # where the ball is when he hits it
    cx = sx * (x_ball - X0) + rng.normal(0, 0.8, n)
    return pd.DataFrame({"stand": "R", "plate_x": px, "vx0": vx0, "vy0": vy0, "ax": 0.0, "ay": ay,
                         aar.CONTACT_X: cx, aar.CONTACT_Y: cy})


def test_plate_calibration_learns_where_he_stands_from_his_swings():
    season = _season_swings()
    cal = aar.plate_calibration(season, (1, 1))
    r = cal["R"]
    assert abs(r["F0"] + 28) < 2.5 and abs(r["X0"] + 30) < 1.5 and r["sx"] == 1 and r["n"] > 250
    flipped = aar.plate_calibration(_season_swings(sx=-1), (1, 1))["R"]       # other axis direction -> found
    assert flipped["sx"] == -1 and abs(flipped["F0"] + 28) < 2.5
    assert aar.plate_calibration(season.head(20), (1, 1)) is None              # too few swings
    # place one swing: 3" out front of the plate, ball over the middle
    one = pd.DataFrame({"stand": ["R"], "cx": [30.0], "cy": [31.0], "px": [0.0], "slope": [0.0], "pa": [1]})
    on = aar.on_plate(one, cal)
    assert abs(on["pf"].iloc[0] - (r["F0"] + 31)) < 1e-9 and abs(on["pxc"].iloc[0]) < 1e-9
    assert len(one.join(on).columns) == 8


def test_field_dims_and_last_30_pa():
    fd = {"gameData": {"venue": {"fieldInfo": {"leftLine": 315, "leftCenter": 370, "center": 404,
                                                "rightCenter": 370, "rightLine": 322, "capacity": 25000}}}}
    assert aar.field_dims(fd) == {"leftLine": 315.0, "leftCenter": 370.0, "center": 404.0, "rightCenter": 370.0,
                                  "rightLine": 322.0}
    assert aar.field_dims({}) == {}
    g = game()
    many = pd.concat([g.assign(game_pk=k, game_date=f"2026-08-{k:02d}") for k in range(1, 11)])
    last = aar.last_pas(many, 30)
    assert last.groupby(["game_pk", "at_bat_number"]).ngroups == 30
    assert last["game_pk"].min() >= 3


def test_box_score_has_aar_links():
    from ui.game import box_html
    bat = [{"id": 1, "name": "Hitter One", "pos": "CF", "sub": False, "AB": 4, "R": 1, "H": 2, "RBI": 2, "BB": 0,
            "SO": 1, "HR": 1, "AVG": ".300"}]
    html = box_html(bat, [], {"name": "X"}, 111, 777)
    assert "/aar?pk=777&pid=1" in html and 'class="gut"' in html and "\U0001f4cb" not in html
    assert 'class="aar-link"' not in box_html(bat, [], {"name": "X"}, 111)
    from ui.game import lineup_html
    lu = {"nine": [{"spot": 1, "id": 5, "name": "A", "pos": "CF", "bats": "R"}], "pitcher": None}
    assert "/aar?pk=777&pid=5" in lineup_html(lu, {"name": "X"}, 111, 777)


def _outline(team="red_sox", lf=310, cf=420, rf=302):
    """A made-up traced park in Statcast hit coordinates: fence from the LF pole to the RF pole + dirt."""
    angs = np.radians(np.linspace(-45, 45, 31))
    dist = np.interp(np.degrees(angs), [-45, -20, 0, 20, 45], [lf, 380, cf, 380, rf])
    fx, fy = dist * np.sin(angs), dist * np.cos(angs)
    rows = [{"team": team, "x": 125.42 + x / 2.5, "y": 198.27 - y / 2.5, "segment": "outfield_outer"}
            for x, y in zip(fx, fy)]
    a2 = np.radians(np.linspace(-45, 45, 9))
    rows += [{"team": team, "x": 125.42 + 95 * np.sin(a) / 2.5, "y": 198.27 - 95 * np.cos(a) / 2.5,
              "segment": "infield_outer"} for a in a2]
    return pd.DataFrame(rows)


def test_outline_slid_onto_statcast_home_plate():
    o = _outline(lf=310, cf=390, rf=302)
    o["y"] = o["y"] + 7.2                                             # this park's file sits 18 ft off
    hp = pd.DataFrame({"team": "red_sox", "x": [125.42 - 3, 125.42 + 3, 125.42], "y": [198.27 + 7.2 - 1,
                       198.27 + 7.2 - 1, 198.27 + 7.2], "segment": "home_plate"})
    got = aar.park_outline("Boston Red Sox", {"leftLine": 310.0, "center": 420.0, "rightLine": 302.0},
                           pd.concat([o, hp]))
    assert got["source"] == "outline", got["note"]                    # 420 = deepest point, not checked at center
    assert abs(got["poles"][0] - 310) < 3 and abs(got["poles"][1] - 302) < 3


def test_fence_path_that_runs_down_the_foul_lines():
    """The real file's fence path goes home -> LF line -> wall -> RF line -> home."""
    o = _outline(lf=310, cf=390, rf=302)
    fence = o[o.segment.eq("outfield_outer")]
    line = lambda ang, far: pd.DataFrame({"team": "red_sox", "segment": "outfield_outer",
                                          "x": 125.42 + np.linspace(5, far, 25) * np.sin(np.radians(ang)) / 2.5,
                                          "y": 198.27 - np.linspace(5, far, 25) * np.cos(np.radians(ang)) / 2.5})
    near_home = pd.DataFrame({"team": "red_sox", "segment": "outfield_outer",       # points by the plate land at
                              "x": 125.42 + np.array([-20, 22]) / 2.5, "y": 198.27 - np.array([12, 11]) / 2.5})  # odd angles
    path = pd.concat([near_home.iloc[:1], line(-45, 309), fence, line(45, 301).iloc[::-1], near_home.iloc[1:],
                      o[o.segment.ne("outfield_outer")]])
    got = aar.park_outline("Boston Red Sox", {"leftLine": 310.0, "center": 420.0, "rightLine": 302.0}, path)
    assert got["source"] == "outline", got["note"]
    assert abs(got["poles"][0] - 310) < 3 and abs(got["poles"][1] - 302) < 3
    fx, fy = got["segments"]["outfield_outer"]
    assert np.hypot(fx, fy).min() > 290                                # no foul-line points left in the wall


def test_park_outline_used_only_when_it_matches_this_venue():
    o = _outline()
    dims = {"leftLine": 310.0, "center": 420.0, "rightLine": 302.0}
    got = aar.park_outline("Boston Red Sox", dims, o)
    assert got["source"] == "outline" and "outfield_outer" in got["segments"]
    moved = aar.park_outline("Boston Red Sox", {"leftLine": 330.0, "center": 400.0, "rightLine": 325.0}, o)
    assert moved["source"] == "distances" and "doesn't match" in moved["note"]
    assert aar.park_outline("Athletics", dims, o)["source"] == "distances"           # no outline on file
    assert abs(aar.fence_distance(*got["segments"]["outfield_outer"], 0) - 420) < 1


def test_spray_draws_real_outline_or_labeled_fallback():
    p = aar.pitches(game(), feed())
    dims = {"leftLine": 310.0, "center": 420.0, "rightLine": 302.0}
    html = aar_ui.spray_svg(p, dims, "Fenway Park", aar.park_outline("Boston Red Sox", dims, _outline()))
    assert "approximate" not in html and ">420<" in html and html.count("<polygon") >= 2
    fb = aar_ui.spray_svg(p, dims, "Some Park", aar.park_outline("Athletics", dims, _outline()))
    assert "approximate fence" in fb


def test_contact_only_fouls_and_balls_in_play():
    g = game()
    g[aar.CONTACT_X], g[aar.CONTACT_Y] = 30.0, 31.0                  # Statcast gives whiffs a point too
    p = aar.pitches(g, feed())
    whiff = p["result"].eq("Swinging strike")
    assert whiff.any() and p.loc[whiff, "cx"].isna().all()
    assert p.loc[p["result"].isin(["Foul", "In play"]), "cx"].notna().all()
    assert p.loc[p["result"].isin(["Ball", "Called strike"]), "cx"].isna().all()


def test_scrambled_outline_is_drawn_in_order():
    o = _outline().sample(frac=1, random_state=1)                    # file rows out of order
    got = aar.park_outline("Boston Red Sox", {"leftLine": 310.0, "center": 420.0, "rightLine": 302.0}, o)
    fx, fy = aar_ui._by_spray(*got["segments"]["outfield_outer"])
    ang = np.degrees(np.arctan2(fx, fy))
    assert (np.diff(ang) > 0).all() and ang[0] < -40 and ang[-1] > 40
