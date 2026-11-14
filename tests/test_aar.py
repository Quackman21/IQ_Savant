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


def test_html_renders_every_piece():
    p = aar.pitches(game(), feed())
    pas = aar.plate_appearances(p)
    html = aar_ui.pa_html(pas, (1, 1))
    assert html.count('<details class="pa"') == 4 and " open" not in html.split("<details")[1][:20]
    assert aar_ui.pa_html(pas, (1, 1), open_all=True).count("<details class=\"pa\" open>") == 4
    assert "sporty-videos?playId=vid-3-1" in html
    assert html.count("<svg") == 11            # zone + spray in each PA; contact only where he made contact
    assert "No measured contact here." in html                # the strikeout PA
    no_contact = aar_ui.charts_html(p, None)
    assert "Point of contact" in no_contact and "Measured, but not enough" in no_contact
    assert "No measured contact" in aar_ui.charts_html(p.assign(cx=np.nan, cy=np.nan), (1, 1))


def test_decision_labels():
    assert aar.decision_label(0.4) == "Great" and aar.decision_label(0.1) == "Good"
    assert aar.decision_label(0.0) == "Neutral" and aar.decision_label(-0.1) == "Bad"
    assert aar.decision_label(-0.4) == "Poor" and aar.decision_label(float("nan")) == ""


def test_box_score_has_aar_links():
    from ui.game import box_html
    bat = [{"id": 1, "name": "Hitter One", "pos": "CF", "sub": False, "AB": 4, "R": 1, "H": 2, "RBI": 2, "BB": 0,
            "SO": 1, "HR": 1, "AVG": ".300"}]
    assert "/aar?pk=777&amp;pid=1" in box_html(bat, [], {"name": "X"}, 111, 777).replace("&pid", "&amp;pid")
    assert 'class="aar-link"' not in box_html(bat, [], {"name": "X"}, 111)
