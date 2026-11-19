import json

import numpy as np
import pandas as pd
import pytest

from core import field_aar as fa
from ui import field_aar as ui


def _game(n=60, seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        la = float(rng.choice([rng.normal(-5, 8), rng.normal(15, 6), rng.normal(32, 8)], p=[.5, .2, .3]))
        ev, spray = float(np.clip(rng.normal(89, 12), 50, 115)), float(rng.uniform(-42, 42))
        if la < 10:
            d, loc = rng.uniform(100, 150), 5 if spray < -22 else 6 if spray < 0 else 4 if spray < 22 else 3
        else:
            d, loc = float(np.clip(150 + (la - 10) * 6 + (ev - 85) * 3, 150, 390)), 7 if spray < -15 else 8 if spray < 15 else 9
        out = rng.random() < 0.65
        evn = "field_out" if out else ("single" if la < 10 else rng.choice(["single", "double"]))
        rows.append({"game_pk": 1, "game_date": "2026-06-14", "at_bat_number": i + 1, "pitch_number": 2, "type": "X",
                     "description": "hit_into_play", "events": evn, "des": "x", "launch_speed": ev, "launch_angle": la,
                     "hit_distance_sc": d if la >= 10 else np.nan,
                     "hc_x": 125.42 + d / 2.5 * np.sin(np.radians(spray)), "hc_y": 198.27 - d / 2.5 * np.cos(np.radians(spray)),
                     "hit_location": float(loc), "bb_type": "ground_ball" if la < 10 else "fly_ball", "stand": "R",
                     "inning": i // 7 + 1, "inning_topbot": "Top", "outs_when_up": int(rng.integers(0, 3)),
                     "on_1b": 99.0 if rng.random() < .35 else np.nan, "on_2b": np.nan, "on_3b": np.nan,
                     "batter": 500 + i, "pitcher": 600, "delta_run_exp": 0.0, **{f"fielder_{k}": 10 + k for k in range(2, 10)}})
    return pd.DataFrame(rows)


def test_direction_names():
    assert fa.direction(-40) == "LF line" and fa.direction(0) == "CF" and fa.direction(25) == "RF" and fa.direction(np.nan) == ""


def test_game_credits_add_up_and_follow_the_rule():
    from models import fielding as fm
    if fm.load() is None:
        pytest.skip("fielding model not trained on this machine")
    g = _game()
    c = fa.credits(g, 18, "Outfield")                      # the CF (fielder_8 = 18)
    p = c["plays"]
    assert not p.empty and set(p["pos"]) == {"CF"}
    made = p[p["made"]]
    assert np.allclose(made["opa"], 1 - made["p_out"], atol=2e-3)                      # out: 1 - out chance
    hit = p[~p["made"]]
    assert np.allclose(hit["opa"], -hit["p_out"] * hit["share"], atol=6e-3)          # hit: -(chance x share)
    assert (hit["share"] >= fa.MIN_SHARE - 0.01).all()
    # the game's totals match the season file's per-play credits (every credit, listed or not)
    pc = fm.play_credits(g.reset_index(drop=True), fm.load())
    mine = pc[pc["player"].eq(18) & pc["pos"].isin([7, 8, 9])]
    ln = fa.line(c, "Outfield")
    assert abs(ln["opa"] - mine["opa"].sum()) < 1e-6 and abs(ln["runs"] - mine["runs"].sum()) < 1e-6
    assert abs(ln["total"] - (ln["runs"] + ln["arm_runs"])) < 1e-9
    inf = fa.credits(g, 16, "Infield")                     # the SS
    assert inf["plays"].empty or set(inf["plays"]["pos"]) <= {"SS"}


def test_page_renders_with_plays_and_without():
    plays = pd.DataFrame([{"n": 1, "inning": 3, "half": "▲", "outs": 1, "on": "1B", "batter": "A", "pitcher": "B",
                           "stand": "R", "ev": 98.0, "la": 24.0, "bb": "Fly ball", "dir": "LCF", "dist": 340.0,
                           "hx": -90.0, "hy": 320.0, "result": "Out (CF)", "des": "flies out", "pos": "CF",
                           "p_out": 0.31, "share": 1.0, "made": True, "opa": 0.69, "runs": 0.55, "rng": 0.69,
                           "conv": 0.0, "stars": 4, "kind": "Out", "video": "abc"}])
    c = {"plays": plays, "dp": pd.DataFrame(), "arm": pd.DataFrame(), "tot": {}}
    ln = fa.line(c, "Outfield")
    assert ln["best"]["p_out"] == 0.31 and ln["outs"] == 1
    html = ui.component_html(c, ln, "Outfield", header="<div id='hdr'></div>",
                             comps={"Outs vs expected": [["Season", 0.1], ["Last 30 days", 0.2], ["MLB avg", 0.0]]})
    data = json.loads(html.split('id="plays-data">')[1].split("</script>")[0])
    assert data[0]["p_out"] == 0.31 and "id='hdr'" in html and "Season" in html and 'id="fld"' in html
    empty = {"plays": pd.DataFrame(), "dp": pd.DataFrame(), "arm": pd.DataFrame(), "tot": {}}
    html2 = ui.component_html(empty, fa.line(empty, "Infield"), "Infield", has_model=False)
    assert "train_fielding" in html2 and "Double plays" in html2
