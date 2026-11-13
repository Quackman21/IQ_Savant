import numpy as np
import pandas as pd

from models import flight_grade as fg


def _season(year=2026, n_hitters=80, pa=760, seed=0):
    """Each hitter gets an air tendency + pull tendency; hitter 1000 is an extreme air-pull hitter."""
    rng = np.random.default_rng(seed)
    air = rng.normal(12, 4, n_hitters)
    pull = rng.normal(0, 8, n_hitters)
    air[0], pull[0] = 24, -22                                   # 1000: lifts and pulls everything (RHB pull = negative)
    air[1], pull[1] = 0, 15                                     # 1001: ground balls the other way
    h = np.repeat(np.arange(n_hitters), pa)
    n = len(h)
    bip = rng.random(n) < .7
    la = rng.normal(air[h], 24 - 6 * (h == 0))
    spray = rng.normal(pull[h], 25)
    # spray_angle = deg(atan((x - 125.42) / (198.27 - y))) * scale  ->  build hc_x / hc_y from the angle
    from core import hit_metrics as hm
    r = 150
    ang = np.radians(spray / getattr(hm, "SPRAY_SCALE", 1.0))
    gp = rng.integers(0, 2430, n)
    return pd.DataFrame({"game_pk": gp + year * 10000, "game_date": f"{year}-06-01", "game_type": "R",
                         "batter": h + 1000, "type": np.where(bip, "X", "S"), "stand": "R",
                         "description": np.where(bip, "hit_into_play", "ball"),
                         "events": np.where(bip, "field_out", np.where(rng.random(n) < .3, "strikeout", None)),
                         "launch_angle": np.where(bip, la, np.nan), "launch_speed": np.where(bip, 90.0, np.nan),
                         "hc_x": 125.42 + r * np.sin(ang), "hc_y": 198.27 - r * np.cos(ang),
                         "home_team": np.array([f"T{g % 30}" for g in gp]),
                         "away_team": np.array([f"T{(g + 15) % 30}" for g in gp])})


def test_air_pull_hitter_grades_high_and_regresses():
    s = _season()
    ref = fg.build_ref(s, 2026)
    assert "sd_blend" in ref and ref["n_qualified"] >= 30
    star = fg.score(s[s.batter == 1000], ref)
    worm = fg.score(s[s.batter == 1001], ref)
    assert 50 + 10 * star["_BF_Z"] > 65 and 50 + 10 * worm["_BF_Z"] < 40
    assert "AirPull%" in star["_BF_PARTS"] and "BIP" in star["_BF_PARTS"]
    few = fg.score(s[s.batter == 1000].head(60), ref)
    assert abs(few["_BF_Z"]) < abs(star["_BF_Z"])              # small sample pulled toward average


def test_exit_velo_never_counts():
    s = _season()
    ref = fg.build_ref(s, 2026)
    one = s[s.batter == 1005]
    a = fg.score(one, ref)["_BF_Z"]
    b = fg.score(one.assign(launch_speed=one.launch_speed + 15), ref)["_BF_Z"]
    assert a == b


def test_bunts_and_tiny_samples_blank():
    s = _season()
    ref = fg.build_ref(s, 2026)
    tiny = s[(s.batter == 1003) & s.type.eq("X")].head(10)
    assert np.isnan(fg.score(tiny, ref)["_BF_Z"])
    bunts = s[(s.batter == 1003) & s.type.eq("X")].head(200).assign(description="bunt_foul_tip", events="sac_bunt")
    assert np.isnan(fg.score(bunts, ref)["_BF_Z"])
    assert np.isnan(fg.score(s[s.batter == 1003], None)["_BF_Z"])
