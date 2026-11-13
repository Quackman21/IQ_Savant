import numpy as np
import pandas as pd
import pytest

from models import contact as cm


def _season(year, n_hitters=90, pitches=2300, seed=0):
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 0.5, n_hitters)
    skill[0] = 2.0                                               # hitter 1000: elite bat-to-ball
    h = np.repeat(np.arange(n_hitters), pitches)
    n = len(h)
    px, pz = rng.normal(0, 0.9, n), rng.normal(2.5, 0.9, n)
    r = np.maximum(np.abs(px) / (17 / 24), np.abs(pz - 2.5) / 0.9)
    swing = rng.random(n) < 1 / (1 + np.exp(-(1.5 - 2.2 * r)))
    contact = swing & (rng.random(n) < 1 / (1 + np.exp(-(2.6 - 1.8 * r + skill[h]))))
    bip = contact & (rng.random(n) < .45)
    desc = np.where(~swing, np.where(r < 1, "called_strike", "ball"),
                    np.where(~contact, "swinging_strike", np.where(bip, "hit_into_play", "foul")))
    gp = rng.integers(0, 2430, n)
    return pd.DataFrame({"game_pk": gp + year * 10000, "game_date": f"{year}-06-01",
                         "game_type": "R", "batter": h + 1000, "description": desc,
                         "type": np.where(bip, "X", "S"), "zone": np.where(r < 1, 5, 13), "plate_x": px, "plate_z": pz,
                         "sz_top": 3.4, "sz_bot": 1.6, "pitch_type": rng.choice(["FF", "SL", "CH"], n),
                         "release_speed": rng.normal(92, 4, n), "pfx_x": rng.normal(0, .8, n),
                         "pfx_z": rng.normal(1, .5, n), "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n),
                         "stand": "R", "p_throws": rng.choice(["R", "L"], n),
                         "events": np.where(rng.random(n) < 1 / 3.9, "field_out", None),
                         "home_team": np.array([f"T{g % 30}" for g in gp]),
                         "away_team": np.array([f"T{(g + 15) % 30}" for g in gp])})


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from pipeline import build_league_reference as bl
    from pipeline import train_contact as tc
    d = tmp_path_factory.mktemp("contact")
    old = (cm.ARTIFACT, bl.pull_season)
    cm.ARTIFACT = d / "contact.joblib"
    seasons = {2025: _season(2025), 2026: _season(2026, seed=1)}
    bl.pull_season = lambda y: seasons[y]
    cm._cache.clear()
    tc.train([2025, 2026])
    ref = cm.build_ref(seasons[2026], 2026)
    yield seasons[2026], ref
    cm.ARTIFACT, bl.pull_season = old
    cm._cache.clear()


def test_elite_contact_grades_high_and_regresses(trained):
    s, ref = trained
    assert "sd_blend" in ref and ref["league_contact_by_zone"]["Heart"] > ref["league_contact_by_zone"]["Chase"]
    star = cm.score(s[s.batter == 1000], ref)
    assert 50 + 10 * star["_B2B_Z"] > 65 and "Cont% over exp" in star["_B2B_PARTS"]
    few = cm.score(s[s.batter == 1000].head(40), ref)
    assert abs(few["_B2B_Z"]) < abs(star["_B2B_Z"])            # small sample pulled toward average


def test_takes_never_count(trained):
    s, ref = trained
    g = s[s.batter == 1005]
    takes = g[g.description.isin(["ball", "called_strike"])]
    more_takes = pd.concat([g, takes.assign(game_pk=takes.game_pk + 1)])  # add more taken pitches
    assert cm.score(g, ref)["_B2B_Z"] == pytest.approx(cm.score(more_takes, ref)["_B2B_Z"])


def test_heart_whiff_costs_more_than_chase_whiff(trained):
    s, _ = trained
    heart = s.iloc[[0]].assign(plate_x=0.0, plate_z=2.5, zone=5, description="swinging_strike")
    chase = s.iloc[[0]].assign(plate_x=1.6, plate_z=0.9, zone=13, description="swinging_strike")
    xh, xc = cm.expected(heart).iloc[0], cm.expected(chase).iloc[0]
    assert (0 - xh) < (0 - xc)                                   # missing a cookie is the bigger penalty
