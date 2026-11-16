import numpy as np
import pandas as pd

from models import pitch_models as pmod


def _league(n=24000, seed=3):
    rng = np.random.default_rng(seed)
    talent = rng.random(n)
    px, pz = rng.normal(0, 0.9, n), rng.normal(2.5, 0.9, n)
    inzone = (px.__abs__() < 0.83) & (pz > 1.6) & (pz < 3.4)
    pt = rng.choice(["FF", "SL"], n, p=[0.6, 0.4])
    swing = rng.random(n) < np.where(inzone, 0.65, 0.3)
    whiff = swing & (rng.random(n) < 0.15 + 0.25 * talent)
    bip = swing & ~whiff & (rng.random(n) < 0.35)
    desc = np.where(~swing, np.where(inzone, "called_strike", "ball"),
                    np.where(whiff, "swinging_strike", np.where(bip, "hit_into_play", "foul")))
    df = pd.DataFrame({"game_date": "2025-06-01", "game_pk": rng.integers(1, 400, n), "pitcher": rng.integers(1, 40, n),
                       "p_throws": "R", "stand": rng.choice(["L", "R"], n), "pitch_type": pt,
                       "release_speed": np.where(pt == "FF", 92 + 5 * talent, 84 + 3 * talent),
                       "pfx_x": np.where(pt == "FF", -0.7, 0.4), "pfx_z": np.where(pt == "FF", 1.3, 0.1),
                       "release_spin_rate": 2300, "release_extension": 6.3, "release_pos_x": -2, "release_pos_z": 5.8,
                       "arm_angle": 40.0, "plate_x": px, "plate_z": pz, "sz_top": 3.4, "sz_bot": 1.6,
                       "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n),
                       "type": np.where(bip, "X", np.where(desc == "ball", "B", "S")), "description": desc,
                       "estimated_woba_using_speedangle": np.where(bip, rng.random(n) * 0.8, np.nan)})
    df["delta_run_exp"] = np.select([desc == "ball", desc == "hit_into_play"], [0.05, 0.1 - 0.15 * talent], -0.05) \
        + rng.normal(0, 0.03, n)
    return df, talent


def test_train_score_and_grade(monkeypatch):
    from models import pitch_quality as pq
    from pipeline import train_pitch_models as t
    monkeypatch.setattr(t, "K_CLUSTERS", 4)
    monkeypatch.setattr(t, "MIN_GROUP", 100)
    df, talent = _league()
    f = pq.features(df)
    b = t.fit(df, f)
    sc = pmod._score_era(df, f, b)
    b["grades"] = t.grade_refs(df, f, sc)
    sw = df["description"].isin(["swinging_strike", "foul", "hit_into_play"])
    assert abs(sc["xsw"].mean() - sw.mean()) < 0.03
    take = ~sw
    assert sc.loc[take, "xbl"].mean() > 0.3
    assert np.corrcoef(sc.loc[sw, "xwf"], talent[sw])[0, 1] > 0.3     # better stuff -> more expected misses
    assert sc["cluster"].notna().all() and sc["cluster"].nunique() <= 4
    # grades: the best-stuff pitcher grades above the worst
    from core import pitch_metrics as pm
    d = df.assign(pitch=df["pitch_type"], take=take, swing=sw, **{c: sc[c] for c in pmod.COLS})
    d["game_date"] = pd.to_datetime(d["game_date"]).dt.date
    hi = d[(d.pitch == "FF") & (talent > 0.8)]
    lo = d[(d.pitch == "FF") & (talent < 0.2)]
    zh, zl = pmod.grade_z(hi, "XRESULTS", b), pmod.grade_z(lo, "XRESULTS", b)
    assert zh is not None and zh > zl
    assert set(b["grades"]) >= {"ARESULTS", "XRESULTS", "ACTL", "ACTR", "SHAPE", "LOCATION"}
    assert all(r["k"] > 0 for r in b["grades"].values())


def test_untrained_is_blank(monkeypatch, tmp_path):
    monkeypatch.setattr(pmod, "ARTIFACT", tmp_path / "none.joblib")
    pmod._cache.clear()
    df, _ = _league(200)
    out = pmod.score(df)
    assert out["xsw"].isna().all() and out["cluster"].isna().all()
    pmod._cache.clear()
