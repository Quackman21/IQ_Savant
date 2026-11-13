import numpy as np
import pandas as pd

from models import fielding as fm
from pipeline import fielding_compare_study as fc


def _t(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    talent = {p: rng.normal() for p in range(240)}
    for y in (2024, 2025, 2026):
        for p in range(240):
            k = 3 + p % 7
            tl = talent[p]
            rows.append({"player": p, "pk": k, "pos": fc.POS[k], "year": y, "inn": 900,
                         "range": tl * .01 + rng.normal(0, .012), "drs_s": tl * 5 + rng.normal(0, 6),
                         "oaa_s": tl * 6 + rng.normal(0, 4), "sis_s": tl * 6 + rng.normal(0, 5),
                         "g_range_raw": 50 + tl * 3, "g_range_talent": 50 + tl * 5, "g_range_rank": 50 + tl * 8})
    return pd.DataFrame(rows)


def test_study_sections_run():
    t = _t()
    names = {p: f"P{p}" for p in range(240)}
    L = "\n".join(fc.agreement(t))
    pl, info = fc.predict(t)
    L += "\n".join(pl + fc.blend(info["m"], info["have"]) + fc.scales(t, names) + fc.disagreements(t, names))
    assert "best w =" in L and "70+:" in L and "we like more" in L and "SIS DRS" in L
    # the noisier measure (ours) shouldn't win the blend outright against a cleaner OAA
    assert "best w = 1.0" not in L


def test_scale_factor_orders():
    raw, tal, rnk = (fm.scale_factor(450, 594, s) for s in ("raw", "talent", "rank"))
    assert raw == 1 and rnk < tal < raw                      # smaller divisor -> wider grades
    spread = {"range": (0.0, 0.02, 450)}
    v = {"_N": 450, "_RNG": 9, "_CONV_N": 0}
    g = {s: fm.grades(v, spread, "IF", scale=s)["RANGE_GRADE"] for s in ("raw", "talent", "rank")}
    assert g["raw"] < g["talent"] < g["rank"]


def test_fetch_drs_saves_fangraphs_format(monkeypatch, tmp_path):
    import sys
    import types
    from pipeline import fetch_drs
    pb = types.ModuleType("pybaseball")
    monkeypatch.setitem(sys.modules, "pybaseball", pb)
    fake = pd.DataFrame({"Name": ["Nico Hoerner", "Bryson Stott"], "Team": ["CHC", "PHI"], "Pos": ["2B", "2B"],
                         "Inn": [1300.0, 1200.0], "DRS": [15, -8], "rPM": [10, -6], "rGFP": [2, -1], "Junk": [1, 2]})
    pb.fielding_stats = lambda *a, **k: fake
    monkeypatch.setattr(fetch_drs, "OUT", tmp_path)
    monkeypatch.setattr(fc, "COMPARE", tmp_path)
    fetch_drs.main(["--season", "2026"])
    s = fc.sis(2026)
    assert list(s["pos"]) == ["2B", "2B"] and s["sis"].tolist() == [15, -8]
    assert s["sis_range"].tolist() == [10, -6] and "sis_gfp" in s
