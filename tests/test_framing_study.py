import numpy as np
import pandas as pd

from pipeline import framing_study as fs


def test_zones_catchers_view():
    xn = pd.Series([0, 0, 0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 2.0])
    zn = pd.Series([0, 1.0, -1.0, 0, 0, 1.0, 1.0, -1.0, -1.0, 0])
    assert list(fs.zone(xn, zn)) == ["heart", "top", "bottom", "glove", "arm", "top-glove", "top-arm",
                                     "bottom-glove", "bottom-arm", "waste"]


def test_adjusted_splits_catcher_from_pitcher():
    rng = np.random.default_rng(1)
    n = 40000
    c, p = rng.integers(0, 2, n), rng.integers(0, 2, n)
    prob = np.full(n, 0.5) + np.where(c == 1, 0.05, -0.05) + np.where(p == 1, 0.1, -0.1)
    t = pd.DataFrame({"catcher": c, "pitcher": p, "ump": 0, "batter": rng.integers(0, 50, n),
                      "y": (rng.random(n) < prob).astype(int), "p": 0.5, "val": 0.125})
    a = fs.adjusted(t, 100)
    assert 0.07 < a[1] - a[0] < 0.13          # true gap 0.10, pitcher effect not leaked in


def test_sis_framing_file_matched_by_name(tmp_path, monkeypatch):
    from pipeline import compare_fielding as cf
    monkeypatch.setattr(cf, "COMPARE_DIR", tmp_path)
    monkeypatch.setattr("core.mlb_api.get_people", lambda ids: {1: "Patrick Bailey", 2: "José Treviño"})
    pd.DataFrame({"Player": ["Patrick Bailey", "Jose Trevino"], "Season": [2025, 2025],
                  "Strike Zone Runs Saved": [12, 7]}).to_csv(tmp_path / "sis_framing.csv", index=False)
    s = fs.sis_framing([1, 2], [2025])
    assert s[2025].to_dict() == {1: 12, 2: 7}
