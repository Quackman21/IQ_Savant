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


def test_savant_framing_rejects_wrong_season(tmp_path, monkeypatch):
    from core import savant
    monkeypatch.setattr(savant, "COMPARE", tmp_path)
    monkeypatch.setattr("time.sleep", lambda s: None)
    ours = pd.Series(np.arange(1000, 7000, 200, dtype=float), index=range(100, 130))       # who caught that year
    right = pd.DataFrame({"id": range(100, 130), "name": "x", "pitches": ours.values / 2, "rv_tot": 1.0})
    wrong = pd.DataFrame({"id": range(500, 530), "name": "x", "pitches": ours.values, "rv_tot": 1.0})  # this season
    calls = []

    def fake(url, tries=3):
        calls.append(url)
        return (wrong, "30 rows") if "season=" not in url else (right, "30 rows")
    monkeypatch.setattr(savant, "_get", fake)
    f = savant.catcher_framing(2019, ours, log=lambda m: None)
    assert f["id"].min() == 100                                          # the ignored-year boards were thrown out
    assert (tmp_path / "savant_framing_2019.csv").exists() and "season=" in (tmp_path / "savant_framing_url.txt").read_text()
    calls.clear()
    assert len(savant.catcher_framing(2019, ours, log=lambda m: None)) == 30 and not calls        # cached
