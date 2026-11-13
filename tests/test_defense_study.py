import pandas as pd

from pipeline import defense_study as ds


def test_for70_spearman_brown():
    # r = .5 at 100 chances a half -> k = 100 -> .70 at 233
    assert round(ds.for70(0.5, 100)) == 233
    assert ds.for70(0, 100) != ds.for70(0, 100)          # NaN when nothing repeats


def test_framing_section_compares_count_and_situational(monkeypatch):
    from core import league_ref
    def fake(y):
        rows = []
        for c in range(30):
            rows.append({"catcher": c, "n": 3000, "strikes": 1500 + c * 3 + (y % 2), "exp": 1500,
                         "runs": c * 0.4, "runs_sit": c * 0.45 + (y % 3) * 0.1, "game_date": f"{y}-05-01"})
        return pd.DataFrame(rows)
    monkeypatch.setattr(league_ref, "load_catching", fake)
    L = "\n".join(ds.framing_section([2023, 2024, 2025]))
    assert "r(count runs, situational runs)" in L and "best predicts NEXT season" in L
