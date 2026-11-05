from datetime import date

import pandas as pd

from core import hit_metrics as hm
from core import league_ref


def test_signature_and_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(league_ref, "DATA", tmp_path)
    from pipeline import update
    monkeypatch.setattr(update, "FIRST", 2024)
    ref = pd.DataFrame({"dims": ["all"], "vals": ["All"], "stat": ["EV"], "avg": [89.0], "sd": [2.0], "n": [1]})
    league_ref.save(ref, 2024, "2024-09-29")
    league_ref.save_catching(pd.DataFrame({"game_date": [], "catcher": []}), 2024)
    assert league_ref.status(2024) == "current" and league_ref.status(2025) == "missing"
    # offseason: only the missing seasons
    assert update.plan(date(2026, 11, 5)) == [(2026, "missing"), (2025, "missing")]
    # change a definition -> the saved season becomes stale
    sig = league_ref.signature()
    monkeypatch.setattr(hm, "SD_MIN_PA", 150)
    assert league_ref.signature() != sig and league_ref.status(2024) == "stale"


def test_in_season_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(league_ref, "DATA", tmp_path)
    from pipeline import update
    monkeypatch.setattr(update, "FIRST", 2026)
    ref = pd.DataFrame({"dims": ["all"], "vals": ["All"], "stat": ["EV"], "avg": [89.0], "sd": [2.0], "n": [1]})
    league_ref.save(ref, 2026)
    league_ref.save_catching(pd.DataFrame({"game_date": [], "catcher": []}), 2026)
    import json
    m = league_ref.meta(2026)
    m["built"] = "2026-06-30T23:00:00"                   # built last night
    (tmp_path / "hitting_2026.json").write_text(json.dumps(m))
    assert update.plan(date(2026, 7, 1)) == [(2026, "new games")]   # in season: add yesterday's games
    assert update.plan(date(2026, 6, 30)) == []                      # already built today
    assert update.plan(date(2026, 11, 5)) == []                      # offseason: nothing new
