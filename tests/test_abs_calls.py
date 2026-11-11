import numpy as np
import pandas as pd

from core import abs_calls
from models import framing


def _pa():
    # one at-bat: 0-0 ball (challenged, overturned -> the strike STOOD), 0-1 called strike, 0-2 ball (upheld), 1-2 K
    return pd.DataFrame({"game_date": "2026-05-01", "game_pk": 9, "at_bat_number": 3, "pitch_number": [1, 2, 3, 4],
                         "balls": [0, 0, 0, 1], "strikes": [0, 1, 2, 2],
                         "description": ["ball", "called_strike", "ball", "called_strike"],
                         "events": [np.nan, np.nan, np.nan, "strikeout"]})


def test_call_that_stood_from_the_count():
    assert abs_calls.call_that_stood(_pa()).tolist() == ["S", "S", "B", "S"]


def test_original_call_restored(tmp_path, monkeypatch):
    monkeypatch.setattr(abs_calls, "RAW", tmp_path)
    pd.DataFrame({"game_pk": [9, 9], "at_bat_number": [3, 3], "pitch_number": [1, 3], "overturned": [True, False],
                  "challenger": [5, 6], "team": 1, "review_type": "ABS", "feed_call": None}).to_csv(
        tmp_path / "abs_2026.csv.gz", index=False)
    out = abs_calls.original_calls(_pa(), fetch=False)
    # pitch 1: Statcast says "ball" but the count shows a strike stood -> overturned -> original call was a BALL
    assert out["description"].tolist() == ["ball", "called_strike", "ball", "called_strike"]
    assert out["abs_overturned"].tolist() == [True, False, False, False]
    df = _pa().assign(description=["called_strike", "called_strike", "ball", "called_strike"])   # feed shows final
    assert abs_calls.original_calls(df, fetch=False)["description"].iloc[0] == "ball"
    old = _pa().assign(game_date="2025-05-01")
    assert abs_calls.original_calls(old, fetch=False)["description"].tolist() == old["description"].tolist()


def test_parse_marks_challenged_pitch():
    pbp = {"allPlays": [{"about": {"atBatIndex": 2}, "playEvents": [
        {"isPitch": True, "pitchNumber": 1, "details": {"call": {"code": "C"}, "hasReview": True},
         "reviewDetails": {"isOverturned": True, "reviewType": "MJ", "challengeTeamId": 111, "player": {"id": 5}}},
        {"isPitch": True, "pitchNumber": 2, "details": {"call": {"code": "B"}}}]}]}
    r = abs_calls.parse(pbp, 9)
    assert len(r) == 1 and r[0]["at_bat_number"] == 3 and r[0]["overturned"] and r[0]["challenger"] == 5


def test_center_league_to_zero():
    rng = np.random.default_rng(0)
    p = pd.Series(rng.uniform(0.05, 0.95, 5000))
    y = pd.Series((rng.random(5000) < (p + 0.03).clip(0, 1)).astype(int))
    c = framing.center(p, y, pd.Series(2025, index=p.index))
    assert abs((y - c).sum()) < 1 and abs((y - p).sum()) > 50
