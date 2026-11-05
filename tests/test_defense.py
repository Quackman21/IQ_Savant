import math

import pandas as pd

from core import defense as dfn
from ui.defense import table_html


def test_mlb_counts_and_positions():
    splits = [{"position": {"abbreviation": "3B"}, "team": {"id": 139},
               "stat": {"gamesPlayed": 20, "chances": 60, "putOuts": 15, "assists": 40, "errors": 5,
                        "doublePlays": 4, "innings": "170.2"}},
              {"position": {"abbreviation": "1B"}, "team": {"id": 139},
               "stat": {"gamesPlayed": 2, "chances": 18, "putOuts": 17, "assists": 1, "errors": 0, "innings": "16.0"}},
              {"position": {"abbreviation": "DH"}, "stat": {"gamesPlayed": 5}}]
    m = dfn.mlb_counts(splits, dfn.INFIELD)
    assert (m["N"], m["E"], m["G"], m["Pos"]) == (78, 5, 22, "1B, 3B")
    assert math.isclose(m["INN"], (170 * 3 + 2 + 48) / 3)
    m3 = dfn.mlb_counts(splits, ["3B"])
    assert m3["N"] == 60 and m3["Pos"] == "3B"


def test_savant_helpers():
    row = pd.Series({"player_id": 1, "rel_league_reaction_distance": 1.2, "n_opp_5stars": 9, "n_5star_percent": 22})
    assert dfn.col(row, "reaction") == 1.2 and dfn.col(row, "5star", "opp") == 9
    assert math.isnan(dfn.col(None, "x"))
    board = pd.DataFrame({"n_5star_percent": [0, 10, 20, 30]})
    assert dfn.pctile(board, 22, "5star", "percent") == 75


def test_framing_zones():
    cols = {f"strike_rate_{z}": [40, 50, 60] for z in (11, 12, 13, 14, 16, 17, 18, 19)}
    board = pd.DataFrame({"id": [1, 2, 3], **cols})
    z = dfn.framing_zones(board.iloc[2], board)
    assert z["F_TOZ"] > 60 and set(z) == {"F_TOZ", "F_BOZ", "F_GS", "F_AS"}


def test_table_blank_grades_and_season_rows():
    rows = [{"lead": {"Year": 2026, "Period": "Season"}, "vals": {"N": 1100, "OAA": 3}, "pct": {}, "season_row": True}]
    html = table_html(dfn.IF_GRADES_TAB, rows)
    assert 'class="season"' in html and "First Step Grade" in html and ">1,100<" in html
    assert "DRS" in html and "Out Probability Added by fielder speed" in html


def test_defense_tabs():
    assert list(dfn.TABS) == ["Infield", "Outfield", "Catching"]
    assert dfn.defense_tabs("P", {}) == []
