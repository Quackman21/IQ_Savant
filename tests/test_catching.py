import numpy as np
import pandas as pd

from models import catching as cm
from pipeline import fetch_events


def test_parse_play_by_play_ties_events_to_pitches():
    pbp = {"allPlays": [{"about": {"atBatIndex": 4}, "playEvents": [
        {"isPitch": True, "pitchNumber": 1, "index": 0}, {"isPitch": True, "pitchNumber": 2, "index": 1},
        {"isPitch": False, "index": 2, "details": {"eventType": "wild_pitch"}},
        {"isPitch": True, "pitchNumber": 3, "index": 3}],
        "runners": [
            {"movement": {"start": "1B", "end": "2B", "isOut": False},
             "details": {"eventType": "wild_pitch", "playIndex": 2, "runner": {"id": 7}}},
            {"movement": {"start": "2B", "end": None, "isOut": True},
             "details": {"eventType": "caught_stealing_3b", "playIndex": 3, "runner": {"id": 7}}},
            {"movement": {"start": None, "end": "1B"}, "details": {"eventType": "single", "playIndex": 3}}]}]}
    rows = fetch_events.parse(pbp, 99)
    assert [(r["at_bat_number"], r["pitch_number"], r["event"]) for r in rows] == \
        [(5, 2, "wild_pitch"), (5, 3, "caught_stealing_3b")]


def _pitches(n=40000, seed=0):
    rng = np.random.default_rng(seed)
    pz = rng.normal(2.0, 1.1, n)
    return pd.DataFrame({"game_date": "2025-06-01", "game_pk": rng.integers(1, 400, n), "at_bat_number": rng.integers(1, 80, n),
                         "pitch_number": rng.integers(1, 7, n), "description": np.where(pz < 0.5, "blocked_ball", "ball"),
                         "type": "B", "plate_x": rng.normal(0, 0.9, n), "plate_z": pz, "sz_bot": 1.6,
                         "release_speed": 90.0, "pfx_x": 0.0, "pfx_z": 1.0, "pitch_type": "FF", "p_throws": "R",
                         "stand": "R", "balls": 1, "strikes": 1, "outs_when_up": 1, "on_1b": 1.0, "on_2b": np.nan,
                         "on_3b": np.nan, "fielder_2": rng.choice([11, 22], n)}).drop_duplicates(
        ["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def test_blocking_and_throwing_credit_the_better_catcher():
    from pipeline import train_catching as t
    df = _pitches()
    rng = np.random.default_rng(1)
    p_wp = np.where(df["plate_z"] < 0.5, 0.25, 0.01) * np.where(df["fielder_2"] == 11, 0.5, 1.5)
    wp = df[rng.random(len(df)) < p_wp]
    steal = df.sample(2000, random_state=2)
    cs = rng.random(len(steal)) < np.where(steal["fielder_2"] == 11, 0.4, 0.15)
    ev = pd.concat([wp.assign(event="wild_pitch"),
                    steal.assign(event=np.where(cs, "caught_stealing_2b", "stolen_base_2b"))])[
        ["game_pk", "at_bat_number", "pitch_number", "event"]]
    b = t.fit(df, ev)
    s = cm.summarize(cm.daily_rows(df, ev, b), "2025-01-01", "2025-12-31")
    assert s[11]["B_RS"] > 0 > s[22]["B_RS"] and s[11]["T_RS"] > 0 > s[22]["T_RS"]
    assert np.isfinite(s[11]["_B_DMID_rate"])


def test_wp_value_by_situation_and_zones_and_opps():
    re = {(0, 0): .48, (0, 1): .25, (0, 2): .10, (1, 0): .86, (1, 2): .21, (2, 0): 1.10, (2, 2): .31,
          (4, 0): 1.35, (4, 2): .36, (1, 1): .51, (2, 1): .65}
    v = cm.wp_values(re)
    assert round(v[(1, 0)], 2) == 0.24              # runner on 1st, 0 out -> 2nd
    assert round(v[(4, 2)], 2) == 0.74              # runner on 3rd, 2 out -> run scores
    z = cm.zone(pd.DataFrame({"plate_z": [2.0, 0.2, 0.2, 0.2], "plate_x": [0, -1.2, 0, 1.2]}))
    assert z.tolist() == ["air", "dirt_glove", "dirt_mid", "dirt_arm"]
    df = pd.DataFrame({"on_1b": [1.0, 1.0, np.nan, 1.0], "on_2b": np.nan, "on_3b": np.nan,
                       "type": ["B", "X", "B", "B"], "description": ["ball", "hit_into_play", "ball", "hit_by_pitch"],
                       "plate_x": 0.0, "plate_z": 1.0})
    assert cm.block_rows(df).tolist() == [True, False, False, False]


def test_run_expectancy_from_pitches():
    df = pd.DataFrame({"game_pk": 1, "inning": 1, "inning_topbot": "Top", "pitch_number": [1, 2, 1, 1],
                       "at_bat_number": [1, 1, 2, 3], "on_1b": [np.nan, np.nan, np.nan, 1.0], "on_2b": np.nan,
                       "on_3b": np.nan, "outs_when_up": [0, 0, 1, 1], "bat_score": [0, 0, 0, 0],
                       "post_bat_score": [0, 0, 0, 2]})
    re = cm.run_expectancy(df)
    assert re[(0, 0)] == 2 and re[(1, 1)] == 2


def test_parse_dropped_third_strike_wild_pitch():
    pbp = {"allPlays": [{"about": {"atBatIndex": 0}, "playEvents": [{"isPitch": True, "pitchNumber": 3}],
                         "runners": [{"movement": {"start": None, "end": "1B"},
                                      "details": {"eventType": "strikeout", "movementReason": "r_wild_pitch",
                                                  "playIndex": 0, "runner": {"id": 5}}}]}]}
    assert fetch_events.parse(pbp, 1)[0]["event"] == "wild_pitch"


def test_blocking_compare_lines_up(monkeypatch):
    from core import savant
    from pipeline import compare_fielding as cf
    ids = list(range(100, 112))
    o = pd.DataFrame({"player": ids, "pos": "C", "drs": 0.0, "inn": 900, "baa": np.arange(12.0) - 5,
                      "b_rs": (np.arange(12.0) - 5) * .3, "b_rs_sc": (np.arange(12.0) - 5) * .25, "opps": 500})
    sv = pd.DataFrame({"player_id": ids, "blocks_above_average": np.arange(12.0) - 5,
                       "blocks_above_average_runs": (np.arange(12.0) - 5) * .25})
    monkeypatch.setattr(savant, "catcher_blocking", lambda y: sv)
    monkeypatch.setattr("core.mlb_api.get_people", lambda ids: {})
    out = "\n".join(cf.blocking_compare(o, 2025))
    assert "r = 1.00  slope = 1.00  (12 catchers)" in out


def test_blocking_video_page(tmp_path, monkeypatch):
    from pipeline import blocking_videos as bv
    monkeypatch.setattr(bv, "HTML", tmp_path / "b.html")
    d = pd.DataFrame({"date": ["2025-05-01"], "game_pk": [1], "catcher": ["X"], "runners_outs": ["1st, 0 out"],
                      "pitch_type": ["SL"], "mph": [86.0], "height_in": [3.0], "zone": ["dirt_glove"],
                      "description": ["ball"], "wp_pb_chance": [0.22], "tier": ["tough"], "result": ["blocked"],
                      "wp_pb_cost": [0.24], "credit_runs": [0.053], "video": [""]})
    bv.write_page(d, "t")
    assert "22.0%" in (tmp_path / "b.html").read_text()


def test_block_table_empty_era_is_skipped():
    assert cm.block_table(pd.DataFrame(), None, {}).empty
