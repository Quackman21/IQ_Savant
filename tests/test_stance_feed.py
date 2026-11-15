import pandas as pd

from core import stance_feed

PAGE_HTML = """
<select id="ddlSeasonStart"><option value="2026">2026</option><option value="2025">2025</option></select>
<select id="ddlMinSwings"><option value="10">10</option><option value="1">1</option></select>
<select id="ddlLeaderboardType"><option value="season">Season</option><option value="month">Month</option>
  <option value="strikes">Strikes: 2 vs &lt; 2</option></select>
<select id="ddlPitchHand"><option value="">All</option><option value="L">Left</option><option value="R">Right</option></select>
<select id="ddlGameType"><option value="Regular">Regular</option><option value="Playoff">Playoff</option></select>
<select id="ddlIsHardHit"><option value="">Any</option><option value="1">Yes</option></select>
"""


class _Page:
    text = PAGE_HTML


class _Session:
    def get(self, url, timeout=60):
        return _Page()


def _fake_fetch(params, s=None, tries=3):
    """A fake Savant that understands minSwings, leaderboardType=month/strikes and pitchHand; ignores isHardHit."""
    rows = [{"id": 1, "name": "A", "side": "R", "avg_batter_y_position": 28.0},
            {"id": 2, "name": "B", "side": "L", "avg_batter_y_position": 31.0}]
    if params.get("minSwings") == "1":
        rows.append({"id": 3, "name": "Rookie", "side": "R", "avg_batter_y_position": 26.0})
    df = pd.DataFrame(rows)
    if params.get("pitchHand") in ("L", "R"):
        df["avg_batter_y_position"] += 0.5 if params["pitchHand"] == "L" else -0.5
    if params.get("group") == "month":
        df = pd.concat([df.assign(month=m) for m in (4, 5)], ignore_index=True)
    if params.get("gameType") == "Playoff":
        df = df.head(1).assign(avg_batter_y_position=27.0)
    return df


def test_discover_reads_the_dropdowns_and_keeps_what_changes_the_data(tmp_path, monkeypatch):
    monkeypatch.setattr(stance_feed, "fetch", _fake_fetch)
    monkeypatch.setattr(stance_feed, "_session", lambda: _Session())
    monkeypatch.setattr(stance_feed, "DIR", tmp_path / "stance")
    monkeypatch.setattr(stance_feed, "FEED", tmp_path / "stance" / "feed.json")
    monkeypatch.setattr(stance_feed.time, "sleep", lambda s: None)
    out = stance_feed.discover(2026, verbose=False)
    assert out["base"] == {"minSwings": "1"}
    assert "leaderboardType" not in out["tried"]                                     # Savant's site menu, skipped
    assert out["tried"]["group"][0]["value"] == "month" and out["tried"]["group"][0]["new_columns"] == ["month"]
    assert not any(t["changed"] for t in out["tried"]["isHardHit"])                # ignored -> not used
    wrote = stance_feed.refresh(2026, verbose=False)
    assert set(wrote) == {"season", "playoff", "vs_L", "vs_R", "group_month"}
    v = stance_feed.views(2026)
    assert v["vs_L"][(1, "R")] == 28.5 and v["vs_R"][(1, "R")] == 27.5 and v["playoff"] == {(1, "R"): 27.0}
    assert len(pd.read_csv(tmp_path / "stance" / "2026" / "season.csv")) == 3      # min 1 -> the rookie is on it
    assert (tmp_path / "batting_stance_2026.csv").exists()
    assert not stance_feed.stale(2026)
