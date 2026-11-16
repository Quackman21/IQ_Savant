import json

import pandas as pd
import pytest

from core import weather as w


def test_air_density_physics():
    sea = w.air_density(59.0, 0.0, 1013.25)
    assert sea == pytest.approx(1.225, abs=0.002)                       # ISA sea level
    assert w.air_density(95.0, 50.0, 1013.25) < w.air_density(50.0, 50.0, 1013.25)    # hot air is thinner
    assert w.air_density(75.0, 90.0, 1013.25) < w.air_density(75.0, 10.0, 1013.25)    # humid air is thinner
    coors = w.air_density(75.0, 40.0, w.pressure_at(5190.0, 1013.25, 0.0))
    assert -20 < 100 * (coors / w.SEA_LEVEL_STD - 1) < -14              # Coors ~17% thinner


def test_wind_on_the_field():
    # park facing due north-east (CF bearing 45); wind FROM the south-west = blowing TOWARD 45 = straight out to CF
    assert w.field_angle((225 + 180) % 360, 45) == pytest.approx(0)
    assert w.along(10, 0, 0) == pytest.approx(10) and w.along(10, 0, 90) == pytest.approx(0, abs=1e-9)
    assert w.along(10, 180, 0) == pytest.approx(-10)                     # in from CF = headwind to CF
    assert w.parse_wind("7 mph, Out To CF") == (7.0, 0.0)
    assert w.parse_wind("12 mph, In From LF") == (12.0, 135.0)
    assert w.parse_wind("0 mph, None")[0] == 0.0 and w.parse_wind("Varies 4 mph") == (4.0, None)
    assert w.roof_closed("Roof Closed", "Retractable") and w.roof_closed("Dome", None) and not w.roof_closed("Sunny", "Open")


def test_conditions_from_the_season_table(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "DIR", tmp_path)
    monkeypatch.setattr(w, "PARKS", tmp_path / "parks.json")
    w._GAMES.clear(); w._HOURLY.clear()
    (tmp_path / "parks.json").write_text(json.dumps({"19": {"name": "Coors Field", "elev_ft": 5190, "cf_bearing": 0.0,
                                                           "roof": "Open", "lat": 39.76, "lon": -104.99}}))
    pd.DataFrame([{"game_pk": 1, "venue_id": 19, "start_utc": "2026-07-04T01:10:00Z", "rep_temp_f": 85,
                   "rep_condition": "Clear", "rep_wind": "10 mph, Out To CF", "rep_wind_mph": 10,
                   "rep_wind_field_deg": 0}]).to_csv(tmp_path / "games_2026.csv", index=False)
    (tmp_path / "2026").mkdir()
    pd.DataFrame({"time": pd.date_range("2026-07-04T00:00Z", periods=4, freq="h"), "temp_f": [88, 86, 82, 80],
                  "rh": [30, 32, 35, 38], "p_hpa": [840, 840, 841, 841], "wind_mph": [9, 10, 11, 12],
                  "wind_from": [180, 180, 180, 180], "grid_elev_ft": 5190}).to_csv(tmp_path / "2026" / "19.csv", index=False)
    c1 = w.conditions(1, 2026, inning=1)
    c9 = w.conditions(1, 2026, inning=9)
    assert c1["density_vs_std_pct"] < -14 and c1["wind_field_deg"] == 0 and c1["wind_out_cf_mph"] > 0
    assert c9["temp_f"] < c1["temp_f"]                                  # later innings get later hours
    assert "out to CF" in w.describe(c1) and "thinner" in w.describe(c1)
    w._GAMES.clear(); w._HOURLY.clear()


def test_dome_has_no_wind(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "DIR", tmp_path)
    monkeypatch.setattr(w, "PARKS", tmp_path / "parks.json")
    w._GAMES.clear(); w._HOURLY.clear()
    (tmp_path / "parks.json").write_text(json.dumps({"12": {"name": "Tropicana Field", "elev_ft": 44, "cf_bearing": 45,
                                                           "roof": "Dome"}}))
    feed = {"gameData": {"venue": {"id": 12}, "datetime": {"dateTime": "2026-10-29T00:08:00Z"},
                         "weather": {"condition": "Dome", "temp": "72", "wind": "0 mph, None"}}}
    c = w.conditions(999, 2026, feed=feed)
    assert c["roof_closed"] and c["wind_mph"] == 0 and c["temp_f"] == 72.0 and "roof closed" in w.describe(c)
    w._GAMES.clear(); w._HOURLY.clear()
