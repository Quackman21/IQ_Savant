import numpy as np

from core import collision as co, swing_sim as ss


def test_contact_layer_grid_and_its_best_cell_is_a_barrel():
    vb = co.ball_velocity([93.0], [-6.0], [0.0])[0]
    vB = co.bat_velocity([74.0], [10.0], [0.0], ["R"])[0]
    ax = co.axis_measured([10.0], [0.0], ["R"], [34.0])[0]
    cl = ss.contact_layer(vb, vB, ax, "R", q=0.24)
    R = np.array(cl["runs"])
    assert R.shape == (len(ss.EFF_GRID), len(ss.D_GRID))
    i, j = np.unravel_index(np.nanargmax(R), R.shape)
    assert cl["eff"][i] == 1.0 and 0.2 <= cl["D"][j] <= 1.3          # sweet spot, a bit under the ball
    assert cl["la"][i][j] > 10 and cl["ev"][i][j] > 95
    k0 = list(cl["D"]).index(-1.0)
    assert R[-1, k0] < R[i, j] and cl["la"][-1][k0] < 0               # topped: on the ground, worth less


def test_price_puts_barrels_above_outs():
    r = ss.price([105, 80, 105], [28, 25, -10], [0, 0, 0], ["R", "R", "R"])
    assert r[0] > r[1] and r[0] > r[2] and r[0] > 0.3


def _scene():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import pandas as pd
    from test_bat_path import _tracked
    from core import bat_path as bp
    df = _tracked(400)
    rng = np.random.default_rng(5)
    poc = pd.DataFrame({"pf": rng.normal(4, 7, len(df)), "pxc": 0.0}, index=df.index)
    df["plate_x"], df["plate_z"] = rng.normal(0, .5, len(df)), rng.normal(2.5, .5, len(df))
    sw = bp.swings(df, poc)
    return sw, bp.group_scenes(sw, [{"types": None}], "R", lambda h, g: 27.0, 26.0, None)


def test_first_group_carries_the_simulator_layers_and_they_serialize():
    import json
    sw, scs = _scene()
    L = scs[0]["layers"]
    assert L and {"contact", "timing", "arsenal", "pt_color"} <= set(L)
    json.dumps(L)                                                   # goes into the page as JSON
    rows = L["timing"]["rows"]
    assert rows[0]["shift"] < 0 < rows[-1]["shift"]
    a = L["arsenal"]
    assert sum(a["counts"].values()) == len(a["rows"]) and a["commit_ms"] == 175
    assert all(r["result"] in ("fair", "foul", "whiff") for r in a["rows"])


def test_the_page_has_the_simulator_tabs():
    from ui import bat_path as ub
    sw, scs = _scene()
    scs[0]["name"], scs[0]["color"] = "A", "#2f5fb3"
    h = ub.path3d_html(scs, "Bat path", "x")
    assert 'id="tabs"' in h and 'id="lay"' in h and "showArsenal" in h and '"layers"' in h


def test_price_grid_reads_the_model_back_and_is_cached(tmp_path, monkeypatch):
    import types
    calls = []

    def fake_model(ev, la, spray, stand):
        calls.append(len(ev))
        return ss._fallback_xwoba(ev, la) + 0.001 * np.asarray(spray)
    art = tmp_path / "m.joblib"
    art.write_text("x")
    import models
    fake = types.SimpleNamespace(ARTIFACT=art)
    monkeypatch.setitem(__import__("sys").modules, "models.xwoba_iq", fake)
    monkeypatch.setattr(models, "xwoba_iq", fake, raising=False)
    monkeypatch.setattr(ss, "_model_xwoba", fake_model)
    monkeypatch.setattr(ss, "_grid_file", lambda: tmp_path / "grid.npz")
    ss._grid.clear()
    r = ss.price([103.0, 70.0], [25.0, -5.0], [10.0, 0.0], ["R", "R"])
    want = ss._fallback_xwoba([103.0, 70.0], [25.0, -5.0]) / ss.WOBA_SCALE - ss.OUT_RUNS
    assert np.allclose(r, want, atol=0.06) and len(calls) == 2           # built once (R + L), then read
    ss._grid.clear()
    ss.price([100.0], [20.0], [0.0], ["L"])
    assert len(calls) == 2                                               # second time: from the saved file
    ss._grid.clear()
