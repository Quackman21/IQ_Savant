import numpy as np
import pandas as pd

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
    assert sum(a["counts"].values()) == len(a["rows"]) and 140 < a["commit_ms"] < 200      # his own commit point
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


def test_commit_point_is_his_swing_time_plus_the_decision():
    short = {"read": {"swing_length": 6.6, "bat_speed": 72.0}}
    long_ = {"read": {"swing_length": 8.0, "bat_speed": 72.0}}
    p = ss.CAL_DEFAULT
    assert ss.commit_s(long_, p) > ss.commit_s(short, p)          # a longer swing has to commit earlier
    assert 0.15 < ss.commit_s(short, p) < 0.19


def test_aim_reads_a_fastball_like_the_one_he_expects_exactly():
    from core.bat_path import synth_pitch
    fb = synth_pitch(*ss.FB_LOOK["R"], "R", 0.0, 2.5)
    d = synth_pitch(*ss.FB_LOOK["R"], "R", 0.3, 2.2)
    tc = ss._t_front(d, 0.0) - 0.17
    A, th = ss.aim(d, tc, 10.0, fb, 0.5)
    true = ss._traj(d, [ss._t_front(d, 10.0)])[0]
    assert np.linalg.norm(A[[0, 2]] - true[[0, 2]]) < 0.6           # same look -> he aims right at it
    assert abs((tc + th) - ss._t_front(d, 10.0)) < 0.003
    # a changeup with the same look at commit arrives later and lower than he expects
    ch = synth_pitch(85.0, 6.0, 14.0, "R", 0.3, 2.2)
    tcc = ss._t_front(ch, 0.0) - 0.17
    Ac, thc = ss.aim(ch, tcc, 10.0, fb, 0.5)
    truec = ss._traj(ch, [ss._t_front(ch, 10.0)])[0]
    assert Ac[2] > truec[2] and tcc + thc < ss._t_front(ch, 10.0)


def test_resolve_whiff_tip_glance_and_contact():
    K = len(ss.D_SHIFTS)

    def raw(d):
        return {"D": np.full(K, d), "lam": np.zeros(K), "a": np.tile([1.0, 0, 0], (K, 1)),
                "vb": np.tile([0.0, 70.0, 0], (K, 1)), "bv": np.tile([0.0, -90.0, -5.0], (K, 1)),
                "at": np.zeros((K, 3)), "t": np.zeros(K), "shifts": ss.D_SHIFTS.astype(float)}
    p = dict(ss.CAL_DEFAULT) | {"reach_max": 0.0, "reach_frac": 0.0, "r_tip": 2.4, "r_foul": 1.8}
    out = ss.resolve([raw(5.0), raw(2.5), raw(2.0), raw(0.4)], p, "R", 0.24)
    assert [o["result"] for o in out[:3]] == ["whiff", "whiff", "foul"]
    assert out[1].get("tip") and out[2].get("glance") and out[3]["result"] in ("fair", "foul") and "ev" in out[3]


def test_his_mix_and_rates_come_from_his_swings():
    sw = pd.DataFrame({"ptype": ["FF"] * 30 + ["SL"] * 20, "hand": "R", "velo": [95.0] * 30 + [86.0] * 20,
                       "ivb": [16.0] * 30 + [1.0] * 20, "hb": [8.0] * 30 + [-5.0] * 20,
                       "result": (["whiff"] * 6 + ["foul"] * 12 + ["fair"] * 12) + (["whiff"] * 10 + ["fair"] * 10)})
    m = ss.his_mix(sw, "R")
    assert abs(m["FF"][0] - 0.6) < 1e-9 and m["SL"][1] == 86.0 and m["SL"][3] == 1.0
    r = ss.his_rates(sw, "R")
    assert r["FF"]["whiff"] == 0.2 and r["SL"]["whiff"] == 0.5 and r["FF"]["n"] == 30
    assert ss.his_mix(sw.head(10), "R") is None


def test_the_painted_barrel_moves_through_the_swing():
    sw, scs = _scene()
    h = scs[0]["layers"]["heat"]
    R = np.array(h["runs"], dtype=float)
    assert R.shape == (len(h["t"]), len(h["lam"]), len(h["D"]))
    best = np.nanmax(R.reshape(len(h["t"]), -1), axis=1)
    k0 = int(np.argmin(np.abs(np.array(h["t"]))))
    assert np.nanmax(best) > 0.5 and best[k0] > np.nanmin(best)      # it lights up through the zone
    a = scs[0]["layers"]["arsenal"]
    assert 140 < a["commit_ms"] < 200 and {"act_whiff", "act_foul"} <= set(a["by_type"][0])


def test_calibration_outcome_matches_the_page_rule():
    from core import swing_cal as cal
    K = len(ss.D_SHIFTS)
    rng = np.random.default_rng(0)
    raws = []
    for d in rng.normal(0, 2.5, 40):
        raws.append({"D": d + 0.1 * ss.D_SHIFTS, "lam": np.zeros(K), "a": np.tile([1.0, 0, 0], (K, 1)),
                     "vb": np.tile([0.0, 70.0, 0], (K, 1)), "bv": np.tile([0.0, -90.0, -5.0], (K, 1)),
                     "at": np.zeros((K, 3)), "t": np.zeros(K), "shifts": ss.D_SHIFTS.astype(float)})
    p = dict(ss.CAL_DEFAULT)
    za, zt = rng.standard_normal(40), rng.standard_normal(40)
    page = [o["result"] for o in ss.resolve(raws, p, "R", 0.24, za, zt, 1.0)]
    vec = cal.outcome(cal._stack(raws), p, za, zt, np.ones(40), np.full(40, "R"), 0.24)["result"].tolist()
    assert page == vec
