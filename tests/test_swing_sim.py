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
