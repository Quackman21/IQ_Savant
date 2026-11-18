import numpy as np

from core import collision as co


def _one(D, aa=0.0, ad=0.0, vaa=0.0, haa=0.0, bs=72.0, pitch=90 / co.PLATE):
    vb = co.ball_velocity([pitch], [vaa], [haa])
    vB = co.bat_velocity([bs], [aa], [ad], ["R"])
    axis = np.array([[-1.0, 0, 0]])                                   # level bat, hands toward 3B (righty)
    return co.collide(vb, vB, axis, [D])


def test_square_hit_matches_nathans_formula():
    r = _one(0.0)
    assert abs(r["ev"][0] - (0.2 * 90 + 1.2 * 72)) < 0.05 and abs(r["la"][0]) < 1e-6 and abs(r["spray"][0]) < 1e-6


def test_undercut_lifts_spins_and_costs_speed():
    r0, r5, r10 = _one(0.0), _one(0.5), _one(1.0)
    assert r0["la"][0] < r5["la"][0] < r10["la"][0]                    # higher with more undercut
    assert r10["ev"][0] < r5["ev"][0] < r0["ev"][0]                    # slower
    assert r10["spin"][0] > r5["spin"][0] > 0                          # backspin
    over = _one(-0.5)
    assert over["la"][0] < 0 and over["spin"][0] < 0                   # topped: down + topspin


def test_attack_angle_and_pitch_angle_set_the_square_launch():
    up = _one(0.0, aa=10.0, vaa=-6.0)
    assert 0 < up["la"][0] < 10                                       # between the pitch's angle and the bat's


def test_offset_solver_inverts_launch_angle():
    vb = co.ball_velocity([95.0] * 3, [-6.0] * 3, [0.0] * 3)
    vB = co.bat_velocity([72.0] * 3, [8.0] * 3, [0.0] * 3, ["R"] * 3)
    ax = np.array([[-1.0, 0, 0.3]] * 3)
    D = np.array([-0.3, 0.4, 0.9])
    la = co.collide(vb, vB, ax, D)["la"]
    assert np.allclose(co.solve_offset(vb, vB, ax, la), D, atol=1e-3)


def test_measured_axis_points_to_the_hands():
    a = co.axis_measured([10.0], [0.0], ["R"], [34.0])
    assert a[0, 0] < 0 and a[0, 2] > 0                                 # toward a righty's hands: 3B side + up
