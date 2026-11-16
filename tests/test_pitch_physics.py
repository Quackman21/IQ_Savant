import math

import numpy as np
import pandas as pd

from core import pitch_physics as pp

G = 32.174


def _pitch(plate_z=2.5, plate_x=0.0):
    return pd.DataFrame({"vx0": [0.0], "vy0": [-138.0], "vz0": [-5.0], "ax": [0.0], "ay": [28.0], "az": [-G + 10.0],
                         "release_extension": [6.5], "plate_x": [plate_x], "plate_z": [plate_z]})


def test_zavaa_equals_vaa_when_already_mid_zone():
    out = pp.compute(_pitch())
    t_p = (138.0 - np.sqrt(138.0 ** 2 - 4 * 14.0 * (50 - 17 / 12))) / (2 * 14.0)
    vyf, vzf = -138.0 + 28.0 * t_p, -5.0 + (-G + 10.0) * t_p
    assert abs(out["zavaa"].iloc[0] + math.degrees(math.atan(vzf / vyf))) < 1e-6 or \
        abs(out["zavaa"].iloc[0] - (-math.degrees(math.atan(vzf / vyf)))) < 1e-6
    assert abs(out["zahaa"].iloc[0]) < 1e-6


def test_low_pitch_re_aimed_up_arrives_flatter():
    mid, low = pp.compute(_pitch(2.5))["zavaa"].iloc[0], pp.compute(_pitch(1.5))["zavaa"].iloc[0]
    assert abs(mid - low) > 0.5


def test_no_weather_or_spin_efficiency():
    assert set(pp.compute(_pitch()).columns) == {"zavaa", "zahaa"}
