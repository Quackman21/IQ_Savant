"""Pitch physics from Statcast's measured trajectory (release velocity + constant acceleration, given at y = 50 ft).

The measured trajectory already reflects the air the pitch was thrown through, so nothing here uses weather.

  ZONE-ADJUSTED APPROACH ANGLES (ZAVAA / ZAHAA): the same release, velocity and accelerations, re-aimed so the pitch
    crosses the middle of the zone (2.5 ft up, centre of the plate) -- exact for the constant-acceleration model.

Not computed: spin efficiency / active spin. MLB's pitch-by-pitch data doesn't carry it, and inferring it from
movement would credit all of the movement to Magnus -- seam-shifted wake can't be separated without seam
orientation (CV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FRONT_FT = 17 / 12
ZONE_MID_FT = 2.5


def _c(df, n):
    return pd.to_numeric(df[n], errors="coerce") if n in df else pd.Series(np.nan, index=df.index)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Per pitch: zavaa / zahaa (degrees)."""
    vx0, vy0, vz0 = _c(df, "vx0"), _c(df, "vy0"), _c(df, "vz0")
    ax, ay, az = _c(df, "ax"), _c(df, "ay"), _c(df, "az")
    y_rel = 60.5 - _c(df, "release_extension").fillna(6.0)

    def t_at(y):
        a, b, c = 0.5 * ay, vy0, 50 - y
        return (-b - np.sqrt(np.clip(b * b - 4 * a * c, 0, None))) / (2 * a)

    t_r, t_p = t_at(y_rel), t_at(FRONT_FT)
    vyf, vzf, vxf = vy0 + ay * t_p, vz0 + az * t_p, vx0 + ax * t_p
    dt = t_p - t_r
    vz_adj = vzf + (ZONE_MID_FT - _c(df, "plate_z")) / dt
    vx_adj = vxf + (0.0 - _c(df, "plate_x")) / dt
    return pd.DataFrame({"zavaa": -np.degrees(np.arctan(vz_adj / vyf)),
                         "zahaa": -np.degrees(np.arctan(vx_adj / vyf))}, index=df.index)
