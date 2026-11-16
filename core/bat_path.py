"""Bat Path tab: point of contact vs the plate for a hitter's swings (same method as the AAR -- MLB's definition).

  front (pf)  = Statcast's contact in front of his center of mass - his stance depth (Savant's board: playoff ->
                vs this pitcher's hand -> season), inches in front of the plate's front edge
  side  (pxc) = where the ball was at that depth, inches from the middle of the plate, catcher's view
Hitters not on the stance board fall back to the AAR's calibrated estimate (flagged).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import aar

POST = {"F", "D", "L", "W"}


def poc(sc: pd.DataFrame, pid: int, season: int, views: dict | None = None, stance: dict | None = None) -> pd.DataFrame:
    """Per row of sc (same index): pf, pxc, depth_src, estimated (bool)."""
    out = pd.DataFrame({"pf": np.nan, "pxc": np.nan, "depth_src": "", "estimated": False}, index=sc.index)
    if sc is None or sc.empty or aar.CONTACT_Y not in sc:
        return out
    num = lambda c: pd.to_numeric(sc[c], errors="coerce") if c in sc else pd.Series(np.nan, index=sc.index)
    made = aar.made_contact(sc)
    p = pd.DataFrame({"cy": num(aar.CONTACT_Y).where(made), "cx": num(aar.CONTACT_X).where(made),
                      "stand": sc["stand"].astype(str), "p_throws": sc["p_throws"].astype(str),
                      "px": num("plate_x"), "slope": aar.plate_slope(sc)}, index=sc.index)
    post = sc["game_type"].astype(str).isin(POST) if "game_type" in sc else pd.Series(False, index=sc.index)
    st = pd.concat([aar.pitch_stance(p[~post], pid, views or {}, False),
                    aar.pitch_stance(p[post], pid, views or {}, True)]).reindex(sc.index)
    p = p.join(st)
    cal = None
    if p["depth"].isna().any() and p["cy"].notna().any():
        try:
            cal = aar.plate_calibration(sc, group=aar.hip_group(season), stance=stance or {})
        except Exception:
            cal = None
    o = aar.on_plate(p, cal)
    out["pf"], out["pxc"] = o["pf"], o["pxc"]
    out["depth_src"] = p["depth_src"].fillna("")
    out["estimated"] = o["pf"].notna() & p["depth"].isna()
    return out
