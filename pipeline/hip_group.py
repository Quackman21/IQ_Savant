"""Calibrates the stand-in used for hitters who aren't on Savant's batting-stance board yet (the AAR uses MLB's own
contact-vs-plate -- contact from the hips minus stance depth -- whenever the board has him).

    python -m pipeline.hip_group ~/Desktop/swing_data/league_2026.csv.gz [batting-stance.csv]

Writes data/hip_group_<season>.json = {"mu": average hips-at-contact (inches behind the front of the plate),
"tau": the REAL hitter-to-hitter spread (raw spread minus fitting noise), "n_hitters"} -- the AAR pulls every hitter
toward it, harder when his own swings are noisy. With a batting-stance CSV (Savant leaderboard) it also checks the
model for every hitter: hips at contact should land 0-12" forward of where he stands. Savant is never an input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core import aar

NO_SHRINK = {"mu": 0.0, "tau": 1e6}


def main(argv: list[str]) -> None:
    lg = pd.read_csv(Path(argv[0]).expanduser(), low_memory=False)
    season = int(str(lg["game_date"].max())[:4])
    asof = str(lg["game_date"].max())[:10]
    raw = {}
    for b, g in lg.groupby("batter"):
        cal = aar.plate_calibration(g, asof=asof, group=NO_SHRINK)
        for side, c in (cal or {}).items():
            raw[(int(b), side)] = (c["raw"], c["se"], c["n"])
    F = np.array([v[0] for v in raw.values()])
    se = np.array([v[1] for v in raw.values()])
    mu = float(F.mean())
    tau = float(np.sqrt(max(F.var(ddof=1) - (se ** 2).mean(), 1.0)))
    ref = {"mu": round(mu, 2), "tau": round(tau, 2), "n_hitters": len(raw)}
    # calibrate the drift fit to Savant's stance board -- it's only the stand-in for hitters not on the board yet
    stance = aar.stance_depths(season)
    pairs = [(stance[k], v[0], v[1]) for k, v in raw.items() if k in stance]
    if len(pairs) >= 20:
        d_, f_, s_ = map(np.array, zip(*pairs))
        off = d_ - f_
        ref.update({"offset": round(float(off.mean()), 2),
                    "offset_sd": round(float(np.sqrt(max(off.var(ddof=1) - (s_ ** 2).mean(), 1.0))), 2),
                    "stance_mu": round(float(np.mean(list(stance.values()))), 2),
                    "stance_sd": round(float(np.std(list(stance.values()), ddof=1)), 2), "n_calibrated": len(pairs)})
    out = Path(__file__).resolve().parents[1] / "data" / f"hip_group_{season}.json"
    out.write_text(json.dumps(ref))
    print(ref)
    print(f"{len(raw)} hitter-sides: MLB average hips {mu:.1f}\" behind the front of the plate, real spread ±{tau:.1f}\" "
          f"(raw spread ±{F.std():.1f}\", fitting noise ±{np.sqrt((se ** 2).mean()):.1f}\") -> {out}")

    if len(argv) > 1:
        st = pd.read_csv(Path(argv[1]).expanduser())
        rows = []
        for (b, side), (r, s_, n) in raw.items():
            m = st[(st["id"] == b) & (st["side"] == side)]
            if m.empty:
                continue
            k = tau ** 2 / (tau ** 2 + s_ ** 2)
            model = mu + k * (r - mu)
            rows.append({"name": m["name"].iloc[0], "side": side, "n": n,
                         "stance": float(m["avg_batter_y_position"].iloc[0]), "raw": r, "model": model})
        d = pd.DataFrame(rows)
        print(f"\nSavant check ({len(d)} hitters with a stance on the leaderboard):")
        for col in ("raw", "model"):
            fwd = d["stance"] - d[col]
            print(f"  {col:5s}: agree {((fwd >= 0) & (fwd <= 12)).mean():.0%}   behind stance {(fwd < 0).mean():.0%}   "
                  f"move forward {fwd.mean():+.1f}\" ± {fwd.std():.1f}\"   corr with stance {d[col].corr(d['stance']):.2f}")
        d["fwd"] = d["stance"] - d["model"]
        print("\n  furthest BEHIND their stance (model):")
        print(d.nsmallest(10, "fwd")[["name", "side", "n", "stance", "raw", "model", "fwd"]].round(1).to_string(index=False))
        print("\n  moved forward the MOST (model):")
        print(d.nlargest(10, "fwd")[["name", "side", "n", "stance", "raw", "model", "fwd"]].round(1).to_string(index=False))
        rep = Path(argv[0]).expanduser().with_name(f"hip_check_{season}.csv")
        d.sort_values("fwd").to_csv(rep, index=False)
        print(f"\n  every hitter -> {rep}")


if __name__ == "__main__":
    main(sys.argv[1:])
