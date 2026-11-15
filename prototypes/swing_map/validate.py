"""Check the hip-position model against Savant's batting-stance leaderboard for EVERY hitter (Savant is the referee,
never an input): hips at contact should land a little forward of where he stands.

    python prototypes/swing_map/validate.py LEAGUE.csv.gz batting-stance.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import build  # noqa: E402

lg = pd.read_csv(sys.argv[1], low_memory=False)
st = pd.read_csv(sys.argv[2])
asof = str(lg["game_date"].max())[:10]
fits = build.hip_fits(lg, asof=asof)
raw = {k: -f["F0"] for k, f in fits.items()}
build.shrink_to_group(fits)
rows = []
for (b, side), f in fits.items():
    m = st[(st["id"] == b) & (st["side"] == side)]
    if m.empty:
        continue
    depth = float(m["avg_batter_y_position"].iloc[0])
    rows.append({"batter": b, "side": side, "name": m["name"].iloc[0], "stance": depth, "raw": raw[(b, side)],
                 "model": -f["F0"], "n": f["n"]})
d = pd.DataFrame(rows)
for col in ["raw", "model"]:
    fwd = d["stance"] - d[col]
    ok = ((fwd >= 0) & (fwd <= 12)).mean()
    print(f"{col:6s}: {len(d)} hitters  agree {ok:.0%}  behind stance {(fwd < 0).mean():.0%}  "
          f"move forward avg {fwd.mean():+.1f}\" spread {fwd.std():.1f}\"  corr(model, stance) {d[col].corr(d['stance']):.2f}")
d["fwd"] = d["stance"] - d["model"]
out = Path(sys.argv[1]).with_name("validation.csv")
d.sort_values("fwd").to_csv(out, index=False)
print("worst (behind stance):"); print(d.nsmallest(8, "fwd")[["name", "side", "stance", "model", "fwd", "n"]].round(1).to_string(index=False))
print("worst (moved a lot):"); print(d.nlargest(8, "fwd")[["name", "side", "stance", "model", "fwd", "n"]].round(1).to_string(index=False))
print("->", out)
