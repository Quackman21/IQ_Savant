import numpy as np
import pandas as pd

from pipeline import decision_study as ds


def test_decision_runs_pieces_add_up_to_total():
    rng = np.random.default_rng(0)
    n = 3000
    d = pd.DataFrame({"game_pk": rng.integers(0, 100, n), "batter": rng.integers(1, 4, n), "year": 2025,
                      "swing": rng.random(n) < .45, "whiff": rng.random(n) < .1, "zone": rng.integers(1, 15, n),
                      "att": rng.choice(["Heart", "Shadow", "Chase", "Waste"], n),
                      "pg": rng.choice(["Fastball", "Breaking", "Offspeed"], n),
                      "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n),
                      "good": rng.choice([0.0, 1.0], n), "quality": rng.normal(0, 30, n), "runs": rng.normal(0, .05, n),
                      "events": None, "woba_value": np.nan, "woba_denom": np.nan, "type": "S",
                      "ev": np.nan, "la": np.nan})
    d["graded"] = True
    d["in_z"] = d["zone"].between(1, 9)
    d["two_k"] = d["strikes"] >= 2
    d["ahead"] = [(b, s) in ds.AHEAD for b, s in zip(d["balls"], d["strikes"])]
    d["first"] = (d["balls"] == 0) & (d["strikes"] == 0)
    t = ds.hitter_table(d)
    for group in (ds.ZONE_ACTION, ds.COUNTS, ds.PITCHES):
        total = sum(t["R: " + b] for b in group)
        assert np.allclose(total, t["DEC RUNS/100"])
