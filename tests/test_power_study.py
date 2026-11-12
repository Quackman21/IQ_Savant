import numpy as np
import pandas as pd

from pipeline import power_study as ps


def test_top8_and_max_adjust():
    assert ps._top8(pd.Series(range(1, 17))) == 15.5
    rng = np.random.default_rng(0)
    n = rng.integers(40, 500, 300)
    t = pd.DataFrame({"BIP": n, "fly_n": n // 3,
                      "max_raw": 105 + 2 * np.log(n) + rng.normal(0, .5, 300),       # max grows with sample size
                      "max_fly_raw": 100 + 2 * np.log(n // 3) + rng.normal(0, .5, 300)})
    a = ps.bip_adjust_max(t)
    assert abs(np.corrcoef(np.log(n), a["Max EV (BIP-adj)"])[0, 1]) < 0.1        # size bias removed


def test_weights_find_the_real_driver():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=800), "b": rng.normal(size=800)})
    y = 2 * X["a"] + rng.normal(size=800)
    w = ps.weights(X, y)
    assert w["a"] > 0.9
