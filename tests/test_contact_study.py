import numpy as np
import pandas as pd

from pipeline import contact_study as cs


def test_partial_r_holds_power_equal():
    rng = np.random.default_rng(0)
    power = rng.normal(size=2000)
    contact = -0.6 * power + rng.normal(size=2000)               # contact hitters have less power
    woba = 0.8 * power + 0.3 * contact + rng.normal(size=2000)
    raw = np.corrcoef(contact, woba)[0, 1]
    part = cs.partial_r(pd.Series(contact), pd.Series(woba), pd.DataFrame({"p": power}))
    assert raw < 0.05 and part > 0.2                              # contact looks useless until power is held equal
