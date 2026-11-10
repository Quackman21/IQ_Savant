"""Swing-decision models: was each pitch a GOOD decision (swing or take)?

Every Swing Decisions stat (GD%, Decision grades) asks good_decision(), so
swapping in your real model is one line: change ACTIVE.

    def iq_v1(df):            # df = Statcast pitches, plus df["attack"] (Heart/Shadow/Chase/Waste)
        ...                   # return 1.0 = good, 0.0 = bad, NaN = no grade for this pitch
    MODELS["iq_v1"] = iq_v1
    ACTIVE = "iq_v1"
Then rebuild the league reference (--force) so league averages use it too.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TAKES_OK_2K = False   # placeholder rule switch: with 2 strikes, is taking a Shadow pitch fine?


def placeholder_v0(df: pd.DataFrame, swing: pd.Series, attack: pd.Series) -> pd.Series:
    """Simple stand-in until the real decision model exists:
       Heart  -> swing is good, take is bad
       Shadow -> either is fine (borderline pitch), except a take with 2 strikes is bad
       Chase / Waste -> take is good, swing is bad
       Unknown location -> no grade."""
    two_k = df["strikes"].fillna(0).ge(2) if "strikes" in df else pd.Series(False, index=df.index)
    good = pd.Series(np.nan, index=df.index)
    good[attack.eq("Heart")] = swing[attack.eq("Heart")].astype(float)
    sh = attack.eq("Shadow")
    good[sh] = np.where(two_k[sh] & ~swing[sh] & (not TAKES_OK_2K), 0.0, 1.0)
    out = attack.isin(["Chase", "Waste"])
    good[out] = (~swing[out]).astype(float)
    return good


def iq_v1(df: pd.DataFrame, swing: pd.Series, attack: pd.Series) -> pd.Series:
    """IQ Swing Decisions (models/decisions_iq.py): run-value decision vs a league-average hitter,
    count first, then location, then runners + outs on close calls. Neutral = no grade.
    Falls back to the placeholder if the model isn't trained yet."""
    from models import decisions_iq
    if decisions_iq.load() is None:
        return placeholder_v0(df, swing, attack)
    return decisions_iq.score(df)["good"].astype(float)


MODELS = {"placeholder_v0": placeholder_v0, "iq_v1": iq_v1}
LABEL = {"placeholder_v0": "Placeholder decision model (zone-based)", "iq_v1": "IQ Swing Decisions v1"}

# Which model the site uses. Set by:  python -m pipeline.train_decisions --activate
_ACTIVE_FILE = Path(__file__).resolve().parent / "artifacts" / "ACTIVE_DECISIONS"


def _read_active() -> str:
    try:
        name = _ACTIVE_FILE.read_text().strip()
    except OSError:
        return "placeholder_v0"
    if name == "iq_v1":
        from models import decisions_iq
        if not decisions_iq.ARTIFACT.exists():
            return "placeholder_v0"
    return name if name in MODELS else "placeholder_v0"


ACTIVE = _read_active()


def set_active(name: str) -> None:
    _ACTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ACTIVE_FILE.write_text(name + "\n")


def good_decision(df, swing, attack, model: str | None = None) -> pd.Series:
    return MODELS[model or ACTIVE](df, swing, attack)


def decision_detail(df, swing, attack, model: str | None = None) -> pd.DataFrame:
    """Per pitch: good (1 / 0 / NaN), quality (x100, capped) and runs vs a league-average hitter.
    Quality / runs exist only for the IQ model -- the placeholder leaves them blank."""
    name = model or ACTIVE
    if name == "iq_v1":
        from models import decisions_iq
        if decisions_iq.load() is not None:
            s = decisions_iq.score(df)
            return pd.DataFrame({"good": s["good"], "quality": s["quality"], "runs": s["runs_vs_avg"]},
                                index=df.index).astype(float)
    good = MODELS[name](df, swing, attack)
    return pd.DataFrame({"good": good, "quality": np.nan, "runs": np.nan}, index=df.index)
