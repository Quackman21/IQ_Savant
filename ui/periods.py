"""Periods picker: one compact button ("Periods: 2026, 2025, …") that opens a
scrollable panel, like the reference site:

    [Default] [Full Seasons] [Curr Season]
    Full Seasons            ☑ 2026  ☑ 2025  ☑ 2024  ☐ 2023 …
    2026 – By Month / Date  ☐ Latest Week  ☐ Last 30 Days  ☐ Oct  ☐ Sep … ☐ Mar
    2025 – By Month / Date  …

Returns a list of Period(code, season, kind, month). Date windows for Latest
Week / Last 30 Days are resolved against the data (last game in that season),
see resolve().
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

MONTHS = [10, 9, 8, 7, 6, 5, 4, 3]          # newest first, like the reference


@dataclass(frozen=True)
class Period:
    code: str            # "2026", "2026-wk", "2026-30", "2026-09"
    season: int
    kind: str            # season | week | last30 | month
    month: int = 0

    @property
    def short(self) -> str:
        if self.kind == "season":
            return str(self.season)
        if self.kind == "week":
            return f"{self.season} | Latest Week"
        if self.kind == "last30":
            return f"{self.season} | Last 30 Days"
        return f"{self.season} | {calendar.month_abbr[self.month]}"


def options(seasons: list[int], by_month: bool = True) -> list[Period]:
    out = [Period(str(y), y, "season") for y in seasons]
    if by_month:
        for y in seasons:
            out += [Period(f"{y}-wk", y, "week"), Period(f"{y}-30", y, "last30")]
            out += [Period(f"{y}-{m:02d}", y, "month", m) for m in MONTHS]
    return out


def resolve(p: Period, game_dates: pd.Series) -> tuple[date, date, str] | None:
    """(start, end, row label) for a period, using the season's game dates.
    Latest Week / Last 30 Days end on the last game played that season."""
    if p.kind == "season":
        return date(p.season, 3, 1), date(p.season, 11, 30), "Season"
    if p.kind == "month":
        last = calendar.monthrange(p.season, p.month)[1]
        return date(p.season, p.month, 1), date(p.season, p.month, last), calendar.month_abbr[p.month]
    gd = pd.to_datetime(game_dates, errors="coerce").dropna()
    if gd.empty:
        return None
    end = gd.max().date()
    start = end - timedelta(days=6 if p.kind == "week" else 29)
    lab = f"{start.month}/{start.day} - {end.month}/{end.day}"
    return start, end, lab


# ----------------------------------------------------------------------------
# Streamlit widget
# ----------------------------------------------------------------------------
def picker(key: str, seasons: list[int], by_month: bool = True, default_n: int = 3) -> list[Period]:
    import streamlit as st

    opts = options(seasons, by_month)
    ck = lambda p: f"{key}__{p.code}"
    defaults = {p.code for p in opts if p.kind == "season" and p.season in seasons[:default_n]}

    # first visit (or new player): tick the defaults
    sig = f"{key}__sig"
    if st.session_state.get(sig) != tuple(seasons):
        for p in opts:
            st.session_state[ck(p)] = p.code in defaults
        st.session_state[sig] = tuple(seasons)

    chosen = [p for p in opts if st.session_state.get(ck(p))]
    names = [p.short for p in chosen]
    label = ("Periods: " + ", ".join(names[:2]) + (", …" if len(names) > 2 else "")) if names else "Periods: none"

    def _set(which: str):
        for p in opts:
            if which == "default":
                v = p.code in defaults
            elif which == "full":
                v = p.kind == "season"
            else:
                v = p.kind == "season" and p.season == seasons[0]
            st.session_state[ck(p)] = v

    with st.popover(label, use_container_width=True):
        b = st.columns(3)
        b[0].button("Default", key=f"{key}__b1", on_click=_set, args=("default",), use_container_width=True)
        b[1].button("Full Seasons", key=f"{key}__b2", on_click=_set, args=("full",), use_container_width=True)
        b[2].button("Curr Season", key=f"{key}__b3", on_click=_set, args=("curr",), use_container_width=True)
        try:
            box = st.container(height=340)
        except TypeError:
            box = st.container()
        with box:
            st.markdown("**Full Seasons**")
            for p in opts:
                if p.kind == "season":
                    st.checkbox(str(p.season), key=ck(p))
            if by_month:
                for y in seasons:
                    st.markdown(f"**{y} – By Month / Date**")
                    for p in opts:
                        if p.season == y and p.kind != "season":
                            st.checkbox(p.short, key=ck(p))
    return [p for p in opts if st.session_state.get(ck(p))]
