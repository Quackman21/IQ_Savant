import math

from core import core4 as c4
from ui.periods import Period, display_order


def test_row_order_months_then_season_last():
    ps = [Period("2025", 2025, "season"), Period("2026", 2026, "season"), Period("2026-04", 2026, "month", 4),
          Period("2026-09", 2026, "month", 9), Period("2026-30", 2026, "last30"), Period("2026-wk", 2026, "week"),
          Period("2025-08", 2025, "month", 8)]
    ps.append(Period("2026-post", 2026, "post"))
    order = [p.code for p in sorted(ps, key=display_order)]
    assert order == ["2026-post", "2026-wk", "2026-30", "2026-09", "2026-04", "2026", "2025-08", "2025"]


def test_grades_blank_until_models():
    assert all(g.stat is None for g in c4.GRADES.values())
    v, z = c4.grade("DECISION GRADE", {"GD%": 0.7}, {"GD%": (0.6, 0.05)}, "20-80")
    assert math.isnan(v) and math.isnan(z)


def test_no_20_80_cap():
    assert c4.to_scale(4.0, "20-80") == 90          # a monster month shows as 90
    assert c4.to_scale(-4.0, "20-80") == 10
    assert c4.to_scale(1.0, "Plus") == 110 and round(c4.to_scale(0, "Percentile")) == 50


def test_default_periods():
    from datetime import date
    from ui.periods import default_codes
    full = {"2026", "2026-post"} | {f"2026-{m:02d}" for m in range(3, 11)}
    assert default_codes([2026, 2025], today=date(2026, 11, 5)) == full                            # offseason
    assert default_codes([2026, 2025], today=date(2026, 10, 10)) == full                           # October
    assert default_codes([2026, 2025], today=date(2026, 6, 15)) == {"2026", "2026-06", "2026-05"}  # in season
    assert "2025-post" in default_codes([2025, 2024], today=date(2026, 2, 1))                      # winter
    assert default_codes([2026], by_month=False) == {"2026"}
