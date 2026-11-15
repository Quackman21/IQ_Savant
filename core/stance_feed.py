"""Savant's batting-stance board, downloaded by the site itself and kept current.

Why: the point of contact vs the plate is MLB's definition -- Statcast's contact (inches in front of his center of
mass) minus his stance depth. The more specific the stance depth (this month, two strikes vs not, vs this pitcher's
hand), the closer each dot is to where he actually stood on that pitch.

Savant's page (baseballsavant.mlb.com/visuals/batting-stance) has dropdowns (id="ddlXxx") for season, month /
strike views, game type, pitcher hand and minimum swings; the download link takes the same names without 'ddl'. The
site DISCOVERS them: it reads every dropdown and its options from the page, tries each option and keeps the ones that
change the downloaded data. Saved to data/stance/feed.json and reused every day.

    python -m core.stance_feed discover        # once (re-run if Savant changes the page)
    python -m core.stance_feed refresh [2026]  # daily during the season (pipeline.update runs it)
    python -m core.stance_feed report [2026]   # how much each split (vs LHP / RHP, playoffs, ...) moves stance
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

PAGE = "https://baseballsavant.mlb.com/visuals/batting-stance"
DIR = Path(__file__).resolve().parents[1] / "data" / "stance"
FEED = DIR / "feed.json"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"}

def _session():
    import requests
    s = requests.Session()
    s.headers.update(UA)
    return s


def fetch(params: dict, s=None, tries: int = 3) -> pd.DataFrame:
    """One CSV download from the stance board (empty on failure)."""
    s = s or _session()
    q = {"csv": "true", **params}
    for i in range(tries):
        try:
            r = s.get(PAGE, params=q, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(10 * (i + 1))
                continue
            text = r.content.decode("utf-8-sig")
            if r.status_code != 200 or text.lstrip().startswith("<"):
                return pd.DataFrame()
            df = pd.read_csv(io.StringIO(text))
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            time.sleep(5 * (i + 1))
    return pd.DataFrame()


def _sig(df: pd.DataFrame) -> str:
    return hashlib.md5(df.to_csv(index=False).encode()).hexdigest() if not df.empty else "empty"


GROUP_KEYS = ("group", "groupBy", "view", "split")
GROUP_VALUES = ("month", "Month", "months", "game_month", "strikes", "strike", "two_strikes", "twoStrikes",
                "2strikes", "count", "Strikes")


def page_selects(s=None) -> dict:
    """Every dropdown on the stance page: {param: [(value, label), ...]}. Savant's dropdowns are id="ddlXxx" and the
    download link uses the same name without 'ddl' (ddlMinSwings -> minSwings, ddlPitchHand -> pitchHand)."""
    s = s or _session()
    try:
        html = s.get(PAGE, timeout=60).text
    except Exception:
        return {}
    out = {}
    for sid, body in re.findall(r'<select[^>]*?id="(ddl\w+)"[^>]*>(.*?)</select>', html, flags=re.S):
        opts = [(v, re.sub(r"<[^>]+>|\s+", " ", lab).strip())
                for v, lab in re.findall(r'<option[^>]*?value="([^"]*)"[^>]*>(.*?)</option>', body, flags=re.S)]
        out[sid[3].lower() + sid[4:]] = opts
    return out


def discover(year: int | None = None, verbose: bool = True) -> dict:
    """Read the page's dropdowns, try every option, keep what changes the data. Saves data/stance/feed.json."""
    year = year or (date.today().year if date.today().month >= 4 else date.today().year - 1)
    s = _session()
    say = print if verbose else (lambda *a, **k: None)
    sel = page_selects(s)
    for k, opts in sel.items():
        say(f"  dropdown {k}: " + ", ".join(f"{v}={lab}" for v, lab in opts))
    base_p = {"minSwings": "1"} if "minSwings" in sel else {}
    base = fetch(base_p, s)
    say(f"baseline ({base_p}): {len(base)} rows, columns {list(base.columns)}")
    tried = {}
    for k, opts in sel.items():
        if k in ("minSwings", "leaderboardType", "minGroupSwings", "seasonEnd"):   # site menu / companions
            continue
        for v, lab in opts:
            if v == "" or v.lower() in ("any", "all"):
                continue
            p = {**base_p, k: v}
            if k == "seasonStart":
                p["seasonEnd"] = v                                   # a season = start and end the same year
            df = fetch(p, s)
            time.sleep(1.0)
            new = [c for c in df.columns if c not in base.columns]
            changed = (not df.empty) and _sig(df) != _sig(base)
            tried.setdefault(k, []).append({"value": v, "label": lab, "rows": len(df), "changed": changed,
                                            "new_columns": new})
            say(f"  {k}={v:<10} ({lab[:24]:24s}) -> {len(df):5d} rows{'  CHANGED' if changed else ''}"
                f"{'  new columns ' + str(new) if new else ''}")
    # Season / Month and "Strikes: 2 vs < 2" are toggle buttons, not dropdowns -- the page's code carries a 'group'
    # parameter; try likely values against the same min-per-group setting and keep any that add columns / rows
    gbase_p = {**base_p, "minGroupSwings": "1"}
    gref = fetch(gbase_p, s)
    for key in GROUP_KEYS:
        for v in GROUP_VALUES:
            df = fetch({**gbase_p, key: v}, s)
            time.sleep(1.0)
            new = [c for c in df.columns if c not in gref.columns]
            if df.empty or (not new and len(df) == len(gref)):
                continue
            tried.setdefault(key, []).append({"value": v, "label": v, "rows": len(df), "changed": True,
                                              "new_columns": new})
            say(f"  {key}={v:<12} -> {len(df):5d} rows (vs {len(gref)}){'  new columns ' + str(new) if new else ''}")
    DIR.mkdir(parents=True, exist_ok=True)
    out = {"selects": {k: [list(o) for o in v] for k, v in sel.items()}, "tried": tried, "base": base_p,
           "checked": date.today().isoformat()}
    FEED.write_text(json.dumps(out, indent=1))
    say(f"saved -> {FEED}")
    return out


def _feed() -> dict:
    try:
        return json.loads(FEED.read_text())
    except Exception:
        return {}


def _views(f: dict, season: int) -> tuple[dict, dict]:
    """(base params, {view name: extra params}) from what discovery found."""
    sel, tried = f.get("selects", {}), f.get("tried", {})
    base = dict(f.get("base", {}))
    if any(o[0] == str(season) for o in sel.get("seasonStart", [])):
        base["seasonStart"] = base["seasonEnd"] = str(season)
    ok = {k: {t["value"] for t in v if t.get("changed")} for k, v in tried.items()}
    views = {"season": {}}
    for v in ("Playoff", "P"):
        if v in ok.get("gameType", set()):
            views["playoff"] = {"gameType": v}
            break
    for h in ("L", "R"):
        if h in ok.get("pitchHand", set()):
            views[f"vs_{h}"] = {"pitchHand": h}
    for key in GROUP_KEYS:
        for v in ok.get(key, set()):
            views[f"{key}_{v}"] = {key: v, "minGroupSwings": "1"}
    return base, views


def refresh(season: int, verbose: bool = True) -> list[str]:
    """Download the season view + every split view that works into data/stance/<season>/, and keep
    data/batting_stance_<season>.csv (what the AAR reads) current."""
    base, views = _views(_feed(), season)
    s = _session()
    d = DIR / str(season)
    d.mkdir(parents=True, exist_ok=True)
    for junk in d.glob("leaderboardType_*.csv"):               # v22.3 mistook Savant's site menu for a split
        junk.unlink()
    wrote = []
    for name, extra in views.items():
        df = fetch({**base, **extra}, s)
        time.sleep(1.0)
        if df.empty:
            if verbose:
                print(f"  {season} {name}: no data")
            continue
        df.to_csv(d / f"{name}.csv", index=False)
        wrote.append(name)
        if verbose:
            print(f"  {season} {name}: {len(df)} rows" + (f", columns {list(df.columns)}" if name != "season" else ""))
        if name == "season" and "avg_batter_y_position" in df:
            df.to_csv(DIR.parent / f"batting_stance_{season}.csv", index=False)
    (d / "refreshed.txt").write_text(date.today().isoformat())
    return wrote


def views(season: int) -> dict:
    """{view: {(batter id, 'R'/'L'): stance depth}} for every downloaded view of a season ('season', 'playoff',
    'vs_L', 'vs_R', ...); 'season' falls back to data/batting_stance_<season>.csv."""
    out = {}
    d = DIR / str(season)
    files = {f.stem: f for f in d.glob("*.csv")} if d.exists() else {}
    if "season" not in files and (DIR.parent / f"batting_stance_{season}.csv").exists():
        files["season"] = DIR.parent / f"batting_stance_{season}.csv"
    for name, f in files.items():
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "avg_batter_y_position" not in df or "id" not in df:
            continue
        side = df["side"] if "side" in df else df.get("bat_side", "R")
        out[name] = {(int(b), str(sd)): float(v) for b, sd, v in zip(df["id"], side, df["avg_batter_y_position"])
                     if pd.notna(v)}
    return out


def report(season: int) -> None:
    """How much each split moves stance depth vs the season number (is the split worth using?)."""
    v = views(season)
    base = v.get("season", {})
    print(f"{season}: views {sorted(v)}")
    for name, m in v.items():
        if name == "season":
            continue
        diff = pd.Series({k: m[k] - base[k] for k in m if k in base})
        if diff.empty:
            continue
        print(f"  {name:12s} {len(diff):4d} hitter-sides   differs from season by: median {diff.abs().median():.1f}\"  "
              f"90% within {diff.abs().quantile(.9):.1f}\"   >2\": {(diff.abs() > 2).mean():.0%}   "
              f">4\": {(diff.abs() > 4).mean():.0%}")


def stale(season: int) -> bool:
    """True when the season's stance files are older than today (during the season) or missing."""
    f = DIR / str(season) / "refreshed.txt"
    if not f.exists():
        return True
    today = date.today()
    in_season = season == today.year and 3 <= today.month <= 11
    return in_season and f.read_text().strip() < today.isoformat()


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "discover":
        discover(int(a[1]) if len(a) > 1 else None)
    elif a and a[0] == "refresh":
        refresh(int(a[1]) if len(a) > 1 else date.today().year)
    elif a and a[0] == "report":
        report(int(a[1]) if len(a) > 1 else date.today().year)
    else:
        print(__doc__)
