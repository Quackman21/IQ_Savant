"""Weather + park layer: the air every pitch and batted ball moved through.

Sources (all free):
  * MLB Stats API -- each park's location, elevation, field orientation (home plate -> CF compass bearing) and roof
    type; every game's venue, first-pitch time and the weather reported at the park (temp, wind, "Roof Closed"/"Dome").
  * Open-Meteo hourly history -- temperature, humidity, surface pressure, wind speed + direction at each park.

Per pitch (game + inning) it gives: temperature, humidity, pressure, AIR DENSITY (what drag + lift depend on) and the
wind as a vector on the field (+ out to CF, + toward RF), so any ball's tail/headwind = the wind along its spray.
Roof closed or dome -> no wind, indoor temperature.

Field angle convention everywhere: 0 = straight to CF, positive toward the RF side, degrees.

    python -m core.weather parks                 # build/refresh data/weather/parks.json
    python -m core.weather season 2026           # schedule + reported weather + hourly history for every game
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "weather"
PARKS = DIR / "parks.json"
STATSAPI = "https://statsapi.mlb.com/api/v1"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
INDOOR = {"temp_f": 72.0, "rh": 50.0}
GAME_TYPES = "R,F,D,L,W"
MIN_PER_HALF = 10.0          # minutes per half-inning, to place a pitch in time (weather moves slowly)

# -------------------------------------------------------------------------------- physics
def air_density(temp_f: float, rh_pct: float, pressure_hpa: float) -> float:
    """Moist-air density, kg/m^3 (dry air + water vapor partial pressures; humid air is LIGHTER)."""
    tc = (temp_f - 32) / 1.8
    tk = tc + 273.15
    pv = (rh_pct / 100.0) * 6.1078 * 10 ** (7.5 * tc / (tc + 237.3)) * 100      # Pa
    pd_ = pressure_hpa * 100 - pv
    return pd_ / (287.058 * tk) + pv / (461.495 * tk)


SEA_LEVEL_STD = air_density(70.0, 50.0, 1013.25)


def pressure_at(elev_ft: float, p_ref_hpa: float, ref_elev_ft: float) -> float:
    """Move a pressure reading between elevations (scale height ~8.4 km)."""
    return p_ref_hpa * math.exp(-((elev_ft - ref_elev_ft) * 0.3048) / 8400.0)


def field_angle(compass_to_deg: float, cf_bearing: float) -> float:
    """Compass direction something is moving TOWARD -> field angle (0 = CF, + toward RF)."""
    return (compass_to_deg - cf_bearing + 180) % 360 - 180


def along(wind_mph: float, wind_field_deg: float, ball_field_deg: float) -> float:
    """Wind component along a ball's direction (+ = tailwind)."""
    return wind_mph * math.cos(math.radians(wind_field_deg - ball_field_deg))


# MLB's reported wind -> field angle the wind is blowing TOWARD
_WIND_DIR = {"out to cf": 0, "out to rf": 45, "out to lf": -45, "in from cf": 180, "in from rf": -135,
             "in from lf": 135, "l to r": 90, "r to l": -90}


def parse_wind(text: str | None) -> tuple[float | None, float | None]:
    """'7 mph, Out To CF' -> (7.0, 0.0); 'Calm'/'None' -> (0, None); 'Varies' -> (speed, None)."""
    if not text:
        return None, None
    t = str(text).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*mph", t)
    spd = float(m.group(1)) if m else (0.0 if any(w in t for w in ("calm", "none")) else None)
    for k, v in _WIND_DIR.items():
        if k in t:
            return spd, float(v)
    return spd, None


def roof_closed(condition: str | None, roof_type: str | None) -> bool:
    c, r = str(condition or "").lower(), str(roof_type or "").lower()
    return "dome" in c or "roof closed" in c or r in ("dome", "closed")


# -------------------------------------------------------------------------------- downloads
def _get(url: str, params: dict, tries: int = 4) -> dict:
    import requests
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(8 * (i + 1))
                continue
            return {}
        except Exception:
            time.sleep(5 * (i + 1))
    return {}


def _venue_row(v: dict) -> dict:
    loc, fi = v.get("location", {}) or {}, v.get("fieldInfo", {}) or {}
    co = loc.get("defaultCoordinates", {}) or {}
    return {"id": v.get("id"), "name": v.get("name"), "lat": co.get("latitude"), "lon": co.get("longitude"),
            "elev_ft": loc.get("elevation"), "cf_bearing": loc.get("azimuthAngle"), "roof": fi.get("roofType"),
            "tz": (v.get("timeZone") or {}).get("id")}


def build_parks(venue_ids: list[int], verbose: bool = True) -> dict:
    """{venue id: park row} from MLB's venues endpoint (location + field info), merged into parks.json."""
    parks = load_parks()
    todo = [int(v) for v in venue_ids if str(int(v)) not in parks]
    for i in range(0, len(todo), 40):
        js = _get(f"{STATSAPI}/venues", {"venueIds": ",".join(map(str, todo[i:i + 40])),
                                          "hydrate": "location,fieldInfo,timezone"})
        for v in js.get("venues", []):
            parks[str(v["id"])] = _venue_row(v)
    DIR.mkdir(parents=True, exist_ok=True)
    PARKS.write_text(json.dumps(parks, indent=1))
    if verbose:
        miss = [p["name"] for p in parks.values() if p.get("cf_bearing") is None or p.get("lat") is None]
        print(f"  parks: {len(parks)}" + (f"; missing location / field bearing: {miss}" if miss else ""))
    return parks


def load_parks() -> dict:
    try:
        return json.loads(PARKS.read_text())
    except Exception:
        return {}


def schedule(season: int) -> pd.DataFrame:
    """Every game: game_pk, venue, first pitch (UTC), reported temp / wind / condition."""
    js = _get(f"{STATSAPI}/schedule", {"sportId": 1, "season": season, "gameType": GAME_TYPES,
                                        "hydrate": "venue,weather"})
    rows = []
    for d in js.get("dates", []):
        for g in d.get("games", []):
            w = g.get("weather", {}) or {}
            spd, ang = parse_wind(w.get("wind"))
            rows.append({"game_pk": g.get("gamePk"), "venue_id": (g.get("venue") or {}).get("id"),
                         "start_utc": g.get("gameDate"), "game_type": g.get("gameType"),
                         "status": (g.get("status") or {}).get("detailedState"),
                         "rep_temp_f": pd.to_numeric(w.get("temp"), errors="coerce"),
                         "rep_condition": w.get("condition"), "rep_wind": w.get("wind"),
                         "rep_wind_mph": spd, "rep_wind_field_deg": ang})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[~df["status"].astype(str).str.contains("Postponed|Cancelled", case=False)]
    return df


def hourly(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Open-Meteo hourly history at a park (UTC): temp F, RH %, surface pressure hPa, wind mph + FROM direction."""
    js = _get(OPEN_METEO, {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                           "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,"
                                     "wind_direction_10m", "temperature_unit": "fahrenheit",
                           "wind_speed_unit": "mph", "timezone": "UTC"})
    h = js.get("hourly") or {}
    if not h:
        return pd.DataFrame()
    df = pd.DataFrame({"time": pd.to_datetime(h["time"], utc=True), "temp_f": h["temperature_2m"],
                       "rh": h["relative_humidity_2m"], "p_hpa": h["surface_pressure"],
                       "wind_mph": h["wind_speed_10m"], "wind_from": h["wind_direction_10m"]})
    df.attrs["grid_elev_ft"] = (js.get("elevation") or 0) / 0.3048
    return df


# -------------------------------------------------------------------------------- season build
def build_season(season: int, verbose: bool = True, refresh_hourly: bool = False) -> Path:
    """data/weather/games_<season>.csv: per game + the hourly record at its park (data/weather/<season>/<venue>.csv)."""
    say = print if verbose else (lambda *a, **k: None)
    sch = schedule(season)
    if sch.empty:
        say(f"  {season}: no schedule")
        return DIR / f"games_{season}.csv"
    parks = build_parks(sorted(sch["venue_id"].dropna().astype(int).unique()), verbose=verbose)
    out_dir = DIR / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, t1 = pd.to_datetime(sch["start_utc"], utc=True).min(), pd.to_datetime(sch["start_utc"], utc=True).max()
    for vid in sorted(sch["venue_id"].dropna().astype(int).unique()):
        f = out_dir / f"{vid}.csv"
        p = parks.get(str(vid), {})
        if (f.exists() and not refresh_hourly) or p.get("lat") is None:
            continue
        h = hourly(p["lat"], p["lon"], f"{t0:%Y-%m-%d}", f"{(t1 + pd.Timedelta(days=1)):%Y-%m-%d}")
        time.sleep(0.5)
        if h.empty:
            say(f"  {season}: no hourly weather for {p.get('name')}")
            continue
        h["grid_elev_ft"] = h.attrs.get("grid_elev_ft", 0)
        h.to_csv(f, index=False)
        say(f"  {season}: hourly weather {p.get('name')} ({len(h):,} hours)")
    sch.to_csv(DIR / f"games_{season}.csv", index=False)
    say(f"  {season}: {len(sch):,} games -> {DIR / f'games_{season}.csv'}")
    return DIR / f"games_{season}.csv"


# -------------------------------------------------------------------------------- lookups
_GAMES: dict = {}
_HOURLY: dict = {}


def _games(season: int) -> pd.DataFrame:
    if season not in _GAMES:
        f = DIR / f"games_{season}.csv"
        _GAMES[season] = pd.read_csv(f).set_index("game_pk") if f.exists() else pd.DataFrame()
    return _GAMES[season]


def _hourly(season: int, vid: int) -> pd.DataFrame:
    k = (season, vid)
    if k not in _HOURLY:
        f = DIR / str(season) / f"{vid}.csv"
        _HOURLY[k] = pd.read_csv(f, parse_dates=["time"]) if f.exists() else pd.DataFrame()
    return _HOURLY[k]


def wind_source() -> str:
    """'reported' (MLB's in-park reading) or 'hourly' (Open-Meteo 10 m) -- whichever the carry check showed explains
    fly-ball carry better (data/calibration/wind_source.json); 'reported' until it's run."""
    try:
        return json.loads((ROOT / "data" / "calibration" / "wind_source.json").read_text()).get("source", "reported")
    except Exception:
        return "reported"


def conditions(game_pk: int, season: int, inning: int = 1, top: bool = True, feed: dict | None = None,
               wind: str | None = None) -> dict:
    """The air for a pitch: temp, humidity, pressure, air density (+ % vs a 70°F sea-level day), wind on the field.
    From the season table when it's built; otherwise from a game feed (gameData.weather/venue) + park table."""
    parks = load_parks()
    g = _games(season)
    row = g.loc[game_pk].to_dict() if (not g.empty and game_pk in g.index) else {}
    if not row and feed:
        gd = feed.get("gameData", {})
        w = gd.get("weather", {}) or {}
        spd, ang = parse_wind(w.get("wind"))
        row = {"venue_id": (gd.get("venue") or {}).get("id"), "start_utc": (gd.get("datetime") or {}).get("dateTime"),
               "rep_temp_f": pd.to_numeric(w.get("temp"), errors="coerce"), "rep_condition": w.get("condition"),
               "rep_wind": w.get("wind"), "rep_wind_mph": spd, "rep_wind_field_deg": ang}
        if row["venue_id"] and str(row["venue_id"]) not in parks:
            try:
                parks = build_parks([int(row["venue_id"])], verbose=False)
            except Exception:
                pass
    if not row or row.get("venue_id") is None or pd.isna(row.get("venue_id")):
        return {}
    vid = int(row["venue_id"])
    park = parks.get(str(vid), {})
    elev = float(park.get("elev_ft") or 0)
    when = pd.to_datetime(row.get("start_utc"), utc=True, errors="coerce")
    if when is not pd.NaT and when is not None:
        when = when + pd.Timedelta(minutes=MIN_PER_HALF * (2 * (max(int(inning), 1) - 1) + (0 if top else 1)))
    out = {"venue": park.get("name"), "elev_ft": elev, "roof_closed": roof_closed(row.get("rep_condition"),
                                                                                    park.get("roof")),
           "reported": {"temp_f": row.get("rep_temp_f"), "wind": row.get("rep_wind"),
                        "condition": row.get("rep_condition")}, "source": []}
    h = _hourly(season, vid)
    hr = None
    if not h.empty and when is not None and when is not pd.NaT:
        i = int(np.argmin(np.abs((h["time"] - when).dt.total_seconds().to_numpy())))
        if abs((h["time"].iloc[i] - when).total_seconds()) <= 2 * 3600:
            hr = h.iloc[i]
    # temperature: the park's reported game temp, moved by the hourly change since first pitch
    temp = row.get("rep_temp_f")
    if hr is not None:
        temp0 = h.iloc[int(np.argmin(np.abs((h["time"] - pd.to_datetime(row.get("start_utc"), utc=True))
                                            .dt.total_seconds().to_numpy())))]["temp_f"]
        temp = (temp if pd.notna(temp) else temp0) + (hr["temp_f"] - temp0)
        out["source"].append("hourly")
    if pd.notna(row.get("rep_temp_f")):
        out["source"].append("MLB reported")
    rh = float(hr["rh"]) if hr is not None else 50.0
    p = pressure_at(elev, float(hr["p_hpa"]), float(hr.get("grid_elev_ft", elev))) if hr is not None \
        else pressure_at(elev, 1013.25, 0.0)
    if out["roof_closed"]:
        temp, rh = INDOOR["temp_f"], INDOOR["rh"]
    if temp is None or pd.isna(temp):
        temp = 70.0
    rho = air_density(float(temp), rh, p)
    out.update({"temp_f": round(float(temp), 1), "rh": round(rh, 0), "pressure_hpa": round(p, 1),
                "air_density": round(rho, 4), "density_vs_std_pct": round(100 * (rho / SEA_LEVEL_STD - 1), 1)})
    # wind on the field: roof closed -> none; else the hourly vector (exact compass direction) if the park's field
    # bearing is known, else MLB's reported direction
    src = wind or wind_source()
    rep_ok = row.get("rep_wind_mph") is not None and pd.notna(row.get("rep_wind_mph"))
    if out["roof_closed"]:
        out.update({"wind_mph": 0.0, "wind_field_deg": None, "wind_src": "roof closed"})
    elif src == "reported" and rep_ok:
        out.update({"wind_mph": float(row["rep_wind_mph"]),
                    "wind_field_deg": None if pd.isna(row.get("rep_wind_field_deg")) else float(row["rep_wind_field_deg"]),
                    "wind_src": "MLB reported (in park)"})
    elif hr is not None and park.get("cf_bearing") is not None:
        out.update({"wind_mph": round(float(hr["wind_mph"]), 1),
                    "wind_field_deg": round(field_angle((float(hr["wind_from"]) + 180) % 360,
                                                        float(park["cf_bearing"])), 0),
                    "wind_src": "hourly (10 m, open-terrain)"})
    elif row.get("rep_wind_mph") is not None and pd.notna(row.get("rep_wind_mph")):
        out.update({"wind_mph": float(row["rep_wind_mph"]),
                    "wind_field_deg": None if pd.isna(row.get("rep_wind_field_deg")) else float(row["rep_wind_field_deg"]),
                    "wind_src": "MLB reported"})
    else:
        out.update({"wind_mph": None, "wind_field_deg": None, "wind_src": "none"})
    if out.get("wind_mph") is not None and out.get("wind_field_deg") is not None:
        out["wind_out_cf_mph"] = round(along(out["wind_mph"], out["wind_field_deg"], 0.0), 1)
    return out


def carry_check(league: pd.DataFrame, season: int, wind: str | None = None) -> dict:
    """Does the weather explain carry? Well-hit fly balls (EV 95+, LA 22-38): distance vs the median for that EV / LA,
    against air density and the wind along each ball. Expect thinner air -> longer, tailwind -> longer."""
    b = league[league["description"].eq("hit_into_play")].dropna(
        subset=["launch_speed", "launch_angle", "hit_distance_sc", "hc_x", "hc_y"]).copy()
    b = b[(b["launch_speed"] >= 95) & b["launch_angle"].between(22, 38)]
    if b.empty:
        return {}
    b["evb"], b["lab"] = (b["launch_speed"] // 2) * 2, (b["launch_angle"] // 2) * 2
    b["resid"] = b["hit_distance_sc"] - b.groupby(["evb", "lab"])["hit_distance_sc"].transform("median")
    cond = {}
    for pk in b["game_pk"].unique():
        try:
            cond[pk] = conditions(int(pk), season, wind=wind)
        except Exception:
            cond[pk] = {}
    b["dens"] = b["game_pk"].map(lambda k: cond.get(k, {}).get("density_vs_std_pct"))
    spray = np.degrees(np.arctan2(b["hc_x"] - 125.42, 198.27 - b["hc_y"]))
    w = b["game_pk"].map(lambda k: cond.get(k, {}).get("wind_mph"))
    a = b["game_pk"].map(lambda k: cond.get(k, {}).get("wind_field_deg"))
    b["tail"] = [along(float(x), float(y), float(z)) if pd.notna(x) and pd.notna(y) else
                 (0.0 if pd.notna(x) and float(x) == 0 else np.nan) for x, y, z in zip(w, a, spray)]
    d = b.dropna(subset=["dens", "tail", "resid"])
    if len(d) < 200:
        return {"n": int(len(d))}
    X = np.column_stack([np.ones(len(d)), d["dens"], d["tail"]])
    coef, *_ = np.linalg.lstsq(X, d["resid"].to_numpy(), rcond=None)
    return {"n": int(len(d)), "ft_per_pct_density": round(float(coef[1]), 2),
            "ft_per_mph_tailwind": round(float(coef[2]), 2),
            "coors_minus_avg_ft": round(float(coef[1]) * (d.loc[d["dens"] < -12, "dens"].mean() - d["dens"].mean()), 1)
            if (d["dens"] < -12).any() else None}


def describe(c: dict) -> str:
    """'84°F · humid 61% · air 3.1% thinner than a 70°F sea-level day · wind 9 mph out to RF'"""
    if not c:
        return ""
    parts = [f'{c["temp_f"]:.0f}°F']
    d = c["density_vs_std_pct"]
    parts.append(f'air {abs(d):.1f}% {"thinner" if d < 0 else "thicker"} than a 70°F sea-level day')
    if c.get("roof_closed"):
        parts.append("roof closed")
    elif c.get("wind_mph") is not None:
        w, a = c["wind_mph"], c.get("wind_field_deg")
        if w < 1:
            parts.append("calm")
        elif a is None:
            parts.append(f"wind {w:.0f} mph")
        else:
            side = ("in from CF" if abs(a) >= 157.5 else "out to CF" if abs(a) < 22.5 else "out to RF" if 22.5 <= a < 67.5 else "out to LF"
                    if -67.5 < a <= -22.5 else "left to right" if 67.5 <= a < 112.5 else "right to left"
                    if -112.5 < a <= -67.5 else "in from LF" if a >= 112.5 else "in from RF")
            parts.append(f"wind {w:.0f} mph {side}")
    return " · ".join(parts)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "season":
        build_season(int(a[1]))
    elif a and a[0] == "parks":
        js = _get(f"{STATSAPI}/teams", {"sportId": 1})
        build_parks([t["venue"]["id"] for t in js.get("teams", []) if t.get("venue")])
    else:
        print(__doc__)
