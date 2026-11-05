# IQ Savant – version history

## v14 – 2026-11-05
- New "Defense" Summary tab (replaces separate Catching / Infield / Outfield tabs); underneath it,
  only the groups the player plays: Catching | Infield | Outfield.
- Infield (incl. 1B via Pos dropdown): Grades + Star tabs. Outfield: Grades + Star tabs.
  Catching: Overview, Framing, Blocking, Throwing. Same Periods picker / Season rows as Core 4.
- Filled now: MLB fielding counts for any period (#, Inn, PO, A, E, DP, SB/CS against), and on Season
  rows Statcast OAA, FRV, outfield jump (reaction / burst / route), catch probability by star,
  catcher framing (+ arm side / glove side / top / bottom percentiles), blocking & throwing runs,
  pop time, exchange, arm strength.
- Blank for our models: DRS family, OPA / FPA / CPA, every grade, infield Star tab, catcher
  blocking-by-location, Frame Speed Grade (reserved for the SAM 3 glove model).
- Level / Org columns kept for MLB/MiLB.
- Selected toggle is always shown (bold, dark); clicking the selected option no longer clears it.

## v13.2 – 2026-11-05
- Core 4: "Grade Scale" renamed "Evaluation Method" (20-80 | Percentile | Plus).
- Row order per season: Latest Week, Last 30 Days, months newest -> oldest, then the full Season row
  last -- bold with thick lines above and below.
- Grade columns left blank (no inputs) until our own grade models are built.
- No 20-80 cap: once grades are back, an extreme month can show 90+ (or below 20).

## v13.1 – 2026-11-05
- New: python -m pipeline.update -- one command that rebuilds only what's needed (missing seasons,
  seasons built with older stat definitions / models, and the current season during the year),
  then commits + pushes. Safe to stop (Ctrl+C) and re-run any time. --check lists what's out of date.
- Each league reference file now records which version of the definitions built it.
- Pages show one quiet note when MLB comparisons are missing or out of date.

## v13 – 2026-11-05
- Core 4 built out: Core 4, Bat Speed, Bat-To-Ball, Swing Decisions, Ball Flight tabs with the
  reference site's columns and hover definitions (proprietary Kinatrax / run-value columns replaced
  by public Statcast equivalents or left for our own models).
- Grades on a Grade Scale toggle: 20-80 (50 avg, 10/SD) | Percentile | Plus (100 avg, 10/SD).
  Inputs in core/core4.py GRADES (Bat Speed <- top 1/8 EV, SC Bat Speed <- top 1/8 bat speed,
  Bat-To-Ball <- Cont%, Decision <- GD%, Ball Flight <- xwOBACON).
- Swing decision placeholder model (models/decisions.py) -- swap for ours later like xwOBA.
- New Periods picker (compact button + scrollable panel): Default / Full Seasons / Curr Season,
  full seasons, and per season Latest Week, Last 30 Days, and each month. Same look on Core 4,
  Catching, Infield, Outfield and Baserunning (those use full seasons).
- Split: Aggregate (row per period) or Combined; Split By Team; Stat Category Regular/Postseason/both;
  Display Stats or Percentiles.
- Barrel% now uses the reference definition (xBA >= .500 and xSLG >= 1.500) everywhere.
- New catalog stats: K%, BB%, LD%, LD+FB%, LAHHB, Attack angle, fly-ball EV, bat speed, contact by
  zone (Z/H/S/C/W, hard/soft), chase by count, swing rates by attack zone, decisions, wOBACON.
- Rebuild the league reference with --force so the new stats get MLB averages.

## v12.1 – 2026-11-05
- Fix: pybaseball returns blanks as <NA>, which crashed the league reference build (and could
  crash Calendar / Player Profile). All Statcast now goes through core/statcast_clean.py first.
- League reference build no longer prints pybaseball's FutureWarning spam.
- League reference build pulls a week at a time with retries (Savant sometimes returns an error
  page instead of data), and skips seasons already saved so a re-run picks up where it stopped.

## v12 – 2026-11-05
- Player search filters as you type (first or last name): "Name · MLB | 2026 | POS | Team".
  "Search all levels" box underneath for minor leaguers.
- Hitting stat catalog (core/hit_metrics.py): every hitting stat's definition, formula, format and
  colour direction in one place. Definitions match the reference site's (Weak GB%, GB%, FB%,
  Weak FB%, Pull/Middle/Oppo, AirPull%, FB-Pull%, Pull-Air%, HHEV = top 1/8 EV, TRK = tracked BIP).
- xwOBA model slot (models/xwoba.py): ACTIVE = "savant" today; add your model and switch one line.
- Hitting tabs: Overview (Counts / Rates / Contact / BIP), Rates, Contact, Exit, Rolling, Spray.
  AAR removed (moving to Game Log later). Zone Grouping (In/Out of Zone or Heart/Shadow/Chase/Waste)
  and Split (Count Group / Platoon / Pitch Group, any combination).
- Hover a column name for its definition. Click a number for MLB AVG, player, Delta and STDEVS.
- Cells coloured by STDEVS (green/red for skills, yellow/purple for style stats).
- League reference builder: python -m pipeline.build_league_reference 2015 2026
  -> data/league/hitting_<season>.csv.gz (league average + spread across 100+ PA hitters).

## v11 – 2026-11-05
- Player Profile rebuilt: search any MLB/MiLB player, white header card with headshot, name,
  level | team, POS, H/W, STATUS, B/T, AGE (years.months), DOB, MLB DEBUT, BORN, team logo,
  and an ENTRY line ("2019 by BAL, Round 1, 1st Overall, Oregon State, Signing Bonus $8.1M" or
  "Signed by CLE out of Dominican Republic").
- Main tabs: Summary | Stats | Game Log | Sandbox.
- Summary sub-tabs depend on where the player has played (last 3 seasons, 10+ games):
  Hitting, Core 4, Catching, Infield, Outfield, Baserunning, Pitching.
- Hitting: Game/Season/Dates, Seasons, League, Stat Category, Level/Platoon split, Show Avgs;
  Overview, Rates, Contact, Exit, Rolling, Spray (with spray chart), AAR (placeholder).
- Core 4: Periods, Aggregate/By Season, Stats/Percentiles; Core 4, Bat Speed, Bat-To-Ball,
  Swing Decisions, Ball Flight.
- Catching: Overview, Framing, Blocking (placeholder), Throwing.
- Infield: Overview, Outs Above Average, Run Value. Outfield: Overview, OAA, Jump,
  Catch Probability, Directional.
- Baserunning: Advancing (placeholder), Base Stealing, Tools.
- Pitching: Overview, Arsenal, Results, Percentiles.
- Stats: year-by-year MLB/MiLB. Game Log by season. Sandbox: open space for later.

## v10 – 2026-11-03
- Fixed team cities: "Tampa Bay" not "St. Petersburg", "Texas" not "Arlington", etc.
- Calendar game cards: Box Score | Lineup | Play by Play links, each opening the game in a new tab.
- New Game page (/game?pk=…&view=…): switch between Box Score (line score, batting, pitching),
  Lineup (batting order, positions, bats, starting/probable pitcher) and Play by Play
  (by half-inning, click a play to see every pitch, "Scoring plays only" toggle).

## v9 – 2026-11-03
- Calendar page rebuilt: "15 MLB Games - 9/1/2026" title with the date picker on the right
  (same style as Standings); opens on the most recent day with games.
- Day-of-competition summary from Statcast: Top Exit Velocity, Top Distances, Top Pitch Velocity,
  Swing & Misses (team logo + headshot), and an All Games totals row.
- Game list: status tag (Final / Live / start time), both teams with logos, score (winner in red).
  No menus under the scores yet.

## v8 – 2026-11-03
- Profile is now a dropdown in the left menu with Player Profile and Team Profile inside
  (removed the buttons at the top of the page). Each profile is its own page.

## v7 – 2026-11-03
- Left-hand menu: Standings (home), Calendar, Leaderboard, Profile.
- Profile opens on a choice between Player Profile and Team Profile.
- Calendar: pick a day, see its games, open the Statcast game report (replaces Game Reports).
- Leaderboard: starter exit velo / barrels board by season.
- Player Profile: starter (replaces the Hitter page). Team Profile: starter with record and last 10 games.
- Requires Streamlit 1.40 or newer.

## v6 – 2026-11-03
- Standings title is one line: "2026 Division Standings". Click "Division" to switch to Wild Card (and back); same font as the rest of the title.
- View and date are kept in the page address (?view=wildcard&date=2026-06-01), so a view can be bookmarked or shared.
- Offseason / after the last regular-season game: shows final standings by default.
- Added this changelog and a VERSION file.

## v5 – 2026-11-03
- Division and Wild Card merged into one Standings page with a toggle.
- Season year in the title; calendar (far right) to view standings as of any date back to Apr 5, 2015.

## v4 – 2026-11-03
- Congrats card: trophy + series record (e.g. 4/5) + winner's logo, for every series clinch.
- Clinch labels: w = Wild Card, y = Division, z = Division & Best Record.
- Division and Wild Card standings pages with W, L, PCT, GB, E#, L10, STRK, RS, RA, DIFF, X-W/L, HOME, AWAY, ≥.500, NEXT GAME.

## v3 – 2026-11-03
- Scoreboard shows on the home (Standings) page only.

## v2 – 2026-11-03
- Postseason scoreboard: last 5 postseason games in order, Off Day box on days without games.

## v1 – 2026-11-03
- Scoreboard strip: top 5 games by importance, switches to today's games at 10:30 AM ET, red stars for clinch chances.
