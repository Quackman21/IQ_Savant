# IQ Savant – version history

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
