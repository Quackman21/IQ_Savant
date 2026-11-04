# IQ Savant – version history

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
