# IQ Savant – version history

## v20.1.2 – 2026-11-13
- FIX (important): 2026 framing was using the ABS-OVERTURNED calls, not the umpire's original calls. The challenge
  data loaded, but writing the challenger ids crashed under pandas 3 ("Invalid value ... for dtype float64"), and
  framing quietly fell back to the calls as recorded. Ids are now cast to numbers; a test reproduces the crash.
  Re-score with:  python -m pipeline.defense_study  (rebuilds every season's framing + fielding files, ~6 min).
- fetch_drs reads FanGraphs' own leaderboard feed (pybaseball's FanGraphs reader comes back with only an IDfg
  column since their redesign).

## v20.1.1 – 2026-11-13
- python -m pipeline.fetch_drs: saves SIS DRS (as published on FanGraphs, with rPM range/positioning and rGFP good
  plays/misplays) to data/compare/drs_<year>.csv for 2016-now, for the fielding comparison study. Fieldingbible.com
  has no download; if FanGraphs blocks the script it prints the copy-paste fallback.

## v20.1 – 2026-11-13
- DRS is now labeled DRS* everywhere (with a disclaimer on hover + the page note): it's IQ's estimate, built the
  way DRS is from public Statcast, without fielder positioning or SIS's video-charted good plays / misplays.
- Fielding grade scale fix: v20.0 divided by the spread of RAW season rates, which still includes the noise the
  regression already removed, so grades got squeezed twice (one infield season topped out ~57). Grades now use the
  spread of TRUE talent ('talent' scale). The comparison study prints raw / talent / rank so you can pick.
- python -m pipeline.fielding_compare_study: ours vs Statcast OAA (downloaded + cached) vs SIS DRS (any
  data/compare/drs_<year>.csv you save) -- same-season agreement, which predicts next season best (each judged on
  the others, not itself), the best ours/OAA Range blend, the three grade scales with names, and the biggest
  disagreements. Pushes results to GitHub and opens the report.

## v20.0.1 – 2026-11-13
- Fix: defense study crashed scoring framing on the PC ("operation 'sub' not supported for dtype 'str'"). A
  2015-2022 season has no 2023+ rows, and that empty slice came back as text on some pandas 3 builds. Empty eras are
  now skipped and every count value is cast to a number before subtracting.

## v20.0 – 2026-11-13
- Fielding tool grades (20-80, regressed for sample size, hover for the why) -- only what public data MEASURES:
  * Infield Range / Conversion / Ball Security (our fielding model, any period). Range + Conversion = OPA:
    Conversion = finishing balls he fielded vs an average fielder on that same ball (a 1B's dropped good throw isn't
    charged to the thrower); Ball Security = fielding errors vs expected (throwing errors excluded).
    New models in pipeline.train_fielding (out chance once fielded, fielding-error chance).
  * Arm Strength (IF + OF) from Statcast's arm strength board; OF Reaction / Acceleration / Direction / Jump from
    Statcast's jump board (full seasons); OF Range from our CPA.
  * Left BLANK on purpose (no stand-ins): First Step, Arm Accuracy, 1B Catch, RS 1B Rec, OF Conversion / Max Spd /
    XBH Prev / Reputation / Transfer / Momentum, catcher Accuracy. Hover says why.
- Framing: every pitch now also stores SITUATIONAL runs (count + runners + outs, from the decision model's tables)
  for after-action reports. The profile keeps count-valued runs (the skill / projection number).
- python -m pipeline.defense_study: re-scores every season's fielding + framing files, measures how many chances
  each grade needs (feeds the regression), year-to-year repeatability, eye test, and count vs situational framing
  (which predicts next season). Pushes results to GitHub and opens the report.

## v19.9 – 2026-11-13
- Ball Flight Grade: Pull-Air% removed (it double-counted AirPull%). New mix: AirPull% 45% (still the gold star) |
  Air% 30% | launch-angle consistency 25%. Predicts next season's ball-flight value better than v19.8 in both
  2015-21 and 2022-25; power leak stays ~0 (r with 90th EV -.03, bat speed -.02). Rebuild refs:
  python -m pipeline.flight_ref --all

## v19.8 – 2026-11-13
- BALL FLIGHT GRADE is live -- the Core 4 is complete. How well he steers the ball, with exit velo kept OUT:
  AirPull% 40% | Air% (10-50 deg) 25% | Pull-Air% 15% | launch-angle consistency 20%. Angle + direction only (no
  exit-velo cut-offs, no xwOBA), so bat speed and contact can't leak in. Weights = the study's AirPull-heavy mix
  (within ~.01 of the data's best fit). Each piece regressed by its own BIP-for-.70 (210 / 310 / 390 / 410), graded
  vs MLB-qualified hitters; hover shows each ingredient's grade + his actual rate.
- python -m pipeline.flight_ref (refs for every season, --top 2025 2026 eye test with Raleigh + PCA called out);
  pipeline.update rebuilds it with the other grades.

## v19.7 – 2026-11-13
- Ball Flight study keeps the other grades OUT of ball flight:
  * FLIGHT PERCENTILE (new core candidate): each ball's angle + direction ranked among all balls hit exactly that
    hard, so power can't magnify it (in the simulation: far steadier than flight runs, ~120 vs ~520 BIP to trust).
  * Angle-only versions of FB% / GB% / pop-up% / AirPull% / Pull-Air% (no exit-velo cut-offs, which let power
    back in), Sweet Spot% (8-32 deg), LA consistency.
  * Leak checks vs the other grades' ingredients (90th EV, bat speed, contact, chase) for every metric.
  * Every test now holds power, contact and chase equal, so a metric only gets credit for what the other three
    grades can't explain.

## v19.6 – 2026-11-13
- python -m pipeline.flight_study (analysis only): how much of a hitter's run value comes from his BALL FLIGHT
  with bat speed / exit velo taken out. FLIGHT RUNS = each ball's SAxwOBA minus an average ball hit exactly that
  hard. Where flight runs come from (batted-ball type x direction), every metric on its own (flight runs + the
  app's 11: GB%, FB%, WEAK FB%, LD%, LD+FB%, PULL%, MIDDLE%, OPPO%, AIRPULL%, FB-PULL%, PULL-AIR%, + LA on hard-hit
  balls) vs next season's actual flight value and runs with tools held equal, whether the 11 add anything, the
  data's weights, eye test. Pushes its results to GitHub when done.

## v19.5 – 2026-11-12
- Decision Grade + 0-0 / <2k / 2k / FB / BB / OS Decision Grades are live (Core 4 + Swing Decisions tabs), built on
  the RUN VALUE of decisions (decision study): DEC RUNS/100 55% | chase + waste swing% before 2 strikes 15% (low) |
  with 2 strikes 15% (low) | heart-minus-chase gap 15%. Swing rate never counts. Splits use the same mix inside the
  situation (no 2-strike piece for 0-0 / <2k, no early piece for 2k). Regressed by each split's sample (from the
  study); graded vs MLB-qualified hitters in that same split; blank under a minimum. Hover for each ingredient.
- python -m pipeline.decision_ref builds data/league/decision_<year>.json (--top <year> = eye test);
  pipeline.update keeps the current season fresh.

## v19.4 – 2026-11-12
- python -m pipeline.decision_study (analysis only), built around the RUN VALUE of swing / take decisions:
  where each hitter's decision runs come from (zone x swing / take, heart takes in hitter's counts, chase + waste
  swings before / with 2 strikes, by count, by pitch type -- typical value, real spread, share of the real
  differences), every metric on its own vs NEXT season's runs with tools held equal (power + contact), walks and
  strikeouts, GD% vs DEC RUNS/100 as the core, 2-strike chase weighting (equal / heavier / lighter), the
  heart-minus-chase gap, the data's own weights, split stability (sets their regression), eye test.
  Pushes its report + hitter file to GitHub when it finishes (--no-push to skip).

## v19.3 – 2026-11-12
- Bat-To-Ball Grade is live (Core 4 + Bat-To-Ball tabs). Contact WHEN HE SWINGS -- takes never count:
  Cont% over expected 45% | Z-Cont% over expected 20% | Cont% 15% | Heart Cont% 10% | Chase Cont% 10%.
  Over expected = each swing vs how often an average hitter makes contact on that exact pitch (league contact
  model: location, pitch type, velo, movement, count, hands), so a whiff over the heart (~89% league contact)
  costs ~4x a whiff on a waste pitch. Regressed for sample size; 50 = average MLB-qualified hitter; hover a grade
  for each ingredient's score. Chosen from the Bat-to-Ball study: within ~.01 of the best mix the data found,
  well ahead of zone contact alone.
- python -m pipeline.train_contact: trains the league contact model (2015-2022 + 2023+, tested on held-out games
  first) and builds data/league/contact_<year>.json references. --top <year> = eye test. pipeline.update keeps the
  current season's reference fresh.

## v19.2 – 2026-11-12
- python -m pipeline.contact_study (analysis only): which contact metrics measure bat-to-ball skill -- the 8 app
  metrics (Cont%, Z-Cont%, Z-Cont% Hard / Soft, Heart / Shadow / Chase / Waste Cont%) plus Cont% and Z-Cont% over
  expected (league contact model: location, pitch type, velo, movement, count, hands) and Squared-Up%. Targets:
  next season's strikeouts avoided, whiffs avoided, and wOBA with power held equal. Stability, overlap, step test,
  weights, stability of weights, eye test -- by era.

## v19.1.1 – 2026-11-12
- Windows PC support: tools\windows\setup_pc.ps1 (one time: Python env, installs, tests, Desktop shortcut with the
  IQ icon) and tools\windows\IQ_Savant.ps1 (the shortcut: sync with GitHub, start the site, add new games every
  3 hours; Open / Restart / Stop when it's already running). tools/IQ_icon.ico.
- requirements.txt pins scikit-learn==1.7.2 (the trained models were saved with it; newer versions can't load them).
- overnight.py: the "already running?" check no longer ends the other job on Windows.

## v19.1 – 2026-11-12
- Bat speed now matches Statcast's method:
  * Competitive swings only: no bunts, and each hitter's slowest tenth of swings dropped (check / defensive swings),
    except any 60+ mph swing hit 90+ mph. Applies to players and the league alike.
  * MLB average for Avg Bat Speed / Top 8th Bat Speed / Swing Length = the average MLB-QUALIFIED hitter
    (3.1 PA per team game: 502 in a full season, 186 in 2020), each hitter counted once -- not every swing pooled.
- Bat Speed Grade + SC Bat Speed Grade: the league (50 and the spread) is now MLB-qualified hitters too, which
  also fixes 2020 (the cutoff scales with the season).
- Needs a full rebuild of the league files: caffeinate -i python -m pipeline.update

## v19.0.3 – 2026-11-12
- FIX: core/hit_metrics.py (SAxwOBA / SAxwOBACON columns) and its test were never in a zip, so GitHub and the Mac
  had the version without them (KeyError: SAxwOBA on Core 4). Whole project checked against GitHub: these were
  the only two files out of step.

## v19.0.2 – 2026-11-12
- FIX: the desktop app installed EVERY IQ_Savant_v*.zip in ~/Dev, including old leftovers, which overwrote newer
  code (SAxwOBA KeyError: model). Now it only installs versions newer than the one installed; older zips are
  moved to ~/Dev/old_zips.

## v19.0.1 – 2026-11-12
- Wording: it is the Bat Speed Grade everywhere (no "power" in anything you see).

## v19.0 – 2026-11-12
- Bat Speed Grade (power) and SC Bat Speed Grade are live (Core 4 + Bat Speed tabs), weights from the power study:
  bat speed 50% (avg 25, top 8th 25) | EV on fly balls 30% (avg 15, 90th 15) | 90th EV + HHEV 10% |
  avg EV + EV>95% 6% | max EV + max fly EV 4% (BIP-adjusted). No bat tracking -> the bat speed half goes to
  90th EV + HHEV. Contact-skill metrics (squared-up, smash, blasts) left out (they're Bat-To-Ball).
  SC Bat Speed Grade = avg + top 8th Statcast bat speed. EVs adjusted for pitch speed; every ingredient regressed
  for sample size; 50 = MLB regular average, 10 = one spread. Hover a grade to see each ingredient's grade.
- models/power.py (shared with the power study, identical definitions), data/league/power_<year>.json references
  (python -m pipeline.power_ref; pipeline.update keeps the current season fresh).
  Eye test: python -m pipeline.power_ref --top 2026

## v18.9 – 2026-11-12
- python -m pipeline.power_study (analysis only): which bat speed / exit velo / application metrics predict next
  season's run value (on contact, ISO, per PA); stability, overlap, step test (bat speed -> application -> exit
  velo, and reverse), ridge weights (positive, cross-validated by hitter), two-part bat speed + application
  version, weight stability by season pair, eye test, and an exit-velo-only version by era. EVs adjusted for
  pitch speed; max EVs adjusted for number of balls in play.

## v18.8 – 2026-11-12
- Desktop app syncs on every launch: installs any IQ_Savant_v*.zip in ~/Dev (oldest first, then deletes it),
  runs the tests, commits local changes, downloads the latest from GitHub (keeping local changes on top) and
  pushes. Tests fail -> nothing is pushed and a notification says so; the site still starts. Runs from a copy of
  itself so a new version can safely replace the launcher. Manual: bash tools/iq_app.sh sync

## v18.7.1 – 2026-11-12
- App icon: rounded macOS-style IQ icon (tools/IQ_icon.png) -- straight sides, rounded corners, Apple margin, soft
  shadow. make_mac_app.sh uses it by default; give it a new logo path and it rounds that one (tools/make_icon.py).

## v18.7 – 2026-11-12
- Desktop app: bash tools/make_mac_app.sh builds "IQ Savant" on the Desktop with the IQ logo
  (~/Dev/images/IQ RB Paisley Logo.png). Double-click: pulls the latest code, starts the site, opens it.
  While the site is open it checks every 3 hours and adds new games (python -m pipeline.update --light);
  full rebuilds are never started on their own -- a Mac notification says when one is due.
  Clicking it again while running: Open / Restart / Stop. Log: data/app_log.txt.

## v18.6.1 – 2026-11-11
- Savant framing download fixed and CHECKED: Savant moved the board (new columns id / pitches / rv_tot) and some
  addresses ignore the year and send the current season. Every season is now verified against who actually caught
  that year (same catchers + matching pitch counts) before it is used, saved in data/compare/, and real errors are
  shown. Slows down and retries if Savant throttles.

## v18.6 – 2026-11-11
- ABS (2026+): framing now uses the umpire's ORIGINAL call. Challenges are a different skill and are kept out.
  core/abs_calls.py reads which pitches were challenged / overturned from MLB's play-by-play (only new games are
  fetched), works out the call that stood from the next pitch's count, and flips overturned calls back.
  python -m core.abs_calls --inspect 2026 shows how MLB marks a challenge (to double-check the fields).
- Framing strike chances are centered each season (league = 0 extra strikes). 2025 was -5,263 and 2026 +3,120,
  which made catchers' numbers not comparable across seasons.
- New: SLAA (regressed) -- best read of framing skill: Extra Strikes / (Framing Opps + 720 before 2023, 1,080 after).
- Framing study: original calls, centering, regressed SLAA, an ABS section (challenges by catcher / pitcher /
  batter, strikes ABS took away, catchers' own challenge results), Savant 2023+ columns fixed.
- data/framing_study/ no longer pushed to GitHub. FutureWarning fixed.

## v18.5 – 2026-11-10
- Framing study: SIS Strike Zone Runs Saved column (data/compare/sis_framing.csv, matched by name) next to Savant.

## v18.4.1 – 2026-11-10
- Fix: blocking_videos crashed on a single-era season (tried to score the empty 2015-2022 half).

## v18.4 – 2026-11-10
- python -m pipeline.framing_study (analysis only, nothing on the site changes): which framing number is most
  accurate (split-half, takes needed, 1st half -> 2nd half, next season, Savant agreement) -- raw extra strikes,
  shadow only, per game, SLAA+, count-valued vs flat 0.125 runs, and ADJUSTED versions that split each call
  between catcher, pitcher, umpire and batter. Zone importance (heart, top, bottom, glove, arm, 4 corners, waste;
  also by batter hand): reliability, true spread, runs at stake, share of real framing differences, next-season
  weights. Downloads each game's home plate umpire once (data/raw/umps_<year>.csv.gz).

## v18.3 – 2026-11-10  (includes v18.1 + v18.2)
- Catcher blocking rebuilt the way Statcast does it:
  - Opportunity = every pitch with a runner on that isn't put in play / HBP / pitchout / intentional / automatic.
  - WP + PB chance per pitch for an average catcher (location, height, pitch type, velo, movement, swing, pitcher
    AND batter hand, count). Blocks Above Average = sum of (chance - what happened). Dropped third strikes that
    go to the backstop now count as wild pitches / passed balls.
  - Tiers on Statcast's lines: Easy < 5%, Medium 5-15%, Tough 15%+. Genuine opportunities = Medium + Tough.
    Easy catches still earn their tiny credit (as Statcast does).
  - Blocking RS (into DRS) is SITUATIONAL: each pitch's credit x what a wild pitch costs in that base / out state
    (from run expectancy, by era). Statcast's flat 0.25-run version shown next to it (Blocking RS (Statcast)).
  - Big Stop runs: the value of every Tough pitch that was blocked.
  - Zones by location only, catcher's view, every catcher right-handed: glove side = catcher's left, arm side =
    catcher's right, middle, and air.
- python -m pipeline.blocking_videos: clickable video check around the 5% line, or any tier / result / chance range.
- compare_fielding now also compares our Blocks Above Average and blocking runs with Savant's catcher blocking board.

## v18.2 – 2026-11-10  (includes v18.1)
- Fielding fix: distance now comes from Statcast's PROJECTED distance (hit_distance_sc, balls in the air) instead
  of the hit coordinates. The coordinates mark where a hit was picked up after rolling, which leaked the result
  and made the out chance near-certain -- shrinking every fielder's credit (our DRS ran about half of SIS DRS).
- python -m pipeline.compare_fielding [--season 2026]: our full-season DRS vs Savant Outs Above Average and
  Fielding Run Value by position (correlation + scale), and vs SIS DRS if you save their leaderboard as
  data/compare/drs_<season>.csv (player, position, DRS).

## v18.1 – 2026-11-10
- Catcher blocking + throwing for ANY period (models/catching.py), era models:
  - Blocking: every pitch with runners on gets a wild pitch / passed ball chance for an average catcher
    (location vs the zone, pitch type, velo, movement, swing, pitcher hand, count). Blocking runs =
    (chance - actual) x 0.28. Location percentiles: Air, Dirt arm side / middle / glove side.
  - Throwing: every steal attempt gets a caught-stealing chance (base, pitch type, velo, location, pitcher hand,
    batter side, outs, count; no runner speed). Throwing runs = (caught - chance) x out-vs-SB run swing.
  - Catching DRS = framing + blocking + throwing on every row (month, last 30 days...), plus all RS / Season rates.
- Play-by-play runner events (MLB stats API, free): python -m pipeline.fetch_events -> data/raw/events_<year>.csv.gz
- python -m pipeline.train_catching ; fielding files now include catcher blocking / throwing.
- Overnight job also downloads events and trains catching.
- Still Statcast full seasons: pop time, exchange, arm strength. Still blank: Accuracy Grade (throw location not public).

## v18.0 – 2026-11-10
- IQ fielding (models/fielding.py): DRS and its parts from public pitch data, any period, era models.
  Out chance per ball (EV, LA, direction, distance, side); responsibility = who makes the out on balls like it;
  OPA / CPA, FPA, RS Field / RS Range (OPA x out-vs-hit run value), DP runs, RS Arm (expected - actual run value
  on balls an outfielder fields with runners on), stars by out / catch chance. No batter speed (USE_SPEED).
- Defense page: Infield DRS = RS Field + DP runs; Outfield DRS = RS Range + RS Arm; DRS / Season per 1,200
  innings + MLB percentiles; our stars (any period) replace Savant's; catcher DRS = framing + Statcast blocking
  + throwing (season rows), every RS / Season rate, Blocking Opps (pitches in the dirt with runners on), Opps Total.
- Still blank (need player-tracking data that isn't public): tool grades, RS 1B receiving, range vs conversion
  splits, catcher blocking by location, Frame Speed Grade.
- python -m pipeline.train_fielding ; the data update now also saves data/league/fielding_<year>.csv.gz.
- pipeline.overnight also trains fielding and Stuff+ / Location+.

## v17.0.1 – 2026-11-10
- Test fix: the placeholder swing-decision test now pins the placeholder model, so it passes when IQ Swing
  Decisions is switched on (it failed on the Mac after the overnight job turned the real model on).

## v17.0 – 2026-11-10
- IQ Stuff+ and Location+ (models/pitch_quality.py), after Driveline's and FanGraphs' approach. Both predict
  each pitch's run value from the pitcher's side (balls in play use expected value of the contact, not the result).
  - Stuff+: velo, IVB, HB (arm side +), spin, extension, release height / side, arm angle, and velo / IVB / HB
    difference from the pitcher's primary fastball -- no location. 100 = MLB-average pitch, 10 = 1 SD.
  - Location+: inside / away vs the hitter, height in his zone, count, pitch type, same / opposite hand --
    no stuff. 100 = average for that pitch type, 10 = 1 SD.
  - Era models (2015-2022, 2023+), each pitch scored by its own era's model.
- python -m pipeline.train_pitch_quality : trains both on every saved pitch (no download), report: averages by
  pitch type, half-to-half repeatability, how well 1st-half numbers predict 2nd-half run value, leaderboards.
- Pitching > Summary: Stuff+ and Location+ columns per pitch type (blank until trained).

## v16.9.1 – 2026-11-10
- Fix: two overnight jobs ran at once and fought over the same temp download file (FileNotFoundError on
  rename), so SAxwOBA / decision training never finished. Now: a lock stops a second overnight job from
  starting; each job writes its own temp file; a season another job already saved is reused.
- Overnight job raises git's upload limit (http.postBuffer 500 MB) -- the data push failed with HTTP 400.

## v16.9 – 2026-11-10  (includes v16.8)
- IQ framing (models/framing.py iq_v1, now active): every taken pitch a catcher received, strike chance from
  the IQ Swing Decisions called-strike model (league-average umpire; location vs the hitter's zone, count,
  batter side, pitch type; era models). Falls back to the placeholder until the decision model is trained.
- New catching metrics (Defense > Catching > Framing, hover for definitions):
  Extra Strikes = sum of (1 if called strike / 0 if ball) - strike chance;  Extra Strikes / Game;
  SLAA = Extra Strikes per framing opportunity;  SLAA+ = 100 x called strikes / expected (100 = average);
  Framing Runs = each extra strike x what a strike instead of a ball is worth in that count.
- Catching data rebuilds with the new model on the next pipeline.update (included in pipeline.overnight).

## v16.8 – 2026-11-10
- python -m pipeline.overnight : the whole overnight job in one command -- git pull, download every pitch
  2015+ once, train SAxwOBA, train + switch on IQ Swing Decisions, push the models, rebuild MLB averages, push.
  Logged to data/overnight_log.txt; a failed step is logged and the rest still run.
- Download once, reuse: finished seasons are saved whole in data/raw/statcast_<year>.csv.gz (~2-3 GB, not
  pushed). The MLB-averages update, SAxwOBA and the decision model all read them -- no second download.
- Era models, no carryover: SAxwOBA and IQ Swing Decisions each train one model on 2015-2022 (shift era) and
  one on 2023+ (shift ban / pitch clock). Every ball / pitch is graded by its own era's model.
- SAxwOBA leaves filled-in tracking data out of training (same exact EV + LA 50+ times in a season --
  Statcast's defaults for missed balls, mostly 2015-2019). Report shows each era separately.
- Decision trainer: --since 2015 (multi-season), loads compactly so 8+ million pitches fit on a laptop.

## v16.7 – 2026-11-10
- Two expected stats side by side. xwOBA / xwOBACON = Savant's (the regular one, unchanged).
  SAxwOBA / SAxwOBACON = ours, spray-adjusted: exit velo + launch angle + spray angle, nothing else
  (no sprint speed, batter side, park adjustment, blur, blend or calibration). models/saxwoba.py.
- python -m pipeline.train_saxwoba : trains it on the saved 2023+ batted balls (minutes), tests it on seasons it
  didn't see (log loss, per-ball error vs Savant, league check, future r), saves models/artifacts/saxwoba.joblib.
- Columns: Hitting > Contact (xwOBA, SAxwOBA, xwOBACON, SAxwOBACON); Core 4 shared columns (SAxwOBA) and
  Ball Flight (SAxwOBACON). Blank until trained. Retraining it marks MLB averages out of date.

## v16.6 – 2026-11-09  (includes v16.5.x)
- python -m pipeline.train_xwoba --best : builds the best spray-adjusted IQ xwOBA by a rule fixed in advance.
  - Spray strength: IQ xwOBA = no_spray + strength x (spray version - no_spray). EV / LA always lead, spray
    adjusts. Candidates air, fly, hr_dir, blur18 at 25 / 50 / 75 / 100% (blends cost no retraining).
  - New per-ball accuracy column BALL ERR (typical gap between a ball's xwOBA and the wOBA it produced).
  - The rule: pull-air credit >= +0.10 AND not reliably worse than no_spray at predicting (gap range reaches 0);
    among those, lowest BALL ERR. Trains the pick on every season and saves it (not switched on unless --activate).
  - Opens the batted-ball video check automatically.
- python -m pipeline.xwoba_videos : outfield balls in the air, every version's xwOBA + Savant's next to what
  happened, with Watch links. Filters: --result double, --direction pull, --min-ev 100, --n 40.
- python -m pipeline.train_xwoba --use-saved --activate : switch on the saved model without retraining.
- --variant accepts strengths, e.g. --variant air@0.5

## v16.5.5 – 2026-11-09
- IQ xwOBA version hr_dir: direction decides ONLY home run vs not. A second, park-neutral model gives
  P(home run) using spray angle on balls in the air (a pulled 105 / 25 clears the fence, to center it's
  caught); out / 1B / 2B / 3B come from the no-direction model, rescaled. Direction never touches
  singles / doubles, where gaps and holes leak the result.
- by_type results (real data): future r 0.486, gap to no_spray -0.043 (range -0.060 to -0.025), pull-air
  credit only +0.05 -- not adopted. Default training version back to no_spray until hr_dir is tested.

## v16.5.4 – 2026-11-09  (includes v16.5.3)
- IQ zones (models/decisions_iq.IQ_ZONES), the coaching tiers: Heart (center -> 50% of the way to the edge),
  Sub-heart (50-80%, the ring around the heart), Edge (80% -> the zone's edge, inside), Edge (off) (ball still
  touching the zone), Chase, Waste. Savant's Shadow straddles the edge, so it mixes sub-heart, edge and off.
- Report tier check now by IQ zone. Video page shows each pitch's IQ zone; decision_videos --zone filter.

## v16.5.3 – 2026-11-09
- Swing decisions: before 2 strikes a swing is judged on DAMAGE (your coaching rule: heart = swing,
  sub-heart = swing to a lesser extent, edges = take). Contact counts only if damage-quality (expected
  xwOBA >= .500, valued at what damage contact is worth in that count); weak contact is valued as an
  out. New damage-chance model per pitch. With 2 strikes contact is still valued realistically.
- Report: tier check -- share of swings / takes graded good in Heart / Shadow / Chase / Waste for 0-0,
  before 2 strikes, and 2 strikes.

## v16.5.2 – 2026-11-09
- python -m pipeline.decision_videos: Savant video links for graded swing decisions. Looks up each pitch's
  playId in MLB's game feed (one call per game) and opens a clickable page
  (models/artifacts/decisions_video_check.html); falls back to the game page if a video ID isn't found.
  Filters for a new sample: --count 0-0 --decision take --grade bad --n 40.

## v16.5.1 – 2026-11-09
- Fix: train_decisions crashed writing the report on a fresh download (pandas <NA> types in the
  attack-zone step). Pitches now go through statcast_clean.clean() first, like the rest of the site.

## v16.5 – 2026-11-09  (includes v16.3 + v16.4)
- IQ xwOBA version by_type (now the default): park-neutral, swing only, direction blurred by ball type --
  exact spray on fly balls (LA 25+), +-12 deg on liners (10-25), +-18 deg on grounders (under 10).
  Keeps the pull-air and oppo-ground skills, drops precise placement between fielders.
- Compare report: "How reliable are the FUTURE r numbers?" -- 95% ranges from 1,000 resamples of the
  hitter-seasons, each version's gap to no_spray with its own range (gap range not crossing 0 = real
  difference), and future r season by season.
- One switch for xwOBA everywhere: the hitting summary and the rolling wOBA / xwOBA chart now use the
  active xwOBA model too (they read Savant's number directly before). Every xwOBA / xwOBACON on the site
  (tables, Core 4, MLB averages, xwOBA MLB / xwOBACON MLB, Ball Flight 600, rolling chart) follows
  python -m pipeline.train_xwoba --activate.

## v16.4 – 2026-11-09  (includes v16.3)
- Three decision numbers per hitter, on Core 4 > Swing Decisions (hover for definitions):
  - GD%: how OFTEN he made the right call (good / (good + bad), coin flips left out).
  - DEC QUAL (Decision Quality): HOW right or wrong -- average count-adjusted decision score x100, each
    pitch capped at +-150, neutral pitches included.
  - DEC RUNS and DEC RUNS/100: runs gained / lost vs what a league-average hitter would choose on the
    same pitches (new league swing-rate model). Not count-adjusted: impact.
  DEC QUAL / DEC RUNS stay blank on the placeholder model.
- Called-strike model now also uses pitch type (fastball / breaking / offspeed).
- Report adds: called-strike check (model's chance vs how often those pitches were really called strikes),
  repeatability + BB% / K% links for all three numbers, GD% vs Decision Quality overlap, best / worst 5
  hitters by Decision Runs. Video-check file adds the model's strike chance and average swing chance.

## v16.3 – 2026-11-09
- IQ Swing Decisions v1 (models/decisions_iq.py): every swing / take graded good / bad / neutral for a
  LEAGUE-AVERAGE hitter (no hitter strengths), by expected runs -- never by what happened.
  - Count sets the stakes (ball / strike / contact values from real run values per count); contact is
    measured against the count you're already in (3-0 contact must beat a great count, 0-2 almost any helps).
  - Location sets the odds: called-strike chance (league umpire, count-aware zone), miss / foul / in-play
    chance and expected contact quality (from the site's xwOBA model) for that pitch.
  - Fouls: before 2 strikes a foul is a strike (no credit); with 2 strikes a foul keeps the at-bat alive.
    Caught foul tips = strikes. Bunts, pitchouts, IBB and pitch-clock automatic balls / strikes not graded.
  - Runners + outs at 50% strength, only on close calls: a clear count + location call can't be flipped.
  - Grade relative to what was at stake in the count; narrow neutral zone for coin flips (no grade).
- python -m pipeline.train_decisions  -- downloads one season of every pitch (step-away job, saved in
  data/train/), tests on half the games with a model trained on the other half, report: count values,
  grade mix by count, flip report, repeatability vs the placeholder, link to BB% / K%, plus
  decisions_video_check.csv (20 random graded pitches to check on video). --activate switches GD% stats.
- Retraining the decision model now marks MLB averages out of date (league reference signature).

## v16.2 – 2026-11-09
- Five more IQ xwOBA versions that keep spray angle as a skill without crediting the holes:
  fly (exact spray on true fly balls, LA 25-45), fly_hard (95+ mph at LA 20-45), and blur8 / blur12 /
  blur18 (spray on every ball, but each ball is valued across +-8/12/18 degrees of its angle and
  averaged, so narrow holes between fielders wash out while pull-air / oppo-ground value stays).
- Compare report: new "Direction as a SKILL" table -- is each hitter's direction credit repeatable
  (1st half vs 2nd half), does it line up with pulled-air-ball rate and oppo-grounder rate, and how big is it.
- Default version for a plain training run is now no_spray (best so far) until the comparison picks one.

## v16.1 – 2026-11-09
- IQ xwOBA versions you can compare and choose from (models/xwoba_iq.VARIANTS). Still swing only, no sprint speed.
  - Park-neutral, done properly: the ballpark is an input while training (so Fenway's wall / short
    porches get blamed on the park, not on EV / LA / direction); when scoring, every ball is valued in
    every MLB park and averaged equally. Parks come from MLB's schedule (one small call per season).
  - Spray: exact angle only on balls in the air (LA >= 10); every ball also gets pull / middle / oppo
    (+-15 deg). On ground balls Statcast's hit coordinate is where the ball was fielded, so the exact
    angle mostly records whether it found a hole.
  - Calibration: each season's league IQ xwOBACON is set equal to the league's actual wOBACON.
- python -m pipeline.train_xwoba --compare  tests every version side by side (no download beyond the
  ballpark list): future r, same-season r, log loss, league drift, vs Savant and actual, plus a
  ground-ball "hole check" chart (models/artifacts/xwoba_gb_spray_check.png).
- python -m pipeline.train_xwoba --variant NAME [--activate]  trains the chosen version (default air_dir).

## v16 – 2026-11-09
- IQ xwOBA v1 (models/xwoba_iq.py): every batted ball valued from batter side + exit velo + launch
  angle + pull-adjusted spray angle -> P(out / 1B / 2B / 3B / HR) x that season's standard wOBA weights.
  Park-neutral (trained across all 30 parks, park never used), no sprint speed. Walks / HBP / Ks keep
  their actual season value. Reached-on-error = out, sac flies counted, sac bunts / IBB excluded.
- Training: python -m pipeline.train_xwoba  -- downloads batted balls only (2023 -> present, kept in
  data/train/ so re-training is instant), tests each season on a model that never saw it, compares with
  Savant's xwOBA, saves models/artifacts/xwoba_iq_v1.joblib + a report.
  --activate switches the whole site to IQ xwOBA; --use-savant switches back.
- Per-ball scoring ready for later (xwoba_iq.score_batted_balls works on any game / player / season).
- Retraining or switching models marks the MLB averages out of date until the next pipeline.update.

## v15.1 – 2026-11-05
- Pitching opens on the current season's averages only (one row per pitch); months, postseason and
  earlier seasons are still in the Periods panel.
- Pitching gets the Evaluation Method toggle (20-80 | Percentile | Plus) like Core 4. Grade columns
  stay blank until our pitch models exist; their hover text follows the chosen method.

## v15 – 2026-11-05
- Pitching rebuilt to match Hitting / Core 4 / Defense: same Periods panel (months, postseason, bold
  Season block), one row per Statcast pitch type in each period, pitch-colored labels.
- Tabs: Summary, Details, Pillars, Rates, Plot, Usage, Location, Trends.
- Filled from public Statcast: #, %, velo / max / top-8th velo, release height / side / extension,
  arm angle (2024+), IVB, HB (arm side +), spin, spin direction, observed movement direction, VAA, HAA,
  actual swing / ball-on-take / whiff rates, actual run value on balls in play; movement plots, usage
  (by batter hand, by count), plate-location maps, 5-game / 400-pitch rolling trends.
- Splits: Games, Batter Hand, Count (First Pitch 0-0 / Hitter's = balls > strikes except 3-2 /
  Pitcher's = strikes > balls plus 2-2; other counts excluded), RISP, Cluster (blank — model later).
- Blank for our models: aResults, xResults, Command, ActL/ActR, Location grade, Shape (vL/vR), Miss,
  expected rates (xSw / xBl / xWf / xIPRV), spin efficiency, seam-shifted movement, ZAVAA / ZAHAA, clusters.
- Hitting sections only show for players who have hit; Pitching only for players who have pitched.

## v14.4 – 2026-11-05
- Defense month / week / last-30 rows now fill in: built from the player's game-by-game fielding log
  (MLB's date-range totals don't support fielding, which is why those rows were missing).
- Baserunning gets the same Periods picker with months, postseason and the bold Season row, in the
  same table style: Advancing, Base Stealing (G, PA, SB attempts, SB, CS, SB%), Tools (sprint speed,
  home to first, bolts — full seasons). RVA, leads and grades blank for our models.

## v14.3 – 2026-11-05
- New "Postseason" period in the Periods panel (Core 4 + Defense): postseason games only, shown as its
  own row above the months.
- Default Periods: offseason (Oct-Mar) = latest season's Postseason + every month (Oct -> Mar) + the
  Season row. In season (Apr-Sep) = this month + last month + the Season row.

## v14.2 – 2026-11-05
- Wide tables: horizontal scrollbar sits right under the table, is always visible and can be grabbed
  (the space reserved for hover definitions used to cover it). Hover definitions now open over the
  header row so they never need extra space.
- Default Periods: the latest season + its two most recent months (in season: this month and last
  month; offseason: Sep and Aug). Season-only sections default to the latest season.

## v14.1 – 2026-11-05
- Catcher framing now comes from our own framing model (models/framing.py) built from public pitch
  data: framing opps, framing runs and arm side / glove side / top / bottom percentiles for ANY
  period (months, latest week). Savant's framing download no longer works with pybaseball.
  Built by the data update (python -m pipeline.update) -> data/league/catching_<season>.csv.gz.
- Fielding Run Value fetched straight from Savant (missing from the installed pybaseball).
- Outfield jump: match Savant's resp_fielder_id column (jump numbers now fill in).
- Year + Period stay pinned on the left while Core 4 / Defense tables scroll sideways.

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
