# NBA SQLite Builder

This utility pulls league reference data through [`nba_api`](https://github.com/swar/nba-api) and writes a ready-to-query SQLite database. The default run captures:

- Team directory metadata (current + historical entries).
- Player directory metadata (active + inactive players).
- League standings for a season and season type you choose.
- Full team rosters for the target season (can be disabled).

## Getting Started

```powershell
cd "C:\Docs\_AI Python Projects\NBA_API Exploration"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python nba_to_sqlite.py --season 2023-24 --season-type "Regular Season"
```

## Streamlit App

Once the SQLite file exists, launch the dashboard locally:

```powershell
streamlit run streamlit_app.py
```

When the app boots it checks for `nba_stats.db`. If the file is missing it automatically rebuilds it by calling the same pipeline as `nba_to_sqlite.py` using the seasons configured in the sidebar’s “NBA API Builder” section (defaults to the current season, 2025-26). Manual rebuilds are available via a button, and you can still point the sidebar inputs to an alternate path or uploaded file if you prefer a pre-generated snapshot. The “Today’s Games” tab calls the NBA Scoreboard API on demand to show the current day’s matchups (with standings context pulled from the local database), surfaces each team’s season + recent scoring profile (totals, averages, medians, last-3/5 averages, 3PM medians), and includes a “Top scorers per matchup” panel plus a dedicated “Matchup Spotlight” tab that highlights the top five players from each team with season averages, recent splits, and opponent defensive allowances so you can quickly identify favorable offensive matchups. Tune the weights and minimum-games threshold from the tab settings.

Key options:

| Flag | Description |
| --- | --- |
| `--db-path` | Output file (defaults to `nba_stats.db` in the current directory). |
| `--season` | Season label understood by `nba_api` (e.g. `2023-24`). |
| `--season-type` | Season type string accepted by the stats API (`Regular Season`, `Playoffs`, etc.). |
| `--include-rosters/--no-include-rosters` | Enable/disable team roster harvesting (enabled by default). |
| `--throttle-seconds` | Delay between roster calls to stay under the public API rate limit (default `0.6`). |
| `--shooting-season` / `--shooting-season-type` | Override the season/season type used for player shooting totals (defaults to the main `--season` inputs). |
| `--top-3pt-view-season` / `--top-3pt-view-season-type` | Control which season is exposed by the `players_2025_top_3pt` view (defaults to `2024-25` Regular Season). |
| `--defense-view-season` / `--defense-view-season-type` | Control which season is exposed by the `teams_2025_defense_3pt` view (defaults match the shooting season). |
| `--top-pts-view-season` / `--top-pts-view-season-type` | Control the season powering the `players_2025_top_pts` scoring view (defaults to the shooting season). |
| `--defense-pts-view-season` / `--defense-pts-view-season-type` | Control the season powering the `teams_2025_defense_pts` scoring-allowed view (defaults to the defensive 3PT season). |
| `--defense-mix-view-season` / `--defense-mix-view-season-type` | Control the season powering the `teams_2026_defense_mix` scoring breakdown view (defaults to `2025-26`). |

The script prints progress as it upserts each dataset. Re-running it is idempotent thanks to SQLite upserts.

## Database Layout

| Table | Purpose |
| --- | --- |
| `teams` | One row per team returned by `nba_api.stats.static.teams`. Contains location/name metadata and flags for active franchises. |
| `players` | One row per player returned by `nba_api.stats.static.players`, with an `is_active` bit. |
| `standings` | Season + season-type standings from `LeagueStandings`, including wins/losses, win percentage, records, ranks, and streak info. |
| `team_rosters` | Detailed roster rows (jersey, position, physicals, school, experience) keyed by season/team/player. |
| `player_season_totals` | Totals from `LeagueDashPlayerStats` (games played, minutes, totals for FGM/3PM/FTM, rebounds, assists, points) keyed by season, season type, and player. |
| `player_game_logs` | One row per game per player from `PlayerGameLogs`, capturing matchup info plus 3PA/3PM and scoring totals (recreated on each run). |
| `team_game_logs` | Team-level game logs from `TeamGameLogs`, including opponent three-point makes/attempts and total points. |

### Offensive Three-Point View

The script now builds (or refreshes) a convenience view named `players_2025_top_3pt` that surfaces league leaders in made threes per game for the season you target via `--top-3pt-view-season`. Columns exposed:

- `total_fg3m`: total makes for the season.
- `avg_fg3m_per_game`: average makes per game.
- `avg_fg3a_per_game`: average attempts per game (requested metric #1).
- `median_fg3m_per_game`: per-player median of game-by-game made threes (metric #2).
- `max_fg3m_per_game`: per-player single-game high for made threes (metric #3).
- `rank_fg3m_per_game`: rank ordered by `avg_fg3m_per_game` descending (metric #4).

```sql
SELECT player_name,
       team_abbreviation,
       total_fg3m,
       avg_fg3a_per_game,
       median_fg3m_per_game,
       max_fg3m_per_game,
       avg_fg3m_per_game,
       rank_fg3m_per_game
FROM players_2025_top_3pt
LIMIT 10;
```

### Defensive Three-Point View

`teams_2025_defense_3pt` ranks teams by how many made threes they allow per game (average, median, and single-game max) for the season dictated by `--defense-view-season`.

```sql
SELECT team_name,
       avg_allowed_fg3m,
       avg_allowed_fg3a,
       median_allowed_fg3m,
       max_allowed_fg3m,
       rank_avg_allowed_fg3m
FROM teams_2025_defense_3pt
ORDER BY rank_avg_allowed_fg3m ASC;
```

### Offensive Points View

`players_2025_top_pts` mirrors the three-point view but focuses on scoring volume. Columns include total points, average/median/max points per game, and the rank (descending by average).

```sql
SELECT player_name,
       team_abbreviation,
       total_points,
       avg_points_per_game,
       median_points_per_game,
       max_points_per_game,
       rank_points_per_game
FROM players_2025_top_pts
LIMIT 10;
```

### Defensive Points View

`teams_2025_defense_pts` highlights teams that concede the most points per game (average, median, and single-game max) along with their rank.

```sql
SELECT team_name,
       avg_allowed_pts,
       median_allowed_pts,
       max_allowed_pts,
       rank_avg_allowed_pts
FROM teams_2025_defense_pts
ORDER BY rank_avg_allowed_pts ASC;
```

### Defensive Mix View (Points vs 3PM)

	eams_2026_defense_mix focuses on how much of each opponent's scoring (total + median per game) comes from made threes, and now surfaces assist/rebound volume allowed to show how teams defend playmaking and the glass.

`sql
SELECT team_name,
       total_allowed_pts,
       avg_allowed_pts,
       median_allowed_pts,
       total_allowed_fg3m,
       median_allowed_fg3m,
       total_allowed_ast,
       avg_allowed_ast,
       total_allowed_reb,
       avg_allowed_reb,
       pct_points_from_3_total,
       pct_points_from_3_median
FROM teams_2026_defense_mix
ORDER BY pct_points_from_3_total DESC;
`


## Predicting Team 3PM Leaders

Use `predict_top_3pm.py` to train a simple model that, for each game, selects the player most likely to lead their team in made threes.

```powershell
python predict_top_3pm.py `
  --db-path nba_stats.db `
  --train-seasons 2024-25 `
  --test-seasons 2025-26 `
  --output-predictions predictions.csv
```

What it does:

- Loads `player_game_logs`, `player_season_totals`, and `team_game_logs` for the requested seasons.
- Engineers pre-game features: rolling/lagged 3PM, attempts, minutes, season averages, plus team + opponent defensive trends.
- Trains a gradient-boosted classifier on the training seasons and evaluates on the test seasons. The CLI reports ROC-AUC, per-game accuracy (did we pick the actual leader), and prints a sample of predictions. Optionally, it writes the per-game picks to CSV.

You can adjust `--train-seasons` / `--test-seasons` to cross-validate across multiple years, increase `--min-history-games` to demand more prior data, or point `--output-predictions` somewhere else for downstream workflows.

Example query after loading:

```sql
SELECT t.full_name,
       s.wins,
       s.losses,
       s.win_pct
FROM standings AS s
JOIN teams AS t ON t.team_id = s.team_id
WHERE s.season = '2023-24'
  AND s.season_type = 'Regular Season'
ORDER BY s.win_pct DESC;
```

You can open the database with any SQLite browser or use the CLI: `sqlite3 nba_stats.db`.
