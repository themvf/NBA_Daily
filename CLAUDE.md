# NBA Daily - Claude Instructions

## Workflow
- All changes must be pushed to GitHub for Streamlit Cloud deployment
- Test locally before publishing to Git
- Add team logos when charts are based on teams

## Before Planning New Features
- Read `docs/PLANNING_CHECKLIST.md` for pre-implementation verification
- Reference `docs/FANDUEL_ODDS_ARCHITECTURE.md` for odds/betting features
- Check schema migration order in `prediction_tracking.py`

## Reviewing Prior Contest Slates (Calibration)

When asked to review past contests or calibrate projections:

1. **Get the latest database from S3** (local copy may be stale):
   ```python
   from s3_backup import S3PredictionStorage
   S3PredictionStorage().download_database()
   ```

2. **Key tables for contest analysis**:
   - `dfs_slate_projections` — per-player projections, salaries, actuals, ownership (columns: `player_name`, `player_id`, `salary`, `proj_fpts`, `actual_fpts`, `ownership_proj`, `actual_ownership`, `slate_date`, `did_play`)
   - `dfs_contest_entries` — top lineup compositions (columns: `slate_date`, `rank`, `points`, `username`, `pg`, `sg`, `sf`, `pf`, `c`, `g`, `f`, `util`)
   - `dfs_contest_meta` — contest-level stats (`contest_id`, `slate_date`, `total_entries`, `unique_users`, `top_score`)
   - `dfs_slate_results` — aggregate accuracy metrics (`proj_mae`, `proj_correlation`, `ownership_mae`, `ownership_correlation`)
   - `predictions` — model predictions with `vegas_implied_fpts`, `projected_ppg`, `proj_floor`, `proj_ceiling`, `role_tier`, `game_date`

3. **JOIN pattern for full analysis** (projections + predictions + Vegas):
   ```sql
   SELECT s.*, p.vegas_implied_fpts, p.role_tier, p.proj_floor, p.proj_ceiling
   FROM dfs_slate_projections s
   LEFT JOIN predictions p ON s.player_id = p.player_id AND s.slate_date = p.game_date
   WHERE s.slate_date = ? AND s.actual_fpts IS NOT NULL AND s.did_play = 1
   ```

4. **Key metrics to compute**:
   - MAE, RMSE, Bias, Correlation (overall and per salary tier: $8K+, $6K-7.9K, $4K-5.9K, <$4K)
   - Ownership MAE, Bias, Correlation (by projected vs actual ownership buckets)
   - Vegas vs Model comparison (MAE, optimal blend weight)
   - Floor/ceiling hit rates (% in range, % below floor, % above ceiling)
   - Winning lineup composition (player frequency in top 10 entries)

5. **Note**: The `predictions` table does NOT have a `dfs_salary` column. Get salary from `dfs_slate_projections` instead.

6. **Current calibration** (applied in `dfs_optimizer.py`):
   - Projection bias corrections: Stars *0.96, Value *1.05, Punt *1.20
   - Ownership model: power-law sharpening (exponent 2.0) with tier-specific base/ceiling
   - Calibrated from 312 players across slates 2026-02-04 to 02-06
   - Re-calibrate periodically as more contest data accumulates

## Odds Pipeline Overview

### Data Flow
```
User clicks "Fetch/Refresh Odds" in Game Stacks UI
  → odds_api.fetch_fanduel_lines_for_date(conn, game_date, force=True)
    ├─ Budget check: get_monthly_api_usage() (500 req/month, blocks at 95%)
    ├─ get_game_odds_bulk(api_key, game_date) → spreads & totals (1 API request, all games)
    │   └─ extract_game_odds_from_response() → [{game_id, home_team, away_team, spread, total}]
    │   └─ vegas_odds.compute_game_environment() → ot_probability, stack_score, pace_score, blowout_risk
    │   └─ vegas_odds.store_game_odds() → INSERT INTO game_odds
    ├─ Per-event player props fetch (1 request per game, all 7 markets)
    │   └─ update_prediction_with_odds() → UPDATE predictions SET fanduel_ou, vegas_implied_fpts, ...
    └─ Logs to odds_fetch_log, auto-backups to S3
  → dfs_optimizer.enrich_players_with_correlation_model(players, conn, game_date)
    ├─ Queries game_odds → builds team_env_lookup keyed by frozenset([home, away])
    ├─ Matches DK game_ids (e.g. "DAL_SAS_0700PM") to game_odds via team pair frozenset
    ├─ Runs Monte Carlo simulation → p_top1, p_top3 per player
    └─ Sets player.stack_score from team_env_lookup
  → dfs_optimizer.apply_correlation_ceiling_boost(players)
    └─ stack_score >= 0.7 → ceiling * 1.08; stars also get leverage * 1.10
  → Game Stacks UI displays games sorted by stack_score with stack type selectors
  → _select_stack_players() forces 2-4 correlated players into each lineup
```

### Key Files and Functions

| File | Function | Line | Purpose |
|------|----------|------|---------|
| `odds_api.py` | `fetch_fanduel_lines_for_date()` | ~853 | Main entry point: fetches game odds + player props, stores everything |
| `odds_api.py` | `get_game_odds_bulk()` | ~258 | Bulk fetch spreads/totals (uses `commenceTimeFrom/To` for date filtering) |
| `odds_api.py` | `update_prediction_with_odds()` | ~650 | **UPDATE** (not upsert) — predictions must exist first |
| `vegas_odds.py` | `compute_game_environment()` | ~284 | Derives stack_score, ot_probability, blowout_risk from spread/total |
| `dfs_optimizer.py` | `enrich_players_with_correlation_model()` | ~1684 | Maps game_odds to DK players via frozenset team matching |
| `dfs_optimizer.py` | `_select_stack_players()` | ~1906 | Primary (3-4 + bring-back), Mini (2), Auto (score-based) |

### Database Tables

**`game_odds`** — Game-level betting environment
- PK: `game_id` (format: `"{date}_{away}_{home}"`, e.g. `"2026-02-07_DAL_SAS"`)
- Core: `game_date`, `home_team`, `away_team`, `spread`, `total`, `home_ml`, `away_ml`, `commence_time`
- Derived: `pace_score` (total/228), `ot_probability` (0.06-0.11), `blowout_risk` (0-1), `stack_score` (0-0.85), `volatility_multiplier`

**`odds_fetch_log`** — API budget tracking
- `fetch_date`, `game_date`, `events_fetched`, `players_matched`, `api_requests_used`, `remaining_requests`

**`predictions`** (FanDuel columns added by odds fetch)
- Props: `fanduel_ou`, `fanduel_reb_ou`, `fanduel_ast_ou`, `fanduel_3pm_ou`, `fanduel_stl_ou`, `fanduel_blk_ou`, `fanduel_pra_ou`
- Derived: `vegas_implied_fpts` (DFS-weighted composite from all props)
- Metadata: `fanduel_over_odds`, `fanduel_under_odds`, `fanduel_fetched_at`, `odds_event_id`

### Critical Gotchas
- **game_id format mismatch**: `game_odds` uses `"{date}_{away}_{home}"`, DK CSV uses `"{away}_{home}_{time}"`. Always match by `frozenset([team1, team2])`, never by raw game_id string
- **Predictions must exist before odds fetch**: `update_prediction_with_odds()` uses UPDATE, not UPSERT. Workflow: Generate Predictions → Fetch Odds
- **Timezone**: Streamlit Cloud runs UTC. All dates in the DFS Builder use `datetime.now(EASTERN_TZ).date()` (defined at line ~84). The Odds API also returns UTC — `get_game_odds_bulk()` converts Eastern date range to UTC via `commenceTimeFrom`/`commenceTimeTo` params
- **API budget**: 500 requests/month. Bulk game odds = 1 request. Player props = 1 request per game (~15/night). Warns at 80%, blocks at 95%
- **Stack score thresholds**: >= 0.75 → Primary Stack (3-4 players), >= 0.50 → Mini Stack (2 players), < 0.50 → No Stack

## Technical Notes
- Use LeagueGameLog endpoint (not BoxScoreTraditionalV2) for player stats
- S3 backup triggers automatically after FanDuel line fetches
- Timezone: All dates use `datetime.now(EASTERN_TZ).date()` — never use `date.today()` in streamlit_app.py
