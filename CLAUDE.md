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

## Technical Notes
- Use LeagueGameLog endpoint (not BoxScoreTraditionalV2) for player stats
- S3 backup triggers automatically after FanDuel line fetches
- Timezone: The Odds API returns UTC, convert to US/Eastern before filtering
