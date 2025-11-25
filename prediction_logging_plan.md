# Prediction Logging System Plan

## Executive Summary

Build a comprehensive prediction tracking system to log Daily Pick Score predictions, compare them against actual game results, and use the data to iteratively improve scoring formulas. This creates a feedback loop for continuous model refinement.

---

## 1. Current State Analysis

### Existing Infrastructure
- **Prediction Tab**: Basic tab exists for displaying 3PT predictions from CSV
- **3PT Prediction Script**: `predict_top_3pm.py` generates predictions and can output to CSV
- **Database**: SQLite database with game logs, player stats, team stats
- **Daily Pick Score**: New unified scoring system (0-100) with grades (Elite/Strong/Solid/Risky/Avoid)

### What's Missing
- ❌ Database table to store Daily Pick Score predictions
- ❌ Automated logging when predictions are made
- ❌ Mechanism to match predictions with actual results
- ❌ Analytics/reporting on prediction accuracy
- ❌ Feedback loop to adjust formulas based on performance

---

## 2. System Architecture

### A. Database Schema

#### Table: `daily_predictions`
```sql
CREATE TABLE IF NOT EXISTS daily_predictions (
    -- Prediction Metadata
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_date DATE NOT NULL,              -- When prediction was made
    game_date DATE NOT NULL,                    -- Game being predicted
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Game Context
    game_id TEXT,                               -- NBA API game_id (if available)
    season TEXT NOT NULL,                       -- e.g., "2025-26"
    season_type TEXT DEFAULT 'Regular Season',

    -- Player & Team
    player_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    team_abbreviation TEXT NOT NULL,
    opponent_abbreviation TEXT NOT NULL,
    home_away TEXT,                             -- 'HOME' or 'AWAY'

    -- Prediction Components
    projected_ppg REAL NOT NULL,                -- Smart projection for this game
    projection_confidence REAL NOT NULL,        -- 0.0 to 1.0
    season_avg_ppg REAL,                        -- Player's season average

    matchup_rating TEXT,                        -- Excellent/Good/Neutral/Difficult/Avoid
    matchup_confidence REAL,                    -- Confidence in matchup rating

    opp_def_rating REAL,                        -- Opponent defensive rating
    league_avg_def_rating REAL DEFAULT 112.0,

    -- Final Score & Grade
    daily_pick_score REAL NOT NULL,             -- 0-100 score
    pick_grade TEXT NOT NULL,                   -- Elite/Strong/Solid/Risky/Avoid
    pick_explanation TEXT,                      -- Narrative explanation

    -- Score Breakdown (for analysis)
    score_base REAL,                            -- Base score from projection
    score_matchup_bonus REAL,                   -- ±20 from matchup
    score_defense_adj REAL,                     -- ±10 from defense
    score_confidence_bonus REAL,                -- 0-15 from confidence

    -- Actual Results (filled in after game)
    actual_ppg REAL,                            -- Actual points scored
    actual_minutes REAL,                        -- Minutes played
    actual_fg_pct REAL,                         -- Field goal percentage
    actual_fg3m INTEGER,                        -- Three-pointers made
    dnp_reason TEXT,                            -- If player didn't play

    -- Prediction Accuracy
    prediction_error REAL,                      -- actual_ppg - projected_ppg
    prediction_error_pct REAL,                  -- error / projected_ppg
    outcome_category TEXT,                      -- Accurate/Overestimate/Underestimate/DNP

    -- Metadata
    formula_version TEXT DEFAULT 'v1.0',        -- Track formula changes
    notes TEXT,

    UNIQUE(game_date, player_id, prediction_date)
);

CREATE INDEX idx_predictions_game_date ON daily_predictions(game_date);
CREATE INDEX idx_predictions_player ON daily_predictions(player_id);
CREATE INDEX idx_predictions_grade ON daily_predictions(pick_grade);
CREATE INDEX idx_predictions_accuracy ON daily_predictions(outcome_category);
```

#### Table: `formula_versions`
```sql
CREATE TABLE IF NOT EXISTS formula_versions (
    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_name TEXT UNIQUE NOT NULL,          -- e.g., "v1.0", "v1.1-boosted-base"
    deployed_date DATE NOT NULL,

    -- Formula Parameters
    base_formula_desc TEXT,                     -- e.g., "(proj - 3) * 1.8"
    matchup_excellent_bonus INTEGER,            -- e.g., 20
    matchup_good_bonus INTEGER,                 -- e.g., 10
    matchup_difficult_penalty INTEGER,          -- e.g., -10
    matchup_avoid_penalty INTEGER,              -- e.g., -20
    defense_max_adj INTEGER,                    -- e.g., ±10
    confidence_max_bonus INTEGER,               -- e.g., 15

    -- Thresholds
    threshold_elite INTEGER,                    -- e.g., 80
    threshold_strong INTEGER,                   -- e.g., 65
    threshold_solid INTEGER,                    -- e.g., 50
    threshold_risky INTEGER,                    -- e.g., 35

    -- Matchup Evaluation Logic
    style_confidence_per_game REAL,             -- e.g., 0.10
    h2h_confidence_per_game REAL,               -- e.g., 0.05
    style_priority BOOLEAN DEFAULT 1,           -- 1 = style first, 0 = h2h first

    -- Performance Metrics (filled in over time)
    games_logged INTEGER DEFAULT 0,
    avg_prediction_error REAL,
    avg_prediction_error_pct REAL,

    notes TEXT
);
```

---

## 3. Implementation Phases

### Phase 1: Core Logging Infrastructure (Week 1)

**Goal**: Store predictions in database automatically

#### Tasks:
1. **Create Database Tables**
   - Add `daily_predictions` table to `nba_to_sqlite.py`
   - Add `formula_versions` table
   - Run migration to create tables in existing database

2. **Build Logging Function**
   ```python
   def log_daily_prediction(
       conn: sqlite3.Connection,
       game_date: str,
       player_id: int,
       player_name: str,
       team_abbrev: str,
       opponent_abbrev: str,
       projection: float,
       proj_confidence: float,
       matchup_rating: str,
       daily_pick_score: float,
       pick_grade: str,
       pick_explanation: str,
       score_breakdown: dict,
       formula_version: str = "v1.0"
   ) -> int:
       """Log a prediction to the database. Returns prediction_id."""
   ```

3. **Integrate with Streamlit App**
   - Add "Log Predictions" button in Daily Matchup tab
   - When clicked, iterate through today's matchup data and log each prediction
   - Display success message with count of predictions logged

#### Deliverables:
- ✅ Database tables created
- ✅ Logging function implemented
- ✅ UI button to manually trigger logging
- ✅ First predictions logged successfully

---

### Phase 2: Automated Result Collection (Week 2)

**Goal**: Automatically fetch actual results after games complete

#### Tasks:
1. **Build Result Fetcher Script**
   ```python
   # fetch_prediction_results.py

   def fetch_game_results(db_path: str, game_date: str):
       """
       For a given date, find all logged predictions and update
       with actual results from player_game_log table.
       """
       # 1. Query all predictions for game_date where actual_ppg IS NULL
       # 2. For each prediction, lookup actual stats from player_game_log
       # 3. Calculate prediction_error, prediction_error_pct
       # 4. Categorize outcome (Accurate/Over/Under/DNP)
       # 5. Update prediction record
   ```

2. **Outcome Classification Logic**
   ```python
   def classify_prediction_outcome(
       projected: float,
       actual: float | None,
       dnp: bool
   ) -> str:
       """
       Classify prediction accuracy:
       - DNP: Player didn't play
       - Accurate: Within ±15% of projection
       - Slight_Over/Under: 15-30% error
       - Major_Over/Under: 30%+ error
       """
   ```

3. **Automated Scheduler (Optional)**
   - Cron job or GitHub Action to run `fetch_prediction_results.py` daily
   - Runs at midnight to fetch previous day's results
   - Could also run at game completion using NBA API game status

#### Deliverables:
- ✅ Result fetcher script working
- ✅ Manual execution updates predictions correctly
- ✅ (Optional) Automated daily execution

---

### Phase 3: Analytics & Reporting (Week 3)

**Goal**: Surface insights about prediction accuracy

#### Tasks:
1. **Build Analytics Module**
   ```python
   # prediction_analytics.py

   def calculate_formula_performance(
       conn: sqlite3.Connection,
       formula_version: str,
       min_games: int = 50
   ) -> dict:
       """
       Calculate key metrics for a formula version:
       - Mean Absolute Error (MAE)
       - Mean Absolute Percentage Error (MAPE)
       - Accuracy rate (within ±15%)
       - Grade reliability (how often Elite picks deliver)
       """

   def grade_calibration_analysis(
       conn: sqlite3.Connection
   ) -> pd.DataFrame:
       """
       For each grade (Elite/Strong/Solid/Risky/Avoid):
       - Average projected PPG
       - Average actual PPG
       - Average prediction error
       - Success rate
       """

   def identify_systematic_biases(
       conn: sqlite3.Connection
   ) -> dict:
       """
       Detect patterns:
       - Do we overestimate home games?
       - Do we underestimate certain matchup ratings?
       - Do we overestimate high-confidence predictions?
       """
   ```

2. **Enhanced Prediction Tab UI**
   ```python
   with predictions_tab:
       st.subheader("Daily Pick Score - Prediction Tracker")

       # Filter controls
       date_range = st.date_input("Date range", ...)
       grade_filter = st.multiselect("Grades", ["Elite", "Strong", ...])

       # Load predictions
       predictions_df = load_predictions(conn, date_range, grade_filter)

       # Summary metrics
       col1, col2, col3, col4 = st.columns(4)
       col1.metric("Total Predictions", len(predictions_df))
       col2.metric("Avg Error", f"{predictions_df['prediction_error'].mean():.1f}")
       col3.metric("MAPE", f"{predictions_df['prediction_error_pct'].abs().mean():.1%}")
       col4.metric("Accuracy Rate", f"{accuracy_rate:.1%}")

       # Grade performance breakdown
       st.subheader("Performance by Grade")
       grade_stats = predictions_df.groupby('pick_grade').agg({
           'projected_ppg': 'mean',
           'actual_ppg': 'mean',
           'prediction_error': 'mean'
       })
       st.dataframe(grade_stats)

       # Detailed predictions table
       st.subheader("All Predictions")
       st.dataframe(predictions_df, use_container_width=True)
   ```

3. **Visualization Dashboard**
   - Scatter plot: Projected vs Actual PPG (by grade)
   - Time series: Prediction error over time
   - Bar chart: Grade distribution and success rates
   - Heatmap: Error by matchup rating × opponent defense rating

#### Deliverables:
- ✅ Analytics functions working
- ✅ Enhanced Prediction Tab with metrics
- ✅ Visualizations showing insights

---

### Phase 4: Formula Optimization (Week 4)

**Goal**: Use data to systematically improve scoring formula

#### Tasks:
1. **A/B Testing Framework**
   - Log multiple formula versions simultaneously
   - Compare performance side-by-side
   - Example: Test `(proj-3)*1.8` vs `(proj-4)*1.6`

2. **Parameter Tuning Analysis**
   ```python
   # tune_formula.py

   def test_threshold_scenarios(
       historical_predictions: pd.DataFrame
   ) -> pd.DataFrame:
       """
       Retroactively test different thresholds:
       - What if Elite was 75+ instead of 80+?
       - What if Strong was 60+ instead of 65+?

       Return performance metrics for each scenario.
       """

   def optimize_matchup_bonuses(
       historical_predictions: pd.DataFrame
   ) -> dict:
       """
       Analyze if matchup bonuses are calibrated correctly:
       - Should Excellent be +20 or +25?
       - Should Difficult be -10 or -15?

       Use regression/correlation to find optimal values.
       """
   ```

3. **Iterative Improvement Process**
   - Weekly review of prediction accuracy
   - Identify systematic biases
   - Propose formula adjustments
   - Test adjustments on historical data
   - Deploy new version
   - Monitor performance

#### Deliverables:
- ✅ A/B testing infrastructure
- ✅ Parameter optimization scripts
- ✅ Process for iterative refinement

---

## 4. Key Design Decisions

### Decision 1: Manual vs Automatic Logging

**Option A: Manual Logging** (Recommended for MVP)
- User clicks "Log Today's Predictions" button in Streamlit
- Logs all predictions for games happening today
- Pros: Simple, user controls when to log
- Cons: Requires manual action daily

**Option B: Automatic Logging**
- Predictions logged automatically when user views Daily Matchup tab
- Pros: No manual action needed
- Cons: Could log duplicate predictions if user refreshes

**Recommendation**: Start with Manual (Option A), add Automatic later if needed

---

### Decision 2: When to Fetch Results

**Option A: Manual Fetch**
- Run `fetch_prediction_results.py` manually after games complete
- Pros: Simple, no scheduling needed
- Cons: Easy to forget

**Option B: Scheduled Fetch**
- Cron job runs daily at 3am ET (after all games complete)
- Pros: Fully automated
- Cons: Requires server/hosting

**Option C: On-Demand Fetch in Streamlit**
- Button in Prediction Tab: "Update Results for [Date]"
- Fetches results for selected date
- Pros: User-controlled, no scheduling
- Cons: Manual action needed

**Recommendation**: Start with Option C (On-Demand), add Option B later

---

### Decision 3: How to Handle DNPs (Did Not Play)

**Challenge**: Player was predicted to score but didn't play (injury, rest, etc.)

**Solution**:
- Store `dnp_reason` field (e.g., "Injury", "Coach's Decision", "Rest")
- Exclude DNPs from accuracy metrics by default
- Show DNPs separately in reporting
- Consider adding "Injury Risk" signal to future predictions

---

### Decision 4: Formula Version Tracking

**Why**: As we improve the formula, we need to know which predictions used which version

**Solution**:
- Every prediction logs `formula_version` (e.g., "v1.0", "v1.1-boosted")
- `formula_versions` table stores parameters for each version
- Can compare performance across versions
- Can recompute historical predictions with new formula (for backtesting)

---

## 5. Success Metrics

### Short-term (After 50 predictions)
- ✅ System successfully logs predictions
- ✅ System successfully fetches results
- ✅ Basic accuracy metrics calculated (MAE, MAPE)

### Medium-term (After 200 predictions)
- ✅ Mean Absolute Percentage Error < 20%
- ✅ Elite picks average 25+ PPG actual
- ✅ Strong picks average 22+ PPG actual
- ✅ Avoid picks average <15 PPG actual
- ✅ Identify at least 1 systematic bias to fix

### Long-term (After 500+ predictions)
- ✅ MAPE < 15%
- ✅ Grade calibration: Elite picks score higher than Strong picks on average
- ✅ 2+ formula iterations deployed based on data
- ✅ Predictive power improves season-over-season

---

## 6. Technical Implementation Details

### File Structure
```
NBA_Daily/
├── nba_to_sqlite.py              # Add prediction tables here
├── log_predictions.py             # New: Logging utility
├── fetch_prediction_results.py   # New: Result fetcher
├── prediction_analytics.py        # New: Analytics module
├── streamlit_app.py               # Update prediction tab
└── prediction_logging_plan.md    # This document
```

### Sample Workflow

**Day 1 (Morning):**
1. User opens Streamlit app
2. Views Daily Matchup tab with Pick Scores
3. Clicks "Log Today's Predictions" button
4. System logs 150 predictions (30 players × 5 games)

**Day 2 (Morning):**
1. User clicks "Fetch Results for Yesterday"
2. System queries player_game_log for actual stats
3. Updates 150 predictions with actual_ppg, prediction_error, etc.
4. User views Prediction Tab to see accuracy metrics

**Weekly Review:**
1. Review prediction accuracy in Prediction Tab
2. Notice: Elite picks underperforming (projected 30 PPG, actual 26 PPG)
3. Hypothesis: Formula too optimistic for neutral matchups
4. Test adjustment: Reduce confidence bonus from 15 to 12
5. Backtest on historical data: MAPE improves from 18% to 15%
6. Deploy new formula version "v1.1-conservative-confidence"
7. Continue monitoring

---

## 7. Future Enhancements

### Phase 5+: Advanced Features

1. **Confidence Intervals**
   - Instead of single projection, show range (e.g., "24-28 PPG")
   - Track how often actual falls within predicted range

2. **Injury/Availability Signals**
   - Integrate injury reports from NBA API
   - Flag predictions where player is questionable/probable
   - Adjust confidence scores based on injury risk

3. **Real-time Updates**
   - Use NBA API to fetch live scores
   - Update predictions during games
   - Show "trending up/down" indicators

4. **Multi-metric Predictions**
   - Extend beyond PPG to predict rebounds, assists, 3PM
   - Create unified "Daily Fantasy Score" prediction

5. **Machine Learning Model**
   - Train ML model on historical predictions + results
   - Use ensemble of formula + ML predictions
   - Compare ML vs formula performance

6. **User Feedback Loop**
   - Allow users to rate predictions
   - "Was this prediction useful?"
   - Track which predictions users actually use

---

## 8. Risk & Mitigation

### Risk 1: Data Quality Issues
**Risk**: Actual results don't match due to data source discrepancies
**Mitigation**:
- Validate game_id matching between predictions and results
- Manual spot-check first 20 predictions
- Add data quality alerts

### Risk 2: Low Prediction Volume
**Risk**: Not enough predictions to draw statistical conclusions
**Mitigation**:
- Start simple with just top players (20+ PPG)
- Log predictions for multiple games per day
- Set minimum threshold (50 predictions) before formula changes

### Risk 3: Overfitting to Recent Data
**Risk**: Optimize formula for recent games, perform poorly on new data
**Mitigation**:
- Use train/validation split when tuning
- Test on multiple seasons
- Prefer simple adjustments over complex ones

### Risk 4: User Abandonment
**Risk**: Manual logging is too tedious, users stop using it
**Mitigation**:
- Make logging one-click easy
- Show immediate value (charts, insights)
- Automate fetching results
- Send weekly summary emails (optional)

---

## 9. Questions to Resolve

Before starting implementation:

1. **Storage location**: Same SQLite DB as stats, or separate predictions DB?
   - Recommendation: Same DB for simplicity

2. **Scope**: Log all players or just top scorers?
   - Recommendation: Start with players averaging 15+ PPG (manageable volume)

3. **UI placement**: New tab or expand existing Prediction tab?
   - Recommendation: Expand existing Prediction tab

4. **Logging trigger**: Button click, page load, or scheduled?
   - Recommendation: Button click for MVP

5. **Privacy**: Are predictions visible to other users?
   - Recommendation: Start with single-user (local), add multi-user later

---

## 10. Next Steps

### Immediate Actions (This Week)
1. ✅ Review and approve this plan
2. ⬜ Create `daily_predictions` and `formula_versions` tables
3. ⬜ Build basic logging function
4. ⬜ Add "Log Predictions" button to Streamlit
5. ⬜ Test with today's games

### Short-term (Next 2 Weeks)
6. ⬜ Build result fetcher script
7. ⬜ Log 50+ predictions and fetch results
8. ⬜ Build analytics dashboard
9. ⬜ Review first accuracy metrics

### Medium-term (Next Month)
10. ⬜ Identify first formula improvement opportunity
11. ⬜ Deploy formula v1.1
12. ⬜ Compare v1.0 vs v1.1 performance
13. ⬜ Document findings and iterate

---

## Appendix: Sample Queries

### Query 1: Grade Performance Summary
```sql
SELECT
    pick_grade,
    COUNT(*) as predictions,
    AVG(projected_ppg) as avg_projected,
    AVG(actual_ppg) as avg_actual,
    AVG(prediction_error) as avg_error,
    AVG(ABS(prediction_error_pct)) as mape
FROM daily_predictions
WHERE actual_ppg IS NOT NULL
GROUP BY pick_grade
ORDER BY
    CASE pick_grade
        WHEN 'Elite' THEN 1
        WHEN 'Strong' THEN 2
        WHEN 'Solid' THEN 3
        WHEN 'Risky' THEN 4
        WHEN 'Avoid' THEN 5
    END;
```

### Query 2: Recent Prediction Accuracy
```sql
SELECT
    game_date,
    player_name,
    pick_grade,
    projected_ppg,
    actual_ppg,
    prediction_error,
    outcome_category
FROM daily_predictions
WHERE game_date >= DATE('now', '-7 days')
AND actual_ppg IS NOT NULL
ORDER BY game_date DESC, daily_pick_score DESC;
```

### Query 3: Systematic Bias Detection
```sql
-- Do we overestimate home games?
SELECT
    home_away,
    AVG(prediction_error) as avg_error
FROM daily_predictions
WHERE actual_ppg IS NOT NULL
GROUP BY home_away;

-- Do we overestimate certain matchup ratings?
SELECT
    matchup_rating,
    AVG(prediction_error) as avg_error,
    COUNT(*) as n
FROM daily_predictions
WHERE actual_ppg IS NOT NULL
GROUP BY matchup_rating;
```

---

**Document Version**: 1.0
**Created**: 2025-11-25
**Author**: Claude (with user input)
**Status**: Ready for Review
