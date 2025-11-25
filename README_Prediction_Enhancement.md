# Prediction Enhancement System

## Overview

This document tracks the evolution of the NBA Daily prediction system, including scoring formula improvements and the prediction logging infrastructure for continuous refinement.

---

## Table of Contents

1. [Current System](#current-system)
2. [Recent Improvements](#recent-improvements)
3. [Prediction Logging System](#prediction-logging-system)
4. [Future Roadmap](#future-roadmap)
5. [Version History](#version-history)

---

## Current System

### Daily Pick Score (v1.0)

**Purpose**: Unified 0-100 score answering "How good is this player as a pick TODAY?"

**Formula Components**:

1. **Base Score**: `(projection - 3) × 1.8`
   - Range: ~3.6 (5 PPG player) to ~66.6 (40 PPG player)
   - More generous than v0.9 to reward elite scorers

2. **Matchup Bonus**: -20 to +20
   - Excellent: +20
   - Good: +10
   - Neutral: 0
   - Difficult: -10
   - Avoid: -20

3. **Defense Adjustment**: -10 to +10
   - Based on opponent's defensive rating vs league average (112.0)
   - Weak defense (118): ~+5 bonus
   - Elite defense (106): ~-5 penalty

4. **Confidence Bonus**: 0 to +15
   - **Additive** (not multiplicative!) to avoid crushing scores
   - High confidence (80%): +12
   - Medium confidence (55%): +8.25
   - Low confidence (30%): +4.5

**Final Score**: `base + matchup + defense + confidence` (capped 0-100)

**Grade Thresholds**:
- **🔥 Elite** (80+): Perfect storm conditions
- **⭐ Strong** (65+): Elite scorer OR favorable matchup
- **✓ Solid** (50+): Reliable option
- **⚠️ Risky** (35+): Uncertainty or mid-tier player
- **❌ Avoid** (<35): Poor conditions

**Narrative Explanations**: Each grade includes contextual explanation:
- "Elite pick: elite scorer, excels vs this matchup, weak defense, high confidence"
- "Caution: strong scorer, struggles vs this defense, elite defense"
- "Avoid: limited volume, historically poor vs opponent, limited data"

---

## Recent Improvements

### v1.0 - Boosted Base + Style-First Matchups (2025-11-25)

#### Problem
- Early season: limited head-to-head history → most matchups rated "Neutral"
- Elite scorers (Giannis 32.5 PPG) only getting "Solid" grades with neutral matchups
- All players showing "Avoid" or "Risky" due to conservative formula
- System not useful for finding top picks

#### Changes Made

**1. Boosted Base Formula**
- **Before**: `(projection - 5) × 1.5` → max ~52.5 for 40 PPG
- **After**: `(projection - 3) × 1.8` → max ~66.6 for 40 PPG
- **Impact**: Elite scorers now reach "Strong" grade even with neutral matchups

**2. Style-First Matchup Evaluation**

Reprioritized defensive style over head-to-head history:

| Metric | v0.9 | v1.0 | Reasoning |
|--------|------|------|-----------|
| Style confidence per game | 5% | **10%** | Larger sample sizes (10+ games vs style) |
| H2H confidence per game | 15% | **5%** | Limited early season (2-3 games vs team) |
| Evaluation priority | H2H first | **Style first** | More predictive signal |

**Logic Flow (v1.0)**:
1. Check defensive style matchup FIRST
   - Good (12%+ better): +10 bonus
   - Difficult (18%+ worse): -10 penalty
2. Use head-to-head ONLY as override for extreme cases
   - If style says "Good" but player historically terrible vs this team → downgrade
   - If style says "Difficult" but player historically dominates this team → upgrade
3. Default to Neutral if insufficient data

**Why Style > Head-to-Head**:
- **Sample size**: 10+ games vs defensive schemes vs 2-3 games vs specific teams
- **Predictive power**: Defensive systems are deliberate; head-to-head includes random variance
- **Early season utility**: Style data accumulates faster and available immediately

#### Results

| Player | Scenario | v0.9 Grade | v1.0 Grade | ✓ |
|--------|----------|------------|------------|---|
| Giannis (32.5 PPG) | Neutral matchup, avg def | Solid | **Strong** | ✓ |
| Luka (28.5 PPG) | Good style matchup | Solid | **Strong** | ✓ |
| AD (27.8 PPG) | Good style matchup | Solid | **Strong** | ✓ |
| Mid-tier (24 PPG) | Neutral matchup | Risky | **Solid** | ✓ |
| Elite (32 PPG) | Excellent + weak def | Strong | **Elite** | ✓ |

**Success**: Top performers now appropriately graded even without perfect matchups.

---

### v0.9 - Contextual Narrative Explanations (2025-11-25)

#### Problem
- Users see grades but don't understand WHY
- "Why is Player A 'Risky' and Player B 'Solid'?"
- No actionable insight from grade alone

#### Solution
Generate dynamic explanations based on component factors:

**Projection Factor**:
- Elite scorer (30+ PPG)
- Strong scorer (25-30 PPG)
- Solid volume (18-25 PPG)
- Limited volume (<18 PPG)

**Matchup Factor**:
- Excels vs this matchup
- Favorable matchup
- Struggles vs this defense
- Historically poor vs opponent

**Defense Factor**:
- Weak defense (118+ rating)
- Elite defense (106- rating)

**Confidence Factor**:
- High confidence (75%+)
- Limited data (<50%)

**Display**: `{Grade}: {explanation}`

**Examples**:
- Elite pick: elite scorer, excels vs this matchup, weak defense, high confidence
- Solid: solid volume, favorable matchup
- Caution: strong scorer, struggles vs this defense, elite defense, high confidence
- Avoid: limited volume, historically poor vs opponent, limited data

#### Result
Users instantly understand what's driving each grade → better decision-making.

---

### v0.8 - Additive Confidence (Fixed "All Avoid" Bug) (2025-11-25)

#### Problem
- **Every single player** showing "Avoid" grade
- Confidence multiplier (0.5 + conf × 0.5) was **crushing** scores
- Example: LeBron (28 PPG, 55% conf) → 24 points → Avoid ❌

#### Root Cause Analysis
```
Base: (28 / 45) × 50 = 31.1
Multiplier: 0.5 + 0.55 × 0.5 = 0.775
Final: 31.1 × 0.775 = 24.1 → Avoid (wrong!)
```

The multiplier penalized scores by **22.5%** for typical 55% confidence.

#### Solution
Changed confidence from **multiplicative** to **additive**:
- **Before**: `score × (0.5 + conf × 0.5)` [crushing penalty]
- **After**: `score + (conf × 15)` [reward bonus]

#### Results

| Player | Scenario | Before | After | ✓ |
|--------|----------|--------|-------|---|
| LeBron (28 PPG, 55%) | Neutral | 24.1 (Avoid) | 42.8 (Risky) | ✓ |
| Curry (30 PPG, 60%) | Good | 38.5 (Risky) | 57.3 (Solid) | ✓ |
| Giannis (32 PPG, 75%) | Excellent | 52.3 (Solid) | 76.8 (Strong) | ✓ |

**Lesson**: Confidence should **reward** high-quality data, not **punish** uncertainty.

---

## Prediction Logging System

### Vision

Build a **feedback loop** to continuously improve predictions:

```
1. Log predictions → 2. Fetch results → 3. Analyze accuracy → 4. Refine formula → 5. Repeat
```

This creates compounding value where each formula iteration learns from real-world performance.

---

### Architecture

#### Database Schema

**`daily_predictions` table**:
```sql
CREATE TABLE daily_predictions (
    prediction_id INTEGER PRIMARY KEY,
    prediction_date DATE,              -- When prediction was made
    game_date DATE,                    -- Game being predicted

    player_id INTEGER,
    player_name TEXT,
    team_abbreviation TEXT,
    opponent_abbreviation TEXT,

    -- Prediction
    projected_ppg REAL,
    projection_confidence REAL,
    matchup_rating TEXT,
    daily_pick_score REAL,
    pick_grade TEXT,
    pick_explanation TEXT,

    -- Score breakdown for analysis
    score_base REAL,
    score_matchup_bonus REAL,
    score_defense_adj REAL,
    score_confidence_bonus REAL,

    -- Actual results (filled after game)
    actual_ppg REAL,
    actual_minutes REAL,
    dnp_reason TEXT,

    -- Accuracy metrics
    prediction_error REAL,              -- actual - projected
    prediction_error_pct REAL,          -- error / projected
    outcome_category TEXT,              -- Accurate/Over/Under/DNP

    formula_version TEXT,

    UNIQUE(game_date, player_id, prediction_date)
);
```

**`formula_versions` table**:
```sql
CREATE TABLE formula_versions (
    version_id INTEGER PRIMARY KEY,
    version_name TEXT UNIQUE,           -- e.g., "v1.0", "v1.1-tuned"
    deployed_date DATE,

    -- Formula parameters
    base_formula_desc TEXT,
    matchup_excellent_bonus INTEGER,
    matchup_good_bonus INTEGER,
    confidence_max_bonus INTEGER,

    -- Thresholds
    threshold_elite INTEGER,
    threshold_strong INTEGER,
    threshold_solid INTEGER,

    -- Performance (filled over time)
    games_logged INTEGER,
    avg_prediction_error REAL,
    avg_prediction_error_pct REAL
);
```

---

### Implementation Phases

#### Phase 1: Core Logging (Week 1)

**Goal**: Store predictions in database

**Tasks**:
1. Create database tables in `nba_to_sqlite.py`
2. Build `log_predictions.py` with logging function
3. Add "Log Today's Predictions" button in Streamlit Daily Matchup tab
4. Test with today's games

**Deliverables**:
- ✅ Database tables created
- ✅ Logging function working
- ✅ First predictions logged

---

#### Phase 2: Result Collection (Week 2)

**Goal**: Automatically fetch actual results

**Tasks**:
1. Build `fetch_prediction_results.py` script
2. Match predictions with `player_game_log` table
3. Calculate prediction errors and classify outcomes
4. Add "Fetch Results" button in Prediction Tab

**Outcome Classification**:
- **Accurate**: Within ±15% of projection
- **Slight Over/Under**: 15-30% error
- **Major Over/Under**: 30%+ error
- **DNP**: Player didn't play

**Deliverables**:
- ✅ Result fetcher working
- ✅ 50+ predictions with results
- ✅ Error metrics calculated

---

#### Phase 3: Analytics & Reporting (Week 3)

**Goal**: Surface insights about accuracy

**Features**:

**1. Summary Metrics Dashboard**:
```
┌──────────────────────────────────────────────┐
│  Total Predictions: 150                      │
│  Avg Error: ±3.2 PPG                         │
│  MAPE: 14.2%                                 │
│  Accuracy Rate: 68% (within ±15%)           │
└──────────────────────────────────────────────┘
```

**2. Grade Performance Breakdown**:
| Grade | Predictions | Avg Projected | Avg Actual | Avg Error | Success Rate |
|-------|-------------|---------------|------------|-----------|--------------|
| Elite | 12 | 32.5 | 28.3 | -4.2 | 67% |
| Strong | 35 | 27.8 | 25.1 | -2.7 | 71% |
| Solid | 68 | 22.3 | 21.8 | -0.5 | 76% |
| Risky | 30 | 17.5 | 16.2 | -1.3 | 63% |
| Avoid | 5 | 12.1 | 11.8 | -0.3 | 60% |

**3. Visualizations**:
- Scatter plot: Projected vs Actual PPG (colored by grade)
- Time series: Prediction error trend over time
- Bar chart: Grade distribution and success rates
- Heatmap: Error by matchup rating × defense rating

**4. Systematic Bias Detection**:
```python
# Do we overestimate home games?
Home games: +2.1 PPG error (overestimate)
Away games: -0.3 PPG error (accurate)

# Do we overestimate certain matchup ratings?
Excellent: -3.5 PPG error (overestimate)
Good: -1.2 PPG error (slight overestimate)
Neutral: +0.1 PPG error (accurate)
Difficult: +0.8 PPG error (slight underestimate)
```

**Deliverables**:
- ✅ Enhanced Prediction Tab with analytics
- ✅ Visualizations working
- ✅ Systematic biases identified

---

#### Phase 4: Formula Optimization (Week 4)

**Goal**: Use data to improve formula

**Capabilities**:

**1. A/B Testing**:
- Log multiple formula versions simultaneously
- Compare side-by-side: v1.0 vs v1.1-tuned
- Deploy winner

**2. Parameter Tuning**:
```python
# Test different thresholds
Current: Elite 80+, Strong 65+, Solid 50+
Test A: Elite 75+, Strong 60+, Solid 45+
Test B: Elite 85+, Strong 70+, Solid 55+

# Optimize matchup bonuses
Current: Excellent +20, Good +10, Difficult -10
Test: Excellent +25, Good +12, Difficult -12
```

**3. Iterative Improvement Process**:
1. Weekly review of prediction accuracy
2. Identify systematic biases (e.g., "we overestimate Excellent matchups by 3.5 PPG")
3. Propose formula adjustment (e.g., "reduce Excellent bonus from +20 to +15")
4. Backtest on historical data
5. Deploy if improvement confirmed
6. Monitor new version performance

**Deliverables**:
- ✅ A/B testing infrastructure
- ✅ Parameter optimization tools
- ✅ 2+ formula iterations deployed

---

### Success Metrics

#### Short-term (After 50 predictions)
- ✅ System logs and fetches results successfully
- ✅ Basic accuracy metrics calculated (MAE, MAPE)

#### Medium-term (After 200 predictions)
- ✅ Mean Absolute Percentage Error < 20%
- ✅ Elite picks average 25+ PPG actual
- ✅ Strong picks average 22+ PPG actual
- ✅ Avoid picks average <15 PPG actual
- ✅ At least 1 systematic bias identified and fixed

#### Long-term (After 500+ predictions)
- ✅ MAPE < 15%
- ✅ Grade calibration: Elite > Strong > Solid > Risky > Avoid (on average)
- ✅ 2+ formula iterations deployed based on data
- ✅ Predictive power improves season-over-season

---

### Sample Workflow

**Monday Morning (Before games):**
1. Open Streamlit app → Daily Matchup tab
2. Review Pick Scores for today's games
3. Click "Log Today's Predictions" button
4. System logs 150 predictions (30 players × 5 games)

**Tuesday Morning (After games):**
1. Open Prediction Tab
2. Click "Fetch Results for Monday"
3. System queries `player_game_log` for actual stats
4. Updates 150 predictions with actual_ppg, errors, outcomes
5. View updated accuracy metrics

**Weekly Review:**
1. Review Prediction Tab analytics
2. Notice: Elite picks underperforming (projected 30 PPG, actual 26 PPG)
3. Hypothesis: Formula too optimistic for Excellent matchups
4. Test adjustment: Reduce Excellent bonus from +20 to +15
5. Backtest on historical data: MAPE improves from 18% to 15%
6. Deploy new formula version "v1.1-conservative-excellent"
7. Continue monitoring v1.1 performance

---

### Key Design Decisions

**1. Manual vs Automatic Logging**
- **Choice**: Manual (button click) for MVP
- **Rationale**: Simple, user-controlled, avoids duplicate logging
- **Future**: Add automatic logging when ready

**2. When to Fetch Results**
- **Choice**: On-demand (button in UI)
- **Rationale**: User-controlled, no scheduling needed
- **Future**: Add scheduled daily fetch at 3am ET

**3. Scope of Predictions**
- **Choice**: Start with players averaging 15+ PPG
- **Rationale**: Manageable volume (~20-30 per game), high-value picks
- **Future**: Expand to all players with 10+ minutes/game

**4. Formula Version Tracking**
- **Why**: As we improve formulas, need to compare performance across versions
- **How**: Every prediction logs `formula_version` field
- **Benefit**: Can A/B test, backtest, and measure improvement over time

**5. DNP (Did Not Play) Handling**
- **Challenge**: Player predicted to score but didn't play
- **Solution**: Store `dnp_reason`, exclude from accuracy metrics by default
- **Future**: Add injury risk signals to reduce DNPs

---

## Future Roadmap

### Phase 5: Advanced Analytics

**1. Confidence Intervals**
- Instead of single projection, show range (e.g., "24-28 PPG with 80% confidence")
- Track how often actual falls within predicted range
- Adjust interval width based on historical variance

**2. Multi-Metric Predictions**
- Extend beyond PPG: rebounds, assists, 3PM, steals, blocks
- Create unified "Daily Fantasy Score" prediction
- Optimize for different DFS platforms (DraftKings, FanDuel)

**3. Real-time Updates**
- Integrate NBA API live scores
- Update predictions during games
- Show "trending up/down" indicators based on first half performance

**4. Injury/Availability Integration**
- Scrape injury reports from NBA API
- Flag predictions where player is questionable/probable
- Adjust confidence scores based on injury risk
- Reduce DNP rate

---

### Phase 6: Machine Learning Enhancement

**1. Hybrid Model**
- Train ML model on historical predictions + results
- Use ensemble: (Formula × 0.5) + (ML × 0.5)
- Compare performance: Formula vs ML vs Hybrid

**2. Feature Engineering**
- Rest days, back-to-back games
- Travel distance, altitude
- Referee tendencies
- Historical performance vs specific opponent players
- Team dynamics (recent trades, coaching changes)

**3. Model Monitoring**
- Track ML model drift over time
- Retrain periodically with new data
- A/B test model versions

---

### Phase 7: User Engagement

**1. Prediction Contests**
- "Pick the top 5 scorers tonight"
- Track user prediction accuracy vs system
- Leaderboard for most accurate users

**2. Feedback Loop**
- "Was this prediction useful?" rating
- Track which predictions users act on
- Use feedback to weight predictions

**3. Personalization**
- Learn user preferences (risk tolerance, favorite players)
- Customize recommendations
- "You historically prefer Strong picks → here are today's top Strong picks"

---

## Version History

### v1.0 - Boosted Base + Style-First (2025-11-25)
- **Base formula**: Changed from `(proj-5)*1.5` to `(proj-3)*1.8`
- **Matchup evaluation**: Prioritize defensive style over head-to-head
- **Impact**: Elite scorers now reach Strong grades with neutral matchups
- **Status**: ✅ Deployed

### v0.9 - Contextual Narratives (2025-11-25)
- **Feature**: Added dynamic explanations to grades
- **Format**: "{Grade}: {explanation}" based on component factors
- **Impact**: Users understand what drives each grade
- **Status**: ✅ Deployed

### v0.8 - Additive Confidence Fix (2025-11-25)
- **Bug**: Confidence multiplier crushing all scores → "All Avoid"
- **Fix**: Changed confidence from multiplicative to additive (+0-15 bonus)
- **Impact**: Scores now realistic (LeBron 28 PPG → Risky, not Avoid)
- **Status**: ✅ Deployed

### v0.7 - Unified Daily Pick Score (2025-11-24)
- **Feature**: Created single 0-100 score combining all factors
- **Components**: Projection + Matchup + Defense + Confidence
- **Grades**: Elite/Strong/Solid/Risky/Avoid with thresholds
- **Status**: ✅ Deployed

### v0.6 - Smart PPG Projections (2025-11-23)
- **Feature**: Multi-factor projections (season avg, recent form, matchup history, defense, pace)
- **Output**: Projection + confidence + floor/ceiling range
- **Status**: ✅ Deployed

### v0.5 - Defense Style Matchups (2025-11-22)
- **Feature**: Classify opponents by defensive style (Perimeter/Interior/Balanced)
- **Data**: Player performance splits vs each style
- **Status**: ✅ Deployed

---

## References

- **Main Plan Document**: `prediction_logging_plan.md` (detailed 10-section plan)
- **Test Scripts**:
  - `test_additive_score.py` - Test additive confidence fix
  - `test_final_score.py` - Test boosted base formula
  - `test_narratives.py` - Test narrative explanations
  - `test_style_emphasis.py` - Test style-first matchup evaluation
- **Diagnostic Scripts**:
  - `diagnose_matchups.py` - Analyze matchup data quality
  - `debug_pick_score.py` - Debug scoring issues
  - `quick_score_test.py` - Quick scoring scenarios

---

## Key Insights Learned

### 1. Avoid Crushing Multipliers
**Problem**: Multiplicative confidence (0.5-1.0×) crushed scores by 20-50%

**Solution**: Additive bonus (0-15) rewards high confidence without punishing uncertainty

**Lesson**: Penalties should be proportional, not exponential

---

### 2. Prioritize Sample Size
**Problem**: Head-to-head history (2-3 games) was weighted higher than style matchups (10+ games)

**Solution**: Prioritize style-first, use head-to-head only for strong signals

**Lesson**: Larger samples → more reliable predictions, especially early season

---

### 3. Make Grades Actionable
**Problem**: Users saw grades but didn't know WHY or what factors mattered

**Solution**: Generate contextual narratives explaining each grade

**Lesson**: Transparency builds trust and helps users learn patterns

---

### 4. Design for Iteration
**Problem**: Hard to improve formula without measuring current performance

**Solution**: Prediction logging system creates feedback loop for continuous refinement

**Lesson**: Build feedback mechanisms into systems from the start

---

### 5. Balance Selectivity vs Utility
**Problem**: Too conservative thresholds (Elite 80+) meant no picks showed as Elite/Strong

**Solution**: Boosted base scoring so elite performers earn appropriate grades

**Lesson**: System must be selective enough to be meaningful but generous enough to be useful

---

## Contact & Contributions

For questions, suggestions, or to contribute improvements:
- Review the detailed plan: `prediction_logging_plan.md`
- Check version history in this document
- Test changes using provided test scripts
- Document all formula changes with version bumps

---

**Last Updated**: 2025-11-25
**Current Formula Version**: v1.0
**Next Milestone**: Implement Phase 1 (Core Logging)
