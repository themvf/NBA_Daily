# Opponent Injury Impact Tracking

**Version:** 1.0
**Date:** December 26, 2025
**Status:** Production Ready

## Overview

The Opponent Injury Impact Tracking system identifies when opponent team injuries create easier defensive matchups for our players, applies appropriate projection/ceiling boosts, and **logs all data to the database** for historical analysis.

### Key Concept

When an opponent's primary defender or rim protector is OUT/DOUBTFUL, our offensive players face easier matchups → predictions are boosted and **every adjustment is tracked for post-game validation**.

### Visual Indicator

Players benefiting from opponent injuries display the **🚑 emoji** in the Analytics column.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Database Schema](#database-schema)
3. [Example Analysis Queries](#example-analysis-queries)
4. [Usage Guide](#usage-guide)
5. [Technical Implementation](#technical-implementation)
6. [Validation & Tuning](#validation--tuning)

---

## How It Works

### Step 1: Identify Opponent Injuries

For each prediction, the system checks the `injury_list` table for opponent players with status IN ('out', 'doubtful').

```python
injuries = get_opponent_injuries(conn, opponent_team_id, game_date)
```

### Step 2: Calculate Player Importance

For each injured opponent player, calculate importance score (0-1) based on:
- **Minutes:** `min(1.0, avg_minutes / 36.0)` — 36 MPG = full score
- **Scoring:** `min(1.0, avg_points / 25.0)` — 25 PPG = full score
- **Usage:** `min(1.0, avg_usage / 0.30)` — 30% usage = full score

**Formula:**
```python
importance_score = (
    minutes_score * 0.40 +
    points_score * 0.30 +
    usage_score * 0.30
)
```

### Step 3: Determine Defensive Matchup

Check if the injured player directly guards our player's position:

```python
DEFENSIVE_MATCHUPS = {
    'C': ['C'],          # Centers guard centers
    'PF': ['PF', 'C'],   # Power forwards guard PF/C
    'SF': ['SF', 'PF'],  # Small forwards guard SF/PF
    'SG': ['SG', 'SF'],  # Shooting guards guard SG/SF
    'PG': ['PG', 'SG'],  # Point guards guard PG/SG
}
```

### Step 4: Calculate Impact

```python
impact = (
    importance_score *                           # 0-1
    POSITION_DEFENSIVE_IMPORTANCE[position] *    # C=1.0, PF=0.8, SG=0.7, SF/PG=0.6
    matchup_multiplier *                         # 1.5x if direct matchup, 1.0x otherwise
    status_factor                                # OUT=1.0, DOUBTFUL=0.5
)
```

### Step 5: Convert to Boost Percentages

```python
# Sum all opponent injury impacts
total_impact = sum(impacts)

# Convert to capped boost percentages
ceiling_boost = min(0.15, total_impact * 0.50)      # Max 15%
projection_boost = min(0.08, total_impact * 0.27)   # Max 8%
```

### Step 6: Apply Boosts

```python
# Apply to projection
if projection_boost > 0:
    projection *= (1 + projection_boost)

# Apply to ceiling
if ceiling_boost > 0:
    ceiling_multiplier += ceiling_boost

# Add visual indicator
if has_significant_injuries:
    analytics_indicators += "🚑"
```

### Step 7: Log to Database

All opponent injury impact data is saved to the `predictions` table for post-game analysis.

---

## Database Schema

### New Columns in `predictions` Table

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `opponent_injury_detected` | BOOLEAN | Whether opponent injuries were detected | 0 or 1 |
| `opponent_injury_boost_projection` | REAL | Projection boost percentage applied | 0.0 to 0.08 |
| `opponent_injury_boost_ceiling` | REAL | Ceiling boost percentage applied | 0.0 to 0.15 |
| `opponent_injured_player_ids` | TEXT | Comma-separated injured player IDs | NULL or "123,456" |
| `opponent_injury_impact_score` | REAL | Total raw impact score | 0.0 to 2.0+ |

### Migration

The schema is automatically upgraded when Streamlit starts:

```python
pt.upgrade_predictions_table_for_opponent_injury(games_conn)
```

Existing databases get new columns with DEFAULT values (0/NULL), preserving all historical data.

---

## Example Analysis Queries

### 1. View All Games with Opponent Injury Advantage

```sql
SELECT
    game_date,
    player_name,
    opponent_name,
    projected_ppg,
    actual_ppg,
    ROUND(opponent_injury_boost_projection * 100, 1) as proj_boost_pct,
    ROUND(opponent_injury_boost_ceiling * 100, 1) as ceiling_boost_pct,
    ROUND(actual_ppg - projected_ppg, 1) as error,
    CASE
        WHEN actual_ppg BETWEEN proj_floor AND proj_ceiling THEN '✅ Hit'
        ELSE '❌ Miss'
    END as floor_ceiling_hit
FROM predictions
WHERE opponent_injury_detected = 1
  AND actual_ppg IS NOT NULL
ORDER BY game_date DESC, opponent_injury_boost_ceiling DESC;
```

### 2. Validate Opponent Injury Boost Effectiveness

**Question:** Does the opponent injury boost actually improve prediction accuracy?

```sql
SELECT
    CASE
        WHEN opponent_injury_detected = 1 THEN 'With Opp Injury Boost'
        ELSE 'No Opp Injury Boost'
    END as boost_status,
    COUNT(*) as games,
    ROUND(AVG(ABS(projected_ppg - actual_ppg)), 2) as MAE,
    ROUND(AVG(opponent_injury_boost_projection) * 100, 1) as avg_proj_boost_pct,
    ROUND(AVG(opponent_injury_boost_ceiling) * 100, 1) as avg_ceiling_boost_pct,
    ROUND(AVG(CASE
        WHEN actual_ppg BETWEEN proj_floor AND proj_ceiling THEN 100.0
        ELSE 0.0
    END), 1) as floor_ceiling_hit_rate
FROM predictions
WHERE actual_ppg IS NOT NULL
  AND actual_ppg > 0
GROUP BY opponent_injury_detected
ORDER BY opponent_injury_detected DESC;
```

### 3. Top 10 Biggest Opponent Injury Boosts

```sql
SELECT
    game_date,
    player_name,
    opponent_name,
    ROUND(opponent_injury_boost_ceiling * 100, 1) as ceiling_boost_pct,
    ROUND(opponent_injury_boost_projection * 100, 1) as proj_boost_pct,
    ROUND(projected_ppg, 1) as projection,
    ROUND(proj_ceiling, 1) as ceiling,
    ROUND(actual_ppg, 1) as actual
FROM predictions
WHERE opponent_injury_detected = 1
ORDER BY opponent_injury_boost_ceiling DESC
LIMIT 10;
```

### 4. Opponent Injury Boost by Player Position

**Question:** Which positions benefit most from opponent injuries?

```sql
SELECT
    p.position,
    COUNT(*) as games_with_opp_injuries,
    ROUND(AVG(pred.opponent_injury_boost_ceiling) * 100, 1) as avg_ceiling_boost_pct,
    ROUND(AVG(pred.actual_ppg), 1) as avg_actual_ppg,
    ROUND(AVG(pred.projected_ppg), 1) as avg_projected_ppg,
    ROUND(AVG(pred.actual_ppg - pred.projected_ppg), 1) as avg_outperformance
FROM predictions pred
JOIN players p ON pred.player_id = p.player_id
WHERE pred.opponent_injury_detected = 1
  AND pred.actual_ppg IS NOT NULL
GROUP BY p.position
ORDER BY avg_ceiling_boost_pct DESC;
```

### 5. Identify Outliers Explained by Opponent Injuries

**Question:** Did opponent injuries explain our big misses?

```sql
SELECT
    game_date,
    player_name,
    projected_ppg,
    actual_ppg,
    actual_ppg - projected_ppg as error,
    opponent_injury_detected,
    ROUND(opponent_injury_boost_projection * 100, 1) as proj_boost_pct,
    analytics_used
FROM predictions
WHERE actual_ppg IS NOT NULL
  AND ABS(actual_ppg - projected_ppg) > 15  -- Big misses only
ORDER BY ABS(actual_ppg - projected_ppg) DESC
LIMIT 20;
```

---

## Usage Guide

### For Daily DFS Workflow

#### 1. Pre-Game (Morning Lock)

1. Generate predictions in "Today's Games" tab
2. Look for **🚑 emoji** in Analytics column
3. Players with 🚑 = opponent's key defender is OUT
4. Review "Injury Admin" tab to see which opponent players are OUT
5. Prioritize 🚑 players for tournament lineups (higher ceiling potential)

#### 2. Post-Game (Evening/Next Morning)

1. Go to "Prediction Log" tab
2. Filter to yesterday's date
3. Look at "Enhanced Metrics" section
4. Check if 🚑 players hit their elevated ceilings

### For Weekly Analysis

Run validation query to check effectiveness:

```sql
-- Copy from "Validate Opponent Injury Boost Effectiveness" above
```

**Target Metrics:**
- **MAE with boost:** Should be ≤ MAE without boost
- **Floor-ceiling hit rate:** Should be ≥60% (same as baseline)
- **Ceiling utilization:** Players with 🚑 should reach higher percentile outcomes

### For Monthly Tuning

If validation shows poor performance:

1. **Boost too aggressive?** → Reduce multipliers in `opponent_injury_impact.py`
2. **Missing injuries?** → Check injury data quality
3. **Wrong matchups?** → Review `DEFENSIVE_MATCHUPS` matrix
4. **Position weights wrong?** → Adjust `POSITION_DEFENSIVE_IMPORTANCE`

---

## Technical Implementation

### Files Modified/Created

| File | Purpose |
|------|---------|
| `opponent_injury_impact.py` | NEW - Core calculation module (345 lines) |
| `streamlit_app.py` | Integration into projection pipeline (+40 lines) |
| `prediction_tracking.py` | Database schema and logging (+65 lines) |

### Key Functions

#### `calculate_opponent_injury_impact()` (opponent_injury_impact.py:159-258)

**Input:**
- Database connection
- Player position (C, PF, SF, SG, PG)
- Opponent team ID
- Game date
- Season

**Output:**
```python
{
    'has_significant_injuries': bool,
    'ceiling_boost_pct': float,        # 0.0 to 0.15
    'projection_boost_pct': float,     # 0.0 to 0.08
    'total_impact_score': float,       # Raw impact
    'injuries': [                      # List of significant injuries
        {
            'player_name': str,
            'position': str,
            'status': str,
            'importance': float,
            'impact': float,
            'is_direct_matchup': bool
        }
    ],
    'all_injuries': [...],             # All opponent injuries
    'reason': str                      # Summary text
}
```

#### Data Flow

```
1. Prediction Request
        ↓
2. calculate_smart_ppg_projection()
        ↓
3. calculate_opponent_injury_impact()
        ↓
4. Apply boosts to projection/ceiling
        ↓
5. Add 🚑 to analytics_indicators
        ↓
6. Store in breakdown dict
        ↓
7. Create Prediction object
        ↓
8. pt.log_prediction(conn, pred)
        ↓
9. INSERT INTO predictions (...)
        ↓
10. Available in database for analysis
```

### Position Matchup Matrix

```python
DEFENSIVE_MATCHUPS = {
    'C': ['C'],          # Centers guard centers
    'PF': ['PF', 'C'],   # Power forwards guard PF/C
    'SF': ['SF', 'PF'],  # Small forwards guard SF/PF
    'SG': ['SG', 'SF'],  # Shooting guards guard SG/SF
    'PG': ['PG', 'SG'],  # Point guards guard PG/SG
}
```

**Rationale:**
- C guards C: Elite rim protectors (Gobert, Jokic, Embiid) directly impact opposing centers
- PF guards PF/C: Can defend both positions (stretch 4s vs centers, traditional 4s vs 4s)
- SF guards SF/PF: Wing versatility (Kawhi, LeBron, Durant can guard 3/4)
- SG guards SG/SF: Shooting guards defend perimeter (2/3 positions)
- PG guards PG/SG: Point guards primarily defend backcourt

### Position Defensive Importance

```python
POSITION_DEFENSIVE_IMPORTANCE = {
    'C': 1.0,   # Rim protectors have highest defensive impact
    'PF': 0.8,  # Secondary rim protection
    'SG': 0.7,  # Perimeter defense
    'SF': 0.6,  # Perimeter defense
    'PG': 0.6,  # Perimeter defense
}
```

**Rationale:**
- Centers protect the rim (blocks, deterrence, interior defense)
- Power forwards provide help defense and secondary rim protection
- Guards/wings handle perimeter defense (less impact on paint scoring)

---

## Validation & Tuning

### Success Criteria

After collecting 50+ games with opponent injury advantages:

**✅ Validation Passes If:**
1. **MAE with boost ≤ MAE without boost** (boost doesn't hurt accuracy)
2. **Floor-ceiling hit rate ≥ 60%** (maintains prediction range quality)
3. **🚑 players reach 70th+ percentile outcomes** (boosts are meaningful)

**❌ Validation Fails If:**
1. **MAE with boost > MAE without boost** (boost hurts accuracy)
2. **Floor-ceiling hit rate < 55%** (ranges too wide)
3. **🚑 players reach 40th- percentile outcomes** (boosts too aggressive)

### Tuning Parameters

If validation fails, adjust these multipliers in `opponent_injury_impact.py`:

```python
# Line 245-246: Boost conversion
ceiling_boost = min(0.15, total_impact * 0.50)      # Adjust 0.50 multiplier
projection_boost = min(0.08, total_impact * 0.27)   # Adjust 0.27 multiplier

# Line 218: Direct matchup bonus
matchup_multiplier = 1.5 if is_defensive_matchup else 1.0  # Adjust 1.5x

# Line 213-216: Position defensive importance
POSITION_DEFENSIVE_IMPORTANCE = {
    'C': 1.0,   # Adjust importance weights
    'PF': 0.8,
    'SG': 0.7,
    'SF': 0.6,
    'PG': 0.6,
}
```

### Example Scenarios

#### Scenario 1: Jokić vs MIN (Christmas Day)

**Setup:**
- Jokić (C) playing vs Minnesota Timberwolves
- Rudy Gobert (C) is OUT
- Gobert stats: 32 MPG, 14 PPG, 18% usage

**Calculation:**
```python
# Gobert importance
minutes_score = min(1.0, 32/36) = 0.89
points_score = min(1.0, 14/25) = 0.56
usage_score = min(1.0, 0.18/0.30) = 0.60
importance = (0.89*0.4) + (0.56*0.3) + (0.60*0.3) = 0.704

# Impact calculation
position_factor = 1.0  # Center
matchup_multiplier = 1.5  # C guards C (direct matchup)
status_factor = 1.0  # OUT

impact = 0.704 * 1.0 * 1.5 * 1.0 = 1.056

# Boosts
ceiling_boost = min(0.15, 1.056 * 0.50) = 0.15  # Maxed at 15%
projection_boost = min(0.08, 1.056 * 0.27) = 0.08  # Maxed at 8%
```

**Result:**
- 24.9 projection → 26.9 (+8%)
- 42 ceiling → 48 (+15%)
- 🚑 indicator shown
- **Actual:** 56 points (extreme outlier, partially explained)

#### Scenario 2: Guard vs Team with OUT Rim Protector

**Setup:**
- Damian Lillard (PG) vs Team with OUT Center
- NOT a direct matchup (PG doesn't guard C)

**Calculation:**
```python
# Same center importance: 0.704
position_factor = 1.0  # Center
matchup_multiplier = 1.0  # NOT direct (PG guards PG/SG, not C)
status_factor = 1.0

impact = 0.704 * 1.0 * 1.0 * 1.0 = 0.704

# Boosts
ceiling_boost = min(0.15, 0.704 * 0.50) = 0.09  # 9%
projection_boost = min(0.08, 0.704 * 0.27) = 0.05  # 5%
```

**Result:**
- Smaller boost (no direct matchup bonus)
- Still benefits from easier paint access
- 🚑 indicator shown

---

## Future Enhancements

### Tier 1 (High Value)

1. **Store Injured Player Details**
   - Populate `opponent_injured_player_ids` field
   - Enable drill-down: "Which specific injuries affected this prediction?"

2. **Injury Severity Weighting**
   - Star player OUT (30+ MPG) = 1.2x multiplier
   - Role player OUT (15-20 MPG) = 0.8x multiplier

3. **Replacement Player Quality**
   - If OUT player has elite backup → reduce boost
   - If OUT player has poor backup → increase boost

### Tier 2 (Advanced)

1. **Defensive Metric Integration**
   - Factor in DRPM, DPIPM, defensive EPM
   - Elite defenders (99th percentile) = 1.3x multiplier

2. **Historical Matchup Data**
   - "Player X scores 8 more PPG when Defender Y is OUT"
   - Personalized injury impact scores

3. **Injury Timeline Tracking**
   - Track how long defender has been OUT
   - Fresh injury (1-2 games) vs extended absence (10+ games)

---

## Changelog

### v1.0 - December 26, 2025 (Initial Release)

**Added:**
- `opponent_injury_impact.py` module (345 lines)
- Database schema for opponent injury tracking
- Integration with prediction pipeline
- 🚑 visual indicator
- Automatic database migration
- Ceiling confidence boost (+4 to +12 points)

**Commits:**
- `694793d` - Add opponent injury impact analysis to predictions
- `282f128` - Add opponent injury impact tracking to predictions database

---

## References

- **Module:** `opponent_injury_impact.py`
- **Integration:** `streamlit_app.py` lines 1800-1834, 2105-2114, 2127-2129
- **Database:** `prediction_tracking.py` lines 150-155, 247-270
- **Documentation:** `PREDICTION_ACCURACY_ENHANCEMENTS.md` Section 7

---

## Support

For questions or issues:
1. Check validation queries to ensure data is being logged correctly
2. Review `opponent_injury_impact.py` for calculation logic
3. Run migration: `pt.upgrade_predictions_table_for_opponent_injury(conn)`
4. Check Streamlit logs for errors during prediction generation

**Example Debug Query:**
```sql
SELECT COUNT(*) as total_predictions,
       SUM(opponent_injury_detected) as with_opp_injuries,
       ROUND(100.0 * SUM(opponent_injury_detected) / COUNT(*), 1) as injury_rate_pct
FROM predictions
WHERE game_date >= '2025-12-01';
```

Expected: 5-15% of predictions should have opponent injury advantages (varies by date).
