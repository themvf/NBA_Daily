# PPM Integration Guide - Phase 2

## Overview
This guide documents how to complete the full PPM (Points Per Minute) integration into the NBA Daily prediction system.

## ✅ Completed (Phase 1)
1. **PPM Stats Module** (`ppm_stats.py`) - Reusable library
   - `get_team_offensive_ppm_stats()` - Team offensive PPM
   - `get_team_defensive_ppm_stats()` - Team defensive PPM
   - `get_player_ppm_stats()` - Player PPM consistency
   - `get_league_avg_ppm()` - League average

2. **Player PPM in Prediction Log** - Display only
   - PPM, Consistency Score, PPM Grade columns added
   - Shows in Streamlit Prediction Log tab

3. **Defensive PPM Analysis** (`analyze_defensive_ppm.py`)
   - Standalone analysis tool
   - Exports defensive_ppm_stats.csv

## 📋 Remaining Work (Phase 2)

### Step 1: Integrate PPM into Prediction Logic

**File:** `prediction_refresh.py` (or equivalent prediction generation file)

**Location to modify:** Find where predictions are generated and opponent defense is factored in.

**Current logic (example):**
```python
# OLD: Uses defensive rating
if opponent_def_rating > 115:
    boost = +2.0
    projected_ppg += boost
```

**New PPM-based logic:**
```python
# Import PPM module at top of file
import ppm_stats

# Cache PPM stats (load once per prediction run, not per player)
# Add this near the start of the prediction function
def generate_predictions(conn, game_date):
    # Cache defensive PPM stats for all teams
    def_ppm_stats = ppm_stats.get_team_defensive_ppm_stats(conn)
    league_avg_ppm = ppm_stats.get_league_avg_ppm(conn)

    # For each player prediction:
    for player in players_to_predict:
        # Get opponent defensive PPM
        opponent_def_ppm = def_ppm_stats[
            def_ppm_stats['team_id'] == player_opponent_id
        ]['avg_def_ppm'].iloc[0]

        # Get player's PPM (from season stats or ppm_stats module)
        player_season_ppm = player_total_points / player_total_minutes

        # Calculate expected minutes (from existing logic)
        expected_minutes = ...  # Your existing minute projection

        # Calculate PPM boost factor
        ppm_boost_factor = opponent_def_ppm / league_avg_ppm

        # New projection using PPM
        projected_ppg = expected_minutes * player_season_ppm * ppm_boost_factor

        # Adjust floor/ceiling based on player PPM consistency
        player_ppm_data = ppm_stats.get_player_ppm_stats(conn, player_name)
        if player_ppm_data:
            consistency_score = player_ppm_data['consistency_score']

            # High consistency = tighter range
            if consistency_score >= 75:  # A/A+ grade
                ceiling_multiplier = 1.8  # Tighten ceiling
                floor_multiplier = 0.6
            # Low consistency = wider range
            elif consistency_score < 55:  # D grade
                ceiling_multiplier = 2.3  # Widen ceiling
                floor_multiplier = 0.4
            else:  # Average
                ceiling_multiplier = 2.0
                floor_multiplier = 0.5

            proj_ceiling = projected_ppg + (ceiling_multiplier * historical_std_dev)
            proj_floor = projected_ppg - (floor_multiplier * historical_std_dev)
```

**Key Changes:**
1. Import `ppm_stats` at top of file
2. Cache defensive PPM stats once per run (not per player)
3. Replace opponent_def_rating logic with opponent_def_ppm
4. Use PPM boost factor instead of fixed +2.0 boosts
5. Adjust floor/ceiling based on player PPM consistency

---

### Step 2: Add PPM Columns to Streamlit

#### A. Defense Mix Tab

**File:** `streamlit_app.py`

**Location:** Find the "Defense Mix" tab section

**Changes:**
```python
# Add after existing defense data loading
import ppm_stats

# Cache PPM stats
@st.cache_data(ttl=3600)
def load_defensive_ppm_stats():
    conn = get_connection()
    return ppm_stats.get_team_defensive_ppm_stats(conn)

def_ppm_df = load_defensive_ppm_stats()

# Merge with existing defense dataframe
defense_df = defense_df.merge(
    def_ppm_df[['team_id', 'avg_def_ppm', 'def_ppm_grade', 'ceiling_factor']],
    on='team_id',
    how='left'
)

# Add to display columns
display_df['Def PPM'] = defense_df['avg_def_ppm']
display_df['Def PPM Grade'] = defense_df['def_ppm_grade']
display_df['Ceiling Factor'] = defense_df['ceiling_factor']
```

#### B. Today's Games Tab

**File:** `streamlit_app.py`

**Location:** Find "Today's Games" tab

**Changes:**
```python
# For each game, show PPM matchup data
import ppm_stats

off_ppm_df = ppm_stats.get_team_offensive_ppm_stats(conn)
def_ppm_df = ppm_stats.get_team_defensive_ppm_stats(conn)

# For each game display:
home_off_ppm = off_ppm_df[off_ppm_df['team_id'] == home_team_id]['avg_off_ppm'].iloc[0]
away_def_ppm = def_ppm_df[def_ppm_df['team_id'] == away_team_id]['avg_def_ppm'].iloc[0]

# Show matchup advantage
ppm_advantage = home_off_ppm - away_def_ppm
if ppm_advantage > 0.05:
    st.success(f"Home offense advantage: {ppm_advantage:.3f} PPM")
elif ppm_advantage < -0.05:
    st.warning(f"Away defense advantage: {abs(ppm_advantage):.3f} PPM")
```

#### C. Tournament Strategy Tab

**File:** `streamlit_app.py`

**Location:** Find Tournament Strategy / GPP Score calculation

**Current GPP Score calculation:**
```python
# Component 3: Defense Variance Bonus (0-15 points)
# OLD: Uses old ceiling factor calculation
```

**New GPP Score with Defensive PPM:**
```python
# Component 3: Defense Variance Bonus (0-15 points)
# Uses defensive PPM ceiling factor

def_ppm_df = ppm_stats.get_team_defensive_ppm_stats(conn)
opponent_ceiling_factor = def_ppm_df[
    def_ppm_df['team_id'] == opponent_team_id
]['ceiling_factor'].iloc[0]

if opponent_ceiling_factor >= 1.15:
    variance_bonus = 15  # Elite variance (OKC, POR)
elif opponent_ceiling_factor >= 1.12:
    variance_bonus = 10  # Good variance (MEM, DAL)
elif opponent_ceiling_factor >= 1.10:
    variance_bonus = 6   # Above average
elif opponent_ceiling_factor >= 1.08:
    variance_bonus = 3   # Slight edge
else:
    variance_bonus = 0   # Tight defense

# Add to GPP Score
gpp_score = ceiling_base + hot_streak_bonus + variance_bonus + defense_adjustment + injury_bonus
```

**Also add Ceiling Factor column to display:**
```python
# Add to tournament display dataframe
display_df['Opp Ceiling Factor'] = ...
display_df['Opp Def PPM Grade'] = ...
```

---

### Step 3: Testing Checklist

**Before deploying:**

1. **Test PPM calculations:**
   ```bash
   python ppm_stats.py
   ```
   - Verify offensive PPM: Denver ~0.521, Miami ~0.509
   - Verify defensive PPM: OKC ~0.438, Houston ~0.450
   - Verify league avg: ~0.462

2. **Test prediction generation:**
   ```bash
   python prediction_refresh.py --date 2025-12-12
   ```
   - Check that projected_ppg values are reasonable
   - Verify PPM boost logic is working
   - Compare old vs new projections for sanity check

3. **Test Streamlit display:**
   ```bash
   streamlit run streamlit_app.py
   ```
   - Check Defense Mix tab has Def PPM columns
   - Check Today's Games shows PPM matchups
   - Check Tournament Strategy has ceiling factor
   - Verify PPM data loads without errors

4. **Validate accuracy:**
   - Run predictions for one game date
   - Wait for actuals to come in
   - Compare PPM-based predictions vs old system
   - Calculate RMSE for both methods

---

### Step 4: Commit Strategy

**Commit 1: Prediction logic**
```bash
git add prediction_refresh.py
git commit -m "Integrate PPM into prediction logic - replace def rating with def PPM"
```

**Commit 2: Streamlit display**
```bash
git add streamlit_app.py
git commit -m "Add PPM columns to Defense Mix, Today's Games, and Tournament tabs"
```

**Commit 3: GPP Score update**
```bash
git add streamlit_app.py
git commit -m "Update GPP Score to use defensive PPM ceiling factor"
```

---

## Expected Results

### Projection Changes
**Old System:**
- Franz Wagner vs Washington: 28.5 PPG (fixed +2.0 boost for bad defense)

**New PPM System:**
- Franz Wagner: 0.726 PPM (player)
- Washington: 0.529 PPM allowed (defense)
- League Avg: 0.462 PPM
- Expected minutes: 35
- Boost factor: 0.529 / 0.462 = 1.145 (14.5% boost)
- Projected: 35 * 0.726 * 1.145 = 29.1 PPG

**Difference:** More accurate, accounts for actual defensive weakness

### Floor/Ceiling Adjustments
**Jaylen Brown (A grade, 81.8 consistency):**
- Old: Floor 22.0, Ceiling 34.0 (±6 range)
- New: Floor 23.5, Ceiling 32.5 (±4.5 range, 25% tighter)

**Franz Wagner (B grade, 74.7 consistency):**
- Old: Floor 20.0, Ceiling 37.0 (±8.5 range)
- New: Floor 20.0, Ceiling 37.0 (unchanged, average consistency)

---

## Troubleshooting

### Issue: PPM stats load slowly
**Solution:** Cache results with `@st.cache_data(ttl=3600)`

### Issue: Player PPM not found
**Solution:** Default to season_ppg / season_minutes if ppm_stats returns None

### Issue: Team ID mismatch
**Solution:** Verify team_id joins between predictions and ppm_stats dataframes

### Issue: Projections seem too high/low
**Solution:** Check league_avg_ppm is correct (~0.462), verify boost factor calculation

---

## Rollback Plan

If PPM integration causes issues:

1. **Quick fix:** Comment out PPM logic, revert to old def_rating
2. **Full rollback:** `git revert <commit-hash>`
3. **Hybrid approach:** Keep PPM display, don't use in predictions yet

---

## Success Metrics

**After 1 week of PPM-based predictions:**
- RMSE should improve by 5-10%
- Prediction accuracy (within floor/ceiling) should increase
- Target defenses (D/D- grade) should have higher scoring actuals
- Elite defenses (A+/A grade) should have lower scoring actuals

---

## Contact Points

**Key files to modify:**
- `prediction_refresh.py` - Prediction generation
- `streamlit_app.py` - Display and GPP Score
- `ppm_stats.py` - PPM calculations (already complete)

**Test files:**
- `test_prediction_system.py` - Add PPM validation tests
- `score_predictions.py` - Compare old vs new accuracy

**Documentation:**
- This file (PPM_INTEGRATION_GUIDE.md)
- TOURNAMENT_GPP_SCORE_GUIDE.md - Update with PPM ceiling factor

---

*Last updated: 2025-12-11*
*Phase 1 complete, Phase 2 pending implementation*
