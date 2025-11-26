# Unified Scoring System for Finding Best Daily Picks

## Problem Analysis

### Current State (Contradictory Systems)

**System 1: Opp Difficulty**
- Input: Opponent's `def_composite_score`
- Logic: Compares opponent defense to league average (percentile-based)
- Output: "Favorable" | "Neutral" | "Hard"
- Issue: **Ignores player-specific matchup history**

**System 2: Matchup Rating**
- Input: Player's avg vs this team OR vs this defense style
- Logic: Compares player's matchup performance to season average
- Output: "Excellent" | "Good" | "Neutral" | "Difficult" | "Avoid"
- Issue: **Ignores opponent's overall defensive quality**

### Why This Creates Contradictions

Example from your screenshot:
```
Player A vs Opponent X:
- Opp Difficulty: "Neutral" (Opponent has league-average defense)
- Matchup Rating: "Good" (Player scores 35.2 vs this style, season avg 30.0)
- Result: Confusing! Which metric should I trust?
```

The player excels vs this STYLE of defense, but the opponent is average overall.
Both can be true, but users don't know which matters more.

## Solution: Unified Daily Pick Score

### Design Principles

1. **Single Primary Metric**: "Daily Pick Score" (0-100)
2. **All Factors Combined**: Player ability + Matchup history + Opponent defense + Recent form + Pace
3. **Clear Interpretation**: Higher = better opportunity
4. **Confidence-Adjusted**: Show reliability of the score

### Unified Daily Pick Score Formula

```python
def calculate_daily_pick_score(
    player_season_avg: float,
    player_projection: float,  # From our smart projection
    projection_confidence: float,  # 0-1
    matchup_rating: str,  # Excellent/Good/Neutral/Difficult/Avoid
    opp_def_rating: float,  # Opponent's def rating
    league_avg_def_rating: float = 112.0,
) -> tuple[float, str, str]:
    """
    Calculate unified Daily Pick Score (0-100) combining all factors.

    Returns:
        (score, grade, explanation)
    """

    # Base score: Projected points (0-50 range, scaled from typical 0-45 PPG)
    base_score = min(50, (player_projection / 45.0) * 50)

    # Matchup bonus/penalty (-20 to +20)
    matchup_adjustments = {
        "Excellent": 20,
        "Good": 10,
        "Neutral": 0,
        "Difficult": -10,
        "Avoid": -20,
    }
    matchup_bonus = matchup_adjustments.get(matchup_rating, 0)

    # Defense adjustment (-10 to +10)
    # Weaker defense = bonus, stronger defense = penalty
    def_diff = (league_avg_def_rating - opp_def_rating) / league_avg_def_rating
    defense_adjustment = def_diff * 10  # Scale to -10 to +10

    # Confidence multiplier (0.5 to 1.0)
    # Low confidence = reduce the score
    confidence_multiplier = 0.5 + (projection_confidence * 0.5)

    # Calculate final score
    raw_score = base_score + matchup_bonus + defense_adjustment
    final_score = raw_score * confidence_multiplier

    # Grade the score
    if final_score >= 80:
        grade = "🔥 Elite"
        explanation = "Exceptional opportunity - high projection with favorable matchup"
    elif final_score >= 65:
        grade = "⭐ Excellent"
        explanation = "Strong opportunity - good projection and matchup"
    elif final_score >= 50:
        grade = "✓ Solid"
        explanation = "Decent opportunity - reliable pick"
    elif final_score >= 35:
        grade = "⚠️ Risky"
        explanation = "Below average opportunity - consider alternatives"
    else:
        grade = "❌ Avoid"
        explanation = "Poor opportunity - unfavorable matchup or low projection"

    return (final_score, grade, explanation)
```

### Example Calculations

**Elite Pick (Score: 85)**
- Player: Stephen Curry (season avg 28.5)
- Projection: 32.0 PPG (confidence 88%)
- Matchup Rating: "Excellent" (+20)
- Opponent Def Rating: 118 (weak defense) (+5)
- Calculation: (32/45)*50 + 20 + 5 = 60.5 * 0.94 = **85 points**

**Risky Pick (Score: 38)**
- Player: Role player (season avg 12.0)
- Projection: 10.5 PPG (confidence 45%)
- Matchup Rating: "Difficult" (-10)
- Opponent Def Rating: 106 (elite defense) (-5)
- Calculation: (10.5/45)*50 + (-10) + (-5) = 11.7 * 0.725 = **38 points**

## Implementation Plan

### Phase 1: Add Daily Pick Score (Core)

```python
# In the matchup loop, after calculating projection
daily_pick_score, pick_grade, pick_explanation = calculate_daily_pick_score(
    player_season_avg=season_avg_pts,
    player_projection=projection,
    projection_confidence=proj_confidence,
    matchup_rating=matchup_rating,
    opp_def_rating=opp_def_rating or league_avg_def_rating,
)

# Add to matchup_rows
matchup_rows.append({
    # ... existing columns ...
    "Daily Pick Score": round(daily_pick_score, 1),
    "Pick Grade": pick_grade,
    # ... rest of columns ...
})
```

### Phase 2: Reorder Table for Decision-Making

**Priority Column Order:**
```python
primary_cols = [
    "Player",
    "Team",
    "Pick Grade",           # 🔥/⭐/✓/⚠️/❌
    "Daily Pick Score",     # 0-100
    "Projected PPG",        # Smart projection
    "Proj Conf",           # Confidence %
    "Season Avg PPG",      # Baseline
]

secondary_cols = [
    # Details (collapsible)
    "Matchup Rating",
    "Warning",
    "Vs This Team",
    "Vs This Style",
    "Opp Defense Style",
    "Opp Difficulty",      # Keep for context, but de-emphasize
    "Last5 Avg Pts",
    "Last3 Avg Pts",
    # ... etc
]
```

### Phase 3: Sort by Daily Pick Score

```python
# Default sort
matchup_df = matchup_df.sort_values("Daily Pick Score", ascending=False)

# Show top picks prominently
st.markdown("### 🎯 Today's Top Picks")
top_picks = matchup_df.head(10)
st.dataframe(top_picks[primary_cols], use_container_width=True)

st.markdown("### All Matchups")
st.dataframe(matchup_df, use_container_width=True)
```

### Phase 4: Add Summary Dashboard

```python
# At top of page
st.title("NBA Daily Picks - [Date]")

col1, col2, col3, col4 = st.columns(4)
with col1:
    elite_picks = len(matchup_df[matchup_df['Pick Grade'].str.contains('Elite')])
    st.metric("🔥 Elite Picks", elite_picks)

with col2:
    excellent_picks = len(matchup_df[matchup_df['Pick Grade'].str.contains('Excellent')])
    st.metric("⭐ Excellent Picks", excellent_picks)

with col3:
    avg_pick_score = matchup_df['Daily Pick Score'].mean()
    st.metric("Avg Pick Quality", f"{avg_pick_score:.1f}/100")

with col4:
    high_conf = len(matchup_df[matchup_df['Proj Conf'].str.replace('%','').astype(float) >= 70])
    st.metric("High Confidence", f"{high_conf}/{len(matchup_df)}")
```

## Expected User Flow

### Before (Confusing):
1. User sees "Neutral" difficulty but "Good" rating → confused
2. Scans multiple columns trying to decide
3. Unclear which metric to prioritize
4. Misses best opportunities buried in data

### After (Clear):
1. User sees "Daily Pick Score: 85" with "🔥 Elite" grade → clear signal
2. Top picks automatically sorted at top
3. Single number combines all factors
4. Confidence shows reliability
5. Details available for those who want to dig deeper

## Handling Edge Cases

### Case 1: High projection, low confidence
- Projection: 30 PPG
- Confidence: 35%
- Result: Score penalized to ~45 (Risky) due to low confidence multiplier
- Message: "High upside but uncertain - speculative pick"

### Case 2: Low projection, elite matchup
- Projection: 12 PPG (role player)
- Matchup: "Excellent" (+20 bonus)
- Result: Score ~35 (Risky) - low ceiling even with good matchup
- Message: "Favorable matchup but limited upside"

### Case 3: Star player, bad matchup
- Projection: 22 PPG (star struggling vs this defense)
- Matchup: "Avoid" (-20 penalty)
- Confidence: 85%
- Result: Score ~40 (Risky) - clear warning despite being a star
- Message: "Proven history of struggling in this matchup"

## Benefits of Unified System

1. **Single Source of Truth**: One score, one decision
2. **All Factors Considered**: Nothing left out
3. **Confidence-Adjusted**: Honest about uncertainty
4. **Sortable**: Easy to find best picks
5. **Explainable**: Users understand what went into the score
6. **Actionable**: Clear grades (Elite/Excellent/Solid/Risky/Avoid)

## Migration Strategy

### Keep Old Metrics for Context
Don't remove "Opp Difficulty" or "Matchup Rating" - they provide useful detail.
Just de-emphasize them in favor of Daily Pick Score.

### Add Explanation Column
```python
# Optional detailed view
if st.checkbox("Show Score Breakdown"):
    matchup_df['Score Breakdown'] = matchup_df.apply(
        lambda row: f"Base: {row['Projected PPG']:.1f} PPG | "
                    f"Matchup: {row['Matchup Rating']} | "
                    f"Defense: {row['Opp Difficulty']} | "
                    f"Confidence: {row['Proj Conf']}",
        axis=1
    )
```

## Alternative: Keep Both Systems But Clarify

If you prefer to keep both systems distinct:

1. **Rename for Clarity:**
   - "Opp Difficulty" → "Opponent Defensive Rank"
   - "Matchup Rating" → "Player-Specific Matchup History"

2. **Add Explanatory Text:**
   ```
   ℹ️ How to read this:
   - Opponent Defensive Rank: How good is this defense vs the league?
   - Player Matchup History: How does THIS player perform vs THIS team/style?
   - Both matter! A player can excel vs a weak defense OR struggle vs a strong one.
   ```

3. **Color Code Consistency:**
   - Green: Both favorable
   - Yellow: Mixed signals
   - Red: Both unfavorable
