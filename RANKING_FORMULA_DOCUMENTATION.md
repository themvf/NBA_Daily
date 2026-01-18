# Top-3 Scorer Ranking Formula Documentation

This document explains the mathematical formulas used to compute `p_top1`, `p_top3`, and the sorting logic for daily top-3 scorer predictions.

---

## Table of Contents

1. [Overview](#overview)
2. [Strategy Definitions](#strategy-definitions)
3. [Backtest Sorting Logic](#backtest-sorting-logic)
4. [Formula 1: TopScorerScore (Heuristic)](#formula-1-topscorerscore-heuristic)
5. [Formula 2: Monte Carlo Simulation](#formula-2-monte-carlo-simulation)
6. [Variance Estimation (σ)](#variance-estimation-σ)
7. [Known Bugs / Investigation Needed](#known-bugs--investigation-needed)

---

## Overview

The system provides two ranking methods:

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| `top_scorer_score` | Fast | Heuristic-based | Quick daily picks |
| `sim_p_top3` | Slower | Monte Carlo simulation | Tournament optimization |

Both methods ultimately rank players by a single numeric field and sort **descending**.

---

## Strategy Definitions

From `backtest_top3.py` lines 29-50:

```python
STRATEGIES = {
    'projection_only': {
        'ranking_field': 'projected_ppg',
        'description': 'Baseline: rank by projected PPG'
    },
    'ceiling_only': {
        'ranking_field': 'proj_ceiling',
        'description': 'Rank by scoring ceiling'
    },
    'top_scorer_score': {
        'ranking_field': 'top_scorer_score',
        'description': 'TSS heuristic formula'
    },
    'sim_p_top3': {
        'ranking_field': 'p_top3',
        'description': 'Monte Carlo P(top-3)'
    },
    'sim_p_first': {
        'ranking_field': 'p_top1',
        'description': 'Monte Carlo P(#1)'
    }
}
```

---

## Backtest Sorting Logic

From `backtest_top3.py` lines 276-293:

```python
def rank_predictions(preds: pd.DataFrame, ranking_field: str) -> pd.DataFrame:
    """
    Rank predictions by the specified field.
    Adds 'pred_rank' column (1 = highest ranked).
    """
    if ranking_field not in preds.columns:
        # ⚠️ FALLBACK: If field doesn't exist, uses projected_ppg
        ranking_field = 'projected_ppg'

    # Sort descending by ranking field
    ranked = preds.sort_values(ranking_field, ascending=False).reset_index(drop=True)

    # Assign ranks (1-indexed)
    ranked['pred_rank'] = range(1, len(ranked) + 1)

    return ranked
```

### Critical Insight

**If `p_top3` column is NULL/missing for a player, they get ranked by `projected_ppg` instead.**

This could explain why stars with good `Proj/Ceil/σ` are ranked #76 — if `p_top3` is NULL, the sort falls back to `projected_ppg`, and the player might have a low projection due to another bug.

---

## Formula 1: TopScorerScore (Heuristic)

From `top3_ranking.py` lines 557-747:

### Total Formula

```
TopScorerScore = calibrated_base
               + ceiling_bonus        (0 to +8)
               + role_score          (-3 to +4)
               + matchup_bonus       (-3 to +5)
               + injury_boost        (0 to +6)
               + star_bonus          (0 to +3)
               - risk_penalty        (0 to -10)
```

### Component Breakdown

#### 1. Calibrated Base (~60% of score)
```python
calibrated_ppg = cal_intercept + cal_slope * projected_ppg
```
- Uses linear calibration from historical prediction accuracy
- Default: `(0.0, 1.0)` = no adjustment

#### 2. Ceiling Bonus (up to +8)
```python
ceiling_upside = proj_ceiling - projected_ppg
proj_factor = min(projected_ppg / 25.0, 1.0)  # 0-1 for 0-25 PPG
ceiling_bonus = min(ceiling_upside * 0.5 * proj_factor, 8.0)
```
- **Problem**: If `proj_ceiling` is 0 or NULL, this produces 0 or negative bonus

#### 3. Role Sustainability Score (ValidityFactor pattern)
```python
hot_ratio = recent_avg_5 / season_avg_ppg

# Raw bonus
if hot_ratio > 1.05: raw_hot_bonus = min((hot_ratio - 1.0) * 20, 4.0)
elif hot_ratio < 0.90: raw_hot_bonus = max((hot_ratio - 1.0) * 15, -3.0)

# ValidityFactor crushes fake streaks
validity_factor = 1.0
if hot_ratio > 1.10 and not is_injury_beneficiary_today:
    validity_factor *= 0.25  # ⚠️ Punishes hot streak without injury context
if proj_confidence < 0.6 and hot_ratio > 1.10:
    validity_factor *= 0.5

role_score = raw_hot_bonus * validity_factor
```
- **Potential Bug**: If `season_avg_ppg = 0`, entire calculation divides by zero or skips

#### 4. Matchup Bonus (-3 to +5)
```python
if opp_def_rating >= 116: matchup_bonus = +5.0   # Bad defense
elif opp_def_rating >= 114: matchup_bonus = +3.0
elif opp_def_rating >= 112: matchup_bonus = +1.0
elif opp_def_rating <= 108: matchup_bonus = -3.0  # Elite defense
elif opp_def_rating <= 110: matchup_bonus = -1.5
```

#### 5. Star Power Bonus (0 to +3)
```python
if season_avg_ppg >= 28: star_bonus = 3.0   # Elite star
elif season_avg_ppg >= 25: star_bonus = 2.0
elif season_avg_ppg >= 22: star_bonus = 1.0
```
- **Problem**: If `season_avg_ppg` is 0/NULL, star gets NO bonus

---

## Formula 2: Monte Carlo Simulation

From `top3_ranking.py` lines 794-969:

### Simulation Loop

```python
for _ in range(n_simulations):  # Default: 5000

    # 1. Game-level factor (affects all players in same game)
    game_factor = np.random.normal(implied_total/220, 0.08)

    # 2. Sample individual scores
    for each player:
        base_mean = calibrated_ppg * game_factor
        sampled_points = max(0, np.random.normal(base_mean, stddev))

    # 3. Teammate cannibalization (negative correlation)
    if any_teammate_spikes > 1.2x mean:
        other_teammates *= 0.95

    # 4. Count who finishes top-3 / top-1
    top3_counts[top3_players] += 1
    top1_counts[top1_player] += 1

# Final probabilities
p_top3 = top3_counts / n_simulations
p_top1 = top1_counts / n_simulations
```

### Key Insight

**P(top-3) is determined by:**
1. **Mean** = calibrated projection
2. **Stddev** = `estimate_scoring_stddev()` — see next section

If `stddev = 0`, the player samples the same value every time and loses variance advantage for tail events.

---

## Variance Estimation (σ)

From `top3_ranking.py` lines 339-394:

### Hierarchy

```python
def estimate_scoring_stddev(player, calibration_table):
    # 1. Try calibrated stddev from historical residuals
    if calibration_table has data for (tier, minutes_band):
        base_std = historical_residual_stddev

    # 2. Fallback: ceiling-floor spread
    elif ceiling > floor:
        base_std = (ceiling - floor) / 3.29  # 99% CI

    # 3. Last resort: 25% of projection
    else:
        base_std = projected_ppg * 0.25

    # Adjustments
    if minutes_confidence < 0.5:
        base_std *= 1.3  # Volatile minutes
    if injury_beneficiary and not role_change:
        base_std *= 1.2  # Temporary role

    return max(base_std, 2.0)  # Floor of 2.0
```

### What Drives σ

| Input | Impact | If Missing |
|-------|--------|------------|
| `proj_ceiling` | Primary | Falls to 25% of proj |
| `proj_floor` | Primary | Falls to 25% of proj |
| `minutes_confidence` | +30% if < 0.5 | Ignored |
| `injury_adjusted` | +20% if temp role | Ignored |

---

## Known Bugs / Investigation Needed

### Bug Hypothesis 1: Missing `p_top3` → Fallback Sort

If `p_top3` is NULL for a player when using `sim_p_top3` strategy:
- `rank_predictions()` falls back to sorting by `projected_ppg`
- Player rank determined by raw projection, not simulation probability

**How to verify**: Check if Murray/Shai have NULL `p_top3` in the predictions table for Jan 17.

### Bug Hypothesis 2: Missing `season_avg_ppg`

If `season_avg_ppg = 0` or NULL:
- Star bonus = 0 (should be +3)
- Hot streak detection divides by zero or skips
- Validity factor logic may misbehave

**How to verify**: Check `SznAvg` column in drilldown for Murray/Shai.

### Bug Hypothesis 3: Zero Variance (σ = 0)

If `proj_ceiling = 0` and `proj_floor = 0`:
- `σ = (0 - 0) / 3.29 = 0`
- Falls to `proj_ppg * 0.25`, but if `proj_ppg` is also 0 → `σ = 0`
- Monte Carlo samples same value every time → low P(top-3)

**How to verify**: Check `Ceil` and `σ` columns in drilldown.

### Bug Hypothesis 4: Simulation Not Run

The simulation (`simulate_top3_probability`) only runs when:
1. User triggers it via Streamlit
2. `Top3Ranker.rank_players(method='simulation')` is called

If predictions are stored WITHOUT running simulation, `p_top3` stays NULL.

**How to verify**: Check if `p_top3` / `p_top1` columns exist and are populated.

---

## Diagnostic Checklist

For any star ranked #40+ despite good fundamentals, check:

| Field | Expected | Bug Indicator |
|-------|----------|---------------|
| `p_top3` | > 0.05 for stars | NULL or 0 |
| `season_avg_ppg` | 25+ for All-Stars | 0 or NULL |
| `proj_ceiling` | 40+ for stars | 0 or NULL |
| `proj_floor` | 15+ for starters | 0 or NULL |
| `σ` (sigma) | 5+ for high-variance | 0 or < 2 |
| `Proj` | Within 20% of SznAvg | << SznAvg |

---

## File References

- **Ranking strategies**: `backtest_top3.py:29-50`
- **Sort logic**: `backtest_top3.py:276-293`
- **TopScorerScore formula**: `top3_ranking.py:557-747`
- **Monte Carlo simulation**: `top3_ranking.py:794-969`
- **Variance estimation**: `top3_ranking.py:339-394`
- **Tier classification**: `top3_ranking.py:148-189`

---

*Generated for debugging star-burial issue (Murray #76, Shai #41, Wemby #56)*
