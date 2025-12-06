# Tournament GPP Score System - User Guide

## Overview

The **GPP Score** (Guaranteed Prize Pool Score) is a tournament-optimized scoring system designed for winner-take-all DFS contests. Unlike the standard DFS Score (which optimizes for accuracy in cash games), the GPP Score maximizes **ceiling potential** and identifies **hot streaks** and **high-variance matchups**.

---

## How GPP Score is Calculated

### Total Range: 0-100 points

The GPP Score combines **6 weighted components**:

### 1. **Ceiling Base Score (0-50 points)** - 50% weight

**Most important component** - Measures explosive scoring potential.

| Ceiling Range | Points | Tier |
|--------------|--------|------|
| 50+ PPG | 50 | Elite |
| 45-49 PPG | 46 | Monster |
| 40-44 PPG | 42 | High |
| 37-39 PPG | 38 | Above Avg |
| 35-36 PPG | 35 | Solid |
| 32-34 PPG | 30 | Decent |
| 30-31 PPG | 25 | Average |

**Why it matters:** In winner-take-all, you need players who can explode for 40-50+ points, not consistent 25-point performances.

---

### 2. **Hot Streak Bonus (0-20 points)** - 20% weight

Compares **Last 5 Games Average (L5)** vs **Season Average**.

| L5 vs Season | Points | Status |
|-------------|--------|--------|
| +25% or more | 20 | ON FIRE 🔥🔥 |
| +15% to +24% | 15 | Very Hot 🔥 |
| +10% to +14% | 12 | Hot |
| +5% to +9% | 10 | Warm |
| -5% to +4% | 5 | Neutral |
| Below -5% | 0 | Cold ❄️ |

**Example:**
- Jaylen Brown: Season avg 27.8, L5 avg 34.0 → +22% → **15 points bonus**
- Giannis: Season avg 25.1, L5 avg 20.2 → -20% → **0 points** (FADE!)

**Why it matters:** Recent form is predictive of future explosions. A player averaging +25% above season average is likely to continue the hot streak.

---

### 3. **Opponent Defense Variance Bonus (0-15 points)** - 15% weight

Based on **Ceiling Factor** = (90th percentile points allowed) / (average points allowed)

| Ceiling Factor | Points | Matchup Type |
|---------------|--------|--------------|
| 1.15+ | 15 | Elite Variance (OKC, POR) |
| 1.12-1.14 | 10 | Good Variance (MEM, DAL) |
| 1.10-1.11 | 6 | Above Average |
| 1.08-1.09 | 3 | Slight Edge |
| Below 1.08 | 0 | Tight Defense |

**Example:**
- Oklahoma City Thunder: Avg 107.6 allowed, 90th% 124.9 → Factor 1.160 → **15 points**
- Portland Trail Blazers: Avg 113.9 allowed, 90th% 132.0 → Factor 1.159 → **15 points**

**Why it matters:** Some defenses are inconsistent and occasionally allow 40-50+ point explosions. These are GOLD for tournaments.

---

### 4. **Matchup History Bonus (0-10 points)** - 10% weight

Rewards players with proven success against specific opponents.

| Career vs Opp | Points |
|--------------|--------|
| 35+ PPG | 10 |
| 30-34 PPG | 7 |
| 25-29 PPG | 5 |
| 20-24 PPG | 3 |

**Why it matters:** Some players consistently dominate certain teams (revenge games, stylistic advantages).

---

### 5. **Defense Quality Adjustment (±5 points)** - 5% weight

Minor bonus/penalty based on opponent's defensive rating vs league average (112.0).

| Defense Rating | Adjustment |
|---------------|-----------|
| 118+ (very bad) | +5 |
| 115-117 (bad) | +3 |
| 109-114 (avg) | 0 |
| 106-108 (good) | -3 |
| <106 (elite) | -5 |

**Why it matters:** Small tiebreaker for elite vs weak defenses.

---

### 6. **Injury Beneficiary Bonus (0-12 points)** - 12% weight 🔥

**CRITICAL FOR TOURNAMENTS** - Players who get boosted usage/minutes when teammates are injured.

| Projection Boost | Points | Flag |
|-----------------|--------|------|
| 5.0+ PPG (star out) | 12 | 🔥 STAR OUT (major usage) |
| 3.0-4.9 PPG (key player out) | 8 | key teammate out |
| 1.5-2.9 PPG (rotation out) | 5 | teammate out |
| <1.5 PPG or no injury | 0 | - |

**Example:**
- **Austin Reaves** when LeBron & AD are OUT:
  - Base projection: 18 PPG
  - Injury boost: +3.5 PPG (teammate analysis shows he averages 21.5 when they sit)
  - Injury bonus: **+8 points** (key teammate out tier)
  - Recent L5: 32.4 PPG (hot streak already captures his recent explosions)
  - **Total GPP Score boost:** 8 points (can jump from "Playable" to "Strong")

**Why this is GOLD for tournaments:**

1. **Low Ownership Edge**: Public can't react to late scratches (6:30 PM injury reports)
   - Field sets lineups by 6 PM
   - Late injury announced at 6:35 PM
   - Only ~5-10% of field can swap in beneficiary
   - You get 35+ point upside at <10% ownership = PURE GOLD

2. **Proven Usage Spike**: System calculates boost from historical data
   - Austin Reaves: Averages +15% usage when LeBron out
   - Not guessing - using actual teammate correlation analytics
   - injury_impact_analytics module tracks every absence

3. **Differentiation**: Even if hot streak is already captured in L5 avg, the injury bonus explicitly flags this as a tournament priority
   - Helps you identify the REASON for the hot streak
   - If it's injury-driven, it's more sustainable (while teammate is out)
   - Different than a random hot streak that might regress

**How to Use:**
1. Check "Injury Admin" tab before tournaments (4 PM daily)
2. Mark late-scratch players as OUT
3. System auto-boosts teammate projections via `apply_injury_adjustments()`
4. Tournament tab shows boosted players with injury bonus in GPP Score
5. Target players with 🔥 STAR OUT flag for maximum edge

---

## GPP Score Grades

| Score Range | Grade | Action |
|------------|-------|--------|
| 85+ | 🔥🔥 GPP Lock | Must-play, build around |
| 75-84 | 🔥 Core Play | Strong tournament option |
| 65-74 | ⭐ Strong | Solid picks |
| 55-64 | ✓ Playable | Viable pivots |
| <55 | ⚠️ Punt/Fade | Avoid in GPP |

---

## How to Use GPP Score in Tournament Strategy

### Step 1: Filter by Ceiling Threshold

In the Tournament Strategy tab, set **Minimum Ceiling** slider to 30-35 PPG to see explosive candidates.

### Step 2: Sort by GPP Score

The table automatically sorts by GPP Score (highest first). This identifies the **best combination** of:
- High ceiling
- Hot streak
- High-variance matchup
- Proven history

### Step 3: Identify Core Plays (75+ GPP Score)

**Core Plays (🔥)** are your tournament anchors:
- Example: Shai Gilgeous-Alexander (75 GPP) = 52.4 ceiling + hot L5 + elite variance vs Dallas

### Step 4: Find Contrarian Pivots (65-74 GPP Score)

**Strong Plays (⭐)** that may be lower-owned:
- Example: Cade Cunningham (66 GPP) = 46.9 ceiling + warm L5 + elite variance vs Portland
- Lower projection (26.8) = likely lower ownership = differentiation edge

### Step 5: FADE Cold Players (<60 GPP Score)

Even with high ceilings, **cold players are tournament poison**:
- Example: Giannis (52 GPP) = 42.1 ceiling BUT L5 20.2 vs Season 25.1 (-20% cold streak)
- Example: Jokić (48 GPP) = 43.5 ceiling BUT L5 23.4 vs Season 26.4 (-11% cold)

**Why:** If they're cold, everyone else has the same high ceiling but better recent form.

---

## Real Example Breakdown (2025-12-05)

### 🔥 Core Play: Shai Gilgeous-Alexander (75 GPP)
```
Ceiling: 52.4 | L5: 35.6 | Season: 30.5 | vs Dallas
GPP Score Breakdown:
- Ceiling Base: 50 (maxed - 52.4 ceiling)
- Hot Streak: 10 (L5 +16% above season)
- Defense Variance: 15 (OKC factor 1.16 - elite)
- Defense Quality: 0 (average)
Total: 75 = Core Play
```

**Why it works:** Shai has a 52+ ceiling, is playing above his season average, and OKC allows occasional huge games. Perfect tournament play.

---

### ⭐ Strong Play: Jaylen Brown (71 GPP)
```
Ceiling: 46.6 | L5: 34.0 | Season: 27.8 | vs LAL
GPP Score Breakdown:
- Ceiling Base: 46 (45+ ceiling tier)
- Hot Streak: 15 (L5 +22% - very hot!)
- Defense Variance: 10 (LAL factor 1.12)
- Defense Quality: 0
Total: 71 = Strong
```

**Why it works:** Jaylen is ON FIRE (34.0 L5 avg!). Even though ceiling isn't maxed, his hot streak makes him a contrarian upside play.

---

### ⚠️ Punt/Fade: Giannis Antetokounmpo (52 GPP)
```
Ceiling: 42.1 | L5: 20.2 | Season: 25.1 | vs PHI
GPP Score Breakdown:
- Ceiling Base: 42 (40+ ceiling tier)
- Hot Streak: 0 (L5 -20% - COLD!)
- Defense Variance: 10 (PHI factor 1.12)
- Defense Quality: 0
Total: 52 = Punt/Fade
```

**Why to fade:** Giannis has a 42+ ceiling BUT he's averaging 20.2 PPG over L5 (20% below season). He's cold, likely to underperform.

---

## GPP Score vs DFS Score Comparison

| Metric | DFS Score (Cash) | GPP Score (Tournament) |
|--------|------------------|------------------------|
| **Ceiling Weight** | 0% (not used) | 50% (most important) |
| **Recent Form (L5)** | 0% (not used) | 20% (heavily weighted) |
| **Defense Variance** | 0% (not used) | 15% (identifies boom/bust) |
| **Projection Weight** | 40% (most important) | 0% (not used) |
| **Consistency** | Valued (low variance) | Ignored (high variance preferred) |

**Cash Game (DFS Score):** Optimize for accuracy and consistency (finish top 40%)

**Tournament (GPP Score):** Optimize for ceiling and variance (win or bust)

---

## Strategy Tips

### 1. **Target 2+ "Core Play" (75+) or "Strong" (65+) Players**

Your 3-player lineup should have:
- 1 Core Play (75+) = Anchor with elite ceiling + hot streak
- 2 Strong (65-74) = Contrarian pivots with upside

### 2. **Fade Players with 0 Hot Streak Bonus**

If L5 < Season Avg, they're cold. Avoid even if ceiling is high.

### 3. **Stack High-Variance Games**

If two players are from the same game with elite variance matchups (1.15+ factor), pick both for correlation.

Example:
- Shai (OKC) + Luka (DAL) = Same game, both have 1.16 ceiling factor opponents

### 4. **Differentiate from Field**

Public consensus focuses on projections (28+ PPG). GPP Score highlights contrarian picks:
- Lower projection BUT higher GPP Score = Low owned + high upside

---

## Troubleshooting

### "All my players are <60 GPP Score"

**Cause:** No one has hot streaks or elite variance matchups today.

**Solution:**
1. Lower minimum ceiling to 28 PPG (expand pool)
2. Focus on defense variance (filter for Opp Def Grade = A+, A, A-)
3. Consider waiting for better slate

---

### "Should I pick player with 85 GPP Score over 75?"

**YES!** Always pick higher GPP Score. The 85+ player has better combination of:
- Ceiling
- Hot streak
- Matchup variance

---

### "Player has 50+ ceiling but 0 hot streak bonus. Play them?"

**DEPENDS:**
- If L5 is neutral (within ±5% of season), yes (42-50 ceiling base alone = 42-50pts)
- If L5 is -10%+ below season, FADE (cold streak is predictive)

---

## Summary: When to Use GPP Score

✅ **Use GPP Score for:**
- Winner-take-all tournaments (3-player, 2,500+ opponents)
- Large-field GPPs (100+ entries)
- Any contest where 1st place pays most

❌ **Don't use GPP Score for:**
- Cash games (50/50, double-ups)
- Small-field contests (20 entries)
- Season-long leagues (consistency matters)

---

## Quick Reference

**GPP Score Formula:**
```
GPP Score = Ceiling Base (50%) + Hot Streak (20%) + Defense Variance (15%) + Matchup History (10%) + Defense Quality (5%) + Injury Beneficiary (12%)

Note: Total can exceed 100% weighting because injury bonus is situational (only applies when teammate is OUT)
```

**Target Lineup:**
- Player 1: 75+ GPP (Core)
- Player 2: 65-74 GPP (Strong pivot)
- Player 3: 65-74 GPP (Strong pivot from same game)
- Combined Ceiling: 105+ PPG
- At least 2 with hot streak bonus (10+ points)

**Avoid:**
- <60 GPP Score (low ceiling or cold streak)
- 0 hot streak bonus (L5 below season avg)
- Players with -5 defense adjustment (elite defense lockdown)

---

**Good luck in tournaments! 🏀🎯**

*Generated: 2025-12-05*
*Commit: 97efaac*
