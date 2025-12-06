# Winner-Take-All Tournament Manual Workflow Guide

## Quick Reference

**Game Format:** Pick 3 players with highest points | 2,500 opponents | Winner-take-all

**Core Strategy:**
1. **Maximize CEILING**, not projection (need explosive games, not consistency)
2. **Differentiate from field** (1-2 contrarian picks minimum)
3. **Game stack** (2 players from same shootout game = correlation edge)
4. **Late swap leverage** (monitor injuries 30 min before games)

**Success Target:** Top 10% finish (250th place) = strong process validation

---

## Game Day Workflow

### STEP 1: Generate Base Data (Morning - 10:00 AM)

**Open Streamlit App → Today's Games Tab**
- Let system generate all predictions
- Predictions are logged to database automatically

**Open Database for Analysis:**
```bash
cd "C:\Docs\_AI Python Projects\NBA_Daily"
sqlite3 nba_stats.db
```

---

### STEP 2: Identify Ceiling Candidates (2:00 PM - 4 hours before games)

**Goal:** Find players with 35+ point ceiling potential

**SQL Query:**
```sql
SELECT
    player_name,
    team_name,
    projected_ppg,
    proj_ceiling,
    proj_floor,
    matchup_rating,
    dfs_score,
    (proj_ceiling - proj_floor) as upside_range,
    ROUND((proj_ceiling - proj_floor) / projected_ppg, 2) as variance_ratio
FROM predictions
WHERE game_date = '2025-12-05'
  AND proj_ceiling >= 35
ORDER BY proj_ceiling DESC
LIMIT 15;
```

**What to Look For:**
- `proj_ceiling >= 35`: Minimum threshold for explosive potential
- `upside_range`: Higher = more variance (good for tournaments)
- `variance_ratio > 0.35`: Players with boom/bust profiles
- `matchup_rating = 'Excellent'`: Favorable defensive matchup

**Save Top 10 Players** - These are your ceiling candidates

---

### STEP 3: Analyze Game Environments (2:15 PM)

**Goal:** Identify high-scoring games for correlation plays

**SQL Query - Find Shootout Games:**
```sql
SELECT
    opponent as game_matchup,
    COUNT(*) as player_count,
    ROUND(SUM(projected_ppg), 1) as projected_game_total,
    ROUND(AVG(proj_ceiling), 1) as avg_ceiling,
    MAX(matchup_rating) as best_matchup_rating
FROM predictions
WHERE game_date = '2025-12-05'
GROUP BY opponent
HAVING projected_game_total >= 230
ORDER BY projected_game_total DESC;
```

**Game Stack Criteria:**
- `projected_game_total >= 230`: High-scoring environment
- Both teams playing fast pace (check matchup_rating)
- Weak defensive matchups on both sides

**Identify Top 3 Shootout Games** - You'll pick 2 players from same game

---

### STEP 4: Cross-Reference Public Projections (2:30 PM)

**Check Public Consensus:**
1. Visit **FantasyLabs** (fantasylabs.com) or **RotoGrinders** (rotogrinders.com)
2. Look at their top projected players for today
3. Identify "chalk" plays (players likely 30%+ owned)

**Chalk Identification Rules:**
- Top 3 projected players = 40-50% ownership (VERY HIGH)
- Players 4-7 = 20-30% ownership (HIGH)
- Players 8-15 = 10-20% ownership (MEDIUM)
- Players 16+ = <10% ownership (LOW)

**Decision Framework:**
- **Chalk with 50+ ceiling:** Consider (1 max in lineup)
- **Chalk with <45 ceiling:** FADE (everyone has them, can't differentiate)
- **Low-owned with 35+ ceiling:** TARGET (contrarian edge)

---

### STEP 5: Build Initial Lineup Candidates (3:00 PM)

**Lineup Construction Rules:**
1. **Max 1 chalk play** (player in public top 5)
2. **At least 2 contrarian picks** (players NOT in public top 10)
3. **2 from same game** (correlation from shootout)
4. **Combined ceiling >= 105** (need 100+ to win typically)

**Example Lineup Combinations:**

**LINEUP A: Contrarian Game Stack**
- Player 1: Highest ceiling from Shootout Game #1 (not chalk)
- Player 2: 2nd highest ceiling from Shootout Game #1 (not chalk)
- Player 3: Highest ceiling from different game (contrarian)
- Strategy: Full differentiation, high correlation

**LINEUP B: Chalk + Pivot**
- Player 1: Top chalk play (if 50+ ceiling)
- Player 2: Low-owned player from same game as chalk
- Player 3: Contrarian from different shootout game
- Strategy: Don't get blown out by chalk, differentiate with pivots

**LINEUP C: Full Fade**
- Player 1: 8th-12th ranked ceiling player (low owned)
- Player 2: Same game as Player 1
- Player 3: Different game, injury beneficiary or pace-up spot
- Strategy: Maximum differentiation, high risk/reward

**Template for Recording Lineups:**
```
LINEUP A:
- [ ] Player 1: ___________ (Team: ___, Ceiling: ___, Ownership: ___)
- [ ] Player 2: ___________ (Team: ___, Ceiling: ___, Ownership: ___)
- [ ] Player 3: ___________ (Team: ___, Ceiling: ___, Ownership: ___)
- Combined Ceiling: ___
- Game Stack: Yes/No
- Chalk Count: ___

LINEUP B:
- [ ] Player 1: ___________ (Team: ___, Ceiling: ___, Ownership: ___)
- [ ] Player 2: ___________ (Team: ___, Ceiling: ___, Ownership: ___)
- [ ] Player 3: ___________ (Team: ___, Ceiling: ___, Ownership: ___)
- Combined Ceiling: ___
- Game Stack: Yes/No
- Chalk Count: ___

LINEUP C:
- [ ] Player 1: ___________ (Team: ___, Ceiling: ___, Ownership: ___)
- [ ] Player 2: ___________ (Team: ___, Ceiling: ___, Ownership: ___)
- [ ] Player 3: ___________ (Team: ___, Ceiling: ___, Ownership: ___)
- Combined Ceiling: ___
- Game Stack: Yes/No
- Chalk Count: ___
```

---

### STEP 6: Check for Injury Beneficiaries (4:00 PM)

**Goal:** Identify late-scratch value plays

**SQL Query - Current Injuries:**
```sql
SELECT
    player_name,
    team_name,
    injury_date,
    notes,
    expected_return_date
FROM injury_list
WHERE status = 'active'
  AND (expected_return_date IS NULL OR expected_return_date >= '2025-12-05')
ORDER BY team_name;
```

**For Each Injured Player:**

**SQL Query - Find Teammates Who Benefit:**
```sql
-- Replace 'PLAYER_ID_HERE' with injured player's ID
SELECT
    player_name,
    team_name,
    projected_ppg,
    proj_ceiling,
    injury_pts_delta,
    adjusted_projection
FROM predictions
WHERE game_date = '2025-12-05'
  AND team_name = 'TEAM_NAME_HERE'  -- Replace with injured player's team
  AND injury_pts_delta > 0
ORDER BY injury_pts_delta DESC
LIMIT 5;
```

**Injury Beneficiary Rules:**
- `injury_pts_delta >= 3.0`: Strong beneficiary (major usage increase)
- `injury_pts_delta >= 5.0`: Elite beneficiary (must consider)
- If injured player is chalk → Teammates likely LOW OWNED (huge edge)

**Add to Lineup Candidates:**
- If beneficiary has 35+ ceiling + low ownership → Swap into Lineup A or C

---

### STEP 7: Late Swap Monitor (6:30 PM - 30 min before games)

**Critical Window:** 6:30-7:00 PM (when late scratches announced)

**Monitoring Sources:**
1. **Twitter/X:** Follow team beat writers
   - Search: "[TEAM NAME] injury report"
   - Example: "Lakers injury report" shows LeBron/AD status
2. **NBA.com Injury Report:** Official status updates
3. **RotoWire:** Real-time injury news aggregator

**Late Scratch Decision Tree:**

**IF high-owned star ruled OUT (e.g., Giannis, LeBron):**
→ **ACTION:** Immediately check injury_impact_analytics
→ **Query teammates** with `injury_pts_delta` (same query as Step 6)
→ **Swap in teammate** if ceiling >= 35 (low owned + volume spike = gold)

**IF low-owned player ruled OUT:**
→ **ACTION:** Minimal impact, hold current lineup

**IF your selected player ruled OUT:**
→ **ACTION:** Replace with next highest ceiling player from same game (maintain correlation)

**Late Swap Execution:**
```sql
-- Quick query to find replacement from same game
SELECT
    player_name,
    team_name,
    opponent,
    projected_ppg,
    proj_ceiling
FROM predictions
WHERE game_date = '2025-12-05'
  AND opponent = 'OPPONENT_NAME_HERE'  -- Same game
  AND player_name != 'INJURED_PLAYER_NAME'
ORDER BY proj_ceiling DESC
LIMIT 5;
```

---

### STEP 8: Final Lineup Lock (6:50 PM - 10 min before games)

**Pre-Lock Checklist:**

✅ **Combined ceiling >= 105** (minimum to have win equity)

✅ **At least 2 players from same game** (correlation boost)

✅ **At least 1 contrarian pick** (player NOT in public top 10)

✅ **Max 1 chalk play** (don't over-expose to field consensus)

✅ **All players confirmed ACTIVE** (no late scratches)

**Final Selection Criteria:**
- Choose lineup with **highest combined ceiling**
- Tiebreaker: **More contrarian picks** (lower total ownership)
- Lock lineup in contest **before 7:00 PM**

---

## Advanced Tactics

### Contrarian Pick Identification

**SQL Query - Find Low-Owned High-Ceiling Players:**
```sql
SELECT
    player_name,
    team_name,
    opponent,
    projected_ppg,
    proj_ceiling,
    matchup_rating,
    dfs_score,
    -- Contrarian score heuristic
    CASE
        WHEN projected_ppg >= 28 THEN 'HIGH_OWNED'
        WHEN projected_ppg >= 23 THEN 'MEDIUM_OWNED'
        WHEN projected_ppg >= 18 THEN 'LOW_OWNED'
        ELSE 'VERY_LOW_OWNED'
    END as estimated_ownership,
    -- Value score
    ROUND(proj_ceiling / projected_ppg, 2) as ceiling_multiplier
FROM predictions
WHERE game_date = '2025-12-05'
  AND proj_ceiling >= 35
  AND projected_ppg < 26  -- Below typical chalk threshold
ORDER BY proj_ceiling DESC;
```

**Target Players:**
- `estimated_ownership = 'LOW_OWNED'` or `'VERY_LOW_OWNED'`
- `ceiling_multiplier >= 1.4` (big upside vs projection)
- `matchup_rating IN ('Excellent', 'Good')`

---

### Game Stack Deep Dive

**SQL Query - Analyze Specific Game:**
```sql
-- Replace 'OPPONENT_NAME' with target game
SELECT
    player_name,
    team_name,
    projected_ppg,
    proj_ceiling,
    proj_floor,
    matchup_rating,
    recent_avg_5,
    -- Ceiling variance
    (proj_ceiling - proj_floor) as upside_range
FROM predictions
WHERE game_date = '2025-12-05'
  AND opponent = 'OPPONENT_NAME_HERE'
ORDER BY proj_ceiling DESC
LIMIT 8;
```

**Stack Combinations:**
1. Pick top ceiling player from Team A
2. Pick top ceiling player from Team B (same game)
3. Verify combined ceiling >= 70 (strong stack)

**Why This Works:**
- If game goes over (e.g., 135-130), both players benefit from pace
- Correlation protects you (if one scores 35, environment likely helped both)
- Low ownership if game isn't the public's favorite shootout

---

### Variance Plays (High Risk/Reward)

**SQL Query - Find Boom/Bust Candidates:**
```sql
SELECT
    player_name,
    team_name,
    projected_ppg,
    proj_ceiling,
    proj_floor,
    -- Variance metrics
    ROUND((proj_ceiling - proj_floor), 1) as range_width,
    ROUND((proj_ceiling - proj_floor) / projected_ppg, 2) as variance_ratio
FROM predictions
WHERE game_date = '2025-12-05'
  AND variance_ratio >= 0.40  -- High variance threshold
  AND proj_ceiling >= 35
ORDER BY variance_ratio DESC
LIMIT 10;
```

**Target:**
- `variance_ratio >= 0.40`: Boom/bust profile
- `range_width >= 12`: Wide outcome distribution
- Good matchup rating (maximize boom probability)

**When to Use:**
- Full fade lineup (Lineup C)
- Late in tournament season (need a win, willing to take risk)
- When chalk is obvious and strong (need max differentiation)

---

## Post-Game Analysis

### Track Tournament Results

**After Contest Ends:**

1. **Record Winning Score:**
   - What was the 1st place total?
   - What was top 10% cutoff (250th place)?

2. **Analyze Winning Lineup:**
   - How many chalk plays did winner have? (0, 1, 2, 3?)
   - Which game produced most points?
   - Were there injury beneficiaries in winning lineup?
   - Combined ceiling of winning players?

3. **Log Your Results:**
```
Date: 2025-12-05
Lineup Selected: A/B/C
Players: ___________, ___________, ___________
Combined Ceiling: ___
Actual Total: ___
Finish Position: ___ / 2,501
Winning Total: ___
Gap to Winner: ___
Notes: (What worked? What didn't?)
```

**Update Strategy:**
- If you're consistently finishing top 20% → Strategy is working, stay disciplined
- If you're finishing bottom 50% → Revisit contrarian picks (too risky?) or chalk (too safe?)
- If chalk keeps winning → Consider 2 chalk + 1 pivot approach
- If contrarian wins often → Lean into full fade strategy

---

## Quick Decision Matrix

### Should I Play This Player?

| Player Type | Ceiling | Ownership | Decision |
|------------|---------|-----------|----------|
| Chalk Star | 50+ | 40%+ | ✅ YES (1 max) |
| Chalk Star | 35-49 | 40%+ | ❌ FADE |
| Contrarian | 40+ | <15% | ✅ YES (priority) |
| Contrarian | 35-39 | <15% | ✅ YES (if good matchup) |
| Mid-Tier | 35-39 | 20-30% | ⚠️ MAYBE (use as pivot) |
| Injury Beneficiary | 35+ | <10% | ✅ YES (elite edge) |
| Game Stack | 35+ (both) | <20% (both) | ✅ YES (correlation) |

---

## Common Mistakes to Avoid

❌ **Picking 3 independent chalk plays**
→ If everyone has them, you can't win even if all 3 hit

❌ **Ignoring late injury news**
→ Biggest edge is when field can't react (6:30+ PM scratches)

❌ **Playing 3 contrarian plays with 30 ceiling**
→ Need CEILING not just differentiation (can't win with 90 total)

❌ **Overthinking public projections**
→ Trust your model's ceiling estimates, not public consensus

❌ **Chasing recent performance**
→ 50-point game 2 nights ago = high ownership tonight (fade)

✅ **What to Do Instead:**
- 1 chalk + 2 contrarian (balanced approach)
- Monitor injuries until 6:50 PM (late swap edge)
- 2 from same shootout game (correlation)
- Trust ceiling over projection (upside > consistency)
- Differentiate from field (can't win with same lineup as 1,000 others)

---

## Bankroll Management

**Risk Guidelines:**
- **Max 5% of bankroll per contest** (e.g., $100 bankroll = $5 max entry)
- **Track weekly ROI:** (Winnings - Entry Fees) / Total Entry Fees
- **10-contest losing streak → Stop and reassess** (strategy flaw or variance?)

**Expected Outcomes:**
- **Normal:** 0-2 wins per 50 contests (winner-take-all variance)
- **Good process:** Top 10% finish rate 20-30% of time
- **Positive ROI:** 1 win per 20-30 contests (covers entry fees + profit)

**Variance Acceptance:**
- Winner-take-all = high variance by design
- Most contests you will lose ($0 payout)
- Focus on process (ceiling optimization, differentiation) not results
- One win can cover 20-30 losing entries

---

## Tools & Resources

### Required Tools (Free)
- **Your NBA_Daily Streamlit App:** Predictions and injury data
- **SQLite Browser:** Query database for analysis
- **FantasyLabs or RotoGrinders:** Public projection reference (free tier)
- **Twitter/X:** Injury news monitoring
- **Spreadsheet:** Track results and iterate strategy

### Optional Tools (Paid)
- **FantasyLabs Premium:** Ownership projections, player correlations
- **RotoGrinders Premium:** Injury alerts, lineup optimizer
- **NBA.com League Pass:** Watch games, understand player usage patterns

---

## Summary Checklist

**2:00 PM - Initial Analysis**
- [ ] Generate predictions in Streamlit app
- [ ] Run ceiling candidates query (35+ ceiling)
- [ ] Identify shootout games (230+ projected total)
- [ ] Check public projections (identify chalk)

**3:00 PM - Build Lineups**
- [ ] Create 3 lineup candidates (A, B, C)
- [ ] Verify game stacks (2 from same game)
- [ ] Ensure contrarian picks (not in public top 10)
- [ ] Check combined ceiling (>=105)

**4:00 PM - Injury Check**
- [ ] Query injury_list for active injuries
- [ ] Identify teammate beneficiaries
- [ ] Update lineups if strong beneficiary found

**6:30 PM - Late Swap Window**
- [ ] Monitor Twitter for late scratches
- [ ] Check injury_impact_analytics if star ruled out
- [ ] Swap if teammate has 35+ ceiling + low ownership

**6:50 PM - Final Lock**
- [ ] Run pre-lock checklist
- [ ] Select highest ceiling lineup
- [ ] Verify all players active
- [ ] Lock in contest

**Post-Game**
- [ ] Record winning total and lineup
- [ ] Log your results
- [ ] Update strategy notes

---

**Remember:** You need explosive games, not consistent games. Finishing 100th vs 500th has the same payout ($0). Optimize for CEILING and DIFFERENTIATION, not accuracy.

**Good luck! 🏀🎯**
