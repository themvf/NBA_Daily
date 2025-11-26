# Injury Impact Analysis - Feature Guide

## Overview

The Injury Impact Analysis feature helps you understand how player absences affect team performance, teammate production, and opponent performance. This analysis is performed without relying on external injury APIs - instead, it intelligently infers absences from missing game log data.

## How to Use

### 1. Access the Feature
- Open your Streamlit app (deployed on Streamlit Cloud or run locally)
- Navigate to the **"Injury Impact"** tab
- The tab appears between "Daily Leaders" and "Standings"

### 2. Configure Your Analysis
Use the filters to customize your analysis:

- **Season**: Select which NBA season to analyze (default: 2025-26)
- **Season Type**: Choose Regular Season, Playoffs, or Pre Season
- **Minimum Games Played**: Filter for players who've played at least N games (5-30)
- **Minimum PPG**: Set minimum points per game threshold (5.0-20.0)

### 3. Select a Player
Choose from the dropdown of significant players. Each option shows:
- Player name
- Team name
- Average points per game

Example: `"James Harden (Clippers) - 22.5 PPG"`

### 4. View Impact Analysis

The dashboard displays three main sections:

#### A. Team Performance Impact
Shows how the team performs with vs. without the selected player:

**Key Metrics:**
- Games played vs. games absent
- Win percentage comparison
- Offensive rating (PPG) comparison
- Defensive rating (points allowed) comparison

**Detailed Table:**
- Win-loss record
- Average points scored/allowed
- Point differential
- Delta columns show the exact impact

#### B. Teammate Usage & Scoring Redistribution
Reveals which teammates absorb the absent player's offensive load:

**Shows:**
- Points per game (with/without key player)
- Usage percentage changes
- Minutes changes
- Visual chart of top beneficiaries (biggest PPG increases)

**Interpretation:**
- Positive deltas = teammate increased production when star is out
- Negative deltas = teammate's production decreased (rare)

#### C. Opponent Performance Impact
Analyzes how opposing teams perform differently:

**Metrics:**
- Opponent points per game
- Opponent 3-pointers made
- Opponent efficiency

**Interpretation:**
- Positive deltas = opponents perform better without the player (shows defensive value)
- Negative deltas = opponents struggle more without the player (unusual)

## Real-World Example Use Cases

### Example 1: Evaluating a Player's True Value
**Question**: "How important is James Harden to the Clippers?"

**Steps**:
1. Select "James Harden (Clippers) - 22.5 PPG"
2. Check team win% delta: If win% drops 15% without him, he's critical
3. Check opponent PPG delta: If opponents score +8 PPG more, he's elite defensively
4. Review teammate redistribution: See who steps up (Kawhi, PG13, etc.)

### Example 2: Fantasy Basketball Decisions
**Question**: "If Luka is injured, which Mavericks player should I target?"

**Steps**:
1. Select "Luka Dončić (Mavericks)"
2. Navigate to "Teammate Usage & Scoring Redistribution"
3. Sort by "Pts Δ" (points delta)
4. Target players with highest positive deltas (biggest usage increases)

### Example 3: Betting/DFS Insights
**Question**: "How do opponents perform when Giannis is out?"

**Steps**:
1. Select "Giannis Antetokounmpo (Bucks)"
2. Check "Opponent Performance Impact"
3. If opponents average +12 PPG without Giannis, target opposing team players in DFS

## Technical Details

### How Player Absences Are Detected
The system compares:
- **Team game logs**: All games the team played
- **Player game logs**: All games the player participated in

**Logic**: If a team game exists but the player has no corresponding game log entry, the player was absent (injury, rest, suspension, etc.)

### Data Requirements
For meaningful analysis, you need:
- Minimum 10+ team games in the season
- At least 3-5 games where the player was absent
- Teammates who played both with and without the key player (min 3 games each scenario)

### Edge Cases Handled
- **Player with no absences**: Shows warning "No impact data available" (e.g., Luka playing every game)
- **Insufficient teammate data**: Shows message "Need at least 3 games in each scenario"
- **Early in season**: May show limited data if not enough games have been played yet

## Files Added/Modified

### New Files
- `injury_impact_analytics.py`: Core analytics module with 3 main functions
  - `calculate_team_impact()`: Team performance analysis
  - `calculate_teammate_redistribution()`: Teammate usage changes
  - `calculate_opponent_impact()`: Opponent performance changes
  - `get_significant_players()`: Helper to filter significant players

### Modified Files
- `streamlit_app.py`: Added "Injury Impact" tab with full UI
- `requirements.txt`: Added `plotly>=5.0,<6` for interactive charts

## Future Enhancements (Potential)

Ideas for expanding this feature:
1. **Historical absence timeline**: Chart showing when player was out across the season
2. **Multi-player comparison**: Compare impact of multiple stars side-by-side
3. **Injury severity classification**: Differentiate between rest, minor injuries, major injuries
4. **Playoff impact analysis**: Special focus on postseason absences
5. **Team logos on charts**: Visual enhancement (as noted in CLAUDE.md)
6. **Export reports**: Download PDF/CSV reports of injury impact analysis

## Troubleshooting

### "No significant players found"
**Solution**: Lower the minimum games or minimum PPG filters

### "No data found for [Player Name]"
**Cause**: Player has played in every single game (no absences)
**Solution**: Select a different player who has missed games

### "Not enough data to analyze teammate redistribution"
**Cause**: Either:
- Player hasn't missed enough games (need 3+)
- Teammates haven't played enough games in both scenarios
**Solution**: Try analyzing the 2024-25 season (more complete data)

### Charts not displaying
**Cause**: Plotly may not be installed
**Solution**: Run `pip install plotly>=5.0` then refresh the app

## Questions?

For issues or enhancement requests, visit the GitHub repository:
https://github.com/themvf/NBA_Daily

---

**Built with**:
- NBA Stats API (nba_api)
- Streamlit
- Plotly
- Pandas
- SQLite

**Generated with Claude Code**
