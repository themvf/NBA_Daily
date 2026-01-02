# FanDuel & The Odds API Integration Architecture

## Overview

This document describes the architecture for integrating FanDuel player points over/under lines via The Odds API, and tracking how our projections compare to FanDuel's lines after games complete.

---

## System Components

### 1. Data Source: The Odds API

**Provider:** [theoddsapi.com](https://the-odds-api.com)
**Budget:** 500 requests/month
**Data:** FanDuel player points over/under lines

**Key Endpoints:**
- `GET /v4/sports/basketball_nba/events` - List today's NBA games
- `GET /v4/sports/basketball_nba/events/{id}/odds?markets=player_points&bookmakers=fanduel` - Player props for a game

**API Key Location:**
```toml
# .streamlit/secrets.toml (local) or Streamlit Cloud Secrets
[theoddsapi]
API_KEY = "your-api-key-here"
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │  The Odds    │────▶│  odds_api.py │────▶│  predictions │        │
│  │  API         │     │              │     │  table       │        │
│  └──────────────┘     └──────────────┘     └──────────────┘        │
│                              │                    │                 │
│                              ▼                    │                 │
│                       ┌──────────────┐           │                 │
│                       │ odds_fetch_  │           │                 │
│                       │ log table    │           │                 │
│                       └──────────────┘           │                 │
│                                                  │                 │
│  ┌──────────────┐                               │                 │
│  │  NBA API     │────▶ actual_ppg ─────────────▶│                 │
│  │  (actuals)   │                               │                 │
│  └──────────────┘                               │                 │
│                                                  │                 │
│                       ┌──────────────────────────┘                 │
│                       │                                            │
│                       ▼                                            │
│                ┌──────────────┐     ┌──────────────┐              │
│                │ prediction_  │────▶│ Streamlit    │              │
│                │ tracking.py  │     │ Pages        │              │
│                └──────────────┘     └──────────────┘              │
│                                            │                       │
│                                            ├── FanDuel Compare     │
│                                            └── Model vs FanDuel    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module: odds_api.py

**Purpose:** Fetch and store FanDuel lines from The Odds API

### Key Functions

| Function | Description |
|----------|-------------|
| `get_api_key()` | Retrieve API key from Streamlit secrets |
| `get_nba_events()` | Fetch list of today's NBA games |
| `get_player_props_for_event()` | Fetch player points props for one game |
| `map_player_name_to_id()` | Fuzzy match FanDuel names to our player_id |
| `fetch_fanduel_lines_for_date()` | Main entry point - fetches all lines for a date |
| `should_fetch_odds()` | Check if we should fetch (budget, already fetched) |
| `get_monthly_api_usage()` | Track API budget usage |

### Player Name Matching

3-tier matching strategy:
1. **Alias lookup** - Check `odds_player_aliases` table for known mappings
2. **Exact match** - Case-insensitive match on `players.full_name`
3. **Fuzzy match** - RapidFuzz with 85% similarity threshold

### Timezone Handling

The Odds API returns UTC timestamps. We convert to US/Eastern before filtering:
```python
eastern = ZoneInfo("America/New_York")
utc_dt = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
eastern_dt = utc_dt.astimezone(eastern)
```

---

## Database Schema

### predictions table (FanDuel columns)

```sql
-- FanDuel line data (populated when fetching odds)
fanduel_ou REAL,              -- FanDuel O/U line (e.g., 23.5)
fanduel_over_odds INTEGER,    -- American odds for over (e.g., -110)
fanduel_under_odds INTEGER,   -- American odds for under (e.g., -110)
fanduel_fetched_at TEXT,      -- Timestamp when line was fetched
odds_event_id TEXT,           -- The Odds API event ID

-- Comparison metrics (calculated after game completes)
our_ou_call TEXT,             -- 'over' or 'under' (what we implied)
actual_ou_result TEXT,        -- 'over' or 'under' (what happened)
ou_call_correct INTEGER,      -- 1 if our call was right, 0 if not
fanduel_error REAL,           -- |actual - fanduel_ou|
we_were_closer INTEGER,       -- 1 if our error < FD error
closer_margin REAL            -- FD_error - our_error (positive = we won)
```

### odds_fetch_log table

Tracks API usage for budget management:
```sql
CREATE TABLE odds_fetch_log (
    log_id INTEGER PRIMARY KEY,
    fetch_date TEXT,          -- When we fetched
    game_date TEXT,           -- Games we fetched for
    events_fetched INTEGER,   -- Number of games
    players_matched INTEGER,  -- Players successfully matched
    api_requests_used INTEGER,-- API calls consumed
    remaining_requests INTEGER,
    error_message TEXT,
    created_at TIMESTAMP
);
```

### odds_player_aliases table

Stores learned name mappings for future lookups:
```sql
CREATE TABLE odds_player_aliases (
    alias_id INTEGER PRIMARY KEY,
    fanduel_name TEXT UNIQUE, -- Name from FanDuel
    player_id INTEGER,        -- Our player_id
    confidence REAL,          -- Match confidence (0-1)
    created_at TIMESTAMP
);
```

---

## API Budget Strategy

**Monthly limit:** 500 requests
**Daily estimate:** 6-16 requests (1 for events + 5-15 for games)

### Safeguards

| Threshold | Action |
|-----------|--------|
| 80% (400) | Warning displayed in UI |
| 95% (475) | Fetching blocked |
| Already fetched today | Skip (use cached data) |

---

## Streamlit Pages

### 1. FanDuel Compare (Pre-Game)

**Purpose:** Show current projections vs FanDuel lines before games

**Features:**
- Date selector
- Fetch button (when API key configured)
- Comparison table: Player, Team, Our Proj, FD O/U, Difference, % Diff
- Toggle: Show all players vs only those with lines
- API usage meter

### 2. Model vs FanDuel (Post-Game Analytics)

**Purpose:** Track accuracy after games complete

**Features:**
- Date range selector
- Summary metrics:
  - Our O/U Accuracy %
  - We Were Closer %
  - Our Avg Error (PPG)
  - FD Avg Error (PPG)
- Detailed comparison table
- Player insights (where we have edge vs where FD beats us)
- Export to CSV

---

## Data Flow Timeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    GAME DAY TIMELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Morning/Afternoon:                                             │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ 1. Generate predictions (prediction_generator.py)     │      │
│  │ 2. Fetch FanDuel lines (odds_api.py) - Step 4        │      │
│  │ 3. Store: projected_ppg + fanduel_ou in predictions  │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
│  Evening:                                                        │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ 4. Games are played                                   │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
│  Next Day:                                                       │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ 5. Fetch actuals from NBA API                        │      │
│  │ 6. Update predictions with actual_ppg                │      │
│  │ 7. Calculate comparison metrics:                     │      │
│  │    - our_ou_call, actual_ou_result, ou_call_correct  │      │
│  │    - fanduel_error, we_were_closer, closer_margin    │      │
│  │ 8. View analytics in "Model vs FanDuel" page         │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Files

| File | Purpose |
|------|---------|
| `odds_api.py` | The Odds API integration (~600 lines) |
| `prediction_tracking.py` | Schema + comparison calculations |
| `streamlit_app.py` | UI pages (FanDuel Compare, Model vs FanDuel) |
| `.streamlit/secrets.toml` | API key storage (local) |

---

## Future Enhancements

- [ ] Charts: O/U accuracy over time (7-day rolling)
- [ ] Charts: Who was closer by player position
- [ ] Alerts: Notify when we have strong disagreement with FD (>3 pts)
- [ ] ROI calculator: If you bet our picks vs FD lines
- [ ] Multi-bookmaker support (DraftKings, BetMGM)

---

## Troubleshooting

### No FanDuel lines showing

1. Check API key is configured in secrets
2. Check API budget isn't exhausted
3. Verify timezone handling (late games may appear next UTC day)

### Players not matching

1. Check `odds_player_aliases` table for failed mappings
2. Try lowering `FUZZY_MATCH_THRESHOLD` (currently 85)
3. Manually add alias via database

### Comparison metrics not calculating

1. Ensure `actual_ppg` is populated (games must complete)
2. Ensure `fanduel_ou` was fetched before games
3. Click "Recalculate Metrics" button on Model vs FanDuel page
