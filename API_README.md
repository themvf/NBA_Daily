# NBA Daily - Model Accuracy API

Local REST API for programmatic access to the NBA prediction model's accuracy data, DFS slate tracking, and backtest results.

## Quick Start

```bash
# Install dependencies (if not already installed)
pip install fastapi uvicorn

# Start the server
python api_server.py

# Server runs at http://127.0.0.1:8000
# Interactive docs at http://127.0.0.1:8000/docs
```

### Options

```bash
python api_server.py --port 9000          # Custom port
python api_server.py --db other.db        # Custom database path
python api_server.py --host 0.0.0.0       # Expose to network
```

## How It Works

The API server sits on top of the same `nba_stats.db` SQLite database that powers the Streamlit dashboard. It opens the database in **read-only mode** so it can never modify your data.

```
                        nba_stats.db
                       /            \
           Streamlit App            API Server (port 8000)
           (dashboard UI)          (JSON endpoints)
                                        |
                                   Your scripts
```

For prediction accuracy and enhanced metrics, the API reuses the existing functions from `prediction_tracking.py` and `prediction_evaluation_metrics.py` directly - so the numbers you see in the API match the Streamlit dashboard exactly.

For DFS and backtest data, the API uses direct SQL queries against the database tables. This avoids importing `dfs_tracking.py` which would pull in the heavy `dfs_optimizer` dependency chain at startup.

## Python Client Examples

### Basic Request

```python
import requests

# Get overall accuracy metrics
r = requests.get("http://127.0.0.1:8000/accuracy/metrics")
data = r.json()
print(f"MAE: {data['mean_absolute_error']:.2f}")
print(f"Hit Rate: {data['hit_rate_floor_ceiling']:.1%}")
```

### Filtering by Date Range

```python
# January 2026 accuracy only
r = requests.get("http://127.0.0.1:8000/accuracy/metrics", params={
    "start_date": "2026-01-01",
    "end_date": "2026-01-31"
})
```

### Searching Predictions

```python
# Find all LeBron predictions
r = requests.get("http://127.0.0.1:8000/predictions", params={
    "player_name": "LeBron",
    "limit": 50
})
for pred in r.json()["data"]:
    print(f"{pred['game_date']}: proj {pred['projected_ppg']:.1f} -> actual {pred['actual_ppg']}")
```

### DFS Slate Analysis

```python
# Get projection accuracy for a specific DFS slate
r = requests.get("http://127.0.0.1:8000/dfs/projections/2026-02-06", params={"limit": 10})
for p in r.json()["data"]:
    diff = p["actual_fpts"] - p["proj_fpts"]
    print(f"{p['player_name']:20s}  proj={p['proj_fpts']:.1f}  actual={p['actual_fpts']:.1f}  ({diff:+.1f})")
```

### Pagination

```python
# Page through all predictions for a date range
offset = 0
all_predictions = []
while True:
    r = requests.get("http://127.0.0.1:8000/predictions", params={
        "start_date": "2026-01-01",
        "limit": 500,
        "offset": offset
    })
    batch = r.json()["data"]
    if not batch:
        break
    all_predictions.extend(batch)
    offset += len(batch)

print(f"Fetched {len(all_predictions)} total predictions")
```

---

## Endpoint Reference

All endpoints are `GET` requests. All date parameters use `YYYY-MM-DD` format.

### Health & Meta

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Database status, all table names, and row counts per table |
| `GET /dates/predictions` | All distinct game dates that have predictions |
| `GET /dates/dfs-slates` | All DFS slate dates with projection data |
| `GET /dates/backtests` | All dates with backtest results |

### Prediction Accuracy

| Endpoint | Parameters | Description |
|----------|------------|-------------|
| `GET /predictions` | `date`, `start_date`, `end_date`, `player_name`, `team`, `limit`, `offset` | Raw predictions with flexible filtering. `player_name` does partial matching. `team` matches full team name (e.g. "Cleveland Cavaliers") |
| `GET /predictions/{game_date}` | | All predictions vs actuals for a single date. Sorted by DFS score |
| `GET /accuracy/metrics` | `start_date`, `end_date`, `min_confidence` | Core accuracy metrics: MAE, RMSE, hit rate (floor-ceiling), confidence-split MAE, analytics-split MAE |
| `GET /accuracy/model-performance` | `start_date`, `end_date` | Overall model summary: total predictions, avg error, hit rate %, over/under projection split |
| `GET /accuracy/enhanced` | `start_date`, `end_date` | Advanced metrics: Spearman/Pearson correlation, calibration by confidence bucket, top-10% miss rate, floor/ceiling coverage |
| `GET /accuracy/by-tier` | `start_date`, `end_date` | Accuracy by player scoring tier (Role 0-10, Bench 10-15, Starter 15-20, Star 20-25, Superstar 25+) |
| `GET /accuracy/best-worst` | `n`, `start_date`, `end_date` | Top N best and worst predictions by absolute error. Default `n=10`, max 100 |
| `GET /accuracy/fanduel-comparison` | `start_date`, `end_date` | Model vs FanDuel O/U accuracy: who's closer more often, per-player breakdown |
| `GET /accuracy/daily-summary` | `start_date`, `end_date` | Per-date rollup: MAE, hit rate, over/under counts for each game date |

### DFS Tracking

| Endpoint | Parameters | Description |
|----------|------------|-------------|
| `GET /dfs/slate-results` | `start_date`, `end_date`, `limit`, `offset` | Aggregate DFS slate metrics: MAE, RMSE, correlation, lineup efficiency, value correlation |
| `GET /dfs/slate-results/{slate_date}` | | Single slate's full metrics (404 if not found) |
| `GET /dfs/projections/{slate_date}` | `limit`, `offset` | Per-player projection vs actual for a slate: fpts, points, rebounds, assists, etc. Only includes players who played |
| `GET /dfs/lineups/{slate_date}` | | All generated lineups and their actual performance (404 if not found) |
| `GET /dfs/pending-slates` | | Slates that have projections saved but no actual results yet |
| `GET /dfs/bias-by-team` | `days` | Projection bias by opponent team over a lookback window. Default `days=30`. Shows which opponents the model over/under-projects against |

### Backtests

| Endpoint | Parameters | Description |
|----------|------------|-------------|
| `GET /backtest/top3` | `strategy`, `start_date`, `end_date`, `limit`, `offset` | Daily top-3 scorer backtest results: who we picked vs who actually scored the most |
| `GET /backtest/top3/summary` | `strategy`, `start_date`, `end_date` | Aggregate hit rates per strategy: hit-any %, hit-2+ %, exact match %, avg overlap, avg closest miss |
| `GET /backtest/top3/strategies` | | List all backtest strategies with slate counts and date ranges |
| `GET /backtest/portfolio` | `start_date`, `end_date`, `limit`, `offset` | Portfolio/lineup backtest results per slate |
| `GET /backtest/portfolio/summary` | | Aggregate portfolio stats: win rate, avg shortfall, rank percentile, lineup efficiency |

---

## Response Format

All list endpoints return this structure:

```json
{
  "count": 88,
  "limit": 100,
  "offset": 0,
  "data": [ ... ]
}
```

Single-item endpoints return the object directly. Endpoints that group data (like `best-worst` or `by-tier`) have their own shapes documented in the Swagger UI.

### Pagination

List endpoints accept `limit` (default 100, max 1000) and `offset` (default 0):

```
GET /predictions?limit=50&offset=100    # rows 101-150
```

### Error Handling

- **404** - Resource not found (e.g. a slate_date with no data)
- **422** - Validation error (e.g. `limit=-1` or invalid parameter type)
- **500** - Server error (check server console for traceback)

---

## Key Metrics Explained

### Accuracy Metrics (`/accuracy/metrics`)

| Field | Meaning |
|-------|---------|
| `mean_absolute_error` | Average distance between projected and actual PPG (lower is better) |
| `rmse` | Root mean squared error - penalizes large misses more heavily |
| `mean_error` | Average signed error. Negative = model over-projects on average |
| `hit_rate_floor_ceiling` | % of predictions where actual fell within the projected floor-ceiling range |
| `high_confidence_mae` | MAE for predictions with confidence >= 70% |
| `low_confidence_mae` | MAE for predictions with confidence < 50% |
| `correlation_used_mae` | MAE when vs-opponent correlation data was available |
| `generic_only_mae` | MAE when only generic season averages were used |

### Enhanced Metrics (`/accuracy/enhanced`)

| Field | Meaning |
|-------|---------|
| `spearman_correlation` | Rank-ordering accuracy (do we correctly rank who scores more?). Closer to 1.0 = better |
| `pearson_correlation` | Linear correlation between projected and actual PPG |
| `calibration_score` | Hit rates grouped by confidence bucket - shows if higher confidence actually means better predictions |
| `top_10_pct_miss_rate` | Average error on the worst 10% of predictions (outlier severity) |
| `floor_coverage` | % of actuals that were above the projected floor |
| `ceiling_coverage` | % of actuals that were below the projected ceiling |

### DFS Slate Results (`/dfs/slate-results`)

| Field | Meaning |
|-------|---------|
| `proj_mae` | Mean absolute error on fantasy point projections |
| `proj_correlation` | Correlation between projected and actual fantasy points |
| `proj_within_range_pct` | % of players whose actual fell within projected floor-ceiling |
| `lineup_efficiency_pct` | Best generated lineup's actual fpts / optimal possible lineup fpts * 100 |
| `value_correlation` | How well fpts-per-dollar predictions tracked actual value |

### Backtest Summary (`/backtest/top3/summary`)

| Field | Meaning |
|-------|---------|
| `hit_any_pct` | % of slates where at least 1 of our top-3 picks was in the actual top 3 |
| `hit_2plus_pct` | % of slates where 2+ of our picks were in the actual top 3 |
| `hit_exact_pct` | % of slates where all 3 picks matched exactly |
| `avg_overlap` | Average number of our picks that were in the actual top 3 (0-3) |
| `avg_closest_miss` | When we miss, how close the nearest miss was (in points) |

---

## Architecture Notes

- **Read-only database access**: The SQLite connection uses `?mode=ro` URI parameter - the API literally cannot write to the database
- **Per-request connections**: Each API call gets its own database connection that closes when the request completes. SQLite handles concurrent reads safely
- **No authentication**: This runs on localhost only by default. If you expose it with `--host 0.0.0.0`, consider adding authentication
- **Swagger UI**: Visit `/docs` for an interactive explorer where you can test every endpoint directly in the browser
