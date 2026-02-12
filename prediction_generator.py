#!/usr/bin/env python3
"""
Standalone prediction generator for NBA_Daily.

Extracts the prediction loop from streamlit_app.py into a reusable function
that can be triggered on-demand via UI button or scheduled job.

This module imports and orchestrates existing calculation functions from
streamlit_app.py rather than duplicating complex logic.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import date
from pathlib import Path
import pandas as pd
import sqlite3

# Import prediction tracking
import prediction_tracking as pt
import injury_adjustment as ia


# ============================================================================
# Uncertainty Multiplier for Tournament Strategy
# ============================================================================

def calculate_uncertainty_multiplier(
    injury_status: str = None,
    minutes_deviation_ratio: float = 0.0,
    starter_changes_recent: int = 0,
    traded_recently: bool = False,
    teammate_questionable: bool = False
) -> float:
    """
    Calculate sigma multiplier based on role uncertainty.

    Higher multiplier = wider ceiling/floor spread = more variance.

    This is valuable for tournament strategy because:
    - Uncertain situations have fatter tails (more upside AND downside)
    - DFS contestants underweight uncertainty
    - News pivots can exploit these situations

    Args:
        injury_status: 'questionable', 'doubtful', 'probable', or None
        minutes_deviation_ratio: |recent_min - season_min| / season_min
                                 (NOT coefficient of variation - renamed for accuracy)
        starter_changes_recent: Number of times starter status changed in L5
        traded_recently: Player traded in last 2 weeks
        teammate_questionable: Star teammate is questionable (usage uncertainty)
                              NOTE: This is TEAMMATE, not opponent!

    Returns:
        Multiplier >= 1.0 to apply to sigma (ceiling - floor spread)
    """
    multiplier = 1.0

    # Injury status uncertainty
    if injury_status:
        status_lower = str(injury_status).lower()
        if status_lower == 'questionable':
            multiplier += 0.20  # +20% sigma
        elif status_lower == 'doubtful':
            multiplier += 0.30  # +30% sigma (might not play at all)
        elif status_lower == 'probable':
            multiplier += 0.05  # Minor uncertainty

    # Minutes volatility (deviation from season average)
    # > 0.15 indicates inconsistent playing time
    if minutes_deviation_ratio > 0.20:
        multiplier += 0.20  # High volatility
    elif minutes_deviation_ratio > 0.15:
        multiplier += 0.10  # Moderate volatility
    elif minutes_deviation_ratio > 0.10:
        multiplier += 0.05  # Slight volatility

    # Starter status changes
    if starter_changes_recent >= 2:
        multiplier += 0.15  # Role in flux
    elif starter_changes_recent == 1:
        multiplier += 0.08  # Recent change

    # Trade uncertainty
    if traded_recently:
        multiplier += 0.25  # New system, unknown role

    # Teammate uncertainty (usage could spike if star teammate out)
    if teammate_questionable:
        multiplier += 0.10  # Usage could spike or not

    # Cap at 2.0 (double sigma) to prevent extreme values
    return min(multiplier, 2.0)


def apply_uncertainty_to_projection(
    proj_floor: float,
    proj_ceiling: float,
    uncertainty_multiplier: float
) -> Tuple[float, float]:
    """
    Apply uncertainty multiplier to ceiling/floor spread.

    Expands the range symmetrically around the midpoint.

    Args:
        proj_floor: Original floor projection
        proj_ceiling: Original ceiling projection
        uncertainty_multiplier: Multiplier >= 1.0

    Returns:
        (new_floor, new_ceiling) tuple
    """
    if uncertainty_multiplier <= 1.0:
        return proj_floor, proj_ceiling

    midpoint = (proj_ceiling + proj_floor) / 2
    half_range = (proj_ceiling - proj_floor) / 2

    # Expand range by multiplier
    new_half_range = half_range * uncertainty_multiplier

    new_floor = max(0, midpoint - new_half_range)  # Floor at 0
    new_ceiling = midpoint + new_half_range

    return new_floor, new_ceiling


def generate_predictions_for_date(
    game_date: date,
    db_path: Path,
    season: str = "2025-26",
    season_type: str = "Regular Season",
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    Generate predictions for all games on a specific date.

    This function:
    1. Loads all required data (games, leaders, defense stats, etc.)
    2. Fetches current injury data
    3. Iterates through games → teams → top 13 players
    4. Calculates projections and saves to database
    5. Reports progress via callback

    Args:
        game_date: Date to generate predictions for
        db_path: Path to SQLite database
        season: NBA season (e.g., "2025-26")
        season_type: "Regular Season", "Playoffs", etc.
        progress_callback: Optional function(current: int, total: int, message: str)
                          Called after processing each game

    Returns:
        Dict with keys:
            - predictions_logged: int - Successfully saved predictions
            - predictions_failed: int - Failed predictions
            - errors: List[str] - Error messages (max 10 shown)
            - games_processed: int - Number of games processed
            - summary: Dict - Additional stats:
                - total_players: int
                - avg_confidence: float
                - avg_dfs_score: float

    Raises:
        ValueError: If no games found for date
        sqlite3.Error: If database connection fails
    """
    # Import streamlit_app functions - use sys.modules to avoid re-executing module
    # When Streamlit runs, the app is loaded as __main__, not streamlit_app
    # We need to check if __main__ actually IS streamlit_app before using it
    import sys

    # Strategy: Check if __main__ has a key function from streamlit_app
    # If yes, use __main__ (we're running inside the Streamlit app)
    # If no, import streamlit_app directly (we're running standalone)
    main_mod = sys.modules.get('__main__')
    if main_mod is not None and hasattr(main_mod, 'build_games_table'):
        # Running inside Streamlit - __main__ IS streamlit_app
        st_app = main_mod
    elif 'streamlit_app' in sys.modules:
        # streamlit_app was already imported, reuse it
        st_app = sys.modules['streamlit_app']
    else:
        # Fresh import for standalone testing
        import streamlit_app as st_app

    # Initialize result tracking
    result = {
        'predictions_logged': 0,
        'predictions_failed': 0,
        'errors': [],
        'games_processed': 0,
        'summary': {
            'total_players': 0,
            'avg_confidence': 0.0,
            'avg_dfs_score': 0.0,
        }
    }

    # Track for summary statistics
    confidence_sum = 0.0
    dfs_score_sum = 0.0
    player_count = 0

    try:
        # Step 1: Load data dependencies
        if progress_callback:
            progress_callback(0, 100, "Loading game data...")

        data = _load_data_dependencies(
            str(db_path),
            season,
            season_type,
            game_date,
            st_app
        )

        if data['games_df'].empty:
            raise ValueError(f"No games found for {game_date}")

        total_games = len(data['games_df'])
        result['games_processed'] = 0

        # Step 2: Iterate through games and generate predictions
        for game_idx, (_, matchup) in enumerate(data['games_df'].iterrows()):
            if progress_callback:
                away_team = matchup.get("Away", "")
                home_team = matchup.get("Home", "")
                progress_callback(
                    game_idx + 1,
                    total_games,
                    f"Processing {away_team} @ {home_team}"
                )

            try:
                logged, failed, player_stats = _generate_predictions_for_game(
                    matchup=matchup,
                    data=data,
                    season=season,
                    season_type=season_type,
                    game_date=game_date,
                    st_app=st_app
                )

                result['predictions_logged'] += logged
                result['predictions_failed'] += failed
                result['games_processed'] += 1

                # Accumulate stats
                for stats in player_stats:
                    confidence_sum += stats['confidence']
                    dfs_score_sum += stats['dfs_score']
                    player_count += 1

            except Exception as e:
                error_msg = f"Game failed ({matchup.get('Away', '?')} @ {matchup.get('Home', '?')}): {str(e)[:100]}"
                result['errors'].append(error_msg)
                result['predictions_failed'] += 1

                # Limit error list size
                if len(result['errors']) > 10:
                    result['errors'] = result['errors'][:9] + ["... (more errors suppressed)"]

        # Step 3: Calculate summary statistics
        if player_count > 0:
            result['summary']['total_players'] = player_count
            result['summary']['avg_confidence'] = confidence_sum / player_count
            result['summary']['avg_dfs_score'] = dfs_score_sum / player_count

        # Step 2.5: Fetch FanDuel odds AFTER predictions are created
        # Uses force=True because Step 2's INSERT OR REPLACE clobbers any
        # previously stored odds data, so we must always re-fetch regardless
        # of what should_fetch_odds() reports from earlier fetches today.
        if progress_callback:
            progress_callback(total_games, total_games, "Fetching FanDuel odds...")

        try:
            import odds_api
            odds_api.create_odds_tables(data['games_conn'])
            odds_api.upgrade_predictions_for_fanduel(data['games_conn'])

            # Always force=True: INSERT OR REPLACE in Step 2 wiped any prior odds
            should_fetch, reason = odds_api.should_fetch_odds(data['games_conn'], game_date)
            if should_fetch or "Already fetched" in reason:
                odds_result = odds_api.fetch_fanduel_lines_for_date(
                    data['games_conn'], game_date, extended_markets=True, force=True
                )
                result['summary']['fanduel_lines_matched'] = odds_result.get('players_matched', 0)
                result['summary']['fanduel_extended_stats'] = odds_result.get('extended_stats_found', 0)
                result['summary']['fanduel_api_requests'] = odds_result.get('api_requests_used', 0)
                if odds_result.get('error'):
                    result['errors'].append(f"FanDuel fetch: {odds_result['error']}")
            else:
                result['summary']['fanduel_lines_matched'] = 0
                result['summary']['fanduel_skip_reason'] = reason
        except ImportError:
            # odds_api module not available - skip silently
            pass
        except Exception as e:
            # Log but don't fail predictions for odds errors
            result['errors'].append(f"FanDuel fetch warning: {str(e)[:50]}")

        # Step 4: Run Monte Carlo simulation and persist results
        # This ensures p_top3, p_top1, top_scorer_score are populated for backtest
        if progress_callback:
            progress_callback(total_games, total_games, "Running top-3 simulation...")

        try:
            sim_result = _run_and_persist_simulation(
                conn=data['games_conn'],
                game_date=str(game_date),
                sim_n=10000,
                progress_callback=progress_callback
            )
            result['summary']['sim_status'] = sim_result.get('status', 'unknown')
            result['summary']['sim_players_updated'] = sim_result.get('updated', 0)
            if sim_result.get('error'):
                result['errors'].append(f"Simulation warning: {sim_result['error']}")
        except Exception as e:
            # Log but don't fail predictions for simulation errors
            result['errors'].append(f"Simulation failed: {str(e)[:50]}")
            result['summary']['sim_status'] = 'failed'

        return result

    except Exception as e:
        result['errors'].append(f"Fatal error: {str(e)}")
        return result

    finally:
        # FIX: Always close the database connection
        # This was leaky before - connections stayed open across multiple runs
        if 'data' in dir() and data.get('games_conn'):
            try:
                data['games_conn'].close()
            except Exception:
                pass  # Already closed or other issue - ignore


def _load_data_dependencies(
    db_path: str,
    season: str,
    season_type: str,
    game_date: date,
    st_app: Any
) -> Dict[str, Any]:
    """
    Load all data required for prediction generation.

    Args:
        db_path: Path to database
        season: NBA season
        season_type: Season type
        game_date: Game date
        st_app: streamlit_app module (passed to avoid circular import)

    Returns:
        Dict with keys:
            - games_df: DataFrame of games for the date
            - leaders_df: DataFrame of player scoring stats
            - scoring_map: Dict[team_id -> scoring stats]
            - defense_stats: DataFrame of team defense
            - defense_map: Dict[team_id -> defense stats]
            - def_style_map: Dict[team_id -> defense style]
            - player_style_splits: Dict[player_id -> style splits]
            - player_vs_team_history: Dict[player_id -> Dict[team_id -> stats]]
            - active_injuries: List of injury dicts
            - injured_player_ids: Set of injured player IDs
            - off_ppm_df, def_ppm_df, league_avg_ppm: PPM stats
            - games_conn: SQLite connection
            - score_column: Which column to sort leaders by
            - classify_difficulty: Function to classify opponent difficulty
            - low_thresh, mid_thresh: Difficulty thresholds
    """
    # load_ppm_stats is in streamlit_app, accessed via st_app

    # Get database connection
    games_conn = sqlite3.connect(db_path)

    # Create predictions table if needed
    pt.create_predictions_table(games_conn)

    # Load games
    games_df, scoring_map = st_app.build_games_table(
        db_path,
        game_date,
        season,
        season_type
    )

    # Load player leaders - first aggregate, then apply weights
    base_stats = st_app.aggregate_player_scoring(db_path, season, season_type)
    leaders_df = st_app.prepare_weighted_scores(
        base_stats,
        st_app.DEFAULT_MIN_GAMES,
        st_app.DEFAULT_WEIGHTS
    )

    # Determine score column
    score_column = (
        "composite_score" if "composite_score" in leaders_df.columns
        else "weighted_score"
    )

    # Load defense stats
    defense_stats = st_app.load_team_defense_stats(
        db_path,
        season,
        season_type
    )

    defense_map: Dict[int, Any] = {
        int(row["team_id"]): row.to_dict()
        for _, row in defense_stats.iterrows()
    }

    # Load PPM stats
    try:
        off_ppm_df, def_ppm_df, league_avg_ppm = st_app.load_ppm_stats(db_path, season)
        ppm_loaded = True
    except Exception:
        off_ppm_df, def_ppm_df, league_avg_ppm = None, None, 0.462
        ppm_loaded = False

    # Load defense style map
    def_style_map: Dict[int, str] = {
        int(row["team_id"]): (str(row.get("defense_style") or "Neutral"))
        for _, row in defense_stats.iterrows()
    }

    # Load player style splits
    player_style_splits = st_app.build_player_style_splits(
        db_path,
        season,
        season_type,
        def_style_map,
    )

    # Load player vs team history
    player_vs_team_history = st_app.build_player_vs_team_history(
        db_path,
        season,
        season_type,
    )

    # Create difficulty classification function
    def_score_series = defense_stats["def_composite_score"].dropna() if not defense_stats.empty else pd.Series(dtype=float)
    if not def_score_series.empty:
        low_thresh = def_score_series.quantile(0.33)
        mid_thresh = def_score_series.quantile(0.66)
    else:
        low_thresh = mid_thresh = None

    def classify_difficulty(score: float | None) -> str:
        if score is None or pd.isna(score):
            return "Unknown"
        if low_thresh is None or mid_thresh is None:
            return "Neutral"
        if score <= low_thresh:
            return "Hard"
        if score <= mid_thresh:
            return "Neutral"
        return "Favorable"

    # Fetch current injuries
    active_injuries = ia.get_active_injuries(games_conn, check_return_dates=True)

    # Build injury_status_map: player_id -> status
    # This is the canonical source since leaders_df may not have injury_status
    injury_status_map: Dict[int, str] = {
        inj['player_id']: inj.get('status', 'unknown').lower()
        for inj in active_injuries
    }

    # ONLY exclude players who are definitively OUT or DOUBTFUL
    # Keep Questionable/Probable players - they're valuable for tournaments!
    excluded_statuses = {'out', 'doubtful'}
    excluded_player_ids = {
        pid for pid, status in injury_status_map.items()
        if status in excluded_statuses
    }

    # Build teammate injury map: team_id -> list of questionable/out star player_ids
    # Used for teammate_questionable uncertainty
    teammate_injury_map: Dict[int, List[int]] = {}
    for inj in active_injuries:
        team_id = inj.get('team_id')
        if team_id and injury_status_map.get(inj['player_id']) in ('questionable', 'out', 'doubtful'):
            if team_id not in teammate_injury_map:
                teammate_injury_map[team_id] = []
            teammate_injury_map[team_id].append(inj['player_id'])

    return {
        'games_df': games_df,
        'leaders_df': leaders_df,
        'scoring_map': scoring_map,
        'defense_stats': defense_stats,
        'defense_map': defense_map,
        'def_style_map': def_style_map,
        'player_style_splits': player_style_splits,
        'player_vs_team_history': player_vs_team_history,
        'active_injuries': active_injuries,
        'injury_status_map': injury_status_map,  # NEW: canonical injury status source
        'excluded_player_ids': excluded_player_ids,  # RENAMED: only Out/Doubtful
        'teammate_injury_map': teammate_injury_map,  # NEW: for teammate uncertainty
        'off_ppm_df': off_ppm_df,
        'def_ppm_df': def_ppm_df,
        'league_avg_ppm': league_avg_ppm,
        'ppm_loaded': ppm_loaded,
        'games_conn': games_conn,
        'db_path': db_path,  # NEW: store path for per-player connections if needed
        'score_column': score_column,
        'classify_difficulty': classify_difficulty,
        'low_thresh': low_thresh,
        'mid_thresh': mid_thresh,
    }


def _generate_predictions_for_game(
    matchup: pd.Series,
    data: Dict[str, Any],
    season: str,
    season_type: str,
    game_date: date,
    st_app: Any
) -> Tuple[int, int, List[Dict[str, float]]]:
    """
    Generate predictions for a single game (both teams, top 13 players each).

    IMPORTANT: We increased from top 8 to top 13 because:
    - Matches DraftKings full rotation depth (10-13 players per team)
    - Vegas Gap Analysis needs all rotation players to compute accurate bench gaps
    - A 6th man or bench player can absolutely be a top-3 scorer on a slate
    - Winner-take-all tournaments need broader coverage

    Args:
        matchup: Game row from games_df
        data: All dependencies from _load_data_dependencies
        season: NBA season
        season_type: Season type
        game_date: Game date
        st_app: streamlit_app module

    Returns:
        (predictions_logged, predictions_failed, player_stats)
        player_stats: List of dicts with 'confidence' and 'dfs_score' for summary
    """
    import player_correlation_analytics as pca
    import defense_type_analytics as dta

    predictions_logged = 0
    predictions_failed = 0
    player_stats = []

    # Extract game info
    away_id = matchup.get("away_team_id")
    home_id = matchup.get("home_team_id")

    if pd.isna(away_id) or pd.isna(home_id):
        return (0, 0, [])

    away_id = int(away_id)
    home_id = int(home_id)

    # INCREASED: top 13 players per team (was 8, originally 5)
    # Matches DraftKings rotation depth for accurate Vegas gap analysis
    PLAYERS_PER_TEAM = 13

    # Pre-compute team categorizations ONCE per game (not per player!)
    # This avoids expensive repeated DB queries
    team_categories_cache = {}
    try:
        team_categories_cache = dta.categorize_teams_by_defense(
            data['games_conn'],
            season,
            season_type
        )
    except Exception:
        pass  # Continue without team categories

    # Process both teams
    for team_label, team_id, team_name in [
        ("Away", away_id, matchup["Away"]),
        ("Home", home_id, matchup["Home"]),
    ]:
        # Get all players for this team
        all_team_players = data['leaders_df'][
            data['leaders_df']["team_id"] == team_id
        ]

        # Filter out ONLY Out/Doubtful players
        # Keep Questionable/Probable - they're valuable for tournaments!
        if data['excluded_player_ids']:
            all_team_players = all_team_players[
                ~all_team_players['player_id'].isin(data['excluded_player_ids'])
            ]

        # Take top N from available players
        team_leaders = all_team_players.nlargest(PLAYERS_PER_TEAM, data['score_column'])

        if team_leaders.empty:
            continue

        # Get opponent info
        opponent_id = home_id if team_label == "Away" else away_id
        opponent_name = matchup["Home"] if team_label == "Away" else matchup["Away"]
        opponent_stats = data['defense_map'].get(int(opponent_id))

        # Get opponent defense metrics
        opp_avg_allowed = st_app.safe_float(
            opponent_stats.get("avg_allowed_pts")
        ) if opponent_stats else None

        opp_recent_allowed = st_app.safe_float(
            opponent_stats.get("avg_allowed_pts_last5")
        ) if opponent_stats else None

        opp_composite = st_app.safe_float(
            opponent_stats.get("def_composite_score")
        ) if opponent_stats else None

        opp_style = data['def_style_map'].get(int(opponent_id), "Neutral")
        opp_difficulty = data['classify_difficulty'](opp_composite)

        opp_def_rating = st_app.safe_float(
            opponent_stats.get("def_rating")
        ) if opponent_stats else None

        opp_pace = st_app.safe_float(
            opponent_stats.get("avg_opp_possessions")
        ) if opponent_stats else None

        # Process each player
        for _, player in team_leaders.iterrows():
            try:
                logged, failed, stats = _generate_prediction_for_player(
                    player=player,
                    team_id=team_id,
                    team_name=team_name,
                    opponent_id=opponent_id,
                    opponent_name=opponent_name,
                    opp_avg_allowed=opp_avg_allowed,
                    opp_recent_allowed=opp_recent_allowed,
                    opp_composite=opp_composite,
                    opp_style=opp_style,
                    opp_difficulty=opp_difficulty,
                    opp_def_rating=opp_def_rating,
                    opp_pace=opp_pace,
                    data=data,
                    season=season,
                    season_type=season_type,
                    game_date=game_date,
                    st_app=st_app,
                    pca=pca,
                    dta=dta,
                    team_categories_cache=team_categories_cache  # NEW: cached team categories
                )

                predictions_logged += logged
                predictions_failed += failed
                if stats:
                    player_stats.append(stats)

            except Exception as e:
                # Individual player failure - continue with next
                predictions_failed += 1
                # Don't append to errors here - let game-level handler deal with it

    return (predictions_logged, predictions_failed, player_stats)


def _run_and_persist_simulation(
    conn: sqlite3.Connection,
    game_date: str,
    sim_n: int = 10000,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation and persist results to predictions table.

    This step computes p_top3, p_top1, top_scorer_score and writes them back
    to the predictions table so they're available for backtest.

    Args:
        conn: Database connection
        game_date: Game date string (YYYY-MM-DD)
        sim_n: Number of simulation iterations
        progress_callback: Optional progress callback

    Returns:
        Dict with keys:
            - status: 'ok', 'skip', 'failed'
            - updated: Number of rows updated
            - error: Error message if any
    """
    try:
        import top3_ranking as t3r

        # Ensure simulation columns exist in predictions table
        pt.ensure_prediction_sim_columns(conn)

        # Create ranker and run simulation
        ranker = t3r.Top3Ranker(conn)

        # Run Monte Carlo simulation
        sim_df = ranker.simulate_top3_probability(
            game_date,
            n_simulations=sim_n,
            use_calibration=True,
            include_minutes=True
        )

        if sim_df.empty:
            return {'status': 'skip', 'updated': 0, 'error': None}

        # Also calculate TopScorerScore for the same players
        score_df = ranker.rank_by_top_scorer_score(game_date, include_components=False)

        # Merge top_scorer_score into sim_df
        if not score_df.empty and 'top_scorer_score' in score_df.columns:
            sim_df = sim_df.merge(
                score_df[['player_id', 'top_scorer_score']],
                on='player_id',
                how='left'
            )

        # Persist simulation results back to predictions table
        updated = pt.persist_sim_results(
            conn=conn,
            game_date=game_date,
            df=sim_df,
            sim_n=sim_n,
            sim_seed=None,
            sim_version='pipeline-v1.0'
        )

        # GUARDRAIL: Verify simulation coverage after persist
        verification = _verify_simulation_coverage(conn, game_date)
        nan_pct = verification.get('nan_pct', 0)
        warnings = []

        if nan_pct > 5.0:
            # Critical: fail the simulation
            return {
                'status': 'failed',
                'updated': updated,
                'error': f'CRITICAL: {nan_pct:.1f}% of predictions missing p_top3 (threshold: 5%)',
                'nan_pct': nan_pct,
                'verification': verification
            }
        elif nan_pct > 1.0:
            # Warning: continue but flag the issue
            warnings.append(f'{nan_pct:.1f}% of predictions missing p_top3')

        # Check probability sanity
        if verification.get('sum_p_top1') is not None:
            sum_p1 = verification['sum_p_top1']
            if abs(sum_p1 - 1.0) > 0.15:
                warnings.append(f'sum(p_top1)={sum_p1:.3f} (expected ~1.0)')

            max_p1 = verification.get('max_p_top1', 0)
            if max_p1 > 0.50:
                warnings.append(f'max(p_top1)={max_p1:.1%} (may indicate variance issue)')

        return {
            'status': 'ok',
            'updated': updated,
            'error': '; '.join(warnings) if warnings else None,
            'nan_pct': nan_pct,
            'verification': verification
        }

    except Exception as e:
        return {'status': 'failed', 'updated': 0, 'error': str(e)}


def _verify_simulation_coverage(conn: sqlite3.Connection, game_date: str) -> Dict[str, Any]:
    """
    Verify simulation coverage for a specific date.

    Returns dict with:
        - total: Total predictions for date
        - with_sim: Predictions with p_top3
        - nan_pct: Percentage missing p_top3
        - sum_p_top1: Sum of p_top1 (should be ~1.0)
        - max_p_top1: Max p_top1 (should be <50%)
    """
    cursor = conn.cursor()

    # Coverage check
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN p_top3 IS NOT NULL THEN 1 ELSE 0 END) as with_sim
        FROM predictions
        WHERE game_date = ?
    """, [game_date])
    total, with_sim = cursor.fetchone()

    nan_pct = ((total - with_sim) / total * 100) if total > 0 else 0

    # Probability sanity check
    cursor.execute("""
        SELECT
            SUM(p_top1) as sum_p1,
            MAX(p_top1) as max_p1
        FROM predictions
        WHERE game_date = ? AND p_top1 IS NOT NULL
    """, [game_date])
    sum_p1, max_p1 = cursor.fetchone()

    return {
        'total': total,
        'with_sim': with_sim,
        'nan_pct': nan_pct,
        'sum_p_top1': sum_p1,
        'max_p_top1': max_p1
    }


def _generate_prediction_for_player(
    player: pd.Series,
    team_id: int,
    team_name: str,
    opponent_id: int,
    opponent_name: str,
    opp_avg_allowed: Optional[float],
    opp_recent_allowed: Optional[float],
    opp_composite: Optional[float],
    opp_style: str,
    opp_difficulty: str,
    opp_def_rating: Optional[float],
    opp_pace: Optional[float],
    data: Dict[str, Any],
    season: str,
    season_type: str,
    game_date: date,
    st_app: Any,
    pca: Any,
    dta: Any,
    team_categories_cache: Dict[int, Any] = None
) -> Tuple[int, int, Optional[Dict[str, float]]]:
    """
    Generate prediction for a single player.

    Args:
        ... (existing params)
        team_categories_cache: Pre-computed team defense categories (avoids per-player DB hit)

    Returns:
        (logged_count, failed_count, stats_dict)
        stats_dict has keys: 'confidence', 'dfs_score' (or None if failed)
    """
    from datetime import date as date_type

    # Extract player stats
    player_id_val = st_app.safe_int(player.get("player_id"))
    avg_pts_last5 = st_app.safe_float(player.get("avg_pts_last5")) or st_app.safe_float(player.get("avg_points"))
    avg_pts_last3 = st_app.safe_float(player.get("avg_pts_last3")) or avg_pts_last5
    season_avg_pts = st_app.safe_float(player.get("avg_points")) or 0.0

    # Extract minutes and usage stats for momentum calculation
    season_avg_minutes = st_app.safe_float(player.get("avg_minutes"))
    avg_minutes_last5 = st_app.safe_float(player.get("avg_minutes_last5"))
    avg_usg_last5 = st_app.safe_float(player.get("avg_usg_last5"))
    usage_pct = st_app.safe_float(player.get("usg_pct"))

    # Get player history vs opponent
    vs_opp_history = None
    avg_vs_opp = None
    games_vs_opp = 0
    std_vs_opp = None

    if player_id_val is not None and opponent_id is not None:
        vs_opp_history = data['player_vs_team_history'].get(player_id_val, {}).get(int(opponent_id))
        if vs_opp_history:
            avg_vs_opp = vs_opp_history["avg_pts"]
            games_vs_opp = vs_opp_history["games"]
            std_vs_opp = vs_opp_history["std_pts"]

    # Get player history vs defense style
    avg_vs_style = None
    games_vs_style = 0
    if player_id_val is not None:
        style_data = data['player_style_splits'].get(player_id_val, {})
        avg_vs_style = style_data.get(opp_style)
        if avg_vs_style is not None:
            games_vs_style = 5  # Conservative estimate

    # Evaluate matchup quality
    matchup_rating, matchup_warning, matchup_confidence = st_app.evaluate_matchup_quality(
        season_avg_pts,
        avg_vs_opp,
        avg_vs_style,
        games_vs_opp,
        games_vs_style,
        std_vs_opp,
    )

    # Get opponent correlation
    # FIX: Use existing connection, not sqlite3.connect(str(conn)) which was broken!
    # str(conn) produces "<sqlite3.Connection object at 0x...>" which is NOT a path
    opponent_correlation_obj = None
    if player_id_val is not None:
        try:
            opponent_corrs = pca.calculate_opponent_correlations(
                data['games_conn'],  # FIX: Use existing connection directly
                player_id_val,
                season,
                season_type,
                min_games_vs=1
            )

            if opponent_corrs and opponent_id is not None:
                for opp_corr in opponent_corrs:
                    if opp_corr.opponent_team_id == int(opponent_id):
                        opponent_correlation_obj = opp_corr
                        break
        except Exception:
            pass

    # Get pace and defense quality splits
    # FIX: Use existing connection and cached team_categories
    pace_split_obj = None
    defense_quality_split_obj = None
    if player_id_val is not None and opponent_id is not None:
        try:
            # FIX: Use existing connection, not creating new broken connection
            defense_splits = dta.calculate_defense_type_splits(
                data['games_conn'],  # FIX: Use existing connection
                player_id_val,
                season,
                season_type,
                min_games=2
            )

            # FIX: Use cached team_categories passed from game level
            # This avoids expensive repeated DB queries per player
            team_categories = team_categories_cache or {}

            if int(opponent_id) in team_categories:
                opp_pace_cat = team_categories[int(opponent_id)]['pace_category']
                opp_def_cat = team_categories[int(opponent_id)]['defense_category']

                for split in defense_splits:
                    if split.defense_type == f"{opp_pace_cat} Pace":
                        pace_split_obj = split
                    elif split.defense_type == f"{opp_def_cat} Defense":
                        defense_quality_split_obj = split

            # FIX: Removed conn_for_splits.close() - we're using shared connection
        except Exception:
            pass

    # Get player position
    player_position = ""
    try:
        cursor = data['games_conn'].cursor()
        cursor.execute(
            "SELECT position FROM players WHERE player_id = ?",
            (player.get("player_id"),)
        )
        position_result = cursor.fetchone()
        if position_result and position_result[0]:
            player_position = position_result[0]
    except Exception:
        pass

    # Calculate projection
    projection, proj_confidence, proj_floor, proj_ceiling, breakdown, analytics_indicators = st_app.calculate_smart_ppg_projection(
        season_avg=season_avg_pts,
        recent_avg_5=avg_pts_last5,
        recent_avg_3=avg_pts_last3,
        vs_opp_team_avg=avg_vs_opp,
        vs_opp_team_games=games_vs_opp,
        vs_defense_style_avg=avg_vs_style,
        vs_defense_style_games=games_vs_style,
        opp_def_rating=opp_def_rating,
        opp_pace=opp_pace,
        opponent_correlation=opponent_correlation_obj,
        pace_split=pace_split_obj,
        defense_quality_split=defense_quality_split_obj,
        opp_team_id=opponent_id,
        player_name=player["player_name"],
        opponent_id=opponent_id,
        def_ppm_df=data['def_ppm_df'] if data['ppm_loaded'] else None,
        league_avg_ppm=data['league_avg_ppm'],
        conn=data['games_conn'],
        player_position=player_position,
        game_date=str(game_date),
        # Momentum calculation parameters
        season_avg_minutes=season_avg_minutes,
        avg_minutes_last5=avg_minutes_last5,
        avg_usg_last5=avg_usg_last5,
        usage_pct=usage_pct,
    )

    # =========================================================================
    # Apply Uncertainty Multiplier for Tournament Strategy
    # =========================================================================
    # Calculate minutes deviation ratio (NOT coefficient of variation)
    # This measures how much recent minutes deviate from season average
    minutes_deviation_ratio = 0.0
    if season_avg_minutes > 0 and avg_minutes_last5 is not None and avg_minutes_last5 > 0:
        minutes_deviation_ratio = abs(avg_minutes_last5 - season_avg_minutes) / season_avg_minutes

    # Get injury status from canonical source (injury_status_map)
    # leaders_df may not have injury_status populated, but injury_status_map
    # is built directly from active_injuries
    injury_status = data.get('injury_status_map', {}).get(player_id_val)

    # Check for questionable TEAMMATE (not opponent!)
    # This is usage uncertainty - if a star teammate is questionable,
    # this player's usage could spike (or not)
    # FIX: The old code used "opponent_injury_detected" which was wrong direction
    teammate_questionable = False
    teammate_injuries = data.get('teammate_injury_map', {}).get(team_id, [])
    if teammate_injuries:
        # Check if any injured teammate is a star (not this player)
        for injured_teammate_id in teammate_injuries:
            if injured_teammate_id != player_id_val:
                teammate_questionable = True
                break

    # Calculate uncertainty multiplier
    uncertainty_mult = calculate_uncertainty_multiplier(
        injury_status=injury_status,
        minutes_deviation_ratio=minutes_deviation_ratio,  # RENAMED from minutes_cv
        starter_changes_recent=0,  # Not tracked yet
        traded_recently=False,     # Not tracked yet
        teammate_questionable=teammate_questionable  # FIX: Now actually uses teammate info
    )

    # Apply uncertainty to ceiling/floor if multiplier > 1
    if uncertainty_mult > 1.0:
        proj_floor, proj_ceiling = apply_uncertainty_to_projection(
            proj_floor, proj_ceiling, uncertainty_mult
        )

    # Calculate DFS score
    daily_pick_score, pick_grade, pick_explanation = st_app.calculate_daily_pick_score(
        player_season_avg=season_avg_pts,
        player_projection=projection,
        projection_confidence=proj_confidence,
        matchup_rating=matchup_rating,
        opp_def_rating=opp_def_rating,
        def_ppm_df=data['def_ppm_df'] if data['ppm_loaded'] else None,
    )

    # Create prediction object
    prediction = pt.Prediction(
        prediction_id=None,
        prediction_date=str(date_type.today()),
        game_date=str(game_date),
        player_id=player_id_val,
        player_name=player["player_name"],
        team_id=team_id,
        team_name=team_name,
        opponent_id=opponent_id if opponent_id else 0,
        opponent_name=opponent_name,
        projected_ppg=projection,
        proj_confidence=proj_confidence,
        proj_floor=proj_floor,
        proj_ceiling=proj_ceiling,
        season_avg_ppg=season_avg_pts,
        recent_avg_3=avg_pts_last3,
        recent_avg_5=avg_pts_last5,
        vs_opponent_avg=avg_vs_opp,
        vs_opponent_games=games_vs_opp,
        analytics_used=analytics_indicators,
        opponent_def_rating=opp_def_rating,
        opponent_pace=opp_pace,
        dfs_score=daily_pick_score,
        dfs_grade=pick_grade,
        opponent_injury_detected=breakdown.get("opponent_injury_detected", False),
        opponent_injury_boost_projection=breakdown.get("opponent_injury_boost_projection", 0.0),
        opponent_injury_boost_ceiling=breakdown.get("opponent_injury_boost_ceiling", 0.0),
        opponent_injured_player_ids=None,
        opponent_injury_impact_score=0.0
    )

    # Log to database
    try:
        pt.log_prediction(data['games_conn'], prediction)
        return (1, 0, {'confidence': proj_confidence, 'dfs_score': daily_pick_score})
    except Exception as e:
        return (0, 1, None)


if __name__ == "__main__":
    # Simple test
    from datetime import date
    from pathlib import Path

    print("Testing prediction generator...")
    print("This requires a database and streamlit_app imports.")
    print("Run from Streamlit app, not standalone.")
