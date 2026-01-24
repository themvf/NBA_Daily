# Enrichment Validation & Iteration Plan

## Overview

This document outlines a systematic approach to validate, monitor, and tune the 4 data enrichments implemented in commit `4744c6d`:

| Enrichment | Module | Key Multiplier |
|------------|--------|----------------|
| Rest/B2B | `rest_days.py` | 0.92 (B2B) to 1.05 (rested) |
| Game Script | `game_script.py` | +/- 5 minutes PPG adjustment |
| Depth Roles | `depth_chart.py` | STAR/STARTER/ROTATION/BENCH |
| Position Defense | `position_ppm_stats.py` | 0.93 to 1.10 factor |

---

## 1. Hypothesis Ledger

### 1.1 Rest/B2B Hypotheses

| ID | Hypothesis | Metric | Success Threshold | Failure Threshold |
|----|------------|--------|-------------------|-------------------|
| R1 | B2B decreases fantasy points by ~8% on average | Mean PPG diff (B2B vs non-B2B) | -6% to -10% | Outside -3% to -15% |
| R2 | B2B effect is stronger for high-minute players | Interaction: B2B * avg_minutes | Negative coefficient, p<0.05 | No significant interaction |
| R3 | 3+ days rest provides 3-5% boost | Mean PPG diff (rested vs normal) | +2% to +6% | <0% or >10% |
| R4 | B2B multiplier reduces MAE for B2B games | MAE(with multiplier) < MAE(without) | >5% MAE reduction | MAE increases |

### 1.2 Game Script Hypotheses

| ID | Hypothesis | Metric | Success Threshold | Failure Threshold |
|----|------------|--------|-------------------|-------------------|
| G1 | Large spreads (>10) reduce star minutes by 3-5 | Mean minutes diff (blowout vs normal) | -2 to -6 minutes | Positive or >-8 |
| G2 | Large spreads increase bench minutes | Mean minutes diff (bench in blowout) | +2 to +6 minutes | Negative |
| G3 | Close games (<3 spread) increase star minutes | Mean minutes diff (close vs normal) | +1 to +3 minutes | Negative |
| G4 | Minutes adjustment improves ceiling accuracy | Ceiling hit rate improvement | >3% improvement | Decreases |
| G5 | Game script tier predicts actual game flow | Correlation: spread vs final margin | r > 0.50 | r < 0.30 |

### 1.3 Depth Role Hypotheses

| ID | Hypothesis | Metric | Success Threshold | Failure Threshold |
|----|------------|--------|-------------------|-------------------|
| D1 | Role tiers correlate with actual minutes | Correlation: tier rank vs avg_minutes | r > 0.70 | r < 0.50 |
| D2 | STARs have lower prediction error (more stable) | MAE by tier | STAR MAE < BENCH MAE | STAR MAE > BENCH MAE |
| D3 | Injury ripple is better explained by same-tier | Boost accuracy when injured=STAR | >10% improvement | <0% improvement |
| D4 | Role tier misclassification rate < 15% | % players with wrong tier | <15% | >25% |

### 1.4 Position Defense Hypotheses

| ID | Hypothesis | Metric | Success Threshold | Failure Threshold |
|----|------------|--------|-------------------|-------------------|
| P1 | Position-specific defense differs from overall | Variance: pos_ppm vs team_ppm | Significant at p<0.05 | Not significant |
| P2 | Position matchup factor improves out-of-sample | MAE reduction with factor | >2% reduction | MAE increases |
| P3 | Grade 'F' opponents yield higher PPG | Mean PPG: F-grade vs A-grade | >15% higher | <5% higher |
| P4 | Position defense is stable across season | Week-to-week correlation | r > 0.70 | r < 0.40 |

---

## 2. Offline Evaluation Design

### 2.1 Ablation Study Matrix

Run backtests comparing these model variants:

| Variant | Rest | Game Script | Roles | Pos Defense | Expected Impact |
|---------|------|-------------|-------|-------------|-----------------|
| Baseline | OFF | OFF | OFF | OFF | Reference |
| +Rest | ON | OFF | OFF | OFF | -3% MAE |
| +GameScript | OFF | ON | OFF | OFF | -2% MAE |
| +Roles | OFF | OFF | ON | OFF | Segmentation only |
| +PosDefense | OFF | OFF | OFF | ON | -1.5% MAE |
| Full Model | ON | ON | ON | ON | -5 to -8% MAE |

### 2.2 Required Metrics

For each variant, report:

**Overall Accuracy:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Bias (Mean Error)
- R-squared

**Calibration by Projection Bucket:**
| Bucket | Predictions | Actual Mean | Predicted Mean | Bias |
|--------|-------------|-------------|----------------|------|
| 5-10 PPG | n | x.x | x.x | +/- |
| 10-15 PPG | n | x.x | x.x | +/- |
| 15-20 PPG | n | x.x | x.x | +/- |
| 20-25 PPG | n | x.x | x.x | +/- |
| 25-30 PPG | n | x.x | x.x | +/- |
| 30+ PPG | n | x.x | x.x | +/- |

**Tail Accuracy:**
- Top decile capture rate (actual top 10% predicted in our top 20%)
- Ceiling hit rate (actual within floor-ceiling range)
- Upside capture (how often our top picks finish top 5)

**Segment Breakdowns:**
- STAR vs STARTER vs ROTATION vs BENCH
- B2B vs non-B2B
- Blowout-risk vs Tossup vs Close-game
- Guard vs Forward vs Center position matchups

### 2.3 Backtest Configuration

```python
BACKTEST_CONFIG = {
    'date_range': {
        'train': '2025-10-21 to 2025-12-31',  # ~70 days
        'test': '2026-01-01 to present',       # Holdout
    },
    'min_games_per_player': 5,
    'min_predictions_per_segment': 30,
    'confidence_level': 0.95,
    'bootstrap_iterations': 1000,
}
```

---

## 3. Production Monitoring

### 3.1 Daily Logging Schema

Create table `enrichment_audit_log`:

```sql
CREATE TABLE IF NOT EXISTS enrichment_audit_log (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_date TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    player_name TEXT,

    -- Enrichment factors applied
    days_rest INTEGER,
    rest_multiplier REAL,
    is_b2b INTEGER,
    game_script_tier TEXT,
    blowout_risk REAL,
    minutes_adjustment REAL,
    role_tier TEXT,
    position_matchup_factor REAL,

    -- Prediction vs Actual
    projected_ppg REAL,
    actual_ppg REAL,
    prediction_error REAL,
    abs_error REAL,

    -- Projection components
    base_projection REAL,  -- Before enrichments
    enriched_projection REAL,  -- After enrichments
    enrichment_delta REAL,  -- Difference

    -- Outcome flags
    ceiling_hit INTEGER,  -- 1 if actual in [floor, ceiling]
    was_top10 INTEGER,    -- 1 if actual in top 10 scorers

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(game_date, player_id)
);
```

### 3.2 Weekly Summary Schema

Create table `enrichment_weekly_summary`:

```sql
CREATE TABLE IF NOT EXISTS enrichment_weekly_summary (
    week_ending TEXT PRIMARY KEY,

    -- Sample sizes
    total_predictions INTEGER,
    b2b_predictions INTEGER,
    blowout_predictions INTEGER,
    close_game_predictions INTEGER,

    -- Rest/B2B metrics
    b2b_mean_error REAL,
    non_b2b_mean_error REAL,
    b2b_effect_observed REAL,  -- Actual B2B penalty %
    rest_multiplier_mae REAL,

    -- Game Script metrics
    blowout_minutes_error REAL,
    close_game_minutes_error REAL,
    game_script_mae_impact REAL,

    -- Role metrics
    star_mae REAL,
    starter_mae REAL,
    rotation_mae REAL,
    bench_mae REAL,

    -- Position Defense metrics
    pos_factor_correlation REAL,
    pos_factor_mae_impact REAL,

    -- Overall
    overall_mae REAL,
    overall_bias REAL,
    ceiling_hit_rate REAL,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### 3.3 Alert Thresholds

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| B2B Effect Flip | B2B players outperform non-B2B for 7+ days | HIGH | Disable B2B multiplier |
| Blowout Overcorrection | Blowout star MAE > 20% worse | MEDIUM | Cap minutes adjustment |
| Position Factor Overfitting | Position factor MAE worse than baseline | MEDIUM | Reset to 1.0 |
| Role Tier Drift | >20% of players change tiers week-over-week | LOW | Investigate roster changes |
| Rest Boost Inversion | Rested players underperform | MEDIUM | Review rest thresholds |

### 3.4 Dashboard Metrics

**Daily View:**
- Predictions with enrichment factors applied
- Error distribution by enrichment type
- Today's B2B players and their outcomes

**Weekly View:**
- Rolling 7-day MAE by enrichment variant
- Enrichment effect estimates with confidence intervals
- Segment breakdown heatmap

**Monthly View:**
- Full ablation study results
- Parameter drift visualization
- Hypothesis validation status

---

## 4. Experimentation Loop

### 4.1 Parameterization

All multipliers are configurable in `enrichment_config.py`:

```python
ENRICHMENT_CONFIG = {
    'rest': {
        'b2b_multiplier': 0.92,       # Current: -8%
        'b2b_min': 0.85,              # Guardrail: max -15%
        'b2b_max': 0.98,              # Guardrail: min -2%
        'rested_multiplier': 1.05,    # Current: +5%
        'rested_min': 1.00,           # Guardrail: no penalty
        'rested_max': 1.10,           # Guardrail: max +10%
        'rested_threshold_days': 3,   # Days for "rested" status
    },
    'game_script': {
        'blowout_spread_threshold': 10.0,
        'close_spread_threshold': 3.0,
        'star_blowout_minutes_adj': -4.0,
        'bench_blowout_minutes_adj': +5.0,
        'star_close_minutes_adj': +2.0,
        'minutes_adj_cap': 6.0,       # Guardrail: max adjustment
        'ppm_conversion': 0.55,
    },
    'roles': {
        'star_min_minutes': 30.0,
        'star_min_ppg': 18.0,
        'starter_min_minutes': 25.0,
        'starter_min_ppg': 10.0,
        'rotation_min_minutes': 15.0,
    },
    'position_defense': {
        'factor_min': 0.90,           # Guardrail: max -10%
        'factor_max': 1.15,           # Guardrail: max +15%
        'grade_factors': {
            'A': 0.93, 'B': 0.97, 'C': 1.00, 'D': 1.05, 'F': 1.10
        },
    },
}
```

### 4.2 Time-Split Validation

```
Training Window: Games from T-60 to T-14
Validation Window: Games from T-14 to T-7
Test Window: Games from T-7 to T (current)

Weekly Update Cycle:
1. Slide windows forward 7 days
2. Retrain on new training window
3. Validate on validation window
4. If validation improves, update parameters
5. Monitor test window for drift
```

### 4.3 Guardrails

**Hard Constraints:**
- No multiplier can exceed 1.20 or go below 0.80
- Minutes adjustments capped at +/- 6 minutes
- Position factors must sum to ~3.0 across positions (no systemic bias)
- Role tiers must have monotonic minutes relationship

**Soft Constraints:**
- Parameter changes limited to +/- 5% per week
- Require 50+ observations per segment for updates
- New parameters must beat current on validation set
- Fallback to neutral (1.0) if confidence interval includes 1.0

### 4.4 Update Cadence

| Frequency | Activity | Scope |
|-----------|----------|-------|
| Daily | Log enrichment factors + outcomes | All predictions |
| Weekly | Calculate rolling metrics, check alerts | Summary stats |
| Weekly | Evaluate parameter adjustments | If thresholds met |
| Monthly | Full rolling backtest | Complete ablation |
| Monthly | Reassess position defense persistence | Team defense table |
| Quarterly | Review hypothesis ledger | Update success/failure |

---

## 5. Implementation Tasks

### Phase 1: Measurement & Logging (Week 1)
**Effort: Medium | Priority: P0**

- [ ] Create `enrichment_audit_log` table
- [ ] Create `enrichment_weekly_summary` table
- [ ] Add `log_enrichment_audit()` function to `prediction_enrichments.py`
- [ ] Add nightly job to populate audit log from predictions + actuals
- [ ] Create `enrichment_config.py` with parameterized multipliers
- [ ] Update `prediction_enrichments.py` to use config file
- [ ] Add unit tests for enrichment bounds

**Acceptance Criteria:**
- Audit log populated for all predictions
- Config file controls all multipliers
- Tests verify guardrails enforced

### Phase 2: Backtesting & Ablations (Week 2-3)
**Effort: Large | Priority: P1**

- [ ] Create `evaluation/` module structure
- [ ] Implement `AblationBacktester` class
- [ ] Add `run_ablation_study()` function
- [ ] Implement segment breakdown reporting
- [ ] Create calibration bucket analysis
- [ ] Add tail accuracy metrics (top decile, ceiling hits)
- [ ] Generate baseline backtest (no enrichments)
- [ ] Generate full ablation matrix (6 variants)
- [ ] Create markdown report generator

**Acceptance Criteria:**
- Ablation study runs in <10 minutes
- All 6 variants compared with confidence intervals
- Segment breakdowns for all 4 enrichment types

### Phase 3: Streamlit Dashboard (Week 3-4)
**Effort: Medium | Priority: P1**

- [ ] Add "Enrichment Validation" page to Streamlit
- [ ] Create daily enrichment factor display
- [ ] Create weekly rolling metrics charts
- [ ] Add hypothesis validation status indicators
- [ ] Create ablation study results visualization
- [ ] Add alert status panel
- [ ] Create parameter tuning interface (optional)

**Acceptance Criteria:**
- Dashboard shows real-time enrichment health
- Alerts visible when thresholds breached
- Historical trends viewable

### Phase 4: Tuning & Guardrails (Week 4-5)
**Effort: Medium | Priority: P2**

- [ ] Implement `ParameterTuner` class
- [ ] Add time-split validation logic
- [ ] Implement guardrail enforcement
- [ ] Add weekly parameter update job
- [ ] Create parameter change audit log
- [ ] Add rollback mechanism
- [ ] Implement A/B test framework for live testing

**Acceptance Criteria:**
- Parameters can auto-tune within guardrails
- All changes logged and reversible
- No parameter exceeds hard constraints

### Phase 5: Ongoing Monitoring (Continuous)
**Effort: Small (ongoing) | Priority: P1**

- [ ] Set up daily cron for audit log population
- [ ] Set up weekly cron for summary calculation
- [ ] Configure alert notifications (email/slack)
- [ ] Create monthly report automation
- [ ] Add CI smoke tests for enrichments
- [ ] Document hypothesis validation results

**Acceptance Criteria:**
- Automated daily/weekly jobs running
- Alerts trigger within 1 hour of threshold breach
- Monthly reports generated automatically

---

## 6. Smoke Tests (CI/CD)

Add to test suite:

```python
def test_enrichment_columns_exist():
    """Verify all enrichment columns are in predictions table."""
    required = ['days_rest', 'rest_multiplier', 'is_b2b',
                'game_script_tier', 'blowout_risk', 'minutes_adjustment',
                'role_tier', 'position_matchup_factor']
    # Assert all columns exist

def test_rest_multiplier_bounds():
    """Verify rest multipliers stay within guardrails."""
    assert 0.85 <= B2B_MULTIPLIER <= 0.98
    assert 1.00 <= RESTED_MULTIPLIER <= 1.10

def test_minutes_adjustment_cap():
    """Verify minutes adjustments don't exceed cap."""
    for spread in [-15, -10, -5, 0, 5, 10, 15]:
        for role in ['STAR', 'STARTER', 'ROTATION', 'BENCH']:
            adj = get_minutes_adjustment(spread, role, True)
            assert abs(adj['minutes_adj']) <= 6.0

def test_position_factor_bounds():
    """Verify position factors stay within guardrails."""
    assert 0.90 <= min(GRADE_FACTORS.values())
    assert max(GRADE_FACTORS.values()) <= 1.15

def test_role_tier_monotonic():
    """Verify role tiers have monotonic minutes relationship."""
    # STAR > STARTER > ROTATION > BENCH in avg_minutes

def test_daily_update_enrichments():
    """Verify daily_update refreshes enrichment tables."""
    # Check player_roles and team_position_defense updated
```

---

## 7. Success Metrics Summary

| Phase | Key Metric | Target | Current |
|-------|------------|--------|---------|
| Phase 1 | Audit log coverage | 100% predictions logged | TBD |
| Phase 2 | Full model MAE reduction | >5% vs baseline | TBD |
| Phase 3 | Dashboard uptime | 99% | TBD |
| Phase 4 | Parameter stability | <5% weekly drift | TBD |
| Phase 5 | Alert response time | <1 hour | TBD |

---

## 8. Timeline

```
Week 1: Phase 1 - Measurement & Logging
Week 2-3: Phase 2 - Backtesting & Ablations
Week 3-4: Phase 3 - Streamlit Dashboard
Week 4-5: Phase 4 - Tuning & Guardrails
Week 5+: Phase 5 - Ongoing Monitoring (continuous)
```

---

## Appendix: File Structure

```
NBA_Daily/
├── evaluation/
│   ├── __init__.py
│   ├── ablation_backtest.py      # AblationBacktester class
│   ├── segment_analysis.py       # Segment breakdown reporting
│   ├── calibration.py            # Bucket calibration analysis
│   ├── hypothesis_tests.py       # Statistical tests for hypotheses
│   └── report_generator.py       # Markdown report generator
├── enrichment_config.py          # Parameterized multipliers
├── enrichment_monitor.py         # Daily/weekly monitoring jobs
├── tests/
│   └── test_enrichments.py       # Smoke tests
└── streamlit_app.py              # Add Enrichment Validation page
```
