# Planning Checklist

Use this checklist before implementing new features to avoid common issues.

---

## 1. Call Path Tracing

Before writing code, trace the full execution path:

- [ ] Identify the UI entry point (button, form, etc.)
- [ ] List all functions that will be called
- [ ] Check for internal safeguards/checks that might block the feature
- [ ] Verify function signatures match expected parameters

**Example (Force Refetch Bug):**
```
UI checkbox → fetch_fanduel_lines_for_date() → should_fetch_odds() ← BLOCKED HERE
```
The internal `should_fetch_odds()` call wasn't visible in the plan.

---

## 2. Schema & Database

- [ ] List all tables/columns the feature requires
- [ ] Check migration dependency order:
  - `upgrade_predictions_table_for_fanduel()` BEFORE
  - `upgrade_predictions_table_for_fanduel_comparison()`
- [ ] Use PRAGMA table_info to check if columns exist before querying
- [ ] Test with empty data scenarios

---

## 3. Imports & Dependencies

- [ ] List all imports the new code requires
- [ ] Verify imports exist at module level in target files
- [ ] Common missing imports:
  - `timedelta` (often forgotten alongside `datetime`)
  - `ZoneInfo` (for timezone handling)
  - `Path` (for file operations)

---

## 4. Edge Cases

### Timezone
- [ ] The Odds API returns UTC timestamps
- [ ] Convert to US/Eastern before date filtering
- [ ] Late-night Eastern games may have next-day UTC dates

### Already-Fetched States
- [ ] Check if `should_fetch_odds()` will block the operation
- [ ] Add `force` parameter if bypass is needed
- [ ] Update both UI logic AND function internals

### Empty/Missing Data
- [ ] Handle empty DataFrames gracefully
- [ ] Check for NULL/None values before calculations
- [ ] Provide user-friendly messages for missing data

---

## 5. Integration Points

When modifying existing functions:

- [ ] Document current function signature
- [ ] List all callers of the function
- [ ] Update callers if signature changes
- [ ] Verify return types match expectations

---

## 6. Post-Implementation

- [ ] Test the happy path locally
- [ ] Test each edge case identified above
- [ ] Verify S3 backup triggered (if data changed)
- [ ] Push to GitHub
- [ ] Verify Streamlit Cloud deployment succeeds

---

## Lessons Learned Log

Add new issues here as they're discovered:

| Date | Issue | Root Cause | Checklist Item Added |
|------|-------|------------|---------------------|
| 2026-01-02 | Force refetch blocked | Internal safeguard not traced | Call Path Tracing |
| 2026-01-02 | Schema migration failed | Wrong upgrade order | Schema & Database |
| 2026-01-02 | NameError: timedelta | Missing import | Imports & Dependencies |
| 2026-01-02 | Late games filtered out | UTC vs Eastern timezone | Edge Cases - Timezone |
