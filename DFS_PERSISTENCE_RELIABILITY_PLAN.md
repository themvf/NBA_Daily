# DFS Persistence Reliability Plan

## Context

On the March 13, 2026 slate, lineups were clearly generated and exported locally, but the
tracking database and postmortem pipeline did not retain the slate data needed for review.

Observed symptoms:

- `dk_lineups_3_13_2026.csv` exists locally and contains generated lineups.
- `dfs_slate_lineups`, `dfs_slate_projections`, and `dfs_contest_entries` had no
  `2026-03-13` rows in the local tracking database.
- Tournament postmortem export returned an empty packet with
  `No contest standings found for this slate.`

## Current Architecture

Current behavior is a hybrid:

- Live reads and writes use a local SQLite database (`nba_stats.db`).
- Cloud storage is used for:
  - uploading the SQLite file as a backup artifact
  - uploading per-slate CSV/JSON artifacts under `dfs_slates/<slate_date>/...`
- The cloud client is GCS-first when GCS secrets are configured.

Important limitation:

- GCS is object storage, not a transactional database engine.
- The current save flow is best-effort and can fail without blocking the UI from reporting
  a successful lineup generation run.

## Reliability Goal

The system should only report a lineup run as successfully saved when:

1. the intended slate date is explicit and correct
2. the local/primary persistence write succeeds
3. saved row counts are verified immediately after write
4. cloud artifacts upload successfully
5. upload targets are visible and auditable

## Recommended Plan

### Phase 1: Immediate Hardening

Estimated effort: 2-4 hours

Scope:

- Log and display the exact `db_path`, `slate_date`, and cloud target before saving.
- Fail loudly if local SQLite save fails.
- Read back saved row counts after `save_slate_projections` and `save_slate_lineups`.
- Fail loudly if cloud upload fails when cloud persistence is enabled.
- Remove silent exception swallowing around lineup-tracking persistence.

Expected outcome:

- We can prove where a lineup run was written.
- A run no longer appears "saved" when persistence actually failed.

### Phase 2: Run-Based Persistence

Estimated effort: 1-2 days

Scope:

- Introduce an immutable `run_id` for every lineup generation batch.
- Add a `dfs_lineup_runs` table for run metadata.
- Store projections, lineups, and manifests by `run_id` instead of only `slate_date`.
- Stop destructive overwrite behavior that deletes all lineup rows for a slate before insert.
- Record save status for:
  - local DB write
  - artifact bundle upload
  - DB backup upload

Expected outcome:

- Every lineup generation run is auditable.
- Multiple runs for the same slate can coexist without overwriting each other.
- Postmortem analysis can target a specific run instead of assuming one slate snapshot.

### Phase 3: Optional Cloud Source of Truth

Estimated effort: 3-7 days

Scope:

- Move operational tracking data to a real cloud database such as Cloud SQL/Postgres.
- Keep GCS only for immutable artifacts and backups.
- Use GCS paths like `dfs_slates/<slate_date>/<run_id>/...`.
- Add manifest verification and checksum/size validation for uploaded artifacts.

Expected outcome:

- The application no longer depends on a local SQLite file as the operational source of truth.
- Cloud persistence becomes transactional and queryable.

## Senior-Engineer Position

The most reliable long-term design is:

- transactional database as source of truth
- immutable run IDs
- append-only persistence for lineup runs
- GCS used for artifacts and backups, not as the live database
- no silent failures in the persistence path

## Practical Recommendation

Implement Phase 1 immediately and Phase 2 next.

That sequence gives a fast reduction in operational risk without requiring a full platform
rewrite. Phase 3 should only be taken if the DFS tracking workflow needs strong cloud-native
guarantees across sessions, deployments, and users.

## Acceptance Criteria

- A generated lineup run cannot be marked saved unless persistence verification passes.
- The UI shows the saved `slate_date`, `run_id`, `db_path`, and cloud artifact prefix.
- Postmortem review can reference a concrete saved run and confirm underlying data exists.
- A missing March 13-style save failure becomes diagnosable in one screen without manual DB
  inspection.
