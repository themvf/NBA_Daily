# Repository Guidelines

## Project Structure & Module Organization
- `nba_to_sqlite.py` orchestrates all ingestion tasks: API pulls, SQLite schema creation, and analytical views. Treat it as the authoritative data builder.
- `predict_top_3pm.py` consumes the SQLite tables to engineer features and train a classifier that predicts per-team three-point leaders.
- `requirements.txt` pins runtime deps; `.venv/` holds local site-packages and must stay untracked. SQLite artifacts such as `nba_stats.db` or derived CSVs belong in the repo root but should be git-ignored if regenerated frequently.

## Build, Test, and Development Commands
- `python -m venv .venv && .\.venv\Scripts\Activate.ps1` creates/activates the project environment.
- `pip install -r requirements.txt` keeps the CLI utilities aligned with the tracked dependency set.
- `python nba_to_sqlite.py --season 2024-25 --season-type "Regular Season"` refreshes the local SQLite warehouse; add season overrides with the documented flags.
- `python predict_top_3pm.py --db-path nba_stats.db --train-seasons 2024-25 --test-seasons 2025-26 --output-predictions predictions.csv` retrains and exports the leaderboard model.

## Coding Style & Naming Conventions
- Follow PEP 8, prefer type annotations (already present) and docstrings for public helpers. Keep functions pure when practical; isolate I/O inside CLI layers.
- Use snake_case for modules, functions, and variables; reserve PascalCase for classes or sklearn estimators.
- Run `ruff` or `black` locally if you add them; otherwise keep formatting consistent with existing files (4-space indents, 88–100 char lines).

## Testing Guidelines
- Ad hoc verification relies on re-running the builder and ensuring tables/views match expectations. If you add logic-heavy modules, create lightweight pytest suites under `tests/`.
- Name tests after the module under test (e.g., `tests/test_predict_top_3pm.py`) and seed fixtures with small SQLite snapshots to avoid hitting the live API.

## Commit & Pull Request Guidelines
- Write commits in the imperative mood with scoped subjects: `Add Streamlit leaderboard page`, `Fix player log rollups`. Group unrelated work into separate commits.
- PRs should describe the user-facing change, list CLI/testing steps (e.g., `python nba_to_sqlite.py --season ...`), and include screenshots for UI changes such as future Streamlit views.
- Link GitHub issues when available and call out schema changes that require rerunning `nba_to_sqlite.py` so downstream users can react accordingly.

## Security & Configuration Tips
- Do not hardcode API keys (nba_api is public) or upload personal SQLite dumps; prefer `.gitignore` entries for large DB snapshots and credentials.
- When sharing predictions or dashboards, sanitize outputs so no locally cached credentials, browser cookies, or OS paths leak into logs.
