# NBA Daily - Claude Instructions

## Workflow
- All changes must be pushed to GitHub for Streamlit Cloud deployment
- Test locally before publishing to Git
- Add team logos when charts are based on teams

## Before Planning New Features
- Read `docs/PLANNING_CHECKLIST.md` for pre-implementation verification
- Reference `docs/FANDUEL_ODDS_ARCHITECTURE.md` for odds/betting features
- Check schema migration order in `prediction_tracking.py`

## Technical Notes
- Use LeagueGameLog endpoint (not BoxScoreTraditionalV2) for player stats
- S3 backup triggers automatically after FanDuel line fetches
- Timezone: The Odds API returns UTC, convert to US/Eastern before filtering
