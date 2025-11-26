# S3 Integration Plan for NBA Daily Predictions

## Problem Statement
Streamlit Cloud uses an ephemeral file system that resets on every redeploy. This means:
- The SQLite database (`nba_stats.db`) is lost on redeploy
- All prediction history is lost
- Accuracy tracking requires persistent storage

## Solution: AWS S3 Integration

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│ Streamlit Cloud App Lifecycle                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. APP START                                            │
│     ├─> Check if local DB exists                        │
│     ├─> Download predictions from S3                    │
│     └─> Load into local SQLite DB                       │
│                                                          │
│  2. DURING SESSION                                       │
│     ├─> Users view predictions (Today's Games tab)      │
│     ├─> Predictions logged to local DB                  │
│     └─> Users can download CSV exports                  │
│                                                          │
│  3. BACKGROUND SYNC                                      │
│     ├─> Periodic upload to S3 (every 5 minutes)         │
│     └─> Upload triggered on prediction logging          │
│                                                          │
│  4. MANUAL EXPORT                                        │
│     └─> Download CSV button (immediate backup)          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Storage Strategy

#### Option 1: SQLite Database Sync (Recommended)
**Pros:**
- Simple implementation
- Complete database backup
- Easy to restore
- All data in one file

**Cons:**
- Potential for concurrent write conflicts
- Larger file size

**Implementation:**
```python
# Upload entire database to S3
s3.upload_file('nba_stats.db', bucket, 'predictions/nba_stats.db')

# Download on startup
s3.download_file(bucket, 'predictions/nba_stats.db', 'nba_stats.db')
```

#### Option 2: CSV Append (Alternative)
**Pros:**
- Human-readable format
- Easy to inspect in S3 console
- No conflict issues (append-only)

**Cons:**
- More complex to restore
- Need to deduplicate on restore
- Slower for large datasets

**Implementation:**
```python
# Append new predictions as CSV
daily_predictions_df.to_csv(f's3://bucket/predictions/{date}.csv')

# On startup, read all CSVs and merge
all_predictions = pd.concat([pd.read_csv(f) for f in s3_files])
```

### Required Components

#### 1. AWS Setup
- [ ] Create AWS account (if needed)
- [ ] Create S3 bucket (e.g., `nba-daily-predictions`)
- [ ] Create IAM user with S3 access
- [ ] Generate access keys
- [ ] Configure bucket policy (private, with user access)

#### 2. Python Dependencies
```txt
boto3>=1.28.0  # AWS SDK for Python
```

#### 3. Streamlit Secrets Configuration
```toml
# .streamlit/secrets.toml (local testing)
# Streamlit Cloud: Configure in app settings

[aws]
access_key_id = "YOUR_ACCESS_KEY"
secret_access_key = "YOUR_SECRET_KEY"
bucket_name = "nba-daily-predictions"
region = "us-east-1"
```

#### 4. New Python Module: `s3_storage.py`
```python
import boto3
from pathlib import Path
import streamlit as st
from datetime import datetime

class S3PredictionStorage:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=st.secrets["aws"]["access_key_id"],
            aws_secret_access_key=st.secrets["aws"]["secret_access_key"],
            region_name=st.secrets["aws"]["region"]
        )
        self.bucket = st.secrets["aws"]["bucket_name"]

    def upload_database(self, db_path: Path) -> bool:
        """Upload SQLite database to S3"""

    def download_database(self, db_path: Path) -> bool:
        """Download SQLite database from S3"""

    def backup_predictions_csv(self, predictions_df) -> str:
        """Backup predictions as CSV (additional safety)"""

    def list_backups(self) -> list:
        """List available backups in S3"""
```

### Implementation Steps

#### Phase 1: Basic S3 Setup (Local Testing)
1. Install boto3: `pip install boto3`
2. Create AWS account and S3 bucket
3. Create IAM user with S3 permissions
4. Configure local secrets in `.streamlit/secrets.toml`
5. Test S3 connection locally

#### Phase 2: Database Sync Implementation
1. Create `s3_storage.py` module
2. Implement upload/download functions
3. Add startup hook to download database
4. Add periodic sync (every 5 minutes)
5. Test locally with mock predictions

#### Phase 3: Streamlit Integration
1. Add S3 sync to app startup
2. Add S3 upload after prediction logging
3. Add manual backup button
4. Add status indicators (last sync time)
5. Add error handling and retry logic

#### Phase 4: Deployment
1. Add boto3 to `requirements.txt`
2. Configure secrets in Streamlit Cloud
3. Deploy and test on Streamlit Cloud
4. Verify predictions persist across redeploys

### File Changes Required

#### New Files
- `s3_storage.py` - S3 integration module
- `test_s3_integration.py` - Test S3 functionality
- `S3_SETUP_GUIDE.md` - User guide for AWS setup

#### Modified Files
- `streamlit_app.py`:
  - Import s3_storage
  - Add startup sync
  - Add periodic background sync
  - Add manual backup button
  - Add sync status display

- `requirements.txt`:
  - Add boto3

- `prediction_tracking.py`:
  - Add callback after successful prediction log

#### Configuration Files
- `.streamlit/secrets.toml` (local, gitignored)
- Streamlit Cloud secrets (web UI)

### Cost Estimation

**AWS S3 Costs (Free Tier):**
- First 5GB storage: FREE
- First 20,000 GET requests: FREE
- First 2,000 PUT requests: FREE

**Expected Usage:**
- Database size: ~10MB (thousands of predictions)
- Requests per day: ~50 (syncs + downloads)
- Monthly cost: **~$0.01** (essentially free)

**Beyond Free Tier:**
- Storage: $0.023/GB/month
- PUT requests: $0.005 per 1,000 requests
- GET requests: $0.0004 per 1,000 requests

### Security Considerations

1. **Secrets Management**
   - Never commit AWS keys to Git
   - Use Streamlit secrets for cloud deployment
   - Rotate keys periodically

2. **Bucket Security**
   - Make bucket private (no public access)
   - Use IAM user with minimal permissions
   - Enable versioning (keep backup history)
   - Enable encryption at rest

3. **Access Control**
   - Create dedicated IAM user for this app only
   - Grant only S3 permissions (no EC2, etc.)
   - Restrict to specific bucket

### Testing Strategy

#### Unit Tests
```python
def test_upload_database():
    """Test uploading database to S3"""

def test_download_database():
    """Test downloading database from S3"""

def test_sync_on_prediction_log():
    """Test automatic sync after logging prediction"""
```

#### Integration Tests
1. Upload database with test predictions
2. Delete local database
3. Download from S3
4. Verify predictions match
5. Add new predictions
6. Upload again
7. Verify no duplicates

### Rollout Plan

#### Week 1: Setup & Development
- [ ] Day 1: AWS setup, create bucket, configure IAM
- [ ] Day 2: Implement s3_storage.py module
- [ ] Day 3: Test locally with mock data
- [ ] Day 4: Integrate with streamlit_app.py

#### Week 2: Testing & Deployment
- [ ] Day 1: End-to-end testing locally
- [ ] Day 2: Configure Streamlit Cloud secrets
- [ ] Day 3: Deploy to Streamlit Cloud
- [ ] Day 4: Monitor and verify persistence

### Alternative: Streamlit Cloud Database Options

If AWS setup is too complex, consider:

1. **Streamlit Community Cloud + GitHub Gist**
   - Store CSV backups in GitHub Gists
   - Simpler than S3
   - Limited to 100MB per Gist

2. **Google Cloud Storage (GCS)**
   - Similar to S3
   - $300 free credit for new users
   - Same architecture as S3 plan

3. **Supabase (PostgreSQL)**
   - Free tier: 500MB database
   - Better for relational data
   - More complex integration

### Next Steps

1. **Decide on storage strategy**: SQLite sync vs CSV append
2. **Create AWS account** (or choose alternative)
3. **Review security requirements**
4. **Begin Phase 1 implementation**

### Questions to Answer

- [ ] Do you have an AWS account already?
- [ ] Are you comfortable with AWS billing (even if minimal)?
- [ ] Should we implement SQLite sync or CSV append?
- [ ] Do you want manual backup only, or automatic sync?
- [ ] Should we add backup to GitHub Gists as additional safety?

---

**Estimated Implementation Time:** 4-6 hours
**Estimated Monthly Cost:** $0.00 - $0.05
**Risk Level:** Low (reversible, non-destructive)
