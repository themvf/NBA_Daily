# Streamlit Cloud Storage Setup (GCS First)

This app now supports cloud persistence through `S3PredictionStorage` with this priority:

1. Google Cloud Storage (`[gcs]`)
2. AWS S3 (`[aws]`) fallback

If `[gcs]` is configured, all database backups and DFS slate artifacts are written to GCS.

## 1. Add GCS Secrets in Streamlit Cloud

In Streamlit Cloud:

1. Open your app
2. Go to `Settings` -> `Secrets`
3. Paste a `[gcs]` block like this:

```toml
[gcs]
bucket_name = "YOUR_GCS_BUCKET"
project = "YOUR_GCP_PROJECT_ID"
service_account_json = """
{
  "type": "service_account",
  "project_id": "YOUR_GCP_PROJECT_ID",
  "private_key_id": "YOUR_PRIVATE_KEY_ID",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "YOUR_SA@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com",
  "client_id": "YOUR_CLIENT_ID",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/..."
}
"""
```

Notes:

- `bucket_name` is required.
- `project` is optional if it is present in the service account JSON.
- `service_account_json` is optional only if ADC is available (usually not the case in Streamlit Cloud).

## 2. Required IAM Permissions

The service account should have bucket/object permissions for:

- upload objects
- download objects
- list objects
- read object metadata

`Storage Object Admin` on the specific bucket is the simplest option.

## 3. Optional S3 Fallback

You can still keep `[aws]` secrets. The app will only use S3 if `[gcs]` is missing or invalid.

```toml
[aws]
access_key_id = "..."
secret_access_key = "..."
bucket_name = "..."
region = "us-east-1"
```

## 4. Verify in the App

After saving secrets and rebooting the app:

1. Upload a DK CSV and generate projections
2. Run AI review and apply changes
3. Open `Cloud Slate Archive` in the DFS app

You should see files under `dfs_slates/<date>/`, including:

- `dk_salaries.csv`
- `projections.csv`
- `ai_review.json`
- `ai_adjustments.json`

The SQLite database backup remains at:

- `predictions/nba_stats.db`

## 5. Security Notes

- Never commit service account keys to Git.
- Keep credentials only in Streamlit Cloud secrets.
- Rotate keys immediately if they were exposed.
