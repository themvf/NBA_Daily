"""Cloud storage integration for persisting app data across redeploys.

Backward compatibility:
- Keeps the existing `S3PredictionStorage` class name and method surface.
- Prefers Google Cloud Storage when `[gcs]` secrets are configured.
- Falls back to S3 when `[aws]` secrets are configured.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

try:
    from google.cloud import storage as gcs_storage
    from google.oauth2 import service_account

    GCS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    gcs_storage = None  # type: ignore[assignment]
    service_account = None  # type: ignore[assignment]
    GCS_AVAILABLE = False


class S3PredictionStorage:
    """Cloud storage client with GCS-first backend selection."""

    def __init__(self):
        self.db_key = "predictions/nba_stats.db"
        self.connected = False
        self.error = ""
        self.backend: Optional[str] = None  # "gcs" | "s3"

        self.bucket_name = ""
        self.s3 = None
        self.gcs_client = None
        self.gcs_bucket = None

        # Prefer GCS when configured, then fall back to S3.
        self._init_gcs_backend()
        if not self.connected:
            self._init_s3_backend()

    # ---------------------------------------------------------------------
    # Backend Initialization
    # ---------------------------------------------------------------------
    def _secrets_block(self, key: str) -> Dict[str, Any]:
        """Read a secrets subsection as a plain dict."""
        try:
            block = st.secrets.get(key, None)
            if block is None:
                return {}
            if isinstance(block, dict):
                return dict(block)
            return dict(block)  # Supports Streamlit Secrets wrappers
        except Exception:
            return {}

    def _init_gcs_backend(self) -> None:
        cfg = self._secrets_block("gcs")
        bucket_name = str(cfg.get("bucket_name", "")).strip()
        if not bucket_name:
            return

        if not GCS_AVAILABLE:
            self.connected = False
            self.error = (
                "GCS secrets found but google-cloud-storage is not installed. "
                "Add google-cloud-storage to requirements."
            )
            return

        try:
            project = str(cfg.get("project", "")).strip() or None
            credentials = None

            # Option 1: full service-account JSON string
            sa_json = cfg.get("service_account_json")
            # Option 2: structured object in secrets
            sa_info = cfg.get("service_account_info")

            if sa_json:
                if isinstance(sa_json, str):
                    info = json.loads(sa_json)
                else:
                    info = dict(sa_json)
                credentials = service_account.Credentials.from_service_account_info(info)
                if not project:
                    project = info.get("project_id")
            elif sa_info:
                if isinstance(sa_info, str):
                    info = json.loads(sa_info)
                else:
                    info = dict(sa_info)
                credentials = service_account.Credentials.from_service_account_info(info)
                if not project:
                    project = info.get("project_id")

            if credentials is not None:
                self.gcs_client = gcs_storage.Client(project=project, credentials=credentials)
            else:
                # Uses GOOGLE_APPLICATION_CREDENTIALS / ADC if present.
                self.gcs_client = gcs_storage.Client(project=project) if project else gcs_storage.Client()

            self.bucket_name = bucket_name
            self.gcs_bucket = self.gcs_client.bucket(self.bucket_name)
            self.backend = "gcs"
            self.connected = True
            self.error = ""
        except Exception as e:
            self.connected = False
            self.error = f"GCS configuration error: {e}"

    def _init_s3_backend(self) -> None:
        cfg = self._secrets_block("aws")
        if not cfg:
            if not self.error:
                self.error = "No [gcs] or [aws] cloud storage secrets configured."
            return

        try:
            self.s3 = boto3.client(
                "s3",
                aws_access_key_id=cfg["access_key_id"],
                aws_secret_access_key=cfg["secret_access_key"],
                region_name=cfg["region"],
            )
            self.bucket_name = cfg["bucket_name"]
            self.backend = "s3"
            self.connected = True
            self.error = ""
        except (KeyError, NoCredentialsError) as e:
            self.connected = False
            if not self.error:
                self.error = str(e)
        except Exception as e:
            self.connected = False
            if not self.error:
                self.error = str(e)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _not_configured(self) -> tuple[bool, str]:
        # Preserve old "S3 not configured" prefix to avoid breaking existing UI checks.
        return False, f"S3 not configured: {self.error}"

    def _backend_label(self) -> str:
        return "GCS" if self.backend == "gcs" else "S3"

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def get_backend(self) -> str:
        """Return the active backend label."""
        if self.backend == "gcs":
            return "GCS"
        if self.backend == "s3":
            return "S3"
        return "Cloud"

    def is_connected(self) -> bool:
        """Check if cloud storage connection is configured."""
        return self.connected

    def upload_database(self, db_path: Path) -> tuple[bool, str]:
        """Upload SQLite database to the active cloud backend."""
        if not self.connected:
            return self._not_configured()
        if not db_path.exists():
            return False, f"Database file not found: {db_path}"

        success, message = self.upload_file(
            local_path=db_path,
            s3_key=self.db_key,
            metadata={
                "upload_time": datetime.now().isoformat(),
                "source": "streamlit_app",
            },
        )
        if not success:
            return False, message

        file_size = db_path.stat().st_size / 1024
        return True, f"Uploaded {file_size:.1f} KB to {self._backend_label()}"

    def download_database(self, db_path: Path) -> tuple[bool, str]:
        """Download SQLite database from the active cloud backend."""
        if not self.connected:
            return self._not_configured()

        try:
            if self.backend == "gcs":
                blob = self.gcs_bucket.blob(self.db_key)
                if not blob.exists(self.gcs_client):
                    return False, "No backup found in GCS (this is normal for first run)"
                db_path.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(db_path))
            else:
                self.s3.head_object(Bucket=self.bucket_name, Key=self.db_key)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(self.bucket_name, self.db_key, str(db_path))

            if db_path.exists():
                file_size = db_path.stat().st_size / 1024
                return True, f"Downloaded {file_size:.1f} KB from {self._backend_label()}"
            return False, "Download completed but file not found"

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "unknown")
            if error_code == "404":
                return False, "No backup found in S3 (this is normal for first run)"
            return False, f"S3 download failed ({error_code}): {e}"
        except Exception as e:
            # Keep "No backup found" phrase for compatibility checks in callers.
            if "not found" in str(e).lower():
                return False, f"No backup found in {self._backend_label()} (this is normal for first run)"
            return False, f"Unexpected error during download: {e}"

    def get_last_backup_time(self) -> Optional[datetime]:
        """Get timestamp of last cloud backup."""
        if not self.connected:
            return None

        try:
            if self.backend == "gcs":
                blob = self.gcs_bucket.blob(self.db_key)
                if not blob.exists(self.gcs_client):
                    return None
                blob.reload(self.gcs_client)
                return blob.updated

            response = self.s3.head_object(Bucket=self.bucket_name, Key=self.db_key)
            return response["LastModified"]
        except Exception:
            return None

    def backup_exists(self) -> bool:
        """Check if DB backup exists in cloud storage."""
        if not self.connected:
            return False

        try:
            if self.backend == "gcs":
                blob = self.gcs_bucket.blob(self.db_key)
                return blob.exists(self.gcs_client)

            self.s3.head_object(Bucket=self.bucket_name, Key=self.db_key)
            return True
        except Exception:
            return False

    def get_backup_info(self) -> dict:
        """Get metadata about the DB backup."""
        if not self.connected:
            return {}

        try:
            if self.backend == "gcs":
                blob = self.gcs_bucket.blob(self.db_key)
                if not blob.exists(self.gcs_client):
                    return {"exists": False}
                blob.reload(self.gcs_client)
                size = int(blob.size or 0)
                return {
                    "exists": True,
                    "last_modified": blob.updated,
                    "size_bytes": size,
                    "size_kb": size / 1024,
                    "metadata": blob.metadata or {},
                    "backend": "gcs",
                }

            response = self.s3.head_object(Bucket=self.bucket_name, Key=self.db_key)
            return {
                "exists": True,
                "last_modified": response["LastModified"],
                "size_bytes": response["ContentLength"],
                "size_kb": response["ContentLength"] / 1024,
                "metadata": response.get("Metadata", {}),
                "backend": "s3",
            }
        except ClientError:
            return {"exists": False}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def upload_file(
        self,
        local_path: Path,
        s3_key: str,
        metadata: Optional[dict] = None,
    ) -> tuple[bool, str]:
        """Upload any local file to an object key in the active backend."""
        if not self.connected:
            return self._not_configured()
        if not local_path.exists():
            return False, f"File not found: {local_path}"

        try:
            default_metadata = {
                "upload_time": datetime.now().isoformat(),
                "source": "streamlit_app",
            }
            final_metadata = {**default_metadata, **(metadata or {})}

            if self.backend == "gcs":
                blob = self.gcs_bucket.blob(s3_key)
                blob.metadata = {str(k): str(v) for k, v in final_metadata.items()}
                blob.upload_from_filename(str(local_path))
            else:
                self.s3.upload_file(
                    str(local_path),
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={"Metadata": {str(k): str(v) for k, v in final_metadata.items()}},
                )

            file_size = local_path.stat().st_size / 1024
            return True, f"Uploaded {file_size:.1f} KB to {s3_key}"
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "unknown")
            return False, f"S3 upload failed ({error_code}): {e}"
        except Exception as e:
            return False, f"Unexpected error during upload: {e}"

    def download_file(self, s3_key: str, local_path: Path) -> tuple[bool, str]:
        """Download any object key from cloud storage to a local path."""
        if not self.connected:
            return self._not_configured()

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if self.backend == "gcs":
                blob = self.gcs_bucket.blob(s3_key)
                if not blob.exists(self.gcs_client):
                    return False, "File not found in GCS"
                blob.download_to_filename(str(local_path))
            else:
                self.s3.head_object(Bucket=self.bucket_name, Key=s3_key)
                self.s3.download_file(self.bucket_name, s3_key, str(local_path))

            if local_path.exists():
                file_size = local_path.stat().st_size / 1024
                return True, f"Downloaded {file_size:.1f} KB"
            return False, "Download completed but file not found"
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "unknown")
            if error_code == "404":
                return False, "File not found in S3"
            return False, f"S3 download failed ({error_code}): {e}"
        except Exception as e:
            return False, f"Unexpected error during download: {e}"

    def list_files(self, prefix: str) -> List[str]:
        """List object keys under a prefix."""
        if not self.connected:
            return []

        try:
            if self.backend == "gcs":
                return [blob.name for blob in self.gcs_client.list_blobs(self.bucket_name, prefix=prefix)]

            keys: List[str] = []
            paginator = self.s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            return keys
        except Exception:
            return []


def sync_database_to_s3(db_path: Path) -> tuple[bool, str]:
    """Convenience function to sync database to configured cloud backend."""
    storage = S3PredictionStorage()
    return storage.upload_database(db_path)


def restore_database_from_s3(db_path: Path) -> tuple[bool, str]:
    """Convenience function to restore database from configured cloud backend."""
    storage = S3PredictionStorage()
    return storage.download_database(db_path)
