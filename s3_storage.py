#!/usr/bin/env python3
"""S3 storage integration for persisting NBA predictions across Streamlit Cloud redeploys."""

import boto3
from pathlib import Path
from datetime import datetime
from typing import Optional
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


class S3PredictionStorage:
    """Handles uploading and downloading the predictions database to/from S3."""

    def __init__(self):
        """Initialize S3 client using Streamlit secrets."""
        try:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=st.secrets["aws"]["access_key_id"],
                aws_secret_access_key=st.secrets["aws"]["secret_access_key"],
                region_name=st.secrets["aws"]["region"]
            )
            self.bucket = st.secrets["aws"]["bucket_name"]
            self.db_key = "predictions/nba_stats.db"
            self.connected = True
        except (KeyError, NoCredentialsError) as e:
            self.connected = False
            self.error = str(e)

    def is_connected(self) -> bool:
        """Check if S3 connection is configured."""
        return self.connected

    def upload_database(self, db_path: Path) -> tuple[bool, str]:
        """
        Upload SQLite database to S3.

        Args:
            db_path: Path to local database file

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.connected:
            return False, f"S3 not configured: {self.error}"

        if not db_path.exists():
            return False, f"Database file not found: {db_path}"

        try:
            # Upload with metadata
            self.s3.upload_file(
                str(db_path),
                self.bucket,
                self.db_key,
                ExtraArgs={
                    'Metadata': {
                        'upload_time': datetime.now().isoformat(),
                        'source': 'streamlit_app'
                    }
                }
            )
            file_size = db_path.stat().st_size / 1024  # KB
            return True, f"Uploaded {file_size:.1f} KB to S3"

        except ClientError as e:
            error_code = e.response['Error']['Code']
            return False, f"S3 upload failed ({error_code}): {e}"
        except Exception as e:
            return False, f"Unexpected error during upload: {e}"

    def download_database(self, db_path: Path) -> tuple[bool, str]:
        """
        Download SQLite database from S3.

        Args:
            db_path: Path where database should be saved locally

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.connected:
            return False, f"S3 not configured: {self.error}"

        try:
            # Check if database exists in S3
            self.s3.head_object(Bucket=self.bucket, Key=self.db_key)

            # Download the file
            self.s3.download_file(
                self.bucket,
                self.db_key,
                str(db_path)
            )

            if db_path.exists():
                file_size = db_path.stat().st_size / 1024  # KB
                return True, f"Downloaded {file_size:.1f} KB from S3"
            else:
                return False, "Download completed but file not found"

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                return False, "No backup found in S3 (this is normal for first run)"
            return False, f"S3 download failed ({error_code}): {e}"
        except Exception as e:
            return False, f"Unexpected error during download: {e}"

    def get_last_backup_time(self) -> Optional[datetime]:
        """
        Get the timestamp of the last backup in S3.

        Returns:
            datetime of last backup, or None if no backup exists
        """
        if not self.connected:
            return None

        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=self.db_key)
            # Get LastModified from response
            return response['LastModified']
        except ClientError:
            return None
        except Exception:
            return None

    def backup_exists(self) -> bool:
        """Check if a backup exists in S3."""
        if not self.connected:
            return False

        try:
            self.s3.head_object(Bucket=self.bucket, Key=self.db_key)
            return True
        except ClientError:
            return False
        except Exception:
            return False

    def get_backup_info(self) -> dict:
        """
        Get information about the S3 backup.

        Returns:
            Dict with backup metadata, or empty dict if no backup exists
        """
        if not self.connected:
            return {}

        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=self.db_key)
            return {
                'exists': True,
                'last_modified': response['LastModified'],
                'size_bytes': response['ContentLength'],
                'size_kb': response['ContentLength'] / 1024,
                'metadata': response.get('Metadata', {})
            }
        except ClientError:
            return {'exists': False}
        except Exception as e:
            return {'exists': False, 'error': str(e)}


def sync_database_to_s3(db_path: Path) -> tuple[bool, str]:
    """
    Convenience function to sync database to S3.

    Args:
        db_path: Path to local database

    Returns:
        Tuple of (success: bool, message: str)
    """
    storage = S3PredictionStorage()
    return storage.upload_database(db_path)


def restore_database_from_s3(db_path: Path) -> tuple[bool, str]:
    """
    Convenience function to restore database from S3.

    Args:
        db_path: Path where database should be saved

    Returns:
        Tuple of (success: bool, message: str)
    """
    storage = S3PredictionStorage()
    return storage.download_database(db_path)
