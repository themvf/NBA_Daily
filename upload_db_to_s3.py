"""
Upload the scored predictions database to S3.

This script uploads nba_stats.db to S3 so Streamlit Cloud can access the latest data.
Run this after scoring predictions to sync the results to the cloud app.
"""

import boto3
from pathlib import Path
from datetime import datetime
import os
from botocore.exceptions import ClientError, NoCredentialsError


def upload_database():
    """Upload nba_stats.db to S3."""

    # Get AWS credentials from environment variables
    # (Set these in your shell or .env file)
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_REGION', 'us-east-1')
    bucket_name = os.getenv('AWS_BUCKET_NAME')

    if not all([access_key, secret_key, bucket_name]):
        print("ERROR: Missing AWS credentials!")
        print("Please set the following environment variables:")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        print("  - AWS_BUCKET_NAME")
        print("  - AWS_REGION (optional, defaults to us-east-1)")
        return False

    # Database details
    db_path = Path("nba_stats.db")
    s3_key = "predictions/nba_stats.db"

    if not db_path.exists():
        print(f"ERROR: Database file not found: {db_path}")
        return False

    try:
        # Create S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )

        print(f"Uploading {db_path} to s3://{bucket_name}/{s3_key}...")

        # Upload with metadata
        s3.upload_file(
            str(db_path),
            bucket_name,
            s3_key,
            ExtraArgs={
                'Metadata': {
                    'upload_time': datetime.now().isoformat(),
                    'source': 'local_upload_script'
                }
            }
        )

        file_size = db_path.stat().st_size / 1024 / 1024  # MB
        print(f"SUCCESS! Uploaded {file_size:.2f} MB to S3")
        print(f"\nNext steps:")
        print("1. Restart your Streamlit Cloud app to download the updated database")
        print("2. Or wait for the next automatic deployment")

        return True

    except NoCredentialsError:
        print("ERROR: Invalid AWS credentials")
        return False
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"ERROR: S3 upload failed ({error_code}): {e}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return False


if __name__ == "__main__":
    print("="*70)
    print("NBA STATS DATABASE S3 UPLOADER")
    print("="*70)
    print()

    success = upload_database()

    if not success:
        print("\nUpload failed. Please check the errors above.")
        exit(1)
