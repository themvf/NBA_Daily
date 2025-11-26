#!/usr/bin/env python3
"""Test S3 connection and upload/download functionality."""

import sys
from pathlib import Path
import tempfile
import streamlit as st

# Mock streamlit secrets for testing outside of streamlit app
if not hasattr(st, 'secrets') or not st.secrets:
    import toml
    secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        st.secrets = toml.load(secrets_path)
    else:
        print(f"[ERROR] Secrets file not found: {secrets_path}")
        sys.exit(1)

import s3_storage


def test_connection():
    """Test basic S3 connection."""
    print("=" * 60)
    print("Testing S3 Connection")
    print("=" * 60)

    storage = s3_storage.S3PredictionStorage()

    if not storage.is_connected():
        print(f"[ERROR] Failed to connect to S3: {storage.error}")
        return False

    print("[OK] S3 client initialized successfully")
    print(f"     Bucket: {storage.bucket}")
    print(f"     Region: {st.secrets['aws']['region']}")
    return True


def test_bucket_access():
    """Test if we can access the bucket."""
    print("\n" + "=" * 60)
    print("Testing Bucket Access")
    print("=" * 60)

    storage = s3_storage.S3PredictionStorage()

    # Try to list objects (this will fail if we don't have access)
    try:
        response = storage.s3.list_objects_v2(
            Bucket=storage.bucket,
            MaxKeys=1
        )
        print(f"[OK] Successfully accessed bucket '{storage.bucket}'")

        if 'Contents' in response:
            print(f"     Bucket contains {response.get('KeyCount', 0)} objects")
        else:
            print("     Bucket is empty (this is normal for new buckets)")

        return True
    except Exception as e:
        print(f"[ERROR] Cannot access bucket: {e}")
        return False


def test_upload():
    """Test uploading a file to S3."""
    print("\n" + "=" * 60)
    print("Testing File Upload")
    print("=" * 60)

    storage = s3_storage.S3PredictionStorage()

    # Create a temporary test file
    test_file = Path(tempfile.gettempdir()) / "test_nba_upload.txt"
    test_content = "This is a test upload from NBA Daily app"
    test_file.write_text(test_content)

    print(f"[INFO] Created test file: {test_file}")
    print(f"[INFO] File size: {test_file.stat().st_size} bytes")

    # Upload test file
    try:
        storage.s3.upload_file(
            str(test_file),
            storage.bucket,
            "test/test_upload.txt"
        )
        print("[OK] Successfully uploaded test file to S3")
        print("     Location: s3://{}/test/test_upload.txt".format(storage.bucket))

        # Clean up local test file
        test_file.unlink()
        return True
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        test_file.unlink()
        return False


def test_download():
    """Test downloading a file from S3."""
    print("\n" + "=" * 60)
    print("Testing File Download")
    print("=" * 60)

    storage = s3_storage.S3PredictionStorage()

    # Download the test file we just uploaded
    download_file = Path(tempfile.gettempdir()) / "test_nba_download.txt"

    try:
        storage.s3.download_file(
            storage.bucket,
            "test/test_upload.txt",
            str(download_file)
        )
        print("[OK] Successfully downloaded test file from S3")

        # Verify content
        content = download_file.read_text()
        expected = "This is a test upload from NBA Daily app"

        if content == expected:
            print("[OK] Downloaded content matches uploaded content")
        else:
            print(f"[WARNING] Content mismatch:")
            print(f"          Expected: {expected}")
            print(f"          Got: {content}")

        # Clean up
        download_file.unlink()
        return True
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def test_database_operations():
    """Test uploading and downloading the actual database."""
    print("\n" + "=" * 60)
    print("Testing Database Upload/Download")
    print("=" * 60)

    db_path = Path(__file__).parent / "nba_stats.db"

    if not db_path.exists():
        print(f"[WARNING] Database not found at {db_path}")
        print("          Creating a dummy database for testing...")
        db_path.write_bytes(b"DUMMY_DATABASE_FOR_TESTING")
        created_dummy = True
    else:
        created_dummy = False
        print(f"[INFO] Using existing database: {db_path}")
        print(f"[INFO] Database size: {db_path.stat().st_size / 1024:.1f} KB")

    storage = s3_storage.S3PredictionStorage()

    # Test upload
    print("\n[TEST] Uploading database to S3...")
    success, message = storage.upload_database(db_path)

    if success:
        print(f"[OK] {message}")
    else:
        print(f"[ERROR] {message}")
        if created_dummy:
            db_path.unlink()
        return False

    # Test backup info
    print("\n[TEST] Getting backup info from S3...")
    info = storage.get_backup_info()

    if info.get('exists'):
        print("[OK] Backup exists in S3")
        print(f"     Last modified: {info['last_modified']}")
        print(f"     Size: {info['size_kb']:.1f} KB")
    else:
        print(f"[ERROR] Backup not found: {info.get('error', 'Unknown error')}")
        if created_dummy:
            db_path.unlink()
        return False

    # Test download to different location
    print("\n[TEST] Downloading database from S3...")
    download_path = Path(tempfile.gettempdir()) / "test_downloaded_db.db"

    success, message = storage.download_database(download_path)

    if success:
        print(f"[OK] {message}")
        print(f"[OK] Downloaded to: {download_path}")

        # Compare sizes
        original_size = db_path.stat().st_size
        downloaded_size = download_path.stat().st_size

        if original_size == downloaded_size:
            print(f"[OK] File sizes match: {original_size} bytes")
        else:
            print(f"[WARNING] Size mismatch: {original_size} vs {downloaded_size}")

        # Clean up
        download_path.unlink()
    else:
        print(f"[ERROR] {message}")
        if created_dummy:
            db_path.unlink()
        return False

    # Clean up dummy database if we created one
    if created_dummy:
        db_path.unlink()
        print("\n[INFO] Cleaned up dummy database")

    return True


def cleanup_test_files():
    """Clean up test files from S3."""
    print("\n" + "=" * 60)
    print("Cleaning Up Test Files")
    print("=" * 60)

    storage = s3_storage.S3PredictionStorage()

    try:
        # Delete test file
        storage.s3.delete_object(
            Bucket=storage.bucket,
            Key="test/test_upload.txt"
        )
        print("[OK] Cleaned up test files from S3")
    except Exception as e:
        print(f"[WARNING] Cleanup failed (this is okay): {e}")


def main():
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print("  S3 Integration Test Suite")
    print("  NBA Daily Prediction Storage")
    print("=" * 60)
    print()

    results = {
        "Connection": test_connection(),
        "Bucket Access": test_bucket_access(),
        "File Upload": test_upload(),
        "File Download": test_download(),
        "Database Operations": test_database_operations(),
    }

    cleanup_test_files()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name:.<40} {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! S3 integration is working correctly.")
        print("\nNext steps:")
        print("1. Integrate S3 sync into streamlit_app.py")
        print("2. Configure secrets in Streamlit Cloud")
        print("3. Deploy and test on Streamlit Cloud")
    else:
        print("Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("- Check AWS credentials in .streamlit/secrets.toml")
        print("- Verify bucket name is correct")
        print("- Ensure IAM user has S3 permissions")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
