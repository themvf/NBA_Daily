# Fix IAM Permissions for S3 Access

## Current Status
✅ S3 connection works
❌ IAM user lacks permissions to access the bucket

## Problem
Your IAM user `nbadaily` can authenticate but cannot:
- List bucket contents (`s3:ListBucket`)
- Upload files (`s3:PutObject`)
- Download files (`s3:GetObject`)
- Delete files (`s3:DeleteObject`)

## Solution: Attach Permissions Policy

### Step 1: Go to IAM Console
1. Open AWS Console: https://console.aws.amazon.com/
2. Search for "IAM" and click on it
3. Click "Users" in the left sidebar
4. Click on your user: **nbadaily**

### Step 2: Create Custom Policy
1. Click "Policies" in the left sidebar
2. Click **"Create policy"** button
3. Click the **"JSON"** tab
4. Delete everything and paste this:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ListBucket",
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetBucketLocation"
            ],
            "Resource": "arn:aws:s3:::nbadaily"
        },
        {
            "Sid": "ReadWriteObjects",
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject",
                "s3:PutObjectAcl"
            ],
            "Resource": "arn:aws:s3:::nbadaily/*"
        }
    ]
}
```

5. Click **"Next"**

### Step 3: Name the Policy
1. **Policy name**: `nbadaily-s3-access`
2. **Description**: Allow read/write access to nbadaily S3 bucket
3. Click **"Create policy"**

### Step 4: Attach Policy to User
1. Go back to **"Users"** in the left sidebar
2. Click on **nbadaily** user
3. Click **"Add permissions"** dropdown
4. Select **"Attach policies directly"**
5. In the search box, type: `nbadaily-s3-access`
6. Check the box next to your new policy
7. Click **"Add permissions"**

### Step 5: Verify Permissions
1. On the user page, click **"Permissions"** tab
2. You should see: `nbadaily-s3-access` listed

---

## After Adding Permissions

Come back and run this command to test:

```bash
python test_s3_connection.py
```

All tests should pass!

---

## Common Issues

**Policy already exists?**
- If you see an error that the policy exists, just search for it and attach it

**Can't find the policy?**
- Make sure you're searching in the **Policies** section when attaching
- Try refreshing the page

**Still getting errors?**
- Double-check the bucket name in the JSON is `nbadaily`
- Make sure the Resource ARN matches your bucket
- Verify the user name is correct

---

## What This Policy Does

- **ListBucket**: Lets the app see what files are in the bucket
- **GetObject**: Lets the app download files from S3
- **PutObject**: Lets the app upload files to S3
- **DeleteObject**: Lets the app delete old backups
- **PutObjectAcl**: Lets the app set permissions on uploaded files

All actions are restricted to ONLY the `nbadaily` bucket. The user can't access any other S3 buckets in your account.

---

Ready? Follow the steps above, then let me know when you're done!
