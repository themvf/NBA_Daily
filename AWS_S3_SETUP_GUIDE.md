# AWS S3 Setup Guide for NBA Daily

This guide will walk you through setting up AWS S3 for storing your NBA predictions.

## Prerequisites
- AWS account (you have this! ✓)
- Admin access to your AWS account

## Estimated Time: 10-15 minutes

---

## Part 1: Create an S3 Bucket

### Step 1: Log into AWS Console
1. Go to https://console.aws.amazon.com/
2. Sign in with your AWS account credentials

### Step 2: Navigate to S3
1. In the AWS Console search bar at the top, type **"S3"**
2. Click on **"S3"** (it should say "Scalable Storage in the Cloud")
3. You'll see the S3 dashboard

### Step 3: Create a New Bucket
1. Click the orange **"Create bucket"** button
2. Fill in the following:

   **Bucket name:** `nba-daily-predictions`
   - Note: Bucket names must be globally unique
   - If this name is taken, try: `nba-daily-predictions-YOURNAME` or `nba-daily-predictions-2025`
   - Remember the exact name you choose!

   **AWS Region:** Choose the region closest to you
   - US East (N. Virginia) = `us-east-1` (recommended)
   - US West (Oregon) = `us-west-2`
   - Remember which region you choose!

3. **Object Ownership:** Leave as "ACLs disabled (recommended)"

4. **Block Public Access settings:**
   - ✅ **KEEP ALL 4 BOXES CHECKED**
   - This ensures your predictions are private
   - Do NOT uncheck anything here

5. **Bucket Versioning:**
   - Select **"Enable"**
   - This keeps backup history of your database

6. **Default encryption:**
   - Select **"Server-side encryption with Amazon S3 managed keys (SSE-S3)"**
   - Encryption type: **"Enable"**

7. Leave all other settings as default

8. Click **"Create bucket"** at the bottom

### Step 4: Verify Bucket Creation
- You should see your new bucket in the list
- Click on the bucket name to open it
- It should be empty (that's normal!)

✅ **Part 1 Complete!** Your S3 bucket is ready.

---

## Part 2: Create IAM User and Access Keys

### Step 5: Navigate to IAM
1. In the AWS Console search bar, type **"IAM"**
2. Click on **"IAM"** (Identity and Access Management)

### Step 6: Create a New User
1. In the left sidebar, click **"Users"**
2. Click the orange **"Create user"** button
3. Fill in the details:

   **User name:** `nba-daily-app`

4. **Set permissions:** Click **"Attach policies directly"**

5. In the search box, type: **"S3"**

6. Find and check the box next to: **"AmazonS3FullAccess"**
   - (We'll restrict this to just your bucket in Step 8)

7. Click **"Next"**

8. Review the settings and click **"Create user"**

### Step 7: Create Access Keys
1. After creating the user, you'll see the user list
2. Click on the username **"nba-daily-app"** you just created
3. Click the **"Security credentials"** tab
4. Scroll down to **"Access keys"** section
5. Click **"Create access key"**
6. Select use case: **"Application running outside AWS"**
7. Check the box: "I understand the above recommendation..."
8. Click **"Next"**
9. (Optional) Add description tag: "NBA Daily Streamlit App"
10. Click **"Create access key"**

### Step 8: Save Your Access Keys
🔴 **CRITICAL:** You can only see the secret key ONCE!

You'll see two values:
- **Access key ID** (starts with "AKIA...")
- **Secret access key** (long random string)

**Save these NOW:**

1. Click **"Download .csv file"** button - save this file somewhere safe
2. OR copy both values to a password manager
3. Keep these secret - never commit to Git or share publicly

Example (yours will be different):
```
Access key ID: AKIAIOSFODNN7EXAMPLE
Secret access key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

✅ **Part 2 Complete!** You have your AWS credentials.

---

## Part 3: Restrict IAM Permissions (Security Best Practice)

Now let's limit this user to ONLY your specific bucket.

### Step 9: Create Custom Policy
1. Still in IAM, click **"Policies"** in the left sidebar
2. Click **"Create policy"**
3. Click the **"JSON"** tab
4. Delete the existing JSON and paste this:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetBucketLocation"
            ],
            "Resource": "arn:aws:s3:::nba-daily-predictions"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject",
                "s3:PutObjectAcl"
            ],
            "Resource": "arn:aws:s3:::nba-daily-predictions/*"
        }
    ]
}
```

⚠️ **IMPORTANT:** Replace `nba-daily-predictions` with YOUR actual bucket name (if different)

5. Click **"Next"**
6. **Policy name:** `NBA-Daily-S3-Access`
7. **Description:** "Allow access only to NBA Daily predictions bucket"
8. Click **"Create policy"**

### Step 10: Attach Custom Policy to User
1. Go back to **"Users"** in the left sidebar
2. Click on **"nba-daily-app"**
3. Click **"Add permissions"** → **"Attach policies directly"**
4. Search for: `NBA-Daily-S3-Access`
5. Check the box next to your new policy
6. Click **"Add permissions"**

### Step 11: Remove Broad S3 Access
1. Still on the user page, find **"Permissions policies"** section
2. Find **"AmazonS3FullAccess"** in the list
3. Click the **X** or **"Remove"** button next to it
4. Confirm removal

Now the user can ONLY access your specific bucket! 🔒

✅ **Part 3 Complete!** Secure permissions configured.

---

## Part 4: Test Connection (We'll do this together)

Once you complete the above steps, we'll:
1. Configure local secrets with your keys
2. Test uploading a file to S3
3. Test downloading from S3
4. Verify everything works

---

## Summary Checklist

Before moving to implementation, verify you have:

- [ ] Created S3 bucket (name: _____________)
- [ ] Noted the AWS region (e.g., us-east-1)
- [ ] Created IAM user: `nba-daily-app`
- [ ] Downloaded access keys CSV file (or saved them securely)
- [ ] Created custom policy: `NBA-Daily-S3-Access`
- [ ] Attached custom policy to user
- [ ] Removed `AmazonS3FullAccess` from user

---

## What You'll Need for Next Steps

Please have ready:
1. **Bucket name** (e.g., `nba-daily-predictions`)
2. **AWS region** (e.g., `us-east-1`)
3. **Access key ID** (starts with AKIA...)
4. **Secret access key** (the long random string)

⚠️ **Security Reminder:**
- NEVER commit these keys to Git
- NEVER share them publicly
- We'll store them in `.streamlit/secrets.toml` (which is gitignored)
- For Streamlit Cloud, we'll use the secure secrets UI

---

## Estimated Costs

**S3 Free Tier (first 12 months):**
- 5 GB storage
- 20,000 GET requests
- 2,000 PUT requests

**Your expected usage:**
- Storage: ~10 MB (well under limit)
- Requests: ~100/day (well under limit)
- **Cost: $0.00/month** ✅

**After free tier (or if you've used AWS >12 months):**
- Storage: $0.023/GB/month = $0.0002/month for 10MB
- Requests: Negligible
- **Cost: <$0.01/month** ✅

---

## Need Help?

If you get stuck on any step:
1. Take a screenshot of where you're stuck
2. Share it with me
3. I'll guide you through!

**Common issues:**
- Bucket name already taken → Try adding your name or year
- Can't find IAM → Make sure you're in the AWS Console (not billing, etc.)
- Lost access keys → Delete and create new ones (it's safe to do this)

---

Ready to proceed? Once you complete these steps, let me know and we'll test the connection!
