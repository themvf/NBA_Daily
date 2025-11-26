# How to Create Access Keys for Your IAM User

## You're Here Because:
- You created an IAM user (great!)
- You need to generate access keys for that user
- The app will use these keys to access S3

---

## Steps to Create Access Keys:

### 1. Go to IAM Console
- In AWS Console search bar, type: **IAM**
- Click on **IAM** (Identity and Access Management)

### 2. Find Your User
- In the left sidebar, click **"Users"**
- You should see your user in the list (e.g., `nba-daily-app`)
- **Click on the username** to open user details

### 3. Go to Security Credentials Tab
- You'll see several tabs at the top
- Click the **"Security credentials"** tab

### 4. Create Access Key
- Scroll down to the **"Access keys"** section
- Click the **"Create access key"** button

### 5. Select Use Case
- You'll see several options
- Select: **"Application running outside AWS"**
- Check the box: "I understand the above recommendation and want to proceed to create an access key."
- Click **"Next"**

### 6. Add Description (Optional)
- Description tag (optional): `NBA Daily Streamlit App`
- Click **"Create access key"**

### 7. SAVE YOUR KEYS! 🔴 IMPORTANT
You'll now see two values:

```
Access key ID: AKIAIOSFODNN7EXAMPLE
Secret access key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

**⚠️ WARNING:** You can only see the secret key ONCE!

**Save them NOW using ONE of these methods:**

**Option A: Download CSV** (Recommended)
- Click **"Download .csv file"**
- Save the file somewhere safe (like Documents folder)
- Don't lose this file!

**Option B: Copy to Notepad**
- Open Notepad
- Copy Access Key ID and paste
- Copy Secret Access Key and paste
- Save the file as `aws-keys.txt` in a safe place

**Option C: Password Manager**
- Copy both values to your password manager (1Password, LastPass, etc.)

### 8. Click "Done"
- After saving, click **"Done"**
- You'll see the access key listed (but secret will be hidden)

---

## What These Keys Look Like:

```
Access Key ID:     AKIAIOSFODNN7EXAMPLE
                   ↑ Starts with "AKIA"
                   ↑ 20 characters long
                   ↑ This one is NOT secret (it's like a username)

Secret Access Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
                   ↑ Long random string (40 characters)
                   ↑ This one IS secret (like a password)
                   ↑ NEVER share this or commit to Git!
```

---

## After You Have the Keys:

Reply back with:
1. ✅ "I have saved my access keys"
2. Your bucket name (e.g., `nba-daily-predictions`)
3. Your AWS region (e.g., `us-east-1`)

Then I'll help you:
- Create the secrets file locally
- Test the connection
- Integrate S3 into your app

---

## Troubleshooting:

**"I don't see 'Create access key' button"**
- Make sure you're on the "Security credentials" tab
- Scroll down - it's below the "Sign-in credentials" section

**"I already created keys but lost them"**
- No problem! Delete the old keys and create new ones
- In the "Access keys" section, click the X next to the old key
- Then create new keys

**"I accidentally closed the window without saving"**
- The access key ID is still visible, but the secret is gone forever
- Delete that key and create a new one
- This time, download the CSV immediately!

---

## Security Reminder:

✅ DO:
- Save keys in a secure location
- Download the CSV file as backup
- Keep keys private

❌ DON'T:
- Commit keys to Git
- Share keys in screenshots
- Email keys to anyone
- Post keys in chat/Slack

---

Ready? Let me know when you have the keys saved!
