import bcrypt
import db_manager as db
import os
import sys

USERNAME = "testuser"
NAME = "Admin User"
PASSWORD = "password123"  # Temporary password

if not os.path.exists(db.DATABASE_FILE):
    print(f"⚠️ Database file not found. Run 'python db_manager.py' first to initialize it.")
    sys.exit(1)

print(f"[INFO] Hashing password for {USERNAME}...")
try:
    hashed_password = bcrypt.hashpw(PASSWORD.encode(), bcrypt.gensalt()).decode()
except Exception as e:
    print(f"[ERROR] Could not hash password: {e}")
    sys.exit(1)

print(f"[INFO] Inserting user '{USERNAME}'...")
existing = db.get_user(USERNAME)
if existing:
    print(f"✅ User '{USERNAME}' already exists.")
else:
    if db.add_user(USERNAME, NAME, hashed_password):
        print(f"✅ SUCCESS: '{USERNAME}' created successfully!")
    else:
        print(f"❌ FAILURE: Could not insert user.")

print("\nNext step ➜ Run `streamlit run app.py`")
