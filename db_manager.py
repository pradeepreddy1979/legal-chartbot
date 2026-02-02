import sqlite3
import datetime
import json
import bcrypt
from typing import List, Dict, Optional

DATABASE_FILE = "user_feedback.db"

# ==============================================================
# DATABASE CONNECTION & SETUP
# ==============================================================

def create_connection():
    """Create a database connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None


def create_tables(conn: sqlite3.Connection):
    """Create tables for users and feedback."""
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL
        );
    """)

    # Feedback / Chat History
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp DATETIME,
            query TEXT NOT NULL,
            response_snippet TEXT,
            retrieved_sections TEXT,
            rating INTEGER,
            FOREIGN KEY(username) REFERENCES users(username)
        );
    """)
    conn.commit()

# ==============================================================
# USER MANAGEMENT
# ==============================================================

def add_user(username: str, name: str, password: str) -> bool:
    """Register a new user with hashed password."""
    conn = create_connection()
    if conn is None:
        return False
    try:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users VALUES (?, ?, ?)", (username, name, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # username exists
    except sqlite3.Error as e:
        print(f"Error adding user: {e}")
        return False
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    """Verify user credentials."""
    conn = create_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username=?", (username,))
        row = cursor.fetchone()
        if not row:
            return False
        return bcrypt.checkpw(password.encode(), row[0].encode())
    except sqlite3.Error as e:
        print(f"Error verifying user: {e}")
        return False
    finally:
        conn.close()


def get_user(username: str) -> Optional[Dict]:
    """Retrieve user info."""
    conn = create_connection()
    if conn is None:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT username, name, password_hash FROM users WHERE username=?", (username,))
        row = cursor.fetchone()
        if row:
            return {"username": row[0], "name": row[1], "password_hash": row[2]}
        return None
    finally:
        conn.close()

# ==============================================================
# FEEDBACK LOGGING
# ==============================================================

def log_interaction(username: str, query: str, response: str, retrieved_sections: List[Dict], rating: Optional[int] = None):
    """Store user query & LLM response."""
    conn = create_connection()
    if conn is None:
        return
    try:
        cursor = conn.cursor()
        retrieved_json = json.dumps(retrieved_sections)
        cursor.execute("""
            INSERT INTO feedback (username, timestamp, query, response_snippet, retrieved_sections, rating)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            username,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query,
            response[:500],
            retrieved_json,
            rating
        ))
        conn.commit()
    finally:
        conn.close()


def delete_chat_history(username: str) -> bool:
    """Delete all chat history for a user."""
    conn = create_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM feedback WHERE username = ?", (username,))
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Error deleting chat history: {e}")
        return False
    finally:
        conn.close()


def delete_single_interaction(interaction_id: int) -> bool:
    """Delete a single chat interaction by ID."""
    conn = create_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM feedback WHERE id = ?", (interaction_id,))
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Error deleting interaction: {e}")
        return False
    finally:
        conn.close()


# ==============================================================
# GOOGLE OAUTH AUTHENTICATION
# ==============================================================

def add_or_update_google_user(email: str, name: str, google_id: str) -> bool:
    """Add or update a Google OAuth user."""
    conn = create_connection()
    if conn is None:
        return False
    try:
        hashed = bcrypt.hashpw(google_id.encode(), bcrypt.gensalt()).decode()
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT username FROM users WHERE username=?", (email,))
        if cursor.fetchone():
            # Update existing user
            cursor.execute(
                "UPDATE users SET name = ? WHERE username = ?",
                (name, email)
            )
        else:
            # Insert new user
            cursor.execute(
                "INSERT INTO users (username, name, password_hash) VALUES (?, ?, ?)",
                (email, name, hashed)
            )
        
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Error adding/updating Google user: {e}")
        return False
    finally:
        conn.close()


def get_user_by_email(email: str) -> Optional[Dict]:
    """Retrieve user info by email (for Google OAuth)."""
    conn = create_connection()
    if conn is None:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT username, name FROM users WHERE username=?", (email,))
        row = cursor.fetchone()
        if row:
            return {"username": row[0], "name": row[1]}
        return None
    finally:
        conn.close()


if __name__ == "__main__":
    conn = create_connection()
    if conn:
        create_tables(conn)
        print(f"✅ Database '{DATABASE_FILE}' initialized successfully.")
        conn.close()


