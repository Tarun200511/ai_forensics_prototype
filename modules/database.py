"""
database.py — SQLite Case Management Module
AI Edge Forensics Prototype
Handles creation, insertion, and retrieval of forensic case records.
"""

import sqlite3
import os
from datetime import datetime


# Path to the SQLite database file
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "forensics.db")


def get_connection():
    """
    Create and return a SQLite database connection.
    Creates the database file automatically if it does not exist.

    Returns:
        sqlite3.Connection: Active database connection object.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
    return conn


def init_db():
    """
    Initialize the database schema.
    Creates the `cases` table and the `analysis_log` table if they do not exist.
    Called once at application startup.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # --- Main Cases Table ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id         TEXT NOT NULL UNIQUE,
            date            TEXT NOT NULL,
            image_path      TEXT,
            weapon_detected TEXT DEFAULT 'Not Run',
            weapon_labels   TEXT DEFAULT '',
            blood_detected  TEXT DEFAULT 'Not Run',
            blood_pattern   TEXT DEFAULT '',
            footprint_match TEXT DEFAULT 'Not Run',
            footprint_score REAL DEFAULT 0.0,
            face_match      TEXT DEFAULT 'Not Run',
            face_name       TEXT DEFAULT '',
            notes           TEXT DEFAULT ''
        )
    """)

    # --- Analysis Event Log Table ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id     TEXT NOT NULL,
            module      TEXT NOT NULL,
            result      TEXT,
            timestamp   TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully.")


def create_case(case_id: str, image_path: str = "") -> bool:
    """
    Insert a new case record into the database.

    Args:
        case_id (str): Unique case identifier (e.g., 'CASE_20240226_001').
        image_path (str): Path to the primary evidence image.

    Returns:
        bool: True if insertion succeeded, False otherwise.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT OR IGNORE INTO cases (case_id, date, image_path)
            VALUES (?, ?, ?)
        """, (case_id, now, image_path))
        conn.commit()
        conn.close()
        print(f"[DB] Case created: {case_id}")
        return True
    except Exception as e:
        print(f"[DB ERROR] Failed to create case: {e}")
        return False


def update_case_results(case_id: str, results: dict) -> bool:
    """
    Update an existing case record with analysis results.

    Args:
        case_id (str): The case identifier to update.
        results (dict): Dictionary of field-value pairs to update.
                        Keys must match column names in the `cases` table.

    Returns:
        bool: True if update succeeded, False otherwise.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Dynamically build SET clause from results dict
        allowed_fields = {
            "weapon_detected", "weapon_labels", "blood_detected",
            "blood_pattern", "footprint_match", "footprint_score",
            "face_match", "face_name", "notes", "image_path"
        }
        filtered = {k: v for k, v in results.items() if k in allowed_fields}

        if not filtered:
            print("[DB] No valid fields to update.")
            return False

        set_clause = ", ".join([f"{k} = ?" for k in filtered.keys()])
        values = list(filtered.values()) + [case_id]

        cursor.execute(f"UPDATE cases SET {set_clause} WHERE case_id = ?", values)
        conn.commit()
        conn.close()
        print(f"[DB] Case {case_id} updated.")
        return True
    except Exception as e:
        print(f"[DB ERROR] Failed to update case: {e}")
        return False


def log_analysis_event(case_id: str, module: str, result: str):
    """
    Add an event entry to the analysis log.

    Args:
        case_id (str): Associated case ID.
        module (str): Module name that ran (e.g., 'weapon_detection').
        result (str): Result summary string.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO analysis_log (case_id, module, result, timestamp)
            VALUES (?, ?, ?, ?)
        """, (case_id, module, result, now))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] Failed to log event: {e}")


def get_all_cases() -> list:
    """
    Retrieve all cases from the database, ordered by most recent first.

    Returns:
        list[sqlite3.Row]: List of case records.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cases ORDER BY date DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"[DB ERROR] Failed to retrieve cases: {e}")
        return []


def get_case_by_id(case_id: str) -> dict:
    """
    Retrieve a single case record by its case_id.

    Args:
        case_id (str): The case identifier to look up.

    Returns:
        dict: Case record as a dictionary, or empty dict if not found.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else {}
    except Exception as e:
        print(f"[DB ERROR] Failed to get case: {e}")
        return {}


def get_case_log(case_id: str) -> list:
    """
    Retrieve all analysis log entries for a given case.

    Args:
        case_id (str): The case identifier.

    Returns:
        list[dict]: List of log entries.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM analysis_log WHERE case_id = ? ORDER BY timestamp DESC",
            (case_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"[DB ERROR] Failed to get log: {e}")
        return []


def delete_case(case_id: str) -> bool:
    """
    Delete a case and all its log entries from the database.

    Args:
        case_id (str): The case identifier to delete.

    Returns:
        bool: True if deletion succeeded.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))
        cursor.execute("DELETE FROM analysis_log WHERE case_id = ?", (case_id,))
        conn.commit()
        conn.close()
        print(f"[DB] Case {case_id} deleted.")
        return True
    except Exception as e:
        print(f"[DB ERROR] Failed to delete case: {e}")
        return False
