# memory.py
import json
import sqlite3
from typing import Any, Dict, Optional

DB_PATH = "planner.db"

def _connect():
    return sqlite3.connect(DB_PATH)

def _init_db():
    con = _connect()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prefs (
            user_id TEXT PRIMARY KEY,
            data TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()

def save_prefs(user_id: str, prefs: Dict[str, Any]) -> None:
    _init_db()
    con = _connect()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO prefs (user_id, data) VALUES (?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET data=excluded.data",
        (user_id, json.dumps(prefs))
    )
    con.commit()
    con.close()

def load_prefs(user_id: str) -> Optional[Dict[str, Any]]:
    _init_db()
    con = _connect()
    cur = con.cursor()
    cur.execute("SELECT data FROM prefs WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None
def init_db():
    """
    Public DB initializer (used by UI / app startup).
    """
    _init_db()
