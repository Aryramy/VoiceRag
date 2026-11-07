import aiosqlite
import os
from typing import List, Dict, Any

DB_PATH = os.path.abspath("./app.db")

async def init_db():
    """Initializes local user-space SQLite tables for sessions and documents."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                title TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                sender TEXT,
                content TEXT,
                citations_json TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                chunk_count INTEGER,
                status TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()

async def save_chat_message(session_id: str, sender: str, content: str, citations_json: str = "[]"):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO sessions (session_id, title) VALUES (?, ?)",
            (session_id, f"Session {session_id[:8]}")
        )
        await db.execute(
            "INSERT INTO chat_history (session_id, sender, content, citations_json) VALUES (?, ?, ?, ?)",
            (session_id, sender, content, citations_json)
        )
        await db.commit()

async def get_chat_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT sender, content, citations_json, timestamp FROM chat_history WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (session_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def record_document(filename: str, chunk_count: int, status: str = "indexed"):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO uploaded_documents (filename, chunk_count, status) VALUES (?, ?, ?)",
            (filename, chunk_count, status)
        )
        await db.commit()

async def list_documents() -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT filename, chunk_count, status, uploaded_at FROM uploaded_documents ORDER BY id DESC") as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def list_sessions() -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT session_id, created_at, title FROM sessions ORDER BY created_at DESC") as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

