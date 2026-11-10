import asyncio
import aiosqlite
import logging
import os
import sqlite3
import time
import weakref
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("database")
DB_PATH = os.path.abspath("./app.db")

# Per-event-loop persistent SQLite connection + mutex lock to prevent per-turn connect overhead
# while remaining safe across concurrent asyncio tasks and separate test event loops.
_loop_connections: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Tuple[aiosqlite.Connection, asyncio.Lock]]" = (
    weakref.WeakKeyDictionary()
)
_loop_init_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)


def init_db_sync():
    """Synchronously initializes local user-space SQLite tables and enables WAL mode."""
    with sqlite3.connect(DB_PATH) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("PRAGMA busy_timeout=5000")
        db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                title TEXT
            )
        """)
        db.execute("""
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
        db.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                chunk_count INTEGER,
                status TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.commit()


# Ensure schema exists immediately on module load
init_db_sync()


async def init_db():
    """Initializes local user-space SQLite tables and warms the persistent async connection."""
    init_db_sync()
    await get_db_connection()


async def get_db_connection() -> Tuple[aiosqlite.Connection, asyncio.Lock]:
    """Returns a persistent WAL-mode aiosqlite connection and lock bound to the active loop."""
    loop = asyncio.get_running_loop()
    entry = _loop_connections.get(loop)
    if entry is not None:
        conn, lock = entry
        if getattr(conn, "_running", False) and getattr(conn, "_connection", None) is not None:
            return conn, lock

    init_lock = _loop_init_locks.get(loop)
    if init_lock is None:
        init_lock = asyncio.Lock()
        _loop_init_locks[loop] = init_lock

    async with init_lock:
        entry = _loop_connections.get(loop)
        if entry is not None:
            conn, lock = entry
            if getattr(conn, "_running", False) and getattr(conn, "_connection", None) is not None:
                return conn, lock

        t0 = time.perf_counter()
        conn = await aiosqlite.connect(DB_PATH)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        await conn.commit()
        lock = asyncio.Lock()
        _loop_connections[loop] = (conn, lock)
        logger.info(
            f"[PERF][db_init] Persistent WAL aiosqlite connection opened in {(time.perf_counter() - t0)*1000:.2f}ms"
        )
        return conn, lock


async def close_db_connection() -> None:
    """Closes the persistent aiosqlite connection for the current event loop, if open."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    entry = _loop_connections.pop(loop, None)
    if entry is not None:
        conn, _ = entry
        try:
            await conn.close()
        except Exception:
            pass


async def save_chat_message(session_id: str, sender: str, content: str, citations_json: str = "[]"):
    t0 = time.perf_counter()
    db, lock = await get_db_connection()
    async with lock:
        await db.execute(
            "INSERT OR IGNORE INTO sessions (session_id, title) VALUES (?, ?)",
            (session_id, f"Session {session_id[:8]}"),
        )
        await db.execute(
            "INSERT INTO chat_history (session_id, sender, content, citations_json) VALUES (?, ?, ?, ?)",
            (session_id, sender, content, citations_json),
        )
        await db.commit()
    logger.info(f"[PERF][db_write] save_chat_message(sender={sender}) took {(time.perf_counter() - t0)*1000:.2f}ms")


async def get_chat_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    t0 = time.perf_counter()
    db, lock = await get_db_connection()
    async with lock:
        async with db.execute(
            "SELECT sender, content, citations_json, timestamp FROM chat_history WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (session_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
    logger.info(f"[PERF][db_read] get_chat_history(rows={len(rows)}) took {(time.perf_counter() - t0)*1000:.2f}ms")
    return [dict(row) for row in rows]


async def record_document(filename: str, chunk_count: int, status: str = "indexed"):
    db, lock = await get_db_connection()
    async with lock:
        await db.execute(
            "INSERT OR REPLACE INTO uploaded_documents (filename, chunk_count, status) VALUES (?, ?, ?)",
            (filename, chunk_count, status),
        )
        await db.commit()


async def delete_document(filename: str):
    db, lock = await get_db_connection()
    async with lock:
        await db.execute("DELETE FROM uploaded_documents WHERE filename = ?", (filename,))
        await db.commit()


async def list_documents() -> List[Dict[str, Any]]:
    db, lock = await get_db_connection()
    async with lock:
        async with db.execute(
            "SELECT filename, chunk_count, status, uploaded_at FROM uploaded_documents ORDER BY id DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def list_sessions() -> List[Dict[str, Any]]:
    db, lock = await get_db_connection()
    async with lock:
        async with db.execute(
            "SELECT session_id, created_at, title FROM sessions ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
