"""
Direct database helpers for storing steps and sessions.

Uses aiosqlite locally and asyncpg on Heroku (detected via DATABASE_URL).
Bypasses Tortoise ORM to avoid context propagation issues with NiceGUI.
"""

import asyncio
import json
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "")
_USE_POSTGRES = DATABASE_URL.startswith("postgres")

# ---------------------------------------------------------------------------
# SQLite helpers (local)
# ---------------------------------------------------------------------------

if not _USE_POSTGRES:
    import aiosqlite

    _DB_PATH = os.path.join(os.path.dirname(__file__), "data", "db.sqlite")
    _lock = asyncio.Lock()

    async def _execute(sql: str, params: tuple = ()):
        async with _lock:
            async with aiosqlite.connect(_DB_PATH) as conn:
                await conn.execute(sql, params)
                await conn.commit()

    async def _fetchall(sql: str, params: tuple = ()) -> list[dict]:
        async with aiosqlite.connect(_DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]

# ---------------------------------------------------------------------------
# Postgres helpers (Heroku)
# ---------------------------------------------------------------------------

else:
    import asyncpg

    _pg_url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    _pool: asyncpg.Pool | None = None

    async def _get_pool() -> asyncpg.Pool:
        global _pool
        if _pool is None:
            _pool = await asyncpg.create_pool(_pg_url, min_size=1, max_size=5)
        return _pool

    async def _execute(sql: str, params: tuple = ()):
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute(sql, *params)

    async def _fetchall(sql: str, params: tuple = ()) -> list[dict]:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [dict(r) for r in rows]

# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

_CREATE_STEPS = """
CREATE TABLE IF NOT EXISTS steps (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    seed         TEXT    NOT NULL,
    timestamp    TEXT    NOT NULL,
    maze_idx     INTEGER NOT NULL,
    maze_step    INTEGER NOT NULL,
    action       TEXT    NOT NULL,
    prev_pos     TEXT    NOT NULL,
    new_pos      TEXT    NOT NULL,
    moved        INTEGER NOT NULL,
    visited_goal INTEGER,
    left_goal    INTEGER,
    reward       INTEGER NOT NULL
)
"""

_CREATE_STEPS_PG = """
CREATE TABLE IF NOT EXISTS steps (
    id           SERIAL  PRIMARY KEY,
    seed         TEXT    NOT NULL,
    timestamp    TEXT    NOT NULL,
    maze_idx     INTEGER NOT NULL,
    maze_step    INTEGER NOT NULL,
    action       TEXT    NOT NULL,
    prev_pos     TEXT    NOT NULL,
    new_pos      TEXT    NOT NULL,
    moved        BOOLEAN NOT NULL,
    visited_goal INTEGER,
    left_goal    INTEGER,
    reward       INTEGER NOT NULL
)
"""

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    seed         TEXT    NOT NULL UNIQUE,
    completed_at TEXT    NOT NULL,
    mode         TEXT    NOT NULL,
    dims         TEXT    NOT NULL,
    n_kinds      TEXT    NOT NULL,
    n_goals      INTEGER NOT NULL,
    rule_dim     TEXT    NOT NULL,
    rule_value   TEXT    NOT NULL,
    total_score  INTEGER NOT NULL,
    total_steps  INTEGER NOT NULL,
    log          TEXT    NOT NULL,
    mazes        TEXT    NOT NULL,
    prolific_pid TEXT,
    study_id     TEXT,
    session_id   TEXT
)
"""

_CREATE_SESSIONS_PG = """
CREATE TABLE IF NOT EXISTS sessions (
    id           SERIAL  PRIMARY KEY,
    seed         TEXT    NOT NULL UNIQUE,
    completed_at TEXT    NOT NULL,
    mode         TEXT    NOT NULL,
    dims         TEXT    NOT NULL,
    n_kinds      TEXT    NOT NULL,
    n_goals      INTEGER NOT NULL,
    rule_dim     TEXT    NOT NULL,
    rule_value   TEXT    NOT NULL,
    total_score  INTEGER NOT NULL,
    total_steps  INTEGER NOT NULL,
    log          TEXT    NOT NULL,
    mazes        TEXT    NOT NULL,
    prolific_pid TEXT,
    study_id     TEXT,
    session_id   TEXT
)
"""

_MIGRATE_SESSIONS_PG = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='sessions' AND column_name='prolific_pid') THEN
        ALTER TABLE sessions ADD COLUMN prolific_pid TEXT;
        ALTER TABLE sessions ADD COLUMN study_id     TEXT;
        ALTER TABLE sessions ADD COLUMN session_id   TEXT;
    END IF;
END$$;
"""

_MIGRATE_SESSIONS_SQLITE = [
    "ALTER TABLE sessions ADD COLUMN prolific_pid TEXT",
    "ALTER TABLE sessions ADD COLUMN study_id     TEXT",
    "ALTER TABLE sessions ADD COLUMN session_id   TEXT",
]


async def init_tables():
    """Create tables if they don't exist. Called once at startup."""
    if _USE_POSTGRES:
        await _execute(_CREATE_STEPS_PG)
        await _execute(_CREATE_SESSIONS_PG)
        await _execute(_MIGRATE_SESSIONS_PG)
    else:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        await _execute(_CREATE_STEPS)
        await _execute(_CREATE_SESSIONS)
        # Add new columns to existing SQLite DBs (fails silently if already present)
        for sql in _MIGRATE_SESSIONS_SQLITE:
            try:
                await _execute(sql)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def insert_step(
    seed: str,
    timestamp: str,
    maze_idx: int,
    maze_step: int,
    action: str,
    prev_pos: str,
    new_pos: str,
    moved: bool,
    visited_goal,
    left_goal,
    reward: int,
):
    await _execute(
        """
        INSERT INTO steps
            (seed, timestamp, maze_idx, maze_step, action,
             prev_pos, new_pos, moved, visited_goal, left_goal, reward)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """ if not _USE_POSTGRES else
        """
        INSERT INTO steps
            (seed, timestamp, maze_idx, maze_step, action,
             prev_pos, new_pos, moved, visited_goal, left_goal, reward)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """,
        (str(seed), timestamp, maze_idx, maze_step, action,
         prev_pos, new_pos, moved, visited_goal, left_goal, reward),
    )


async def upsert_session(
    seed: str,
    completed_at: str,
    mode: str,
    dims: list,
    n_kinds: dict,
    n_goals: int,
    rule_dim: str,
    rule_value: str,
    total_score: int,
    total_steps: int,
    log: list,
    mazes: list,
    prolific_pid: str | None = None,
    study_id: str | None = None,
    session_id: str | None = None,
):
    sql = (
        """
        INSERT INTO sessions
            (seed, completed_at, mode, dims, n_kinds, n_goals,
             rule_dim, rule_value, total_score, total_steps, log, mazes,
             prolific_pid, study_id, session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(seed) DO UPDATE SET
            completed_at=excluded.completed_at,
            mode=excluded.mode, dims=excluded.dims,
            n_kinds=excluded.n_kinds, n_goals=excluded.n_goals,
            rule_dim=excluded.rule_dim, rule_value=excluded.rule_value,
            total_score=excluded.total_score, total_steps=excluded.total_steps,
            log=excluded.log, mazes=excluded.mazes,
            prolific_pid=excluded.prolific_pid,
            study_id=excluded.study_id, session_id=excluded.session_id
        """
        if not _USE_POSTGRES else
        """
        INSERT INTO sessions
            (seed, completed_at, mode, dims, n_kinds, n_goals,
             rule_dim, rule_value, total_score, total_steps, log, mazes,
             prolific_pid, study_id, session_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        ON CONFLICT(seed) DO UPDATE SET
            completed_at=EXCLUDED.completed_at,
            mode=EXCLUDED.mode, dims=EXCLUDED.dims,
            n_kinds=EXCLUDED.n_kinds, n_goals=EXCLUDED.n_goals,
            rule_dim=EXCLUDED.rule_dim, rule_value=EXCLUDED.rule_value,
            total_score=EXCLUDED.total_score, total_steps=EXCLUDED.total_steps,
            log=EXCLUDED.log, mazes=EXCLUDED.mazes,
            prolific_pid=EXCLUDED.prolific_pid,
            study_id=EXCLUDED.study_id, session_id=EXCLUDED.session_id
        """
    )
    await _execute(sql, (
        str(seed), completed_at, mode,
        json.dumps(dims), json.dumps(n_kinds), n_goals,
        rule_dim, rule_value, total_score, total_steps,
        json.dumps(log), json.dumps(mazes),
        prolific_pid, study_id, session_id,
    ))


async def fetch_sessions(seed: str | None = None) -> list[dict]:
    if seed:
        rows = await _fetchall("SELECT * FROM sessions WHERE seed = ?", (seed,))
    else:
        rows = await _fetchall("SELECT * FROM sessions ORDER BY completed_at")
    for r in rows:
        r["dims"]   = json.loads(r["dims"])
        r["n_kinds"] = json.loads(r["n_kinds"])
        r["log"]    = json.loads(r["log"])
        r["mazes"]  = json.loads(r["mazes"])
    return rows


async def fetch_steps(seed: str | None = None) -> list[dict]:
    if seed:
        return await _fetchall(
            "SELECT * FROM steps WHERE seed = ? ORDER BY maze_idx, maze_step", (seed,)
        )
    return await _fetchall("SELECT * FROM steps ORDER BY seed, maze_idx, maze_step")
