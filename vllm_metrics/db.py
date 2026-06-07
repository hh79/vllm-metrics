"""
Database layer for vLLM metrics storage.

Schema designed for multi-server, multi-model data with daily rollups
for efficient long-term queries across years.
"""

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


SCHEMA = """
-- Tracked vLLM server instances
CREATE TABLE IF NOT EXISTS servers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    url         TEXT NOT NULL,
    notes       TEXT DEFAULT '',
    added_at    REAL NOT NULL,
    last_seen   REAL
);

-- Models discovered on each server
CREATE TABLE IF NOT EXISTS models (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    server_id   INTEGER NOT NULL,
    model_name  TEXT NOT NULL,
    first_seen  REAL NOT NULL,
    last_seen   REAL,
    FOREIGN KEY (server_id) REFERENCES servers(id),
    UNIQUE(server_id, model_name)
);

-- Raw counter/gauge snapshots (one row per scrape, per model)
CREATE TABLE IF NOT EXISTS raw_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    server_id       INTEGER NOT NULL,
    model_id        INTEGER,
    timestamp       REAL NOT NULL,
    timestring      TEXT NOT NULL,

    -- Token counters
    prompt_tokens_total             REAL DEFAULT 0,
    generation_tokens_total         REAL DEFAULT 0,
    prompt_tokens_cached_total      REAL DEFAULT 0,
    request_success_total           REAL DEFAULT 0,
    request_prompt_tokens_total     REAL DEFAULT 0,
    request_generation_tokens_total REAL DEFAULT 0,
    num_preemptions_total           REAL DEFAULT 0,
    prefix_cache_queries_total      REAL DEFAULT 0,
    prefix_cache_hits_total         REAL DEFAULT 0,
    mm_cache_queries_total          REAL DEFAULT 0,
    mm_cache_hits_total             REAL DEFAULT 0,

    -- Gauges
    num_requests_running    REAL DEFAULT NULL,
    num_requests_waiting    REAL DEFAULT NULL,
    kv_cache_usage_perc     REAL DEFAULT NULL,

    -- Histogram sums/counts
    ttft_count      REAL DEFAULT NULL,
    ttft_sum        REAL DEFAULT NULL,
    itl_count       REAL DEFAULT NULL,
    itl_sum         REAL DEFAULT NULL,
    e2e_count       REAL DEFAULT NULL,
    e2e_sum         REAL DEFAULT NULL,
    queue_count     REAL DEFAULT NULL,
    queue_sum       REAL DEFAULT NULL,
    prefill_count   REAL DEFAULT NULL,
    prefill_sum     REAL DEFAULT NULL,
    decode_count    REAL DEFAULT NULL,
    decode_sum      REAL DEFAULT NULL,

    FOREIGN KEY (server_id) REFERENCES servers(id),
    FOREIGN KEY (model_id) REFERENCES models(id)
);

CREATE INDEX IF NOT EXISTS idx_raw_ts     ON raw_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_raw_server ON raw_snapshots(server_id);
CREATE INDEX IF NOT EXISTS idx_raw_model  ON raw_snapshots(model_id);

-- Last known counter values per (server, model) for computing deltas
-- model_id is stored as -1 for "no model" (unlabeled) to avoid NULL in UNIQUE
CREATE TABLE IF NOT EXISTS last_values (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    server_id   INTEGER NOT NULL,
    model_id    INTEGER NOT NULL DEFAULT -1,
    metric_key  TEXT NOT NULL,
    value       REAL NOT NULL,
    updated_at  REAL NOT NULL,
    UNIQUE(server_id, model_id, metric_key),
    FOREIGN KEY (server_id) REFERENCES servers(id)
);

-- Daily aggregated statistics (for fast year-scale queries)
CREATE TABLE IF NOT EXISTS daily_stats (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    date    TEXT NOT NULL,   -- 'YYYY-MM-DD'
    server_id INTEGER NOT NULL,
    model_id  INTEGER,

    -- Aggregated token counts for the day
    prompt_tokens               REAL DEFAULT 0,
    generation_tokens           REAL DEFAULT 0,
    prompt_tokens_cached        REAL DEFAULT 0,
    completed_requests          REAL DEFAULT 0,
    preemptions                 REAL DEFAULT 0,

    -- Min/max/avg gauges
    avg_running      REAL DEFAULT NULL,
    min_running      REAL DEFAULT NULL,
    max_running      REAL DEFAULT NULL,
    avg_waiting      REAL DEFAULT NULL,
    avg_kv_cache_pct REAL DEFAULT NULL,

    -- Histogram averages for the day
    avg_ttft_ms      REAL DEFAULT NULL,
    avg_itl_ms       REAL DEFAULT NULL,
    avg_e2e_s        REAL DEFAULT NULL,
    avg_queue_s      REAL DEFAULT NULL,
    avg_prefill_s    REAL DEFAULT NULL,
    avg_decode_s     REAL DEFAULT NULL,

    -- Sample count
    num_snapshots    INTEGER DEFAULT 0,

    UNIQUE(date, server_id, model_id),
    FOREIGN KEY (server_id) REFERENCES servers(id),
    FOREIGN KEY (model_id) REFERENCES models(id)
);

CREATE INDEX IF NOT EXISTS idx_daily_date     ON daily_stats(date);
CREATE INDEX IF NOT EXISTS idx_daily_server   ON daily_stats(server_id);
CREATE INDEX IF NOT EXISTS idx_daily_model    ON daily_stats(model_id);
"""


def get_db_path(path: str | None = None) -> str:
    if path:
        return os.path.expanduser(path)
    return os.path.expanduser('~/.vllm-metrics.db')


def connect(db_path: str) -> sqlite3.Connection:
    """Open or create the database with schema."""
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def upsert_server(conn: sqlite3.Connection, name: str, url: str, notes: str = '') -> int:
    now = time.time()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO servers (name, url, notes, added_at, last_seen)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            url = excluded.url,
            notes = COALESCE(NULLIF(excluded.notes, ''), servers.notes),
            last_seen = excluded.last_seen
    """, (name, url, notes, now, now))
    conn.commit()
    cursor.execute("SELECT id FROM servers WHERE name = ?", (name,))
    return cursor.fetchone()[0]


def get_servers(conn: sqlite3.Connection) -> list[dict]:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM servers ORDER BY name")
    return [dict(row) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def upsert_model(conn: sqlite3.Connection, server_id: int, model_name: str) -> int:
    now = time.time()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO models (server_id, model_name, first_seen, last_seen)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(server_id, model_name) DO UPDATE SET
            last_seen = excluded.last_seen
    """, (server_id, model_name, now, now))
    conn.commit()
    cursor.execute(
        "SELECT id FROM models WHERE server_id = ? AND model_name = ?",
        (server_id, model_name),
    )
    return cursor.fetchone()[0]


# ---------------------------------------------------------------------------
# Storing snapshots
# ---------------------------------------------------------------------------

def store_snapshot(
    conn: sqlite3.Connection,
    server_id: int,
    model_id: int | None,
    timestamp: float,
    stats: dict[str, float],
) -> int:
    """Store a raw snapshot row from extracted stats dict."""
    timestr = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    # Map flat keys to column names
    col_map = {
        'vllm:prompt_tokens_total': 'prompt_tokens_total',
        'vllm:generation_tokens_total': 'generation_tokens_total',
        'vllm:prompt_tokens_cached_total': 'prompt_tokens_cached_total',
        'vllm:request_success_total': 'request_success_total',
        'vllm:request_prompt_tokens_total': 'request_prompt_tokens_total',
        'vllm:request_generation_tokens_total': 'request_generation_tokens_total',
        'vllm:num_preemptions_total': 'num_preemptions_total',
        'vllm:prefix_cache_queries_total': 'prefix_cache_queries_total',
        'vllm:prefix_cache_hits_total': 'prefix_cache_hits_total',
        'vllm:mm_cache_queries_total': 'mm_cache_queries_total',
        'vllm:mm_cache_hits_total': 'mm_cache_hits_total',
        'vllm:num_requests_running': 'num_requests_running',
        'vllm:num_requests_waiting': 'num_requests_waiting',
        'vllm:kv_cache_usage_perc': 'kv_cache_usage_perc',
        'vllm:time_to_first_token_seconds_count': 'ttft_count',
        'vllm:time_to_first_token_seconds_sum': 'ttft_sum',
        'vllm:inter_token_latency_seconds_count': 'itl_count',
        'vllm:inter_token_latency_seconds_sum': 'itl_sum',
        'vllm:e2e_request_latency_seconds_count': 'e2e_count',
        'vllm:e2e_request_latency_seconds_sum': 'e2e_sum',
        'vllm:request_queue_time_seconds_count': 'queue_count',
        'vllm:request_queue_time_seconds_sum': 'queue_sum',
        'vllm:request_prefill_time_seconds_count': 'prefill_count',
        'vllm:request_prefill_time_seconds_sum': 'prefill_sum',
        'vllm:request_decode_time_seconds_count': 'decode_count',
        'vllm:request_decode_time_seconds_sum': 'decode_sum',
    }

    cols = ['server_id', 'model_id', 'timestamp', 'timestring']
    vals = [server_id, model_id, timestamp, timestr]
    placeholders = ['?', '?', '?', '?']

    for metric_key, db_col in col_map.items():
        if metric_key in stats:
            cols.append(db_col)
            vals.append(stats[metric_key])
            placeholders.append('?')

    cursor = conn.cursor()
    sql = f"INSERT INTO raw_snapshots ({', '.join(cols)}) VALUES ({', '.join(placeholders)})"
    cursor.execute(sql, vals)
    conn.commit()
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# Last value tracking (for delta computation across restarts)
# ---------------------------------------------------------------------------

def get_last_values(conn: sqlite3.Connection, server_id: int, model_id: int | None) -> dict[str, float]:
    cursor = conn.cursor()
    mid = model_id if model_id is not None else -1
    cursor.execute(
        "SELECT metric_key, value FROM last_values WHERE server_id = ? AND model_id = ?",
        (server_id, mid),
    )
    return {row['metric_key']: row['value'] for row in cursor.fetchall()}


def save_last_values(
    conn: sqlite3.Connection,
    server_id: int,
    model_id: int | None,
    stats: dict[str, float],
):
    now = time.time()
    cursor = conn.cursor()
    mid = model_id if model_id is not None else -1
    for key, value in stats.items():
        cursor.execute("""
            INSERT INTO last_values (server_id, model_id, metric_key, value, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(server_id, model_id, metric_key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
        """, (server_id, mid, key, value, now))
    conn.commit()


# ---------------------------------------------------------------------------
# Daily rollup
# ---------------------------------------------------------------------------

def compute_daily_rollup(conn: sqlite3.Connection, date_str: str):
    """Aggregate raw snapshots for a given date into daily_stats."""
    cursor = conn.cursor()

    # Get all raw snapshots for this date, grouped by server + model
    cursor.execute("""
        SELECT
            r.server_id,
            r.model_id,
            COUNT(*)                                               AS num_snapshots,
            SUM(r.prompt_tokens_total)                             AS prompt_tokens,
            SUM(r.generation_tokens_total)                         AS generation_tokens,
            SUM(r.prompt_tokens_cached_total)                      AS prompt_tokens_cached,
            SUM(r.request_success_total)                           AS completed_requests,
            SUM(r.num_preemptions_total)                           AS preemptions,
            AVG(r.num_requests_running)    AS avg_running,
            MIN(r.num_requests_running)    AS min_running,
            MAX(r.num_requests_running)    AS max_running,
            AVG(r.num_requests_waiting)    AS avg_waiting,
            AVG(r.kv_cache_usage_perc)     AS avg_kv_cache,
            AVG(r.ttft_sum / NULLIF(r.ttft_count, 0)) * 1000  AS avg_ttft_ms,
            AVG(r.itl_sum / NULLIF(r.itl_count, 0)) * 1000    AS avg_itl_ms,
            AVG(r.e2e_sum / NULLIF(r.e2e_count, 0))           AS avg_e2e_s,
            AVG(r.queue_sum / NULLIF(r.queue_count, 0))       AS avg_queue_s,
            AVG(r.prefill_sum / NULLIF(r.prefill_count, 0))   AS avg_prefill_s,
            AVG(r.decode_sum / NULLIF(r.decode_count, 0))     AS avg_decode_s
        FROM raw_snapshots r
        WHERE DATE(r.timestring) = ?
        GROUP BY r.server_id, r.model_id
    """, (date_str,))

    rows = cursor.fetchall()
    if not rows:
        return 0

    inserted = 0
    for row in rows:
        cursor.execute("""
            INSERT INTO daily_stats (
                date, server_id, model_id,
                prompt_tokens, generation_tokens, prompt_tokens_cached,
                completed_requests, preemptions,
                avg_running, min_running, max_running,
                avg_waiting, avg_kv_cache_pct,
                avg_ttft_ms, avg_itl_ms, avg_e2e_s,
                avg_queue_s, avg_prefill_s, avg_decode_s,
                num_snapshots
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, server_id, model_id) DO UPDATE SET
                prompt_tokens        = excluded.prompt_tokens,
                generation_tokens    = excluded.generation_tokens,
                prompt_tokens_cached = excluded.prompt_tokens_cached,
                completed_requests   = excluded.completed_requests,
                preemptions          = excluded.preemptions,
                avg_running   = excluded.avg_running,
                min_running   = excluded.min_running,
                max_running   = excluded.max_running,
                avg_waiting   = excluded.avg_waiting,
                avg_kv_cache_pct = excluded.avg_kv_cache_pct,
                avg_ttft_ms   = excluded.avg_ttft_ms,
                avg_itl_ms    = excluded.avg_itl_ms,
                avg_e2e_s     = excluded.avg_e2e_s,
                avg_queue_s   = excluded.avg_queue_s,
                avg_prefill_s = excluded.avg_prefill_s,
                avg_decode_s  = excluded.avg_decode_s,
                num_snapshots = excluded.num_snapshots
        """, (
            date_str,
            row['server_id'],
            row['model_id'],
            row['prompt_tokens'] or 0,
            row['generation_tokens'] or 0,
            row['prompt_tokens_cached'] or 0,
            row['completed_requests'] or 0,
            row['preemptions'] or 0,
            row['avg_running'],
            row['min_running'],
            row['max_running'],
            row['avg_waiting'],
            row['avg_kv_cache'],
            row['avg_ttft_ms'],
            row['avg_itl_ms'],
            row['avg_e2e_s'],
            row['avg_queue_s'],
            row['avg_prefill_s'],
            row['avg_decode_s'],
            row['num_snapshots'],
        ))
        inserted += 1

    conn.commit()
    return inserted


def prune_raw_snapshots(conn: sqlite3.Connection, retention_days: int):
    """Delete raw snapshots older than retention_days."""
    if retention_days <= 0:
        return 0
    cutoff = time.time() - (retention_days * 86400)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM raw_snapshots WHERE timestamp < ?", (cutoff,))
    deleted = cursor.rowcount
    conn.commit()
    return deleted


def rollup_and_prune(conn: sqlite3.Connection, retention_days: int = 90):
    """Compute daily rollups for all dates that have raw data but no rollup,
    then prune old raw data."""
    cursor = conn.cursor()

    # Find dates with raw data that may not have daily rollup yet
    # We process yesterday and all prior dates that don't have rollups
    yesterday = datetime.now(timezone.utc).date().isoformat()

    cursor.execute("""
        SELECT DISTINCT DATE(r.timestring) AS d
        FROM raw_snapshots r
        LEFT JOIN daily_stats ds ON DATE(r.timestring) = ds.date
            AND r.server_id = ds.server_id
            AND (r.model_id = ds.model_id OR (r.model_id IS NULL AND ds.model_id IS NULL))
        WHERE ds.id IS NULL
        ORDER BY d
    """)

    dates = [row[0] for row in cursor.fetchall()]

    rolled = 0
    for d in dates:
        if d >= yesterday:
            continue  # don't rollup today - still accumulating
        rolled += compute_daily_rollup(conn, d)

    pruned = prune_raw_snapshots(conn, retention_days)

    return rolled, pruned
