"""SQLite persistence for benchmark runs.

Kept intentionally simple (stdlib sqlite3, JSON blob per run). Swap for
PostgreSQL by replacing these four functions; the API layer only speaks
BenchmarkResult objects.
"""
import json
import os
import sqlite3
from typing import List, Optional

from . import config
from .schemas import BenchmarkResult


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(config.DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            inference_mode TEXT NOT NULL,
            num_prompts INTEGER NOT NULL,
            results_json TEXT NOT NULL
        )
        """
    )
    return conn


def save_benchmark(result: BenchmarkResult) -> int:
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO benchmark_runs (created_at, inference_mode, num_prompts, results_json) "
            "VALUES (?, ?, ?, ?)",
            (
                result.created_at,
                result.inference_mode,
                result.num_prompts,
                json.dumps([m.model_dump() for m in result.results]),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def load_benchmarks(limit: int = 20) -> List[BenchmarkResult]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, created_at, inference_mode, num_prompts, results_json "
            "FROM benchmark_runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [
        BenchmarkResult(
            id=row[0],
            created_at=row[1],
            inference_mode=row[2],
            num_prompts=row[3],
            results=json.loads(row[4]),
        )
        for row in rows
    ]


def load_latest_benchmark() -> Optional[BenchmarkResult]:
    runs = load_benchmarks(limit=1)
    return runs[0] if runs else None
