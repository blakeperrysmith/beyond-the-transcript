"""Latency log. Stores timings only: no audio, no IP address, no user id.

SQLite for local runs. Set DATABASE_URL to a Postgres URL to keep the counts across
restarts on a host with no persistent disk. Supabase works: use its *transaction
pooler* connection string (port 6543). The table is created in a private schema
(btt) that Supabase's public REST API does not expose, with row-level security on
as a second lock. If the database cannot be reached the app falls back to local
storage and says so; it never fails an analysis because of the log.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import Boolean, Column, Float, Integer, MetaData, String, Table, create_engine, func, select, text, update
from sqlalchemy.pool import StaticPool

log = logging.getLogger("btt.store")
SCHEMA = "btt"  # Postgres only; Supabase does not expose custom schemas over its REST API

_MAX_ROWS_FOR_PERCENTILES = 100_000


def _normalise_url(url: str) -> str:
    # Hosts often hand out postgres:// or postgresql:// URLs; SQLAlchemy needs a driver name.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def _summ(values: list[float]) -> dict | None:
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }


class LatencyStore:
    def __init__(self, url: str = "sqlite:///latency.db"):
        url = _normalise_url(url)
        self.kind = "postgres" if url.startswith("postgresql") else "sqlite"
        kwargs: dict = {"pool_pre_ping": True}
        if self.kind == "postgres":
            kwargs["connect_args"] = {"connect_timeout": 8}
            if "pooler" in url or ":6543" in url:  # pgbouncer, transaction mode: no server-side prepared statements
                kwargs["connect_args"]["prepare_threshold"] = None
        elif ":memory:" in url:
            kwargs.update(poolclass=StaticPool, connect_args={"check_same_thread": False})
        self.engine = create_engine(url, **kwargs)
        self.persistent = self.kind == "postgres"
        self.note = ""
        schema = SCHEMA if self.kind == "postgres" else None
        meta = MetaData(schema=schema)
        self.runs = Table(
            "runs",
            meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("ts", Float, nullable=False),
            Column("source", String(16), nullable=False),
            Column("model_version", String(32), nullable=False),
            Column("cold", Boolean, nullable=False),
            Column("features_ms", Float, nullable=False),
            Column("prosody_ms", Float, nullable=False),
            Column("tract_ms", Float, nullable=False),
            Column("server_ms", Float, nullable=False),
            Column("e2e_ms", Float, nullable=True),
        )
        if schema:
            with self.engine.begin() as conn:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        meta.create_all(self.engine)
        if schema:
            try:  # second lock: no policies means the REST roles see nothing
                with self.engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {schema}.runs ENABLE ROW LEVEL SECURITY"))
            except Exception as e:  # noqa: BLE001
                log.warning("could not enable row-level security: %s", e)
        self._lock = threading.Lock()
        self._cache: tuple[float, dict] | None = None
        self._first_ts: float | None = None

    def add(self, *, source: str, model_version: str, cold: bool, features_ms: float,
            prosody_ms: float, tract_ms: float, server_ms: float) -> int:
        with self.engine.begin() as conn:
            res = conn.execute(
                self.runs.insert().values(
                    ts=time.time(), source=source, model_version=model_version, cold=cold,
                    features_ms=features_ms, prosody_ms=prosody_ms, tract_ms=tract_ms,
                    server_ms=server_ms, e2e_ms=None,
                )
            )
            self._cache = None
            return int(res.inserted_primary_key[0])

    def set_e2e(self, run_id: int, e2e_ms: float) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(self.runs).where(self.runs.c.id == run_id).values(e2e_ms=float(e2e_ms)))
        self._cache = None

    def summary(self, ttl: float = 2.0) -> dict:
        with self._lock:
            if self._cache and time.time() - self._cache[0] < ttl:
                return self._cache[1]
            with self.engine.connect() as conn:
                total = conn.execute(select(func.count()).select_from(self.runs)).scalar_one()
                first = conn.execute(select(func.min(self.runs.c.ts))).scalar_one()
                rows = conn.execute(
                    select(self.runs.c.cold, self.runs.c.tract_ms, self.runs.c.server_ms, self.runs.c.e2e_ms)
                    .order_by(self.runs.c.id.desc())
                    .limit(_MAX_ROWS_FOR_PERCENTILES)
                ).all()
            warm = [r for r in rows if not r[0]]
            out = {
                "n": int(total),
                "n_warm": len(warm),
                "n_cold": len(rows) - len(warm),
                "since": datetime.fromtimestamp(first, timezone.utc).isoformat(timespec="seconds") if first else None,
                "window_rows": len(rows),
                "tract": _summ([r[1] for r in warm]),
                "server": _summ([r[2] for r in warm]),
                "round_trip": _summ([r[3] for r in warm if r[3] is not None]),
            }
            self._cache = (time.time(), out)
            return out


def open_store(url: str, fallback_path: str) -> LatencyStore:
    """Open the configured database; fall back to local disk, then memory, never raise."""
    reason = ""
    try:
        return LatencyStore(url)
    except Exception as e:  # noqa: BLE001
        log.warning("database unavailable (%s); falling back to local storage", type(e).__name__)
        reason = "The configured database could not be reached, so counts are kept on this server and reset when it restarts."
    for fb in (f"sqlite:///{fallback_path}", "sqlite:///:memory:"):
        try:
            st = LatencyStore(fb)
            st.note = reason
            return st
        except Exception as e:  # noqa: BLE001
            log.warning("fallback %s failed: %s", fb, type(e).__name__)
    raise RuntimeError("no usable storage")
