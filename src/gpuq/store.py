"""Persistent job state for gpuq."""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, cast

ACTIVE_STATES = ("allocated", "running", "quarantined")
TERMINAL_STATES = ("succeeded", "failed", "cancelled", "lost")


class JobNotFoundError(LookupError):
    pass


class JobPermissionError(PermissionError):
    pass


class JobStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._initialize()
        database_path.chmod(0o600)

    @contextmanager
    def _connect(self, write: bool = True) -> Iterator[sqlite3.Connection]:
        """Serialize state transitions from their first read, and always close.

        SQLite's connection context manager commits but does not close, and a
        deferred transaction does not protect a SELECT followed by an UPDATE.
        """
        connection = sqlite3.connect(str(self.database_path), timeout=10)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 10000")
            if write:
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    yield connection
            else:
                with connection:
                    yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        init_conn = sqlite3.connect(str(self.database_path), timeout=10)
        try:
            init_conn.execute("PRAGMA journal_mode = WAL")
        finally:
            init_conn.close()
        with self._connect(write=True) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uid INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    token TEXT NOT NULL,
                    name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    gpu_count INTEGER NOT NULL,
                    gpu_ids TEXT,
                    command_json TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    log_path TEXT,
                    supervisor_pid INTEGER,
                    process_pid INTEGER,
                    return_code INTEGER,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    heartbeat_at REAL,
                    started_at REAL,
                    finished_at REAL,
                    message TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_state_id ON jobs(state, id);
                CREATE INDEX IF NOT EXISTS jobs_uid_id ON jobs(uid, id);
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        result["command"] = json.loads(result.pop("command_json"))
        gpu_ids = result.get("gpu_ids")
        result["gpu_ids"] = json.loads(gpu_ids) if gpu_ids else []
        result["cancel_requested"] = bool(result["cancel_requested"])
        result.pop("token", None)
        return result

    def create_job(
        self,
        *,
        uid: int,
        username: str,
        name: str,
        gpu_count: int,
        command: Sequence[str],
        cwd: str,
        log_path: Optional[str],
    ) -> Tuple[int, str]:
        if gpu_count < 1:
            raise ValueError("gpu_count must be at least 1")
        if not command:
            raise ValueError("command must not be empty")
        now = time.time()
        token = secrets.token_urlsafe(32)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO jobs (
                    uid, username, token, name, state, gpu_count, command_json,
                    cwd, log_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
                """,
                (
                    uid,
                    username,
                    token,
                    name,
                    gpu_count,
                    json.dumps(list(command)),
                    cwd,
                    log_path,
                    now,
                    now,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a job id")
            return int(cursor.lastrowid), token

    def _authorized_row(
        self,
        connection: sqlite3.Connection,
        job_id: int,
        uid: int,
        *,
        token: Optional[str] = None,
        admin_uid: Optional[int] = None,
    ) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFoundError(f"job {job_id} does not exist")
        if uid != row["uid"] and uid != admin_uid:
            raise JobPermissionError(f"job {job_id} belongs to another user")
        if token is not None and not secrets.compare_digest(str(row["token"]), token):
            raise JobPermissionError("invalid job token")
        return cast(sqlite3.Row, row)

    def get_job(self, job_id: int, uid: int, *, admin_uid: Optional[int] = None) -> Dict[str, Any]:
        with self._connect(write=False) as connection:
            return self._decode(self._authorized_row(connection, job_id, uid, admin_uid=admin_uid))

    def list_jobs(self, uid: int, *, include_all: bool, admin_uid: int) -> List[Dict[str, Any]]:
        with self._connect(write=False) as connection:
            if include_all:
                rows = connection.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 200").fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM jobs WHERE uid = ? ORDER BY id DESC LIMIT 200", (uid,)
                ).fetchall()
        return [self._decode(row) for row in rows]

    def register_supervisor(self, job_id: int, uid: int, token: str, supervisor_pid: int) -> Dict[str, Any]:
        now = time.time()
        with self._connect() as connection:
            row = self._authorized_row(connection, job_id, uid, token=token)
            if row["state"] in TERMINAL_STATES:
                return self._decode(row)
            if row["supervisor_pid"] is not None:
                raise ValueError(f"job {job_id} already has a supervisor")
            if row["state"] == "quarantined":
                raise ValueError(f"job {job_id} is quarantined")
            if supervisor_pid <= 0:
                raise ValueError("supervisor_pid must be positive")
            connection.execute(
                """
                UPDATE jobs SET supervisor_pid = ?, heartbeat_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (supervisor_pid, now, now, job_id),
            )
            updated = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            assert updated is not None
            return self._decode(updated)

    def heartbeat(self, job_id: int, uid: int, token: str) -> Dict[str, Any]:
        now = time.time()
        with self._connect() as connection:
            row = self._authorized_row(connection, job_id, uid, token=token)
            if row["state"] in ("pending", "allocated", "running"):
                connection.execute(
                    "UPDATE jobs SET heartbeat_at = ?, updated_at = ? WHERE id = ?",
                    (now, now, job_id),
                )
                row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                assert row is not None
            return self._decode(row)

    def mark_started(self, job_id: int, uid: int, token: str, process_pid: int) -> Dict[str, Any]:
        now = time.time()
        with self._connect() as connection:
            row = self._authorized_row(connection, job_id, uid, token=token)
            if row["state"] != "allocated":
                raise ValueError(f"job {job_id} cannot start from state {row['state']}")
            if row["cancel_requested"]:
                raise ValueError(f"job {job_id} has been cancelled")
            if row["supervisor_pid"] is None:
                raise ValueError(f"job {job_id} has no registered supervisor")
            if process_pid <= 0:
                raise ValueError("process_pid must be positive")
            connection.execute(
                """
                UPDATE jobs
                SET state = 'running', process_pid = ?, started_at = ?,
                    heartbeat_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (process_pid, now, now, now, job_id),
            )
            updated = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            assert updated is not None
            return self._decode(updated)

    def finish(self, job_id: int, uid: int, token: str, return_code: int, message: str = "") -> Dict[str, Any]:
        now = time.time()
        with self._connect() as connection:
            row = self._authorized_row(connection, job_id, uid, token=token)
            if row["state"] in TERMINAL_STATES:
                return self._decode(row)
            if row["cancel_requested"]:
                state = "cancelled"
            else:
                state = "succeeded" if return_code == 0 else "failed"
            connection.execute(
                """
                UPDATE jobs
                SET state = ?, return_code = ?, finished_at = ?, updated_at = ?, message = ?
                WHERE id = ?
                """,
                (state, return_code, now, now, message or None, job_id),
            )
            updated = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            assert updated is not None
            return self._decode(updated)

    def cancel(
        self,
        job_id: int,
        uid: int,
        *,
        admin_uid: int,
        force: bool = False,
        reason: str = "",
    ) -> Dict[str, Any]:
        now = time.time()
        with self._connect() as connection:
            row = self._authorized_row(connection, job_id, uid, admin_uid=admin_uid)
            if row["state"] in TERMINAL_STATES:
                return self._decode(row)
            if force:
                if uid != admin_uid:
                    raise JobPermissionError("only administrator can force cancel a job")
                if row["state"] != "quarantined":
                    raise ValueError(f"force cancel is only permitted for quarantined jobs, not '{row['state']}'")
                reason_clean = reason.strip()
                if not reason_clean:
                    raise ValueError("reason is required for force cancel")
                message = f"force-cancelled by admin (uid={uid}): {reason_clean}"
                connection.execute(
                    """
                    UPDATE jobs
                    SET state = 'cancelled', cancel_requested = 1,
                        finished_at = ?, updated_at = ?, message = ?
                    WHERE id = ?
                    """,
                    (now, now, message, job_id),
                )
            elif row["state"] == "pending":
                connection.execute(
                    """
                    UPDATE jobs
                    SET state = 'cancelled', cancel_requested = 1,
                        finished_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (now, now, job_id),
                )
            else:
                connection.execute(
                    "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                    (now, job_id),
                )
            updated = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            assert updated is not None
            return self._decode(updated)

    def schedule(self, gpu_ids: Iterable[int], externally_busy: Iterable[int], *, policy: str = "fifo") -> None:
        if policy not in ("fifo", "fit"):
            raise ValueError("schedule policy must be 'fifo' or 'fit'")
        all_gpu_ids = sorted(set(gpu_ids))
        external = set(externally_busy)
        now = time.time()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT gpu_ids FROM jobs WHERE state IN ('allocated', 'running', 'quarantined')"
            ).fetchall()
            leased = set()
            for row in rows:
                if row["gpu_ids"]:
                    leased.update(json.loads(row["gpu_ids"]))
            available = [gpu_id for gpu_id in all_gpu_ids if gpu_id not in leased and gpu_id not in external]

            pending = connection.execute(
                "SELECT id, gpu_count, supervisor_pid FROM jobs WHERE state = 'pending' ORDER BY id"
            ).fetchall()
            for row in pending:
                if row["supervisor_pid"] is None:
                    # A failed detached launch must never acquire a lease.
                    if policy == "fifo":
                        break
                    continue
                requested = int(row["gpu_count"])
                if requested > len(available):
                    if policy == "fifo":
                        # Strict FIFO avoids starving large requests.
                        break
                    # Fit admits later jobs using otherwise idle cards. Without
                    # runtime estimates this does not guarantee a future start
                    # time for the skipped job, so it is explicitly opt-in.
                    continue
                assigned = available[:requested]
                del available[:requested]
                connection.execute(
                    """
                    UPDATE jobs
                    SET state = 'allocated', gpu_ids = ?, updated_at = ?
                    WHERE id = ? AND state = 'pending'
                    """,
                    (json.dumps(assigned), now, row["id"]),
                )

    def expire_stale(self, timeout_seconds: float) -> None:
        cutoff = time.time() - timeout_seconds
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET state = CASE WHEN state = 'pending' THEN 'lost' ELSE 'quarantined' END,
                    finished_at = CASE WHEN state = 'pending' THEN ? ELSE NULL END,
                    updated_at = ?,
                    message = 'supervisor heartbeat expired'
                WHERE state IN ('pending', 'allocated', 'running')
                  AND (
                    (supervisor_pid IS NULL AND created_at < ?)
                    OR
                    (supervisor_pid IS NOT NULL AND (heartbeat_at IS NULL OR heartbeat_at < ?))
                  )
                """,
                (now, now, cutoff, cutoff),
            )

    def recover_active(self) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET heartbeat_at = ?, updated_at = ?
                WHERE state IN ('pending', 'allocated', 'running')
                  AND supervisor_pid IS NOT NULL
                """,
                (now, now),
            )

    def active_allocations(self) -> Dict[int, int]:
        allocations: Dict[int, int] = {}
        with self._connect(write=False) as connection:
            rows = connection.execute(
                "SELECT id, gpu_ids FROM jobs WHERE state IN ('allocated', 'running', 'quarantined')"
            ).fetchall()
        for row in rows:
            if row["gpu_ids"]:
                for gpu_id in json.loads(row["gpu_ids"]):
                    allocations[int(gpu_id)] = int(row["id"])
        return allocations

    def active_allocations_summary(self) -> Tuple[Dict[int, int], Dict[int, str]]:
        allocations: Dict[int, int] = {}
        allocation_states: Dict[int, str] = {}
        with self._connect(write=False) as connection:
            rows = connection.execute(
                "SELECT id, state, gpu_ids FROM jobs WHERE state IN ('allocated', 'running', 'quarantined')"
            ).fetchall()
        for row in rows:
            job_id = int(row["id"])
            allocation_states[job_id] = str(row["state"])
            if row["gpu_ids"]:
                for gpu_id in json.loads(row["gpu_ids"]):
                    allocations[int(gpu_id)] = job_id
        return allocations, allocation_states
