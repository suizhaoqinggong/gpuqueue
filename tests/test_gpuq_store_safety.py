from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any, Tuple

import pytest

from gpuq.store import JobStore


def create_job(store: JobStore) -> Tuple[int, str]:
    return store.create_job(
        uid=os.getuid(), username="tester", name="safety", gpu_count=1,
        command=["true"], cwd="/tmp", log_path=None,
    )


@pytest.mark.parametrize("started", [False, True])
def test_stale_lease_is_quarantined_until_cleanup(tmp_path: Path, started: bool) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    uid = os.getuid()
    job_id, token = create_job(store)
    store.register_supervisor(job_id, uid, token, 100)
    store.schedule([0], [])
    if started:
        store.mark_started(job_id, uid, token, 101)
    waiting_id, _ = create_job(store)
    store.expire_stale(-1)
    assert store.get_job(waiting_id, uid)["state"] == "lost"
    assert store.get_job(job_id, uid)["state"] == "quarantined"
    assert store.get_job(job_id, uid)["finished_at"] is None
    assert store.active_allocations() == {0: job_id}

    store.recover_active()
    assert store.heartbeat(job_id, uid, token)["state"] == "quarantined"
    with pytest.raises(ValueError, match="cannot start"):
        store.mark_started(job_id, uid, token, 101)
    next_id, next_token = create_job(store)
    store.register_supervisor(next_id, uid, next_token, 102)
    store.schedule([0], [])
    assert store.get_job(next_id, uid)["state"] == "pending"

    store.finish(job_id, uid, token, 125, "client confirmed process cleanup")
    store.schedule([0], [])
    assert store.get_job(next_id, uid)["state"] == "allocated"


def test_duplicate_supervisor_and_cancelled_start_rejected(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    uid = os.getuid()
    job_id, token = create_job(store)
    store.register_supervisor(job_id, uid, token, 100)
    for pid in (100, 101):
        with pytest.raises(ValueError, match="already has a supervisor"):
            store.register_supervisor(job_id, uid, token, pid)
    store.schedule([0], [])
    store.cancel(job_id, uid, admin_uid=uid)
    with pytest.raises(ValueError, match="cancelled"):
        store.mark_started(job_id, uid, token, 102)
    assert store.active_allocations() == {0: job_id}
    assert store.finish(job_id, uid, token, 125)["state"] == "cancelled"
    assert store.active_allocations() == {}


def test_expiration_cannot_interleave_start_read_and_write(tmp_path: Path) -> None:
    read_complete = Event()
    allow_start_write = Event()
    expiration_entered = Event()
    expiration_complete = Event()

    class PausingStore(JobStore):
        pause = False

        def _authorized_row(self, connection: sqlite3.Connection, *args: Any, **kwargs: Any) -> sqlite3.Row:
            row = super()._authorized_row(connection, *args, **kwargs)
            if self.pause:
                assert connection.in_transaction
                read_complete.set()
                assert allow_start_write.wait(5)
            return row

    database = tmp_path / "jobs.sqlite3"
    store = PausingStore(database)
    competing = JobStore(database)
    uid = os.getuid()
    job_id, token = create_job(store)
    store.register_supervisor(job_id, uid, token, 100)
    store.schedule([0], [])
    store.pause = True

    def expire_and_reschedule() -> None:
        expiration_entered.set()
        competing.expire_stale(-1)
        next_id, next_token = create_job(competing)
        competing.register_supervisor(next_id, uid, next_token, 102)
        competing.schedule([0], [])
        assert competing.get_job(next_id, uid)["state"] == "pending"
        expiration_complete.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        starter = pool.submit(store.mark_started, job_id, uid, token, 101)
        try:
            assert read_complete.wait(5)
            expirer = pool.submit(expire_and_reschedule)
            assert expiration_entered.wait(5)
            assert not expiration_complete.wait(0.1)
        finally:
            allow_start_write.set()
        assert starter.result(timeout=5)["state"] == "running"
        expirer.result(timeout=5)
    assert competing.get_job(job_id, uid)["state"] == "quarantined"
    assert competing.active_allocations() == {0: job_id}


def test_store_connections_close_on_success_and_error(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    with store._connect() as connection:
        assert connection.in_transaction
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with pytest.raises(ValueError, match="rollback"):
        with store._connect() as failed_connection:
            failed_connection.execute("UPDATE jobs SET state = 'failed'")
            raise ValueError("rollback")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        failed_connection.execute("SELECT 1")


@pytest.mark.parametrize("policy,small_state", [("fifo", "pending"), ("fit", "allocated")])
def test_scheduler_policy_controls_overtaking(tmp_path: Path, policy: str, small_state: str) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    uid = os.getuid()
    large_id, large_token = store.create_job(
        uid=uid, username="tester", name="large", gpu_count=2,
        command=["true"], cwd="/tmp", log_path=None,
    )
    small_id, small_token = create_job(store)
    store.register_supervisor(large_id, uid, large_token, 100)
    store.register_supervisor(small_id, uid, small_token, 101)
    store.schedule([0, 1], [1], policy=policy)
    assert store.get_job(large_id, uid)["state"] == "pending"
    assert store.get_job(small_id, uid)["state"] == small_state
    if policy == "fit":
        assert store.get_job(small_id, uid)["gpu_ids"] == [0]


def test_invalid_schedule_policy_does_not_allocate(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    job_id, _ = create_job(store)
    with pytest.raises(ValueError, match="schedule policy"):
        store.schedule([0], [], policy="unknown")
    assert store.get_job(job_id, os.getuid())["state"] == "pending"


@pytest.mark.parametrize("policy,next_state", [("fifo", "pending"), ("fit", "allocated")])
def test_unregistered_job_never_acquires_a_lease(tmp_path: Path, policy: str, next_state: str) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    uid = os.getuid()
    orphan_id, _ = create_job(store)
    ready_id, ready_token = create_job(store)
    store.register_supervisor(ready_id, uid, ready_token, 100)
    store.schedule([0], [], policy=policy)
    assert store.get_job(orphan_id, uid)["state"] == "pending"
    assert store.get_job(orphan_id, uid)["gpu_ids"] == []
    assert store.get_job(ready_id, uid)["state"] == next_state
    with store._connect() as connection:
        connection.execute("UPDATE jobs SET created_at = 0 WHERE id = ?", (orphan_id,))
    store.expire_stale(60)
    assert store.get_job(orphan_id, uid)["state"] == "lost"
    store.schedule([0], [], policy=policy)
    assert store.get_job(ready_id, uid)["state"] == "allocated"
