"""Failure injection and queue visibility regressions from the code review."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import pytest

from gpuq import cli
from gpuq.daemon import GPUQService, validate_gpu_topology
from gpuq.gpu import GPU
from gpuq.store import JobPermissionError, JobStore


def create(store: JobStore, root: Path, count: int = 1) -> tuple[int, str]:
    return store.create_job(
        uid=os.getuid(), username="tester", name="regression", gpu_count=count,
        command=[sys.executable, "-c", "print('workload ran')"],
        cwd=str(root), log_path=str(root / "job.log"),
    )


@pytest.mark.parametrize("expire", [False, True])
def test_lost_registration_reply_recovers_or_cleans_expired_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expire: bool,
) -> None:
    monkeypatch.setenv("GPUQ_GPU_IDS", "0")
    store = JobStore(tmp_path / "jobs.db")
    service = GPUQService(store, [GPU(0, "GPUQ-FAKE-0", "fake", 0)], os.getuid())
    job_id, token = create(store, tmp_path)
    attempts = 0

    def request(payload: Any, socket_path: Path) -> Any:
        nonlocal attempts
        response = service.handle(os.getuid(), payload)
        if payload["action"] == "register":
            attempts += 1
            if attempts == 1:
                store.schedule([0], [])
                if expire:
                    store.expire_stale(-1)
                raise cli.DaemonUnavailableError("registration reply lost after commit")
        return response

    monkeypatch.setattr(cli, "request", request)
    kwargs = dict(
        job_id=job_id, token=token, socket_path=tmp_path / "q.sock", interactive=False,
        mirror_output=False, local_job={
            "command": [sys.executable, "-c", "print('workload ran')"],
            "cwd": str(tmp_path), "log_path": str(tmp_path / "job.log"),
        },
    )
    if expire:
        with pytest.raises(cli.ClientError, match="quarantined"):
            cli.supervise(**kwargs)
        assert not (tmp_path / "job.log").exists()
    else:
        assert cli.supervise(**kwargs) == 0
        assert "workload ran" in (tmp_path / "job.log").read_text()
    assert attempts == 2
    assert store.active_allocations() == {}
    assert store.get_job(job_id, os.getuid())["state"] == ("failed" if expire else "succeeded")
    with pytest.raises(JobPermissionError, match="invalid job token"):
        store.register_supervisor(job_id, os.getuid(), "wrong-token", os.getpid())


@pytest.mark.parametrize("cancel", ["none", "remote", "signal"])
def test_pending_task_survives_transport_failure_and_observes_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: str,
) -> None:
    monkeypatch.setenv("GPUQ_GPU_IDS", "0")
    store = JobStore(tmp_path / "jobs.db")
    service = GPUQService(store, [GPU(0, "GPUQ-FAKE-0", "fake", 0)], os.getuid())
    service.heartbeat_interval = 0.01
    job_id, token = create(store, tmp_path)
    heartbeats = 0

    def request(payload: Any, socket_path: Path) -> Any:
        nonlocal heartbeats
        if payload["action"] == "heartbeat":
            heartbeats += 1
            if heartbeats == 1:
                if cancel == "signal":
                    signal.raise_signal(signal.SIGTERM)
                raise cli.DaemonUnavailableError("brief daemon outage")
            if cancel == "remote":
                store.cancel(job_id, os.getuid(), admin_uid=os.getuid())
            else:
                store.schedule([0], [])
        return service.handle(os.getuid(), payload)

    monkeypatch.setattr(cli, "request", request)
    code = cli.supervise(
        job_id=job_id, token=token, socket_path=tmp_path / "q.sock", interactive=False,
        mirror_output=False, local_job={
            "command": [sys.executable, "-c", "print('workload ran')"],
            "cwd": str(tmp_path), "log_path": str(tmp_path / "job.log"),
        },
    )
    assert heartbeats >= 2
    assert code == {"none": 0, "remote": 1, "signal": 130}[cancel]
    assert store.get_job(job_id, os.getuid())["state"] == ("succeeded" if cancel == "none" else "cancelled")
    assert (tmp_path / "job.log").exists() is (cancel == "none")
    assert store.active_allocations() == {}


def test_retry_does_not_swallow_business_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def denied(payload: Any, socket_path: Path) -> Any:
        calls.append(payload)
        raise cli.ClientError("invalid job token")

    monkeypatch.setattr(cli, "request", denied)
    with pytest.raises(cli.ClientError, match="invalid job token"):
        cli._request_with_retry({"action": "register"}, Path("unused"))
    assert len(calls) == 1


@pytest.mark.parametrize("raw", [b"", b'{"ok":true', b'{"ok":'])
def test_incomplete_rpc_response_is_retryable(monkeypatch: pytest.MonkeyPatch, raw: bytes) -> None:
    monkeypatch.setenv("GPUQ_DAEMON_UID", str(os.getuid()))
    with tempfile.TemporaryDirectory(prefix="gq-frame-", dir="/tmp") as directory:
        path = Path(directory) / "q.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(path))
            server.listen(1)

            def respond() -> None:
                connection, _ = server.accept()
                with connection:
                    connection.recv(4096)
                    connection.sendall(raw)

            thread = threading.Thread(target=respond, daemon=True)
            thread.start()
            try:
                with pytest.raises(cli.DaemonUnavailableError):
                    cli.request({"action": "ping"}, path)
            finally:
                thread.join(timeout=3)


def test_list_keeps_old_active_jobs_and_paginates_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any,
) -> None:
    store = JobStore(tmp_path / "jobs.db")
    service = GPUQService(store, [GPU(0, "GPU-test", "fake", 0)], os.getuid())
    old_id, token = create(store, tmp_path)
    store.register_supervisor(old_id, os.getuid(), token, 100)
    store.schedule([0], [])
    store.mark_started(old_id, os.getuid(), token, 101)
    finished_ids = []
    for _ in range(205):
        job_id, token = create(store, tmp_path)
        store.finish(job_id, os.getuid(), token, 0)
        finished_ids.append(job_id)
    # More than one active page also must be rendered, including the old job.
    for _ in range(205):
        create(store, tmp_path)
    monkeypatch.setattr(cli, "request", lambda payload, path: service.handle(os.getuid(), payload))
    args = argparse.Namespace(all=False, socket=tmp_path / "q.sock", history=False, limit=10, before=None)
    assert cli.command_list(args) == 0
    output = capsys.readouterr().out
    listed_ids = {int(line.split()[0]) for line in output.splitlines() if line.split()[0].isdigit()}
    assert old_id in listed_ids
    assert len(listed_ids) == 206 + 10
    assert "more history:" in output
    args.history = True
    args.before = finished_ids[-10]
    cli.command_list(args)
    output = capsys.readouterr().out
    listed_ids = {int(line.split()[0]) for line in output.splitlines() if line.split()[0].isdigit()}
    assert listed_ids == set(finished_ids[-20:-10])


def test_queue_summary_stays_small_with_large_commands_and_names(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    service = GPUQService(store, [GPU(0, "GPU-test", "fake", 0)], os.getuid())
    for _ in range(50):
        job_id, _ = store.create_job(
            uid=os.getuid(), username="tester", name="n" * 10000, gpu_count=1,
            command=["true", "x" * 90000], cwd=str(tmp_path), log_path=None,
        )
    response = service.handle(os.getuid(), {"action": "list"})
    assert len(response["jobs"]) == 50
    assert len(json.dumps(response).encode()) < 100000
    assert all("command" not in job and len(job["name"]) <= 128 for job in response["jobs"])
    details = service.handle(os.getuid(), {"action": "get", "job_id": job_id})["job"]
    assert len(details["command"][1]) == 90000
    assert len(details["name"]) == 10000


@pytest.mark.parametrize("payload", [{"limit": 0}, {"limit": 201}, {"before_id": -1}])
def test_list_rejects_invalid_page_parameters(tmp_path: Path, payload: Any) -> None:
    service = GPUQService(JobStore(tmp_path / "jobs.db"), [], os.getuid())
    with pytest.raises(ValueError):
        service.handle(os.getuid(), {"action": "list", **payload})
